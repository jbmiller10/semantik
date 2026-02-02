"""
Model lifecycle manager with lazy loading and automatic unloading.

This module manages embedding model lifecycle using the plugin-aware provider system.
Providers are created via EmbeddingProviderFactory which auto-detects the appropriate
provider for each model name, enabling support for third-party embedding plugins.
"""

import asyncio
import contextlib
import gc
import hashlib
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor
from threading import RLock
from typing import TYPE_CHECKING, Any

from shared.config import settings
from shared.embedding.factory import EmbeddingProviderFactory
from shared.embedding.types import EmbeddingMode
from shared.plugins.loader import load_plugins

from .memory_utils import InsufficientMemoryError, get_gpu_memory_info
from .reranker import CrossEncoderReranker

if TYPE_CHECKING:
    from shared.embedding.plugin_base import BaseEmbeddingPlugin

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages embedding model lifecycle with lazy loading and automatic unloading.

    This class uses the plugin-aware embedding provider system to enable:
    - Built-in providers (DenseLocalEmbeddingProvider, MockEmbeddingProvider)
    - Third-party plugins registered via semantik.plugins entry points

    The provider is auto-detected based on model name via EmbeddingProviderFactory.
    """

    def __init__(self, unload_after_seconds: int = 300):  # 5 minutes default
        """Initialize the model manager.

        Args:
            unload_after_seconds: Unload model after this many seconds of inactivity
        """
        # Provider-based embedding (replaces legacy EmbeddingService)
        self._provider: BaseEmbeddingPlugin | None = None
        self._provider_name: str | None = None  # Tracks which provider type is active

        # Reranker (still uses legacy CrossEncoderReranker)
        self.reranker: CrossEncoderReranker | None = None

        # Executor for sync operations (reranking)
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="model_manager")

        self.unload_after_seconds = unload_after_seconds
        self.last_used: float = 0
        self.last_reranker_used: float = 0
        self.current_model_key: str | None = None
        self.current_reranker_key: str | None = None

        # Lock for provider operations (async-safe via short-lived usage)
        self.lock = RLock()
        self.reranker_lock = RLock()

        self.unload_task: asyncio.Task[None] | None = None
        self.reranker_unload_task: asyncio.Task[None] | None = None

        # Guards provider initialization / switching against concurrent async calls.
        # This avoids races where multiple requests try to initialize the provider
        # at the same time and one observes a half-initialized provider.
        self._provider_init_lock = asyncio.Lock()

        # Mock mode from settings
        self.is_mock_mode = settings.USE_MOCK_EMBEDDINGS

    def _get_model_key(self, model_name: str, quantization: str) -> str:
        """Generate a unique key for model/quantization combination.

        Note: Uses underscore separator. Quantization values (int8, float16, etc.)
        should not contain underscores, making rsplit("_", 1) safe for parsing.
        """
        if not model_name:
            raise ValueError("model_name cannot be empty")
        if not quantization:
            raise ValueError("quantization cannot be empty")
        if "_" in quantization:
            raise ValueError(f"quantization cannot contain underscore (would break key parsing), got '{quantization}'")
        return f"{model_name}_{quantization}"

    def _parse_model_key(self, model_key: str) -> tuple[str, str] | None:
        """Parse a model key back into (model_name, quantization).

        Returns:
            Tuple of (model_name, quantization) or None if parsing fails.
        """
        if not model_key:
            return None
        parts = model_key.rsplit("_", 1)
        if len(parts) != 2:
            logger.warning("Invalid model key format: %s", model_key)
            return None
        model_name, quantization = parts
        if not model_name or not quantization:
            logger.warning("Empty model_name or quantization in key: %s", model_key)
            return None
        return model_name, quantization

    def _update_last_used(self) -> None:
        """Update the last used timestamp."""
        self.last_used = time.time()

    def _update_last_reranker_used(self) -> None:
        """Update the last reranker used timestamp - must be called within reranker_lock."""
        self.last_reranker_used = time.time()

    async def _ensure_provider_initialized(self, model_name: str, quantization: str) -> "BaseEmbeddingPlugin":
        """Ensure the embedding provider is initialized for the given model.

        This method:
        1. Registers built-in providers and loads plugins if not done
        2. Routes to MockEmbeddingProvider if mock mode is enabled
        3. Auto-detects the appropriate provider for the model via factory
        4. Handles provider switching when model changes

        Args:
            model_name: HuggingFace model name or other model identifier
            quantization: Quantization type (float32, float16, int8)

        Returns:
            An initialized embedding provider

        Raises:
            ValueError: If no provider supports the model
            RuntimeError: If provider initialization fails
        """
        model_key = self._get_model_key(model_name, quantization)

        # Fast path: already initialized with correct model
        if self._provider is not None and self.current_model_key == model_key:
            self._update_last_used()
            return self._provider

        async with self._provider_init_lock:
            # Re-check under the lock in case another task initialized while we awaited.
            if self._provider is not None and self.current_model_key == model_key:
                self._update_last_used()
                return self._provider

            # Ensure providers are registered (idempotent)
            load_plugins(plugin_types={"embedding"})

            # Mock mode handling - use MockEmbeddingProvider
            if self.is_mock_mode:
                if self._provider is None or self._provider_name != "mock":
                    if self._provider is not None:
                        logger.info("Switching to mock provider, cleaning up previous provider")
                        await self._provider.cleanup()
                    logger.info("Creating mock embedding provider")
                    self._provider = EmbeddingProviderFactory.create_provider_by_name("mock")
                    self._provider_name = "mock"
                    await self._provider.initialize(model_name)
                    self.current_model_key = model_key
                self._update_last_used()
                return self._provider

            # Real provider via factory auto-detection
            new_provider_name = EmbeddingProviderFactory.get_provider_for_model(model_name)
            if new_provider_name is None:
                available = EmbeddingProviderFactory.list_available_providers()
                raise ValueError(f"No provider found for model: {model_name}. Available providers: {available}")

            # Switch providers if needed (different provider type or different model)
            if self._provider is None or self._provider_name != new_provider_name or self.current_model_key != model_key:
                if self._provider is not None:
                    logger.info(
                        "Switching provider from '%s' to '%s' for model '%s'",
                        self._provider_name,
                        new_provider_name,
                        model_name,
                    )
                    await self._provider.cleanup()

                logger.info(
                    "Creating embedding provider '%s' for model '%s' with %s quantization",
                    new_provider_name,
                    model_name,
                    quantization,
                )
                self._provider = EmbeddingProviderFactory.create_provider(model_name)
                self._provider_name = new_provider_name
                await self._provider.initialize(model_name, quantization=quantization)
                self.current_model_key = model_key

            self._update_last_used()
            return self._provider

    async def _schedule_unload(self) -> None:
        """Schedule model unloading after inactivity."""
        if self.unload_task:
            self.unload_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.unload_task

        async def unload_after_delay() -> None:
            try:
                await asyncio.sleep(self.unload_after_seconds)
                if time.time() - self.last_used >= self.unload_after_seconds:
                    logger.info(f"Unloading model after {self.unload_after_seconds}s of inactivity")
                    await self.unload_model_async()
            except asyncio.CancelledError:
                # Task was cancelled, this is expected behavior
                raise
            except Exception as e:
                logger.error(f"Failed to unload model after inactivity: {e}", exc_info=True)

        self.unload_task = asyncio.create_task(unload_after_delay())

    async def unload_model_async(self) -> None:
        """Unload the current embedding model to free memory.

        This is the primary method for unloading models. It properly cleans up
        the provider's resources asynchronously.
        """
        if self._provider is not None:
            logger.info("Unloading current embedding model")
            await self._provider.cleanup()
            self._provider = None
            self._provider_name = None
            self.current_model_key = None

            # Force garbage collection
            gc.collect()

            # Clear GPU cache if using CUDA
            # IMPORTANT: synchronize() must be called before empty_cache()
            # because CUDA operations are async and tensors may still be in use
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    def unload_model(self) -> None:
        """Unload the current model (sync wrapper for shutdown).

        Note: Prefer using unload_model_async() in async contexts.
        This sync method is provided for shutdown scenarios.
        """
        if self._provider is not None:
            logger.info("Unloading current embedding model (sync)")
            # Use sync cleanup if available, otherwise skip
            # Most shutdown paths can just let the process exit
            self._provider = None
            self._provider_name = None
            self.current_model_key = None

            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    async def generate_embedding_async(
        self,
        text: str,
        model_name: str,
        quantization: str,
        instruction: str | None = None,
        mode: str | None = None,
    ) -> list[float] | None:
        """Generate embedding with lazy model loading via plugin-aware provider.

        Args:
            text: Text to embed
            model_name: Model to use (provider is auto-detected)
            quantization: Quantization type (float32, float16, int8)
            instruction: Optional instruction for the model
            mode: Embedding mode - 'query' for search, 'document' for indexing

        Returns:
            Embedding vector as list of floats, or None if failed

        Raises:
            ValueError: If no provider supports the model
            RuntimeError: If embedding generation fails
        """
        # Initialize provider (lazy loading)
        provider = await self._ensure_provider_initialized(model_name, quantization)

        # Schedule unloading after inactivity
        await self._schedule_unload()

        # Convert mode string to EmbeddingMode enum
        embedding_mode: EmbeddingMode | None = None
        if mode == "query":
            embedding_mode = EmbeddingMode.QUERY
        elif mode == "document":
            embedding_mode = EmbeddingMode.DOCUMENT

        # Generate embedding using provider (native async)
        embedding = await provider.embed_single(text, mode=embedding_mode, instruction=instruction)
        result: list[float] = embedding.tolist()
        return result

    async def generate_embeddings_batch_async(
        self,
        texts: list[str],
        model_name: str,
        quantization: str,
        instruction: str | None = None,
        batch_size: int = 32,
        mode: str | None = None,
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts via plugin-aware provider.

        Args:
            texts: List of texts to embed
            model_name: Model to use (provider is auto-detected)
            quantization: Quantization type (float32, float16, int8)
            instruction: Optional instruction for the model
            batch_size: Batch size for processing
            mode: Embedding mode - 'query' for search, 'document' for indexing

        Returns:
            List of embedding vectors

        Raises:
            ValueError: If no provider supports the model
            RuntimeError: If embedding generation fails
        """
        # Initialize provider (lazy loading)
        provider = await self._ensure_provider_initialized(model_name, quantization)

        # Schedule unloading after inactivity
        await self._schedule_unload()

        # Convert mode string to EmbeddingMode enum
        embedding_mode: EmbeddingMode | None = None
        if mode == "query":
            embedding_mode = EmbeddingMode.QUERY
        elif mode == "document":
            embedding_mode = EmbeddingMode.DOCUMENT

        # Generate embeddings using provider (native async)
        embeddings_array = await provider.embed_texts(
            texts, batch_size=batch_size, mode=embedding_mode, instruction=instruction
        )

        # Convert numpy array to list of lists
        result: list[list[float]] = embeddings_array.tolist()
        return result

    def ensure_reranker_loaded(self, model_name: str, quantization: str) -> bool:
        """
        Ensure the specified reranker model is loaded

        Args:
            model_name: Name of the reranker model
            quantization: Quantization type (float16, int8, etc.)

        Returns:
            True if reranker is loaded successfully, False otherwise
        """
        if self.is_mock_mode:
            return True

        reranker_key = self._get_model_key(model_name, quantization)

        with self.reranker_lock:
            # Check if correct reranker is already loaded
            if self.current_reranker_key == reranker_key and self.reranker is not None:
                self._update_last_reranker_used()
                return True

            # Need to load the reranker
            logger.info(f"Loading reranker: {model_name} with {quantization}")

            # Unload current reranker if different
            if self.reranker is not None:
                self.reranker.unload_model()

            # Create and load new reranker
            self.reranker = CrossEncoderReranker(model_name=model_name, quantization=quantization)

            try:
                self.reranker.load_model()
                self.current_reranker_key = reranker_key
                self._update_last_reranker_used()
                return True
            except Exception as e:
                logger.error(f"Failed to load reranker: {e}")
                self.reranker = None
                self.current_reranker_key = None

                # If it's an OOM error, provide helpful message
                if "out of memory" in str(e).lower() or "CUDA" in str(e):
                    free_mb, total_mb = get_gpu_memory_info()
                    raise InsufficientMemoryError(
                        f"GPU out of memory while loading reranker. "
                        f"Current free memory: {free_mb}MB / {total_mb}MB total. "
                        f"Try using a smaller model or enabling quantization (int8/float16)."
                    ) from e
                raise

    async def _schedule_reranker_unload(self) -> None:
        """Schedule reranker unloading after inactivity"""
        if self.reranker_unload_task:
            self.reranker_unload_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.reranker_unload_task

        async def unload_after_delay() -> None:
            await asyncio.sleep(self.unload_after_seconds)
            with self.reranker_lock:
                if time.time() - self.last_reranker_used >= self.unload_after_seconds:
                    logger.info(f"Unloading reranker after {self.unload_after_seconds}s of inactivity")
                    self.unload_reranker()

        self.reranker_unload_task = asyncio.create_task(unload_after_delay())

    def unload_reranker(self) -> None:
        """Unload the current reranker to free memory"""
        with self.reranker_lock:
            if self.reranker is not None:
                logger.info("Unloading current reranker")
                self.reranker.unload_model()
                self.reranker = None
                self.current_reranker_key = None

    async def rerank_async(
        self,
        query: str,
        documents: list[str],
        top_k: int,
        model_name: str,
        quantization: str,
        instruction: str | None = None,
    ) -> list[tuple[int, float]]:
        """
        Perform async reranking with lazy model loading

        Args:
            query: Search query
            documents: List of documents to rerank
            top_k: Number of top documents to return
            model_name: Reranker model to use
            quantization: Quantization type
            instruction: Optional instruction for reranking

        Returns:
            List of (index, score) tuples sorted by relevance
        """
        # Ensure reranker is loaded
        if not self.ensure_reranker_loaded(model_name, quantization):
            raise RuntimeError(f"Failed to load reranker {model_name}")

        # Schedule unloading
        await self._schedule_reranker_unload()

        # Perform reranking
        if self.is_mock_mode:
            # Mock reranking - just return indices with fake scores
            seed_source = f"{query}|{len(documents)}|{top_k}|{'|'.join(documents)}"
            seed = int(hashlib.sha256(seed_source.encode()).hexdigest()[:8], 16)
            rng = random.Random(seed)
            scores = [(i, rng.random()) for i in range(len(documents))]
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:top_k]

        # Use real reranker
        loop = asyncio.get_running_loop()
        assert self.reranker is not None  # Already checked in ensure_reranker_loaded
        return await loop.run_in_executor(
            self.executor,
            self.reranker.rerank,
            query,
            documents,
            top_k,
            instruction,
            True,  # return_scores
        )

    def get_status(self) -> dict[str, Any]:
        """Get current status of the model manager."""
        status: dict[str, Any] = {
            "embedding_model_loaded": self.current_model_key is not None,
            "current_embedding_model": self.current_model_key,
            "embedding_provider": self._provider_name,
            "embedding_last_used": self.last_used,
            "embedding_seconds_since_last_use": int(time.time() - self.last_used) if self.last_used > 0 else None,
            "reranker_loaded": self.current_reranker_key is not None,
            "current_reranker": self.current_reranker_key,
            "reranker_last_used": self.last_reranker_used,
            "reranker_seconds_since_last_use": (
                int(time.time() - self.last_reranker_used) if self.last_reranker_used > 0 else None
            ),
            "unload_after_seconds": self.unload_after_seconds,
            "is_mock_mode": self.is_mock_mode,
        }

        # Add provider info if initialized
        if self._provider is not None and self._provider.is_initialized:
            status["provider_info"] = self._provider.get_model_info()

        # Add reranker info if loaded
        if self.reranker is not None:
            status["reranker_info"] = self.reranker.get_model_info()

        return status

    def shutdown(self) -> None:
        """Shutdown the model manager"""
        if self.unload_task:
            self.unload_task.cancel()
        if self.reranker_unload_task:
            self.reranker_unload_task.cancel()
        self.unload_model()
        self.unload_reranker()
        if self.executor:
            self.executor.shutdown(wait=True)
