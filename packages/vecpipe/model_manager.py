"""
Model lifecycle manager with lazy loading and automatic unloading
"""

import asyncio
import contextlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from threading import RLock
from typing import Any

from shared.config import settings
from shared.embedding import EmbeddingService

from .memory_utils import InsufficientMemoryError, check_memory_availability, get_gpu_memory_info
from .reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages embedding model lifecycle with lazy loading and automatic unloading"""

    def __init__(self, unload_after_seconds: int = 300):  # 5 minutes default
        """
        Initialize the model manager

        Args:
            unload_after_seconds: Unload model after this many seconds of inactivity
        """
        self.embedding_service: EmbeddingService | None = None
        self.reranker: CrossEncoderReranker | None = None
        # Initialize executor immediately to avoid hasattr checks
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="model_manager")
        self.unload_after_seconds = unload_after_seconds
        self.last_used: float = 0
        self.last_reranker_used: float = 0
        self.current_model_key: str | None = None
        self.current_reranker_key: str | None = None
        self.lock = RLock()
        self.reranker_lock = RLock()
        self.unload_task: asyncio.Task | None = None
        self.reranker_unload_task: asyncio.Task | None = None
        self.is_mock_mode = False

    def _get_model_key(self, model_name: str, quantization: str) -> str:
        """Generate a unique key for model/quantization combination"""
        return f"{model_name}_{quantization}"

    def _ensure_service_initialized(self) -> None:
        """Ensure the embedding service is initialized"""
        with self.lock:
            if self.embedding_service is None:
                logger.info("Initializing embedding service")
                self.embedding_service = EmbeddingService(mock_mode=settings.USE_MOCK_EMBEDDINGS)
                self.is_mock_mode = self.embedding_service.mock_mode

    def _update_last_used(self) -> None:
        """Update the last used timestamp - must be called within lock"""
        self.last_used = time.time()

    def _update_last_reranker_used(self) -> None:
        """Update the last reranker used timestamp - must be called within reranker_lock"""
        self.last_reranker_used = time.time()

    async def _schedule_unload(self) -> None:
        """Schedule model unloading after inactivity"""
        if self.unload_task:
            self.unload_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.unload_task

        async def unload_after_delay() -> None:
            await asyncio.sleep(self.unload_after_seconds)
            with self.lock:
                if time.time() - self.last_used >= self.unload_after_seconds:
                    logger.info(f"Unloading model after {self.unload_after_seconds}s of inactivity")
                    self.unload_model()

        self.unload_task = asyncio.create_task(unload_after_delay())

    def ensure_model_loaded(self, model_name: str, quantization: str) -> bool:
        """
        Ensure the specified model is loaded

        Returns:
            True if model is loaded successfully, False otherwise
        """
        self._ensure_service_initialized()

        if self.is_mock_mode:
            return True

        model_key = self._get_model_key(model_name, quantization)

        with self.lock:
            # Check if correct model is already loaded
            if self.current_model_key == model_key:
                self._update_last_used()
                return True

            # Need to load the model
            logger.info(f"Loading model: {model_name} with {quantization}")
            if self.embedding_service is not None:
                # Run load_model in executor to avoid async/sync deadlock
                # Note: EmbeddingService.load_model is thread-safe as it uses internal locking
                try:
                    # Use executor with timeout to prevent hanging
                    future = self.executor.submit(self.embedding_service.load_model, model_name, quantization)
                    success = future.result(timeout=300)  # 5 minute timeout

                    if success:
                        self.current_model_key = model_key
                        self._update_last_used()
                        return True
                except TimeoutError:
                    logger.error(f"Model loading timed out for {model_name} with {quantization} after 5 minutes")
                except Exception as e:
                    logger.error(f"Unexpected error loading model {model_name}: {type(e).__name__}: {e}")

            logger.error(f"Failed to load model: {model_name}")
            return False

    def unload_model(self) -> None:
        """Unload the current model to free memory"""
        with self.lock:
            if self.embedding_service:
                logger.info("Unloading current model")
                self.embedding_service.unload_model()
                self.current_model_key = None

                # Force garbage collection
                import gc

                gc.collect()

                # Clear GPU cache if using CUDA
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

    async def generate_embedding_async(
        self, text: str, model_name: str, quantization: str, instruction: str | None = None
    ) -> list[float] | None:
        """
        Generate embedding with lazy model loading

        Args:
            text: Text to embed
            model_name: Model to use
            quantization: Quantization type
            instruction: Optional instruction for the model

        Returns:
            Embedding vector or None if failed
        """
        # Ensure model is loaded
        if not self.ensure_model_loaded(model_name, quantization):
            raise RuntimeError(f"Failed to load model {model_name}")

        # Schedule unloading
        await self._schedule_unload()

        # Generate embedding
        if self.is_mock_mode:
            # Use mock embedding
            import hashlib

            hash_bytes = hashlib.sha256(text.encode()).digest()
            values = []
            for i in range(0, len(hash_bytes), 4):
                chunk = hash_bytes[i : i + 4]
                if len(chunk) == 4:
                    val = int.from_bytes(chunk, byteorder="big") / (2**32)
                    values.append(val * 2 - 1)
            # Pad to standard size
            while len(values) < 256:
                values.append(0.0)
            return values[:1024]  # Standard mock size

        # Use real embedding service
        loop = asyncio.get_event_loop()
        assert self.embedding_service is not None  # Already checked in ensure_model_loaded
        return await loop.run_in_executor(
            self.executor, self.embedding_service.generate_single_embedding, text, model_name, quantization, instruction
        )

    async def generate_embeddings_batch_async(
        self, texts: list[str], model_name: str, quantization: str, instruction: str | None = None, batch_size: int = 32
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts with lazy model loading using batch processing

        Args:
            texts: List of texts to embed
            model_name: Model to use
            quantization: Quantization type
            instruction: Optional instruction for the model
            batch_size: Initial batch size for processing (will be adapted based on GPU memory)

        Returns:
            List of embedding vectors

        Raises:
            RuntimeError: If model loading fails
        """
        # Ensure model is loaded
        if not self.ensure_model_loaded(model_name, quantization):
            raise RuntimeError(f"Failed to load model {model_name}")

        # Schedule unloading
        await self._schedule_unload()

        # Generate embeddings
        if self.is_mock_mode:
            # Use mock embeddings for all texts
            import hashlib

            embeddings = []
            for text in texts:
                hash_bytes = hashlib.sha256(text.encode()).digest()
                values = []
                for i in range(0, len(hash_bytes), 4):
                    chunk = hash_bytes[i : i + 4]
                    if len(chunk) == 4:
                        val = int.from_bytes(chunk, byteorder="big") / (2**32)
                        values.append(val * 2 - 1)
                # Pad to standard size
                while len(values) < 256:
                    values.append(0.0)
                embeddings.append(values[:1024])  # Standard mock size
            return embeddings

        # Use real embedding service with batch processing
        loop = asyncio.get_event_loop()
        assert self.embedding_service is not None  # Already checked in ensure_model_loaded

        # Call the batch processing method
        embeddings_array = await loop.run_in_executor(
            self.executor,
            self.embedding_service.generate_embeddings,
            texts,
            model_name,
            quantization,
            batch_size,  # Initial batch size - the service will handle adaptive sizing
            False,  # show_progress
            instruction,
        )

        if embeddings_array is None:
            raise RuntimeError("Failed to generate embeddings")

        # Convert numpy array to list of lists
        result: list[list[float]] = embeddings_array.tolist()
        return result

    def ensure_reranker_loaded(self, model_name: str, quantization: str) -> bool:
        """
        Ensure the specified reranker model is loaded

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

            # Check memory availability before loading
            current_models = {}
            if self.current_model_key:
                model_name_parts = self.current_model_key.split("_")
                if len(model_name_parts) >= 2:
                    current_models["embedding"] = ("_".join(model_name_parts[:-1]), model_name_parts[-1])

            can_load, memory_msg = check_memory_availability(model_name, quantization, current_models)
            logger.info(f"Memory check: {memory_msg}")

            if not can_load and "Can free" in memory_msg:
                # Memory pressure - inform user instead of silent fallback
                raise InsufficientMemoryError(
                    f"Cannot load reranker due to insufficient GPU memory. {memory_msg}. "
                    f"Consider using a smaller model or different quantization."
                )
            if not can_load:
                # Even with unloading, not enough memory
                raise InsufficientMemoryError(
                    f"Cannot load reranker: {memory_msg}. This GPU cannot run both models simultaneously."
                )

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
            import random

            scores = [(i, random.random()) for i in range(len(documents))]
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:top_k]

        # Use real reranker
        loop = asyncio.get_event_loop()
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
        """Get current status of the model manager"""
        status = {
            "embedding_model_loaded": self.current_model_key is not None,
            "current_embedding_model": self.current_model_key,
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

        # Add reranker info if loaded
        if self.reranker is not None:
            status["reranker_info"] = self.reranker.get_model_info()  # type: ignore[assignment]

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
