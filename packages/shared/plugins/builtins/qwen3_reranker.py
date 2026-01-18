"""Qwen3 Reranker Plugin - wraps the existing VecPipe reranker as a plugin."""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, ClassVar

from shared.plugins.manifest import PluginManifest
from shared.plugins.types.reranker import RerankerCapabilities, RerankerPlugin, RerankResult

logger = logging.getLogger(__name__)

# Default model configuration
DEFAULT_MODEL = "Qwen/Qwen3-Reranker-0.6B"
DEFAULT_QUANTIZATION = "float16"

# Supported model variants
SUPPORTED_MODELS = [
    "Qwen/Qwen3-Reranker-0.6B",
    "Qwen/Qwen3-Reranker-4B",
    "Qwen/Qwen3-Reranker-8B",
]


class Qwen3RerankerPlugin(RerankerPlugin):
    """Plugin wrapper for Qwen3 cross-encoder reranker.

    This plugin wraps the existing CrossEncoderReranker from VecPipe,
    exposing it through the unified plugin interface.
    """

    PLUGIN_TYPE: ClassVar[str] = "reranker"
    PLUGIN_ID: ClassVar[str] = "qwen3-reranker"
    PLUGIN_VERSION: ClassVar[str] = "1.0.0"

    METADATA: ClassVar[dict[str, Any]] = {
        "display_name": "Qwen3 Reranker",
        "description": "Cross-encoder reranking using Qwen3-Reranker models for improved search relevance",
        "author": "Semantik",
        "license": "Apache-2.0",
    }

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the reranker plugin.

        Args:
            config: Plugin configuration with optional keys:
                - model_name: Model to use (default: Qwen/Qwen3-Reranker-0.6B)
                - quantization: Quantization type (default: float16)
                - device: Device to run on (default: cuda)
                - max_length: Maximum sequence length (default: 512)
        """
        super().__init__(config)
        self._reranker = None
        self._model_lock = threading.Lock()
        self._model_name = self._config.get("model_name", DEFAULT_MODEL)
        self._quantization = self._config.get("quantization", DEFAULT_QUANTIZATION)
        self._device = self._config.get("device", "cuda")
        self._max_length = self._config.get("max_length", 512)

    @classmethod
    def get_manifest(cls) -> PluginManifest:
        """Return plugin manifest."""
        return PluginManifest(
            id=cls.PLUGIN_ID,
            type=cls.PLUGIN_TYPE,
            version=cls.PLUGIN_VERSION,
            display_name=cls.METADATA["display_name"],
            description=cls.METADATA["description"],
            author=cls.METADATA.get("author"),
            license=cls.METADATA.get("license"),
            capabilities={
                "max_documents": 200,
                "max_query_length": 512,
                "max_doc_length": 512,
                "supports_batching": True,
                "models": SUPPORTED_MODELS,
            },
        )

    @classmethod
    def get_capabilities(cls) -> RerankerCapabilities:
        """Return reranker capabilities."""
        return RerankerCapabilities(
            max_documents=200,
            max_query_length=512,
            max_doc_length=512,
            supports_batching=True,
            models=SUPPORTED_MODELS,
        )

    @classmethod
    def get_config_schema(cls) -> dict[str, Any]:
        """Return JSON Schema for plugin configuration."""
        return {
            "type": "object",
            "properties": {
                "model_name": {
                    "type": "string",
                    "description": "Qwen3 reranker model to use",
                    "enum": SUPPORTED_MODELS,
                    "default": DEFAULT_MODEL,
                },
                "quantization": {
                    "type": "string",
                    "description": "Model quantization type",
                    "enum": ["float32", "float16", "bfloat16", "int8"],
                    "default": DEFAULT_QUANTIZATION,
                },
                "device": {
                    "type": "string",
                    "description": "Device to run on",
                    "enum": ["cuda", "cpu"],
                    "default": "cuda",
                },
                "max_length": {
                    "type": "integer",
                    "description": "Maximum input sequence length",
                    "minimum": 128,
                    "maximum": 2048,
                    "default": 512,
                },
            },
        }

    @classmethod
    async def health_check(cls, config: dict[str, Any] | None = None) -> bool:
        """Check if the reranker can be loaded."""
        try:
            # Check if torch and transformers are available
            import torch
            from transformers import AutoTokenizer

            # Check CUDA availability if configured for GPU
            device = (config or {}).get("device", "cuda")
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA not available, reranker will fall back to CPU")

            # Try to load tokenizer for default model (lightweight check)
            model_name = (config or {}).get("model_name", DEFAULT_MODEL)
            _ = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

            return True
        except Exception as e:
            logger.warning("Qwen3 reranker health check failed: %s", e)
            return False

    async def initialize(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the reranker (lazy loading - model loads on first use)."""
        await super().initialize(config)

        # Update config from initialize call
        if config:
            self._model_name = config.get("model_name", self._model_name)
            self._quantization = config.get("quantization", self._quantization)
            self._device = config.get("device", self._device)
            self._max_length = config.get("max_length", self._max_length)

        # Don't load the model yet - it will be loaded on first rerank call
        logger.info(
            "Qwen3 reranker initialized (model: %s, quantization: %s)",
            self._model_name,
            self._quantization,
        )

    def _ensure_model_loaded(self) -> None:
        """Ensure the underlying reranker model is loaded (blocking)."""
        if self._reranker is not None:
            return

        with self._model_lock:
            if self._reranker is not None:
                return

            try:
                from vecpipe.reranker import CrossEncoderReranker

                reranker = CrossEncoderReranker(
                    model_name=self._model_name,
                    device=self._device,
                    quantization=self._quantization,
                    max_length=self._max_length,
                )
                reranker.load_model()
                self._reranker = reranker
            except ImportError as e:
                raise RuntimeError("VecPipe not available. Qwen3 reranker requires vecpipe package.") from e

    async def _ensure_model_loaded_async(self) -> None:
        """Ensure the underlying reranker model is loaded (async-safe).

        Runs the blocking model loading in a thread pool to avoid blocking
        the async event loop during model initialization.
        """
        if self._reranker is None:
            await asyncio.to_thread(self._ensure_model_loaded)

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[RerankResult]:
        """Rerank documents by relevance to query.

        Args:
            query: The search query.
            documents: List of document texts to rerank.
            top_k: Number of results to return. If None, return all.
            metadata: Optional metadata for each document.

        Returns:
            List of RerankResult sorted by relevance (highest first).
        """
        if not documents:
            return []

        # Ensure model is loaded (async-safe to avoid blocking event loop)
        await self._ensure_model_loaded_async()
        assert self._reranker is not None  # Type narrowing

        # Determine top_k
        effective_top_k = top_k if top_k is not None else len(documents)
        effective_top_k = min(effective_top_k, len(documents))

        # Use the underlying reranker
        # rerank() returns list of (index, score) tuples
        reranked = await asyncio.to_thread(
            self._reranker.rerank,
            query=query,
            documents=documents,
            top_k=effective_top_k,
            instruction=None,  # Use default instruction
            return_scores=True,
        )

        # Convert to RerankResult objects
        results = []
        for idx, score in reranked:
            doc_metadata = metadata[idx] if metadata and idx < len(metadata) else {}
            results.append(
                RerankResult(
                    index=idx,
                    score=score,
                    document=documents[idx],
                    metadata=doc_metadata,
                )
            )

        return results

    async def cleanup(self) -> None:
        """Clean up reranker resources."""
        if self._reranker is not None:
            self._reranker.unload_model()
            self._reranker = None
            logger.info("Qwen3 reranker cleaned up")
