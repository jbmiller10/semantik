"""SPLADE Sparse Indexer Plugin - Learned sparse representations.

SPLADE (SParse Lexical AnD Expansion) uses a masked language model to generate
sparse vector representations where:
- Indices are token IDs from the model's vocabulary
- Values are learned term importance weights

Unlike BM25, SPLADE:
- Captures semantic relationships (e.g., synonyms)
- Learns term importance from data
- Requires no corpus statistics (stateless)
- Needs GPU for efficient inference
"""

from __future__ import annotations

import asyncio
import logging
from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from shared.plugins.types.sparse_indexer import (
    SparseIndexerCapabilities,
    SparseIndexerPlugin,
    SparseQueryVector,
    SparseVector,
)

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "naver/splade-cocondenser-ensembledistil"
DEFAULT_MAX_LENGTH = 512
DEFAULT_BATCH_SIZE = 32

# Recommended batch sizes by GPU VRAM tier
# Based on float16 precision with ~400MB base model memory
RECOMMENDED_BATCH_SIZES: dict[str, int] = {
    "4GB": 8,  # GTX 1650, RTX 3050
    "6GB": 16,  # RTX 2060, RTX 3060
    "8GB": 32,  # RTX 3070, RTX 4060
    "12GB": 64,  # RTX 3080, RTX 4070
    "24GB": 128,  # RTX 3090, RTX 4090
}


def get_recommended_batch_size(available_memory_mb: int) -> int:
    """Get recommended batch size based on available GPU memory.

    Args:
        available_memory_mb: Available GPU memory in megabytes.

    Returns:
        Recommended batch size for the given memory budget.

    Example:
        >>> get_recommended_batch_size(8000)  # 8GB GPU
        32
        >>> get_recommended_batch_size(4000)  # 4GB GPU
        8
    """
    if available_memory_mb >= 24000:
        return 128
    if available_memory_mb >= 12000:
        return 64
    if available_memory_mb >= 8000:
        return 32
    if available_memory_mb >= 6000:
        return 16
    if available_memory_mb >= 4000:
        return 8
    return 4  # Minimum for very low memory


class SPLADESparseIndexerPlugin(SparseIndexerPlugin):
    """SPLADE sparse indexer plugin.

    Generates sparse vectors using SPLADE models. SPLADE is a learned sparse
    representation model that uses a masked language model to predict term
    importance weights for each token in the vocabulary.

    Configuration options:
        model_name: HuggingFace model ID (default: naver/splade-cocondenser-ensembledistil)
        device: Device to run on (cuda/cpu/auto, default: auto)
        quantization: Model precision (float32/float16/int8, default: float16)
        max_length: Maximum sequence length (default: 512)
        batch_size: Batch size for encoding (default: 32)
        top_k_tokens: Keep only top-k tokens per vector (optional, default: None)

    Performance characteristics:
        - GPU throughput: ~10-50 docs/sec (batch_size=32, RTX 3080)
        - CPU throughput: ~0.3-1 docs/sec (much slower, use GPU if possible)
        - Query latency: <100ms (GPU), <1s (CPU)
        - Model load time: ~30s (GPU), ~60s (CPU)
        - Memory: ~400MB base + ~50MB per batch_size=8 increment

    Comparison with BM25:
        - BM25: Fast (~1000 docs/sec), keyword-only, requires corpus stats
        - SPLADE: Slower (~10-50 docs/sec), captures semantics, stateless

    Use get_recommended_batch_size(available_memory_mb) to select batch size
    based on GPU memory. See RECOMMENDED_BATCH_SIZES for quick reference.

    Usage:
        plugin = SPLADESparseIndexerPlugin()
        await plugin.initialize({"model_name": "naver/splade-v3"})

        vectors = await plugin.encode_documents([
            {"content": "hello world", "chunk_id": "chunk-1"},
        ])

        query_vector = await plugin.encode_query("hello")
    """

    PLUGIN_TYPE: ClassVar[str] = "sparse_indexer"
    PLUGIN_ID: ClassVar[str] = "splade"
    PLUGIN_VERSION: ClassVar[str] = "1.0.0"
    SPARSE_TYPE: ClassVar[str] = "splade"

    METADATA: ClassVar[dict[str, Any]] = {
        "display_name": "SPLADE Sparse Indexer",
        "description": "Learned sparse representations using SPLADE models",
        "author": "Semantik",
        "license": "Apache-2.0",
    }

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the SPLADE plugin.

        Args:
            config: Plugin configuration. See class docstring for options.
        """
        super().__init__(config)

        # Model configuration
        self._model_name: str = self._config.get("model_name", DEFAULT_MODEL)
        self._device: str = self._config.get("device", "auto")
        self._quantization: str = self._config.get("quantization", "float16")
        self._max_length: int = self._config.get("max_length", DEFAULT_MAX_LENGTH)
        self._batch_size: int = self._config.get("batch_size", DEFAULT_BATCH_SIZE)
        self._top_k_tokens: int | None = self._config.get("top_k_tokens")
        self._inference_timeout: float = self._config.get("inference_timeout", 300.0)

        # Model state (loaded lazily via initialize())
        self._model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizer | None = None
        self._actual_device: str | None = None

    async def initialize(self, config: dict[str, Any] | None = None) -> None:
        """Initialize plugin and load model.

        Args:
            config: Optional config overrides to merge with existing config.
        """
        if config:
            self._config.update(config)
            # Re-read config values after update
            self._model_name = self._config.get("model_name", DEFAULT_MODEL)
            self._device = self._config.get("device", "auto")
            self._quantization = self._config.get("quantization", "float16")
            self._max_length = self._config.get("max_length", DEFAULT_MAX_LENGTH)
            self._batch_size = self._config.get("batch_size", DEFAULT_BATCH_SIZE)
            self._top_k_tokens = self._config.get("top_k_tokens")
            self._inference_timeout = self._config.get("inference_timeout", 300.0)

        await self._load_model()

    async def cleanup(self) -> None:
        """Cleanup resources and unload model."""
        await self._unload_model()

    async def _load_model(self) -> None:
        """Load SPLADE model with GPU memory management.

        Uses device_map={"": 0} (not "auto") for proper memory tracking
        that allows integration with the GPU memory governor.
        """
        # Resolve device
        if self._device == "auto":
            self._actual_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._actual_device = self._device

        logger.info(
            "Loading SPLADE model %s on %s with %s precision",
            self._model_name,
            self._actual_device,
            self._quantization,
        )

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)

        # Build model kwargs
        model_kwargs: dict[str, Any] = {"trust_remote_code": True}

        if self._actual_device == "cuda":
            # Use explicit device_map (NOT "auto") for proper memory management
            # This ensures the model can be properly tracked by the memory governor
            model_kwargs["device_map"] = {"": 0}

            if self._quantization == "float16":
                model_kwargs["torch_dtype"] = torch.float16
            elif self._quantization == "int8":
                try:
                    from transformers import BitsAndBytesConfig

                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_8bit=True,
                        bnb_8bit_compute_dtype=torch.float16,
                    )
                except ImportError:
                    logger.warning("bitsandbytes not available for int8 quantization, falling back to float16")
                    model_kwargs["torch_dtype"] = torch.float16
            # float32 is the default, no special handling needed

        # Load model
        model = AutoModelForMaskedLM.from_pretrained(
            self._model_name,
            **model_kwargs,
        )

        # Move to device if not using device_map (CPU case)
        if self._actual_device != "cuda" or "device_map" not in model_kwargs:
            model = model.to(self._actual_device)

        # Set to eval mode and disable gradients for inference
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        self._model = model

        logger.info("SPLADE model loaded successfully on %s", self._actual_device)

    async def _unload_model(self) -> None:
        """Unload model and free GPU memory.

        Properly synchronizes CUDA operations before clearing cache
        to ensure memory is actually freed.
        """
        if self._model is not None:
            del self._model
            self._model = None

        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        # Sync and clear CUDA cache
        # CRITICAL: synchronize() before empty_cache() as CUDA ops are async
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        self._actual_device = None
        logger.info("SPLADE model unloaded")

    # === Batch inference helper methods ===

    def _tokenize_batch(self, texts: list[str]) -> dict[str, torch.Tensor]:
        """Tokenize a batch of texts for SPLADE inference.

        Args:
            texts: List of document texts to tokenize.

        Returns:
            Tokenizer output dict with input_ids, attention_mask tensors on device.

        Note:
            Must be called after initialize() - assumes tokenizer is loaded.
        """
        assert self._tokenizer is not None, "Tokenizer not loaded"
        assert self._actual_device is not None, "Device not set"
        encoded = self._tokenizer(
            texts,
            max_length=self._max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        return dict(encoded.to(self._actual_device))

    def _extract_sparse_vectors(
        self,
        logits: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> list[tuple[tuple[int, ...], tuple[float, ...]]]:
        """Extract sparse vectors from SPLADE model logits.

        Args:
            logits: Model output logits (batch_size, seq_len, vocab_size).
            attention_mask: Attention mask (batch_size, seq_len).

        Returns:
            List of (indices, values) tuples, one per document in batch.
        """
        # SPLADE activation: log(1 + ReLU(x))
        activated = torch.log1p(torch.relu(logits))

        # Mask padding tokens (expand attention_mask to vocab dimension)
        # Shape: (batch_size, seq_len, 1) -> broadcast to (batch_size, seq_len, vocab_size)
        mask = attention_mask.unsqueeze(-1).float()
        activated = activated * mask

        # Max pool over sequence dimension
        # Shape: (batch_size, vocab_size)
        sparse_reps, _ = activated.max(dim=1)

        results = []
        for i in range(sparse_reps.size(0)):
            doc_sparse = sparse_reps[i]

            # Find non-zero indices
            nonzero_mask = doc_sparse > 0
            indices = nonzero_mask.nonzero(as_tuple=True)[0]
            values = doc_sparse[indices]

            # Apply top_k if configured
            if self._top_k_tokens is not None and len(indices) > self._top_k_tokens:
                topk_values, topk_idx = values.topk(self._top_k_tokens)
                indices = indices[topk_idx]
                values = topk_values

            # Sort by index (required by protocol)
            sorted_order = indices.argsort()
            indices = indices[sorted_order]
            values = values[sorted_order]

            # Convert to tuples
            indices_tuple = tuple(indices.cpu().tolist())
            values_tuple = tuple(values.cpu().tolist())

            results.append((indices_tuple, values_tuple))

        return results

    def _encode_single_batch(
        self,
        texts: list[str],
    ) -> list[tuple[tuple[int, ...], tuple[float, ...]]]:
        """Encode a single batch of texts to sparse vectors (sync, on device).

        Args:
            texts: List of document texts.

        Returns:
            List of (indices, values) tuples.

        Note:
            Must be called after initialize() - assumes model is loaded.
        """
        assert self._model is not None, "Model not loaded"

        # Tokenize
        encoded = self._tokenize_batch(texts)

        # Model inference
        with torch.no_grad():
            output = self._model(**encoded)

        # Extract sparse vectors
        return self._extract_sparse_vectors(
            output.logits,
            encoded["attention_mask"],
        )

    def _encode_single_query(
        self,
        query: str,
    ) -> tuple[tuple[int, ...], tuple[float, ...]]:
        """Encode a single query to sparse vector (sync, on device).

        Args:
            query: Query text to encode.

        Returns:
            Tuple of (indices, values) for the sparse query vector.

        Note:
            Must be called after initialize() - assumes model is loaded.
        """
        assert self._model is not None, "Model not loaded"

        # Tokenize as batch of 1
        encoded = self._tokenize_batch([query])

        # Model inference
        with torch.no_grad():
            output = self._model(**encoded)

        # Extract sparse vector (returns list of 1 item)
        results = self._extract_sparse_vectors(output.logits, encoded["attention_mask"])
        return results[0]

    async def _encode_batch_with_recovery(
        self,
        texts: list[str],
        chunk_ids: list[str],
        metadatas: list[dict[str, Any]],
    ) -> list[SparseVector]:
        """Encode batch with OOM recovery via batch splitting.

        Args:
            texts: Document texts.
            chunk_ids: Chunk identifiers.
            metadatas: Document metadata dicts.

        Returns:
            List of SparseVector instances.
        """
        try:
            # Run inference in thread pool to not block event loop
            loop = asyncio.get_event_loop()
            sparse_results = await loop.run_in_executor(
                None,
                partial(self._encode_single_batch, texts),
            )

            # Build SparseVector objects
            results = []
            for i, (indices, values) in enumerate(sparse_results):
                results.append(
                    SparseVector(
                        indices=indices,
                        values=values,
                        chunk_id=chunk_ids[i],
                        metadata=metadatas[i],
                    )
                )
            return results

        except RuntimeError as e:
            if "out of memory" in str(e).lower() and len(texts) > 1:
                # Clear cache and retry with smaller batch
                torch.cuda.empty_cache()

                mid = len(texts) // 2
                logger.warning(
                    "CUDA OOM with batch_size=%d, splitting to %d + %d",
                    len(texts),
                    mid,
                    len(texts) - mid,
                )

                # Recursively process both halves
                left = await self._encode_batch_with_recovery(
                    texts[:mid],
                    chunk_ids[:mid],
                    metadatas[:mid],
                )
                right = await self._encode_batch_with_recovery(
                    texts[mid:],
                    chunk_ids[mid:],
                    metadatas[mid:],
                )
                return left + right
            raise

    # === Abstract method implementations ===

    async def encode_documents(
        self,
        documents: list[dict[str, Any]],
    ) -> list[SparseVector]:
        """Generate sparse vectors for documents using SPLADE.

        Args:
            documents: List of documents with keys:
                - content (str): Document/chunk text
                - chunk_id (str): Unique chunk identifier
                - metadata (dict, optional): Additional metadata

        Returns:
            List of SparseVector instances, one per input document.

        Raises:
            RuntimeError: If model not loaded (call initialize() first).
            TimeoutError: If encoding exceeds inference_timeout.
        """
        if self._model is None or self._tokenizer is None:
            msg = "Model not loaded. Call initialize() first."
            raise RuntimeError(msg)

        if not documents:
            return []

        # Extract data from documents
        texts = [doc.get("content", "") for doc in documents]
        chunk_ids = [doc["chunk_id"] for doc in documents]
        metadatas = [doc.get("metadata", {}) for doc in documents]

        # Process in batches with timeout
        results: list[SparseVector] = []
        batch_size = self._batch_size

        try:
            async with asyncio.timeout(self._inference_timeout):
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i : i + batch_size]
                    batch_chunk_ids = chunk_ids[i : i + batch_size]
                    batch_metadatas = metadatas[i : i + batch_size]

                    batch_results = await self._encode_batch_with_recovery(
                        batch_texts,
                        batch_chunk_ids,
                        batch_metadatas,
                    )
                    results.extend(batch_results)

                    logger.debug(
                        "SPLADE encode progress: %d/%d documents",
                        len(results),
                        len(documents),
                    )

        except TimeoutError:
            logger.error(
                "SPLADE encoding timed out after %.1f seconds with %d/%d documents processed",
                self._inference_timeout,
                len(results),
                len(documents),
            )
            raise

        return results

    async def encode_query(self, query: str) -> SparseQueryVector:
        """Generate sparse vector for a search query using SPLADE.

        Uses the same SPLADE model and sparse vector extraction as document
        encoding, but optimized for single query inference.

        Args:
            query: Search query text.

        Returns:
            SparseQueryVector with indices and values.

        Raises:
            RuntimeError: If model not loaded (call initialize() first).
            TimeoutError: If encoding exceeds inference_timeout.
        """
        if self._model is None or self._tokenizer is None:
            msg = "Model not loaded. Call initialize() first."
            raise RuntimeError(msg)

        # Handle empty query
        if not query or not query.strip():
            return SparseQueryVector(indices=(), values=())

        # Run inference in thread pool to not block event loop
        loop = asyncio.get_event_loop()
        try:
            async with asyncio.timeout(self._inference_timeout):
                indices, values = await loop.run_in_executor(
                    None,
                    partial(self._encode_single_query, query),
                )
        except TimeoutError:
            logger.error(
                "SPLADE query encoding timed out after %.1f seconds",
                self._inference_timeout,
            )
            raise

        return SparseQueryVector(indices=indices, values=values)

    async def remove_documents(self, chunk_ids: list[str]) -> None:
        """No-op for SPLADE - stateless plugin.

        SPLADE does not maintain any per-document state like IDF statistics,
        so document removal requires no action from the plugin.

        Args:
            chunk_ids: List of chunk IDs being removed (ignored).
        """
        # No-op: SPLADE is stateless, no cleanup needed
        # Suppress unused parameter warning
        _ = chunk_ids

    @classmethod
    def get_capabilities(cls) -> SparseIndexerCapabilities:
        """Return SPLADE capabilities and limits.

        Returns:
            SparseIndexerCapabilities with SPLADE-specific settings.
        """
        return SparseIndexerCapabilities(
            sparse_type="splade",
            max_tokens=512,  # Typical BERT-based limit
            vocabulary_handling="direct",  # Use model vocabulary directly
            supports_batching=True,
            max_batch_size=32,
            requires_corpus_stats=False,  # Stateless - no IDF needed
            max_terms_per_vector=None,  # Can be set via top_k_tokens config
            vocabulary_size=30522,  # BERT vocabulary size
            supports_filters=False,
            idf_storage="file",  # N/A for SPLADE, but required field
            supported_languages=None,  # Multilingual if model supports
        )

    @classmethod
    def get_config_schema(cls) -> dict[str, Any]:
        """Return JSON schema for plugin configuration.

        Returns:
            JSON schema dict describing valid configuration options.
        """
        return {
            "type": "object",
            "properties": {
                "model_name": {
                    "type": "string",
                    "default": DEFAULT_MODEL,
                    "description": "HuggingFace model ID for SPLADE",
                },
                "device": {
                    "type": "string",
                    "enum": ["cuda", "cpu", "auto"],
                    "default": "auto",
                    "description": "Device to run inference on",
                },
                "quantization": {
                    "type": "string",
                    "enum": ["float32", "float16", "int8"],
                    "default": "float16",
                    "description": "Model precision",
                },
                "max_length": {
                    "type": "integer",
                    "default": DEFAULT_MAX_LENGTH,
                    "minimum": 64,
                    "maximum": 512,
                    "description": "Maximum sequence length",
                },
                "batch_size": {
                    "type": "integer",
                    "default": DEFAULT_BATCH_SIZE,
                    "minimum": 1,
                    "maximum": 256,
                    "description": "Batch size for encoding",
                },
                "top_k_tokens": {
                    "type": ["integer", "null"],
                    "default": None,
                    "minimum": 1,
                    "description": "Keep only top-k tokens per vector (optional)",
                },
                "inference_timeout": {
                    "type": "number",
                    "default": 300.0,
                    "minimum": 10.0,
                    "maximum": 3600.0,
                    "description": "Timeout in seconds for encoding operations",
                },
            },
            "additionalProperties": False,
        }

    @classmethod
    async def health_check(cls, config: dict[str, Any] | None = None) -> bool:
        """Check if SPLADE dependencies are available.

        Args:
            config: Optional configuration (unused).

        Returns:
            True if transformers and torch are available.
        """
        del config  # Unused parameter
        try:
            import transformers

            # Verify transformers is usable
            _ = transformers.__version__
            return True
        except ImportError:
            return False
