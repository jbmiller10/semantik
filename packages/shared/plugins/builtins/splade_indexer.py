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

import logging
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

    # === Abstract method implementations ===
    # NOTE: encode_documents() and encode_query() are stubs that return empty vectors.
    # Full inference pipeline will be implemented in Phase 5b and 5c.

    async def encode_documents(
        self,
        documents: list[dict[str, Any]],
    ) -> list[SparseVector]:
        """Generate sparse vectors for documents.

        NOTE: This is a stub implementation for Phase 5a. Returns empty vectors.
        Full inference pipeline will be implemented in Phase 5b.

        Args:
            documents: List of documents with keys:
                - content (str): Document/chunk text
                - chunk_id (str): Unique chunk identifier
                - metadata (dict, optional): Additional metadata

        Returns:
            List of SparseVector instances, one per input document.
        """
        if self._model is None:
            msg = "Model not loaded. Call initialize() first."
            raise RuntimeError(msg)

        # Placeholder - returns empty vectors
        # Phase 5b will implement actual SPLADE inference
        return [
            SparseVector(
                indices=(),
                values=(),
                chunk_id=doc["chunk_id"],
            )
            for doc in documents
        ]

    async def encode_query(self, query: str) -> SparseQueryVector:
        """Generate sparse vector for a search query.

        NOTE: This is a stub implementation for Phase 5a. Returns empty vector.
        Full inference pipeline will be implemented in Phase 5c.

        Args:
            query: Search query text.

        Returns:
            SparseQueryVector with indices and values.
        """
        # Suppress unused parameter warning - query will be used in Phase 5c
        _ = query

        if self._model is None:
            msg = "Model not loaded. Call initialize() first."
            raise RuntimeError(msg)

        # Placeholder - returns empty vector
        # Phase 5c will implement actual query encoding
        return SparseQueryVector(indices=(), values=())

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
