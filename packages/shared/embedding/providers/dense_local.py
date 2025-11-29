"""Local dense embedding provider using sentence-transformers and transformers.

This provider supports:
- Sentence-transformers models (all-MiniLM, all-mpnet, BGE, etc.)
- Qwen embedding models with instruction-aware last-token pooling
- Quantization: float32, float16, int8
- Adaptive batch sizing with OOM recovery
"""

from __future__ import annotations

import asyncio
import gc
import logging
import os
from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from sentence_transformers import SentenceTransformer
from torch import Tensor
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel

from shared.embedding.models import MODEL_CONFIGS, ModelConfig, get_model_config
from shared.embedding.plugin_base import BaseEmbeddingPlugin, EmbeddingProviderDefinition
from shared.embedding.types import EmbeddingMode
from shared.metrics.prometheus import record_batch_size_reduction, record_oom_error, update_current_batch_size

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    from shared.config.vecpipe import VecpipeConfig

logger = logging.getLogger(__name__)


def check_int8_compatibility() -> tuple[bool, str]:
    """Check if INT8 quantization is available.

    Returns:
        tuple: (is_compatible, message)
    """
    if not torch.cuda.is_available():
        return False, "INT8 requires CUDA GPU"

    if not os.environ.get("CC"):
        logger.warning("CC environment variable not set, setting to 'gcc'")
        os.environ["CC"] = "gcc"
        os.environ["CXX"] = "g++"

    try:
        import bitsandbytes as bnb

        logger.debug(f"bitsandbytes version: {bnb.__version__}")
    except ImportError:
        return False, "bitsandbytes not installed"

    try:
        from bitsandbytes.nn import Linear8bitLt

        test_layer = Linear8bitLt(16, 16).cuda()
        test_input = torch.randn(1, 16).cuda()
        _ = test_layer(test_input)
        del test_layer, test_input
        torch.cuda.empty_cache()
    except Exception as e:
        return False, f"INT8 layer test failed: {str(e)}"

    return True, "INT8 quantization available"


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Pool the last token for models like Qwen that use last-token pooling."""
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


class DenseLocalEmbeddingProvider(BaseEmbeddingPlugin):
    """Local dense embedding provider using sentence-transformers and transformers.

    This provider supports sentence-transformers models and Qwen embedding models
    with features including:
    - Quantization (float32, float16, int8)
    - Adaptive batch sizing with OOM recovery
    - GPU acceleration with CUDA
    - Instruction-aware embeddings (Qwen models)
    """

    INTERNAL_NAME: ClassVar[str] = "dense_local"
    API_ID: ClassVar[str] = "dense_local"
    PROVIDER_TYPE: ClassVar[str] = "local"

    METADATA: ClassVar[dict[str, Any]] = {
        "display_name": "Local Dense Embeddings",
        "description": "High-quality local embeddings using sentence-transformers or Qwen models",
        "best_for": ["semantic_search", "document_similarity", "clustering"],
        "pros": [
            "No data leaves your server",
            "High quality embeddings",
            "GPU acceleration",
            "Supports quantization",
        ],
        "cons": [
            "Requires GPU for best performance",
            "Large models need significant memory",
        ],
    }

    # Model detection patterns
    _SENTENCE_TRANSFORMER_PATTERNS: ClassVar[tuple[str, ...]] = (
        "sentence-transformers/",
        "BAAI/bge-",
        "intfloat/",
    )

    _QWEN_PATTERNS: ClassVar[tuple[str, ...]] = (
        "Qwen/Qwen3-Embedding",
        "Qwen/Qwen2-Embedding",
    )

    def __init__(self, config: VecpipeConfig | None = None, **_kwargs: Any) -> None:
        """Initialize the dense local embedding provider.

        Args:
            config: Optional VecpipeConfig for configuration
            **_kwargs: Additional options (unused in base init)
        """
        self.model: PreTrainedModel | SentenceTransformer | None = None
        self.tokenizer: PreTrainedTokenizerBase | None = None
        self.model_name: str | None = None
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_qwen_model: bool = False
        self.dimension: int | None = None
        self.max_sequence_length: int = 512
        self._initialized: bool = False

        self.config = config

        # Quantization settings
        self.quantization: str = "float32"
        self.dtype: torch.dtype = torch.float32

        # Adaptive batch size management
        self.original_batch_size: int | None = None
        self.current_batch_size: int | None = None
        if config is not None:
            self.min_batch_size = config.MIN_BATCH_SIZE
            self.batch_size_increase_threshold = config.BATCH_SIZE_INCREASE_THRESHOLD
            self.enable_adaptive_batch_size = config.ENABLE_ADAPTIVE_BATCH_SIZE
        else:
            self.min_batch_size = 1
            self.batch_size_increase_threshold = 10
            self.enable_adaptive_batch_size = True
        self.successful_batches: int = 0

    @classmethod
    def get_definition(cls) -> EmbeddingProviderDefinition:
        """Return the canonical definition for this provider."""
        return EmbeddingProviderDefinition(
            api_id=cls.API_ID,
            internal_id=cls.INTERNAL_NAME,
            display_name="Local Dense Embeddings",
            description="Local embedding generation using sentence-transformers or Qwen models",
            provider_type="local",
            supports_quantization=True,
            supports_instruction=True,  # Qwen models support instructions
            supports_batch_processing=True,
            supports_asymmetric=True,  # Handles query/document mode differently
            supported_models=tuple(MODEL_CONFIGS.keys()),
            default_config={
                "quantization": "float16",
                "batch_size": 32,
            },
            performance_characteristics={
                "latency": "low_to_moderate",
                "throughput": "high",
                "memory_usage": "moderate_to_high",
            },
            is_plugin=False,
        )

    @classmethod
    def supports_model(cls, model_name: str) -> bool:
        """Check if this provider supports the given model."""
        # Check known models first
        if model_name in MODEL_CONFIGS:
            return True

        # Check sentence-transformer patterns
        for pattern in cls._SENTENCE_TRANSFORMER_PATTERNS:
            if model_name.startswith(pattern):
                return True

        # Check Qwen patterns
        return any(model_name.startswith(pattern) for pattern in cls._QWEN_PATTERNS)

    @classmethod
    def get_model_config(cls, model_name: str) -> ModelConfig | None:
        """Get configuration for a specific model."""
        return get_model_config(model_name)

    @classmethod
    def list_supported_models(cls) -> list[ModelConfig]:
        """List all supported models."""
        return list(MODEL_CONFIGS.values())

    @property
    def is_initialized(self) -> bool:
        """Check if the service is initialized."""
        return self._initialized

    async def initialize(self, model_name: str, **kwargs: Any) -> None:
        """Initialize the embedding model.

        Args:
            model_name: HuggingFace model name
            **kwargs: Additional options:
                - quantization: "float32", "float16", "int8"
                - device: "cuda" or "cpu"
                - trust_remote_code: bool

        Raises:
            ValueError: If input parameters are invalid
            RuntimeError: If model initialization fails
        """
        try:
            if not isinstance(model_name, str) or not model_name.strip():
                raise ValueError("model_name must be a non-empty string")

            quantization = kwargs.get("quantization", "float32")
            if quantization not in ["float32", "float16", "int8"]:
                raise ValueError(f"quantization must be one of ['float32', 'float16', 'int8'], got '{quantization}'")

            device = kwargs.get("device", self.device)
            if device not in ["cuda", "cpu"]:
                raise ValueError(f"device must be 'cuda' or 'cpu', got '{device}'")

            if device == "cuda" and not torch.cuda.is_available():
                if kwargs.get("allow_fallback", True):
                    logger.warning("CUDA requested but not available, falling back to CPU")
                    device = "cpu"
                else:
                    raise RuntimeError("CUDA device requested but CUDA is not available")

            await self.cleanup()

            self.model_name = model_name.strip()
            self.quantization = quantization
            self.device = device

            # Check int8 compatibility if requested
            if self.quantization == "int8" and self.device == "cuda":
                is_compatible, msg = check_int8_compatibility()
                if not is_compatible:
                    error_msg = f"INT8 quantization not available: {msg}"
                    logger.error(error_msg)
                    if kwargs.get("allow_fallback", True):
                        logger.warning("Falling back to float32")
                        self.quantization = "float32"
                    else:
                        raise ValueError(error_msg)

            # Check if this is a Qwen model
            self.is_qwen_model = "Qwen" in model_name and "Embedding" in model_name

            logger.info(f"Initializing {model_name} with {self.quantization} on {self.device}")

            # Run model loading in thread pool
            await asyncio.get_running_loop().run_in_executor(None, self._load_model_sync, model_name, kwargs)

            # Get dimension from model
            if self.is_qwen_model and self.model is not None:
                assert hasattr(self.model, "config"), "Qwen model should have config attribute"
                self.dimension = getattr(self.model.config, "hidden_size", None)
            elif self.model is not None and hasattr(self.model, "get_sentence_embedding_dimension"):
                self.dimension = self.model.get_sentence_embedding_dimension()  # type: ignore[operator]
            else:
                test_embedding = await self._embed_single_internal("test")
                self.dimension = len(test_embedding)

            self._initialized = True
            logger.info(f"Model initialized successfully. Dimension: {self.dimension}")

        except Exception as e:
            logger.error(f"Failed to initialize embedding model {model_name}: {e}")
            self._initialized = False
            raise RuntimeError(f"Embedding model initialization failed for {model_name}: {e}") from e

    def _load_model_sync(self, model_name: str, kwargs: dict[str, Any]) -> None:
        """Synchronously load the model (runs in thread pool)."""
        if self.is_qwen_model:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, padding_side="left", trust_remote_code=kwargs.get("trust_remote_code", False)
            )

            model_kwargs = self._get_model_kwargs()
            self.model = AutoModel.from_pretrained(model_name, **model_kwargs)
            self.max_sequence_length = 32768  # Qwen3 supports long context

        else:
            model_kwargs = {}
            if self.device == "cuda":
                model_kwargs["device"] = self.device

            self.model = SentenceTransformer(model_name, **model_kwargs)

            if self.quantization == "float16" and self.device == "cuda":
                self.model = self.model.half()
                self.dtype = torch.float16

            self.max_sequence_length = getattr(self.model, "max_seq_length", 512)

    def _get_model_kwargs(self) -> dict[str, Any]:
        """Get model loading kwargs based on quantization settings."""
        kwargs: dict[str, Any] = {}

        if self.device == "cuda":
            if self.quantization == "float16":
                kwargs["torch_dtype"] = torch.float16
                self.dtype = torch.float16
            elif self.quantization == "int8":
                kwargs["torch_dtype"] = torch.float32
                self.dtype = torch.float32

            kwargs["device_map"] = {"": 0}

        return kwargs

    def _apply_mode_transform(
        self,
        texts: list[str],
        mode: EmbeddingMode,
        instruction: str | None = None,
    ) -> tuple[list[str], str | None]:
        """Apply query/document transformation based on model type and mode.

        For DOCUMENT mode, we typically return texts unchanged (no prefix).
        For QUERY mode, we apply model-specific prefixes or instructions.

        Args:
            texts: Original texts to transform
            mode: QUERY or DOCUMENT mode
            instruction: Optional custom instruction override

        Returns:
            Tuple of (transformed_texts, effective_instruction)
        """
        # Document mode: return texts unchanged, no instruction
        if mode == EmbeddingMode.DOCUMENT:
            return texts, None

        # Query mode: apply model-specific transformations
        config = get_model_config(self.model_name) if self.model_name else None

        if self.is_qwen_model:
            # Qwen uses instruction-based format, handled in _embed_qwen_texts
            # Just pass through the instruction (or use default from config)
            effective_instruction = instruction
            if effective_instruction is None and config and config.default_query_instruction:
                effective_instruction = config.default_query_instruction
            return texts, effective_instruction

        # Prefix-based models (BGE, E5, etc.)
        if config and config.is_asymmetric and config.query_prefix:
            # Apply query prefix
            transformed = [f"{config.query_prefix}{text}" for text in texts]
            return transformed, instruction

        # Symmetric model or no special handling
        return texts, instruction

    async def _embed_texts_internal(
        self, texts: list[str], batch_size: int = 32, *, mode: EmbeddingMode | None = None, **kwargs: Any
    ) -> NDArray[np.float32]:
        """Internal method for embedding texts."""
        if not texts:
            return np.array([]).reshape(0, self.dimension or 384)

        texts = [text if text.strip() else " " for text in texts]

        # Default to QUERY mode for backward compatibility
        effective_mode = mode if mode is not None else EmbeddingMode.QUERY

        normalize = kwargs.get("normalize", True)
        show_progress = kwargs.get("show_progress", False)
        instruction = kwargs.get("instruction", None)

        # Apply mode-specific transformation
        transformed_texts, effective_instruction = self._apply_mode_transform(
            texts, effective_mode, instruction
        )

        return await asyncio.get_running_loop().run_in_executor(
            None, self._embed_texts_sync, transformed_texts, batch_size, normalize, show_progress, effective_instruction
        )

    def _embed_texts_sync(
        self, texts: list[str], batch_size: int, normalize: bool, show_progress: bool, instruction: str | None
    ) -> NDArray[np.float32]:
        """Synchronously embed texts (runs in thread pool)."""
        if self.is_qwen_model:
            return self._embed_qwen_texts(texts, batch_size, normalize, instruction)
        return self._embed_sentence_transformer_texts(texts, batch_size, normalize, show_progress)

    def _embed_qwen_texts(
        self, texts: list[str], batch_size: int, normalize: bool, instruction: str | None
    ) -> NDArray[np.float32]:
        """Embed texts using Qwen model with adaptive batch sizing."""
        if instruction:
            texts = [f"Instruct: {instruction}\nQuery:{text}" for text in texts]

        if self.enable_adaptive_batch_size and self.device == "cuda":
            if self.original_batch_size is None:
                self.original_batch_size = batch_size
                self.current_batch_size = batch_size

            current_batch_size = self.current_batch_size or batch_size
        else:
            current_batch_size = batch_size

        all_embeddings = []
        i = 0

        while i < len(texts):
            batch_texts = texts[i : i + current_batch_size]

            try:
                if self.tokenizer is None:
                    raise RuntimeError("Tokenizer not initialized")
                batch_dict = self.tokenizer(
                    batch_texts, padding=True, truncation=True, max_length=self.max_sequence_length, return_tensors="pt"
                ).to(self.device)

                if self.model is None:
                    raise RuntimeError("Model not initialized")
                assert isinstance(self.model, PreTrainedModel)
                with torch.no_grad():
                    if self.dtype == torch.float16:
                        with torch.cuda.amp.autocast(dtype=torch.float16):
                            outputs = cast(Any, self.model)(**batch_dict)
                    else:
                        outputs = cast(Any, self.model)(**batch_dict)

                    embeddings = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])

                    if normalize:
                        embeddings = F.normalize(embeddings, p=2, dim=1)

                    all_embeddings.append(embeddings.cpu().numpy())

                if self.enable_adaptive_batch_size and self.device == "cuda":
                    self.successful_batches += 1

                    if (
                        self.successful_batches >= self.batch_size_increase_threshold
                        and self.original_batch_size is not None
                        and current_batch_size < self.original_batch_size
                    ):
                        new_size = min(current_batch_size * 2, self.original_batch_size)
                        logger.info(
                            f"Increasing batch size from {current_batch_size} to {new_size} "
                            f"after {self.successful_batches} successes"
                        )
                        current_batch_size = new_size
                        self.current_batch_size = new_size
                        self.successful_batches = 0

                        if self.model_name and self.quantization:
                            update_current_batch_size(self.model_name, self.quantization, new_size)

                i += len(batch_texts)

            except torch.cuda.OutOfMemoryError:
                if not self.enable_adaptive_batch_size or self.device != "cuda":
                    raise

                if self.model_name and self.quantization:
                    record_oom_error(self.model_name, self.quantization)

                if current_batch_size > self.min_batch_size:
                    torch.cuda.empty_cache()
                    new_batch_size = max(self.min_batch_size, current_batch_size // 2)
                    logger.warning(
                        f"OOM with batch size {current_batch_size}, reducing to {new_batch_size} "
                        f"for model {self.model_name} with quantization {self.quantization}"
                    )

                    if self.model_name and self.quantization:
                        record_batch_size_reduction(self.model_name, self.quantization)
                        update_current_batch_size(self.model_name, self.quantization, new_batch_size)

                    current_batch_size = new_batch_size
                    self.current_batch_size = new_batch_size
                    self.successful_batches = 0

                else:
                    logger.error(f"OOM even with minimum batch size {self.min_batch_size}")
                    raise RuntimeError(
                        f"Unable to process batch even with minimum batch size {self.min_batch_size}"
                    ) from None

        return np.vstack(all_embeddings)

    def _embed_sentence_transformer_texts(
        self, texts: list[str], batch_size: int, normalize: bool, show_progress: bool
    ) -> NDArray[np.float32]:
        """Embed texts using sentence-transformers with adaptive batch sizing."""
        if self.model is None:
            raise RuntimeError("Model not initialized")
        assert isinstance(self.model, SentenceTransformer)

        if self.enable_adaptive_batch_size and self.device == "cuda":
            if self.original_batch_size is None:
                self.original_batch_size = batch_size
                self.current_batch_size = batch_size

            current_batch_size = self.current_batch_size or batch_size
        else:
            current_batch_size = batch_size

        embeddings: NDArray[np.float32] | None = None

        while current_batch_size >= self.min_batch_size:
            try:
                logger.debug(
                    f"Attempting to encode with batch_size={current_batch_size}, quantization={self.quantization}"
                )

                embeddings = self.model.encode(
                    texts,
                    batch_size=current_batch_size,
                    normalize_embeddings=normalize,
                    convert_to_numpy=True,
                    show_progress_bar=show_progress,
                )

                if self.enable_adaptive_batch_size and self.device == "cuda":
                    self.successful_batches += 1

                    if (
                        self.successful_batches >= self.batch_size_increase_threshold
                        and self.original_batch_size is not None
                        and current_batch_size < self.original_batch_size
                    ):
                        new_size = min(current_batch_size * 2, self.original_batch_size)
                        logger.info(
                            f"Increasing batch size from {current_batch_size} to {new_size} "
                            f"after {self.successful_batches} successes"
                        )
                        self.current_batch_size = new_size
                        self.successful_batches = 0

                        if self.model_name and self.quantization:
                            update_current_batch_size(self.model_name, self.quantization, new_size)

                break

            except torch.cuda.OutOfMemoryError:
                if not self.enable_adaptive_batch_size or self.device != "cuda":
                    raise

                if self.model_name and self.quantization:
                    record_oom_error(self.model_name, self.quantization)

                if current_batch_size > self.min_batch_size:
                    torch.cuda.empty_cache()
                    new_batch_size = max(self.min_batch_size, current_batch_size // 2)
                    logger.warning(
                        f"OOM with batch size {current_batch_size}, reducing to {new_batch_size} "
                        f"for model {self.model_name} with quantization {self.quantization}"
                    )

                    if self.model_name and self.quantization:
                        record_batch_size_reduction(self.model_name, self.quantization)
                        update_current_batch_size(self.model_name, self.quantization, new_batch_size)

                    current_batch_size = new_batch_size
                    self.current_batch_size = new_batch_size
                    self.successful_batches = 0
                else:
                    logger.error(f"OOM even with minimum batch size {self.min_batch_size}")
                    raise RuntimeError(
                        f"Unable to process batch even with minimum batch size {self.min_batch_size}"
                    ) from None

        if embeddings is None:
            raise RuntimeError("Failed to generate embeddings after all retries")

        return embeddings

    async def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 32,
        *,
        mode: EmbeddingMode | None = None,
        **kwargs: Any,
    ) -> NDArray[np.float32]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process at once
            mode: Embedding mode - QUERY for search queries, DOCUMENT for indexing.
                  Defaults to QUERY for backward compatibility.
            **kwargs: Additional options (instruction, normalize, show_progress)
        """
        if not self._initialized:
            raise RuntimeError("Embedding service not initialized. Call initialize() first.")

        if not isinstance(texts, list):
            raise ValueError("texts must be a list of strings")

        if not texts:
            logger.warning("Empty text list provided to embed_texts")
            return np.array([]).reshape(0, self.dimension or 384)

        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        if not all(isinstance(text, str) for text in texts):
            raise ValueError("All items in texts must be strings")

        empty_indices = [i for i, text in enumerate(texts) if not text.strip()]
        if empty_indices:
            logger.warning(
                f"Found {len(empty_indices)} empty or whitespace-only texts at indices: {empty_indices[:5]}..."
            )

        return await self._embed_texts_internal(texts, batch_size, mode=mode, **kwargs)

    async def _embed_single_internal(
        self, text: str, *, mode: EmbeddingMode | None = None, **kwargs: Any
    ) -> NDArray[np.float32]:
        """Internal method for embedding a single text."""
        embeddings = await self._embed_texts_internal([text], batch_size=1, mode=mode, **kwargs)
        result: NDArray[np.float32] = embeddings[0]
        return result

    async def embed_single(
        self,
        text: str,
        *,
        mode: EmbeddingMode | None = None,
        **kwargs: Any,
    ) -> NDArray[np.float32]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed
            mode: Embedding mode - QUERY for search queries, DOCUMENT for indexing.
                  Defaults to QUERY for backward compatibility.
            **kwargs: Additional options (instruction, normalize)
        """
        if not self._initialized:
            raise RuntimeError("Embedding service not initialized. Call initialize() first.")

        if not isinstance(text, str):
            raise ValueError("text must be a string")

        embeddings = await self.embed_texts([text], batch_size=1, mode=mode, **kwargs)
        result: NDArray[np.float32] = embeddings[0]
        return result

    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        if not self._initialized or self.dimension is None:
            raise RuntimeError("Service not initialized")
        return self.dimension

    def get_model_info(self) -> dict[str, Any]:
        """Get model information."""
        if not self._initialized:
            raise RuntimeError("Service not initialized")

        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "device": self.device,
            "max_sequence_length": self.max_sequence_length,
            "quantization": self.quantization,
            "is_qwen": self.is_qwen_model,
            "dtype": str(self.dtype),
            "provider": self.INTERNAL_NAME,
        }

    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        self._initialized = False
        self.dimension = None

        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        logger.info("Dense local embedding provider cleaned up")
