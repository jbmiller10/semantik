"""Dense embedding service using sentence-transformers and transformers."""

import asyncio
import gc
import logging
import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from sentence_transformers import SentenceTransformer
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

from .base import BaseEmbeddingService

logger = logging.getLogger(__name__)


def check_int8_compatibility() -> tuple[bool, str]:
    """Check if INT8 quantization is available.

    Returns:
        tuple: (is_compatible, message)
    """
    # Check CUDA availability
    if not torch.cuda.is_available():
        return False, "INT8 requires CUDA GPU"

    # Check for C compiler
    if not os.environ.get("CC"):
        logger.warning("CC environment variable not set, setting to 'gcc'")
        os.environ["CC"] = "gcc"
        os.environ["CXX"] = "g++"

    # Check bitsandbytes
    try:
        import bitsandbytes as bnb

        logger.debug(f"bitsandbytes version: {bnb.__version__}")
    except ImportError:
        return False, "bitsandbytes not installed"

    # Check if we can create INT8 layers
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


class DenseEmbeddingService(BaseEmbeddingService):
    """Dense embedding service supporting both sentence-transformers and custom models like Qwen."""

    def __init__(self, mock_mode: bool = False) -> None:
        self.model: Any = None
        self.tokenizer: Any = None
        self.model_name: str | None = None
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_qwen_model: bool = False
        self.dimension: int | None = None
        self.max_sequence_length: int = 512
        self._initialized: bool = False
        self.mock_mode: bool = mock_mode

        # Quantization settings
        self.quantization: str = "float32"
        self.dtype: torch.dtype = torch.float32

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
                - mock_mode: bool
        """
        try:
            # Clean up previous model if exists
            await self.cleanup()

            self.model_name = model_name
            self.quantization = kwargs.get("quantization", "float32")
            self.device = kwargs.get("device", self.device)
            self.mock_mode = kwargs.get("mock_mode", self.mock_mode)

            # If in mock mode, skip actual model loading
            if self.mock_mode:
                logger.info(f"Mock mode: simulating initialization of {model_name}")
                # Get dimension from config if available
                from .models import get_model_config

                config = get_model_config(model_name)
                self.dimension = config.dimension if config else 384
                self._initialized = True
                return

            # Check int8 compatibility if requested
            if self.quantization == "int8" and self.device == "cuda":
                is_compatible, msg = check_int8_compatibility()
                if not is_compatible:
                    error_msg = f"INT8 quantization not available: {msg}"
                    logger.error(error_msg)
                    # Fallback to float32 if allowed
                    if kwargs.get("allow_fallback", True):
                        logger.warning("Falling back to float32")
                        self.quantization = "float32"
                    else:
                        raise ValueError(error_msg)

            # Check if this is a Qwen model
            self.is_qwen_model = "Qwen" in model_name and "Embedding" in model_name

            logger.info(f"Initializing {model_name} with {self.quantization} on {self.device}")

            # Run model loading in thread pool to avoid blocking
            await asyncio.get_running_loop().run_in_executor(None, self._load_model_sync, model_name, kwargs)

            # Test the model and get dimension
            test_embedding = await self.embed_single("test")
            self.dimension = len(test_embedding)

            self._initialized = True
            logger.info(f"Model initialized successfully. Dimension: {self.dimension}")

        except Exception as e:
            logger.error(f"Failed to initialize model {model_name}: {e}")
            self._initialized = False
            raise RuntimeError(f"Model initialization failed: {e}") from e

    def _load_model_sync(self, model_name: str, kwargs: dict[str, Any]) -> None:
        """Synchronously load the model (runs in thread pool)."""
        if self.is_qwen_model:
            # Load Qwen model using transformers
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, padding_side="left", trust_remote_code=kwargs.get("trust_remote_code", False)
            )

            # Set dtype based on quantization
            model_kwargs = self._get_model_kwargs()
            self.model = AutoModel.from_pretrained(model_name, **model_kwargs)
            self.max_sequence_length = 32768  # Qwen3 supports long context

        else:
            # Load sentence-transformers model
            model_kwargs = {}
            if self.device == "cuda":
                model_kwargs["device"] = self.device

            self.model = SentenceTransformer(model_name, **model_kwargs)

            # Apply quantization
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
                # For int8, we'd need bitsandbytes config
                # Keeping it simple for now
                kwargs["torch_dtype"] = torch.float32
                self.dtype = torch.float32

            kwargs["device_map"] = {"": 0}

        return kwargs

    async def embed_texts(self, texts: list[str], batch_size: int = 32, **kwargs: Any) -> np.ndarray:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            **kwargs: Additional options:
                - normalize: bool (default: True)
                - show_progress: bool (default: False)
                - instruction: str (for Qwen models)
        """
        if not self._initialized:
            raise RuntimeError("Service not initialized. Call initialize() first.")

        if not texts:
            return np.array([])

        # Mock mode - return random embeddings
        if self.mock_mode:
            logger.debug(f"Mock mode: generating embeddings for {len(texts)} texts")
            embeddings = np.random.randn(len(texts), self.dimension or 384).astype(np.float32)
            if kwargs.get("normalize", True):
                # Normalize embeddings
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / norms
            return embeddings

        normalize = kwargs.get("normalize", True)
        show_progress = kwargs.get("show_progress", False)
        instruction = kwargs.get("instruction", None)

        # Process in thread pool to avoid blocking
        return await asyncio.get_running_loop().run_in_executor(
            None, self._embed_texts_sync, texts, batch_size, normalize, show_progress, instruction
        )

    def _embed_texts_sync(
        self, texts: list[str], batch_size: int, normalize: bool, show_progress: bool, instruction: str | None
    ) -> np.ndarray:
        """Synchronously embed texts (runs in thread pool)."""
        if self.is_qwen_model:
            return self._embed_qwen_texts(texts, batch_size, normalize, instruction)
        return self._embed_sentence_transformer_texts(texts, batch_size, normalize, show_progress)

    def _embed_qwen_texts(
        self, texts: list[str], batch_size: int, normalize: bool, instruction: str | None
    ) -> np.ndarray:
        """Embed texts using Qwen model."""
        # Apply instruction if provided
        if instruction:
            texts = [f"Instruct: {instruction}\nQuery:{text}" for text in texts]

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # Tokenize
            batch_dict = self.tokenizer(
                batch_texts, padding=True, truncation=True, max_length=self.max_sequence_length, return_tensors="pt"
            ).to(self.device)

            # Generate embeddings
            with torch.no_grad():
                if self.dtype == torch.float16:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        outputs = self.model(**batch_dict)
                else:
                    outputs = self.model(**batch_dict)

                embeddings = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])

                if normalize:
                    embeddings = F.normalize(embeddings, p=2, dim=1)

                all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    def _embed_sentence_transformer_texts(
        self, texts: list[str], batch_size: int, normalize: bool, show_progress: bool
    ) -> np.ndarray:
        """Embed texts using sentence-transformers."""
        embeddings: np.ndarray = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return embeddings

    async def embed_single(self, text: str, **kwargs: Any) -> np.ndarray:
        """Generate embedding for a single text."""
        embeddings = await self.embed_texts([text], batch_size=1, **kwargs)
        result: np.ndarray = embeddings[0]
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

        # Force garbage collection
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        logger.info("Embedding service cleaned up")


class EmbeddingService:
    """Synchronous wrapper for DenseEmbeddingService for backwards compatibility.

    This provides a synchronous interface that matches the old API while using
    the new async implementation underneath.
    """

    def __init__(self, mock_mode: bool = False) -> None:
        self._service = DenseEmbeddingService()
        self._loop: asyncio.AbstractEventLoop | None = None
        self.mock_mode = mock_mode
        self.allow_quantization_fallback = True  # For backwards compatibility

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create event loop for sync operations."""
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            if self._loop is None or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
            return self._loop

    def load_model(self, model_name: str, quantization: str = "float32") -> bool:
        """Load a model synchronously."""
        try:
            loop = self._get_loop()
            loop.run_until_complete(
                self._service.initialize(
                    model_name,
                    quantization=quantization,
                    allow_fallback=self.allow_quantization_fallback,
                    mock_mode=self.mock_mode,
                )
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def get_model_info(self, model_name: str, quantization: str = "float32") -> dict[str, Any]:
        """Get model info synchronously."""
        try:
            # In mock mode, return mock info without loading
            if self.mock_mode and not self._service.is_initialized:
                from .models import get_model_config

                config = get_model_config(model_name)
                return {
                    "model_name": model_name,
                    "dimension": config.dimension if config else 384,
                    "device": self._service.device,
                    "max_sequence_length": 512,
                    "quantization": quantization,
                    "is_qwen": False,
                    "dtype": "torch.float32",
                }

            # Ensure model is loaded
            if (not self._service.is_initialized or self._service.model_name != model_name) and not self.load_model(
                model_name, quantization
            ):
                return {"error": f"Failed to load model {model_name}"}

            return self._service.get_model_info()
        except Exception as e:
            return {"error": str(e)}

    def generate_embeddings(
        self,
        texts: list[str],
        model_name: str,
        quantization: str = "float32",
        batch_size: int = 32,
        show_progress: bool = True,
        instruction: str | None = None,
        **kwargs: Any,
    ) -> np.ndarray | None:
        """Generate embeddings synchronously."""
        try:
            # Mock mode - return random embeddings
            if self.mock_mode:
                logger.info(f"Mock mode: generating embeddings for {len(texts)} texts")
                # Get dimension from model config if available
                from .models import get_model_config

                config = get_model_config(model_name)
                dim = config.dimension if config else 384
                return np.random.randn(len(texts), dim).astype(np.float32)

            # Ensure model is loaded
            if (not self._service.is_initialized or self._service.model_name != model_name) and not self.load_model(
                model_name, quantization
            ):
                return None

            loop = self._get_loop()
            return loop.run_until_complete(
                self._service.embed_texts(
                    texts, batch_size=batch_size, show_progress=show_progress, instruction=instruction, **kwargs
                )
            )
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return None

    def generate_single_embedding(
        self, text: str, model_name: str, quantization: str = "float32", instruction: str | None = None, **kwargs: Any
    ) -> list[float] | None:
        """Generate single embedding synchronously."""
        embeddings = self.generate_embeddings(
            [text], model_name, quantization, batch_size=1, show_progress=False, instruction=instruction, **kwargs
        )
        if embeddings is not None and len(embeddings) > 0:
            result: list[float] = embeddings[0].tolist()
            return result
        return None

    @property
    def current_model(self) -> Any:
        """Get current model for compatibility."""
        return self._service.model

    @property
    def current_tokenizer(self) -> Any:
        """Get current tokenizer for compatibility."""
        return self._service.tokenizer

    @property
    def current_model_name(self) -> str | None:
        """Get current model name for compatibility."""
        return self._service.model_name

    @property
    def current_quantization(self) -> str:
        """Get current quantization for compatibility."""
        return self._service.quantization

    @property
    def device(self) -> str:
        """Get device for compatibility."""
        return self._service.device

    def unload_model(self) -> None:
        """Unload the current model to free memory."""
        if self._service.is_initialized:
            loop = self._get_loop()
            loop.run_until_complete(self._service.cleanup())


# Import centralized model configurations
from .models import POPULAR_MODELS, QUANTIZED_MODEL_INFO

# Create global instances for backwards compatibility
embedding_service = EmbeddingService()
enhanced_embedding_service = embedding_service  # Alias
