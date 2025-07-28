"""Dense embedding service using sentence-transformers and transformers."""

import asyncio
import contextlib
import gc
import logging
import os
import threading
from collections.abc import Coroutine
from typing import Any, Protocol, TypeVar, Union, cast

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
from shared.config.vecpipe import VecpipeConfig
from torch import Tensor
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .base import BaseEmbeddingService

logger = logging.getLogger(__name__)

# Type variable for async return types
T = TypeVar("T")


class EmbeddingServiceProtocol(Protocol):
    """Protocol defining the EmbeddingService interface for better type checking."""

    mock_mode: bool
    allow_quantization_fallback: bool

    def load_model(self, model_name: str, quantization: str = "float32") -> bool:
        """Load a model synchronously."""
        ...

    def get_model_info(self, model_name: str, quantization: str = "float32") -> dict[str, Any]:
        """Get model info synchronously."""
        ...

    def generate_embeddings(
        self,
        texts: list[str],
        model_name: str,
        quantization: str = "float32",
        batch_size: int = 32,
        show_progress: bool = True,
        instruction: str | None = None,
        **kwargs: Any,
    ) -> NDArray[np.float32] | None:
        """Generate embeddings synchronously."""
        ...

    def generate_single_embedding(
        self,
        text: str,
        model_name: str,
        quantization: str = "float32",
        instruction: str | None = None,
        **kwargs: Any,
    ) -> list[float] | None:
        """Generate single embedding synchronously."""
        ...

    def unload_model(self) -> None:
        """Unload the current model to free memory."""
        ...

    def shutdown(self) -> None:
        """Shutdown the service and clean up resources."""
        ...

    @property
    def current_model(self) -> Union[PreTrainedModel, "SentenceTransformer", None]:
        """Get current model for compatibility."""
        ...

    @property
    def current_tokenizer(self) -> PreTrainedTokenizerBase | None:
        """Get current tokenizer for compatibility."""
        ...

    @property
    def current_model_name(self) -> str | None:
        """Get current model name for compatibility."""
        ...

    @property
    def current_quantization(self) -> str:
        """Get current quantization for compatibility."""
        ...

    @property
    def device(self) -> str:
        """Get device for compatibility."""
        ...


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

    def __init__(self, config: VecpipeConfig | None = None, mock_mode: bool | None = None) -> None:
        self.model: PreTrainedModel | "SentenceTransformer" | None = None
        self.tokenizer: PreTrainedTokenizerBase | None = None
        self.model_name: str | None = None
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_qwen_model: bool = False
        self.dimension: int | None = None
        self.max_sequence_length: int = 512
        self._initialized: bool = False

        # Use config if provided, otherwise fall back to direct parameter
        self.config: VecpipeConfig | None = config
        if config is not None:
            self.mock_mode = config.USE_MOCK_EMBEDDINGS
        else:
            # For backward compatibility
            self.mock_mode = mock_mode if mock_mode is not None else False

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

        Raises:
            ValueError: If input parameters are invalid
            RuntimeError: If model initialization fails
        """
        try:
            # Validate input parameters
            if not isinstance(model_name, str) or not model_name.strip():
                raise ValueError("model_name must be a non-empty string")

            quantization = kwargs.get("quantization", "float32")
            if quantization not in ["float32", "float16", "int8"]:
                raise ValueError(f"quantization must be one of ['float32', 'float16', 'int8'], got '{quantization}'")

            device = kwargs.get("device", self.device)
            if device not in ["cuda", "cpu"]:
                raise ValueError(f"device must be 'cuda' or 'cpu', got '{device}'")

            # Check CUDA availability if requested
            if device == "cuda" and not torch.cuda.is_available():
                if kwargs.get("allow_fallback", True):
                    logger.warning("CUDA requested but not available, falling back to CPU")
                    device = "cpu"
                else:
                    raise RuntimeError("CUDA device requested but CUDA is not available")

            # Clean up previous model if exists
            await self.cleanup()

            self.model_name = model_name.strip()
            self.quantization = quantization
            self.device = device
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

            # Get dimension from the model directly
            if self.is_qwen_model and self.model is not None:
                # For Qwen models, get dimension from model config
                # Type assertion: Qwen models are always AutoModel
                assert hasattr(self.model, "config"), "Qwen model should have config attribute"
                self.dimension = getattr(self.model.config, "hidden_size", None)
            elif self.model is not None and hasattr(self.model, "get_sentence_embedding_dimension"):
                # For sentence-transformers
                self.dimension = self.model.get_sentence_embedding_dimension()  # type: ignore[operator]
            else:
                # Fallback: generate test embedding to determine dimension
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

    async def _embed_texts_internal(self, texts: list[str], batch_size: int = 32, **kwargs: Any) -> NDArray[np.float32]:
        """Internal method for embedding texts without validation checks.

        This is used during initialization and other internal operations.
        """
        # Handle empty texts gracefully for internal use
        if not texts:
            return np.array([]).reshape(0, self.dimension or 384)

        # Ensure no empty strings for tokenization
        texts = [text if text.strip() else " " for text in texts]

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
    ) -> NDArray[np.float32]:
        """Synchronously embed texts (runs in thread pool)."""
        if self.is_qwen_model:
            return self._embed_qwen_texts(texts, batch_size, normalize, instruction)
        return self._embed_sentence_transformer_texts(texts, batch_size, normalize, show_progress)

    def _embed_qwen_texts(
        self, texts: list[str], batch_size: int, normalize: bool, instruction: str | None
    ) -> NDArray[np.float32]:
        """Embed texts using Qwen model."""
        # Apply instruction if provided
        if instruction:
            texts = [f"Instruct: {instruction}\nQuery:{text}" for text in texts]

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # Tokenize
            if self.tokenizer is None:
                raise RuntimeError("Tokenizer not initialized")
            batch_dict = self.tokenizer(
                batch_texts, padding=True, truncation=True, max_length=self.max_sequence_length, return_tensors="pt"
            ).to(self.device)

            # Generate embeddings
            if self.model is None:
                raise RuntimeError("Model not initialized")
            # Type assertion: For Qwen models, self.model is an AutoModel instance
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

        return np.vstack(all_embeddings)

    def _embed_sentence_transformer_texts(
        self, texts: list[str], batch_size: int, normalize: bool, show_progress: bool
    ) -> NDArray[np.float32]:
        """Embed texts using sentence-transformers."""
        if self.model is None:
            raise RuntimeError("Model not initialized")
        # Type assertion: This method is only called when we have a SentenceTransformer
        assert isinstance(self.model, SentenceTransformer)
        embeddings: NDArray[np.float32] = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return embeddings

    async def embed_texts(self, texts: list[str], batch_size: int = 32, **kwargs: Any) -> NDArray[np.float32]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            **kwargs: Additional options:
                - normalize: bool (default: True)
                - show_progress: bool (default: False)
                - instruction: str (for Qwen models)

        Raises:
            RuntimeError: If service is not initialized
            ValueError: If input parameters are invalid
        """
        # Validate service state
        if not self._initialized:
            raise RuntimeError("Embedding service not initialized. Call initialize() first.")

        # Validate input parameters
        if not isinstance(texts, list):
            raise ValueError("texts must be a list of strings")

        if not texts:
            logger.warning("Empty text list provided to embed_texts")
            return np.array([]).reshape(0, self.dimension or 384)

        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        # Validate text content
        if not all(isinstance(text, str) for text in texts):
            raise ValueError("All items in texts must be strings")

        # Check for empty strings
        empty_indices = [i for i, text in enumerate(texts) if not text.strip()]
        if empty_indices:
            logger.warning(
                f"Found {len(empty_indices)} empty or whitespace-only texts at indices: {empty_indices[:5]}..."
            )

        # Delegate to internal method
        return await self._embed_texts_internal(texts, batch_size, **kwargs)

    async def _embed_single_internal(self, text: str, **kwargs: Any) -> NDArray[np.float32]:
        """Internal method for embedding a single text without validation."""
        embeddings = await self._embed_texts_internal([text], batch_size=1, **kwargs)
        result: NDArray[np.float32] = embeddings[0]
        return result

    async def embed_single(self, text: str, **kwargs: Any) -> NDArray[np.float32]:
        """Generate embedding for a single text."""
        if not self._initialized:
            raise RuntimeError("Embedding service not initialized. Call initialize() first.")

        if not isinstance(text, str):
            raise ValueError("text must be a string")

        embeddings = await self.embed_texts([text], batch_size=1, **kwargs)
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

    def __init__(self, config: VecpipeConfig | None = None, mock_mode: bool | None = None) -> None:
        # Create service with config or mock_mode
        if config is not None:
            self._service = DenseEmbeddingService(config=config)
            self.mock_mode = config.USE_MOCK_EMBEDDINGS
        else:
            self._service = DenseEmbeddingService(mock_mode=mock_mode if mock_mode is not None else False)
            self.mock_mode = mock_mode if mock_mode is not None else False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._loop_lock = threading.Lock()  # For thread-safe loop management
        self.allow_quantization_fallback = True  # For backwards compatibility

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create persistent event loop for sync operations.

        This method creates a persistent event loop that runs in a separate thread
        to avoid the overhead of creating/destroying loops per operation.
        """
        try:
            # If we're already in an async context, we can't use a separate event loop
            return asyncio.get_running_loop()
        except RuntimeError:
            # No running loop, create our persistent loop
            pass

        with self._loop_lock:
            # Check if we need to create a new loop
            if self._loop is None or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()

                # Start the loop in a separate daemon thread
                def run_loop() -> None:
                    asyncio.set_event_loop(self._loop)
                    try:
                        if self._loop is not None:
                            self._loop.run_forever()
                    except Exception as e:
                        logger.error(f"Event loop thread error: {e}")
                    finally:
                        if self._loop is not None:
                            self._loop.close()

                self._loop_thread = threading.Thread(target=run_loop, daemon=True)
                self._loop_thread.start()

                # Wait for the loop to be ready
                while not self._loop.is_running():
                    threading.Event().wait(0.001)  # Small delay

                logger.debug("Created persistent event loop for sync operations")

            return self._loop

    def _run_async(self, coro: Coroutine[Any, Any, T]) -> T:
        """Helper to run async coroutines from sync context with proper loop management."""
        try:
            # Check if we're already in an async context
            current_loop = asyncio.get_running_loop()
            # If we're in an async context, we need to use run_coroutine_threadsafe
            loop = self._get_loop()
            if loop != current_loop:
                future = asyncio.run_coroutine_threadsafe(coro, loop)
                return future.result()

            # This shouldn't happen in practice, but handle it gracefully
            raise RuntimeError("Cannot run async operation from same event loop")
        except RuntimeError:
            # No running loop, use our persistent loop
            loop = self._get_loop()
            if loop.is_running():
                future = asyncio.run_coroutine_threadsafe(coro, loop)
                return future.result()

            # Fallback to run_until_complete
            return loop.run_until_complete(coro)

    def load_model(self, model_name: str, quantization: str = "float32") -> bool:
        """Load a model synchronously.

        Args:
            model_name: HuggingFace model name
            quantization: Quantization type

        Returns:
            True if successful, False otherwise
        """
        try:
            self._run_async(
                self._service.initialize(
                    model_name,
                    quantization=quantization,
                    allow_fallback=self.allow_quantization_fallback,
                    mock_mode=self.mock_mode,
                )
            )
            logger.info(f"Successfully loaded model {model_name} with quantization {quantization}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model {model_name} with quantization {quantization}: {e}")
            return False

    def get_model_info(self, model_name: str, quantization: str = "float32") -> dict[str, Any]:
        """Get model info synchronously.

        Args:
            model_name: HuggingFace model name
            quantization: Quantization type

        Returns:
            Dictionary with model information or error details
        """
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
                error_msg = f"Failed to load model {model_name} with quantization {quantization}"
                logger.error(error_msg)
                return {"error": error_msg}

            return self._service.get_model_info()
        except Exception as e:
            error_msg = f"Failed to get model info for {model_name}: {e}"
            logger.error(error_msg)
            return {"error": error_msg}

    def generate_embeddings(
        self,
        texts: list[str],
        model_name: str,
        quantization: str = "float32",
        batch_size: int = 32,
        show_progress: bool = True,
        instruction: str | None = None,
        **kwargs: Any,
    ) -> NDArray[np.float32] | None:
        """Generate embeddings synchronously.

        Args:
            texts: List of texts to embed
            model_name: HuggingFace model name
            quantization: Quantization type
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            instruction: Optional instruction for Qwen models
            **kwargs: Additional options

        Returns:
            Numpy array of embeddings or None on error

        Raises:
            ValueError: If input parameters are invalid
        """
        try:
            # Validate input parameters
            if not isinstance(texts, list):
                raise ValueError("texts must be a list of strings")

            if not texts:
                logger.warning("Empty text list provided to generate_embeddings")
                return np.array([]).reshape(0, 384)  # Default dimension

            if not isinstance(model_name, str) or not model_name.strip():
                raise ValueError("model_name must be a non-empty string")

            if quantization not in ["float32", "float16", "int8"]:
                raise ValueError(f"quantization must be one of ['float32', 'float16', 'int8'], got '{quantization}'")

            if batch_size <= 0:
                raise ValueError(f"batch_size must be positive, got {batch_size}")

            if not all(isinstance(text, str) for text in texts):
                raise ValueError("All items in texts must be strings")

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

            return self._run_async(
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
        """Generate single embedding synchronously.

        Args:
            text: Text to embed
            model_name: HuggingFace model name
            quantization: Quantization type
            instruction: Optional instruction for Qwen models
            **kwargs: Additional options

        Returns:
            List of floats representing the embedding, or None on error
        """
        try:
            embeddings = self.generate_embeddings(
                [text], model_name, quantization, batch_size=1, show_progress=False, instruction=instruction, **kwargs
            )
            if embeddings is not None and len(embeddings) > 0:
                result: list[float] = embeddings[0].tolist()
                return result

            logger.warning(f"No embeddings generated for text with model {model_name}")
            return None
        except Exception as e:
            logger.error(f"Failed to generate single embedding for text with model {model_name}: {e}")
            return None

    @property
    def current_model(self) -> Union[PreTrainedModel, "SentenceTransformer", None]:
        """Get current model for compatibility."""
        return self._service.model

    @property
    def current_tokenizer(self) -> PreTrainedTokenizerBase | None:
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
        """Unload the current model to free memory.

        Raises:
            RuntimeError: If model unloading fails
        """
        try:
            if self._service.is_initialized:
                self._run_async(self._service.cleanup())
                logger.info("Model unloaded successfully")
            else:
                logger.info("No model to unload")
        except Exception as e:
            logger.error(f"Failed to unload model: {e}")
            raise RuntimeError(f"Failed to unload model: {e}") from e

    def shutdown(self) -> None:
        """Shutdown the service and clean up resources.

        This method properly closes the persistent event loop and cleans up all resources.
        """
        try:
            # First unload the model
            if self._service.is_initialized:
                self.unload_model()

            # Stop the event loop if it's running
            with self._loop_lock:
                if self._loop is not None and not self._loop.is_closed():
                    # Stop the loop gracefully
                    self._loop.call_soon_threadsafe(self._loop.stop)

                    # Wait for the loop thread to finish
                    if self._loop_thread is not None and self._loop_thread.is_alive():
                        self._loop_thread.join(timeout=5.0)
                        if self._loop_thread.is_alive():
                            logger.warning("Event loop thread did not shut down cleanly")

                    # Close the loop
                    if not self._loop.is_closed():
                        self._loop.close()

                    self._loop = None
                    self._loop_thread = None

                    logger.info("EmbeddingService shutdown completed")

        except Exception as e:
            logger.error(f"Error during EmbeddingService shutdown: {e}")

    def __del__(self) -> None:
        """Destructor to ensure proper cleanup."""
        with contextlib.suppress(Exception):
            self.shutdown()

    def __enter__(self) -> "EmbeddingService":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        """Context manager exit with cleanup."""
        self.shutdown()


# Import centralized model configurations

# Global instances for backwards compatibility (lazy initialization)
_embedding_service: EmbeddingService | None = None
_enhanced_embedding_service: EmbeddingService | None = None


def _get_global_embedding_service() -> EmbeddingService:
    """Get or create the global embedding service instance with lazy initialization."""
    global _embedding_service
    if _embedding_service is None:
        # Try to import webui settings if available
        try:
            from shared.config import settings

            logger.info(
                f"Initializing global embedding service with settings: mock_mode={settings.USE_MOCK_EMBEDDINGS}"
            )
            _embedding_service = EmbeddingService(mock_mode=settings.USE_MOCK_EMBEDDINGS)
        except ImportError as e:
            # Fall back to default configuration
            logger.warning(
                f"Could not import shared.config.settings ({e}), falling back to default embedding service configuration"
            )
            _embedding_service = EmbeddingService()
            logger.info("Global embedding service initialized with default configuration (mock_mode=False)")
        except Exception as e:
            # Handle other potential errors during service creation
            logger.error(f"Error creating embedding service with settings: {e}")
            logger.warning("Falling back to default embedding service configuration")
            _embedding_service = EmbeddingService()
            logger.info("Global embedding service initialized with default configuration after error fallback")
    return _embedding_service


def configure_global_embedding_service(config: VecpipeConfig | None = None, mock_mode: bool | None = None) -> None:
    """Configure the global embedding service instance.

    This allows webui code to properly configure the global instance with settings.
    """
    global _embedding_service
    if config is not None:
        _embedding_service = EmbeddingService(config=config)
    elif mock_mode is not None:
        _embedding_service = EmbeddingService(mock_mode=mock_mode)
    else:
        # Try to use settings
        try:
            from shared.config import settings

            _embedding_service = EmbeddingService(mock_mode=settings.USE_MOCK_EMBEDDINGS)
        except ImportError:
            _embedding_service = EmbeddingService()


# Create lazy-initialized global instances
class _LazyEmbeddingService:
    """Lazy wrapper for embedding service to delay initialization.

    This class implements the EmbeddingServiceProtocol interface and provides
    lazy initialization of the actual EmbeddingService instance.
    """

    def __init__(self) -> None:
        self._instance: EmbeddingService | None = None

    def _get_instance(self) -> EmbeddingService:
        """Get the underlying embedding service instance, creating it if needed."""
        if self._instance is None:
            self._instance = _get_global_embedding_service()
        return self._instance

    def __getattr__(self, name: str) -> Any:
        # Delegate to the actual instance
        return getattr(self._get_instance(), name)

    # Implement the protocol methods explicitly for better type checking
    @property
    def mock_mode(self) -> bool:
        """Get mock mode setting."""
        result: bool = self._get_instance().mock_mode
        return result

    @mock_mode.setter
    def mock_mode(self, value: bool) -> None:
        """Set mock mode setting."""
        self._get_instance().mock_mode = value

    @property
    def allow_quantization_fallback(self) -> bool:
        """Get quantization fallback setting."""
        return self._get_instance().allow_quantization_fallback

    @allow_quantization_fallback.setter
    def allow_quantization_fallback(self, value: bool) -> None:
        """Set quantization fallback setting."""
        self._get_instance().allow_quantization_fallback = value

    def load_model(self, model_name: str, quantization: str = "float32") -> bool:
        """Load a model synchronously."""
        return self._get_instance().load_model(model_name, quantization)

    def get_model_info(self, model_name: str, quantization: str = "float32") -> dict[str, Any]:
        """Get model info synchronously."""
        return self._get_instance().get_model_info(model_name, quantization)

    def generate_embeddings(
        self,
        texts: list[str],
        model_name: str,
        quantization: str = "float32",
        batch_size: int = 32,
        show_progress: bool = True,
        instruction: str | None = None,
        **kwargs: Any,
    ) -> NDArray[np.float32] | None:
        """Generate embeddings synchronously."""
        return self._get_instance().generate_embeddings(
            texts, model_name, quantization, batch_size, show_progress, instruction, **kwargs
        )

    def generate_single_embedding(
        self,
        text: str,
        model_name: str,
        quantization: str = "float32",
        instruction: str | None = None,
        **kwargs: Any,
    ) -> list[float] | None:
        """Generate single embedding synchronously."""
        return self._get_instance().generate_single_embedding(text, model_name, quantization, instruction, **kwargs)

    def unload_model(self) -> None:
        """Unload the current model to free memory."""
        return self._get_instance().unload_model()

    @property
    def current_model(self) -> Union[PreTrainedModel, "SentenceTransformer", None]:
        """Get current model for compatibility."""
        return self._get_instance().current_model

    @property
    def current_tokenizer(self) -> PreTrainedTokenizerBase | None:
        """Get current tokenizer for compatibility."""
        return self._get_instance().current_tokenizer

    @property
    def current_model_name(self) -> str | None:
        """Get current model name for compatibility."""
        return self._get_instance().current_model_name

    @property
    def current_quantization(self) -> str:
        """Get current quantization for compatibility."""
        return self._get_instance().current_quantization

    @property
    def device(self) -> str:
        """Get device for compatibility."""
        return self._get_instance().device

    def __call__(self, *_args: Any, **_kwargs: Any) -> Any:
        if self._instance is None:
            self._instance = _get_global_embedding_service()
        # This method shouldn't be called since EmbeddingService isn't callable
        # But keeping for compatibility
        raise AttributeError("EmbeddingService object is not callable")

    def shutdown(self) -> None:
        """Shutdown the lazy embedding service."""
        if self._instance is not None:
            self._instance.shutdown()
            self._instance = None


# Create global instances for backwards compatibility
embedding_service = _LazyEmbeddingService()
enhanced_embedding_service = embedding_service  # Alias
