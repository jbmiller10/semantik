"""Base abstraction for embedding services."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .types import EmbeddingMode


class BaseEmbeddingService(ABC):
    """Abstract base class for embedding services.

    This defines the minimal interface that all embedding services must implement.
    Implementations can add additional methods, but must support these core operations.
    """

    @abstractmethod
    async def initialize(self, model_name: str, **kwargs: Any) -> None:
        """Initialize the embedding model.

        Args:
            model_name: The model identifier (e.g., HuggingFace model name)
            **kwargs: Implementation-specific configuration options

        Raises:
            ValueError: If the model cannot be loaded
            RuntimeError: If initialization fails
        """

    @abstractmethod
    async def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 32,
        *,
        mode: "EmbeddingMode | None" = None,
        **kwargs: Any,
    ) -> NDArray[np.float32]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process at once
            mode: Embedding mode - QUERY for search queries, DOCUMENT for indexing.
                  If None, defaults to QUERY for backward compatibility.
                  For asymmetric models (E5, BGE, Qwen), this affects prefix/instruction.
            **kwargs: Implementation-specific options (e.g., instruction, normalize)

        Returns:
            numpy array of shape (n_texts, embedding_dim)

        Raises:
            ValueError: If texts is empty or invalid
            RuntimeError: If embedding generation fails
        """

    @abstractmethod
    async def embed_single(
        self,
        text: str,
        *,
        mode: "EmbeddingMode | None" = None,
        **kwargs: Any,
    ) -> NDArray[np.float32]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed
            mode: Embedding mode - QUERY for search queries, DOCUMENT for indexing.
                  If None, defaults to QUERY for backward compatibility.
            **kwargs: Implementation-specific options

        Returns:
            numpy array of shape (embedding_dim,)

        Raises:
            ValueError: If text is empty or invalid
            RuntimeError: If embedding generation fails
        """

    @abstractmethod
    def get_dimension(self) -> int:
        """Get the embedding dimension.

        Returns:
            The dimension of embeddings produced by this service

        Raises:
            RuntimeError: If called before initialization
        """

    @abstractmethod
    def get_model_info(self) -> dict[str, Any]:
        """Get information about the current model.

        Returns:
            Dictionary containing at least:
            - model_name: str
            - dimension: int
            - device: str (e.g., "cuda", "cpu")
            - max_sequence_length: int

        Raises:
            RuntimeError: If called before initialization
        """

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources (models, memory, etc.).

        This should be called when the service is no longer needed.
        After cleanup, the service must be re-initialized before use.
        """

    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        """Check if the service is initialized and ready to use."""

    async def __aenter__(self) -> "BaseEmbeddingService":
        """Async context manager entry.

        Returns the service instance for use in async with statements.

        Example:
            async with service:
                embeddings = await service.embed_texts(["hello"])
        """
        return self

    async def __aexit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any) -> bool:
        """Async context manager exit.

        Ensures cleanup is called when exiting the context, even if an
        exception occurred. This provides automatic resource management.

        Args:
            exc_type: Exception type if an exception occurred
            exc_val: Exception value if an exception occurred
            exc_tb: Exception traceback if an exception occurred

        Returns:
            False to propagate any exception that occurred
        """
        try:
            await self.cleanup()
        except Exception as e:
            # Log but don't raise cleanup errors to avoid masking original exception
            import logging

            logging.getLogger(__name__).error(f"Error during context cleanup: {e}")
        return False  # Don't suppress exceptions
