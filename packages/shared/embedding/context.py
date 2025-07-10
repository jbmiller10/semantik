"""Context managers for embedding service lifecycle management.

This module provides context managers to ensure proper resource cleanup
for embedding services, preventing GPU memory leaks and ensuring
consistent state management.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from .base import BaseEmbeddingService
from .service import get_embedding_service

logger = logging.getLogger(__name__)


@asynccontextmanager
async def embedding_service_context(**kwargs: Any) -> AsyncIterator[BaseEmbeddingService]:
    """Context manager for embedding service lifecycle.

    Ensures that the embedding service is properly cleaned up after use,
    even if an exception occurs. This is the recommended way to use
    embedding services for temporary operations.

    Args:
        **kwargs: Arguments passed to get_embedding_service()
            - config: Optional VecpipeConfig object
            - mock_mode: Whether to use mock embeddings
            - Other service-specific options

    Yields:
        BaseEmbeddingService: The initialized embedding service

    Example:
        async with embedding_service_context(mock_mode=True) as service:
            embeddings = await service.embed_texts(["hello world"])
            # Service will be automatically cleaned up after this block
    """
    service = None
    try:
        logger.debug("Creating embedding service in context manager")
        service = await get_embedding_service(**kwargs)
        yield service
    except Exception as e:
        logger.error(f"Error in embedding service context: {e}")
        raise
    finally:
        if service is not None:
            try:
                logger.debug("Cleaning up embedding service")
                await service.cleanup()
            except Exception as e:
                logger.error(f"Error during embedding service cleanup: {e}")
                # Don't re-raise cleanup errors to avoid masking the original exception


@asynccontextmanager
async def temporary_embedding_service(
    model_name: str, service_class: type[BaseEmbeddingService] | None = None, **kwargs: Any
) -> AsyncIterator[BaseEmbeddingService]:
    """Create a temporary embedding service instance with a specific model.

    This context manager creates a new service instance that is independent
    of the global singleton. Useful for:
    - Loading different models temporarily
    - Testing with specific configurations
    - Isolated operations that shouldn't affect global state

    Args:
        model_name: The model to load (e.g., "BAAI/bge-base-en-v1.5")
        service_class: Optional service class to use (defaults to DenseEmbeddingService)
        **kwargs: Additional arguments for service initialization
            - quantization: Model quantization ("float32", "float16", "int8")
            - device: Device to use ("cuda", "cpu")
            - max_memory: Maximum GPU memory to use

    Yields:
        BaseEmbeddingService: A new, isolated service instance

    Example:
        async with temporary_embedding_service("sentence-transformers/all-MiniLM-L6-v2") as service:
            # This won't affect the global embedding service
            embeddings = await service.embed_texts(["test"])
    """
    if service_class is None:
        from .dense import DenseEmbeddingService

        service_class = DenseEmbeddingService

    service = None
    try:
        logger.info(f"Creating temporary embedding service with model: {model_name}")
        service = service_class(**kwargs)
        await service.initialize(model_name, **kwargs)
        yield service
    except Exception as e:
        logger.error(f"Error in temporary embedding service: {e}")
        raise
    finally:
        if service is not None:
            try:
                logger.info(f"Cleaning up temporary embedding service for model: {model_name}")
                await service.cleanup()
            except Exception as e:
                logger.error(f"Error during temporary service cleanup: {e}")


class ManagedEmbeddingService:
    """A wrapper that ensures embedding service cleanup using context managers.

    This class provides both sync and async context manager interfaces
    for the embedding service, making it easier to use in different contexts.
    """

    def __init__(self, **kwargs: Any):
        """Initialize with arguments for get_embedding_service.

        Args:
            **kwargs: Arguments passed to get_embedding_service()
        """
        self.kwargs = kwargs
        self._service: BaseEmbeddingService | None = None

    async def __aenter__(self) -> BaseEmbeddingService:
        """Async context manager entry."""
        self._service = await get_embedding_service(**self.kwargs)
        return self._service

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._service is not None:
            try:
                await self._service.cleanup()
            except Exception as e:
                logger.error(f"Error during managed service cleanup: {e}")
        return False  # Don't suppress exceptions

    def __enter__(self):
        """Sync context manager entry - not supported."""
        raise RuntimeError(
            "Synchronous context manager not supported. Use 'async with ManagedEmbeddingService()' instead."
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Sync context manager exit - not supported."""
