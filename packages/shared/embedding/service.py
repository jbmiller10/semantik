"""Embedding service singleton and factory.

This module provides the primary interface for embedding services throughout the application.
It manages service lifecycle and provides both sync and async interfaces.
"""

import asyncio
import logging
from typing import Any

from .base import BaseEmbeddingService
from .dense import DenseEmbeddingService

logger = logging.getLogger(__name__)

# Global service instance
_embedding_service: BaseEmbeddingService | None = None
_service_lock = asyncio.Lock()


async def get_embedding_service() -> BaseEmbeddingService:
    """Get or create the singleton embedding service instance.

    Returns:
        The initialized embedding service

    Raises:
        RuntimeError: If service cannot be created
    """
    global _embedding_service

    async with _service_lock:
        if _embedding_service is None:
            logger.info("Creating new embedding service instance")
            _embedding_service = DenseEmbeddingService()

        return _embedding_service


async def initialize_embedding_service(model_name: str, **kwargs: Any) -> BaseEmbeddingService:
    """Initialize the embedding service with a specific model.

    Args:
        model_name: The model to load
        **kwargs: Additional configuration options

    Returns:
        The initialized embedding service
    """
    service = await get_embedding_service()

    if not service.is_initialized or service.get_model_info().get("model_name") != model_name:
        await service.initialize(model_name, **kwargs)

    return service


def get_embedding_service_sync() -> BaseEmbeddingService:
    """Get the embedding service synchronously.

    This is a convenience method for code that hasn't been converted to async yet.
    It will create a new event loop if needed.

    Returns:
        The embedding service instance (may not be initialized)
    """
    try:
        loop = asyncio.get_running_loop()
        # We're in an async context, can't use sync
        raise RuntimeError("Cannot use get_embedding_service_sync() from async context")
    except RuntimeError:
        # No running loop, we can proceed
        pass

    # Create new event loop for sync access
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        service = loop.run_until_complete(get_embedding_service())
        return service
    finally:
        loop.close()
        asyncio.set_event_loop(None)


# Export convenience functions for common operations
async def embed_texts(texts: list[str], model_name: str, batch_size: int = 32, **kwargs: Any) -> Any:
    """Convenience function to embed texts with a specific model.

    This will initialize the service if needed.
    """
    service = await initialize_embedding_service(model_name, **kwargs)
    return await service.embed_texts(texts, batch_size, **kwargs)


async def embed_single(text: str, model_name: str, **kwargs: Any) -> Any:
    """Convenience function to embed a single text with a specific model.

    This will initialize the service if needed.
    """
    service = await initialize_embedding_service(model_name, **kwargs)
    return await service.embed_single(text, **kwargs)


# Clean up on module unload
async def cleanup() -> None:
    """Clean up the embedding service."""
    global _embedding_service

    async with _service_lock:
        if _embedding_service is not None:
            await _embedding_service.cleanup()
            _embedding_service = None
