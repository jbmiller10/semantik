"""Embedding service singleton and factory.

This module provides the primary interface for embedding services throughout the application.
It manages service lifecycle and provides both sync and async interfaces.

The service now uses the plugin-based provider system, with auto-detection of the
appropriate provider based on model name.
"""

import asyncio
import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from shared.config.vecpipe import VecpipeConfig

from .base import BaseEmbeddingService

logger = logging.getLogger(__name__)

# Global service instance
_embedding_service: BaseEmbeddingService | None = None
_service_lock = asyncio.Lock()

# Track current model for service reuse
_current_model_name: str | None = None


def _ensure_providers_registered() -> None:
    """Ensure all providers (built-in and plugins) are registered before use."""
    # Import triggers registration in providers/__init__.py (built-ins first)
    from . import providers as _  # noqa: F401

    # Load plugins if enabled (idempotent - only runs once)
    from .plugin_loader import load_embedding_plugins

    load_embedding_plugins()


async def get_embedding_service(
    config: VecpipeConfig | None = None,
    model_name: str | None = None,
    **kwargs: Any,
) -> BaseEmbeddingService:
    """Get or create the singleton embedding service instance.

    Uses the plugin-based provider system to auto-detect the appropriate
    provider based on model name.

    Args:
        config: Optional configuration object for dependency injection
        model_name: Optional model name for provider auto-detection
        **kwargs: Options passed to service creation (e.g., mock_mode)

    Returns:
        The embedding service instance (may need initialization)

    Raises:
        RuntimeError: If service cannot be created
    """
    global _embedding_service, _current_model_name

    _ensure_providers_registered()

    try:
        async with _service_lock:
            # Determine if we need a new service
            need_new_service = False

            if _embedding_service is None:
                need_new_service = True
            elif model_name is not None and model_name != _current_model_name:
                # Different model requested, may need different provider
                from .factory import EmbeddingProviderFactory

                current_provider = EmbeddingProviderFactory.get_provider_for_model(_current_model_name or "")
                new_provider = EmbeddingProviderFactory.get_provider_for_model(model_name)

                if current_provider != new_provider:
                    # Need a different provider type
                    logger.info(f"Switching providers: {current_provider} -> {new_provider} for model {model_name}")
                    if _embedding_service.is_initialized:
                        await _embedding_service.cleanup()
                    need_new_service = True

            if need_new_service:
                logger.info("Creating new embedding service instance")

                # Check for mock mode
                mock_mode = kwargs.get("mock_mode", False)
                if config is not None and hasattr(config, "USE_MOCK_EMBEDDINGS"):
                    mock_mode = config.USE_MOCK_EMBEDDINGS

                if mock_mode:
                    # Use mock provider explicitly
                    from .factory import EmbeddingProviderFactory

                    _embedding_service = EmbeddingProviderFactory.create_provider_by_name(
                        "mock", config=config, **kwargs
                    )
                    _current_model_name = "mock"
                elif model_name is not None:
                    # Use factory to create appropriate provider
                    from .factory import EmbeddingProviderFactory

                    _embedding_service = EmbeddingProviderFactory.create_provider(model_name, config=config, **kwargs)
                    _current_model_name = model_name
                else:
                    # Fall back to DenseEmbeddingService for backward compatibility
                    from .dense import DenseEmbeddingService

                    if config is not None:
                        _embedding_service = DenseEmbeddingService(config=config)
                    else:
                        _embedding_service = DenseEmbeddingService(**kwargs)
                    _current_model_name = None

            if _embedding_service is None:
                raise RuntimeError("Embedding service not initialized")
            return _embedding_service
    except Exception as e:
        logger.error(f"Failed to create embedding service: {e}", exc_info=True)
        raise RuntimeError(f"Failed to create embedding service: {e}") from e


async def initialize_embedding_service(
    model_name: str, config: VecpipeConfig | None = None, **kwargs: Any
) -> BaseEmbeddingService:
    """Initialize the embedding service with a specific model.

    Args:
        model_name: The model to load
        config: Optional configuration object for dependency injection
        **kwargs: Additional configuration options

    Returns:
        The initialized embedding service

    Raises:
        RuntimeError: If service initialization fails
    """
    try:
        # Extract service creation kwargs
        mock_mode = kwargs.get("mock_mode", False)
        service = await get_embedding_service(config=config, model_name=model_name, mock_mode=mock_mode)

        if not service.is_initialized or (hasattr(service, "model_name") and service.model_name != model_name):
            await service.initialize(model_name, **kwargs)

        return service
    except Exception as e:
        logger.error(f"Failed to initialize embedding service with model {model_name}: {e}", exc_info=True)
        raise RuntimeError(f"Failed to initialize embedding service with model {model_name}: {e}") from e


def get_embedding_service_sync(config: VecpipeConfig | None = None) -> BaseEmbeddingService:
    """Get the embedding service synchronously.

    This is a convenience method for code that hasn't been converted to async yet.
    It will create a new event loop if needed.

    Args:
        config: Optional configuration object for dependency injection

    Returns:
        The embedding service instance (may not be initialized)

    Raises:
        RuntimeError: If service cannot be created or if called from async context
    """
    try:
        loop = asyncio.get_running_loop()
        # We're in an async context, can't use sync
        error_msg = "Cannot use get_embedding_service_sync() from async context. Use get_embedding_service() instead."
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    except RuntimeError as e:
        if "async context" in str(e):
            raise  # Re-raise async context error
        # No running loop, we can proceed

    # Create new event loop for sync access
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(get_embedding_service(config=config))
    except Exception as e:
        logger.error(f"Failed to get embedding service synchronously: {e}", exc_info=True)
        raise RuntimeError(f"Failed to get embedding service synchronously: {e}") from e
    finally:
        loop.close()
        asyncio.set_event_loop(None)


# Export convenience functions for common operations
async def embed_texts(texts: list[str], model_name: str, batch_size: int = 32, **kwargs: Any) -> NDArray[np.float32]:
    """Convenience function to embed texts with a specific model.

    This will initialize the service if needed.

    Args:
        texts: List of texts to embed
        model_name: HuggingFace model name
        batch_size: Batch size for processing
        **kwargs: Additional options

    Returns:
        Embeddings array

    Raises:
        RuntimeError: If embedding fails
    """
    try:
        service = await initialize_embedding_service(model_name, **kwargs)
        return await service.embed_texts(texts, batch_size, **kwargs)
    except Exception as e:
        logger.error(f"Failed to embed texts with model {model_name}: {e}", exc_info=True)
        raise RuntimeError(f"Failed to embed texts with model {model_name}: {e}") from e


async def embed_single(text: str, model_name: str, **kwargs: Any) -> NDArray[np.float32]:
    """Convenience function to embed a single text with a specific model.

    This will initialize the service if needed.

    Args:
        text: Text to embed
        model_name: HuggingFace model name
        **kwargs: Additional options

    Returns:
        Single embedding array

    Raises:
        RuntimeError: If embedding fails
    """
    try:
        service = await initialize_embedding_service(model_name, **kwargs)
        return await service.embed_single(text, **kwargs)
    except Exception as e:
        logger.error(f"Failed to embed single text with model {model_name}: {e}", exc_info=True)
        raise RuntimeError(f"Failed to embed single text with model {model_name}: {e}") from e


# Clean up on module unload
async def cleanup() -> None:
    """Clean up the embedding service.

    Raises:
        RuntimeError: If cleanup fails
    """
    global _embedding_service

    try:
        async with _service_lock:
            if _embedding_service is not None:
                await _embedding_service.cleanup()
                _embedding_service = None
                logger.info("Embedding service cleaned up successfully")
    except Exception as e:
        logger.error(f"Failed to clean up embedding service: {e}", exc_info=True)
        raise RuntimeError(f"Failed to clean up embedding service: {e}") from e
