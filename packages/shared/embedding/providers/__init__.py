"""Built-in embedding providers.

This module exports the built-in providers and auto-registers them on import.
"""

from .dense_local import DenseLocalEmbeddingProvider
from .mock import MockEmbeddingProvider

__all__ = [
    "DenseLocalEmbeddingProvider",
    "MockEmbeddingProvider",
]


def _register_builtin_providers() -> None:
    """Register all built-in embedding providers.

    This is called automatically on module import.
    """
    from shared.embedding.factory import EmbeddingProviderFactory
    from shared.embedding.provider_registry import register_provider_definition

    # Register dense local provider
    EmbeddingProviderFactory.register_provider(
        DenseLocalEmbeddingProvider.INTERNAL_NAME,
        DenseLocalEmbeddingProvider,
    )
    register_provider_definition(DenseLocalEmbeddingProvider.get_definition())

    # Register mock provider
    EmbeddingProviderFactory.register_provider(
        MockEmbeddingProvider.INTERNAL_NAME,
        MockEmbeddingProvider,
    )
    register_provider_definition(MockEmbeddingProvider.get_definition())


# Auto-register on import
_register_builtin_providers()
