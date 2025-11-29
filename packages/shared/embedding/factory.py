"""Factory for creating embedding provider instances.

This module provides the central dispatch point for obtaining embedding providers.
The factory auto-detects the appropriate provider based on model name.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from shared.config.vecpipe import VecpipeConfig

    from .plugin_base import BaseEmbeddingPlugin

logger = logging.getLogger(__name__)

# Registry of provider classes - maps internal_name to provider class
_PROVIDER_CLASSES: dict[str, type[BaseEmbeddingPlugin]] = {}


class EmbeddingProviderFactory:
    """Factory for creating embedding provider instances.

    This factory manages provider registration and creation. It can auto-detect
    the appropriate provider for a given model name using the supports_model()
    class method on each registered provider.

    Example:
        # Create provider for a specific model (auto-detection)
        provider = EmbeddingProviderFactory.create_provider(
            model_name="Qwen/Qwen3-Embedding-0.6B",
            config=vecpipe_config,
        )

        # Create provider by explicit name
        provider = EmbeddingProviderFactory.create_provider_by_name(
            provider_name="dense_local",
            config=vecpipe_config,
        )
    """

    @classmethod
    def register_provider(
        cls,
        internal_name: str,
        provider_class: type[BaseEmbeddingPlugin],
    ) -> None:
        """Register a provider class.

        Args:
            internal_name: Internal identifier for the provider
            provider_class: The provider class to register
        """
        _PROVIDER_CLASSES[internal_name] = provider_class
        logger.debug("Registered embedding provider: %s -> %s", internal_name, provider_class.__name__)

    @classmethod
    def unregister_provider(cls, internal_name: str) -> None:
        """Unregister a provider class.

        Primarily used for testing to clean up registered providers.

        Args:
            internal_name: Internal identifier of the provider to remove
        """
        if internal_name in _PROVIDER_CLASSES:
            del _PROVIDER_CLASSES[internal_name]
            logger.debug("Unregistered embedding provider: %s", internal_name)

    @classmethod
    def create_provider(
        cls,
        model_name: str,
        config: VecpipeConfig | None = None,
        **kwargs: Any,
    ) -> BaseEmbeddingPlugin:
        """Create a provider instance for the given model.

        Auto-detects the appropriate provider based on model name using
        each provider's supports_model() class method.

        Args:
            model_name: HuggingFace model name or other model identifier
            config: Optional VecpipeConfig for provider configuration
            **kwargs: Additional kwargs passed to provider constructor

        Returns:
            An uninitialized provider instance

        Raises:
            ValueError: If no provider supports the model
        """
        for internal_name, provider_cls in _PROVIDER_CLASSES.items():
            if provider_cls.supports_model(model_name):
                logger.info(
                    "Creating provider '%s' (%s) for model '%s'",
                    internal_name,
                    provider_cls.__name__,
                    model_name,
                )
                return provider_cls(config=config, **kwargs)

        available = list(_PROVIDER_CLASSES.keys())
        raise ValueError(
            f"No provider found for model: {model_name}. "
            f"Available providers: {available}"
        )

    @classmethod
    def create_provider_by_name(
        cls,
        provider_name: str,
        config: VecpipeConfig | None = None,
        **kwargs: Any,
    ) -> BaseEmbeddingPlugin:
        """Create a provider by its internal name.

        Use this when you want to explicitly specify the provider rather
        than relying on auto-detection.

        Args:
            provider_name: Internal name of the provider
            config: Optional VecpipeConfig for provider configuration
            **kwargs: Additional kwargs passed to provider constructor

        Returns:
            An uninitialized provider instance

        Raises:
            ValueError: If provider name is not registered
        """
        if provider_name not in _PROVIDER_CLASSES:
            available = list(_PROVIDER_CLASSES.keys())
            raise ValueError(
                f"Unknown provider: {provider_name}. "
                f"Available: {available}"
            )

        provider_cls = _PROVIDER_CLASSES[provider_name]
        logger.info("Creating provider by name: %s (%s)", provider_name, provider_cls.__name__)
        return provider_cls(config=config, **kwargs)

    @classmethod
    def get_provider_for_model(cls, model_name: str) -> str | None:
        """Get the internal provider name for a model.

        Args:
            model_name: HuggingFace model name or other model identifier

        Returns:
            Internal provider name if found, None otherwise
        """
        for internal_name, provider_cls in _PROVIDER_CLASSES.items():
            if provider_cls.supports_model(model_name):
                return internal_name
        return None

    @classmethod
    def is_model_supported(cls, model_name: str) -> bool:
        """Check if any provider supports the given model.

        Args:
            model_name: HuggingFace model name or other model identifier

        Returns:
            True if a provider supports the model, False otherwise
        """
        return cls.get_provider_for_model(model_name) is not None

    @classmethod
    def list_available_providers(cls) -> list[str]:
        """List all registered provider internal names.

        Returns:
            List of internal provider names
        """
        return list(_PROVIDER_CLASSES.keys())

    @classmethod
    def get_provider_class(cls, internal_name: str) -> type[BaseEmbeddingPlugin] | None:
        """Get the provider class for an internal name.

        Args:
            internal_name: Internal provider name

        Returns:
            Provider class if found, None otherwise
        """
        return _PROVIDER_CLASSES.get(internal_name)

    @classmethod
    def clear_providers(cls) -> None:
        """Clear all registered providers.

        Primarily used for testing to reset state.
        """
        _PROVIDER_CLASSES.clear()
        logger.debug("Cleared all embedding providers")


def get_all_supported_models() -> list[dict[str, Any]]:
    """Get model configurations from all registered providers.

    Aggregates models from all providers for API/UI exposure.

    Returns:
        List of model config dicts with provider information
    """
    models: list[dict[str, Any]] = []

    for internal_name, provider_cls in _PROVIDER_CLASSES.items():
        try:
            provider_models = provider_cls.list_supported_models()
            for model_config in provider_models:
                model_dict = model_config.to_dict() if hasattr(model_config, "to_dict") else {}
                model_dict["provider"] = internal_name
                model_dict["model_name"] = getattr(model_config, "name", "")
                models.append(model_dict)
        except Exception as e:
            logger.warning("Failed to get models from provider %s: %s", internal_name, e)

    return models


def get_model_config_from_providers(model_name: str) -> Any:
    """Get model configuration from the appropriate provider.

    Args:
        model_name: HuggingFace model name or other model identifier

    Returns:
        ModelConfig if found by any provider, None otherwise
    """
    for provider_cls in _PROVIDER_CLASSES.values():
        if provider_cls.supports_model(model_name):
            config = provider_cls.get_model_config(model_name)
            if config is not None:
                return config
    return None
