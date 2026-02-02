"""Factory for creating embedding provider instances.

This module provides the central dispatch point for obtaining embedding providers.
The factory auto-detects the appropriate provider based on model name.

Plugin Configuration:
    When creating providers, the factory checks the shared plugin state file
    for plugin-specific configuration. This allows external plugins to receive
    configuration (e.g., API keys, model settings) set via WebUI.
"""

from __future__ import annotations

import inspect
import logging
import sys
from threading import Lock
from typing import TYPE_CHECKING, Any

from shared.plugins.state import get_plugin_config

# Import shims exist for this repo (`shared/` points at `packages/shared/`).
# Ensure this module is a singleton even if imported via both names.
if __name__ in {"shared.embedding.factory", "packages.shared.embedding.factory"}:
    sys.modules.setdefault("shared.embedding.factory", sys.modules[__name__])
    sys.modules.setdefault("packages.shared.embedding.factory", sys.modules[__name__])

if TYPE_CHECKING:
    from shared.config.vecpipe import VecpipeConfig
    from shared.plugins.protocols import EmbeddingProtocol

    from .plugin_base import BaseEmbeddingPlugin

logger = logging.getLogger(__name__)

# Registry of provider classes - maps internal_name to provider class
# Accepts both ABC-based plugins (BaseEmbeddingPlugin) and Protocol-based plugins
_PROVIDER_CLASSES: dict[str, type[BaseEmbeddingPlugin] | type[EmbeddingProtocol]] = {}
_PROVIDER_CLASSES_LOCK = Lock()


def _provider_accepts_kwarg(provider_cls: type[Any], kwarg_name: str) -> bool:
    """Return True if provider_cls.__init__ can accept kwarg_name."""

    try:
        signature = inspect.signature(provider_cls.__init__)
    except (TypeError, ValueError):
        # If we can't introspect, be permissive and let the constructor raise if needed.
        return True

    if kwarg_name in signature.parameters:
        return True

    return any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())


def _merge_plugin_config_into_config(config: Any, plugin_config: dict[str, Any]) -> dict[str, Any]:
    """Merge plugin_config into config for providers that don't accept plugin_config kwarg.

    Explicit config values win over plugin_config values.
    """

    if config is None:
        return dict(plugin_config)
    if isinstance(config, dict):
        return {**plugin_config, **config}
    logger.warning(
        "Embedding provider config is %s, expected dict for protocol provider; using plugin state config only",
        type(config).__name__,
    )
    return dict(plugin_config)


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
        provider_class: type[BaseEmbeddingPlugin] | type[EmbeddingProtocol],
    ) -> None:
        """Register a provider class.

        Accepts both ABC-based (inheriting from BaseEmbeddingPlugin) and
        Protocol-based (implementing EmbeddingProtocol) provider classes.

        Args:
            internal_name: Internal identifier for the provider
            provider_class: The provider class to register (ABC or Protocol)
        """
        with _PROVIDER_CLASSES_LOCK:
            if internal_name in _PROVIDER_CLASSES:
                logger.warning("Embedding provider '%s' already registered, skipping", internal_name)
                return
            _PROVIDER_CLASSES[internal_name] = provider_class
        logger.debug("Registered embedding provider: %s -> %s", internal_name, provider_class.__name__)

    @classmethod
    def unregister_provider(cls, internal_name: str) -> None:
        """Unregister a provider class.

        Primarily used for testing to clean up registered providers.

        Args:
            internal_name: Internal identifier of the provider to remove
        """
        with _PROVIDER_CLASSES_LOCK:
            if internal_name in _PROVIDER_CLASSES:
                del _PROVIDER_CLASSES[internal_name]
                logger.debug("Unregistered embedding provider: %s", internal_name)

    @classmethod
    def create_provider(
        cls,
        model_name: str,
        config: VecpipeConfig | None = None,
        **kwargs: Any,
    ) -> BaseEmbeddingPlugin | EmbeddingProtocol:
        """Create a provider instance for the given model.

        Auto-detects the appropriate provider based on model name using
        each provider's supports_model() class method.

        Returns an instance that satisfies the EmbeddingProtocol interface,
        which may be either an ABC-based or Protocol-based implementation.

        Plugin Configuration:
            If no plugin_config is provided in kwargs, the factory will
            check the shared plugin state file for configuration. This
            allows plugins to receive settings (API keys, etc.) from WebUI.

        Args:
            model_name: HuggingFace model name or other model identifier
            config: Optional VecpipeConfig for provider configuration
            **kwargs: Additional kwargs passed to provider constructor.
                      May include 'plugin_config' for plugin-specific settings.

        Returns:
            An uninitialized provider instance satisfying EmbeddingProtocol

        Raises:
            ValueError: If no provider supports the model
        """
        with _PROVIDER_CLASSES_LOCK:
            providers = list(_PROVIDER_CLASSES.items())
        for internal_name, provider_cls in providers:
            if provider_cls.supports_model(model_name):
                logger.info(
                    "Creating provider '%s' (%s) for model '%s'",
                    internal_name,
                    provider_cls.__name__,
                    model_name,
                )
                # Load plugin config from state file if not explicitly provided
                if "plugin_config" not in kwargs:
                    plugin_id = getattr(provider_cls, "API_ID", None) or internal_name
                    state_config = get_plugin_config(plugin_id, resolve_secrets=True)
                    if state_config:
                        kwargs["plugin_config"] = state_config
                        logger.debug("Loaded config for plugin '%s' from state file", plugin_id)

                plugin_config = kwargs.get("plugin_config")
                if plugin_config and not _provider_accepts_kwarg(provider_cls, "plugin_config"):
                    kwargs.pop("plugin_config", None)
                    config = _merge_plugin_config_into_config(config, plugin_config)

                return provider_cls(config=config, **kwargs)

        with _PROVIDER_CLASSES_LOCK:
            available = list(_PROVIDER_CLASSES.keys())
        raise ValueError(f"No provider found for model: {model_name}. Available providers: {available}")

    @classmethod
    def create_provider_by_name(
        cls,
        provider_name: str,
        config: VecpipeConfig | None = None,
        **kwargs: Any,
    ) -> BaseEmbeddingPlugin | EmbeddingProtocol:
        """Create a provider by its internal name.

        Use this when you want to explicitly specify the provider rather
        than relying on auto-detection.

        Returns an instance that satisfies the EmbeddingProtocol interface,
        which may be either an ABC-based or Protocol-based implementation.

        Args:
            provider_name: Internal name of the provider
            config: Optional VecpipeConfig for provider configuration
            **kwargs: Additional kwargs passed to provider constructor.
                      May include 'plugin_config' for plugin-specific settings.

        Returns:
            An uninitialized provider instance satisfying EmbeddingProtocol

        Raises:
            ValueError: If provider name is not registered
        """
        with _PROVIDER_CLASSES_LOCK:
            if provider_name not in _PROVIDER_CLASSES:
                available = list(_PROVIDER_CLASSES.keys())
                raise ValueError(f"Unknown provider: {provider_name}. Available: {available}")

            provider_cls = _PROVIDER_CLASSES[provider_name]

        # Load plugin config from state file if not explicitly provided
        if "plugin_config" not in kwargs:
            plugin_id = getattr(provider_cls, "API_ID", None) or provider_name
            state_config = get_plugin_config(plugin_id, resolve_secrets=True)
            if state_config:
                kwargs["plugin_config"] = state_config
                logger.debug("Loaded config for plugin '%s' from state file", plugin_id)

        logger.info("Creating provider by name: %s (%s)", provider_name, provider_cls.__name__)
        plugin_config = kwargs.get("plugin_config")
        if plugin_config and not _provider_accepts_kwarg(provider_cls, "plugin_config"):
            kwargs.pop("plugin_config", None)
            config = _merge_plugin_config_into_config(config, plugin_config)

        return provider_cls(config=config, **kwargs)

    @classmethod
    def get_provider_for_model(cls, model_name: str) -> str | None:
        """Get the internal provider name for a model.

        Args:
            model_name: HuggingFace model name or other model identifier

        Returns:
            Internal provider name if found, None otherwise
        """
        with _PROVIDER_CLASSES_LOCK:
            providers = list(_PROVIDER_CLASSES.items())
        for internal_name, provider_cls in providers:
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
        with _PROVIDER_CLASSES_LOCK:
            return list(_PROVIDER_CLASSES.keys())

    @classmethod
    def get_provider_class(cls, internal_name: str) -> type[BaseEmbeddingPlugin] | type[EmbeddingProtocol] | None:
        """Get the provider class for an internal name.

        Returns the registered class, which may be ABC-based or Protocol-based.

        Args:
            internal_name: Internal provider name

        Returns:
            Provider class if found, None otherwise
        """
        with _PROVIDER_CLASSES_LOCK:
            return _PROVIDER_CLASSES.get(internal_name)

    @classmethod
    def clear_providers(cls) -> None:
        """Clear all registered providers.

        Primarily used for testing to reset state.
        """
        with _PROVIDER_CLASSES_LOCK:
            _PROVIDER_CLASSES.clear()
        logger.debug("Cleared all embedding providers")


def get_all_supported_models() -> list[dict[str, Any]]:
    """Get model configurations from all registered providers.

    Aggregates models from all providers for API/UI exposure.
    Handles both ABC-based and Protocol-based providers gracefully.

    Returns:
        List of model config dicts with provider information
    """
    models: list[dict[str, Any]] = []

    with _PROVIDER_CLASSES_LOCK:
        providers = list(_PROVIDER_CLASSES.items())
    for internal_name, provider_cls in providers:
        try:
            # list_supported_models is optional for Protocol-based plugins
            list_models_fn = getattr(provider_cls, "list_supported_models", None)
            if list_models_fn is None or not callable(list_models_fn):
                continue
            provider_models = list_models_fn()
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

    Handles both ABC-based and Protocol-based providers gracefully.

    Args:
        model_name: HuggingFace model name or other model identifier

    Returns:
        ModelConfig if found by any provider, None otherwise
    """
    with _PROVIDER_CLASSES_LOCK:
        providers = list(_PROVIDER_CLASSES.values())
    for provider_cls in providers:
        if provider_cls.supports_model(model_name):
            # get_model_config is optional for Protocol-based plugins
            get_config_fn = getattr(provider_cls, "get_model_config", None)
            if get_config_fn is not None and callable(get_config_fn):
                config = get_config_fn(model_name)
                if config is not None:
                    return config
    return None


def resolve_model_config(model_name: str) -> Any:
    """Resolve model configuration from providers or built-in configs.

    Provides unified model config resolution that:
    1. First checks registered providers (including plugins)
    2. Falls back to built-in MODEL_CONFIGS if no provider has the config

    This ensures plugin models are visible to all code that needs
    model configuration (batch sizing, dimension validation, etc.).

    Args:
        model_name: HuggingFace model name or other model identifier

    Returns:
        ModelConfig if found by any provider or in built-ins, None otherwise

    Example:
        >>> config = resolve_model_config("my-plugin/custom-model")
        >>> if config:
        ...     dimension = config.dimension
        ... else:
        ...     dimension = DEFAULT_DIMENSION  # safe fallback
    """
    # First try providers (includes plugins)
    config = get_model_config_from_providers(model_name)
    if config is not None:
        return config

    # Fall back to built-in configs
    from .models import get_model_config as builtin_get_model_config

    return builtin_get_model_config(model_name)
