"""Runtime loader for external embedding provider plugins.

Plugins register an entry point under the group ``semantik.embedding_providers``.
Each entry point should resolve to a BaseEmbeddingPlugin subclass or a
callable returning one. Minimal contract expected on the class:

    INTERNAL_NAME: str   # required internal identifier
    API_ID: str          # required API-facing identifier
    PROVIDER_TYPE: str   # required: "local", "remote", or "hybrid"
    METADATA: dict       # optional metadata for UI/API

The plugin must implement:
    - get_definition() -> EmbeddingProviderDefinition
    - supports_model(model_name: str) -> bool
    - All BaseEmbeddingService abstract methods
"""

from __future__ import annotations

import logging
import os
from importlib import metadata
from typing import Any

logger = logging.getLogger(__name__)

ENTRYPOINT_GROUP = "semantik.embedding_providers"
ENV_FLAG = "SEMANTIK_ENABLE_EMBEDDING_PLUGINS"

# Module-level flag for idempotent plugin loading
_plugins_loaded = False
_registered_plugins: list[str] = []


def _should_enable_plugins() -> bool:
    """Check env flag to allow disabling plugin loading."""
    value = os.getenv(ENV_FLAG, "true").lower()
    return value not in {"0", "false", "no", "off"}


def _coerce_class(obj: Any) -> type | None:
    """Return a class object if obj is a class or a callable returning a class."""
    if isinstance(obj, type):
        return obj
    if callable(obj):
        maybe_cls = obj()
        if isinstance(maybe_cls, type):
            return maybe_cls
        # If callable returned an instance (not None), extract its class
        if maybe_cls is not None:
            return maybe_cls.__class__  # type: ignore[no-any-return]
    return None


def _validate_plugin_contract(cls: type) -> tuple[bool, str | None]:
    """Validate that a class meets the embedding plugin contract.

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check required class attributes
    required_attrs = ["INTERNAL_NAME", "API_ID", "PROVIDER_TYPE"]
    for attr in required_attrs:
        value = getattr(cls, attr, None)
        if not value:
            return False, f"missing required attribute: {attr}"

    # Validate PROVIDER_TYPE value
    provider_type = getattr(cls, "PROVIDER_TYPE", "")
    if provider_type not in ("local", "remote", "hybrid"):
        return False, f"invalid PROVIDER_TYPE: {provider_type}"

    # Check required methods
    required_methods = ["get_definition", "supports_model", "initialize", "embed_texts", "embed_single", "cleanup"]
    for method_name in required_methods:
        method = getattr(cls, method_name, None)
        if not callable(method):
            return False, f"missing required method: {method_name}"

    return True, None


def load_embedding_plugins() -> list[str]:
    """Discover and register embedding provider plugins via entry points.

    This function:
    1. Queries the entry point group 'semantik.embedding_providers'
    2. Loads each entry point and validates the plugin contract
    3. Registers valid plugins with the factory and provider registry

    This function is idempotent - subsequent calls return the cached result
    without re-querying entry points.

    Returns:
        List of api_ids successfully registered.
    """
    global _plugins_loaded, _registered_plugins

    # Idempotent: only load once
    if _plugins_loaded:
        return list(_registered_plugins)

    from .factory import EmbeddingProviderFactory
    from .provider_registry import register_provider_definition

    if not _should_enable_plugins():
        logger.info("Embedding plugins disabled via %s", ENV_FLAG)
        _plugins_loaded = True
        return []

    try:
        eps = metadata.entry_points()
        # Handle both old and new entry_points API
        if hasattr(eps, "select"):
            ep_group = list(eps.select(group=ENTRYPOINT_GROUP))
        else:
            ep_group = list(eps.get(ENTRYPOINT_GROUP, []))
    except Exception as exc:
        logger.warning("Unable to query entry points for embedding plugins: %s", exc)
        return []

    registered: list[str] = []

    for ep in ep_group:
        ep_name = getattr(ep, "name", "unknown")
        try:
            loaded = ep.load()
            plugin_cls = _coerce_class(loaded)

            if plugin_cls is None:
                raise TypeError(f"Entry point {ep_name} did not resolve to a class.")

            # Validate plugin contract
            is_valid, error = _validate_plugin_contract(plugin_cls)
            if not is_valid:
                logger.warning(
                    "Skipping invalid embedding plugin '%s': %s",
                    ep_name,
                    error,
                )
                continue

            # Get definition from plugin
            try:
                # Cast to access get_definition method
                from .plugin_base import BaseEmbeddingPlugin

                assert issubclass(plugin_cls, BaseEmbeddingPlugin)
                definition = plugin_cls.get_definition()
            except Exception as e:
                logger.warning(
                    "Skipping embedding plugin '%s': get_definition() failed: %s",
                    ep_name,
                    e,
                )
                continue

            # Mark as external plugin
            if not definition.is_plugin:
                # Create new definition with is_plugin=True
                from .plugin_base import EmbeddingProviderDefinition

                definition = EmbeddingProviderDefinition(
                    api_id=definition.api_id,
                    internal_id=definition.internal_id,
                    display_name=definition.display_name,
                    description=definition.description,
                    provider_type=definition.provider_type,
                    supports_quantization=definition.supports_quantization,
                    supports_instruction=definition.supports_instruction,
                    supports_batch_processing=definition.supports_batch_processing,
                    supports_asymmetric=definition.supports_asymmetric,
                    supported_models=definition.supported_models,
                    default_config=dict(definition.default_config),
                    performance_characteristics=dict(definition.performance_characteristics),
                    is_plugin=True,
                )

            # Register in provider factory
            EmbeddingProviderFactory.register_provider(
                definition.internal_id,
                plugin_cls,
            )

            # Register metadata for API exposure
            register_provider_definition(definition)

            registered.append(definition.api_id)
            logger.info(
                "Registered embedding plugin '%s' (internal: %s, class: %s)",
                definition.api_id,
                definition.internal_id,
                plugin_cls.__name__,
            )

        except Exception as exc:
            logger.warning(
                "Failed to load embedding plugin %s: %s",
                ep_name,
                exc,
            )
            continue

    # Mark as loaded and cache result
    _plugins_loaded = True
    _registered_plugins = registered
    return registered


def ensure_providers_registered() -> None:
    """Ensure built-in providers are registered.

    This calls the builtin provider registration function directly to ensure
    providers are registered even if the module was already imported and the
    registry was cleared. It's idempotent - calling multiple times is safe.
    """
    # Import and call registration function directly to handle case where
    # module was already imported but registry was cleared
    from .providers import _register_builtin_providers

    _register_builtin_providers()


def _reset_plugin_loader_state() -> None:
    """Reset plugin loader state for testing.

    This allows tests to re-run plugin loading with fresh state.
    Should only be used in tests.
    """
    global _plugins_loaded, _registered_plugins
    _plugins_loaded = False
    _registered_plugins = []
