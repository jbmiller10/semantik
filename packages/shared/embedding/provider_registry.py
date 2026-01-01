"""Canonical embedding provider registry.

This module centralizes all embedding provider metadata so that the service layer,
API endpoints, and factories can share a single source of truth.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from functools import lru_cache
from threading import Lock
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .plugin_base import EmbeddingProviderDefinition

logger = logging.getLogger(__name__)
_REGISTRY_LOCK = Lock()

# Registry storage - maps api_id to definition
_PROVIDERS: dict[str, EmbeddingProviderDefinition] = {}


def register_provider_definition(definition: EmbeddingProviderDefinition) -> None:
    """Register a provider definition at runtime.

    This is called by built-in providers during module import and by
    the plugin loader when discovering external plugins.

    Args:
        definition: The provider definition to register
    """
    with _REGISTRY_LOCK:
        if definition.api_id in _PROVIDERS:
            logger.warning("Embedding provider '%s' already registered, skipping", definition.api_id)
            return
        _PROVIDERS[definition.api_id] = definition
        _clear_caches()


def unregister_provider_definition(api_id: str) -> None:
    """Unregister a provider definition.

    Primarily used for testing to clean up registered providers.

    Args:
        api_id: The API identifier of the provider to remove
    """
    with _REGISTRY_LOCK:
        if api_id in _PROVIDERS:
            del _PROVIDERS[api_id]
            _clear_caches()


def _clear_caches() -> None:
    """Clear lru-cached lookups after registry mutation."""
    list_provider_definitions.cache_clear()
    get_api_to_internal_map.cache_clear()
    get_internal_to_api_map.cache_clear()
    list_provider_metadata.cache_clear()


@lru_cache(maxsize=1)
def list_provider_definitions() -> tuple[EmbeddingProviderDefinition, ...]:
    """List all registered provider definitions.

    Returns:
        Tuple of all registered EmbeddingProviderDefinition objects
    """
    with _REGISTRY_LOCK:
        return tuple(_PROVIDERS.values())


@lru_cache(maxsize=1)
def get_api_to_internal_map() -> dict[str, str]:
    """Return mapping from API provider identifiers to internal names.

    Returns:
        Dict mapping api_id -> internal_id
    """
    with _REGISTRY_LOCK:
        return {k: v.internal_id for k, v in _PROVIDERS.items()}


@lru_cache(maxsize=1)
def get_internal_to_api_map() -> dict[str, str]:
    """Return mapping from internal names to API identifiers.

    When multiple providers share an internal_id, returns the first registered.

    Returns:
        Dict mapping internal_id -> api_id
    """
    with _REGISTRY_LOCK:
        mapping: dict[str, str] = {}
        for api_id, defn in _PROVIDERS.items():
            mapping.setdefault(defn.internal_id, api_id)
        return mapping


def get_provider_definition(identifier: str) -> EmbeddingProviderDefinition | None:
    """Get provider definition by API ID or internal ID.

    Args:
        identifier: Either the api_id or internal_id of the provider

    Returns:
        The provider definition if found, None otherwise
    """
    # Try direct API ID lookup first
    with _REGISTRY_LOCK:
        if identifier in _PROVIDERS:
            return _PROVIDERS[identifier]

        # Try internal ID lookup
        for defn in _PROVIDERS.values():
            if defn.internal_id == identifier:
                return defn

    return None


@lru_cache(maxsize=1)
def list_provider_metadata() -> tuple[dict[str, Any], ...]:
    """Return metadata for all providers formatted for API consumers.

    Returns:
        Tuple of metadata dicts suitable for JSON serialization
    """
    return tuple(defn.to_metadata_dict() for defn in list_provider_definitions())


def list_provider_metadata_list() -> list[dict[str, Any]]:
    """Return metadata for all providers as a list.

    Convenience method that returns a mutable list instead of tuple.

    Returns:
        List of metadata dicts suitable for JSON serialization
    """
    return [deepcopy(m) for m in list_provider_metadata()]


def get_provider_metadata(identifier: str) -> dict[str, Any] | None:
    """Get metadata for a specific provider.

    Args:
        identifier: Either the api_id or internal_id of the provider

    Returns:
        Metadata dict if found, None otherwise
    """
    defn = get_provider_definition(identifier)
    if defn is None:
        return None
    return defn.to_metadata_dict()


def is_provider_registered(identifier: str) -> bool:
    """Check if a provider is registered.

    Args:
        identifier: Either the api_id or internal_id of the provider

    Returns:
        True if provider is registered, False otherwise
    """
    return get_provider_definition(identifier) is not None


def get_providers_by_type(provider_type: str) -> list[EmbeddingProviderDefinition]:
    """Get all providers of a specific type.

    Args:
        provider_type: "local", "remote", or "hybrid"

    Returns:
        List of matching provider definitions
    """
    return [defn for defn in list_provider_definitions() if defn.provider_type == provider_type]


def get_registered_provider_ids() -> list[str]:
    """Get all registered provider API IDs.

    Returns:
        List of api_id strings
    """
    with _REGISTRY_LOCK:
        return list(_PROVIDERS.keys())
