"""Connector registry derived from connector classes.

This module provides cached access to connector definitions, avoiding
redundant plugin loading on every request. Cache invalidation should be
triggered after plugin install/uninstall/enable/disable operations.
"""

from __future__ import annotations

from functools import lru_cache
from threading import RLock
from typing import Any, TypedDict

from shared.connectors.base import BaseConnector
from shared.plugins.loader import load_plugins
from shared.plugins.registry import PluginSource, plugin_registry

# Cache lock for thread-safe operations
_CONNECTOR_CACHE_LOCK = RLock()
_PLUGINS_LOADED = False


def _ensure_plugins_loaded() -> None:
    """Ensure connector plugins are loaded (idempotent)."""
    global _PLUGINS_LOADED
    if not _PLUGINS_LOADED:
        with _CONNECTOR_CACHE_LOCK:
            if not _PLUGINS_LOADED:
                load_plugins(plugin_types={"connector"})
                _PLUGINS_LOADED = True


def invalidate_connector_cache() -> None:
    """Invalidate connector caches after plugin changes.

    Call this after:
    - Installing a new connector plugin
    - Uninstalling a connector plugin
    - Enabling/disabling a connector

    This clears both the plugin load state and the LRU caches.
    """
    global _PLUGINS_LOADED
    with _CONNECTOR_CACHE_LOCK:
        _PLUGINS_LOADED = False
        get_connector_catalog.cache_clear()
        get_connector_definition.cache_clear()


class FieldOption(TypedDict, total=False):
    value: str
    label: str


class ShowWhen(TypedDict, total=False):
    field: str
    equals: str | list[str]


class FieldDefinition(TypedDict, total=False):
    name: str
    type: str
    label: str
    description: str
    required: bool
    default: Any
    placeholder: str
    options: list[FieldOption]
    show_when: ShowWhen
    min: int | float
    max: int | float
    step: int | float


class SecretDefinition(TypedDict, total=False):
    name: str
    label: str
    description: str
    required: bool
    show_when: ShowWhen
    is_multiline: bool


class ConnectorDefinition(TypedDict, total=False):
    name: str
    description: str
    icon: str
    fields: list[FieldDefinition]
    secrets: list[SecretDefinition]
    supports_sync: bool
    preview_endpoint: str


def _build_definition(connector_cls: type[BaseConnector]) -> ConnectorDefinition:
    metadata = getattr(connector_cls, "METADATA", {}) or {}
    return {
        "name": metadata.get("name") or metadata.get("display_name") or connector_cls.__name__,
        "description": metadata.get("description", ""),
        "icon": metadata.get("icon", "plug"),
        "fields": list(connector_cls.get_config_fields()),
        "secrets": list(connector_cls.get_secret_fields()),
        "supports_sync": metadata.get("supports_sync", True),
        "preview_endpoint": metadata.get("preview_endpoint", ""),
    }


@lru_cache(maxsize=1)
def get_connector_catalog() -> dict[str, ConnectorDefinition]:
    """Return connector catalog derived from registered connectors.

    Results are cached for performance. Call invalidate_connector_cache()
    after plugin changes to clear the cache.
    """
    with _CONNECTOR_CACHE_LOCK:
        _ensure_plugins_loaded()
        connectors = plugin_registry.get_by_type("connector")
        disabled = plugin_registry.disabled_ids()
        return {
            plugin_id: _build_definition(record.plugin_class)
            for plugin_id, record in connectors.items()
            if not (record.source == PluginSource.EXTERNAL and plugin_id in disabled)
        }


@lru_cache(maxsize=64)
def get_connector_definition(connector_type: str) -> ConnectorDefinition | None:
    """Return connector definition for a specific type.

    Results are cached for performance. Call invalidate_connector_cache()
    after plugin changes to clear the cache.
    """
    # Use the cached catalog lookup for consistency
    catalog = get_connector_catalog()
    return catalog.get(connector_type)
