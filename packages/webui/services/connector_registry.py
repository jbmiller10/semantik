"""Connector registry derived from connector classes."""

from __future__ import annotations

from typing import Any, TypedDict

from shared.plugins.loader import load_plugins
from shared.plugins.registry import PluginSource, plugin_registry


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


def _build_definition(connector_cls: type) -> ConnectorDefinition:
    metadata = getattr(connector_cls, "METADATA", {}) or {}
    return {
        "name": metadata.get("name") or metadata.get("display_name") or connector_cls.__name__,
        "description": metadata.get("description", ""),
        "icon": metadata.get("icon", "plug"),
        "fields": list(getattr(connector_cls, "get_config_fields")()),
        "secrets": list(getattr(connector_cls, "get_secret_fields")()),
        "supports_sync": metadata.get("supports_sync", True),
        "preview_endpoint": metadata.get("preview_endpoint", ""),
    }


def get_connector_catalog() -> dict[str, ConnectorDefinition]:
    """Return connector catalog derived from registered connectors."""
    load_plugins(plugin_types={"connector"})
    connectors = plugin_registry.get_by_type("connector")
    disabled = plugin_registry.disabled_ids()
    return {
        plugin_id: _build_definition(record.plugin_class)
        for plugin_id, record in connectors.items()
        if not (record.source == PluginSource.EXTERNAL and plugin_id in disabled)
    }


def get_connector_definition(connector_type: str) -> ConnectorDefinition | None:
    """Return connector definition for a specific type."""
    load_plugins(plugin_types={"connector"})
    record = plugin_registry.get("connector", connector_type)
    if record is None:
        return None
    if record.source == PluginSource.EXTERNAL and plugin_registry.is_disabled(record.plugin_id):
        return None
    return _build_definition(record.plugin_class)
