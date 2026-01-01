"""Factory for creating document source connectors."""

from __future__ import annotations

import logging
from typing import Any

from shared.plugins.loader import load_plugins
from shared.plugins.registry import PluginSource, plugin_registry

logger = logging.getLogger(__name__)


class ConnectorFactory:
    """Factory for creating connector instances by source type."""

    @classmethod
    def get_connector(
        cls,
        source_type: str,
        config: dict[str, Any],
    ) -> Any:
        """Create a connector instance for the given source type."""
        normalized_type = source_type.lower().strip()
        load_plugins(plugin_types={"connector"})
        record = plugin_registry.get("connector", normalized_type)
        if record is None:
            available = cls.list_available_types()
            raise ValueError(f"Unknown source type: {source_type!r}. Available types: {', '.join(available)}")
        if record.source == PluginSource.EXTERNAL and plugin_registry.is_disabled(record.plugin_id):
            raise ValueError(f"Connector plugin '{source_type}' is disabled")
        logger.debug("Creating connector for source_type=%s", normalized_type)
        return record.plugin_class(config)

    @classmethod
    def get_connector_class(cls, source_type: str) -> type | None:
        """Return connector class for a source type."""
        normalized_type = source_type.lower().strip()
        load_plugins(plugin_types={"connector"})
        record = plugin_registry.get("connector", normalized_type)
        if record is None:
            return None
        if record.source == PluginSource.EXTERNAL and plugin_registry.is_disabled(record.plugin_id):
            return None
        return record.plugin_class

    @classmethod
    def list_available_types(cls) -> list[str]:
        """List all registered connector types."""
        load_plugins(plugin_types={"connector"})
        disabled = plugin_registry.disabled_ids()
        return sorted(
            plugin_id
            for plugin_id, record in plugin_registry.get_by_type("connector").items()
            if not (record.source == PluginSource.EXTERNAL and plugin_id in disabled)
        )
