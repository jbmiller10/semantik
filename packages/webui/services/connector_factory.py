"""Factory for creating document source connectors.

Plugin Configuration:
    When creating connectors, the factory checks the shared plugin state file
    for plugin-specific configuration. This allows external plugins to receive
    configuration (e.g., API keys, default settings) set via WebUI.

    The plugin config is merged with source-specific config, with source config
    taking precedence (per-source settings override plugin defaults).
"""

from __future__ import annotations

import logging
from typing import Any, cast

from shared.plugins.loader import load_plugins
from shared.plugins.registry import PluginSource, plugin_registry
from shared.plugins.state import get_plugin_config

logger = logging.getLogger(__name__)


class ConnectorFactory:
    """Factory for creating connector instances by source type."""

    @classmethod
    def get_connector(
        cls,
        source_type: str,
        config: dict[str, Any],
    ) -> Any:
        """Create a connector instance for the given source type.

        Args:
            source_type: Type of connector (e.g., "local_directory", "git_repo")
            config: Source-specific configuration dict

        Returns:
            Initialized connector instance

        Note:
            Plugin-specific config from state file is merged with source config.
            Source config takes precedence (per-source settings override defaults).
        """
        normalized_type = source_type.lower().strip()
        load_plugins(plugin_types={"connector"})
        record = plugin_registry.get("connector", normalized_type)
        if record is None:
            available = cls.list_available_types()
            raise ValueError(f"Unknown source type: {source_type!r}. Available types: {', '.join(available)}")
        if record.source == PluginSource.EXTERNAL and plugin_registry.is_disabled(record.plugin_id):
            raise ValueError(f"Connector plugin '{source_type}' is disabled")

        # Load plugin config from state file and merge with source config
        # Source config takes precedence (per-source settings override plugin defaults)
        plugin_config = get_plugin_config(record.plugin_id, resolve_secrets=True)
        if plugin_config:
            merged_config = {**plugin_config, **config}
            logger.debug(
                "Merged plugin config for connector '%s' (plugin defaults + source overrides)",
                normalized_type,
            )
        else:
            merged_config = config

        logger.debug("Creating connector for source_type=%s", normalized_type)
        return record.plugin_class(merged_config)

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
        return cast(type, record.plugin_class)

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
