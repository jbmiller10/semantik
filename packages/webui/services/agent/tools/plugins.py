"""Agent tools for plugin discovery and inspection.

These tools allow the agent to explore available plugins and their capabilities
when building pipeline configurations.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from shared.plugins.registry import plugin_registry
from webui.services.agent.tools.base import BaseTool

logger = logging.getLogger(__name__)


class ListPluginsTool(BaseTool):
    """List available plugins, optionally filtered by type.

    Returns a summary of each plugin including ID, display name, description,
    and type. Useful for discovering what processing options are available.
    """

    NAME: ClassVar[str] = "list_plugins"
    DESCRIPTION: ClassVar[str] = (
        "List available plugins for pipeline configuration. "
        "Can filter by plugin type (parser, chunker, embedder, extractor, reranker, sparse_indexer). "
        "Returns plugin IDs, names, descriptions, and types."
    )
    PARAMETERS: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "plugin_type": {
                "type": "string",
                "description": (
                    "Optional type to filter by. Valid types: "
                    "parser, chunker, embedder, extractor, reranker, sparse_indexer, connector"
                ),
                "enum": [
                    "parser",
                    "chunker",
                    "embedding",
                    "extractor",
                    "reranker",
                    "sparse_indexer",
                    "connector",
                ],
            },
            "include_disabled": {
                "type": "boolean",
                "description": "Whether to include disabled plugins in the list",
                "default": False,
            },
        },
        "required": [],
    }

    async def execute(
        self,
        plugin_type: str | None = None,
        include_disabled: bool = False,
    ) -> dict[str, Any]:
        """Execute the plugin listing.

        Args:
            plugin_type: Optional type to filter by
            include_disabled: Whether to include disabled plugins

        Returns:
            Dictionary with plugins list and metadata
        """
        try:
            # Get all plugin records, optionally filtered by type
            records = plugin_registry.list_records(plugin_type=plugin_type)

            plugins = []
            for record in records:
                # Skip disabled plugins unless explicitly requested
                if not include_disabled and plugin_registry.is_disabled(record.plugin_id):
                    continue

                manifest = record.manifest
                plugin_info = {
                    "id": record.plugin_id,
                    "type": record.plugin_type,
                    "display_name": manifest.display_name,
                    "description": manifest.description,
                    "version": record.plugin_version,
                    "is_builtin": record.source.value == "builtin",
                    "is_disabled": plugin_registry.is_disabled(record.plugin_id),
                }

                # Include agent hints if available
                if manifest.agent_hints:
                    plugin_info["agent_hints"] = {
                        "purpose": manifest.agent_hints.purpose,
                        "best_for": manifest.agent_hints.best_for,
                        "not_recommended_for": manifest.agent_hints.not_recommended_for,
                    }

                plugins.append(plugin_info)

            # Sort by type, then by display name
            plugins.sort(key=lambda p: (p["type"], p["display_name"]))

            return {
                "plugins": plugins,
                "count": len(plugins),
                "filter": plugin_type,
                "available_types": sorted(plugin_registry.list_types()),
            }

        except Exception as e:
            logger.error(f"Failed to list plugins: {e}", exc_info=True)
            return {
                "error": str(e),
                "plugins": [],
                "count": 0,
            }


class GetPluginDetailsTool(BaseTool):
    """Get detailed information about a specific plugin.

    Returns the full manifest including configuration schema, capabilities,
    and agent hints for reasoning about when to use the plugin.
    """

    NAME: ClassVar[str] = "get_plugin_details"
    DESCRIPTION: ClassVar[str] = (
        "Get detailed information about a specific plugin by ID. "
        "Returns full manifest including configuration schema, capabilities, "
        "input/output types, and usage hints."
    )
    PARAMETERS: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "plugin_id": {
                "type": "string",
                "description": "The unique ID of the plugin to get details for",
            },
        },
        "required": ["plugin_id"],
    }

    async def execute(self, plugin_id: str) -> dict[str, Any]:
        """Execute the plugin details lookup.

        Args:
            plugin_id: The ID of the plugin to look up

        Returns:
            Dictionary with plugin details or error
        """
        try:
            record = plugin_registry.find_by_id(plugin_id)

            if not record:
                return {
                    "found": False,
                    "error": f"Plugin '{plugin_id}' not found",
                    "available_plugins": plugin_registry.list_ids()[:20],  # Sample
                }

            manifest = record.manifest

            # Build detailed response
            details: dict[str, Any] = {
                "found": True,
                "id": record.plugin_id,
                "type": record.plugin_type,
                "version": record.plugin_version,
                "display_name": manifest.display_name,
                "description": manifest.description,
                "author": manifest.author,
                "license": manifest.license,
                "homepage": manifest.homepage,
                "source": record.source.value,
                "is_disabled": plugin_registry.is_disabled(record.plugin_id),
                "requires": manifest.requires,
                "semantik_version": manifest.semantik_version,
                "capabilities": manifest.capabilities,
            }

            # Include agent hints if available
            if manifest.agent_hints:
                details["agent_hints"] = manifest.agent_hints.to_dict()

            return details

        except Exception as e:
            logger.error(f"Failed to get plugin details for '{plugin_id}': {e}", exc_info=True)
            return {
                "found": False,
                "error": str(e),
            }
