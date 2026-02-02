"""SDK MCP Server creation for assisted flow.

This module creates an in-process MCP server with all the tools
needed for the pipeline configuration assistant.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from claude_agent_sdk import create_sdk_mcp_server, tool

from shared.plugins.registry import plugin_registry

if TYPE_CHECKING:
    from claude_agent_sdk.types import McpSdkServerConfig

    from webui.services.assisted_flow.context import ToolContext

logger = logging.getLogger(__name__)


def create_mcp_server(ctx: ToolContext) -> McpSdkServerConfig:
    """Create an SDK MCP server with assisted flow tools.

    Args:
        ctx: Shared context for all tools

    Returns:
        SDK MCP server configuration
    """

    @tool(
        "list_plugins",
        "List available plugins for pipeline configuration. Can filter by type.",
        {
            "type": "object",
            "properties": {
                "plugin_type": {
                    "type": "string",
                    "description": "Filter by type: parser, chunker, embedding, extractor, reranker",
                    "enum": ["parser", "chunking", "embedding", "extractor", "reranker", "sparse_indexer"],
                },
                "include_disabled": {
                    "type": "boolean",
                    "description": "Include disabled plugins",
                    "default": False,
                },
            },
        },
    )
    async def list_plugins(args: dict[str, Any]) -> dict[str, Any]:
        """List available plugins."""
        plugin_type = args.get("plugin_type")
        include_disabled = args.get("include_disabled", False)

        try:
            records = plugin_registry.list_records(plugin_type=plugin_type)
            plugins = []

            for record in records:
                if not include_disabled and plugin_registry.is_disabled(record.plugin_id):
                    continue

                manifest = record.manifest
                plugin_info: dict[str, Any] = {
                    "id": record.plugin_id,
                    "type": record.plugin_type,
                    "display_name": manifest.display_name,
                    "description": manifest.description,
                    "version": record.plugin_version,
                }

                if manifest.agent_hints:
                    plugin_info["agent_hints"] = {
                        "purpose": manifest.agent_hints.purpose,
                        "best_for": manifest.agent_hints.best_for,
                    }

                plugins.append(plugin_info)

            plugins.sort(key=lambda p: (p["type"], p["display_name"]))

            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({
                            "plugins": plugins,
                            "count": len(plugins),
                            "filter": plugin_type,
                        }),
                    }
                ]
            }
        except Exception as e:
            logger.error(f"list_plugins failed: {e}", exc_info=True)
            return {"content": [{"type": "text", "text": json.dumps({"error": str(e)})}]}

    @tool(
        "get_plugin_details",
        "Get detailed information about a specific plugin by ID.",
        {
            "type": "object",
            "properties": {
                "plugin_id": {
                    "type": "string",
                    "description": "The plugin ID to look up",
                },
            },
            "required": ["plugin_id"],
        },
    )
    async def get_plugin_details(args: dict[str, Any]) -> dict[str, Any]:
        """Get plugin details."""
        plugin_id = args.get("plugin_id", "")

        try:
            record = plugin_registry.find_by_id(plugin_id)

            if not record:
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps({
                                "found": False,
                                "error": f"Plugin '{plugin_id}' not found",
                            }),
                        }
                    ]
                }

            manifest = record.manifest
            result: dict[str, Any] = {
                "found": True,
                "id": record.plugin_id,
                "type": record.plugin_type,
                "version": record.plugin_version,
                "display_name": manifest.display_name,
                "description": manifest.description,
                "capabilities": manifest.capabilities,
            }

            if manifest.agent_hints:
                result["agent_hints"] = manifest.agent_hints.to_dict()

            return {"content": [{"type": "text", "text": json.dumps(result)}]}

        except Exception as e:
            logger.error(f"get_plugin_details failed: {e}", exc_info=True)
            return {"content": [{"type": "text", "text": json.dumps({"error": str(e)})}]}

    @tool(
        "build_pipeline",
        "Create or update the pipeline configuration. Builds a DAG with nodes and edges.",
        {
            "type": "object",
            "properties": {
                "nodes": {
                    "type": "array",
                    "description": "Pipeline nodes (parser, chunker, embedder)",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "type": {
                                "type": "string",
                                "enum": ["parser", "chunker", "extractor", "embedder"],
                            },
                            "plugin_id": {"type": "string"},
                            "config": {"type": "object"},
                        },
                        "required": ["id", "type", "plugin_id"],
                    },
                },
                "edges": {
                    "type": "array",
                    "description": "Edges connecting nodes with optional predicates",
                    "items": {
                        "type": "object",
                        "properties": {
                            "from_node": {"type": "string"},
                            "to_node": {"type": "string"},
                            "when": {
                                "type": "object",
                                "description": "Predicate for conditional routing",
                            },
                            "parallel": {"type": "boolean", "default": False},
                        },
                        "required": ["from_node", "to_node"],
                    },
                },
            },
            "required": ["nodes", "edges"],
        },
    )
    async def build_pipeline(args: dict[str, Any]) -> dict[str, Any]:
        """Build pipeline configuration."""
        nodes = args.get("nodes", [])
        edges = args.get("edges", [])

        try:
            # Validate plugins exist
            missing_plugins = []
            for node in nodes:
                if not plugin_registry.find_by_id(node["plugin_id"]):
                    missing_plugins.append(node["plugin_id"])

            if missing_plugins:
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps({
                                "success": False,
                                "error": f"Unknown plugins: {', '.join(missing_plugins)}",
                            }),
                        }
                    ]
                }

            # Build DAG structure
            pipeline = {
                "nodes": nodes,
                "edges": edges,
            }

            # Store in context
            ctx.pipeline_state = pipeline

            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({
                            "success": True,
                            "pipeline": pipeline,
                            "node_count": len(nodes),
                            "edge_count": len(edges),
                        }),
                    }
                ]
            }

        except Exception as e:
            logger.error(f"build_pipeline failed: {e}", exc_info=True)
            return {"content": [{"type": "text", "text": json.dumps({"error": str(e)})}]}

    @tool(
        "apply_pipeline",
        "Apply the current pipeline configuration to create a collection. "
        "Validates the pipeline and prepares it for collection creation.",
        {
            "type": "object",
            "properties": {
                "collection_name": {
                    "type": "string",
                    "description": "Name for the new collection",
                },
                "collection_description": {
                    "type": "string",
                    "description": "Optional description for the collection",
                },
            },
            "required": ["collection_name"],
        },
    )
    async def apply_pipeline(args: dict[str, Any]) -> dict[str, Any]:
        """Apply pipeline configuration to create a collection."""
        collection_name = args.get("collection_name", "").strip()
        collection_description = args.get("collection_description")

        try:
            # Validate collection name
            if not collection_name:
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps({
                                "success": False,
                                "error": "collection_name is required",
                            }),
                        }
                    ]
                }

            # Check for pipeline
            if not ctx.pipeline_state:
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps({
                                "success": False,
                                "error": "No pipeline configured. Use build_pipeline first.",
                            }),
                        }
                    ]
                }

            # Validate pipeline structure using PipelineDAG
            from shared.pipeline.types import PipelineDAG

            try:
                dag = PipelineDAG.from_dict(ctx.pipeline_state)
                known_plugins = set(plugin_registry.list_ids())
                errors = dag.validate(known_plugins)

                if errors:
                    return {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps({
                                    "success": False,
                                    "error": "Pipeline has validation errors",
                                    "validation_errors": [
                                        {"rule": e.rule, "message": e.message}
                                        for e in errors
                                    ],
                                }),
                            }
                        ]
                    }
            except Exception as e:
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps({
                                "success": False,
                                "error": f"Invalid pipeline configuration: {e}",
                            }),
                        }
                    ]
                }

            # Store the applied configuration in context
            ctx.applied_config = {
                "collection_name": collection_name,
                "collection_description": collection_description,
                "pipeline_config": ctx.pipeline_state,
                "source_id": ctx.source_id,
            }

            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({
                            "success": True,
                            "message": f"Pipeline validated and ready to create collection '{collection_name}'",
                            "collection_name": collection_name,
                            "pipeline_node_count": len(ctx.pipeline_state.get("nodes", [])),
                            "pipeline_edge_count": len(ctx.pipeline_state.get("edges", [])),
                        }),
                    }
                ]
            }

        except Exception as e:
            logger.error(f"apply_pipeline failed: {e}", exc_info=True)
            return {"content": [{"type": "text", "text": json.dumps({"error": str(e)})}]}

    return create_sdk_mcp_server(
        name="semantik-assisted-flow",
        version="1.0.0",
        tools=[list_plugins, get_plugin_details, build_pipeline, apply_pipeline],
    )
