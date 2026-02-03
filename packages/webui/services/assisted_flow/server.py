"""SDK MCP Server creation for assisted flow.

This module creates an in-process MCP server with all the tools
needed for the pipeline configuration assistant.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from claude_agent_sdk import create_sdk_mcp_server, tool

from shared.connectors.base import BaseConnector  # noqa: TCH001 - used in runtime type hints
from shared.database.database import ensure_async_sessionmaker
from shared.database.repositories.collection_source_repository import CollectionSourceRepository
from shared.pipeline.types import FileReference, PipelineDAG
from shared.plugins.registry import plugin_registry
from webui.services.connector_factory import ConnectorFactory

if TYPE_CHECKING:
    from claude_agent_sdk.types import McpSdkServerConfig

    from webui.services.assisted_flow.context import ToolContext

logger = logging.getLogger(__name__)


def _error_result(message: str) -> dict[str, Any]:
    """Create standardized error response for MCP tools."""
    return {"content": [{"type": "text", "text": json.dumps({"success": False, "error": message})}]}


def _get_parser_recommendations(extension_counts: dict[str, int]) -> dict[str, str]:
    """Map file extensions to recommended parser plugin IDs.

    Args:
        extension_counts: Dictionary of extension -> count

    Returns:
        Dictionary mapping extensions to recommended parser plugin IDs
    """
    # Extension to parser mapping
    extension_parser_map = {
        # Text parser handles
        ".txt": "text",
        ".md": "text",
        ".markdown": "text",
        ".rst": "text",
        ".py": "text",
        ".js": "text",
        ".ts": "text",
        ".tsx": "text",
        ".jsx": "text",
        ".java": "text",
        ".c": "text",
        ".cpp": "text",
        ".h": "text",
        ".hpp": "text",
        ".go": "text",
        ".rs": "text",
        ".rb": "text",
        ".php": "text",
        ".sh": "text",
        ".bash": "text",
        ".zsh": "text",
        ".yaml": "text",
        ".yml": "text",
        ".json": "text",
        ".xml": "text",
        ".html": "text",
        ".css": "text",
        ".scss": "text",
        ".sql": "text",
        ".toml": "text",
        ".ini": "text",
        ".cfg": "text",
        ".conf": "text",
        # Unstructured parser handles
        ".pdf": "unstructured",
        ".docx": "unstructured",
        ".doc": "unstructured",
        ".pptx": "unstructured",
        ".ppt": "unstructured",
        ".xlsx": "unstructured",
        ".xls": "unstructured",
        ".eml": "unstructured",
        ".msg": "unstructured",
        ".epub": "unstructured",
        ".rtf": "unstructured",
        ".odt": "unstructured",
        ".ods": "unstructured",
        ".odp": "unstructured",
    }

    recommendations = {}
    for ext in extension_counts:
        # Default to text parser for unknown extensions
        recommendations[ext] = extension_parser_map.get(ext, "text")

    return recommendations


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
                        "text": json.dumps(
                            {
                                "plugins": plugins,
                                "count": len(plugins),
                                "filter": plugin_type,
                            }
                        ),
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
                            "text": json.dumps(
                                {
                                    "found": False,
                                    "error": f"Plugin '{plugin_id}' not found",
                                }
                            ),
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
                "id": {
                    "type": "string",
                    "description": "Optional pipeline DAG id. Defaults to 'agent-recommended'.",
                },
                "version": {
                    "type": "string",
                    "description": "Optional pipeline DAG schema version. Defaults to '1'.",
                },
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
        pipeline_id = str(args.get("id") or "agent-recommended")
        version = str(args.get("version") or "1")
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
                            "text": json.dumps(
                                {
                                    "success": False,
                                    "error": f"Unknown plugins: {', '.join(missing_plugins)}",
                                }
                            ),
                        }
                    ]
                }

            # Ensure there's at least one catch-all edge from _source so the DAG
            # validates even if the agent forgets it.
            has_source_edge = any(isinstance(e, dict) and e.get("from_node") == "_source" for e in edges)
            if not has_source_edge:
                first_parser = next((n for n in nodes if isinstance(n, dict) and n.get("type") == "parser"), None)
                if first_parser and first_parser.get("id"):
                    edges = [{"from_node": "_source", "to_node": first_parser["id"], "when": None}, *edges]

            # Build DAG structure
            pipeline = {
                "id": pipeline_id,
                "version": version,
                "nodes": nodes,
                "edges": edges,
            }

            # Store in context
            ctx.pipeline_state = pipeline

            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "success": True,
                                "pipeline": pipeline,
                                "node_count": len(nodes),
                                "edge_count": len(edges),
                            }
                        ),
                    }
                ]
            }

        except Exception as e:
            logger.error(f"build_pipeline failed: {e}", exc_info=True)
            return {"content": [{"type": "text", "text": json.dumps({"error": str(e)})}]}

    @tool(
        "apply_pipeline",
        "Apply the pipeline configuration. For new sources, creates a collection and starts indexing. "
        "For existing sources, updates the collection's pipeline and triggers reindexing.",
        {
            "type": "object",
            "properties": {
                "collection_name": {
                    "type": "string",
                    "description": "Name for the collection (required for new sources, optional for existing)",
                },
                "collection_description": {
                    "type": "string",
                    "description": "Optional description for the collection",
                },
            },
        },
    )
    async def apply_pipeline(args: dict[str, Any]) -> dict[str, Any]:
        """Apply pipeline: create or update collection, then start indexing."""
        collection_name = args.get("collection_name", "").strip()
        collection_description = args.get("collection_description", "")

        if not ctx.pipeline_state:
            return _error_result("No pipeline configured. Use build_pipeline first.")

        if not ctx.get_session:
            return _error_result("Database access not configured")

        try:
            # Validate DAG first
            dag = PipelineDAG.from_dict(ctx.pipeline_state)
            known_plugins = set(plugin_registry.list_ids())
            errors = dag.validate(known_plugins)
            if errors:
                error_msgs = [f"{e.rule}: {e.message}" for e in errors]
                return _error_result(f"Pipeline validation failed: {', '.join(error_msgs)}")

            # Import services (avoid circular imports)
            from webui.services.factory import create_collection_service, create_source_service

            async with ctx.get_session() as session:
                collection_service = create_collection_service(session)

                # EXISTING SOURCE MODE: Update existing collection's pipeline
                if ctx.source_id is not None:
                    # Get source to find its collection
                    source_service = create_source_service(session)
                    source_obj = await source_service.get_source(ctx.user_id, ctx.source_id)
                    # get_source returns CollectionSource directly when include_secret_types=False
                    if isinstance(source_obj, tuple):
                        source_obj = source_obj[0]
                    source = source_obj

                    # Update collection's pipeline_config and trigger REINDEX
                    reindex_result = await collection_service.reindex_collection(
                        collection_id=str(source.collection_id),
                        user_id=ctx.user_id,
                        config_updates={"pipeline_config": ctx.pipeline_state},
                    )

                    # Get updated collection info
                    collection = await collection_service.collection_repo.get_by_uuid(str(source.collection_id))

                    await session.commit()

                    ctx.applied_config = {
                        "collection_id": str(source.collection_id),
                        "collection_name": collection.name if collection else "Unknown",
                        "source_id": ctx.source_id,
                        "operation_id": reindex_result.get("uuid"),
                        "mode": "reindex",
                    }

                    return {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(
                                    {
                                        "success": True,
                                        "mode": "reindex",
                                        "collection_id": str(source.collection_id),
                                        "collection_name": collection.name if collection else "Unknown",
                                        "operation_id": reindex_result.get("uuid"),
                                        "message": "Pipeline updated. Reindexing started for collection.",
                                    }
                                ),
                            }
                        ]
                    }

                # INLINE SOURCE MODE: Create new collection + source
                if not collection_name:
                    return _error_result("collection_name is required for new sources")

                if not ctx.inline_source_config:
                    return _error_result("No source configuration provided")

                # Create collection with pipeline config
                collection_config = {"pipeline_config": ctx.pipeline_state}

                collection_result, index_op_result = await collection_service.create_collection(
                    user_id=ctx.user_id,
                    name=collection_name,
                    description=collection_description,
                    config=collection_config,
                )

                collection_id = collection_result["id"]

                # Determine source_type from inline config
                source_type = ctx.inline_source_config.get("source_type", "directory")
                # Remove source_type from config dict (it's a separate param)
                source_config = {k: v for k, v in ctx.inline_source_config.items() if k != "source_type"}

                # Derive source_path from config
                if source_type == "directory":
                    source_path = source_config.get("path", collection_name)
                elif source_type == "git":
                    source_path = source_config.get("repo_url", source_config.get("repository_url", collection_name))
                else:
                    source_path = collection_name

                # Create source
                source_service = create_source_service(session)
                new_source, _secret_types = await source_service.create_source(
                    user_id=ctx.user_id,
                    collection_id=collection_id,
                    source_type=source_type,
                    source_path=source_path,
                    source_config=source_config,
                    secrets=ctx.inline_secrets,
                )
                new_source_id = int(new_source.id)

                # Trigger APPEND operation to index the source
                append_result = await source_service.run_now(
                    user_id=ctx.user_id,
                    source_id=new_source_id,
                )

                await session.commit()

                ctx.applied_config = {
                    "collection_id": collection_id,
                    "collection_name": collection_name,
                    "source_id": new_source_id,
                    "index_operation_id": index_op_result.get("uuid"),
                    "append_operation_id": append_result.get("uuid"),
                    "mode": "create",
                }

                return {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "success": True,
                                    "mode": "create",
                                    "collection_id": collection_id,
                                    "collection_name": collection_name,
                                    "source_id": new_source_id,
                                    "index_operation_id": index_op_result.get("uuid"),
                                    "append_operation_id": append_result.get("uuid"),
                                    "message": f"Collection '{collection_name}' created. Indexing started.",
                                }
                            ),
                        }
                    ]
                }

        except Exception as e:
            logger.error(f"apply_pipeline failed: {e}", exc_info=True)
            return _error_result(str(e))

    @tool(
        "sample_files",
        "Sample files from the source to understand content types and structure.",
        {
            "type": "object",
            "properties": {
                "count": {
                    "type": "integer",
                    "description": "Number of files to sample (default 10, max 50)",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 50,
                },
                "filter_extension": {
                    "type": "string",
                    "description": "Filter by file extension (e.g., '.pdf')",
                },
            },
        },
    )
    async def sample_files(args: dict[str, Any]) -> dict[str, Any]:
        """Sample files from the configured source."""
        count = min(args.get("count", 10), 50)
        filter_extension = args.get("filter_extension")

        if filter_extension and not filter_extension.startswith("."):
            filter_extension = f".{filter_extension}"

        if ctx.source_id is None:
            return _error_result("No source configured for this session")

        try:
            # Get source from database
            sessionmaker = await ensure_async_sessionmaker()
            async with sessionmaker() as session:
                repo = CollectionSourceRepository(session)
                source = await repo.get_by_id(ctx.source_id)

                if not source:
                    return _error_result(f"Source {ctx.source_id} not found")

                # Create connector
                connector: BaseConnector = ConnectorFactory.get_connector(
                    source.source_type,
                    source.source_config or {},
                )

                # Authenticate
                if not await connector.authenticate():
                    return _error_result("Failed to authenticate with source")

                # Enumerate files
                sampled_files: list[dict[str, Any]] = []
                async for file_ref in connector.enumerate(source.id):
                    # Apply extension filter if specified
                    if filter_extension and file_ref.extension != filter_extension.lower():
                        continue

                    sampled_files.append(
                        {
                            "uri": file_ref.uri,
                            "filename": file_ref.filename,
                            "extension": file_ref.extension,
                            "mime_type": file_ref.mime_type,
                            "size_bytes": file_ref.size_bytes,
                            "content_type": file_ref.content_type,
                        }
                    )

                    if len(sampled_files) >= count:
                        break

                return {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "success": True,
                                    "files": sampled_files,
                                    "count": len(sampled_files),
                                    "filter_extension": filter_extension,
                                    "source_type": source.source_type,
                                }
                            ),
                        }
                    ]
                }

        except Exception as e:
            logger.error(f"sample_files failed: {e}", exc_info=True)
            return _error_result(str(e))

    @tool(
        "preview_content",
        "Preview the content of a specific file (first 2000 characters).",
        {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "File URI from sample_files result",
                },
            },
            "required": ["file_path"],
        },
    )
    async def preview_content(args: dict[str, Any]) -> dict[str, Any]:
        """Preview content of a specific file."""
        file_path = args.get("file_path", "")

        if not file_path:
            return _error_result("file_path is required")

        if ctx.source_id is None:
            return _error_result("No source configured for this session")

        try:
            # Get source from database
            sessionmaker = await ensure_async_sessionmaker()
            async with sessionmaker() as session:
                repo = CollectionSourceRepository(session)
                source = await repo.get_by_id(ctx.source_id)

                if not source:
                    return _error_result(f"Source {ctx.source_id} not found")

                # Create connector
                connector: BaseConnector = ConnectorFactory.get_connector(
                    source.source_type,
                    source.source_config or {},
                )

                # Authenticate
                if not await connector.authenticate():
                    return _error_result("Failed to authenticate with source")

                # Find the matching file reference
                target_file: FileReference | None = None
                async for file_ref in connector.enumerate(source.id):
                    if file_ref.uri == file_path:
                        target_file = file_ref
                        break

                if not target_file:
                    return _error_result(f"File not found: {file_path}")

                # Load content
                content_bytes = await connector.load_content(target_file)

                # Decode to text (try UTF-8, fall back to latin-1)
                try:
                    content_text = content_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    content_text = content_bytes.decode("latin-1")

                # Truncate to 2000 chars
                max_chars = 2000
                truncated = len(content_text) > max_chars
                preview = content_text[:max_chars]

                return {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "success": True,
                                    "uri": target_file.uri,
                                    "filename": target_file.filename,
                                    "mime_type": target_file.mime_type,
                                    "size_bytes": target_file.size_bytes,
                                    "preview": preview,
                                    "truncated": truncated,
                                    "preview_length": len(preview),
                                    "total_length": len(content_text),
                                }
                            ),
                        }
                    ]
                }

        except Exception as e:
            logger.error(f"preview_content failed: {e}", exc_info=True)
            return _error_result(str(e))

    @tool(
        "detect_patterns",
        "Analyze sampled files to detect patterns and recommend parsers.",
        {
            "type": "object",
            "properties": {
                "sample_count": {
                    "type": "integer",
                    "description": "Number of files to analyze (default 20, max 100)",
                    "default": 20,
                    "minimum": 1,
                    "maximum": 100,
                },
            },
        },
    )
    async def detect_patterns(args: dict[str, Any]) -> dict[str, Any]:
        """Detect file patterns and recommend parsers."""
        sample_count = min(args.get("sample_count", 20), 100)

        if ctx.source_id is None:
            return _error_result("No source configured for this session")

        try:
            # Get source from database
            sessionmaker = await ensure_async_sessionmaker()
            async with sessionmaker() as session:
                repo = CollectionSourceRepository(session)
                source = await repo.get_by_id(ctx.source_id)

                if not source:
                    return _error_result(f"Source {ctx.source_id} not found")

                # Create connector
                connector: BaseConnector = ConnectorFactory.get_connector(
                    source.source_type,
                    source.source_config or {},
                )

                # Authenticate
                if not await connector.authenticate():
                    return _error_result("Failed to authenticate with source")

                # Collect file statistics
                extension_counts: dict[str, int] = {}
                mime_type_counts: dict[str, int] = {}
                sizes: list[int] = []
                content_types: dict[str, int] = {}
                files_analyzed = 0

                async for file_ref in connector.enumerate(source.id):
                    files_analyzed += 1

                    # Track extensions
                    ext = file_ref.extension or "(no extension)"
                    extension_counts[ext] = extension_counts.get(ext, 0) + 1

                    # Track MIME types
                    mime = file_ref.mime_type or "(unknown)"
                    mime_type_counts[mime] = mime_type_counts.get(mime, 0) + 1

                    # Track sizes
                    sizes.append(file_ref.size_bytes)

                    # Track content types
                    ct = file_ref.content_type
                    content_types[ct] = content_types.get(ct, 0) + 1

                    if files_analyzed >= sample_count:
                        break

                # Compute size statistics
                size_stats = {
                    "min_bytes": min(sizes) if sizes else 0,
                    "max_bytes": max(sizes) if sizes else 0,
                    "avg_bytes": sum(sizes) // len(sizes) if sizes else 0,
                    "total_bytes": sum(sizes),
                }

                # Get parser recommendations
                parser_recommendations = _get_parser_recommendations(extension_counts)

                # Sort counts by frequency
                sorted_extensions = sorted(extension_counts.items(), key=lambda x: -x[1])
                sorted_mime_types = sorted(mime_type_counts.items(), key=lambda x: -x[1])

                return {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "success": True,
                                    "files_analyzed": files_analyzed,
                                    "extension_counts": dict(sorted_extensions),
                                    "mime_type_counts": dict(sorted_mime_types),
                                    "content_types": content_types,
                                    "size_stats": size_stats,
                                    "parser_recommendations": parser_recommendations,
                                    "source_type": source.source_type,
                                }
                            ),
                        }
                    ]
                }

        except Exception as e:
            logger.error(f"detect_patterns failed: {e}", exc_info=True)
            return _error_result(str(e))

    @tool(
        "validate_pipeline",
        "Validate current pipeline configuration against sample files.",
        {
            "type": "object",
            "properties": {
                "sample_count": {
                    "type": "integer",
                    "description": "Number of files to validate against (default 5, max 20)",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 20,
                },
            },
        },
    )
    async def validate_pipeline(args: dict[str, Any]) -> dict[str, Any]:
        """Validate pipeline configuration against sample files."""
        sample_count = min(args.get("sample_count", 5), 20)

        if ctx.pipeline_state is None:
            return _error_result("No pipeline configured. Use build_pipeline first.")

        try:
            # Validate pipeline structure using PipelineDAG
            dag = PipelineDAG.from_dict(ctx.pipeline_state)
            known_plugins = set(plugin_registry.list_ids())
            validation_errors = dag.validate(known_plugins)

            if validation_errors:
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "success": False,
                                    "error": "Pipeline has validation errors",
                                    "validation_errors": [
                                        {"rule": e.rule, "message": e.message} for e in validation_errors
                                    ],
                                }
                            ),
                        }
                    ]
                }

            # If no source, just validate DAG structure
            if ctx.source_id is None:
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "success": True,
                                    "dag_valid": True,
                                    "message": "Pipeline DAG is valid (no source to test routing)",
                                    "node_count": len(dag.nodes),
                                    "edge_count": len(dag.edges),
                                }
                            ),
                        }
                    ]
                }

            # Get source and sample files for routing validation
            sessionmaker = await ensure_async_sessionmaker()
            async with sessionmaker() as session:
                repo = CollectionSourceRepository(session)
                source = await repo.get_by_id(ctx.source_id)

                if not source:
                    return _error_result(f"Source {ctx.source_id} not found")

                # Create connector
                connector: BaseConnector = ConnectorFactory.get_connector(
                    source.source_type,
                    source.source_config or {},
                )

                # Authenticate
                if not await connector.authenticate():
                    return _error_result("Failed to authenticate with source")

                # Sample files and check routing
                from shared.pipeline.router import PipelineRouter

                router = PipelineRouter(dag)
                file_results: list[dict[str, Any]] = []
                files_checked = 0

                async for file_ref in connector.enumerate(source.id):
                    files_checked += 1

                    # Get entry nodes from _source
                    try:
                        entry_nodes = router.get_entry_nodes(file_ref)
                        matched_nodes = [n.id for n, _ in entry_nodes]
                        has_route = len(matched_nodes) > 0
                    except Exception as e:
                        matched_nodes = []
                        has_route = False
                        logger.warning(f"Routing failed for {file_ref.uri}: {e}")

                    file_results.append(
                        {
                            "uri": file_ref.uri,
                            "filename": file_ref.filename,
                            "extension": file_ref.extension,
                            "mime_type": file_ref.mime_type,
                            "has_route": has_route,
                            "matched_nodes": matched_nodes,
                        }
                    )

                    if files_checked >= sample_count:
                        break

                # Summary
                routed_count = sum(1 for f in file_results if f["has_route"])
                unrouted_count = len(file_results) - routed_count

                return {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "success": True,
                                    "dag_valid": True,
                                    "files_checked": files_checked,
                                    "routed_count": routed_count,
                                    "unrouted_count": unrouted_count,
                                    "all_files_routed": unrouted_count == 0,
                                    "file_results": file_results,
                                    "node_count": len(dag.nodes),
                                    "edge_count": len(dag.edges),
                                }
                            ),
                        }
                    ]
                }

        except Exception as e:
            logger.error(f"validate_pipeline failed: {e}", exc_info=True)
            return _error_result(str(e))

    return create_sdk_mcp_server(
        name="semantik-assisted-flow",
        version="1.0.0",
        tools=[
            list_plugins,
            get_plugin_details,
            build_pipeline,
            apply_pipeline,
            sample_files,
            preview_content,
            detect_patterns,
            validate_pipeline,
        ],
    )
