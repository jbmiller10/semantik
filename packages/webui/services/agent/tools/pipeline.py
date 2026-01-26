"""Agent tools for pipeline configuration and management.

These tools allow the agent to view, build, and apply pipeline configurations
for document processing in collections.
"""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Any, ClassVar

from shared.pipeline.types import (
    NodeType,
    PipelineDAG,
    PipelineEdge,
    PipelineNode,
)
from shared.plugins.registry import plugin_registry
from webui.services.agent.tools.base import BaseTool

if TYPE_CHECKING:
    from webui.services.agent.models import AgentConversation

logger = logging.getLogger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================


def _get_source_path(source_type: str, source_config: dict[str, Any]) -> str:
    """Derive source_path from config based on connector type.

    Args:
        source_type: Type of connector (directory, git, imap)
        source_config: Connector-specific configuration

    Returns:
        Human-readable source path/identifier
    """
    if source_type == "directory":
        return source_config.get("path", "")
    if source_type == "git":
        return source_config.get("repo_url", source_config.get("repository_url", ""))
    if source_type == "imap":
        username = source_config.get("username", "")
        host = source_config.get("host", "")
        return f"{username}@{host}" if username and host else (username or host)
    # Fallback: try common path-like fields
    return source_config.get("path", source_config.get("url", str(source_config)))


def _map_secret_key_to_type(secret_key: str) -> str | None:
    """Map secret keys from connector forms to valid ConnectorSecret types.

    Args:
        secret_key: Key from the secrets dict (e.g., 'password', 'access_token')

    Returns:
        Valid secret type or None if not mappable
    """
    # Valid secret types from ConnectorSecretRepository
    valid_types = {"password", "token", "ssh_key", "ssh_passphrase"}

    # Direct match
    if secret_key in valid_types:
        return secret_key

    # Common mappings
    key_lower = secret_key.lower()
    if "password" in key_lower or "pass" in key_lower:
        return "password"
    if "token" in key_lower or "api_key" in key_lower or "access" in key_lower:
        return "token"
    if "ssh_key" in key_lower or "private_key" in key_lower:
        return "ssh_key"
    if "passphrase" in key_lower:
        return "ssh_passphrase"

    # No mapping found - log warning and skip
    logger.warning(f"Unknown secret key '{secret_key}' - skipping storage")
    return None


# =============================================================================
# Tool Classes
# =============================================================================


class GetPipelineStateTool(BaseTool):
    """Get the current pipeline configuration from conversation state.

    Returns the current pipeline DAG being built, or null if none configured yet.
    Optionally validates the DAG against registered plugins.
    """

    NAME: ClassVar[str] = "get_pipeline_state"
    DESCRIPTION: ClassVar[str] = (
        "Get the current pipeline configuration from the conversation. "
        "Returns the pipeline DAG structure or null if not yet configured. "
        "Can optionally validate the DAG against available plugins."
    )
    PARAMETERS: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "validate": {
                "type": "boolean",
                "description": "Whether to validate the pipeline against registered plugins",
                "default": False,
            },
        },
        "required": [],
    }

    async def execute(self, validate: bool = False) -> dict[str, Any]:
        """Execute the pipeline state retrieval.

        Args:
            validate: Whether to validate the pipeline

        Returns:
            Dictionary with pipeline state and optional validation
        """
        try:
            conversation: AgentConversation | None = self.context.get("conversation")

            if not conversation:
                return {
                    "has_pipeline": False,
                    "error": "No conversation context available",
                }

            pipeline_config = conversation.current_pipeline

            if not pipeline_config:
                return {
                    "has_pipeline": False,
                    "pipeline": None,
                    "message": "No pipeline configured yet. Use build_pipeline to create one.",
                }

            result: dict[str, Any] = {
                "has_pipeline": True,
                "pipeline": pipeline_config,
            }

            # Optionally validate
            if validate:
                try:
                    dag = PipelineDAG.from_dict(pipeline_config)
                    known_plugins = set(plugin_registry.list_ids())
                    errors = dag.validate(known_plugins)

                    result["validation"] = {
                        "is_valid": len(errors) == 0,
                        "errors": [
                            {
                                "rule": e.rule,
                                "message": e.message,
                                "node_id": e.node_id,
                                "edge_index": e.edge_index,
                            }
                            for e in errors
                        ],
                    }
                except Exception as e:
                    result["validation"] = {
                        "is_valid": False,
                        "errors": [{"rule": "parse_error", "message": str(e)}],
                    }

            return result

        except Exception as e:
            logger.error(f"Failed to get pipeline state: {e}", exc_info=True)
            return {
                "has_pipeline": False,
                "error": str(e),
            }


class BuildPipelineTool(BaseTool):
    """Build or modify a pipeline configuration.

    Creates a new pipeline from a template or custom specification,
    with optional tunable parameter overrides. The pipeline is validated
    and stored in the conversation state.
    """

    NAME: ClassVar[str] = "build_pipeline"
    DESCRIPTION: ClassVar[str] = (
        "Build or modify a pipeline configuration. Can create from a template "
        "with tunable overrides, or build a custom pipeline from nodes and edges. "
        "Validates the result and stores it in conversation state."
    )
    PARAMETERS: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "template_id": {
                "type": "string",
                "description": ("ID of a template to use as base. Use list_templates to see available templates."),
            },
            "tunable_values": {
                "type": "object",
                "description": (
                    "Override values for tunable parameters, keyed by path "
                    "(e.g., {'nodes.chunker.config.max_tokens': 1024})"
                ),
            },
            "nodes": {
                "type": "array",
                "description": "Custom nodes for a pipeline (when not using template). Each node: {id, type, plugin_id, config}",
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
                "description": "Custom edges for a pipeline (when not using template). Each edge: {from_node, to_node, when}",
                "items": {
                    "type": "object",
                    "properties": {
                        "from_node": {"type": "string"},
                        "to_node": {"type": "string"},
                        "when": {"type": "object"},
                    },
                    "required": ["from_node", "to_node"],
                },
            },
            "pipeline_id": {
                "type": "string",
                "description": "Custom ID for the pipeline (defaults to template ID or 'custom')",
            },
        },
        "required": [],
    }

    async def execute(
        self,
        template_id: str | None = None,
        tunable_values: dict[str, Any] | None = None,
        nodes: list[dict[str, Any]] | None = None,
        edges: list[dict[str, Any]] | None = None,
        pipeline_id: str | None = None,
    ) -> dict[str, Any]:
        """Execute the pipeline building.

        Args:
            template_id: ID of template to use as base
            tunable_values: Override values for tunable parameters
            nodes: Custom nodes (when not using template)
            edges: Custom edges (when not using template)
            pipeline_id: Custom ID for the pipeline

        Returns:
            Dictionary with built pipeline and validation status
        """
        try:
            conversation: AgentConversation | None = self.context.get("conversation")

            if not conversation:
                return {
                    "success": False,
                    "error": "No conversation context available",
                }

            dag: PipelineDAG | None = None

            # Build from template
            if template_id:
                from shared.pipeline.templates import load_template, resolve_tunable_path

                template = load_template(template_id)
                if not template:
                    from shared.pipeline.templates import list_templates

                    available = [t.id for t in list_templates()]
                    return {
                        "success": False,
                        "error": f"Template '{template_id}' not found",
                        "available_templates": available,
                    }

                # Deep copy the pipeline to avoid modifying the template
                pipeline_dict = copy.deepcopy(template.pipeline.to_dict())
                dag = PipelineDAG.from_dict(pipeline_dict)

                # Apply tunable value overrides
                if tunable_values:
                    applied: list[str] = []
                    failed: list[str] = []

                    for path, value in tunable_values.items():
                        node, config_key = resolve_tunable_path(template, path)
                        if node:
                            # Find the node in our copied DAG and update it
                            for dag_node in dag.nodes:
                                if dag_node.id == node.id and config_key is not None:
                                    dag_node.config[config_key] = value
                                    applied.append(path)
                                    break
                        else:
                            failed.append(path)

                    if failed:
                        return {
                            "success": False,
                            "error": f"Invalid tunable paths: {failed}",
                            "valid_paths": [t.path for t in template.tunable],
                        }

                # Use template ID as pipeline ID if not specified
                if pipeline_id:
                    # Create new DAG with custom ID
                    dag = PipelineDAG(
                        id=pipeline_id,
                        version=dag.version,
                        nodes=dag.nodes,
                        edges=dag.edges,
                    )

            # Build from custom nodes/edges
            elif nodes is not None:
                if not edges:
                    return {
                        "success": False,
                        "error": "Both 'nodes' and 'edges' are required for custom pipelines",
                    }

                try:
                    pipeline_nodes = [
                        PipelineNode(
                            id=n["id"],
                            type=NodeType(n["type"]),
                            plugin_id=n["plugin_id"],
                            config=n.get("config", {}),
                        )
                        for n in nodes
                    ]

                    pipeline_edges = [
                        PipelineEdge(
                            from_node=e["from_node"],
                            to_node=e["to_node"],
                            when=e.get("when"),
                        )
                        for e in edges
                    ]

                    dag = PipelineDAG(
                        id=pipeline_id or "custom",
                        version="1.0",
                        nodes=pipeline_nodes,
                        edges=pipeline_edges,
                    )

                except (KeyError, ValueError) as e:
                    return {
                        "success": False,
                        "error": f"Invalid node/edge specification: {e}",
                    }

            else:
                return {
                    "success": False,
                    "error": "Must specify either template_id or nodes/edges",
                }

            # Validate the DAG
            known_plugins = set(plugin_registry.list_ids())
            validation_errors = dag.validate(known_plugins)

            pipeline_dict = dag.to_dict()

            result: dict[str, Any] = {
                "success": len(validation_errors) == 0,
                "pipeline": pipeline_dict,
                "validation": {
                    "is_valid": len(validation_errors) == 0,
                    "errors": [
                        {
                            "rule": e.rule,
                            "message": e.message,
                            "node_id": e.node_id,
                            "edge_index": e.edge_index,
                        }
                        for e in validation_errors
                    ],
                },
            }

            # Store in conversation state (even if invalid, for debugging)
            # Note: The actual persistence would be handled by the orchestrator
            # after the tool call. We update the context for immediate access.
            conversation.current_pipeline = pipeline_dict

            if validation_errors:
                result["message"] = "Pipeline built but has validation errors"
            else:
                result["message"] = "Pipeline built and validated successfully"

            return result

        except Exception as e:
            logger.error(f"Failed to build pipeline: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }


class ApplyPipelineTool(BaseTool):
    """Apply the current pipeline to create a collection.

    Takes the current pipeline configuration and creates a new collection
    with it, optionally triggering the indexing operation.
    """

    NAME: ClassVar[str] = "apply_pipeline"
    DESCRIPTION: ClassVar[str] = (
        "Apply the current pipeline configuration to create a collection. "
        "Checks for blocking uncertainties first. Returns the created collection ID."
    )
    PARAMETERS: ClassVar[dict[str, Any]] = {
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
            "force": {
                "type": "boolean",
                "description": "Apply even if there are blocking uncertainties",
                "default": False,
            },
            "start_indexing": {
                "type": "boolean",
                "description": "Whether to immediately start indexing after creation",
                "default": True,
            },
        },
        "required": ["collection_name"],
    }

    async def execute(
        self,
        collection_name: str,
        collection_description: str | None = None,
        force: bool = False,
        start_indexing: bool = True,
    ) -> dict[str, Any]:
        """Execute the pipeline application.

        Args:
            collection_name: Name for the new collection
            collection_description: Optional description
            force: Apply even with blocking uncertainties
            start_indexing: Whether to start indexing immediately

        Returns:
            Dictionary with collection details or error
        """
        try:
            conversation: AgentConversation | None = self.context.get("conversation")
            session = self.context.get("session")
            user_id = self.context.get("user_id")

            if not conversation:
                return {
                    "success": False,
                    "error": "No conversation context available",
                }

            if not session:
                return {
                    "success": False,
                    "error": "No database session available",
                }

            if not user_id:
                return {
                    "success": False,
                    "error": "No user ID available",
                }

            # Check for pipeline
            if not conversation.current_pipeline:
                return {
                    "success": False,
                    "error": "No pipeline configured. Use build_pipeline first.",
                }

            # Validate pipeline
            try:
                dag = PipelineDAG.from_dict(conversation.current_pipeline)
                known_plugins = set(plugin_registry.list_ids())
                errors = dag.validate(known_plugins)

                if errors:
                    return {
                        "success": False,
                        "error": "Pipeline has validation errors",
                        "validation_errors": [{"rule": e.rule, "message": e.message} for e in errors],
                    }
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Invalid pipeline configuration: {e}",
                }

            # Check for blocking uncertainties
            if not force:
                from webui.services.agent.repository import AgentConversationRepository

                repo = AgentConversationRepository(session)
                blocking = await repo.get_blocking_uncertainties(conversation.id)

                if blocking:
                    return {
                        "success": False,
                        "error": "Cannot apply: blocking uncertainties exist",
                        "blocking_uncertainties": [
                            {"message": u.message, "severity": u.severity.value} for u in blocking
                        ],
                        "hint": "Resolve the uncertainties or use force=true to override",
                    }

            # Create the collection with pipeline config
            from webui.services.factory import create_collection_service

            collection_service = create_collection_service(session)

            # Extract embedding model from pipeline config
            embedder_node = next(
                (n for n in dag.nodes if n.type == NodeType.EMBEDDER),
                None,
            )
            embedding_model = (
                embedder_node.config.get("model", "BAAI/bge-base-en-v1.5") if embedder_node else "BAAI/bge-base-en-v1.5"
            )

            # Extract chunking config from pipeline
            chunker_node = next(
                (n for n in dag.nodes if n.type == NodeType.CHUNKER),
                None,
            )
            chunk_size = chunker_node.config.get("max_tokens", 512) if chunker_node else 512
            chunk_overlap = chunker_node.config.get("overlap", 50) if chunker_node else 50

            # Build collection config
            config: dict[str, Any] = {
                "embedding_model": embedding_model,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "pipeline_config": conversation.current_pipeline,
            }

            # Create collection (without source initially)
            collection_result, operation_result = await collection_service.create_collection(
                user_id=user_id,
                name=collection_name,
                description=collection_description,
                config=config,
            )

            collection_id = collection_result["id"]

            # Update conversation with collection link
            from webui.services.agent.repository import AgentConversationRepository

            repo = AgentConversationRepository(session)
            await repo.set_collection(
                conversation_id=conversation.id,
                user_id=user_id,
                collection_id=collection_id,
            )

            # If conversation has inline source config, create the source now
            source_id: int | None = conversation.source_id
            if conversation.inline_source_config and not conversation.source_id:
                from shared.database.repositories.collection_source_repository import (
                    CollectionSourceRepository,
                )
                from shared.database.repositories.connector_secret_repository import (
                    ConnectorSecretRepository,
                )

                inline_config = conversation.inline_source_config
                source_type = inline_config.get("source_type", "")
                source_config = inline_config.get("source_config", {})
                pending_secrets = inline_config.get("_pending_secrets", {})

                # Derive source_path from config based on connector type
                source_path = _get_source_path(source_type, source_config)

                # Create the source
                source_repo = CollectionSourceRepository(session)
                new_source = await source_repo.create(
                    collection_id=collection_id,
                    source_type=source_type,
                    source_path=source_path,
                    source_config=source_config,
                )
                # Cast to int for type safety (new_source.id is typed as Column[int])
                new_source_id: int = int(new_source.id)  # type: ignore[arg-type]
                source_id = new_source_id

                # Store secrets encrypted if any
                # Note: secrets may already be encrypted (as base64 strings) from agent.py
                if pending_secrets:
                    from shared.utils.encryption import (
                        DecryptionError,
                        EncryptionNotConfiguredError,
                        decrypt_secret,
                    )

                    secret_repo = ConnectorSecretRepository(session)
                    for secret_key, value in pending_secrets.items():
                        # Map common secret keys to valid secret types
                        secret_type = _map_secret_key_to_type(secret_key)
                        if secret_type:
                            # Try to decrypt the value (it may be encrypted from agent.py)
                            plaintext_value = value
                            try:
                                plaintext_value = decrypt_secret(value)
                            except EncryptionNotConfiguredError:
                                logger.warning(
                                    f"Encryption not configured - using raw secret "
                                    f"for source {new_source_id}"
                                )
                            except DecryptionError as e:
                                logger.error(
                                    f"Failed to decrypt secret '{secret_key}' "
                                    f"for source {new_source_id}: {e}"
                                )
                            except ValueError as e:
                                logger.error(
                                    f"Invalid secret format for '{secret_key}' "
                                    f"on source {new_source_id}: {e}"
                                )

                            await secret_repo.set_secret(
                                source_id=new_source_id,
                                secret_type=secret_type,
                                plaintext=plaintext_value,
                            )

                # Update conversation with the new source ID and clear pending secrets
                await repo.set_source_id(
                    conversation_id=conversation.id,
                    user_id=user_id,
                    source_id=new_source_id,
                )

                logger.info(f"Created source {source_id} from inline config for collection {collection_id}")

            # Note: The orchestrator's _persist_state_changes() handles the commit

            return {
                "success": True,
                "collection_id": collection_id,
                "collection_name": collection_result["name"],
                "source_id": source_id,
                "operation_id": operation_result.get("id") if start_indexing else None,
                "status": "indexing" if start_indexing else "created",
                "message": f"Collection '{collection_name}' created successfully",
            }

        except Exception as e:
            logger.error(f"Failed to apply pipeline: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }
