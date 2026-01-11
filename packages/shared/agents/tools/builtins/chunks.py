"""Chunk retrieval tool for agent use.

This tool enables LLM agents to retrieve specific text chunks
from Semantik collections.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from uuid import UUID

from shared.agents.exceptions import ToolExecutionError
from shared.agents.tools.base import AgentTool, ToolDefinition, ToolParameter
from shared.database import pg_connection_manager
from shared.database.repositories.chunk_repository import ChunkRepository
from shared.database.repositories.collection_repository import CollectionRepository

if TYPE_CHECKING:
    from shared.agents.types import AgentContext

logger = logging.getLogger(__name__)


class GetChunkTool(AgentTool):
    """Retrieve a specific chunk by ID.

    This tool allows agents to fetch the full content of a specific
    text chunk, typically identified from search results.
    """

    @property
    def definition(self) -> ToolDefinition:
        """Return the tool definition."""
        return ToolDefinition(
            name="get_chunk",
            description=(
                "Retrieve a specific text chunk by its ID. Use this to get the full "
                "content of a chunk returned from search results. The chunk_id is "
                "typically in the format 'document_uuid_0001'."
            ),
            parameters=[
                ToolParameter(
                    name="chunk_id",
                    type="string",
                    description="The chunk identifier (from search results metadata.chunk_id or chunk_id field).",
                    required=True,
                ),
                ToolParameter(
                    name="collection_id",
                    type="string",
                    description="The UUID of the collection containing the chunk.",
                    required=True,
                ),
            ],
            category="search",
            requires_context=True,
            is_destructive=False,
            timeout_seconds=15.0,
        )

    async def execute(
        self,
        args: dict[str, Any],
        context: AgentContext | None = None,
    ) -> dict[str, Any]:
        """Execute the chunk retrieval.

        Args:
            args: Tool arguments (chunk_id, collection_id).
            context: Agent execution context with user_id.

        Returns:
            Chunk content and metadata.
        """
        # Parse and validate parameters
        chunk_id = args.get("chunk_id", "").strip()
        collection_id = args.get("collection_id", "").strip()

        if not chunk_id:
            return {"error": "chunk_id is required"}
        if not collection_id:
            return {"error": "collection_id is required"}

        # Validate collection_id is a valid UUID
        try:
            validated_collection_id = str(UUID(collection_id))
        except ValueError as e:
            return {"error": f"Invalid collection_id UUID format: {e}"}

        # Get user_id from context
        if not context or not context.user_id:
            return {"error": "User context required for chunk retrieval"}

        try:
            user_id = int(context.user_id)
        except (ValueError, TypeError):
            return {"error": "Invalid user_id in context"}

        try:
            async with pg_connection_manager.get_session() as session:
                # First verify access to collection
                collection_repo = CollectionRepository(session)
                collection = await collection_repo.get_by_uuid_with_permission_check(validated_collection_id, user_id)

                if not collection:
                    return {
                        "error": "Collection not found or access denied",
                        "collection_id": collection_id,
                    }

                # Get the chunk by metadata chunk_id
                chunk_repo = ChunkRepository(session)
                chunk = await chunk_repo.get_chunk_by_metadata_chunk_id(
                    chunk_id=chunk_id,
                    collection_id=validated_collection_id,
                )

                if not chunk:
                    return {
                        "error": "Chunk not found",
                        "chunk_id": chunk_id,
                        "collection_id": collection_id,
                    }

                # Build response
                metadata = chunk.metadata if hasattr(chunk, "metadata") else (chunk.meta or {})

                return {
                    "id": str(chunk.id),
                    "chunk_id": chunk_id,
                    "content": chunk.content,
                    "document_id": str(chunk.document_id),
                    "chunk_index": chunk.chunk_index,
                    "collection_id": str(chunk.collection_id),
                    "metadata": metadata,
                }

        except ValueError as e:
            # Handle validation errors from repository
            return {"error": str(e)}
        except Exception as e:
            logger.exception("Chunk retrieval failed")
            error_msg = str(e)
            if "Access denied" in error_msg or "AccessDeniedError" in str(type(e)):
                return {"error": "Access denied to collection"}
            raise ToolExecutionError(
                f"Chunk retrieval failed: {error_msg}",
                tool_name="get_chunk",
                cause=error_msg,
            ) from e
