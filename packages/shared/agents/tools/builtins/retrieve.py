"""Document retrieval tool for agent use.

This tool enables LLM agents to retrieve document metadata and
optionally content from Semantik collections.
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
from shared.database.repositories.document_repository import DocumentRepository

if TYPE_CHECKING:
    from shared.agents.types import AgentContext

logger = logging.getLogger(__name__)


class DocumentRetrieveTool(AgentTool):
    """Retrieve a document by ID.

    This tool allows agents to fetch document metadata and optionally
    the full content or list of chunks for a specific document.
    """

    @property
    def definition(self) -> ToolDefinition:
        """Return the tool definition."""
        return ToolDefinition(
            name="retrieve_document",
            description=(
                "Retrieve a document by its ID. Returns document metadata including "
                "file name, status, and chunk count. Optionally retrieves full content "
                "or list of chunks."
            ),
            parameters=[
                ToolParameter(
                    name="document_id",
                    type="string",
                    description="The UUID of the document to retrieve.",
                    required=True,
                ),
                ToolParameter(
                    name="collection_id",
                    type="string",
                    description="The UUID of the collection containing the document.",
                    required=True,
                ),
                ToolParameter(
                    name="include_chunks",
                    type="boolean",
                    description="Whether to include a list of chunk IDs and indices.",
                    required=False,
                    default=False,
                ),
            ],
            category="search",
            requires_context=True,
            is_destructive=False,
            timeout_seconds=30.0,
        )

    async def execute(
        self,
        args: dict[str, Any],
        context: AgentContext | None = None,
    ) -> dict[str, Any]:
        """Execute the document retrieval.

        Args:
            args: Tool arguments (document_id, collection_id, include_chunks).
            context: Agent execution context with user_id.

        Returns:
            Document metadata and optionally chunk list.
        """
        # Parse and validate parameters
        document_id = args.get("document_id", "").strip()
        collection_id = args.get("collection_id", "").strip()
        include_chunks = args.get("include_chunks", False)

        if not document_id:
            return {"error": "document_id is required"}
        if not collection_id:
            return {"error": "collection_id is required"}

        # Validate UUIDs
        try:
            validated_doc_id = str(UUID(document_id))
            validated_collection_id = str(UUID(collection_id))
        except ValueError as e:
            return {"error": f"Invalid UUID format: {e}"}

        # Get user_id from context
        if not context or not context.user_id:
            return {"error": "User context required for document retrieval"}

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

                # Get the document
                doc_repo = DocumentRepository(session)
                document = await doc_repo.get_by_id(validated_doc_id)

                if not document:
                    return {
                        "error": "Document not found",
                        "document_id": document_id,
                    }

                # Verify document belongs to the specified collection
                if str(document.collection_id) != validated_collection_id:
                    return {
                        "error": "Document does not belong to the specified collection",
                        "document_id": document_id,
                        "collection_id": collection_id,
                    }

                # Build response
                result: dict[str, Any] = {
                    "id": str(document.id),
                    "file_name": document.file_name,
                    "file_path": document.file_path,
                    "mime_type": document.mime_type,
                    "status": document.status.value if hasattr(document.status, "value") else str(document.status),
                    "chunk_count": document.chunk_count or 0,
                    "file_size": document.file_size,
                    "created_at": document.created_at.isoformat() if document.created_at else None,
                    "updated_at": document.updated_at.isoformat() if document.updated_at else None,
                    "metadata": document.meta or {},
                }

                # Optionally include chunk list
                if include_chunks:
                    chunk_repo = ChunkRepository(session)
                    chunks = await chunk_repo.get_chunks_by_document(
                        document_id=validated_doc_id,
                        collection_id=validated_collection_id,
                        limit=500,  # Reasonable limit
                    )

                    result["chunks"] = [
                        {
                            "id": str(chunk.id),
                            "index": chunk.chunk_index,
                            "metadata_chunk_id": chunk.metadata.get("chunk_id") if chunk.metadata else None,
                        }
                        for chunk in chunks
                    ]

                return result

        except Exception as e:
            logger.exception("Document retrieval failed")
            error_msg = str(e)
            if "Access denied" in error_msg or "AccessDeniedError" in str(type(e)):
                return {"error": "Access denied to collection"}
            if "not found" in error_msg.lower():
                return {"error": "Document or collection not found"}
            raise ToolExecutionError(
                f"Document retrieval failed: {error_msg}",
                tool_name="retrieve_document",
                cause=error_msg,
            ) from e
