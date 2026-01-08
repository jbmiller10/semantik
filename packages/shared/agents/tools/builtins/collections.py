"""Collection listing tool for agent use.

This tool enables LLM agents to discover available document collections.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from shared.agents.exceptions import ToolExecutionError
from shared.agents.tools.base import AgentTool, ToolDefinition, ToolParameter
from shared.database import pg_connection_manager
from shared.database.repositories.collection_repository import CollectionRepository

if TYPE_CHECKING:
    from shared.agents.types import AgentContext

logger = logging.getLogger(__name__)


class ListCollectionsTool(AgentTool):
    """List available document collections.

    This tool allows agents to discover what document collections are
    available for searching, including both owned and public collections.
    """

    @property
    def definition(self) -> ToolDefinition:
        """Return the tool definition."""
        return ToolDefinition(
            name="list_collections",
            description=(
                "List available document collections. Returns collection names, "
                "descriptions, document counts, and status. Use this to discover "
                "what collections are available before searching."
            ),
            parameters=[
                ToolParameter(
                    name="include_public",
                    type="boolean",
                    description="Whether to include public collections in the list.",
                    required=False,
                    default=True,
                ),
                ToolParameter(
                    name="status_filter",
                    type="string",
                    description="Filter by collection status.",
                    required=False,
                    enum=["ready", "processing", "pending", "error", "degraded"],
                ),
                ToolParameter(
                    name="limit",
                    type="integer",
                    description="Maximum number of collections to return (1-100).",
                    required=False,
                    default=50,
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
        """Execute the collection listing.

        Args:
            args: Tool arguments (include_public, status_filter, limit).
            context: Agent execution context with user_id.

        Returns:
            List of collections with metadata.
        """
        # Parse parameters
        include_public = args.get("include_public", True)
        status_filter = args.get("status_filter")
        limit = min(max(args.get("limit", 50), 1), 100)

        # Get user_id from context
        if not context or not context.user_id:
            return {"error": "User context required for listing collections"}

        try:
            user_id = int(context.user_id)
        except (ValueError, TypeError):
            return {"error": "Invalid user_id in context"}

        try:
            async with pg_connection_manager.get_session() as session:
                collection_repo = CollectionRepository(session)

                collections, total = await collection_repo.list_for_user(
                    user_id=user_id,
                    offset=0,
                    limit=limit,
                    include_public=include_public,
                )

                # Format results
                formatted_collections = []
                for coll in collections:
                    # Apply status filter if specified
                    coll_status = coll.status.value if hasattr(coll.status, "value") else str(coll.status)
                    if status_filter and coll_status.lower() != status_filter.lower():
                        continue

                    formatted_collections.append({
                        "id": str(coll.id),
                        "name": coll.name,
                        "description": coll.description,
                        "status": coll_status,
                        "document_count": getattr(coll, "document_count", None),
                        "embedding_model": coll.embedding_model,
                        "is_public": coll.is_public,
                        "is_owned": coll.owner_id == user_id,
                        "created_at": coll.created_at.isoformat() if coll.created_at else None,
                    })

                return {
                    "total": len(formatted_collections),
                    "collections": formatted_collections,
                }

        except Exception as e:
            logger.exception("Collection listing failed")
            raise ToolExecutionError(
                f"Collection listing failed: {e!s}",
                tool_name="list_collections",
                cause=str(e),
            ) from e
