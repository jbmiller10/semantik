"""Semantic search tool for agent use.

This tool enables LLM agents to search documents by semantic similarity
within Semantik collections.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from uuid import UUID

from shared.agents.exceptions import ToolExecutionError
from shared.agents.tools.base import AgentTool, ToolDefinition, ToolParameter
from shared.database import pg_connection_manager
from shared.database.repositories.collection_repository import CollectionRepository

if TYPE_CHECKING:
    from shared.agents.types import AgentContext

logger = logging.getLogger(__name__)


class SemanticSearchTool(AgentTool):
    """Search documents by semantic similarity.

    This tool allows agents to search across one or more collections
    using semantic (vector) search, hybrid search, or keyword search.

    Configuration options (set at registration time):
        - use_reranker: Whether to apply reranking to results
        - reranker_id: Reranker plugin ID
        - hybrid_alpha: Weight for hybrid search
        - default_search_type: Default search type
    """

    def __init__(
        self,
        *,
        use_reranker: bool = False,
        reranker_id: str | None = None,
        hybrid_alpha: float = 0.5,
        default_search_type: str = "semantic",
    ) -> None:
        """Initialize the semantic search tool.

        Args:
            use_reranker: Whether to apply reranking to search results.
            reranker_id: Reranker plugin ID (e.g., "qwen3-reranker").
            hybrid_alpha: Weight for hybrid search (0=keyword, 1=semantic).
            default_search_type: Default search type if not specified.
        """
        self._use_reranker = use_reranker
        self._reranker_id = reranker_id
        self._hybrid_alpha = hybrid_alpha
        self._default_search_type = default_search_type

    @property
    def definition(self) -> ToolDefinition:
        """Return the tool definition."""
        return ToolDefinition(
            name="semantic_search",
            description=(
                "Search documents by semantic similarity. Returns the most relevant "
                "text chunks from the specified collections. Use this to find information "
                "related to a query across your document collections."
            ),
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="The search query text. Be specific and descriptive.",
                    required=True,
                ),
                ToolParameter(
                    name="top_k",
                    type="integer",
                    description="Number of results to return (1-50).",
                    required=False,
                    default=10,
                ),
                ToolParameter(
                    name="collection_ids",
                    type="array",
                    description="List of collection UUIDs to search. If not provided, uses the current context collection.",
                    required=False,
                    items={"type": "string"},
                ),
                ToolParameter(
                    name="score_threshold",
                    type="number",
                    description="Minimum similarity score threshold (0.0-1.0).",
                    required=False,
                    default=0.0,
                ),
                ToolParameter(
                    name="search_type",
                    type="string",
                    description="Type of search to perform.",
                    required=False,
                    default=self._default_search_type,
                    enum=["semantic", "hybrid", "keyword"],
                ),
            ],
            category="search",
            requires_context=True,
            is_destructive=False,
            timeout_seconds=60.0,
        )

    async def execute(
        self,
        args: dict[str, Any],
        context: AgentContext | None = None,
    ) -> dict[str, Any]:
        """Execute the semantic search.

        Args:
            args: Tool arguments (query, top_k, collection_ids, etc.).
            context: Agent execution context with user_id, collection_id, etc.

        Returns:
            Search results with content, scores, and metadata.
        """
        # Validate query
        query = args.get("query", "").strip()
        if not query:
            return {"error": "Query cannot be empty"}

        # Parse parameters
        top_k = min(max(args.get("top_k", 10), 1), 50)
        score_threshold = args.get("score_threshold", 0.0)
        search_type = args.get("search_type", self._default_search_type)

        # Resolve collection IDs
        collection_ids = args.get("collection_ids", [])
        if not collection_ids and context and context.collection_id:
            collection_ids = [context.collection_id]

        if not collection_ids:
            return {
                "error": "No collection specified. Provide collection_ids or set collection in context."
            }

        # Validate collection IDs are valid UUIDs
        try:
            validated_ids = [str(UUID(cid)) for cid in collection_ids]
        except ValueError as e:
            return {"error": f"Invalid collection UUID format: {e}"}

        # Get user_id from context
        if not context or not context.user_id:
            return {"error": "User context required for search"}

        try:
            user_id = int(context.user_id)
        except (ValueError, TypeError):
            return {"error": "Invalid user_id in context"}

        # Perform search
        try:
            # Import here to avoid circular dependency
            from webui.services.search_service import SearchService

            async with pg_connection_manager.get_session() as session:
                collection_repo = CollectionRepository(session)
                search_service = SearchService(session, collection_repo)

                result = await search_service.multi_collection_search(
                    user_id=user_id,
                    collection_uuids=validated_ids,
                    query=query,
                    k=top_k,
                    search_type=search_type,
                    score_threshold=score_threshold,
                    use_reranker=self._use_reranker,
                    reranker_id=self._reranker_id,
                    hybrid_alpha=self._hybrid_alpha,
                )

            # Format results
            results = result.get("results", [])
            formatted_results = []

            for item in results:
                formatted_results.append({
                    "content": item.get("content", ""),
                    "score": item.get("reranked_score") or item.get("score", 0.0),
                    "doc_id": item.get("doc_id"),
                    "chunk_id": item.get("chunk_id"),
                    "collection_id": item.get("collection_id"),
                    "collection_name": item.get("collection_name"),
                    "metadata": item.get("metadata", {}),
                })

            return {
                "query": query,
                "total": len(formatted_results),
                "results": formatted_results,
            }

        except Exception as e:
            logger.exception("Search execution failed")
            # Return error as data, not exception, so LLM can handle gracefully
            error_msg = str(e)
            if "Access denied" in error_msg or "AccessDeniedError" in str(type(e)):
                return {"error": "Access denied to one or more collections"}
            if "not found" in error_msg.lower():
                return {"error": "One or more collections not found"}
            raise ToolExecutionError(
                f"Search failed: {error_msg}",
                tool_name="semantic_search",
                cause=error_msg,
            ) from e
