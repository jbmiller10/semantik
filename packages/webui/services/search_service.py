"""Search Service for managing search-related business logic."""

import asyncio
import logging
import time
from typing import Any

import httpx
from shared.config import settings
from shared.database.exceptions import AccessDeniedError, EntityNotFoundError
from shared.database.models import Collection, CollectionStatus
from shared.database.repositories.collection_repository import CollectionRepository
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class SearchService:
    """Service for managing search-related business logic."""

    def __init__(
        self,
        db_session: AsyncSession,
        collection_repo: CollectionRepository,
    ):
        """Initialize the search service."""
        self.db_session = db_session
        self.collection_repo = collection_repo

    async def validate_collection_access(self, collection_uuids: list[str], user_id: int) -> list[Collection]:
        """Validate user has access to all requested collections.

        Args:
            collection_uuids: List of collection UUIDs to validate
            user_id: ID of the user requesting access

        Returns:
            List of Collection objects user has access to

        Raises:
            AccessDeniedError: If user doesn't have access to any collection
            EntityNotFoundError: If any collection is not found
        """
        collections: list[Collection] = []

        for uuid in collection_uuids:
            try:
                collection = await self.collection_repo.get_by_uuid_with_permission_check(
                    collection_uuid=uuid, user_id=user_id
                )
                collections.append(collection)
            except (EntityNotFoundError, AccessDeniedError) as e:
                logger.error(f"Error accessing collection {uuid}: {e}")
                raise AccessDeniedError(f"Access denied or collection not found: {uuid}") from e

        return collections

    async def search_single_collection(
        self,
        collection: Collection,
        query: str,
        k: int,
        search_params: dict[str, Any],
        timeout: httpx.Timeout | None = None,
    ) -> tuple[Collection, list[dict[str, Any]] | None, str | None]:
        """Search a single collection and return results.

        Args:
            collection: Collection to search
            query: Search query
            k: Number of results to return
            search_params: Additional search parameters
            timeout: HTTP timeout settings

        Returns:
            Tuple of (collection, results, error_message)
        """
        # Skip collections that aren't ready
        if collection.status != CollectionStatus.READY:
            return (collection, None, f"Collection {collection.name} is not ready for search")

        # Use default timeout if not provided
        if timeout is None:
            timeout = httpx.Timeout(timeout=30.0, connect=5.0, read=30.0, write=5.0)

        # Build search request for this collection
        collection_search_params = {
            **search_params,
            "query": query,
            "k": k * settings.SEARCH_CANDIDATE_MULTIPLIER,  # Get more candidates for re-ranking
            "collection": collection.vector_store_name,
            "model_name": collection.embedding_model,
            "quantization": collection.quantization,
            "include_content": True,
            "use_reranker": False,  # We'll do re-ranking after merging
        }

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(f"{settings.SEARCH_API_URL}/search", json=collection_search_params)
                response.raise_for_status()

            result = response.json()
            return (collection, result.get("results", []), None)

        except httpx.ReadTimeout:
            # Retry with longer timeout
            logger.warning(f"Search timeout for collection {collection.name}, retrying...")
            extended_timeout = httpx.Timeout(timeout=120.0, connect=5.0, read=120.0, write=5.0)

            try:
                async with httpx.AsyncClient(timeout=extended_timeout) as client:
                    response = await client.post(f"{settings.SEARCH_API_URL}/search", json=collection_search_params)
                    response.raise_for_status()

                result = response.json()
                return (collection, result.get("results", []), None)

            except httpx.HTTPStatusError as e:
                return self._handle_http_error(e, collection, retry=True)
            except Exception as e:
                return (collection, None, f"Search failed after retry: {str(e)}")

        except httpx.HTTPStatusError as e:
            return self._handle_http_error(e, collection, retry=False)

        except httpx.ConnectError:
            return (collection, None, f"Cannot connect to search service for collection '{collection.name}'")

        except httpx.RequestError as e:
            return (collection, None, f"Network error searching collection '{collection.name}': {str(e)}")

        except Exception as e:
            logger.error(f"Unexpected error searching collection {collection.name}: {e}")
            return (collection, None, f"Unexpected error searching collection '{collection.name}': {str(e)}")

    def _handle_http_error(
        self, error: httpx.HTTPStatusError, collection: Collection, retry: bool
    ) -> tuple[Collection, None, str]:
        """Handle HTTP status errors during search."""
        retry_suffix = " after retry" if retry else ""
        status_code = error.response.status_code

        if status_code == 404:
            return (collection, None, f"Collection '{collection.name}' not found in vector store{retry_suffix}")
        if status_code == 403:
            return (collection, None, f"Access denied to collection '{collection.name}'{retry_suffix}")
        if status_code == 429:
            return (collection, None, f"Rate limit exceeded for collection '{collection.name}'{retry_suffix}")
        if status_code >= 500:
            return (
                collection,
                None,
                f"Search service unavailable for collection '{collection.name}'{retry_suffix} (status: {status_code})",
            )
        return (
            collection,
            None,
            f"Search failed for collection '{collection.name}'{retry_suffix} (status: {status_code})",
        )

    async def rerank_merged_results(
        self,
        query: str,
        results: list[tuple[Collection, dict[str, Any]]],
        rerank_model: str | None = None,
        k: int = 10,
    ) -> list[tuple[Collection, dict[str, Any], float]]:
        """Re-rank merged results from multiple collections.

        Args:
            query: Original search query
            results: List of (collection, result) tuples
            rerank_model: Optional reranker model name
            k: Number of top results to return

        Returns:
            List of (collection, result, reranked_score) tuples
        """
        if not results:
            return []

        # Prepare documents for re-ranking
        documents = []
        for _, result in results:
            # Use content if available, otherwise use metadata
            content = result.get("content", "")
            if not content and result.get("metadata"):
                content = str(result.get("metadata", {}))
            documents.append(content)

        # Build re-ranking request
        rerank_params = {
            "query": query,
            "documents": documents,
            "model_name": rerank_model or "Qwen/Qwen3-Reranker",
            "k": min(k, len(documents)),
        }

        try:
            timeout = httpx.Timeout(timeout=60.0, connect=5.0, read=60.0, write=5.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(f"{settings.SEARCH_API_URL}/rerank", json=rerank_params)
                response.raise_for_status()

            rerank_result = response.json()
            reranked_indices = rerank_result.get("results", [])

            # Build reranked results with new scores
            reranked_results = []
            for idx, score in reranked_indices:
                if 0 <= idx < len(results):
                    collection, result = results[idx]
                    reranked_results.append((collection, result, score))

            return reranked_results[:k]

        except Exception as e:
            logger.error(f"Re-ranking failed: {e}")
            # Fall back to original scores
            scored_results = [(collection, result, result.get("score", 0.0)) for collection, result in results]
            scored_results.sort(key=lambda x: x[2], reverse=True)
            return scored_results[:k]

    async def multi_collection_search(
        self,
        user_id: int,
        collection_uuids: list[str],
        query: str,
        k: int = 10,
        search_type: str = "semantic",
        score_threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
        rerank_model: str | None = None,
        hybrid_alpha: float = 0.7,
        hybrid_search_mode: str = "weighted",
    ) -> dict[str, Any]:
        """Search across multiple collections with result aggregation and re-ranking.

        Args:
            user_id: ID of the user performing the search
            collection_uuids: List of collection UUIDs to search
            query: Search query
            k: Number of results to return
            search_type: Type of search (semantic, hybrid, etc.)
            score_threshold: Minimum score threshold for results
            metadata_filter: Optional metadata filters
            rerank_model: Optional reranker model
            hybrid_alpha: Weight for hybrid search
            hybrid_search_mode: Mode for hybrid search

        Returns:
            Dictionary with search results and metadata
        """
        start_time = time.time()

        # Validate collection access
        collections = await self.validate_collection_access(collection_uuids, user_id)

        # Build common search parameters
        search_params = {
            "search_type": search_type,
            "score_threshold": score_threshold,
            "filters": metadata_filter,
        }

        # Add hybrid search parameters if applicable
        if search_type == "hybrid":
            search_params.update(
                {
                    "hybrid_alpha": hybrid_alpha,
                    "hybrid_search_mode": hybrid_search_mode,
                }
            )

        # Create timeout for searches
        timeout = httpx.Timeout(timeout=30.0, connect=5.0, read=30.0, write=5.0)

        # Execute searches in parallel
        search_tasks = [
            self.search_single_collection(collection, query, k, search_params, timeout) for collection in collections
        ]

        search_results = await asyncio.gather(*search_tasks)

        # Process results
        all_results = []
        collection_results = []
        errors = []

        for collection, results, error in search_results:
            if error:
                errors.append(error)
                collection_results.append(
                    {
                        "collection_id": collection.id,
                        "collection_name": collection.name,
                        "result_count": 0,
                        "error": error,
                    }
                )
            else:
                if results:
                    # Add collection info to each result
                    for result in results:
                        all_results.append((collection, result))

                collection_results.append(
                    {
                        "collection_id": collection.id,
                        "collection_name": collection.name,
                        "result_count": len(results) if results else 0,
                    }
                )

        # Re-rank merged results
        reranked_results = await self.rerank_merged_results(query, all_results, rerank_model, k)

        # Format final results
        final_results = []
        for collection, result, score in reranked_results:
            final_results.append(
                {
                    "collection_id": collection.id,
                    "collection_name": collection.name,
                    **result,
                    "reranked_score": score,
                }
            )

        processing_time = time.time() - start_time

        return {
            "results": final_results,
            "metadata": {
                "total_results": len(final_results),
                "collections_searched": len(collections),
                "collection_details": collection_results,
                "processing_time": processing_time,
                "errors": errors if errors else None,
            },
        }

    async def single_collection_search(
        self,
        user_id: int,
        collection_uuid: str,
        query: str,
        k: int = 10,
        search_type: str = "semantic",
        score_threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
        use_reranker: bool = True,
        rerank_model: str | None = None,
        hybrid_alpha: float = 0.7,
        hybrid_search_mode: str = "weighted",
        include_content: bool = True,
    ) -> dict[str, Any]:
        """Search a single collection with optional re-ranking.

        Args:
            user_id: ID of the user performing the search
            collection_uuid: UUID of the collection to search
            query: Search query
            k: Number of results to return
            search_type: Type of search
            score_threshold: Minimum score threshold
            metadata_filter: Optional metadata filters
            use_reranker: Whether to use re-ranking
            rerank_model: Optional reranker model
            hybrid_alpha: Weight for hybrid search
            hybrid_search_mode: Mode for hybrid search
            include_content: Whether to include document content

        Returns:
            Dictionary with search results
        """
        # Validate collection access
        collections = await self.validate_collection_access([collection_uuid], user_id)
        collection = collections[0]

        # Build search parameters
        search_params = {
            "query": query,
            "k": k,
            "collection": collection.vector_store_name,
            "model_name": collection.embedding_model,
            "quantization": collection.quantization,
            "search_type": search_type,
            "score_threshold": score_threshold,
            "filters": metadata_filter,
            "use_reranker": use_reranker,
            "rerank_model": rerank_model,
            "include_content": include_content,
        }

        # Add hybrid search parameters if applicable
        if search_type == "hybrid":
            search_params.update(
                {
                    "hybrid_alpha": hybrid_alpha,
                    "hybrid_search_mode": hybrid_search_mode,
                }
            )

        try:
            timeout = httpx.Timeout(timeout=60.0, connect=5.0, read=60.0, write=5.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(f"{settings.SEARCH_API_URL}/search", json=search_params)
                response.raise_for_status()

            return response.json()

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise EntityNotFoundError(f"Collection '{collection.name}' not found in vector store") from e
            if e.response.status_code == 403:
                raise AccessDeniedError(f"Access denied to collection '{collection.name}'") from e
            logger.error(f"Search failed with status {e.response.status_code}: {e}")
            raise

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
