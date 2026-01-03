"""Search Service for managing search-related business logic."""

import asyncio
import logging
import time
from typing import Any

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from shared.config import settings
from shared.contracts.search import normalize_hybrid_mode, normalize_keyword_mode
from shared.database.exceptions import AccessDeniedError, EntityNotFoundError
from shared.database.models import Collection, CollectionStatus
from shared.database.repositories.collection_repository import CollectionRepository
from shared.plugins.loader import load_plugins
from shared.plugins.registry import plugin_registry

logger = logging.getLogger(__name__)


def _resolve_reranker_id_to_model(reranker_id: str | None) -> str | None:
    """Resolve a reranker plugin ID to its model name.

    Args:
        reranker_id: The plugin ID (e.g., "qwen3-reranker")

    Returns:
        The reranker model name from the plugin, or None if not found.
    """
    if not reranker_id:
        return None

    # Ensure reranker plugins are loaded
    load_plugins(plugin_types={"reranker"})

    record = plugin_registry.get("reranker", reranker_id)
    if record is None:
        logger.warning("Reranker plugin not found: %s", reranker_id)
        return None

    # Get the model name from capabilities
    capabilities = record.manifest.capabilities
    models = capabilities.get("models", [])
    if models:
        return models[0]  # Return the first (primary) model

    logger.warning("Reranker plugin %s has no models defined", reranker_id)
    return None


class SearchService:
    """Service for managing search-related business logic."""

    def __init__(
        self,
        db_session: AsyncSession,
        collection_repo: CollectionRepository,
        default_timeout: httpx.Timeout | None = None,
        retry_timeout_multiplier: float = 4.0,
    ):
        """Initialize the search service.

        Args:
            db_session: Database session for transactions
            collection_repo: Repository for collection data access
            default_timeout: Default timeout configuration for HTTP requests.
                            Defaults to Timeout(timeout=30.0, connect=5.0, read=30.0, write=5.0)
            retry_timeout_multiplier: Multiplier applied to timeout values when retrying failed requests.
                                    Defaults to 4.0
        """
        self.db_session = db_session
        self.collection_repo = collection_repo
        self.default_timeout = default_timeout or httpx.Timeout(timeout=30.0, connect=5.0, read=30.0, write=5.0)
        self.retry_timeout_multiplier = retry_timeout_multiplier

    @staticmethod
    def _result_sort_key(result: dict[str, Any]) -> float:
        """Sort by reranked_score when present, otherwise fall back to base score."""

        reranked = result.get("reranked_score")
        if reranked is not None:
            try:
                return float(reranked)
            except (TypeError, ValueError):
                return 0.0
        try:
            return float(result.get("score", 0.0))
        except (TypeError, ValueError):
            return 0.0

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
                logger.error("Error accessing collection %s: %s", uuid, e, exc_info=True)
                raise AccessDeniedError(str(user_id), "collection", uuid) from e

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
            timeout = self.default_timeout

        # Build search request for this collection
        base_params = dict(search_params)
        if "hybrid_search_mode" in base_params:
            raise ValueError("legacy field 'hybrid_search_mode' is no longer supported")

        if "hybrid_mode" in base_params and base_params["hybrid_mode"] is not None:
            base_params["hybrid_mode"] = normalize_hybrid_mode(base_params["hybrid_mode"])

        if "keyword_mode" in base_params:
            base_params["keyword_mode"] = normalize_keyword_mode(base_params.get("keyword_mode"))

        collection_search_params = {
            **base_params,
            "query": query,
            "k": k,  # Request exactly k results (vecpipe will handle candidate multiplier if reranking)
            "collection": collection.vector_store_name,
            "model_name": collection.embedding_model,
            "quantization": collection.quantization,
            "include_content": True,
        }
        # Ensure legacy hybrid_search_mode never leaks into the payload we send to vecpipe
        collection_search_params.pop("hybrid_search_mode", None)

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(f"{settings.SEARCH_API_URL}/search", json=collection_search_params)
                response.raise_for_status()

            result = response.json()
            return (collection, result.get("results", []), None)

        except httpx.ReadTimeout:
            # Retry with longer timeout
            logger.warning("Search timeout for collection %s, retrying...", collection.name, exc_info=True)
            # Calculate extended timeout by multiplying current timeout values
            # Calculate a general timeout based on the maximum of individual timeouts
            # Cap the retry timeout to a reasonable limit (e.g., 60s) to prevent excessive hangs
            max_timeout = min(
                max(timeout.connect or 0, timeout.read or 0, timeout.write or 0, timeout.pool or 0)
                * self.retry_timeout_multiplier,
                60.0,
            )
            extended_timeout = httpx.Timeout(
                timeout=max_timeout if max_timeout > 0 else 60.0,
                connect=min(timeout.connect * self.retry_timeout_multiplier if timeout.connect else 20.0, 30.0),
                read=min(timeout.read * self.retry_timeout_multiplier if timeout.read else 60.0, 60.0),
                write=min(timeout.write * self.retry_timeout_multiplier if timeout.write else 20.0, 30.0),
            )

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
            logger.error("Unexpected error searching collection %s: %s", collection.name, e, exc_info=True)
            return (collection, None, f"Unexpected error searching collection '{collection.name}': {str(e)}")

    def _handle_http_error(
        self, error: httpx.HTTPStatusError, collection: Collection, retry: bool
    ) -> tuple[Collection, None, str]:
        """Handle HTTP status errors during search operations.

        Maps HTTP status codes to appropriate error messages for user feedback.

        Args:
            error: The HTTP status error that occurred
            collection: The collection that was being searched
            retry: Whether this error occurred during a retry attempt

        Returns:
            Tuple of (collection, None for results, error message)
        """
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

    async def multi_collection_search(
        self,
        user_id: int,
        collection_uuids: list[str],
        query: str,
        k: int = 10,
        search_type: str = "semantic",
        score_threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
        use_reranker: bool = True,
        rerank_model: str | None = None,
        reranker_id: str | None = None,
        hybrid_alpha: float = 0.7,
        hybrid_mode: str = "weighted",
        keyword_mode: str = "any",
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
            use_reranker: Whether to use reranking
            rerank_model: Optional reranker model name
            reranker_id: Optional reranker plugin ID (takes precedence over rerank_model)
            hybrid_alpha: Weight for hybrid search
            hybrid_mode: Mode for hybrid search
            keyword_mode: Keyword matching mode when using hybrid search

        Returns:
            Dictionary with search results and metadata
        """
        start_time = time.time()

        # Validate collection access
        collections = await self.validate_collection_access(collection_uuids, user_id)

        # Validate requested hybrid/keyword modes
        hybrid_mode = normalize_hybrid_mode(hybrid_mode)
        keyword_mode = normalize_keyword_mode(keyword_mode)

        # Resolve reranker_id to model name if provided (takes precedence)
        effective_rerank_model = rerank_model
        if reranker_id:
            resolved_model = _resolve_reranker_id_to_model(reranker_id)
            if resolved_model:
                effective_rerank_model = resolved_model

        # Build common search parameters
        search_params = {
            "search_type": search_type,
            "score_threshold": score_threshold,
            "filters": metadata_filter,
            "use_reranker": use_reranker,
            "rerank_model": effective_rerank_model,
        }

        # Add hybrid search parameters if applicable
        if search_type == "hybrid":
            search_params.update(
                {
                    "hybrid_alpha": hybrid_alpha,
                    "hybrid_mode": hybrid_mode,
                    "keyword_mode": keyword_mode,
                }
            )

        # Create timeout for searches
        timeout = self.default_timeout

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
                        all_results.append(
                            {
                                "collection_id": collection.id,
                                "collection_name": collection.name,
                                **result,
                            }
                        )

                collection_results.append(
                    {
                        "collection_id": collection.id,
                        "collection_name": collection.name,
                        "result_count": len(results) if results else 0,
                    }
                )

        # Sort merged results by score (results are already reranked by vecpipe if reranking was enabled)
        all_results.sort(key=self._result_sort_key, reverse=True)

        # Limit to requested k results
        final_results = all_results[:k]

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
        reranker_id: str | None = None,
        hybrid_alpha: float = 0.7,
        hybrid_mode: str = "weighted",
        keyword_mode: str = "any",
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
            rerank_model: Optional reranker model name
            reranker_id: Optional reranker plugin ID (takes precedence over rerank_model)
            hybrid_alpha: Weight for hybrid search
            hybrid_mode: Mode for hybrid search
            keyword_mode: Keyword matching mode when using hybrid search
            include_content: Whether to include document content

        Returns:
            Dictionary with search results
        """
        # Validate collection access
        collections = await self.validate_collection_access([collection_uuid], user_id)
        collection = collections[0]

        # Build search parameters
        hybrid_mode = normalize_hybrid_mode(hybrid_mode)
        keyword_mode = normalize_keyword_mode(keyword_mode)

        # Resolve reranker_id to model name if provided (takes precedence)
        effective_rerank_model = rerank_model
        if reranker_id:
            resolved_model = _resolve_reranker_id_to_model(reranker_id)
            if resolved_model:
                effective_rerank_model = resolved_model

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
            "rerank_model": effective_rerank_model,
            "include_content": include_content,
        }

        # Add hybrid search parameters if applicable
        if search_type == "hybrid":
            search_params.update(
                {
                    "hybrid_alpha": hybrid_alpha,
                    "hybrid_mode": hybrid_mode,
                    "keyword_mode": keyword_mode,
                }
            )

        try:
            # Use a longer timeout for single collection searches
            # Calculate timeout based on the read timeout (usually the longest operation)
            general_timeout = self.default_timeout.read * 2 if self.default_timeout.read else 60.0
            timeout = httpx.Timeout(
                timeout=general_timeout,
                connect=self.default_timeout.connect if self.default_timeout.connect else 5.0,
                read=self.default_timeout.read * 2 if self.default_timeout.read else 60.0,
                write=self.default_timeout.write if self.default_timeout.write else 5.0,
            )
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(f"{settings.SEARCH_API_URL}/search", json=search_params)
                response.raise_for_status()

            result: dict[str, Any] = response.json()
            return result

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise EntityNotFoundError("collection", collection_uuid) from e
            if e.response.status_code == 403:
                raise AccessDeniedError(str(user_id), "collection", collection_uuid) from e
            logger.error(
                "Search failed with status %s: %s",
                e.response.status_code,
                e,
                exc_info=True,
            )
            raise

        except Exception as e:
            logger.error("Search failed: %s", e, exc_info=True)
            raise
