"""Search Service for managing search-related business logic."""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Literal

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from shared.config import settings
from shared.config.internal_api_key import ensure_internal_api_key
from shared.database.exceptions import AccessDeniedError, EntityNotFoundError
from shared.database.models import Collection, CollectionStatus
from shared.database.repositories.collection_repository import CollectionRepository
from shared.database.repositories.user_preferences_repository import UserPreferencesRepository
from shared.llm.exceptions import LLMError, LLMNotConfiguredError
from shared.llm.factory import LLMServiceFactory
from shared.llm.hyde import HyDEConfig, HyDEResult, generate_hyde_expansion
from shared.llm.types import LLMQualityTier, LLMResponse
from shared.llm.usage_tracking import record_llm_usage
from shared.plugins.loader import load_plugins
from shared.plugins.registry import plugin_registry

# Type alias for search mode
SearchMode = Literal["dense", "sparse", "hybrid"]

logger = logging.getLogger(__name__)


def _vecpipe_headers() -> dict[str, str]:
    """Return auth headers for internal vecpipe requests."""
    try:
        key = ensure_internal_api_key(settings)
    except RuntimeError as exc:
        logger.error("Internal API key not configured for vecpipe requests: %s", exc)
        raise
    return {"X-Internal-Api-Key": key}


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
        return str(models[0])  # Return the first (primary) model

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
            except EntityNotFoundError as e:
                logger.error("Collection not found during access validation (user=%s, id=%s): %s", user_id, uuid, e)
                raise
            except AccessDeniedError as e:
                logger.error("Access denied during collection access validation (user=%s, id=%s): %s", user_id, uuid, e)
                raise

        return collections

    async def search_single_collection(
        self,
        collection: Collection,
        query: str,
        k: int,
        search_params: dict[str, Any],
        timeout: httpx.Timeout | None = None,
    ) -> tuple[Collection, dict[str, Any] | None, str | None]:
        """Search a single collection and return results.

        Args:
            collection: Collection to search
            query: Search query
            k: Number of results to return
            search_params: Additional search parameters
            timeout: HTTP timeout settings

        Returns:
            Tuple of (collection, search_response, error_message)
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

        # Remove any legacy hybrid mode parameters
        base_params.pop("hybrid_mode", None)
        base_params.pop("keyword_mode", None)
        base_params.pop("hybrid_alpha", None)

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
            headers = _vecpipe_headers()
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{settings.SEARCH_API_URL}/search",
                    json=collection_search_params,
                    headers=headers,
                )
                response.raise_for_status()

            result: dict[str, Any] = response.json()
            # Preserve sparse/hybrid fallback metadata returned by vecpipe.
            result.setdefault("search_mode_used", collection_search_params.get("search_mode", "dense"))
            if not isinstance(result.get("warnings"), list):
                result["warnings"] = []
            return (collection, result, None)

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
                    response = await client.post(
                        f"{settings.SEARCH_API_URL}/search",
                        json=collection_search_params,
                        headers=headers,
                    )
                    response.raise_for_status()

                retry_result: dict[str, Any] = response.json()
                # Preserve sparse/hybrid fallback metadata returned by vecpipe.
                retry_result.setdefault("search_mode_used", collection_search_params.get("search_mode", "dense"))
                if not isinstance(retry_result.get("warnings"), list):
                    retry_result["warnings"] = []
                return (collection, retry_result, None)

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

    async def _resolve_hyde_settings(
        self,
        user_id: int,
        use_hyde_override: bool | None,
    ) -> tuple[bool, LLMQualityTier, int]:
        """Resolve HyDE settings from user preferences with optional override.

        Args:
            user_id: User ID for preferences lookup
            use_hyde_override: Per-request override (None = use preference)

        Returns:
            Tuple of (should_use_hyde, quality_tier, timeout_seconds)
        """
        prefs_repo = UserPreferencesRepository(self.db_session)
        prefs = await prefs_repo.get_or_create(user_id)

        # Per-request override takes precedence
        should_use = use_hyde_override if use_hyde_override is not None else prefs.search_use_hyde

        tier = LLMQualityTier.HIGH if prefs.search_hyde_quality_tier == "high" else LLMQualityTier.LOW
        timeout = prefs.search_hyde_timeout_seconds

        return should_use, tier, timeout

    async def _generate_hyde_if_enabled(
        self,
        user_id: int,
        query: str,
        use_hyde_override: bool | None,
    ) -> tuple[HyDEResult | None, LLMResponse | None, float]:
        """Generate HyDE expansion if enabled for user.

        Args:
            user_id: User ID for LLM provider and preferences
            query: Original search query
            use_hyde_override: Per-request override

        Returns:
            Tuple of (HyDEResult or None, LLMResponse or None, generation_time_ms)
        """
        should_use, tier, timeout = await self._resolve_hyde_settings(user_id, use_hyde_override)

        if not should_use:
            return None, None, 0.0

        start_time = time.time()

        try:
            factory = LLMServiceFactory(self.db_session)
            provider = await factory.create_provider_for_tier(user_id, tier)

            async with provider:
                config = HyDEConfig(
                    timeout_seconds=timeout,
                )
                result, response = await generate_hyde_expansion(provider, query, config=config)

                # Track usage if we got a response
                if response is not None:
                    await record_llm_usage(
                        self.db_session,
                        user_id,
                        response,
                        feature="hyde",
                        quality_tier=tier.value,
                    )

                generation_time = (time.time() - start_time) * 1000
                return result, response, generation_time

        except LLMNotConfiguredError:
            logger.debug("HyDE skipped: LLM not configured for user %s", user_id)
            return (
                HyDEResult(
                    expanded_query=query,
                    original_query=query,
                    success=False,
                    warning="HyDE skipped: LLM not configured",
                ),
                None,
                0.0,
            )

        except LLMError as e:
            logger.warning("HyDE generation failed for user %s: %s", user_id, e)
            generation_time = (time.time() - start_time) * 1000
            return (
                HyDEResult(
                    expanded_query=query,
                    original_query=query,
                    success=False,
                    warning=f"HyDE generation failed: {type(e).__name__}",
                ),
                None,
                generation_time,
            )

        except Exception as e:
            logger.warning("Unexpected error during HyDE generation for user %s: %s", user_id, e)
            generation_time = (time.time() - start_time) * 1000
            return (
                HyDEResult(
                    expanded_query=query,
                    original_query=query,
                    success=False,
                    warning=f"HyDE generation failed: {type(e).__name__}",
                ),
                None,
                generation_time,
            )

    async def multi_collection_search(
        self,
        user_id: int,
        collection_uuids: list[str],
        query: str,
        k: int = 10,
        search_type: str = "semantic",
        search_mode: SearchMode = "dense",
        rrf_k: int = 60,
        score_threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
        use_reranker: bool = True,
        rerank_model: str | None = None,
        reranker_id: str | None = None,
        use_hyde: bool | None = None,
    ) -> dict[str, Any]:
        """Search across multiple collections with result aggregation and re-ranking.

        Args:
            user_id: ID of the user performing the search
            collection_uuids: List of collection UUIDs to search
            query: Search query
            k: Number of results to return
            search_type: Type of search (semantic, question, code)
            search_mode: Search mode (dense, sparse, hybrid)
            rrf_k: RRF constant k for hybrid mode ranking
            score_threshold: Minimum score threshold for results
            metadata_filter: Optional metadata filters
            use_reranker: Whether to use reranking
            rerank_model: Optional reranker model name
            reranker_id: Optional reranker plugin ID (takes precedence over rerank_model)
            use_hyde: Enable HyDE query expansion (None = use user preference)

        Returns:
            Dictionary with search results and metadata
        """
        start_time = time.time()
        warnings: list[str] = []
        search_mode_used: SearchMode = search_mode
        search_modes_used: set[SearchMode] = set()
        warning_set: set[str] = set()

        # Validate collection access
        collections = await self.validate_collection_access(collection_uuids, user_id)

        # Generate HyDE expansion if enabled
        hyde_result, hyde_response, hyde_time_ms = await self._generate_hyde_if_enabled(user_id, query, use_hyde)

        # Determine the query to use for dense embedding
        dense_query: str | None = None
        if hyde_result is not None:
            if hyde_result.success:
                dense_query = hyde_result.expanded_query
            elif hyde_result.warning and hyde_result.warning not in warning_set:
                warning_set.add(hyde_result.warning)
                warnings.append(hyde_result.warning)

        # Resolve reranker_id to model name if provided (takes precedence)
        effective_rerank_model = rerank_model
        if reranker_id:
            resolved_model = _resolve_reranker_id_to_model(reranker_id)
            if resolved_model:
                effective_rerank_model = resolved_model

        # Build common search parameters
        search_params: dict[str, Any] = {
            "search_type": search_type,
            "search_mode": search_mode,
            "rrf_k": rrf_k,
            "score_threshold": score_threshold,
            "filters": metadata_filter,
            "use_reranker": use_reranker,
            "rerank_model": effective_rerank_model,
            "dense_query": dense_query,  # HyDE expanded query for embedding (None if HyDE disabled/failed)
        }

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

        for collection, response, error in search_results:
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
                response = response or {}
                collection_warnings = response.get("warnings", [])
                if isinstance(collection_warnings, list):
                    for warning in collection_warnings:
                        if isinstance(warning, str) and warning and warning not in warning_set:
                            warning_set.add(warning)
                            warnings.append(warning)

                collection_search_mode_used = response.get("search_mode_used", search_mode)
                if collection_search_mode_used in ("dense", "sparse", "hybrid"):
                    search_modes_used.add(collection_search_mode_used)

                results = response.get("results", [])
                if not isinstance(results, list):
                    results = []

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
                        "search_mode_used": collection_search_mode_used,
                        "warnings": collection_warnings if isinstance(collection_warnings, list) else [],
                    }
                )

        # If sparse/hybrid search was requested, vecpipe may fall back to dense per collection.
        # Report the most conservative mode when collection modes differ.
        if search_modes_used:
            if len(search_modes_used) == 1:
                search_mode_used = next(iter(search_modes_used))
            else:
                if "dense" in search_modes_used:
                    search_mode_used = "dense"
                elif "hybrid" in search_modes_used:
                    search_mode_used = "hybrid"
                else:
                    search_mode_used = "sparse"

        # Sort merged results by score (results are already reranked by vecpipe if reranking was enabled)
        all_results.sort(key=self._result_sort_key, reverse=True)

        # Limit to requested k results
        final_results = all_results[:k]

        processing_time = time.time() - start_time

        # Build HyDE metadata for response
        hyde_used = hyde_result is not None and hyde_result.success
        hyde_info: dict[str, Any] | None = None
        if hyde_used and hyde_result is not None:
            hyde_info = {
                "expanded_query": hyde_result.expanded_query,
                "generation_time_ms": hyde_time_ms,
                "tokens_used": hyde_response.total_tokens if hyde_response else None,
                "provider": hyde_response.provider if hyde_response else None,
                "model": hyde_response.model if hyde_response else None,
            }

        return {
            "results": final_results,
            "metadata": {
                "total_results": len(final_results),
                "collections_searched": len(collections),
                "collection_details": collection_results,
                "processing_time": processing_time,
                "errors": errors if errors else None,
                "search_mode_used": search_mode_used,
                "warnings": warnings,
                "hyde_used": hyde_used,
                "hyde_info": hyde_info,
            },
        }

    async def single_collection_search(
        self,
        user_id: int,
        collection_uuid: str,
        query: str,
        k: int = 10,
        search_type: str = "semantic",
        search_mode: SearchMode = "dense",
        rrf_k: int = 60,
        score_threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
        use_reranker: bool = True,
        rerank_model: str | None = None,
        reranker_id: str | None = None,
        include_content: bool = True,
        use_hyde: bool | None = None,
    ) -> dict[str, Any]:
        """Search a single collection with optional re-ranking.

        Args:
            user_id: ID of the user performing the search
            collection_uuid: UUID of the collection to search
            query: Search query
            k: Number of results to return
            search_type: Type of search (semantic, question, code)
            search_mode: Search mode (dense, sparse, hybrid)
            rrf_k: RRF constant k for hybrid mode ranking
            score_threshold: Minimum score threshold
            metadata_filter: Optional metadata filters
            use_reranker: Whether to use re-ranking
            rerank_model: Optional reranker model name
            reranker_id: Optional reranker plugin ID (takes precedence over rerank_model)
            include_content: Whether to include document content
            use_hyde: Enable HyDE query expansion (None = use user preference)

        Returns:
            Dictionary with search results
        """
        warnings: list[str] = []
        search_mode_used: SearchMode = search_mode

        # Validate collection access
        collections = await self.validate_collection_access([collection_uuid], user_id)
        collection = collections[0]

        # Generate HyDE expansion if enabled
        hyde_result, hyde_response, hyde_time_ms = await self._generate_hyde_if_enabled(user_id, query, use_hyde)

        # Determine the query to use for dense embedding
        dense_query: str | None = None
        if hyde_result is not None:
            if hyde_result.success:
                dense_query = hyde_result.expanded_query
            elif hyde_result.warning:
                warnings.append(hyde_result.warning)

        # Resolve reranker_id to model name if provided (takes precedence)
        effective_rerank_model = rerank_model
        if reranker_id:
            resolved_model = _resolve_reranker_id_to_model(reranker_id)
            if resolved_model:
                effective_rerank_model = resolved_model

        search_params: dict[str, Any] = {
            "query": query,
            "k": k,
            "collection": collection.vector_store_name,
            "model_name": collection.embedding_model,
            "quantization": collection.quantization,
            "search_type": search_type,
            "search_mode": search_mode,
            "rrf_k": rrf_k,
            "score_threshold": score_threshold,
            "filters": metadata_filter,
            "use_reranker": use_reranker,
            "rerank_model": effective_rerank_model,
            "include_content": include_content,
            "dense_query": dense_query,  # HyDE expanded query for embedding
        }

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
            headers = _vecpipe_headers()
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(f"{settings.SEARCH_API_URL}/search", json=search_params, headers=headers)
                response.raise_for_status()

            result: dict[str, Any] = response.json()
            # Add search_mode metadata to result
            result["search_mode_used"] = result.get("search_mode_used", search_mode_used)

            # Merge warnings from HyDE and vecpipe
            vecpipe_warnings = result.get("warnings", [])
            if not isinstance(vecpipe_warnings, list):
                vecpipe_warnings = []
            result["warnings"] = warnings + vecpipe_warnings

            # Add HyDE metadata to result
            hyde_used = hyde_result is not None and hyde_result.success
            result["hyde_used"] = hyde_used
            if hyde_used and hyde_result is not None:
                result["hyde_info"] = {
                    "expanded_query": hyde_result.expanded_query,
                    "generation_time_ms": hyde_time_ms,
                    "tokens_used": hyde_response.total_tokens if hyde_response else None,
                    "provider": hyde_response.provider if hyde_response else None,
                    "model": hyde_response.model if hyde_response else None,
                }
            else:
                result["hyde_info"] = None

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

    async def benchmark_search(
        self,
        collection: Collection,
        query: str,
        search_mode: SearchMode,
        use_reranker: bool,
        top_k: int,
        rrf_k: int = 60,
        score_threshold: float | None = None,
    ) -> "BenchmarkSearchResult":
        """Low-level search for benchmarking.

        This method differs from the standard search methods in several ways:
        - No HyDE expansion (disabled for reproducibility)
        - No auth checks (collection passed in pre-validated)
        - Returns raw chunk-level results with document IDs
        - Returns detailed timing breakdowns

        Args:
            collection: Pre-fetched Collection instance (already validated)
            query: Search query text
            search_mode: Search mode (dense, sparse, hybrid)
            use_reranker: Whether to use reranking
            top_k: Number of results to return
            rrf_k: RRF constant k for hybrid mode (default: 60)
            score_threshold: Minimum score threshold (optional)

        Returns:
            BenchmarkSearchResult with chunks, timing, and total results
        """
        search_params: dict[str, Any] = {
            "query": query,
            "k": top_k,
            "collection": collection.vector_store_name,
            "model_name": collection.embedding_model,
            "quantization": collection.quantization,
            "search_mode": search_mode,
            "rrf_k": rrf_k,
            "score_threshold": score_threshold,
            "use_reranker": use_reranker,
            "include_content": False,  # Not needed for metrics calculation
        }

        # Note: Reranker model selection is handled by vecpipe based on the embedding model
        # We don't override it here for benchmark consistency

        start = time.perf_counter()

        try:
            headers = _vecpipe_headers()
            async with httpx.AsyncClient(timeout=self.default_timeout) as client:
                response = await client.post(
                    f"{settings.SEARCH_API_URL}/search",
                    json=search_params,
                    headers=headers,
                )
                response.raise_for_status()

            elapsed_ms = int((time.perf_counter() - start) * 1000)
            data: dict[str, Any] = response.json()

            # Extract timing from response metadata if available
            metadata = data.get("metadata", {})
            rerank_time_ms: int | None = metadata.get("rerank_time_ms")
            search_time_ms = elapsed_ms - (rerank_time_ms or 0)

            # Format chunks for metrics calculation
            # Each result has document_id, chunk_id, and score
            chunks: list[dict[str, Any]] = []
            for r in data.get("results", []):
                chunks.append(
                    {
                        "doc_id": r.get("document_id"),
                        "chunk_id": r.get("chunk_id"),
                        "score": r.get("score", 0.0),
                    }
                )

            return BenchmarkSearchResult(
                chunks=chunks,
                search_time_ms=search_time_ms,
                rerank_time_ms=rerank_time_ms,
                total_results=len(chunks),
            )

        except httpx.HTTPStatusError as e:
            logger.error(
                "Benchmark search failed for collection %s with status %s",
                collection.name,
                e.response.status_code,
                exc_info=True,
            )
            # Return empty result on error
            return BenchmarkSearchResult(
                chunks=[],
                search_time_ms=int((time.perf_counter() - start) * 1000),
                rerank_time_ms=None,
                total_results=0,
            )

        except Exception as e:
            logger.error(
                "Benchmark search failed for collection %s: %s",
                collection.name,
                e,
                exc_info=True,
            )
            return BenchmarkSearchResult(
                chunks=[],
                search_time_ms=int((time.perf_counter() - start) * 1000),
                rerank_time_ms=None,
                total_results=0,
            )


@dataclass
class BenchmarkSearchResult:
    """Result from benchmark search with timing details.

    Attributes:
        chunks: List of raw chunks with doc_id, chunk_id, and score
        search_time_ms: Time spent on vector search in milliseconds
        rerank_time_ms: Time spent on reranking (if applicable)
        total_results: Total number of results returned
    """

    chunks: list[dict[str, Any]]
    search_time_ms: int
    rerank_time_ms: int | None
    total_results: int
