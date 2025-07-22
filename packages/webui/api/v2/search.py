"""
Search API v2 endpoints with multi-collection support.

This module provides search endpoints that support searching across multiple
collections with result aggregation and re-ranking.
"""

import asyncio
import logging
import time
from typing import Any, cast

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from packages.shared.config import settings
from packages.shared.database import get_db
from packages.shared.database.models import Collection, CollectionStatus
from packages.shared.database.repositories.collection_repository import CollectionRepository
from packages.webui.api.schemas import ErrorResponse
from packages.webui.api.v2.schemas import (
    CollectionSearchRequest,
    CollectionSearchResponse,
    CollectionSearchResult,
    SingleCollectionSearchRequest,
)
from packages.webui.auth import get_current_user
from packages.webui.rate_limiter import limiter
from packages.webui.dependencies import get_collection_repository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/search", tags=["search-v2"])


async def validate_collection_access(collection_uuids: list[str], user_id: int, repository: CollectionRepository) -> list[Collection]:
    """
    Validate user has access to all requested collections.

    Returns list of Collection objects user has access to.
    Raises HTTPException if any collection is not found or access is denied.
    """
    collections: list[Collection] = []

    for uuid in collection_uuids:
        try:
            collection = await repository.get_by_uuid_with_permission_check(collection_uuid=uuid, user_id=user_id)
            collections.append(collection)
        except Exception as e:
            logger.error(f"Error accessing collection {uuid}: {e}")
            raise HTTPException(status_code=403, detail=f"Access denied or collection not found: {uuid}") from e

    return collections


async def search_single_collection(
    collection: Collection,
    query: str,
    k: int,
    search_params: dict[str, Any],
    timeout: httpx.Timeout,
) -> tuple[Collection, list[dict[str, Any]] | None, str | None]:
    """
    Search a single collection and return results.

    Returns: (collection, results, error_message)
    """
    # Skip collections that aren't ready
    if collection.status != CollectionStatus.READY:
        return (collection, None, f"Collection {collection.name} is not ready for search")

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
            # Handle specific HTTP status codes even on retry
            if e.response.status_code == 404:
                return (collection, None, f"Collection '{collection.name}' not found in vector store (after retry)")
            if e.response.status_code >= 500:
                return (
                    collection,
                    None,
                    f"Search service unavailable for collection '{collection.name}' after retry (status: {e.response.status_code})",
                )
            return (
                collection,
                None,
                f"Search failed for collection '{collection.name}' after retry (status: {e.response.status_code})",
            )
        except Exception as e:
            return (collection, None, f"Search failed after retry: {str(e)}")

    except httpx.HTTPStatusError as e:
        # Handle specific HTTP status codes
        if e.response.status_code == 404:
            return (collection, None, f"Collection '{collection.name}' not found in vector store")
        if e.response.status_code == 403:
            return (collection, None, f"Access denied to collection '{collection.name}'")
        if e.response.status_code == 429:
            return (collection, None, f"Rate limit exceeded for collection '{collection.name}'")
        if e.response.status_code >= 500:
            return (
                collection,
                None,
                f"Search service unavailable for collection '{collection.name}' (status: {e.response.status_code})",
            )
        return (
            collection,
            None,
            f"Search failed for collection '{collection.name}' (status: {e.response.status_code})",
        )

    except httpx.ConnectError:
        return (collection, None, f"Cannot connect to search service for collection '{collection.name}'")

    except httpx.RequestError as e:
        return (collection, None, f"Network error searching collection '{collection.name}': {str(e)}")

    except Exception as e:
        logger.error(f"Unexpected error searching collection {collection.name}: {e}")
        return (collection, None, f"Unexpected error searching collection '{collection.name}': {str(e)}")


async def rerank_merged_results(
    query: str,
    results: list[tuple[Collection, dict[str, Any]]],
    rerank_model: str | None = None,
    k: int = 10,
) -> list[tuple[Collection, dict[str, Any], float]]:
    """
    Re-rank merged results from multiple collections.

    Returns list of (collection, result, reranked_score) tuples.
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


@router.post(
    "",
    response_model=CollectionSearchResponse,
    responses={
        403: {"model": ErrorResponse, "description": "Access denied to one or more collections"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
@limiter.limit("30/minute")
async def multi_collection_search(
    request: Request,  # noqa: ARG001
    search_request: CollectionSearchRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
    collection_repo: CollectionRepository = Depends(get_collection_repository),
) -> CollectionSearchResponse:
    """
    Search across multiple collections with result aggregation and re-ranking.

    This endpoint allows searching across up to 10 collections simultaneously.
    Results are aggregated and re-ranked to provide globally relevant results.
    """
    start_time = time.time()

    # Validate collection access
    user_id = int(current_user["id"])
    collections = await validate_collection_access(search_request.collection_uuids, user_id, collection_repo)

    # Build common search parameters
    search_params = {
        "search_type": search_request.search_type,
        "score_threshold": search_request.score_threshold,
        "filters": search_request.metadata_filter,
    }

    # Add hybrid search parameters if applicable
    if search_request.search_type == "hybrid":
        search_params.update(
            {
                "hybrid_alpha": search_request.hybrid_alpha,
                "hybrid_mode": search_request.hybrid_mode,
                "keyword_mode": search_request.keyword_mode,
            }
        )

    # Search collections in parallel
    timeout = httpx.Timeout(timeout=60.0, connect=5.0, read=60.0, write=5.0)

    search_tasks = [
        search_single_collection(
            collection,
            search_request.query,
            search_request.k,
            search_params,
            timeout,
        )
        for collection in collections
    ]

    search_start = time.time()
    search_results = await asyncio.gather(*search_tasks)
    search_time_ms = (time.time() - search_start) * 1000

    # Process results and handle failures
    all_results: list[tuple[Collection, dict[str, Any]]] = []
    failed_collections: list[dict[str, str]] = []
    collections_info: list[dict[str, Any]] = []

    for collection, results, error in search_results:
        if error:
            failed_collections.append(
                {
                    "collection_id": str(collection.id),
                    "collection_name": str(collection.name),
                    "error": error,
                }
            )
        else:
            # Add collection info to results
            for result in results or []:
                all_results.append((collection, result))

            collections_info.append(
                {
                    "id": str(collection.id),
                    "name": str(collection.name),
                    "embedding_model": str(collection.embedding_model),
                    "document_count": collection.document_count or 0,
                }
            )

    # Check if we need to re-rank (different models or user requested)
    unique_models = {col.embedding_model for col in collections}
    needs_reranking = len(unique_models) > 1 or search_request.use_reranker

    reranking_time_ms = None
    final_results: list[CollectionSearchResult] = []

    if needs_reranking and all_results:
        # Re-rank merged results
        rerank_start = time.time()
        reranked_results = await rerank_merged_results(
            search_request.query,
            all_results,
            search_request.rerank_model,
            search_request.k,
        )
        reranking_time_ms = (time.time() - rerank_start) * 1000

        # Convert to response format
        for collection, result, reranked_score in reranked_results:
            final_results.append(
                CollectionSearchResult(
                    document_id=result.get("doc_id", ""),
                    chunk_id=result.get("chunk_id", ""),
                    score=reranked_score,
                    original_score=result.get("score", 0.0),
                    reranked_score=reranked_score,
                    text=result.get("content", ""),
                    metadata=result.get("metadata", {}),
                    file_name=result.get("path", "").split("/")[-1] if result.get("path") else "Unknown",
                    file_path=result.get("path", ""),
                    collection_id=str(collection.id),
                    collection_name=str(collection.name),
                    embedding_model=str(collection.embedding_model),
                )
            )
    else:
        # No re-ranking needed, just merge and sort by score
        all_results.sort(key=lambda x: x[1].get("score", 0.0), reverse=True)

        for collection, result in all_results[: search_request.k]:
            final_results.append(
                CollectionSearchResult(
                    document_id=result.get("doc_id", ""),
                    chunk_id=result.get("chunk_id", ""),
                    score=result.get("score", 0.0),
                    original_score=result.get("score", 0.0),
                    reranked_score=None,
                    text=result.get("content", ""),
                    metadata=result.get("metadata", {}),
                    file_name=result.get("path", "").split("/")[-1] if result.get("path") else "Unknown",
                    file_path=result.get("path", ""),
                    collection_id=str(collection.id),
                    collection_name=str(collection.name),
                    embedding_model=str(collection.embedding_model),
                )
            )

    total_time_ms = (time.time() - start_time) * 1000

    return CollectionSearchResponse(
        query=search_request.query,
        results=final_results,
        total_results=len(final_results),
        collections_searched=collections_info,
        search_type=search_request.search_type,
        reranking_used=needs_reranking,
        reranker_model=search_request.rerank_model if needs_reranking else None,
        search_time_ms=search_time_ms,
        reranking_time_ms=reranking_time_ms,
        total_time_ms=total_time_ms,
        partial_failure=bool(failed_collections),
        failed_collections=failed_collections if failed_collections else None,
    )


@router.post(
    "/single",
    response_model=CollectionSearchResponse,
    responses={
        403: {"model": ErrorResponse, "description": "Access denied to collection"},
        404: {"model": ErrorResponse, "description": "Collection not found"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
@limiter.limit("60/minute")
async def single_collection_search(
    request: Request,  # noqa: ARG001
    search_request: SingleCollectionSearchRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
    collection_repo: CollectionRepository = Depends(get_collection_repository),
) -> CollectionSearchResponse:
    """
    Search within a single collection (backward compatibility).

    This endpoint provides a simpler interface for searching within a single
    collection without the overhead of multi-collection aggregation.
    """
    # Convert to multi-collection request with single UUID
    multi_request = CollectionSearchRequest(
        collection_uuids=[search_request.collection_id],
        query=search_request.query,
        k=search_request.k,
        search_type=search_request.search_type,
        use_reranker=search_request.use_reranker,
        rerank_model=None,  # Single collection request doesn't specify rerank model
        score_threshold=search_request.score_threshold,
        metadata_filter=search_request.metadata_filter,
        include_content=search_request.include_content,
    )

    # Delegate to multi-collection search
    response = await multi_collection_search(request, multi_request, current_user, collection_repo)
    return cast(CollectionSearchResponse, response)
