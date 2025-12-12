"""
Search API v2 endpoints with multi-collection support.

This module provides search endpoints that support searching across multiple
collections with result aggregation and re-ranking.
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from shared.database.exceptions import AccessDeniedError, EntityNotFoundError
from webui.api.schemas import ErrorResponse
from webui.api.v2.schemas import (
    CollectionSearchRequest,
    CollectionSearchResponse,
    CollectionSearchResult,
    SingleCollectionSearchRequest,
)
from webui.auth import get_current_user
from webui.rate_limiter import limiter
from webui.services.factory import get_search_service
from webui.services.search_service import SearchService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/search", tags=["search-v2"])


# Local helper functions have been moved to SearchService


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
    service: SearchService = Depends(get_search_service),
) -> CollectionSearchResponse:
    """
    Search across multiple collections with result aggregation and re-ranking.

    This endpoint allows searching across up to 10 collections simultaneously.
    Results are aggregated and re-ranked to provide globally relevant results.
    """
    try:
        result = await service.multi_collection_search(
            user_id=int(current_user["id"]),
            collection_uuids=search_request.collection_uuids,
            query=search_request.query,
            k=search_request.k,
            search_type=search_request.search_type,
            score_threshold=search_request.score_threshold,
            metadata_filter=search_request.metadata_filter,
            use_reranker=search_request.use_reranker,
            rerank_model=search_request.rerank_model,
            hybrid_alpha=search_request.hybrid_alpha,
            hybrid_mode=search_request.hybrid_mode,
            keyword_mode=search_request.keyword_mode,
        )

        # Convert service result to API response format
        final_results = []
        for res in result["results"]:
            final_results.append(
                CollectionSearchResult(
                    document_id=res.get("doc_id", ""),
                    chunk_id=res.get("chunk_id", ""),
                    score=res.get("reranked_score", res.get("score", 0.0)),
                    original_score=res.get("score", 0.0),
                    reranked_score=res.get("reranked_score"),
                    text=res.get("content", ""),
                    metadata=res.get("metadata", {}),
                    file_name=res.get("path", "").split("/")[-1] if res.get("path") else "Unknown",
                    file_path=res.get("path", ""),
                    collection_id=res.get("collection_id"),
                    collection_name=res.get("collection_name"),
                    embedding_model=res.get("embedding_model", ""),
                )
            )

        metadata = result["metadata"]
        return CollectionSearchResponse(
            query=search_request.query,
            results=final_results,
            total_results=metadata["total_results"],
            collections_searched=[
                {
                    "id": str(cd["collection_id"]),
                    "name": cd["collection_name"],
                    "result_count": cd["result_count"],
                }
                for cd in metadata["collection_details"]
                if "error" not in cd
            ],
            search_type=search_request.search_type,
            reranking_used=search_request.use_reranker,
            reranker_model=search_request.rerank_model,
            search_time_ms=metadata["processing_time"] * 1000,
            reranking_time_ms=None,  # Not available in new format
            total_time_ms=metadata["processing_time"] * 1000,
            partial_failure=bool(metadata.get("errors")),
            failed_collections=(
                [
                    {
                        "collection_id": str(cd["collection_id"]),
                        "collection_name": cd["collection_name"],
                        "error": cd["error"],
                    }
                    for cd in metadata["collection_details"]
                    if "error" in cd
                ]
                if metadata.get("errors")
                else None
            ),
        )

    except AccessDeniedError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Search failed") from e


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
    service: SearchService = Depends(get_search_service),
) -> CollectionSearchResponse:
    """
    Search within a single collection (backward compatibility).

    This endpoint provides a simpler interface for searching within a single
    collection without the overhead of multi-collection aggregation.
    """
    try:
        result = await service.single_collection_search(
            user_id=int(current_user["id"]),
            collection_uuid=search_request.collection_id,
            query=search_request.query,
            k=search_request.k,
            search_type=search_request.search_type,
            score_threshold=search_request.score_threshold,
            metadata_filter=search_request.metadata_filter,
            use_reranker=search_request.use_reranker,
            include_content=search_request.include_content,
        )

        # Convert service result to API response format
        final_results = []
        for res in result.get("results", []):
            final_results.append(
                CollectionSearchResult(
                    document_id=res.get("doc_id", ""),
                    chunk_id=res.get("chunk_id", ""),
                    score=res.get("score", 0.0),
                    original_score=res.get("score", 0.0),
                    reranked_score=res.get("reranked_score"),
                    text=res.get("content", ""),
                    metadata=res.get("metadata", {}),
                    file_name=res.get("path", "").split("/")[-1] if res.get("path") else "Unknown",
                    file_path=res.get("path", ""),
                    collection_id=search_request.collection_id,
                    collection_name="",  # Not available in single search response
                    embedding_model="",  # Not available in single search response
                )
            )

        return CollectionSearchResponse(
            query=search_request.query,
            results=final_results,
            total_results=len(final_results),
            collections_searched=[],  # Single collection search doesn't provide this info
            search_type=search_request.search_type,
            reranking_used=search_request.use_reranker,
            reranker_model=None,
            search_time_ms=result.get("processing_time_ms", 0),
            reranking_time_ms=None,
            total_time_ms=result.get("processing_time_ms", 0),
            partial_failure=False,
            failed_collections=None,
        )

    except EntityNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except AccessDeniedError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail="Search failed") from e
