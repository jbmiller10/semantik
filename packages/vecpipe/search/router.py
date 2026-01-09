"""HTTP router for the vecpipe search API."""

from __future__ import annotations

import inspect
import logging
import secrets
from typing import Any, cast

from fastapi import APIRouter, Body, Depends, Header, HTTPException, Query

from shared.config import settings
from shared.contracts.search import (
    BatchSearchRequest,
    BatchSearchResponse,
    SearchRequest,
    SearchResponse,
)
from vecpipe.search import service, state as search_state
from vecpipe.search.schemas import EmbedRequest, EmbedResponse, UpsertRequest, UpsertResponse

logger = logging.getLogger(__name__)

router = APIRouter()


def require_internal_api_key(x_internal_api_key: str | None = Header(default=None, alias="X-Internal-Api-Key")) -> None:
    """Verify the internal API key for protected endpoints."""
    expected_key = settings.INTERNAL_API_KEY
    if not expected_key:
        raise HTTPException(status_code=500, detail="Internal API key is not configured")
    if not x_internal_api_key or not secrets.compare_digest(x_internal_api_key, expected_key):
        raise HTTPException(status_code=401, detail="Invalid or missing internal API key")


@router.get("/model/status")
async def model_status() -> dict[str, Any]:
    return cast(dict[str, Any], await service.model_status())


@router.get("/")
async def root() -> dict[str, Any]:
    """Health check with collection info."""
    try:
        cfg = service._get_settings()

        if search_state.qdrant_client is None:
            raise HTTPException(status_code=503, detail="Qdrant client not initialized")
        response = await search_state.qdrant_client.get(f"/collections/{cfg.DEFAULT_COLLECTION}")
        maybe_coro = response.raise_for_status()
        if inspect.isawaitable(maybe_coro):
            await maybe_coro
        info = response.json()["result"]

        health_info: dict[str, Any] = {
            "status": "healthy",
            "collection": {
                "name": cfg.DEFAULT_COLLECTION,
                "points_count": info["points_count"],
                "vector_size": info["config"]["params"]["vectors"]["size"] if "config" in info else None,
            },
            "embedding_mode": "mock" if cfg.USE_MOCK_EMBEDDINGS else "real",
        }

        # Get embedding status from model manager
        if search_state.model_manager:
            mgr_status = search_state.model_manager.get_status()
            health_info["embedding_service"] = {
                "current_model": mgr_status.get("current_embedding_model"),
                "provider": mgr_status.get("embedding_provider"),
                "model_info": mgr_status.get("provider_info"),
                "is_mock_mode": mgr_status.get("is_mock_mode"),
            }

        return health_info
    except Exception as e:  # pragma: no cover - health fallback
        logger.error("Health check failed: %s", e)
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}") from e


@router.get("/health")
async def health() -> dict[str, Any]:
    return cast(dict[str, Any], await service.health())


@router.get("/search", response_model=SearchResponse, dependencies=[Depends(require_internal_api_key)])
async def search_get(
    q: str = Query(..., description="Search query"),
    k: int = Query(service.DEFAULT_K, ge=1, le=100, description="Number of results to return"),
    collection: str | None = Query(None, description="Collection name"),
    search_type: str = Query("semantic", description="Type of search: semantic, question, code, hybrid"),
    model_name: str | None = Query(None, description="Override embedding model"),
    quantization: str | None = Query(None, description="Override quantization"),
) -> SearchResponse:
    request = SearchRequest(
        query=q,
        top_k=k,
        search_type=search_type,
        model_name=model_name,
        quantization=quantization,
        collection=collection,
        filters=None,
        include_content=False,
        operation_uuid=None,
        use_reranker=False,
        rerank_model=None,
        rerank_quantization=None,
        score_threshold=0.0,
    )
    result = await service.perform_search(request)
    return SearchResponse(**result.model_dump())


@router.post("/search", response_model=SearchResponse, dependencies=[Depends(require_internal_api_key)])
async def search_post(request: SearchRequest = Body(...)) -> SearchResponse:
    return await service.perform_search(request)


@router.post("/search/batch", response_model=BatchSearchResponse, dependencies=[Depends(require_internal_api_key)])
async def batch_search(request: BatchSearchRequest = Body(...)) -> BatchSearchResponse:
    return await service.perform_batch_search(request)


@router.get("/collection/info")
async def collection_info() -> dict[str, Any]:
    try:
        cfg = service._get_settings()
        client = service._get_qdrant_client()
        if client is None:
            raise HTTPException(status_code=503, detail="Qdrant client not initialized")
        response = await client.get(f"/collections/{cfg.DEFAULT_COLLECTION}")
        maybe_coro = response.raise_for_status()
        if inspect.isawaitable(maybe_coro):
            await maybe_coro
        result = response.json()
        if inspect.isawaitable(result):
            result = await result
        return dict(result["result"])
    except Exception as e:  # pragma: no cover - fallback
        logger.error("Failed to get collection info: %s", e)
        raise HTTPException(status_code=502, detail="Failed to get collection info") from e


@router.get("/models")
async def list_models() -> dict[str, Any]:
    return cast(dict[str, Any], await service.list_models())


@router.post("/models/load", dependencies=[Depends(require_internal_api_key)])
async def load_model(
    model_name: str = Body(..., description="Model name to load"),
    quantization: str = Body("float32", description="Quantization type"),
) -> dict[str, Any]:
    return cast(dict[str, Any], await service.load_model(model_name, quantization))


@router.get("/models/suggest")
async def suggest_models() -> dict[str, Any]:
    return cast(dict[str, Any], await service.suggest_models())


@router.get("/embedding/info")
async def embedding_info() -> dict[str, Any]:
    return cast(dict[str, Any], await service.embedding_info())


@router.post("/embed", response_model=EmbedResponse, dependencies=[Depends(require_internal_api_key)])
async def embed_texts(request: EmbedRequest = Body(...)) -> EmbedResponse:
    return await service.embed_texts(request)


@router.post("/upsert", response_model=UpsertResponse, dependencies=[Depends(require_internal_api_key)])
async def upsert_points(request: UpsertRequest = Body(...)) -> UpsertResponse:
    return await service.upsert_points(request)
