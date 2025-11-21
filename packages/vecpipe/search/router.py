"""HTTP router for the vecpipe search API."""

from __future__ import annotations

import logging
import inspect
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query

from shared.config import settings
from shared.contracts.search import (
    BatchSearchRequest,
    BatchSearchResponse,
    HybridSearchResponse,
    SearchRequest,
    SearchResponse,
)
from vecpipe.search import service
from vecpipe.search.schemas import EmbedRequest, EmbedResponse, UpsertRequest, UpsertResponse
from vecpipe.search import state as search_state

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/model/status")
async def model_status() -> dict[str, Any]:
    return await service.model_status()


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

        if not cfg.USE_MOCK_EMBEDDINGS and search_state.embedding_service:
            model_info = search_state.embedding_service.get_model_info()
            health_info["embedding_service"] = {
                "current_model": search_state.embedding_service.current_model_name,
                "quantization": search_state.embedding_service.current_quantization,
                "device": search_state.embedding_service.device,
                "model_info": model_info,
            }

        return health_info
    except Exception as e:  # pragma: no cover - health fallback
        logger.error("Health check failed: %s", e)
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}") from e


@router.get("/health")
async def health() -> dict[str, Any]:
    return await service.health()


@router.get("/search", response_model=SearchResponse)
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
        hybrid_alpha=0.7,
        hybrid_mode="weighted",
        keyword_mode="any",
    )
    result = await service.perform_search(request)
    return SearchResponse(**result.model_dump())


@router.post("/search", response_model=SearchResponse)
async def search_post(request: SearchRequest = Body(...)) -> SearchResponse:
    return await service.perform_search(request)


@router.get("/hybrid_search", response_model=HybridSearchResponse)
async def hybrid_search(
    q: str = Query(..., description="Search query"),
    k: int = Query(service.DEFAULT_K, ge=1, le=100, description="Number of results to return"),
    collection: str | None = Query(None, description="Collection name"),
    mode: str = Query("filter", description="Hybrid search mode: 'filter' or 'weighted'"),
    keyword_mode: str = Query("any", description="Keyword matching: 'any' or 'all'"),
    score_threshold: float | None = Query(None, description="Minimum similarity score threshold"),
    model_name: str | None = Query(None, description="Override embedding model"),
    quantization: str | None = Query(None, description="Override quantization"),
) -> HybridSearchResponse:
    return await service.perform_hybrid_search(
        query=q,
        k=k,
        collection=collection,
        mode=mode,
        keyword_mode=keyword_mode,
        score_threshold=score_threshold,
        model_name=model_name,
        quantization=quantization,
    )


@router.post("/search/batch", response_model=BatchSearchResponse)
async def batch_search(request: BatchSearchRequest = Body(...)) -> BatchSearchResponse:
    return await service.perform_batch_search(request)


@router.get("/keyword_search", response_model=HybridSearchResponse)
async def keyword_search(
    q: str = Query(..., description="Keywords to search for"),
    k: int = Query(service.DEFAULT_K, ge=1, le=100, description="Number of results to return"),
    collection: str | None = Query(None, description="Collection name"),
    mode: str = Query("any", description="Keyword matching: 'any' or 'all'"),
) -> HybridSearchResponse:
    return await service.perform_keyword_search(query=q, k=k, collection=collection, mode=mode)


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
    return await service.list_models()


@router.post("/models/load")
async def load_model(
    model_name: str = Body(..., description="Model name to load"),
    quantization: str = Body("float32", description="Quantization type"),
) -> dict[str, Any]:
    return await service.load_model(model_name, quantization)


@router.get("/models/suggest")
async def suggest_models() -> dict[str, Any]:
    return await service.suggest_models()


@router.get("/embedding/info")
async def embedding_info() -> dict[str, Any]:
    return await service.embedding_info()


@router.post("/embed", response_model=EmbedResponse)
async def embed_texts(request: EmbedRequest = Body(...)) -> EmbedResponse:
    return await service.embed_texts(request)


@router.post("/upsert", response_model=UpsertResponse)
async def upsert_points(request: UpsertRequest = Body(...)) -> UpsertResponse:
    return await service.upsert_points(request)
