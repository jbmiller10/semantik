"""HTTP router for the vecpipe search API."""

from __future__ import annotations

import logging
from typing import Any, cast

from fastapi import APIRouter, Body, Depends, HTTPException, Query

from shared.contracts.search import BatchSearchRequest, BatchSearchResponse, SearchMode, SearchRequest, SearchResponse
from vecpipe.search import service
from vecpipe.search.auth import require_internal_api_key
from vecpipe.search.deps import get_runtime
from vecpipe.search.errors import maybe_raise_for_status, response_json
from vecpipe.search.runtime import VecpipeRuntime
from vecpipe.search.schemas import EmbedRequest, EmbedResponse, UpsertRequest, UpsertResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/model/status")
async def model_status(runtime: VecpipeRuntime = Depends(get_runtime)) -> dict[str, Any]:
    return cast(dict[str, Any], await service.model_status(runtime=runtime))


@router.get("/")
async def root(runtime: VecpipeRuntime = Depends(get_runtime)) -> dict[str, Any]:
    """Health check with collection info."""
    try:
        cfg = service._get_settings()

        response = await runtime.qdrant_http.get(f"/collections/{cfg.DEFAULT_COLLECTION}")
        await maybe_raise_for_status(response)
        info = (await response_json(response))["result"]

        health_info: dict[str, Any] = {
            "status": "healthy",
            "collection": {
                "name": cfg.DEFAULT_COLLECTION,
                "points_count": info["points_count"],
                "vector_size": info["config"]["params"]["vectors"]["size"] if "config" in info else None,
            },
            "embedding_mode": "mock" if cfg.USE_MOCK_EMBEDDINGS else "real",
        }

        mgr_status = runtime.model_manager.get_status()
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
async def health(runtime: VecpipeRuntime = Depends(get_runtime)) -> dict[str, Any]:
    return cast(dict[str, Any], await service.health(runtime=runtime))


@router.get("/search", response_model=SearchResponse, dependencies=[Depends(require_internal_api_key)])
async def search_get(
    runtime: VecpipeRuntime = Depends(get_runtime),
    q: str = Query(..., description="Search query"),
    k: int = Query(service.DEFAULT_K, ge=1, le=100, description="Number of results to return"),
    collection: str | None = Query(None, description="Collection name"),
    search_type: str = Query("semantic", description="Type of search: semantic, question, code, hybrid"),
    search_mode: SearchMode | None = Query(
        None,
        description="Search mode: 'dense' (vector only), 'sparse' (BM25/SPLADE only), 'hybrid' (dense + sparse with RRF)",
    ),
    rrf_k: int = Query(60, ge=1, le=1000, description="RRF constant for hybrid search score fusion"),
    model_name: str | None = Query(None, description="Override embedding model"),
    quantization: str | None = Query(None, description="Override quantization"),
) -> SearchResponse:
    # Backward-compat: legacy callers may pass search_type=hybrid without specifying search_mode.
    if search_mode is not None:
        effective_search_mode: SearchMode = search_mode
    elif search_type == "hybrid":
        effective_search_mode = "hybrid"
    else:
        effective_search_mode = "dense"
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
        search_mode=effective_search_mode,
        rrf_k=rrf_k,
    )
    result = await service.perform_search(request, runtime=runtime)
    return SearchResponse(**result.model_dump())


@router.post("/search", response_model=SearchResponse, dependencies=[Depends(require_internal_api_key)])
async def search_post(
    request: SearchRequest = Body(...),
    runtime: VecpipeRuntime = Depends(get_runtime),
) -> SearchResponse:
    # Backward-compat: legacy callers may send search_type=hybrid without specifying search_mode.
    # Preserve explicit search_mode when provided.
    if request.search_type == "hybrid" and "search_mode" not in request.model_fields_set:
        request = request.model_copy(update={"search_mode": "hybrid"})
    return await service.perform_search(request, runtime=runtime)


@router.post("/search/batch", response_model=BatchSearchResponse, dependencies=[Depends(require_internal_api_key)])
async def batch_search(
    request: BatchSearchRequest = Body(...),
    runtime: VecpipeRuntime = Depends(get_runtime),
) -> BatchSearchResponse:
    return await service.perform_batch_search(request, runtime=runtime)


@router.get("/collection/info")
async def collection_info(runtime: VecpipeRuntime = Depends(get_runtime)) -> dict[str, Any]:
    try:
        cfg = service._get_settings()
        response = await runtime.qdrant_http.get(f"/collections/{cfg.DEFAULT_COLLECTION}")
        await maybe_raise_for_status(response)
        result = await response_json(response)
        return dict(result["result"])
    except Exception as e:  # pragma: no cover - fallback
        logger.error("Failed to get collection info: %s", e)
        raise HTTPException(status_code=502, detail="Failed to get collection info") from e


@router.get("/models")
async def list_models(runtime: VecpipeRuntime = Depends(get_runtime)) -> dict[str, Any]:
    return cast(dict[str, Any], await service.list_models(runtime=runtime))


@router.post("/models/load", dependencies=[Depends(require_internal_api_key)])
async def load_model(
    runtime: VecpipeRuntime = Depends(get_runtime),
    model_name: str = Body(..., description="Model name to load"),
    quantization: str = Body("float32", description="Quantization type"),
) -> dict[str, Any]:
    return cast(dict[str, Any], await service.load_model(model_name, quantization, runtime=runtime))


@router.get("/models/suggest")
async def suggest_models(runtime: VecpipeRuntime = Depends(get_runtime)) -> dict[str, Any]:
    return cast(dict[str, Any], await service.suggest_models(runtime=runtime))


@router.get("/embedding/info")
async def embedding_info(runtime: VecpipeRuntime = Depends(get_runtime)) -> dict[str, Any]:
    return cast(dict[str, Any], await service.embedding_info(runtime=runtime))


@router.post("/embed", response_model=EmbedResponse, dependencies=[Depends(require_internal_api_key)])
async def embed_texts(request: EmbedRequest = Body(...), runtime: VecpipeRuntime = Depends(get_runtime)) -> EmbedResponse:
    return await service.embed_texts(request, runtime=runtime)


@router.post("/upsert", response_model=UpsertResponse, dependencies=[Depends(require_internal_api_key)])
async def upsert_points(
    request: UpsertRequest = Body(...),
    runtime: VecpipeRuntime = Depends(get_runtime),
) -> UpsertResponse:
    return await service.upsert_points(request, runtime=runtime)
