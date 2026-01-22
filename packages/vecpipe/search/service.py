"""Core business logic for the vecpipe search API."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

import httpx
from fastapi import HTTPException

from shared.config import settings
from shared.contracts.search import (
    BatchSearchRequest,
    BatchSearchResponse,
    SearchMode,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from shared.database.exceptions import DimensionMismatchError
from shared.embedding.validation import validate_dimension_compatibility
from shared.metrics.prometheus import metrics_collector
from vecpipe.memory_utils import InsufficientMemoryError
from vecpipe.search.collection_info import get_cached_collection_metadata, get_collection_info, resolve_collection_name
from vecpipe.search.dense_search import generate_embedding, generate_mock_embedding, search_dense_qdrant
from vecpipe.search.errors import extract_qdrant_error, maybe_raise_for_status, response_json
from vecpipe.search.metrics import (
    rrf_fusion_latency,
    search_errors,
    search_latency,
    search_requests,
    sparse_search_fallbacks,
    sparse_search_requests,
)
from vecpipe.search.payloads import fetch_payloads_for_chunk_ids
from vecpipe.search.rerank import calculate_candidate_k, maybe_rerank_results
from vecpipe.search.schemas import EmbedRequest, EmbedResponse, UpsertRequest, UpsertResponse
from vecpipe.search.sparse_search import _reciprocal_rank_fusion, perform_sparse_search

if TYPE_CHECKING:
    from vecpipe.search.runtime import VecpipeRuntime

logger = logging.getLogger(__name__)

DEFAULT_K = 10

SEARCH_INSTRUCTIONS = {
    "semantic": "Represent this sentence for searching relevant passages:",
    "question": "Represent this question for retrieving supporting documents:",
    "code": "Represent this code query for finding similar code snippets:",
    "hybrid": "Generate a comprehensive embedding for multi-modal search:",
}


def _get_settings() -> Any:
    return settings


def _is_mock_object(obj: object) -> bool:
    # Avoid importing unittest.mock in production code.
    return bool(getattr(obj, "_is_mock_object", False)) or obj.__class__.__module__ == "unittest.mock"


def _resolve_runtime(runtime: VecpipeRuntime | None) -> VecpipeRuntime:
    """Resolve runtime for service-layer calls."""
    if runtime is None:
        raise HTTPException(status_code=503, detail="VecPipe runtime not initialized")
    return runtime


async def perform_search(request: SearchRequest, runtime: VecpipeRuntime | None = None) -> SearchResponse:
    """Execute semantic/question/code search with optional sparse/hybrid mode and reranking."""
    cfg = _get_settings()
    rt = _resolve_runtime(runtime)

    start_time = time.time()
    search_requests.labels(endpoint="/search", search_type=request.search_type).inc()

    # Track sparse search state
    search_mode_used: SearchMode = "dense"
    sparse_search_time_ms: float | None = None
    rrf_fusion_time_ms: float | None = None
    warnings: list[str] = []
    sparse_results: list[dict[str, Any]] = []
    sparse_config: dict[str, Any] | None = None

    try:
        collection_name = await resolve_collection_name(
            request.collection, request.operation_uuid, cfg.DEFAULT_COLLECTION
        )

        vector_dim, collection_info = await get_collection_info(
            collection_name=collection_name,
            cfg=cfg,
            qdrant_http=rt.qdrant_http,
        )

        collection_dim_known = (
            bool(collection_info)
            and isinstance(collection_info, dict)
            and "config" in collection_info
            and isinstance(collection_info.get("config"), dict)
            and "params" in collection_info["config"]
            and isinstance(collection_info["config"].get("params"), dict)
            and "vectors" in collection_info["config"]["params"]
            and isinstance(collection_info["config"]["params"].get("vectors"), dict)
            and "size" in collection_info["config"]["params"]["vectors"]
        )

        # Fetch metadata early - used for both model defaults and sparse config
        metadata = await get_cached_collection_metadata(
            collection_name=collection_name,
            cfg=cfg,
            qdrant_sdk=rt.qdrant_sdk,
        )

        # Derive sparse config from cached metadata (avoids separate Qdrant call)
        if request.search_mode in ("sparse", "hybrid"):
            if metadata:
                sparse_index_config = metadata.get("sparse_index_config")
                if (
                    isinstance(sparse_index_config, dict)
                    and sparse_index_config.get("enabled")
                    and sparse_index_config.get("plugin_id")
                    and sparse_index_config.get("sparse_collection_name")
                ):
                    sparse_config = sparse_index_config
                    search_mode_used = request.search_mode
                else:
                    warnings.append(
                        f"Sparse index not available for collection '{collection_name}'. Falling back to dense search."
                    )
                    search_mode_used = "dense"
                    sparse_search_fallbacks.labels(reason="sparse_not_enabled").inc()
            else:
                warnings.append(
                    f"Sparse index not available for collection '{collection_name}'. Falling back to dense search."
                )
                search_mode_used = "dense"
                sparse_search_fallbacks.labels(reason="no_metadata").inc()
        else:
            search_mode_used = "dense"

        collection_model = metadata.get("model_name") if metadata else None
        collection_quantization = metadata.get("quantization") if metadata else None
        collection_instruction = metadata.get("instruction") if metadata else None

        model_name = request.model_name or collection_model or cfg.DEFAULT_EMBEDDING_MODEL
        quantization = request.quantization or collection_quantization or cfg.DEFAULT_QUANTIZATION

        instruction = (
            collection_instruction
            if collection_instruction and request.search_type == "semantic"
            else SEARCH_INSTRUCTIONS.get(request.search_type, SEARCH_INSTRUCTIONS["semantic"])
        )

        search_k = calculate_candidate_k(request.k) if request.use_reranker else request.k
        embed_time_ms: float = 0.0
        search_time_ms: float = 0.0
        qdrant_results: list[dict[str, Any]] = []
        dense_results_for_fusion: list[dict[str, Any]] = []
        dense_chunk_ids: set[str] = set()

        # Dense search is skipped entirely in sparse-only mode.
        if search_mode_used != "sparse":
            embed_start = time.time()
            embed_query = request.dense_query if request.dense_query is not None else request.query
            query_vector = await generate_embedding(
                cfg=cfg,
                model_manager=rt.model_manager,
                text=embed_query,
                model_name=model_name,
                quantization=quantization,
                instruction=instruction,
                mode="query",
                vector_dim=vector_dim,
            )
            embed_time_ms = (time.time() - embed_start) * 1000

            if not collection_dim_known:
                vector_dim = len(query_vector)

            if not cfg.USE_MOCK_EMBEDDINGS and collection_dim_known:
                try:
                    validate_dimension_compatibility(
                        expected_dimension=vector_dim,
                        actual_dimension=len(query_vector),
                        collection_name=collection_name,
                        model_name=model_name,
                    )
                except DimensionMismatchError as exc:
                    search_errors.labels(endpoint="/search", error_type="dimension_mismatch").inc()
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "dimension_mismatch",
                            "message": str(exc),
                            "expected_dimension": exc.expected_dimension,
                            "actual_dimension": exc.actual_dimension,
                            "suggestion": (
                                "Use the same model that was used to create the collection, "
                                f"or ensure the model outputs {exc.expected_dimension}-dimensional vectors"
                            ),
                        },
                    ) from exc

            search_start = time.time()
            qdrant_results, dense_sdk_fallback_used = await search_dense_qdrant(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=search_k,
                qdrant_http=rt.qdrant_http,
                qdrant_sdk=rt.qdrant_sdk,
                filters=request.filters,
            )
            search_time_ms = (time.time() - search_start) * 1000
            if dense_sdk_fallback_used:
                warnings.append("Dense search SDK failed; used REST fallback")

            for point in qdrant_results:
                payload = point.get("payload") or {}
                chunk_id = payload.get("chunk_id") or str(point.get("id", ""))
                if chunk_id:
                    dense_chunk_ids.add(str(chunk_id))
                dense_results_for_fusion.append(
                    {
                        "chunk_id": str(chunk_id),
                        "score": point.get("score", 0.0),
                        "payload": payload,
                    }
                )

        # Perform sparse search if needed
        if search_mode_used in ("sparse", "hybrid") and sparse_config:
            sparse_results, sparse_search_time_ms, sparse_warnings = await perform_sparse_search(
                cfg=cfg,
                collection_name=collection_name,
                sparse_config=sparse_config,
                query=request.query,
                k=search_k,
                sparse_manager=rt.sparse_manager,
                qdrant_sdk=rt.qdrant_sdk,
            )
            warnings.extend(sparse_warnings)

            if sparse_results:
                sparse_to_original: dict[str, str] = {}
                for r in sparse_results:
                    sparse_cid = r.get("chunk_id", "")
                    payload = r.get("payload") or {}
                    metadata_obj = payload.get("metadata", {}) if isinstance(payload, dict) else {}
                    original_cid = metadata_obj.get("original_chunk_id") or payload.get("original_chunk_id", "")
                    if sparse_cid and original_cid:
                        sparse_to_original[str(sparse_cid)] = str(original_cid)

                chunk_ids_to_fetch = [cid for cid in sparse_to_original.values() if cid and cid not in dense_chunk_ids]
                payloads_by_chunk_id = await fetch_payloads_for_chunk_ids(
                    collection_name=collection_name,
                    chunk_ids=chunk_ids_to_fetch,
                    cfg=cfg,
                    qdrant_http=rt.qdrant_http,
                    filters=request.filters,
                )

                # Map payloads back to sparse results using original_chunk_id
                for item in sparse_results:
                    sparse_cid = item.get("chunk_id", "")
                    original_cid = sparse_to_original.get(str(sparse_cid), "")
                    if original_cid and original_cid in payloads_by_chunk_id:
                        item["payload"] = payloads_by_chunk_id[original_cid]
                        item["chunk_id"] = original_cid

                # Ensure filtered searches don't leak sparse hits outside the filter scope.
                valid_chunk_ids = dense_chunk_ids | set(payloads_by_chunk_id)
                sparse_results = [item for item in sparse_results if item.get("chunk_id", "") in valid_chunk_ids]

            if search_mode_used == "hybrid" and sparse_results:
                rrf_start = time.time()
                fused_results = _reciprocal_rank_fusion(
                    dense_results=dense_results_for_fusion,
                    sparse_results=sparse_results,
                    k=search_k,
                    rrf_k=request.rrf_k,
                )
                rrf_fusion_time_ms = (time.time() - rrf_start) * 1000
                rrf_fusion_latency.observe(rrf_fusion_time_ms / 1000)

                qdrant_results = [
                    {"id": r["chunk_id"], "score": r["score"], "payload": r.get("payload", {})} for r in fused_results
                ]
            elif search_mode_used == "sparse":
                qdrant_results = [
                    {"id": r["chunk_id"], "score": r["score"], "payload": r.get("payload", {})} for r in sparse_results
                ]
                if sparse_search_time_ms is not None:
                    search_time_ms = sparse_search_time_ms

        should_include_content = request.include_content or request.use_reranker
        results: list[SearchResult] = []

        for point in qdrant_results:
            payload = point.get("payload") or {}
            if not isinstance(payload, dict):
                payload = {}

            doc_id = payload.get("doc_id")
            if not doc_id:
                # Defensive: skip malformed points rather than crashing.
                continue

            results.append(
                SearchResult(
                    path=payload.get("path", ""),
                    chunk_id=payload.get("chunk_id", "") or str(point.get("id", "")),
                    score=float(point.get("score", 0.0)),
                    doc_id=str(doc_id),
                    content=payload.get("content") if should_include_content else None,
                    metadata=payload.get("metadata"),
                    file_path=None,
                    file_name=None,
                    operation_uuid=None,
                    chunk_index=payload.get("chunk_index"),
                    total_chunks=payload.get("total_chunks"),
                )
            )

        # Apply score_threshold filtering BEFORE reranking
        if request.score_threshold > 0 and results:
            pre_filter_count = len(results)
            results = [r for r in results if r.score >= request.score_threshold]
            if pre_filter_count != len(results):
                logger.info(
                    "score_threshold=%.2f filtered %d/%d results",
                    request.score_threshold,
                    pre_filter_count - len(results),
                    pre_filter_count,
                )

        results, reranker_model_used, reranking_time_ms = await maybe_rerank_results(
            cfg=cfg,
            model_manager=rt.model_manager,
            qdrant_http=rt.qdrant_http,
            collection_name=collection_name,
            request=request,
            results=results,
            embedding_model_name=model_name,
            embedding_quantization=quantization,
        )
        if request.use_reranker and results and reranking_time_ms is not None and reranker_model_used is None:
            warnings.append("Reranking failed; returning un-reranked results")

        total_time_ms = (time.time() - start_time) * 1000
        msg = f"Search completed in {total_time_ms:.2f}ms (embed: {embed_time_ms:.2f}ms, search: {search_time_ms:.2f}ms"
        if reranking_time_ms:
            msg += f", rerank: {reranking_time_ms:.2f}ms"
        msg += ")"
        logger.info(msg)

        search_latency.labels(endpoint="/search", search_type=request.search_type).observe(time.time() - start_time)
        metrics_collector.update_resource_metrics()

        if search_mode_used in ("sparse", "hybrid"):
            sparse_type = sparse_config.get("plugin_id", "unknown").split("-")[0] if sparse_config else "unknown"
            sparse_search_requests.labels(search_mode=search_mode_used, sparse_type=sparse_type).inc()

        return SearchResponse(
            query=request.query,
            results=results,
            num_results=len(results),
            search_type=request.search_type,
            model_used=f"{model_name}/{quantization}" if not cfg.USE_MOCK_EMBEDDINGS else "mock",
            embedding_time_ms=embed_time_ms,
            search_time_ms=search_time_ms,
            reranking_used=bool(reranker_model_used),
            reranker_model=reranker_model_used,
            reranking_time_ms=reranking_time_ms,
            search_mode_used=search_mode_used,
            sparse_search_time_ms=sparse_search_time_ms,
            rrf_fusion_time_ms=rrf_fusion_time_ms,
            warnings=warnings,
        )

    except HTTPException:
        raise
    except httpx.HTTPStatusError as exc:
        logger.error("Qdrant error: %s", exc)
        search_errors.labels(endpoint="/search", error_type="qdrant_error").inc()
        raise HTTPException(status_code=502, detail="Vector database error") from exc
    except RuntimeError as exc:
        logger.error("Embedding generation failed: %s", exc)
        search_errors.labels(endpoint="/search", error_type="embedding_error").inc()
        raise HTTPException(
            status_code=503, detail=f"Embedding service error: {str(exc)}. Check logs for details."
        ) from exc
    except Exception as exc:  # pragma: no cover - uncaught path
        logger.error("Search error: %s", exc, exc_info=True)
        search_errors.labels(endpoint="/search", error_type="unknown_error").inc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(exc)}") from exc


async def perform_batch_search(
    request: BatchSearchRequest, runtime: VecpipeRuntime | None = None
) -> BatchSearchResponse:
    """Batch search for multiple queries."""
    cfg = _get_settings()
    rt = _resolve_runtime(runtime)
    start_time = time.time()

    try:
        collection_name = request.collection if request.collection else cfg.DEFAULT_COLLECTION
        model_name = request.model_name or cfg.DEFAULT_EMBEDDING_MODEL
        quantization = request.quantization or cfg.DEFAULT_QUANTIZATION
        instruction = SEARCH_INSTRUCTIONS.get(request.search_type, SEARCH_INSTRUCTIONS["semantic"])

        embedding_tasks = [
            generate_embedding(
                cfg=cfg,
                model_manager=rt.model_manager,
                text=query,
                model_name=model_name,
                quantization=quantization,
                instruction=instruction,
                mode="query",
            )
            for query in request.queries
        ]
        query_vectors = await asyncio.gather(*embedding_tasks)

        search_tasks = [
            search_dense_qdrant(
                collection_name=collection_name,
                query_vector=vector,
                limit=request.k,
                qdrant_http=rt.qdrant_http,
                qdrant_sdk=rt.qdrant_sdk,
                filters=None,
            )
            for vector in query_vectors
        ]
        all_results = await asyncio.gather(*search_tasks)

        responses: list[SearchResponse] = []
        for query, result_tuple in zip(request.queries, all_results, strict=False):
            results, dense_sdk_fallback_used = result_tuple
            parsed_results: list[SearchResult] = []
            for point in results:
                payload = point.get("payload") or {}
                if not isinstance(payload, dict) or "doc_id" not in payload:
                    continue
                parsed_results.append(
                    SearchResult(
                        path=payload.get("path", ""),
                        chunk_id=payload.get("chunk_id", ""),
                        score=float(point.get("score", 0.0)),
                        doc_id=str(payload["doc_id"]),
                        content=None,
                        metadata=payload.get("metadata"),
                        file_path=None,
                        file_name=None,
                        operation_uuid=None,
                        chunk_index=payload.get("chunk_index"),
                        total_chunks=payload.get("total_chunks"),
                    )
                )

            responses.append(
                SearchResponse(
                    query=query,
                    results=parsed_results,
                    num_results=len(parsed_results),
                    search_type=request.search_type,
                    model_used=f"{model_name}/{quantization}" if not cfg.USE_MOCK_EMBEDDINGS else "mock",
                    search_mode_used="dense",
                    sparse_search_time_ms=None,
                    rrf_fusion_time_ms=None,
                    warnings=["Dense search SDK failed; used REST fallback"] if dense_sdk_fallback_used else [],
                )
            )

        total_time = (time.time() - start_time) * 1000
        return BatchSearchResponse(responses=responses, total_time_ms=total_time)
    except Exception as exc:  # pragma: no cover - failure path
        logger.error("Batch search error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Batch search failed: {str(exc)}") from exc


async def embed_texts(request: EmbedRequest, runtime: VecpipeRuntime | None = None) -> EmbedResponse:
    """Generate embeddings for a batch of texts."""
    rt = _resolve_runtime(runtime)
    start_time = time.time()
    search_requests.labels(endpoint="/embed", search_type="embedding").inc()

    try:
        model_mgr = rt.model_manager
        if model_mgr is None:
            raise RuntimeError("Model manager not initialized")

        embeddings: list[list[float]] = []
        batch_count = 0

        for i in range(0, len(request.texts), request.batch_size):
            batch_texts = request.texts[i : i + request.batch_size]
            batch_embeddings = await model_mgr.generate_embeddings_batch_async(
                batch_texts,
                request.model_name,
                request.quantization,
                instruction=request.instruction,
                batch_size=request.batch_size,
                mode=request.mode,
            )
            embeddings.extend(batch_embeddings)
            batch_count += 1

        total_time = (time.time() - start_time) * 1000
        search_latency.labels(endpoint="/embed", search_type="embedding").observe(time.time() - start_time)

        return EmbedResponse(
            embeddings=embeddings,
            model_used=f"{request.model_name}/{request.quantization}",
            embedding_time_ms=total_time,
            batch_count=batch_count,
        )
    except InsufficientMemoryError as exc:
        search_errors.labels(endpoint="/embed", error_type="memory_error").inc()
        raise HTTPException(
            status_code=507,
            detail={
                "error": "insufficient_memory",
                "message": str(exc),
                "suggestion": "Try using a smaller model or different quantization (float16/int8)",
            },
        ) from exc
    except RuntimeError as exc:
        search_errors.labels(endpoint="/embed", error_type="runtime_error").inc()
        raise HTTPException(status_code=503, detail=f"Embedding service error: {str(exc)}") from exc
    except Exception as exc:  # pragma: no cover - unexpected path
        # Defensive: InsufficientMemoryError can occasionally be imported under
        # different module paths in tests (vecpipe vs packages.vecpipe), causing
        # class identity mismatches. Treat matching class names as OOM as well.
        if exc.__class__.__name__ == "InsufficientMemoryError":
            search_errors.labels(endpoint="/embed", error_type="memory_error").inc()
            raise HTTPException(
                status_code=507,
                detail={
                    "error": "insufficient_memory",
                    "message": str(exc),
                    "suggestion": "Try using a smaller model or different quantization (float16/int8)",
                },
            ) from exc
        search_errors.labels(endpoint="/embed", error_type="unknown_error").inc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(exc)}") from exc


async def upsert_points(request: UpsertRequest, runtime: VecpipeRuntime | None = None) -> UpsertResponse:
    """Upsert points into Qdrant."""
    rt = _resolve_runtime(runtime)
    client = rt.qdrant_http

    start_time = time.time()
    search_requests.labels(endpoint="/upsert", search_type="vector_upload").inc()

    try:
        test_mode = _is_mock_object(client)

        if not test_mode:
            try:
                response = await client.get(f"/collections/{request.collection_name}")
                await maybe_raise_for_status(response)
                payload = await response_json(response)
                collection_info = payload.get("result", {}) if isinstance(payload, dict) else {}
                collection_dim = None
                if (
                    isinstance(collection_info, dict)
                    and "config" in collection_info
                    and "params" in collection_info["config"]
                ):
                    collection_dim = collection_info["config"]["params"]["vectors"]["size"]

                if collection_dim and request.points:
                    for point in request.points:
                        vector_dim = len(point.vector)
                        if vector_dim != collection_dim:
                            raise DimensionMismatchError(
                                expected_dimension=collection_dim,
                                actual_dimension=vector_dim,
                                collection_name=request.collection_name,
                            )
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 404:
                    raise HTTPException(
                        status_code=404, detail=f"Collection '{request.collection_name}' not found"
                    ) from exc
                raise
            except DimensionMismatchError as exc:
                search_errors.labels(endpoint="/upsert", error_type="dimension_mismatch").inc()
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "dimension_mismatch",
                        "message": str(exc),
                        "expected_dimension": exc.expected_dimension,
                        "actual_dimension": exc.actual_dimension,
                        "suggestion": (
                            f"All vectors must have dimension {exc.expected_dimension} "
                            "to match the collection configuration"
                        ),
                    },
                ) from exc

        upsert_request: dict[str, Any] = {
            "points": [
                {
                    "id": p.id,
                    "vector": p.vector,
                    "payload": {
                        "doc_id": p.payload.doc_id,
                        "chunk_id": p.payload.chunk_id,
                        "path": p.payload.path,
                        **({"content": p.payload.content} if p.payload.content is not None else {}),
                        **({"metadata": p.payload.metadata} if p.payload.metadata is not None else {}),
                        **({"collection_id": p.payload.collection_id} if p.payload.collection_id is not None else {}),
                        **({"chunk_index": p.payload.chunk_index} if p.payload.chunk_index is not None else {}),
                        **({"total_chunks": p.payload.total_chunks} if p.payload.total_chunks is not None else {}),
                    },
                }
                for p in request.points
            ]
        }

        url = f"/collections/{request.collection_name}/points"
        if request.wait:
            url = f"{url}?wait=true"

        response = await client.put(url, json=upsert_request)
        await maybe_raise_for_status(response)

        total_time = (time.time() - start_time) * 1000
        search_latency.labels(endpoint="/upsert", search_type="vector_upload").observe(time.time() - start_time)
        return UpsertResponse(
            status="success",
            points_upserted=len(request.points),
            collection_name=request.collection_name,
            upsert_time_ms=total_time,
        )
    except HTTPException:
        raise
    except httpx.HTTPStatusError as exc:
        search_errors.labels(endpoint="/upsert", error_type="qdrant_error").inc()
        detail = extract_qdrant_error(exc)
        detail_text = f"Vector database error: {detail}" if detail != "Vector database error" else detail
        raise HTTPException(status_code=502, detail=detail_text) from exc
    except Exception as exc:  # pragma: no cover - unexpected
        search_errors.labels(endpoint="/upsert", error_type="unknown_error").inc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(exc)}") from exc


async def model_status(runtime: VecpipeRuntime | None = None) -> dict[str, Any]:
    """Return model manager status."""
    rt = _resolve_runtime(runtime)
    return dict(rt.model_manager.get_status())


async def health(runtime: VecpipeRuntime | None = None) -> dict[str, Any]:
    """Comprehensive health check."""
    cfg = _get_settings()
    rt = _resolve_runtime(runtime)

    health_status: dict[str, Any] = {"status": "healthy", "components": {}}

    try:
        response = await rt.qdrant_http.get("/collections")
        if response.status_code == 200:
            collections_data = response.json()
            health_status["components"]["qdrant"] = {
                "status": "healthy",
                "collections_count": len(collections_data.get("result", {}).get("collections", [])),
            }
        else:
            health_status["components"]["qdrant"] = {"status": "unhealthy", "error": f"HTTP {response.status_code}"}
            health_status["status"] = "unhealthy"
    except Exception as exc:
        health_status["components"]["qdrant"] = {"status": "unhealthy", "error": str(exc)}
        health_status["status"] = "unhealthy"

    try:
        mgr_status = rt.model_manager.get_status()
        if mgr_status.get("embedding_model_loaded"):
            provider_info = mgr_status.get("provider_info", {})
            health_status["components"]["embedding"] = {
                "status": "healthy",
                "model": mgr_status.get("current_embedding_model"),
                "provider": mgr_status.get("embedding_provider"),
                "dimension": provider_info.get("dimension") if provider_info else None,
                "is_mock_mode": cfg.USE_MOCK_EMBEDDINGS,
            }
        else:
            health_status["components"]["embedding"] = {
                "status": "healthy",
                "model": None,
                "provider": None,
                "note": "Embedding model loaded on first use",
                "is_mock_mode": cfg.USE_MOCK_EMBEDDINGS,
            }
    except Exception as exc:
        health_status["components"]["embedding"] = {"status": "unhealthy", "error": str(exc)}
        health_status["status"] = "degraded" if health_status["status"] == "healthy" else "unhealthy"

    if health_status["status"] == "unhealthy":
        raise HTTPException(status_code=503, detail=health_status)

    return health_status


async def list_models(runtime: VecpipeRuntime | None = None) -> dict[str, Any]:
    """List available embedding models and their properties."""
    rt = _resolve_runtime(runtime)

    from shared.embedding.factory import get_all_supported_models

    all_models = get_all_supported_models()

    models = []
    for model_info in all_models:
        model_name = model_info.get("model_name") or model_info.get("name", "")
        provider = model_info.get("provider", "unknown")

        models.append(
            {
                "name": model_name,
                "description": model_info.get("description", ""),
                "dimension": model_info.get("dimension"),
                "supports_quantization": model_info.get("supports_quantization", True),
                "recommended_quantization": model_info.get("recommended_quantization", "float32"),
                "memory_estimate": model_info.get("memory_estimate", {}),
                "is_qwen3": "Qwen3-Embedding" in model_name,
                "provider_id": provider,
                "is_plugin": provider not in ("dense_local", "mock"),
            }
        )

    current_model = None
    current_quantization = None
    mgr_status = rt.model_manager.get_status()
    model_key = mgr_status.get("current_embedding_model")
    if model_key:
        parts = str(model_key).rsplit("_", 1)
        current_model = parts[0] if len(parts) > 1 else model_key
        current_quantization = parts[1] if len(parts) > 1 else "float32"

    return {
        "models": models,
        "current_model": current_model,
        "current_quantization": current_quantization,
    }


async def load_model(
    model_name: str, quantization: str = "float32", runtime: VecpipeRuntime | None = None
) -> dict[str, Any]:
    """Load a specific embedding model by forcing a warm-up embedding."""
    cfg = _get_settings()
    rt = _resolve_runtime(runtime)

    if cfg.USE_MOCK_EMBEDDINGS:
        raise HTTPException(status_code=400, detail="Cannot load models when using mock embeddings")

    try:
        await rt.model_manager.generate_embedding_async("warm-up", model_name, quantization)
        mgr_status = rt.model_manager.get_status()
        model_info = mgr_status.get("provider_info", {})
        return {
            "status": "success",
            "model": model_name,
            "quantization": quantization,
            "provider": mgr_status.get("embedding_provider"),
            "info": model_info,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - fallback
        logger.error("Model load error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Model load failed: {str(exc)}") from exc


async def suggest_models(runtime: VecpipeRuntime | None = None) -> dict[str, Any]:
    """Suggest optimal model configuration based on available GPU memory."""
    rt = _resolve_runtime(runtime)

    from vecpipe.memory_utils import get_gpu_memory_info, suggest_model_configuration

    free_mb, total_mb = get_gpu_memory_info()

    if total_mb == 0:
        return {
            "gpu_available": False,
            "message": "No GPU detected. CPU mode will be used.",
            "suggestion": {
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "embedding_quantization": "float32",
                "reranker_model": None,
                "reranker_quantization": None,
                "notes": ["CPU mode - using lightweight models"],
            },
        }

    suggestions = suggest_model_configuration(free_mb)

    return {
        "gpu_available": True,
        "gpu_memory": {
            "free_mb": free_mb,
            "total_mb": total_mb,
            "used_mb": total_mb - free_mb,
            "usage_percent": round((total_mb - free_mb) / total_mb * 100, 1),
        },
        "suggestion": suggestions,
        "current_models": {
            "embedding": getattr(rt.model_manager, "current_model_key", None),
            "reranker": getattr(rt.model_manager, "current_reranker_key", None),
        },
    }


async def embedding_info(runtime: VecpipeRuntime | None = None) -> dict[str, Any]:
    """Return information about the embedding configuration."""
    cfg = _get_settings()
    rt = _resolve_runtime(runtime)

    model_status = rt.model_manager.get_status() if rt.model_manager else {}

    info: dict[str, Any] = {
        "mode": "mock" if cfg.USE_MOCK_EMBEDDINGS else "real",
        "available": rt.model_manager is not None,
        "is_mock_mode": cfg.USE_MOCK_EMBEDDINGS,
    }

    if model_status.get("embedding_model_loaded"):
        provider_info = model_status.get("provider_info", {})
        current_model_key = model_status.get("current_embedding_model", "")

        if "_" in str(current_model_key):
            parts = str(current_model_key).rsplit("_", 1)
            model_name = parts[0]
            quant = parts[1] if len(parts) > 1 else "unknown"
        else:
            model_name = str(current_model_key)
            quant = provider_info.get("quantization", "unknown")

        info.update(
            {
                "current_model": model_name,
                "quantization": quant,
                "device": provider_info.get("device"),
                "provider": model_status.get("embedding_provider"),
                "dimension": provider_info.get("dimension"),
                "model_details": provider_info,
            }
        )
    else:
        info["note"] = "Embedding model loaded on first use"
        info.update(
            {
                "default_model": cfg.DEFAULT_EMBEDDING_MODEL,
                "default_quantization": cfg.DEFAULT_QUANTIZATION,
            }
        )

    return info


__all__ = [
    "DEFAULT_K",
    "SEARCH_INSTRUCTIONS",
    "perform_search",
    "perform_batch_search",
    "embed_texts",
    "upsert_points",
    "model_status",
    "health",
    "list_models",
    "load_model",
    "suggest_models",
    "embedding_info",
    "generate_mock_embedding",
]
