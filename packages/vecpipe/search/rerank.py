"""Reranking helpers for VecPipe search."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException

if TYPE_CHECKING:
    import httpx

    from shared.contracts.search import SearchRequest, SearchResult

from vecpipe.memory_utils import InsufficientMemoryError
from vecpipe.qwen3_search_config import RERANK_CONFIG, RERANKING_INSTRUCTIONS, get_reranker_for_embedding_model
from vecpipe.search.metrics import rerank_content_fetch_latency, rerank_fallbacks
from vecpipe.search.payloads import fetch_payloads_for_chunk_ids

logger = logging.getLogger(__name__)


def calculate_candidate_k(requested_k: int) -> int:
    """Calculate how many candidates to fetch before reranking."""
    multiplier_raw = RERANK_CONFIG.get("candidate_multiplier", 5)
    min_candidates_raw = RERANK_CONFIG.get("min_candidates", 20)
    max_candidates_raw = RERANK_CONFIG.get("max_candidates", 200)

    multiplier = int(multiplier_raw) if isinstance(multiplier_raw, int | float | str) else 5
    min_candidates = int(min_candidates_raw) if isinstance(min_candidates_raw, int | float | str) else 20
    max_candidates = int(max_candidates_raw) if isinstance(max_candidates_raw, int | float | str) else 200

    return max(min_candidates, min(requested_k * multiplier, max_candidates))


async def maybe_rerank_results(
    *,
    cfg: Any,
    model_manager: Any,
    qdrant_http: httpx.AsyncClient,
    collection_name: str,
    request: SearchRequest,
    results: list[SearchResult],
    embedding_model_name: str,
    embedding_quantization: str,
) -> tuple[list[SearchResult], str | None, float | None]:
    """Optionally rerank results, returning (results, reranker_model_used, reranking_time_ms)."""
    if not request.use_reranker or not results:
        return results[: request.k], None, None

    rerank_start = time.time()
    try:
        reranker_model = request.rerank_model or get_reranker_for_embedding_model(embedding_model_name)
        reranker_quantization = request.rerank_quantization or embedding_quantization

        if not all(r.content for r in results):
            chunk_ids_to_fetch = [r.chunk_id for r in results if not r.content]
            if chunk_ids_to_fetch:
                content_fetch_start = time.time()
                payloads = await fetch_payloads_for_chunk_ids(
                    collection_name=collection_name,
                    chunk_ids=chunk_ids_to_fetch,
                    cfg=cfg,
                    qdrant_http=qdrant_http,
                    filters=request.filters,
                )
                for r in results:
                    if not r.content and r.chunk_id in payloads:
                        r.content = payloads[r.chunk_id].get("content")
                rerank_content_fetch_latency.observe(time.time() - content_fetch_start)

        documents = [r.content if r.content else f"Document from {r.path} (chunk {r.chunk_id})" for r in results]
        instruction = RERANKING_INSTRUCTIONS.get(request.search_type, RERANKING_INSTRUCTIONS["general"])

        if model_manager is None:
            raise RuntimeError("Model manager not initialized")

        reranked_indices = await model_manager.rerank_async(
            query=request.query,
            documents=documents,
            top_k=request.k,
            model_name=reranker_model,
            quantization=reranker_quantization,
            instruction=instruction,
        )

        reranked_results: list[SearchResult] = []
        for idx, score in reranked_indices:
            if 0 <= idx < len(results):
                result = results[idx]
                result.score = score
                reranked_results.append(result)

        final_results = reranked_results if reranked_results else results[: request.k]
        model_used = f"{reranker_model}/{reranker_quantization}"
        reranking_time_ms = (time.time() - rerank_start) * 1000
        return final_results, model_used, reranking_time_ms

    except InsufficientMemoryError as exc:
        logger.error("Insufficient memory for reranking: %s", exc)
        raise HTTPException(
            status_code=507,
            detail={
                "error": "insufficient_memory",
                "message": str(exc),
                "suggestion": "Try using a smaller model or different quantization (float16/int8)",
            },
        ) from exc
    except Exception as exc:  # pragma: no cover - safety path
        logger.error("Reranking failed: %s", exc, exc_info=True)
        rerank_fallbacks.labels(reason="error").inc()

        # Check if user wants to fail on error instead of fallback
        if getattr(request, "rerank_on_error", "fallback") == "error":
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "reranking_failed",
                    "message": f"Reranking failed: {exc}",
                    "suggestion": "Set rerank_on_error='fallback' to return un-reranked results on failure",
                },
            ) from exc

        logger.warning("Falling back to vector search results due to reranking failure")
        return results[: request.k], None, (time.time() - rerank_start) * 1000
