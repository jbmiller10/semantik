"""Sparse and hybrid search helpers."""

from __future__ import annotations

import inspect
import logging
import time
from typing import Any

from vecpipe.search.metrics import (
    qdrant_ad_hoc_client_total,
    sparse_encode_query_latency,
    sparse_search_fallbacks,
    sparse_search_latency,
)
from vecpipe.sparse import search_sparse_collection

logger = logging.getLogger(__name__)


def _reciprocal_rank_fusion(
    dense_results: list[dict[str, Any]],
    sparse_results: list[dict[str, Any]],
    k: int,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """Combine dense and sparse results using Reciprocal Rank Fusion (RRF)."""
    if not dense_results and not sparse_results:
        return []

    dense_ranks: dict[str, int] = {r["chunk_id"]: i + 1 for i, r in enumerate(dense_results)}
    sparse_ranks: dict[str, int] = {r["chunk_id"]: i + 1 for i, r in enumerate(sparse_results)}

    all_chunk_ids = set(dense_ranks.keys()) | set(sparse_ranks.keys())

    rrf_scores: dict[str, float] = {}
    for chunk_id in all_chunk_ids:
        score = 0.0
        if chunk_id in dense_ranks:
            score += 1.0 / (rrf_k + dense_ranks[chunk_id])
        if chunk_id in sparse_ranks:
            score += 1.0 / (rrf_k + sparse_ranks[chunk_id])
        rrf_scores[chunk_id] = score

    sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[:k]

    dense_payload_map = {r["chunk_id"]: r for r in dense_results}
    sparse_payload_map = {r["chunk_id"]: r for r in sparse_results}
    dense_score_map = {r["chunk_id"]: r.get("score", 0.0) for r in dense_results}
    sparse_score_map = {r["chunk_id"]: r.get("score", 0.0) for r in sparse_results}

    fused_results: list[dict[str, Any]] = []
    for chunk_id in sorted_chunk_ids:
        base_result = dense_payload_map.get(chunk_id) or sparse_payload_map.get(chunk_id)
        if base_result:
            fused_result = {**base_result, "score": rrf_scores[chunk_id]}
            fused_result["_dense_rank"] = dense_ranks.get(chunk_id)
            fused_result["_sparse_rank"] = sparse_ranks.get(chunk_id)
            fused_result["_dense_score"] = dense_score_map.get(chunk_id)
            fused_result["_sparse_score"] = sparse_score_map.get(chunk_id)
            fused_results.append(fused_result)

    return fused_results


async def perform_sparse_search(
    *,
    cfg: Any,
    collection_name: str,
    sparse_config: dict[str, Any],
    query: str,
    k: int,
    sparse_manager: Any | None,
    qdrant_sdk: Any | None,
) -> tuple[list[dict[str, Any]], float, list[str]]:
    """Perform sparse-only search using the configured sparse indexer."""
    start_time = time.time()
    warnings: list[str] = []

    try:
        plugin_id = sparse_config.get("plugin_id")
        if not plugin_id:
            logger.warning("No plugin_id in sparse config for collection %s", collection_name)
            sparse_search_fallbacks.labels(reason="no_plugin_id").inc()
            warnings.append("Sparse search skipped: missing sparse plugin_id")
            return [], 0.0, warnings

        plugin_config = sparse_config.get("model_config") or {}
        if not isinstance(plugin_config, dict):
            plugin_config = {}

        # Encode sparse query
        if sparse_manager is not None:
            encode_start = time.time()
            query_config = dict(plugin_config)
            query_config["collection_name"] = collection_name
            query_vector = await sparse_manager.encode_query(plugin_id=plugin_id, query=query, config=query_config)
            encode_time = time.time() - encode_start

            sparse_type = getattr(query_vector, "_sparse_type", None)
            if sparse_type is None:
                from shared.plugins import load_plugins, plugin_registry

                load_plugins(plugin_types={"sparse_indexer"})
                record = plugin_registry.get("sparse_indexer", plugin_id)
                sparse_type = getattr(record.plugin_class, "SPARSE_TYPE", "unknown") if record else "unknown"

            sparse_encode_query_latency.labels(sparse_type=sparse_type).observe(encode_time)
            logger.debug("Sparse query encoding took %.3fs for plugin %s (managed)", encode_time, plugin_id)
        else:
            from shared.plugins import load_plugins, plugin_registry

            load_plugins(plugin_types={"sparse_indexer"})
            record = plugin_registry.get("sparse_indexer", plugin_id)
            if not record:
                logger.warning("Sparse indexer plugin '%s' not found", plugin_id)
                sparse_search_fallbacks.labels(reason="plugin_not_found").inc()
                warnings.append(f"Sparse search skipped: plugin '{plugin_id}' not found")
                return [], 0.0, warnings

            sparse_type = getattr(record.plugin_class, "SPARSE_TYPE", "unknown")
            plugin = record.plugin_class()
            init_config = dict(plugin_config)
            init_config.setdefault("collection_name", collection_name)

            try:
                initialize_fn = getattr(plugin, "initialize", None)
                if callable(initialize_fn):
                    maybe_coro = initialize_fn(init_config)
                    if inspect.isawaitable(maybe_coro):
                        await maybe_coro

                encode_start = time.time()
                query_vector = await plugin.encode_query(query)
                encode_time = time.time() - encode_start
                sparse_encode_query_latency.labels(sparse_type=sparse_type).observe(encode_time)
                logger.debug("Sparse query encoding took %.3fs for plugin %s (direct)", encode_time, plugin_id)
            finally:
                cleanup_fn = getattr(plugin, "cleanup", None)
                if callable(cleanup_fn):
                    try:
                        maybe_coro = cleanup_fn()
                        if inspect.isawaitable(maybe_coro):
                            await maybe_coro
                    except Exception as cleanup_exc:
                        logger.debug(
                            "Sparse plugin cleanup failed for plugin %s: %s",
                            plugin_id,
                            cleanup_exc,
                            exc_info=True,
                        )

        # Extract indices/values
        if hasattr(query_vector, "indices") and hasattr(query_vector, "values"):
            query_indices = list(query_vector.indices)
            query_values = list(query_vector.values)
        elif isinstance(query_vector, dict):
            query_indices = list(query_vector.get("indices") or [])
            query_values = list(query_vector.get("values") or [])
        else:
            logger.warning("Unsupported sparse query vector type: %s", type(query_vector))
            sparse_search_fallbacks.labels(reason="invalid_query_vector").inc()
            warnings.append("Sparse search skipped: sparse query encoder returned unsupported vector type")
            return [], (time.time() - start_time) * 1000, warnings

        if len(query_indices) != len(query_values):
            logger.warning(
                "Sparse query vector indices/values length mismatch: %d != %d",
                len(query_indices),
                len(query_values),
            )
            sparse_search_fallbacks.labels(reason="invalid_query_vector").inc()
            warnings.append("Sparse search skipped: sparse query indices/values mismatch")
            return [], (time.time() - start_time) * 1000, warnings

        if not query_indices:
            return [], (time.time() - start_time) * 1000, warnings

        # Search sparse collection
        sparse_collection_name = sparse_config.get("sparse_collection_name")
        if not sparse_collection_name:
            logger.warning("No sparse_collection_name in sparse config")
            sparse_search_fallbacks.labels(reason="no_collection_name").inc()
            warnings.append("Sparse search skipped: missing sparse_collection_name")
            return [], 0.0, warnings

        sdk_client = qdrant_sdk
        created_client = False
        if sdk_client is None:
            from qdrant_client import AsyncQdrantClient

            sdk_client = AsyncQdrantClient(
                url=f"http://{cfg.QDRANT_HOST}:{cfg.QDRANT_PORT}", api_key=cfg.QDRANT_API_KEY
            )
            qdrant_ad_hoc_client_total.labels(location="sparse_search").inc()
            created_client = True

        try:
            search_start = time.time()
            results = await search_sparse_collection(
                sparse_collection_name=sparse_collection_name,
                query_indices=query_indices,
                query_values=query_values,
                limit=k,
                qdrant_client=sdk_client,
            )
            qdrant_search_time = time.time() - search_start
            sparse_search_latency.labels(sparse_type=sparse_type).observe(qdrant_search_time)

            total_time = (time.time() - start_time) * 1000
            logger.debug(
                "Sparse search for %s: %d results in %.1fms (encode: %.1fms, qdrant: %.1fms)",
                collection_name,
                len(results),
                total_time,
                encode_time * 1000,
                qdrant_search_time * 1000,
            )
            return results, total_time, warnings
        finally:
            if created_client and sdk_client is not None:
                try:
                    await sdk_client.close()
                except Exception as close_exc:
                    logger.debug("Error closing ad-hoc Qdrant client: %s", close_exc, exc_info=True)

    except Exception as exc:
        logger.error("Sparse search failed for collection %s: %s", collection_name, exc, exc_info=True)
        sparse_search_fallbacks.labels(reason="error").inc()
        warnings.append(f"Sparse search failed ({type(exc).__name__}); returning empty sparse results")
        return [], (time.time() - start_time) * 1000, warnings
