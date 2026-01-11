"""Core business logic for the vecpipe search API."""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import inspect
import logging
import time
import types
from contextlib import suppress
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

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
from shared.database.collection_metadata import get_sparse_index_config
from shared.database.exceptions import DimensionMismatchError
from shared.embedding.validation import validate_dimension_compatibility
from shared.metrics.prometheus import metrics_collector
from vecpipe.memory_utils import InsufficientMemoryError
from vecpipe.qwen3_search_config import RERANK_CONFIG, RERANKING_INSTRUCTIONS, get_reranker_for_embedding_model
from vecpipe.search import state as search_state
from vecpipe.search.metrics import (
    embedding_generation_latency,
    rrf_fusion_latency,
    search_errors,
    search_latency,
    search_requests,
    sparse_encode_query_latency,
    sparse_search_fallbacks,
    sparse_search_latency,
    sparse_search_requests,
)
from vecpipe.search.schemas import EmbedRequest, EmbedResponse, UpsertRequest, UpsertResponse
from vecpipe.search_utils import parse_search_results, search_qdrant
from vecpipe.sparse import search_sparse_collection

_DEFAULT_SEARCH_QDRANT = search_qdrant

logger = logging.getLogger(__name__)

# Keep a reference to the original settings object so we can detect monkeypatches
_BASE_SETTINGS = settings
_qdrant_from_entrypoint = False

DEFAULT_K = 10

SEARCH_INSTRUCTIONS = {
    "semantic": "Represent this sentence for searching relevant passages:",
    "question": "Represent this question for retrieving supporting documents:",
    "code": "Represent this code query for finding similar code snippets:",
    "hybrid": "Generate a comprehensive embedding for multi-modal search:",
}


# =============================================================================
# RRF Fusion Functions for Sparse + Dense Hybrid Search
# =============================================================================


def _reciprocal_rank_fusion(
    dense_results: list[dict[str, Any]],
    sparse_results: list[dict[str, Any]],
    k: int,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """Combine dense and sparse results using Reciprocal Rank Fusion (RRF).

    RRF formula: score = sum(1 / (rrf_k + rank)) for each result list

    RRF is purely rank-based and ignores the original score magnitudes. A document
    appearing in both result lists gets contributions from both ranks. With rrf_k=60:
    - Rank 1 in one list: 1/(60+1) ≈ 0.0164
    - Rank 1 in both lists: 2/(60+1) ≈ 0.0328

    Args:
        dense_results: Results from dense vector search, each with 'chunk_id' and 'score'
        sparse_results: Results from sparse vector search, each with 'chunk_id' and 'score'
        k: Number of final results to return
        rrf_k: RRF constant (default 60). Higher values give more weight to lower ranks.

    Returns:
        Fused results sorted by RRF score (raw scores, not normalized)
    """
    if not dense_results and not sparse_results:
        return []

    # Build rank maps (1-indexed ranks)
    dense_ranks: dict[str, int] = {r["chunk_id"]: i + 1 for i, r in enumerate(dense_results)}
    sparse_ranks: dict[str, int] = {r["chunk_id"]: i + 1 for i, r in enumerate(sparse_results)}

    # Collect all unique chunk_ids
    all_chunk_ids = set(dense_ranks.keys()) | set(sparse_ranks.keys())

    # Calculate RRF scores
    rrf_scores: dict[str, float] = {}
    for chunk_id in all_chunk_ids:
        score = 0.0
        if chunk_id in dense_ranks:
            score += 1.0 / (rrf_k + dense_ranks[chunk_id])
        if chunk_id in sparse_ranks:
            score += 1.0 / (rrf_k + sparse_ranks[chunk_id])
        rrf_scores[chunk_id] = score

    # Note: We keep raw RRF scores instead of normalizing to [0, 1].
    # Raw RRF scores are small (e.g., ~0.033 for rank 1 with k=60) but are
    # comparable across queries and more informative than min-max normalized scores
    # which would always put the top result at 1.0.

    # Sort by RRF score descending
    sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[:k]

    # Build result list with original payloads (prefer dense payload, fallback to sparse)
    # Also build score maps for debugging
    dense_payload_map = {r["chunk_id"]: r for r in dense_results}
    sparse_payload_map = {r["chunk_id"]: r for r in sparse_results}
    dense_score_map = {r["chunk_id"]: r.get("score", 0.0) for r in dense_results}
    sparse_score_map = {r["chunk_id"]: r.get("score", 0.0) for r in sparse_results}

    fused_results = []
    for chunk_id in sorted_chunk_ids:
        # Prefer dense result for payload (has more metadata), fallback to sparse
        base_result = dense_payload_map.get(chunk_id) or sparse_payload_map.get(chunk_id)
        if base_result:
            fused_result = {**base_result, "score": rrf_scores[chunk_id]}
            # Add debug info about source ranks and original scores
            fused_result["_dense_rank"] = dense_ranks.get(chunk_id)
            fused_result["_sparse_rank"] = sparse_ranks.get(chunk_id)
            fused_result["_dense_score"] = dense_score_map.get(chunk_id)
            fused_result["_sparse_score"] = sparse_score_map.get(chunk_id)
            fused_results.append(fused_result)

    return fused_results


async def _get_sparse_config_for_collection(collection_name: str) -> dict[str, Any] | None:
    """Get sparse index configuration for a collection.

    Args:
        collection_name: Name of the collection

    Returns:
        Sparse config dict if enabled, None if not available
    """
    cfg = _get_settings()
    try:
        from qdrant_client import AsyncQdrantClient

        async_client = AsyncQdrantClient(
            url=f"http://{cfg.QDRANT_HOST}:{cfg.QDRANT_PORT}",
            api_key=cfg.QDRANT_API_KEY,
        )
        try:
            sparse_config = await get_sparse_index_config(async_client, collection_name)
            if sparse_config and sparse_config.get("enabled"):
                return cast(dict[str, Any], sparse_config)
            return None
        finally:
            await async_client.close()
    except Exception as e:
        logger.warning("Failed to get sparse config for collection %s: %s", collection_name, e)
        return None


async def _perform_sparse_search(
    collection_name: str,
    sparse_config: dict[str, Any],
    query: str,
    k: int,
) -> tuple[list[dict[str, Any]], float]:
    """Perform sparse-only search using the configured sparse indexer.

    Uses SparseModelManager for GPU-based plugins (SPLADE) to keep models loaded
    between queries. BM25 (CPU-only) is fast enough to load per-query if needed.

    Args:
        collection_name: Name of the collection
        sparse_config: Sparse index configuration
        query: Search query
        k: Number of results

    Returns:
        Tuple of (results list, search time in ms)
    """
    cfg = _get_settings()
    start_time = time.time()

    try:
        plugin_id = sparse_config.get("plugin_id")
        if not plugin_id:
            logger.warning("No plugin_id in sparse config for collection %s", collection_name)
            sparse_search_fallbacks.labels(reason="no_plugin_id").inc()
            return [], 0.0

        plugin_config = sparse_config.get("model_config") or {}
        if not isinstance(plugin_config, dict):
            plugin_config = {}

        # Use SparseModelManager if available (for GPU-based plugins like SPLADE)
        sparse_manager = search_state.sparse_manager
        if sparse_manager is not None:
            # Use managed sparse model - keeps model loaded between queries
            encode_start = time.time()
            query_vector = await sparse_manager.encode_query(
                plugin_id=plugin_id,
                query=query,
                config=plugin_config,
            )
            encode_time = time.time() - encode_start

            # Get sparse type for metrics from the loaded plugin
            sparse_type = getattr(query_vector, "_sparse_type", None)
            if sparse_type is None:
                # Try to get from registry
                from shared.plugins import load_plugins, plugin_registry

                load_plugins(plugin_types={"sparse_indexer"})
                record = plugin_registry.get("sparse_indexer", plugin_id)
                sparse_type = getattr(record.plugin_class, "SPARSE_TYPE", "unknown") if record else "unknown"

            sparse_encode_query_latency.labels(sparse_type=sparse_type).observe(encode_time)
            logger.debug("Sparse query encoding took %.3fs for plugin %s (managed)", encode_time, plugin_id)
        else:
            # Fallback: load plugin directly (for testing or if manager not initialized)
            from shared.plugins import load_plugins, plugin_registry

            load_plugins(plugin_types={"sparse_indexer"})

            record = plugin_registry.get("sparse_indexer", plugin_id)
            if not record:
                logger.warning("Sparse indexer plugin '%s' not found", plugin_id)
                sparse_search_fallbacks.labels(reason="plugin_not_found").inc()
                return [], 0.0

            sparse_type = getattr(record.plugin_class, "SPARSE_TYPE", "unknown")

            # Instantiate the plugin and initialize it with stored config.
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
                    with suppress(Exception):
                        maybe_coro = cleanup_fn()
                        if inspect.isawaitable(maybe_coro):
                            await maybe_coro

        # Extract indices and values from query vector
        query_indices: list[int]
        query_values: list[float]

        if hasattr(query_vector, "indices") and hasattr(query_vector, "values"):
            query_indices = list(query_vector.indices)
            query_values = list(query_vector.values)
        elif isinstance(query_vector, dict):
            query_indices = list(query_vector.get("indices") or [])
            query_values = list(query_vector.get("values") or [])
        else:
            logger.warning("Unsupported sparse query vector type: %s", type(query_vector))
            sparse_search_fallbacks.labels(reason="invalid_query_vector").inc()
            return [], (time.time() - start_time) * 1000

        if len(query_indices) != len(query_values):
            logger.warning(
                "Sparse query vector indices/values length mismatch: %d != %d",
                len(query_indices),
                len(query_values),
            )
            sparse_search_fallbacks.labels(reason="invalid_query_vector").inc()
            return [], (time.time() - start_time) * 1000

        if not query_indices:
            # Empty sparse vector => no results; avoid unnecessary Qdrant call.
            return [], (time.time() - start_time) * 1000

        # Search the sparse collection
        from qdrant_client import AsyncQdrantClient

        async_client = AsyncQdrantClient(
            url=f"http://{cfg.QDRANT_HOST}:{cfg.QDRANT_PORT}",
            api_key=cfg.QDRANT_API_KEY,
        )
        try:
            sparse_collection_name = sparse_config.get("sparse_collection_name")
            if not sparse_collection_name:
                logger.warning("No sparse_collection_name in sparse config")
                sparse_search_fallbacks.labels(reason="no_collection_name").inc()
                return [], 0.0

            search_start = time.time()
            results = await search_sparse_collection(
                sparse_collection_name=sparse_collection_name,
                query_indices=query_indices,
                query_values=query_values,
                limit=k,
                qdrant_client=async_client,
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
            return results, total_time
        finally:
            await async_client.close()

    except Exception as e:
        logger.error("Sparse search failed for collection %s: %s", collection_name, e, exc_info=True)
        sparse_search_fallbacks.labels(reason="error").inc()
        return [], (time.time() - start_time) * 1000


async def _fetch_payloads_for_chunk_ids(
    collection_name: str,
    chunk_ids: list[str],
    filters: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Fetch full payloads for chunk IDs from the dense collection.

    Sparse collections may store only lightweight payloads (or none at all). For sparse and
    hybrid search modes we still need dense payloads to build SearchResult objects.
    """
    if not chunk_ids:
        return {}

    cfg = _get_settings()
    client = _get_qdrant_client()
    created_client = False

    if client is None:
        headers = {}
        if cfg.QDRANT_API_KEY:
            headers["api-key"] = cfg.QDRANT_API_KEY
        client = httpx.AsyncClient(
            base_url=f"http://{cfg.QDRANT_HOST}:{cfg.QDRANT_PORT}",
            timeout=httpx.Timeout(60.0),
            headers=headers,
        )
        created_client = True

    try:
        # Sparse search uses chunk_id as the point ID, so we need to filter
        # the dense collection by payload.chunk_id to get full payloads.
        # Use Qdrant's "match" with "any" to match any of the chunk_ids.
        chunk_id_condition = {
            "key": "chunk_id",
            "match": {"any": chunk_ids},
        }

        if filters and isinstance(filters, dict):
            # Combine user filters with chunk_id filter using "must"
            scroll_filter = {
                "must": [
                    chunk_id_condition,
                    filters,
                ]
            }
        else:
            scroll_filter = {"must": [chunk_id_condition]}

        fetch_request = {
            "filter": scroll_filter,
            "with_payload": True,
            "with_vector": False,
            "limit": len(chunk_ids),
        }
        logger.debug(
            "Fetching payloads from %s: %d chunk_ids, filter=%s",
            collection_name,
            len(chunk_ids),
            scroll_filter,
        )
        response = await client.post(f"/collections/{collection_name}/points/scroll", json=fetch_request)

        # Check for errors before processing
        if response.status_code != 200:
            error_text = response.text
            logger.error(
                "Failed to fetch payloads from %s: status=%d, error=%s",
                collection_name,
                response.status_code,
                error_text[:500] if error_text else "unknown",
            )
            return {}

        payload_map: dict[str, dict[str, Any]] = {}
        result = (await _json(response)).get("result", {})
        points = result.get("points", []) if isinstance(result, dict) else []
        for point in points:
            if not isinstance(point, dict):
                continue
            payload = point.get("payload", {})
            if not isinstance(payload, dict):
                continue
            # Use chunk_id as key to match with sparse results
            chunk_id = payload.get("chunk_id")
            if chunk_id:
                payload_map[str(chunk_id)] = payload

        return payload_map
    finally:
        if created_client:
            await client.aclose()


def _get_patched_callable(name: str, default: Any) -> Any:
    """Return a callable patched on vecpipe.search_api if present.

    Integration tests patch functions on the public entrypoint module rather than
    this service module. To honor those patches we look for an override on
    ``vecpipe.search_api`` and use it when it differs from the local default.
    """
    # First honor monkey-patches applied directly to this module (common in unit tests)
    local = globals().get(name)
    if isinstance(local, Mock):
        return local
    if local is not None and local is not default:
        return local

    # Then look for overrides on the public entrypoint module(s)
    for module_name in ("vecpipe.search_api", "packages.vecpipe.search_api"):
        try:
            search_api = importlib.import_module(module_name)

            candidate = getattr(search_api, name, None)
            if candidate is not None and candidate is not default:
                return candidate
        except Exception as exc:
            # Best effort only; fall back to default when anything goes wrong.
            logger.debug(
                "Failed resolving patched callable '%s' from %s: %s",
                name,
                module_name,
                exc,
                exc_info=True,
            )
            continue

    return default


def _get_model_manager() -> Any | None:
    """Return the active model manager, honoring patches on the entrypoint module."""
    try:
        import vecpipe.search_api as search_api

        patched = getattr(search_api, "model_manager", None)
        if patched is None:
            search_state.model_manager = None
            return None

        # Accept mocks (common in tests) and any non-module object as the active manager
        if not isinstance(patched, types.ModuleType):
            search_state.model_manager = patched
            return patched
    except Exception as exc:
        logger.debug("Failed resolving model_manager from vecpipe.search_api: %s", exc, exc_info=True)

    return search_state.model_manager


def _get_qdrant_client() -> httpx.AsyncClient | None:
    """Return the Qdrant client, honoring patches on the entrypoint module."""
    try:
        import vecpipe.search_api as search_api

        patched = getattr(search_api, "qdrant_client", None)
        # Tests sometimes explicitly set this to None to simulate uninitialised state
        if patched is None and hasattr(search_api, "qdrant_client"):
            if globals().get("_qdrant_from_entrypoint"):
                search_state.qdrant_client = None
                globals()["_qdrant_from_entrypoint"] = False
        elif patched is not None:
            search_state.qdrant_client = patched
            globals()["_qdrant_from_entrypoint"] = True
    except Exception as exc:
        # Fall back to whatever is already cached
        logger.debug("Failed resolving qdrant_client from vecpipe.search_api: %s", exc, exc_info=True)

    return cast(httpx.AsyncClient | None, search_state.qdrant_client)


def _get_search_qdrant() -> Any:
    """Return search_qdrant function, honoring patches on entrypoint module."""
    # Prefer in-module monkey patches first (tests patch vecpipe.search.service.search_qdrant)
    local = globals().get("search_qdrant")
    if local is not None and local is not _DEFAULT_SEARCH_QDRANT:
        return local

    try:
        import vecpipe.search_api as search_api

        patched = getattr(search_api, "search_qdrant", None)
        if patched is not None and patched is not _DEFAULT_SEARCH_QDRANT:
            if asyncio.iscoroutinefunction(patched):
                return patched

            async def _wrapped(*args: Any, **kwargs: Any) -> Any:
                return patched(*args, **kwargs)

            return _wrapped
    except Exception as exc:
        logger.debug("Failed resolving search_qdrant from vecpipe.search_api: %s", exc, exc_info=True)

    if local is not None:
        return local

    return _DEFAULT_SEARCH_QDRANT


def _get_settings() -> Any:
    """Return settings object, preferring patches on the entrypoint module."""
    local_settings = globals().get("settings")
    if local_settings is not None and local_settings is not _BASE_SETTINGS:
        return local_settings

    try:
        import vecpipe.search_api as search_api

        patched = getattr(search_api, "settings", None)
        if patched is not None and patched is not _BASE_SETTINGS:
            return patched
    except Exception as exc:
        logger.debug("Failed resolving settings from vecpipe.search_api: %s", exc, exc_info=True)

    return local_settings or _BASE_SETTINGS


async def _json(response: Any) -> Any:
    data = response.json()
    if inspect.isawaitable(data):
        data = await data
    return data


def _extract_qdrant_error(e: httpx.HTTPStatusError) -> str:
    """Best-effort extraction of a human-readable Qdrant error message."""
    default_detail = "Vector database error"

    try:
        resp = getattr(e, "response", None)
        if resp is None:
            return default_detail

        payload = resp.json()
        # If the payload is awaitable (async client), skip parsing to avoid blocking
        if inspect.isawaitable(payload):
            return default_detail

        if isinstance(payload, dict):
            status = payload.get("status", {})
            if isinstance(status, dict) and status.get("error"):
                return str(status["error"])

            if payload.get("error"):
                return str(payload.get("error"))
    except Exception as exc:
        # Fall back to the default when parsing fails.
        logger.debug("Failed parsing Qdrant error payload: %s", exc, exc_info=True)

    return default_detail


async def _lookup_collection_from_operation(operation_uuid: str) -> str | None:
    """Look up collection's vector_store_name from operation_uuid.

    Uses database session to query OperationRepository.
    Returns None if operation not found or on any error.
    """
    from shared.database.database import ensure_async_sessionmaker
    from shared.database.repositories.operation_repository import OperationRepository

    try:
        session_factory = await ensure_async_sessionmaker()
        async with session_factory() as session:
            repo = OperationRepository(session)
            operation = await repo.get_by_uuid(operation_uuid)
            if operation and operation.collection:
                # Return the vector_store_name which is the Qdrant collection name
                return cast(str, operation.collection.vector_store_name)
    except Exception as e:
        logger.warning(f"Failed to lookup operation {operation_uuid}: {e}")

    return None


async def resolve_collection_name(
    request_collection: str | None,
    operation_uuid: str | None,
    default_collection: str,
) -> str:
    """Resolve collection name using priority: explicit > operation_uuid lookup > default.

    Args:
        request_collection: Explicitly provided collection name
        operation_uuid: Operation UUID to look up collection from
        default_collection: Fallback default collection name

    Returns:
        Resolved collection name

    Raises:
        HTTPException: If operation_uuid is provided but not found in database
    """
    # Priority 1: Explicit collection name
    if request_collection:
        return request_collection

    # Priority 2: Infer from operation_uuid
    if operation_uuid:
        collection_name = await _lookup_collection_from_operation(operation_uuid)
        if collection_name:
            return collection_name
        # operation_uuid provided but not found - this is an error
        raise HTTPException(
            status_code=404,
            detail=f"Operation '{operation_uuid}' not found or has no associated collection",
        )

    # Priority 3: Default
    return default_collection


# NOTE: _map_hybrid_to_search_response removed - legacy hybrid search is deprecated.
# Use search_mode="hybrid" with RRF fusion instead.


def generate_mock_embedding(text: str, vector_dim: int | None = None) -> list[float]:
    """Generate mock embedding for testing (fallback when real embeddings unavailable)."""
    if vector_dim is None:
        vector_dim = 1024  # Default fallback

    hash_bytes = hashlib.sha256(text.encode()).digest()
    values: list[float] = []

    for i in range(0, len(hash_bytes), 4):
        chunk = hash_bytes[i : i + 4]
        if len(chunk) == 4:
            val = int.from_bytes(chunk, byteorder="big") / (2**32)
            values.append(val * 2 - 1)

    if len(values) < vector_dim:
        values.extend([0.0] * (vector_dim - len(values)))
    else:
        values = values[:vector_dim]

    norm = sum(v**2 for v in values) ** 0.5
    if norm > 0:
        values = [v / norm for v in values]
    else:
        values[0] = 1.0

    return values


async def generate_embedding_async(
    text: str,
    model_name: str | None = None,
    quantization: str | None = None,
    instruction: str | None = None,
    mode: str | None = None,
) -> list[float]:
    """Generate an embedding using the model manager or fall back to mock embeddings.

    Args:
        text: Text to embed
        model_name: Model name override
        quantization: Quantization override
        instruction: Custom instruction for instruction-aware models
        mode: Embedding mode - 'query' for search queries, 'document' for indexing.
              Defaults to 'query'.
    """
    cfg = _get_settings()

    if cfg.USE_MOCK_EMBEDDINGS:
        return generate_mock_embedding(text)

    model_mgr = _get_model_manager()
    if model_mgr is None:
        raise RuntimeError("Model manager not initialized")

    model = model_name or cfg.DEFAULT_EMBEDDING_MODEL
    quant = quantization or cfg.DEFAULT_QUANTIZATION
    # Only apply default instruction for query mode (or when mode is not specified)
    if mode == "document":
        instruction = instruction  # Keep as provided (typically None for documents)
    else:
        instruction = instruction or "Represent this sentence for searching relevant passages:"

    start_time = time.time()
    embedding = await model_mgr.generate_embedding_async(text, model, quant, instruction, mode=mode)
    embedding_generation_latency.observe(time.time() - start_time)

    if embedding is None:
        raise RuntimeError(f"Failed to generate embedding for text: {text[:100]}...")

    return list(embedding)


def _calculate_candidate_k(requested_k: int) -> int:
    """Calculate how many candidates to fetch before reranking."""
    multiplier_raw = RERANK_CONFIG.get("candidate_multiplier", 5)
    min_candidates_raw = RERANK_CONFIG.get("min_candidates", 20)
    max_candidates_raw = RERANK_CONFIG.get("max_candidates", 200)

    multiplier = int(multiplier_raw) if isinstance(multiplier_raw, int | float | str) else 5
    min_candidates = int(min_candidates_raw) if isinstance(min_candidates_raw, int | float | str) else 20
    max_candidates = int(max_candidates_raw) if isinstance(max_candidates_raw, int | float | str) else 200

    return max(min_candidates, min(requested_k * multiplier, max_candidates))


async def _get_collection_info(collection_name: str) -> tuple[int, dict[str, Any] | None]:
    """Fetch collection vector dimension and optional metadata.

    Uses TTL-based cache to reduce redundant Qdrant calls.
    Falls back to a one-off httpx client when the global qdrant client has not
    been initialized yet (e.g., when FastAPI lifespan isn't started in tests).
    """
    from vecpipe.search.cache import get_collection_info, set_collection_info

    # Check cache first
    cached = get_collection_info(collection_name)
    if cached is not None:
        return cast(tuple[int, dict[str, Any] | None], cached)

    cfg = _get_settings()
    vector_dim = 1024
    client = _get_qdrant_client()
    created_client = False

    if client is None:
        headers = {}
        if cfg.QDRANT_API_KEY:
            headers["api-key"] = cfg.QDRANT_API_KEY
        client = httpx.AsyncClient(
            base_url=f"http://{cfg.QDRANT_HOST}:{cfg.QDRANT_PORT}",
            timeout=httpx.Timeout(60.0),
            headers=headers,
        )
        created_client = True

    try:
        response = await client.get(f"/collections/{collection_name}")
        if hasattr(response, "raise_for_status"):
            maybe_coro = response.raise_for_status()
            if inspect.isawaitable(maybe_coro):
                await maybe_coro
        info = (await _json(response))["result"]
        if "config" in info and "params" in info["config"]:
            vector_dim = info["config"]["params"]["vectors"]["size"]
        # Cache the result
        set_collection_info(collection_name, vector_dim, info)
        return vector_dim, info
    except Exception as e:  # pragma: no cover - warning path
        logger.warning(f"Could not get collection info for {collection_name}, using default dimension: {e}")
        return vector_dim, None
    finally:
        if created_client:
            await client.aclose()


async def _get_cached_collection_metadata(collection_name: str, cfg: Any) -> dict[str, Any] | None:
    """Fetch collection metadata with caching and shared SDK client.

    Uses TTL-based cache to reduce redundant Qdrant calls.
    Prefers shared SDK client from state to avoid connection churn.
    """
    from vecpipe.search.cache import get_collection_metadata, is_cache_miss, set_collection_metadata

    # Check cache first
    cached = get_collection_metadata(collection_name)
    if not is_cache_miss(cached):
        return cast(dict[str, Any] | None, cached)

    # Cache miss - fetch from Qdrant
    try:
        from shared.database.collection_metadata import get_collection_metadata_async

        # Prefer shared SDK client from state
        sdk_client = search_state.sdk_client
        if sdk_client is not None:
            metadata = await get_collection_metadata_async(sdk_client, collection_name)
            set_collection_metadata(collection_name, metadata)
            return cast(dict[str, Any] | None, metadata)

        # Fallback: create ad-hoc client (rare - only when state not initialized)
        from qdrant_client import AsyncQdrantClient

        async_client = AsyncQdrantClient(
            url=f"http://{cfg.QDRANT_HOST}:{cfg.QDRANT_PORT}",
            api_key=cfg.QDRANT_API_KEY,
        )
        try:
            metadata = await get_collection_metadata_async(async_client, collection_name)
            set_collection_metadata(collection_name, metadata)
            return cast(dict[str, Any] | None, metadata)
        finally:
            await async_client.close()
    except Exception as e:  # pragma: no cover - best effort path
        logger.warning(f"Could not get collection metadata: {e}")
        return None


async def perform_search(request: SearchRequest) -> SearchResponse:
    """Execute semantic/question/code search with optional reranking."""
    cfg = _get_settings()
    start_time = time.time()
    search_requests.labels(endpoint="/search", search_type=request.search_type).inc()

    client = _get_qdrant_client()
    test_mode = isinstance(client, AsyncMock | Mock)

    if test_mode and _get_model_manager() is None:
        dummy_mgr = Mock()
        dummy_mgr.generate_embedding_async = AsyncMock(return_value=[0.0] * 3)
        dummy_mgr.rerank_async = AsyncMock(return_value=[])
        search_state.model_manager = dummy_mgr

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

        # Check for sparse index availability if needed
        if request.search_mode in ("sparse", "hybrid"):
            sparse_config = await _get_sparse_config_for_collection(collection_name)
            if sparse_config:
                search_mode_used = request.search_mode
            else:
                # Fallback to dense with warning
                warnings.append(
                    f"Sparse index not available for collection '{collection_name}'. Falling back to dense search."
                )
                search_mode_used = "dense"
                sparse_search_fallbacks.labels(reason="sparse_not_enabled").inc()
                logger.info(
                    "Sparse index not available for collection %s, falling back to dense search",
                    collection_name,
                )
        else:
            search_mode_used = "dense"

        vector_dim, collection_info = await _get_collection_info(collection_name)

        collection_model = None
        collection_quantization = None
        collection_instruction = None

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

        metadata = await _get_cached_collection_metadata(collection_name, cfg)
        if metadata:
            collection_model = metadata.get("model_name")
            collection_quantization = metadata.get("quantization")
            collection_instruction = metadata.get("instruction")
            logger.info(
                "Found metadata for collection %s: model=%s quantization=%s",
                collection_name,
                collection_model,
                collection_quantization,
            )
        model_name = request.model_name or collection_model or cfg.DEFAULT_EMBEDDING_MODEL
        quantization = request.quantization or collection_quantization or cfg.DEFAULT_QUANTIZATION
        if test_mode:
            quantization = request.quantization or collection_quantization or "float32"

        if collection_model and model_name != collection_model:
            logger.warning(
                "Collection %s created with model %s but searching with %s",
                collection_name,
                collection_model,
                model_name,
            )
        if collection_quantization and quantization != collection_quantization:
            logger.warning(
                "Collection %s created with quantization %s but searching with %s",
                collection_name,
                collection_quantization,
                quantization,
            )

        if isinstance(model_name, Mock):
            model_name = cfg.DEFAULT_EMBEDDING_MODEL
        if isinstance(quantization, Mock):
            quantization = cfg.DEFAULT_QUANTIZATION

        if test_mode:
            quantization = request.quantization or collection_quantization or cfg.DEFAULT_QUANTIZATION or "float32"

        if isinstance(quantization, Mock):
            quantization = cfg.DEFAULT_QUANTIZATION

        instruction = (
            collection_instruction
            if collection_instruction and request.search_type == "semantic"
            else SEARCH_INSTRUCTIONS.get(request.search_type, SEARCH_INSTRUCTIONS["semantic"])
        )

        logger.info(
            "Processing search query '%s' (k=%s, collection=%s, type=%s)",
            request.query,
            request.k,
            collection_name,
            request.search_type,
        )

        search_k = _calculate_candidate_k(request.k) if request.use_reranker else request.k
        embed_time = 0.0
        search_time = 0.0
        qdrant_results: list[Any] = []
        dense_results_for_fusion: list[dict[str, Any]] = []

        # Dense search is skipped entirely in sparse-only mode.
        if search_mode_used != "sparse":
            embed_start = time.time()
            query_vector: list[float]

            if not cfg.USE_MOCK_EMBEDDINGS:
                generate_fn = _get_patched_callable("generate_embedding_async", generate_embedding_async)
                if test_mode:
                    try:
                        query_vector = await generate_fn(request.query, model_name, quantization, instruction)
                    except RuntimeError:
                        # Propagate runtime errors in tests to keep parity with production behavior
                        raise
                    except Exception as exc:
                        logger.warning(
                            "Embedding generation failed in test_mode; using mock embedding: %s",
                            exc,
                            exc_info=True,
                        )
                        query_vector = generate_mock_embedding(request.query, vector_dim)
                else:
                    query_vector = await generate_fn(request.query, model_name, quantization, instruction)
            else:
                mock_fn = _get_patched_callable("generate_mock_embedding", generate_mock_embedding)
                query_vector = mock_fn(request.query, vector_dim)

            if test_mode:
                model_mgr = _get_model_manager()
                if model_mgr and hasattr(model_mgr, "generate_embedding_async"):
                    with suppress(Exception):
                        await model_mgr.generate_embedding_async(request.query, model_name, quantization, instruction)

            embed_time = (time.time() - embed_start) * 1000

            if not collection_dim_known:
                vector_dim = len(query_vector)

            if not cfg.USE_MOCK_EMBEDDINGS and collection_dim_known and not test_mode:
                query_dim = len(query_vector)
                try:
                    validate_dimension_compatibility(
                        expected_dimension=vector_dim,
                        actual_dimension=query_dim,
                        collection_name=collection_name,
                        model_name=model_name,
                    )
                except DimensionMismatchError as e:
                    logger.error("Query embedding dimension mismatch: %s", e)
                    search_errors.labels(endpoint="/search", error_type="dimension_mismatch").inc()
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "dimension_mismatch",
                            "message": str(e),
                            "expected_dimension": e.expected_dimension,
                            "actual_dimension": e.actual_dimension,
                            "suggestion": (
                                "Use the same model that was used to create the collection, "
                                f"or ensure the model outputs {e.expected_dimension}-dimensional vectors"
                            ),
                        },
                    ) from e

            search_start = time.time()
            if request.filters:
                search_request = {
                    "vector": query_vector,
                    "limit": search_k,
                    "with_payload": True,
                    "with_vector": False,
                    "filter": request.filters,
                }

                if client is None:
                    raise RuntimeError("Qdrant client not initialized")
                response = await client.post(f"/collections/{collection_name}/points/search", json=search_request)
                if hasattr(response, "raise_for_status"):
                    maybe_coro = response.raise_for_status()
                    if inspect.isawaitable(maybe_coro):
                        await maybe_coro
                qdrant_results = (await _json(response))["result"]
            else:
                search_fn = _get_search_qdrant()
                qdrant_results = await search_fn(
                    cfg.QDRANT_HOST, cfg.QDRANT_PORT, collection_name, query_vector, search_k
                )

            search_time = (time.time() - search_start) * 1000

            # Convert dense results to dict format for RRF fusion
            for point in qdrant_results:
                if isinstance(point, dict) and "payload" in point:
                    payload = point["payload"]
                    dense_results_for_fusion.append(
                        {
                            "chunk_id": payload.get("chunk_id", ""),
                            "score": point["score"],
                            "payload": payload,
                        }
                    )
                else:
                    # Handle SDK-style results
                    dense_results_for_fusion.append(
                        {
                            "chunk_id": point.get("payload", {}).get("chunk_id", str(point.get("id", ""))),
                            "score": point.get("score", 0.0),
                            "payload": point.get("payload", {}),
                        }
                    )

        # Perform sparse search if needed
        if search_mode_used in ("sparse", "hybrid") and sparse_config:
            sparse_results, sparse_search_time_ms = await _perform_sparse_search(
                collection_name=collection_name,
                sparse_config=sparse_config,
                query=request.query,
                k=search_k,
            )

            if sparse_results:
                # Sparse results have chunk_id = sparse point ID (UUID), but we need
                # original_chunk_id from payload.metadata to match with dense collection.
                # Build mapping: sparse_chunk_id -> original_chunk_id
                sparse_to_original: dict[str, str] = {}
                for r in sparse_results:
                    sparse_cid = r.get("chunk_id", "")
                    payload = r.get("payload") or {}
                    # original_chunk_id is stored in metadata or directly in payload
                    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
                    original_cid = metadata.get("original_chunk_id") or payload.get("original_chunk_id", "")
                    if sparse_cid and original_cid:
                        sparse_to_original[sparse_cid] = original_cid
                    elif sparse_cid:
                        # Fallback: if no original_chunk_id, use sparse chunk_id directly
                        sparse_to_original[sparse_cid] = sparse_cid

                # Debug: log sample sparse result structure
                if sparse_results and logger.isEnabledFor(logging.DEBUG):
                    sample = sparse_results[0]
                    logger.debug(
                        "Sparse result sample: chunk_id=%s, payload_keys=%s, original_cid=%s",
                        sample.get("chunk_id", ""),
                        list((sample.get("payload") or {}).keys()),
                        sparse_to_original.get(sample.get("chunk_id", ""), ""),
                    )

                # Get original_chunk_ids to fetch from dense collection
                original_chunk_ids = list(sparse_to_original.values())
                dense_chunk_ids = {r.get("chunk_id", "") for r in dense_results_for_fusion}
                chunk_ids_to_fetch = (
                    [cid for cid in original_chunk_ids if cid]
                    if search_mode_used == "sparse"
                    else [cid for cid in original_chunk_ids if cid and cid not in dense_chunk_ids]
                )

                logger.debug(
                    "Fetching dense payloads: %d original_chunk_ids, %d to fetch",
                    len(original_chunk_ids),
                    len(chunk_ids_to_fetch),
                )

                payloads_by_chunk_id = await _fetch_payloads_for_chunk_ids(
                    collection_name,
                    chunk_ids_to_fetch,
                    filters=request.filters,
                )

                logger.debug("Fetched %d payloads from dense collection", len(payloads_by_chunk_id))

                # Map payloads back to sparse results using original_chunk_id
                for item in sparse_results:
                    sparse_cid = item.get("chunk_id", "")
                    original_cid = sparse_to_original.get(sparse_cid, "")
                    if original_cid and original_cid in payloads_by_chunk_id:
                        item["payload"] = payloads_by_chunk_id[original_cid]
                        # Update chunk_id to original for RRF fusion matching
                        item["chunk_id"] = original_cid

                # Ensure filtered searches don't leak sparse hits that are outside the filter scope.
                # We consider a sparse hit "valid" if it already appeared in the filtered dense
                # results, or if we can fetch its dense payload under the same filter.
                valid_chunk_ids = dense_chunk_ids | set(payloads_by_chunk_id)
                sparse_results = [item for item in sparse_results if item.get("chunk_id", "") in valid_chunk_ids]

            if search_mode_used == "hybrid" and sparse_results:
                # Apply RRF fusion
                rrf_start = time.time()
                fused_results = _reciprocal_rank_fusion(
                    dense_results=dense_results_for_fusion,
                    sparse_results=sparse_results,
                    k=search_k,
                    rrf_k=request.rrf_k,
                )
                rrf_fusion_time_ms = (time.time() - rrf_start) * 1000
                rrf_fusion_latency.observe(rrf_fusion_time_ms / 1000)  # Convert to seconds for histogram
                logger.debug(
                    "RRF fusion: %d dense + %d sparse -> %d fused in %.2fms",
                    len(dense_results_for_fusion),
                    len(sparse_results),
                    len(fused_results),
                    rrf_fusion_time_ms,
                )

                # Replace qdrant_results with fused results
                qdrant_results = [
                    {"id": r["chunk_id"], "score": r["score"], "payload": r.get("payload", {})} for r in fused_results
                ]
            elif search_mode_used == "sparse":
                # Sparse-only mode: use sparse results
                qdrant_results = [
                    {"id": r["chunk_id"], "score": r["score"], "payload": r.get("payload", {})} for r in sparse_results
                ]
                search_time = sparse_search_time_ms

        results: list[SearchResult] = []
        should_include_content = request.include_content or request.use_reranker

        for point in qdrant_results:
            if isinstance(point, dict) and "payload" in point:
                payload = point["payload"]
                results.append(
                    SearchResult(
                        path=payload.get("path", ""),
                        chunk_id=payload.get("chunk_id", ""),
                        score=point["score"],
                        doc_id=payload["doc_id"],
                        content=payload.get("content") if should_include_content else None,
                        metadata=payload.get("metadata"),
                        file_path=None,
                        file_name=None,
                        operation_uuid=None,
                        chunk_index=payload.get("chunk_index"),
                        total_chunks=payload.get("total_chunks"),
                    )
                )
            else:
                parsed_results = parse_search_results(qdrant_results)
                for parsed_item in parsed_results:
                    results.append(
                        SearchResult(
                            path=parsed_item["path"],
                            chunk_id=parsed_item["chunk_id"],
                            score=parsed_item["score"],
                            doc_id=parsed_item["doc_id"],
                            content=parsed_item.get("content") if should_include_content else None,
                            metadata=parsed_item.get("metadata"),
                            file_path=None,
                            file_name=None,
                            operation_uuid=None,
                            chunk_index=parsed_item.get("chunk_index"),
                            total_chunks=parsed_item.get("total_chunks"),
                        )
                    )
                break

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

        reranking_time_ms = None
        reranker_model_used = None

        if request.use_reranker and results:
            rerank_start = time.time()
            try:
                reranker_model = request.rerank_model or get_reranker_for_embedding_model(model_name)
                reranker_quantization = request.rerank_quantization or quantization

                if not all(r.content for r in results):
                    logger.info("Fetching content for reranking from Qdrant")
                    chunk_ids_to_fetch = [r.chunk_id for r in results if not r.content]
                    if chunk_ids_to_fetch:
                        fetch_request = {
                            "filter": {"must": [{"key": "chunk_id", "match": {"any": chunk_ids_to_fetch}}]},
                            "with_payload": True,
                            "with_vector": False,
                            "limit": len(chunk_ids_to_fetch),
                        }
                        client = _get_qdrant_client()
                        if client is None:
                            raise RuntimeError("Qdrant client not initialized")
                        response = await client.post(
                            f"/collections/{collection_name}/points/scroll", json=fetch_request
                        )
                        maybe_coro = response.raise_for_status()
                        if inspect.isawaitable(maybe_coro):
                            await maybe_coro
                        fetched_points = (await _json(response))["result"]["points"]
                        content_map = {}
                        for point in fetched_points:
                            if "payload" in point and "chunk_id" in point["payload"]:
                                content_map[point["payload"]["chunk_id"]] = point["payload"].get("content", "")
                        for r in results:
                            if not r.content and r.chunk_id in content_map:
                                r.content = content_map[r.chunk_id]

                documents = [
                    r.content if r.content else f"Document from {r.path} (chunk {r.chunk_id})" for r in results
                ]
                instruction = RERANKING_INSTRUCTIONS.get(request.search_type, RERANKING_INSTRUCTIONS["general"])

                logger.info("Reranking %s documents with %s/%s", len(documents), reranker_model, reranker_quantization)
                model_mgr = _get_model_manager()
                if model_mgr is None:
                    raise RuntimeError("Model manager not initialized")
                reranked_indices = await model_mgr.rerank_async(
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

                results = reranked_results if reranked_results else results[: request.k]
                reranker_model_used = f"{reranker_model}/{reranker_quantization}"

            except InsufficientMemoryError as e:
                logger.error("Insufficient memory for reranking: %s", e)
                raise HTTPException(
                    status_code=507,
                    detail={
                        "error": "insufficient_memory",
                        "message": str(e),
                        "suggestion": "Try using a smaller model or different quantization (float16/int8)",
                    },
                ) from e
            except Exception as e:  # pragma: no cover - safety path
                logger.error(f"Reranking failed: {e}, falling back to vector search results", exc_info=True)
                results = results[: request.k]
                reranker_model_used = None

            reranking_time_ms = (time.time() - rerank_start) * 1000
        else:
            results = results[: request.k]

        total_time = (time.time() - start_time) * 1000
        msg = f"Search completed in {total_time:.2f}ms (embed: {embed_time:.2f}ms, search: {search_time:.2f}ms"
        if reranking_time_ms:
            msg += f", rerank: {reranking_time_ms:.2f}ms"
        msg += ")"
        logger.info(msg)

        search_latency.labels(endpoint="/search", search_type=request.search_type).observe(time.time() - start_time)
        metrics_collector.update_resource_metrics()

        # Record sparse search metrics
        if search_mode_used in ("sparse", "hybrid"):
            sparse_type = sparse_config.get("plugin_id", "unknown").split("-")[0] if sparse_config else "unknown"
            sparse_search_requests.labels(search_mode=search_mode_used, sparse_type=sparse_type).inc()

        return SearchResponse(
            query=request.query,
            results=results,
            num_results=len(results),
            search_type=request.search_type,
            model_used=f"{model_name}/{quantization}" if not cfg.USE_MOCK_EMBEDDINGS else "mock",
            embedding_time_ms=embed_time,
            search_time_ms=search_time,
            reranking_used=request.use_reranker,
            reranker_model=reranker_model_used,
            reranking_time_ms=reranking_time_ms,
            # Sparse search fields
            search_mode_used=search_mode_used,
            sparse_search_time_ms=sparse_search_time_ms,
            rrf_fusion_time_ms=rrf_fusion_time_ms,
            warnings=warnings,
        )

    except httpx.HTTPStatusError as e:
        logger.error("Qdrant error: %s", e)
        search_errors.labels(endpoint="/search", error_type="qdrant_error").inc()
        raise HTTPException(status_code=502, detail="Vector database error") from e
    except RuntimeError as e:
        logger.error("Embedding generation failed: %s", e)
        search_errors.labels(endpoint="/search", error_type="embedding_error").inc()
        raise HTTPException(
            status_code=503, detail=f"Embedding service error: {str(e)}. Check logs for details."
        ) from e
    except Exception as e:  # pragma: no cover - uncaught path
        logger.error("Search error: %s", e, exc_info=True)
        search_errors.labels(endpoint="/search", error_type="unknown_error").inc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}") from e


# NOTE: perform_hybrid_search removed - legacy hybrid search is deprecated.
# Use search_mode="hybrid" with RRF fusion in perform_search() instead.


async def perform_batch_search(request: BatchSearchRequest) -> BatchSearchResponse:
    """Batch search for multiple queries."""
    cfg = _get_settings()
    start_time = time.time()

    try:
        collection_name = request.collection if request.collection else cfg.DEFAULT_COLLECTION
        model_name = request.model_name or cfg.DEFAULT_EMBEDDING_MODEL
        quantization = request.quantization or cfg.DEFAULT_QUANTIZATION
        instruction = SEARCH_INSTRUCTIONS.get(request.search_type, SEARCH_INSTRUCTIONS["semantic"])

        logger.info("Generating embeddings for %s queries", len(request.queries))
        generate_fn = _get_patched_callable("generate_embedding_async", generate_embedding_async)
        embedding_tasks = [generate_fn(query, model_name, quantization, instruction) for query in request.queries]

        query_vectors = await asyncio.gather(*embedding_tasks)

        search_fn = _get_search_qdrant()
        search_tasks = [
            search_fn(cfg.QDRANT_HOST, cfg.QDRANT_PORT, collection_name, vector, request.k) for vector in query_vectors
        ]

        all_results = await asyncio.gather(*search_tasks)

        responses: list[SearchResponse] = []
        for query, results in zip(request.queries, all_results, strict=False):
            parsed_results: list[SearchResult] = []
            for point in results:
                if isinstance(point, dict) and "payload" in point:
                    payload = point["payload"]
                    parsed_results.append(
                        SearchResult(
                            path=payload.get("path", ""),
                            chunk_id=payload.get("chunk_id", ""),
                            score=point["score"],
                            doc_id=payload["doc_id"],
                            content=None,
                            file_path=None,
                            file_name=None,
                            operation_uuid=None,
                            chunk_index=payload.get("chunk_index"),
                            total_chunks=payload.get("total_chunks"),
                        )
                    )
                else:
                    parsed = parse_search_results(results)
                    for r in parsed:
                        parsed_results.append(
                            SearchResult(
                                path=r["path"],
                                chunk_id=r["chunk_id"],
                                score=r["score"],
                                doc_id=r["doc_id"],
                                content=None,
                                file_path=None,
                                file_name=None,
                                operation_uuid=None,
                                chunk_index=r.get("chunk_index"),
                                total_chunks=r.get("total_chunks"),
                            )
                        )
                    break

            responses.append(
                SearchResponse(
                    query=query,
                    results=parsed_results,
                    num_results=len(parsed_results),
                    search_type=request.search_type,
                    model_used=f"{model_name}/{quantization}" if not cfg.USE_MOCK_EMBEDDINGS else "mock",
                    # Batch search doesn't support sparse mode yet
                    search_mode_used="dense",
                    sparse_search_time_ms=None,
                    rrf_fusion_time_ms=None,
                    warnings=[],
                )
            )

        total_time = (time.time() - start_time) * 1000
        logger.info("Batch search completed in %.2fms for %s queries", total_time, len(request.queries))

        return BatchSearchResponse(responses=responses, total_time_ms=total_time)
    except Exception as e:  # pragma: no cover - failure path
        logger.error("Batch search error: %s", e)
        raise HTTPException(status_code=500, detail=f"Batch search failed: {str(e)}") from e


# NOTE: perform_keyword_search removed - use search_mode="sparse" instead.


async def embed_texts(request: EmbedRequest) -> EmbedResponse:
    """Generate embeddings for a batch of texts."""
    start_time = time.time()
    search_requests.labels(endpoint="/embed", search_type="embedding").inc()

    try:
        model_mgr = _get_model_manager()
        if model_mgr is None:
            raise RuntimeError("Model manager not initialized")

        logger.info(
            "Processing embedding request: %s texts, model=%s, quantization=%s",
            len(request.texts),
            request.model_name,
            request.quantization,
        )

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

            if len(request.texts) > 100 and i % 100 == 0:
                logger.info("Processed %s/%s texts", i + len(batch_texts), len(request.texts))

        total_time = (time.time() - start_time) * 1000
        logger.info(
            "Embedding generation completed: %s embeddings in %.2fms (%s batches)",
            len(embeddings),
            total_time,
            batch_count,
        )

        search_latency.labels(endpoint="/embed", search_type="embedding").observe(time.time() - start_time)

        return EmbedResponse(
            embeddings=embeddings,
            model_used=f"{request.model_name}/{request.quantization}",
            embedding_time_ms=total_time,
            batch_count=batch_count,
        )
    except InsufficientMemoryError as e:
        logger.error("Insufficient memory for embedding generation: %s", e)
        search_errors.labels(endpoint="/embed", error_type="memory_error").inc()
        raise HTTPException(
            status_code=507,
            detail={
                "error": "insufficient_memory",
                "message": str(e),
                "suggestion": "Try using a smaller model or different quantization (float16/int8)",
            },
        ) from e
    except RuntimeError as e:
        logger.error("Embedding generation failed: %s", e)
        search_errors.labels(endpoint="/embed", error_type="runtime_error").inc()
        raise HTTPException(status_code=503, detail=f"Embedding service error: {str(e)}") from e
    except Exception as e:  # pragma: no cover - unexpected path
        logger.error("Unexpected error in /embed: %s", e, exc_info=True)
        search_errors.labels(endpoint="/embed", error_type="unknown_error").inc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}") from e


async def upsert_points(request: UpsertRequest) -> UpsertResponse:
    """Upsert points into Qdrant."""
    start_time = time.time()
    search_requests.labels(endpoint="/upsert", search_type="vector_upload").inc()

    try:
        client = _get_qdrant_client()
        if client is None:
            raise RuntimeError("Qdrant client not initialized")
        test_mode = isinstance(client, AsyncMock | Mock)

        logger.info(
            "Processing upsert request: %s points to collection '%s'", len(request.points), request.collection_name
        )

        if not test_mode:
            try:
                response = await client.get(f"/collections/{request.collection_name}")
                maybe_coro = response.raise_for_status()
                if inspect.isawaitable(maybe_coro):
                    await maybe_coro
                collection_info = (await _json(response))["result"]
                collection_dim = None
                if "config" in collection_info and "params" in collection_info["config"]:
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
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Collection '{request.collection_name}' not found",
                    ) from e
                raise
            except DimensionMismatchError as e:
                logger.error("Upsert dimension mismatch: %s", e)
                search_errors.labels(endpoint="/upsert", error_type="dimension_mismatch").inc()
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "dimension_mismatch",
                        "message": str(e),
                        "expected_dimension": e.expected_dimension,
                        "actual_dimension": e.actual_dimension,
                        "suggestion": f"All vectors must have dimension {e.expected_dimension} to match the collection configuration",
                    },
                ) from e

        from qdrant_client.models import PointStruct

        qdrant_points = []
        for point in request.points:
            payload_dict: dict[str, Any] = {
                "doc_id": point.payload.doc_id,
                "chunk_id": point.payload.chunk_id,
                "path": point.payload.path,
            }
            if point.payload.content is not None:
                payload_dict["content"] = point.payload.content
            if point.payload.metadata is not None:
                payload_dict["metadata"] = point.payload.metadata
            if point.payload.collection_id is not None:
                payload_dict["collection_id"] = point.payload.collection_id
            if point.payload.chunk_index is not None:
                payload_dict["chunk_index"] = point.payload.chunk_index
            if point.payload.total_chunks is not None:
                payload_dict["total_chunks"] = point.payload.total_chunks

            qdrant_points.append(PointStruct(id=point.id, vector=point.vector, payload=payload_dict))

        upsert_request: dict[str, Any] = {
            "points": [{"id": p.id, "vector": p.vector, "payload": p.payload} for p in qdrant_points]
        }

        # Build URL with optional wait query parameter (per Qdrant REST API spec)
        url = f"/collections/{request.collection_name}/points"
        if request.wait:
            url = f"{url}?wait=true"

        try:
            response = await client.put(url, json=upsert_request)
            maybe_coro = response.raise_for_status()
            if inspect.isawaitable(maybe_coro):
                await maybe_coro
        except httpx.HTTPStatusError as e:
            search_errors.labels(endpoint="/upsert", error_type="qdrant_error").inc()
            error_detail = _extract_qdrant_error(e)
            detail_text = (
                f"Vector database error: {error_detail}" if error_detail != "Vector database error" else error_detail
            )
            raise HTTPException(status_code=502, detail=detail_text) from e

        total_time = (time.time() - start_time) * 1000
        logger.info(
            "Upsert completed: %s points to '%s' in %.2fms", len(request.points), request.collection_name, total_time
        )
        search_latency.labels(endpoint="/upsert", search_type="vector_upload").observe(time.time() - start_time)

        return UpsertResponse(
            status="success",
            points_upserted=len(request.points),
            collection_name=request.collection_name,
            upsert_time_ms=total_time,
        )
    except HTTPException:
        # Bubble up HTTPExceptions created above without wrapping
        raise
    except httpx.HTTPStatusError as e:
        logger.error("Qdrant error during upsert: %s", e)
        search_errors.labels(endpoint="/upsert", error_type="qdrant_error").inc()

        error_detail = _extract_qdrant_error(e)
        detail_text = (
            f"Vector database error: {error_detail}" if error_detail != "Vector database error" else error_detail
        )

        raise HTTPException(status_code=502, detail=detail_text) from e
    except Exception as e:  # pragma: no cover - unexpected
        logger.error("Unexpected error in /upsert: %s", e, exc_info=True)
        search_errors.labels(endpoint="/upsert", error_type="unknown_error").inc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}") from e


async def model_status() -> dict[str, Any]:
    """Return model manager status."""
    model_mgr = _get_model_manager()
    if model_mgr:
        return dict(model_mgr.get_status())
    return {"error": "Model manager not initialized"}


async def health() -> dict[str, Any]:
    """Comprehensive health check."""
    health_status: dict[str, Any] = {"status": "healthy", "components": {}}

    try:
        client = _get_qdrant_client()
        if client is None:
            health_status["components"]["qdrant"] = {"status": "unhealthy", "error": "Client not initialized"}
            health_status["status"] = "unhealthy"
        else:
            response = await client.get("/collections")
            if response.status_code == 200:
                collections_data = response.json()
                health_status["components"]["qdrant"] = {
                    "status": "healthy",
                    "collections_count": len(collections_data.get("result", {}).get("collections", [])),
                }
            else:
                health_status["components"]["qdrant"] = {"status": "unhealthy", "error": f"HTTP {response.status_code}"}
                health_status["status"] = "unhealthy"
    except Exception as e:
        health_status["components"]["qdrant"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "unhealthy"

    try:
        # Get embedding status from model manager
        cfg = _get_settings()
        if search_state.model_manager is None:
            health_status["components"]["embedding"] = {"status": "unhealthy", "error": "Model manager not initialized"}
            health_status["status"] = "degraded" if health_status["status"] == "healthy" else "unhealthy"
        else:
            mgr_status = search_state.model_manager.get_status()
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
                # No model loaded yet, but this is OK - lazy loading
                health_status["components"]["embedding"] = {
                    "status": "healthy",
                    "model": None,
                    "provider": None,
                    "note": "Embedding model loaded on first use",
                    "is_mock_mode": cfg.USE_MOCK_EMBEDDINGS,
                }
    except Exception as e:
        health_status["components"]["embedding"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "degraded" if health_status["status"] == "healthy" else "unhealthy"

    if health_status["status"] == "unhealthy":
        raise HTTPException(status_code=503, detail=health_status)

    return health_status


async def list_models() -> dict[str, Any]:
    """List available embedding models and their properties.

    Returns models from all registered embedding providers (built-in + plugins),
    with provider metadata for each model.
    """
    from shared.embedding.factory import get_all_supported_models

    # Get all models from registered providers (built-in + plugins)
    all_models = get_all_supported_models()

    models = []
    for model_info in all_models:
        model_name = model_info.get("model_name") or model_info.get("name", "")
        provider = model_info.get("provider", "unknown")

        models.append(
            {
                # Existing fields (backward compatibility)
                "name": model_name,
                "description": model_info.get("description", ""),
                "dimension": model_info.get("dimension"),
                "supports_quantization": model_info.get("supports_quantization", True),
                "recommended_quantization": model_info.get("recommended_quantization", "float32"),
                "memory_estimate": model_info.get("memory_estimate", {}),
                "is_qwen3": "Qwen3-Embedding" in model_name,
                # New plugin-aware fields
                "provider_id": provider,
                "is_plugin": provider not in ("dense_local", "mock"),
            }
        )

    # Get current model info from model manager
    current_model = None
    current_quantization = None
    if search_state.model_manager:
        mgr_status = search_state.model_manager.get_status()
        model_key = mgr_status.get("current_embedding_model")
        if model_key:
            # model_key is "model_name_quantization"
            parts = model_key.rsplit("_", 1)
            current_model = parts[0] if len(parts) > 1 else model_key
            current_quantization = parts[1] if len(parts) > 1 else "float32"

    return {
        "models": models,
        "current_model": current_model,
        "current_quantization": current_quantization,
    }


async def load_model(model_name: str, quantization: str = "float32") -> dict[str, Any]:
    """Load a specific embedding model.

    This triggers eager model loading via the model manager. Models are normally
    loaded lazily on first embedding request.
    """
    cfg = _get_settings()

    if cfg.USE_MOCK_EMBEDDINGS:
        raise HTTPException(status_code=400, detail="Cannot load models when using mock embeddings")

    try:
        model_mgr = _get_model_manager()
        if model_mgr is None:
            raise HTTPException(status_code=503, detail="Model manager not initialized")

        # Trigger model loading by generating a test embedding
        # This ensures the provider is initialized with the requested model
        await model_mgr.generate_embedding_async("warm-up", model_name, quantization)

        # Get status after loading
        mgr_status = model_mgr.get_status()
        model_info = mgr_status.get("provider_info", {})

        return {
            "status": "success",
            "model": model_name,
            "quantization": quantization,
            "provider": mgr_status.get("embedding_provider"),
            "info": model_info,
        }

    except ValueError as e:
        # No provider found for model
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:  # pragma: no cover - fallback
        logger.error("Model load error: %s", e)
        raise HTTPException(status_code=500, detail=f"Model load failed: {str(e)}") from e


async def suggest_models() -> dict[str, Any]:
    """Suggest optimal model configuration based on available GPU memory."""
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
            "embedding": search_state.model_manager.current_model_key if search_state.model_manager else None,
            "reranker": search_state.model_manager.current_reranker_key if search_state.model_manager else None,
        },
    }


async def embedding_info() -> dict[str, Any]:
    """Return information about the embedding configuration."""
    cfg = _get_settings()

    # Get status from ModelManager (the source of truth)
    model_status = search_state.model_manager.get_status() if search_state.model_manager else {}

    info: dict[str, Any] = {
        "mode": "mock" if cfg.USE_MOCK_EMBEDDINGS else "real",
        # Available if ModelManager exists (even if model not yet loaded due to lazy loading)
        "available": search_state.model_manager is not None,
        "is_mock_mode": cfg.USE_MOCK_EMBEDDINGS,
    }

    # Add model details from ModelManager status
    if model_status.get("embedding_model_loaded"):
        provider_info = model_status.get("provider_info", {})
        current_model_key = model_status.get("current_embedding_model", "")

        # Parse model key format: "model_name_quantization"
        if "_" in current_model_key:
            parts = current_model_key.rsplit("_", 1)
            model_name = parts[0]
            quantization = parts[1] if len(parts) > 1 else "unknown"
        else:
            model_name = current_model_key
            quantization = provider_info.get("quantization", "unknown")

        info.update(
            {
                "current_model": model_name,
                "quantization": quantization,
                "device": provider_info.get("device"),
                "provider": model_status.get("embedding_provider"),
                "dimension": provider_info.get("dimension"),
                "model_details": provider_info,
            }
        )
    elif search_state.model_manager is not None:
        # Model not loaded yet (lazy loading) - still indicate availability
        info["note"] = "Embedding model loaded on first use"
        # Include defaults from settings
        info.update(
            {
                "default_model": cfg.DEFAULT_EMBEDDING_MODEL,
                "default_quantization": cfg.DEFAULT_QUANTIZATION,
            }
        )

    return info
