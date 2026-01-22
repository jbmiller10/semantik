"""Dense embedding and Qdrant vector search helpers."""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any

import httpx

from vecpipe.search.errors import maybe_raise_for_status, response_json
from vecpipe.search.metrics import embedding_generation_latency

logger = logging.getLogger(__name__)


def generate_mock_embedding(text: str, vector_dim: int | None = None) -> list[float]:
    """Generate a deterministic mock embedding for testing."""
    if vector_dim is None:
        vector_dim = 1024

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


async def generate_embedding(
    *,
    cfg: Any,
    model_manager: Any,
    text: str,
    model_name: str,
    quantization: str,
    instruction: str | None = None,
    mode: str | None = None,
    vector_dim: int | None = None,
) -> list[float]:
    """Generate an embedding using the runtime model manager or mock embeddings."""
    if cfg.USE_MOCK_EMBEDDINGS:
        return generate_mock_embedding(text, vector_dim)

    if model_manager is None:
        raise RuntimeError("Model manager not initialized")

    start_time = time.time()
    embedding = await model_manager.generate_embedding_async(text, model_name, quantization, instruction, mode=mode)
    embedding_generation_latency.observe(time.time() - start_time)

    if embedding is None:
        raise RuntimeError(f"Failed to generate embedding for text: {text[:100]}...")

    return list(embedding)


async def search_dense_qdrant(
    *,
    collection_name: str,
    query_vector: list[float],
    limit: int,
    qdrant_http: httpx.AsyncClient,
    qdrant_sdk: Any,
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Search Qdrant for dense vectors, preferring SDK with REST fallback.

    REST is required for filtered search in this codebase (kept as fallback).
    """
    if filters:
        search_request = {
            "vector": query_vector,
            "limit": limit,
            "with_payload": True,
            "with_vector": False,
            "filter": filters,
        }
        response = await qdrant_http.post(f"/collections/{collection_name}/points/search", json=search_request)
        await maybe_raise_for_status(response)
        return list((await response_json(response))["result"])

    try:
        results = await qdrant_sdk.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=True,
        )
        converted: list[dict[str, Any]] = []
        for point in results:
            if isinstance(point, dict):
                converted.append(
                    {
                        "id": str(point.get("id", "")),
                        "score": float(point.get("score", 0.0)),
                        "payload": point.get("payload") or {},
                    }
                )
            else:
                converted.append(
                    {
                        "id": str(getattr(point, "id", "")),
                        "score": float(getattr(point, "score", 0.0)),
                        "payload": getattr(point, "payload", {}) or {},
                    }
                )
        return converted
    except Exception as exc:  # pragma: no cover - best effort fallback
        logger.warning("SDK dense search failed; falling back to REST: %s", exc, exc_info=True)
        search_request = {
            "vector": query_vector,
            "limit": limit,
            "with_payload": True,
            "with_vector": False,
        }
        response = await qdrant_http.post(f"/collections/{collection_name}/points/search", json=search_request)
        await maybe_raise_for_status(response)
        return list((await response_json(response))["result"])

