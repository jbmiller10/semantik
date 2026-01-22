"""Helpers for fetching Qdrant payloads and normalizing filters."""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from vecpipe.search.errors import response_json
from vecpipe.search.metrics import payload_fetch_latency, qdrant_ad_hoc_client_total

logger = logging.getLogger(__name__)


def normalize_qdrant_filter(filters: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize filter to standard form with must/should/must_not keys."""
    if filters is None:
        return {"must": []}
    if any(k in filters for k in ("must", "should", "must_not")):
        return filters
    return {"must": [filters]}


def merge_filters(base: dict[str, Any], additional_must: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge additional must conditions into base filter, avoiding invalid nesting."""
    result = dict(base)
    result["must"] = additional_must + list(result.get("must") or [])
    return result


async def fetch_payloads_for_chunk_ids(
    *,
    collection_name: str,
    chunk_ids: list[str],
    cfg: Any,
    qdrant_http: httpx.AsyncClient | None,
    filters: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Fetch full payloads for chunk IDs from the dense collection.

    Sparse collections may store only lightweight payloads (or none at all). For sparse and
    hybrid search modes we still need dense payloads to build SearchResult objects.
    """
    if not chunk_ids:
        return {}

    client = qdrant_http
    created_client = False

    if client is None:
        headers: dict[str, str] = {}
        if cfg.QDRANT_API_KEY:
            headers["api-key"] = cfg.QDRANT_API_KEY
        client = httpx.AsyncClient(
            base_url=f"http://{cfg.QDRANT_HOST}:{cfg.QDRANT_PORT}",
            timeout=httpx.Timeout(60.0),
            headers=headers,
        )
        qdrant_ad_hoc_client_total.labels(location="payload_fetch").inc()
        created_client = True

    try:
        chunk_id_condition = {"key": "chunk_id", "match": {"any": chunk_ids}}
        normalized = normalize_qdrant_filter(filters)
        scroll_filter = merge_filters(normalized, [chunk_id_condition])

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
        fetch_start = time.time()
        response = await client.post(f"/collections/{collection_name}/points/scroll", json=fetch_request)

        if response.status_code != 200:
            error_text = response.text
            logger.error(
                "Failed to fetch payloads from %s: status=%d, error=%s",
                collection_name,
                response.status_code,
                error_text[:500] if error_text else "unknown",
            )
            payload_fetch_latency.observe(time.time() - fetch_start)
            return {}

        payload_map: dict[str, dict[str, Any]] = {}
        result = (await response_json(response)).get("result", {})
        points = result.get("points", []) if isinstance(result, dict) else []
        for point in points:
            if not isinstance(point, dict):
                continue
            payload = point.get("payload", {})
            if not isinstance(payload, dict):
                continue
            chunk_id = payload.get("chunk_id")
            if chunk_id:
                payload_map[str(chunk_id)] = payload

        payload_fetch_latency.observe(time.time() - fetch_start)
        return payload_map
    finally:
        if created_client and client is not None:
            await client.aclose()
