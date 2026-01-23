"""Collection resolution and cached metadata helpers for VecPipe search."""

from __future__ import annotations

import logging
import time
from typing import Any, cast

import httpx
from fastapi import HTTPException

from vecpipe.search.cache import (
    get_collection_info as get_cached_collection_info,
    get_collection_metadata,
    is_cache_miss,
    set_collection_info as set_cached_collection_info,
    set_collection_metadata,
)
from vecpipe.search.errors import maybe_raise_for_status, response_json
from vecpipe.search.metrics import collection_metadata_fetch_latency, qdrant_ad_hoc_client_total

logger = logging.getLogger(__name__)


async def lookup_collection_from_operation(operation_uuid: str) -> str | None:
    """Look up collection's vector_store_name from an operation UUID."""
    from shared.database.database import ensure_async_sessionmaker
    from shared.database.repositories.operation_repository import OperationRepository

    try:
        session_factory = await ensure_async_sessionmaker()
        async with session_factory() as session:
            repo = OperationRepository(session)
            operation = await repo.get_by_uuid(operation_uuid)
            if operation and operation.collection:
                return cast(str, operation.collection.vector_store_name)
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("Failed to lookup operation %s: %s", operation_uuid, exc)

    return None


async def resolve_collection_name(
    request_collection: str | None,
    operation_uuid: str | None,
    default_collection: str,
) -> str:
    """Resolve collection name using priority: explicit > operation_uuid lookup > default."""
    if request_collection:
        return request_collection

    if operation_uuid:
        collection_name = await lookup_collection_from_operation(operation_uuid)
        if collection_name:
            return collection_name
        raise HTTPException(
            status_code=404,
            detail=f"Operation '{operation_uuid}' not found or has no associated collection",
        )

    return default_collection


async def get_collection_info(
    *,
    collection_name: str,
    cfg: Any,
    qdrant_http: httpx.AsyncClient | None,
) -> tuple[int, dict[str, Any] | None]:
    """Fetch collection vector dimension and optional info, using a TTL cache."""
    cached = get_cached_collection_info(collection_name)
    if cached is not None:
        return cast(tuple[int, dict[str, Any] | None], cached)

    vector_dim = 1024
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
        qdrant_ad_hoc_client_total.labels(location="collection_info").inc()
        created_client = True

    try:
        response = await client.get(f"/collections/{collection_name}")
        await maybe_raise_for_status(response)
        info = (await response_json(response))["result"]
        if "config" in info and "params" in info["config"]:
            vector_dim = info["config"]["params"]["vectors"]["size"]
        set_cached_collection_info(collection_name, vector_dim, info)
        return vector_dim, info
    except Exception as exc:  # pragma: no cover - warning path
        logger.warning("Could not get collection info for %s, using default dimension: %s", collection_name, exc)
        return vector_dim, None
    finally:
        if created_client and client is not None:
            await client.aclose()


async def get_cached_collection_metadata(
    *,
    collection_name: str,
    cfg: Any,
    qdrant_sdk: Any | None,
) -> dict[str, Any] | None:
    """Fetch collection metadata with caching, preferring a shared SDK client."""
    cached = get_collection_metadata(collection_name)
    if not is_cache_miss(cached):
        return cast(dict[str, Any] | None, cached)

    metadata_fetch_start = time.time()
    try:
        from shared.database.collection_metadata import get_collection_metadata_async

        sdk_client = qdrant_sdk
        created_client = False
        if sdk_client is None:
            from qdrant_client import AsyncQdrantClient

            sdk_client = AsyncQdrantClient(
                url=f"http://{cfg.QDRANT_HOST}:{cfg.QDRANT_PORT}",
                api_key=cfg.QDRANT_API_KEY,
            )
            qdrant_ad_hoc_client_total.labels(location="metadata_fetch").inc()
            created_client = True

        try:
            metadata = await get_collection_metadata_async(sdk_client, collection_name)
            set_collection_metadata(collection_name, metadata)
            collection_metadata_fetch_latency.observe(time.time() - metadata_fetch_start)
            return cast(dict[str, Any] | None, metadata)
        finally:
            if created_client and sdk_client is not None:
                await sdk_client.close()
    except Exception as exc:  # pragma: no cover - best effort path
        logger.warning("Could not get collection metadata: %s", exc)
        collection_metadata_fetch_latency.observe(time.time() - metadata_fetch_start)
        return None
