from datetime import UTC, datetime

import pytest

from webui.services.chunking.cache import ChunkingCache


@pytest.mark.asyncio()
async def test_clear_preview_by_id_removes_id_and_primary_keys(fake_redis_client):
    cache = ChunkingCache(redis_client=fake_redis_client)
    preview_id = "preview-123"
    content_hash = "content-hash-abc"
    strategy = "recursive"
    config = {"chunk_size": 100}
    preview_payload = {
        "preview_id": preview_id,
        "strategy": strategy,
        "config": config,
        "chunks": [],
        "total_chunks": 0,
        "performance_metrics": {},
        "processing_time_ms": 12,
        "expires_at": datetime.now(UTC).isoformat(),
        "correlation_id": "corr-1",
    }

    cache_key = await cache.cache_preview(
        content_hash,
        strategy,
        config,
        preview_payload,
        preview_id=preview_id,
        ttl=60,
    )

    alias_key = f"{cache.PREVIEW_CACHE_PREFIX}:id:{preview_id}"
    assert await fake_redis_client.exists(alias_key) == 1
    assert await fake_redis_client.exists(cache_key) == 1

    deleted = await cache.clear_preview_by_id(preview_id)

    assert deleted >= 1
    assert await fake_redis_client.exists(alias_key) == 0
    assert await fake_redis_client.exists(cache_key) == 0
