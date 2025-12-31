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


@pytest.mark.asyncio()
async def test_delete_keys_accepts_string_count():
    class DummyRedis:
        async def delete(self, *keys):
            return "2"

    cache = ChunkingCache(redis_client=DummyRedis())

    deleted = await cache._delete_keys(["alpha", "beta"])

    assert deleted == 2


class TestChunkingCacheDisabled:
    """Tests for cache disabled scenarios."""

    @pytest.mark.asyncio()
    async def test_get_cached_preview_returns_none_when_disabled(self):
        """Test get_cached_preview() returns None when cache is disabled."""
        cache = ChunkingCache(redis_client=None)

        result = await cache.get_cached_preview(
            content_hash="abc123",
            strategy="recursive",
            config={},
        )

        assert result is None

    @pytest.mark.asyncio()
    async def test_cache_preview_returns_empty_string_when_disabled(self):
        """Test cache_preview() returns empty string when cache is disabled."""
        cache = ChunkingCache(redis_client=None)

        result = await cache.cache_preview(
            content_hash="abc123",
            strategy="recursive",
            config={},
            preview_data={"chunks": []},
        )

        assert result == ""

    @pytest.mark.asyncio()
    async def test_clear_cache_returns_zero_when_disabled(self):
        """Test clear_cache() returns 0 when cache is disabled."""
        cache = ChunkingCache(redis_client=None)

        result = await cache.clear_cache()

        assert result == 0

    @pytest.mark.asyncio()
    async def test_get_cached_by_id_returns_none_when_disabled(self):
        """Test get_cached_by_id() returns None when disabled."""
        cache = ChunkingCache(redis_client=None)

        result = await cache.get_cached_by_id("test-id")

        assert result is None

    @pytest.mark.asyncio()
    async def test_cache_with_id_returns_empty_string_when_disabled(self):
        """Test cache_with_id() returns empty string when disabled."""
        cache = ChunkingCache(redis_client=None)

        result = await cache.cache_with_id({"data": "test"})

        assert result == ""

    @pytest.mark.asyncio()
    async def test_get_usage_metrics_returns_empty_when_disabled(self):
        """Test get_usage_metrics() returns empty dict when cache is disabled."""
        cache = ChunkingCache(redis_client=None)

        result = await cache.get_usage_metrics()

        assert result == {}


class TestChunkingCacheErrorHandling:
    """Tests for error handling in cache operations."""

    @pytest.mark.asyncio()
    async def test_clear_preview_by_id_handles_missing_cache_key(self, fake_redis_client):
        """Test clear_preview_by_id() handles case where cache_key is not in cached data."""
        cache = ChunkingCache(redis_client=fake_redis_client)

        # Set up cache entry without cache_key field
        preview_id = "preview-no-key"
        alias_key = f"{cache.PREVIEW_CACHE_PREFIX}:id:{preview_id}"

        import json

        await fake_redis_client.set(alias_key, json.dumps({"preview_id": preview_id}))

        deleted = await cache.clear_preview_by_id(preview_id)

        # Should still work, just delete the alias key
        assert deleted >= 1

    @pytest.mark.asyncio()
    async def test_track_usage_handles_redis_errors(self):
        """Test track_usage() gracefully handles Redis errors without raising."""
        from unittest.mock import AsyncMock

        mock_redis = AsyncMock()
        mock_redis.hincrby = AsyncMock(side_effect=Exception("Redis error"))

        cache = ChunkingCache(redis_client=mock_redis)

        # Should not raise
        await cache.track_usage(user_id=1, strategy="recursive", cache_hit=True)

    @pytest.mark.asyncio()
    async def test_get_cached_preview_handles_json_decode_error(self):
        """Test get_cached_preview() handles malformed JSON."""
        from unittest.mock import AsyncMock

        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value="not-valid-json{")

        cache = ChunkingCache(redis_client=mock_redis)

        result = await cache.get_cached_preview(
            content_hash="abc",
            strategy="test",
            config={},
        )

        # Should return None on JSON error
        assert result is None


class TestChunkingCacheKeyGeneration:
    """Tests for cache key generation."""

    def test_generate_preview_key_is_deterministic(self):
        """Test _generate_preview_key produces deterministic keys."""
        cache = ChunkingCache(redis_client=None)

        key1 = cache._generate_preview_key("hash1", "recursive", {"size": 100})
        key2 = cache._generate_preview_key("hash1", "recursive", {"size": 100})

        assert key1 == key2

    def test_generate_preview_key_varies_by_content(self):
        """Test _generate_preview_key produces different keys for different content."""
        cache = ChunkingCache(redis_client=None)

        key1 = cache._generate_preview_key("hash1", "recursive", {"size": 100})
        key2 = cache._generate_preview_key("hash2", "recursive", {"size": 100})

        assert key1 != key2

    def test_generate_content_hash(self):
        """Test generate_content_hash() produces consistent hashes."""
        cache = ChunkingCache(redis_client=None)

        hash1 = cache.generate_content_hash("test content")
        hash2 = cache.generate_content_hash("test content")

        assert hash1 == hash2
        assert len(hash1) == 16  # SHA256 truncated to 16 chars
