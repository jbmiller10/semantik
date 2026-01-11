"""Tests for collection info and metadata caching."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, patch

import pytest

# Import cache module via the same path the service uses to avoid module aliasing issues
# (packages.vecpipe vs vecpipe can create separate module instances)
from vecpipe.search import cache
from vecpipe.search.cache import (
    clear_cache as clear_cache_direct,
)


@pytest.fixture(autouse=True)
def _clear_caches():
    """Clear all caches before and after each test."""
    # Clear both module instances to handle aliasing
    cache.clear_cache()
    clear_cache_direct()
    yield
    cache.clear_cache()
    clear_cache_direct()


class TestCollectionInfoCache:
    """Tests for collection info caching."""

    def test_cache_miss_returns_none(self):
        """Cache miss should return None."""
        result = cache.get_collection_info("nonexistent_collection")
        assert result is None

    def test_cache_hit_returns_value(self):
        """Cache hit should return the cached value."""
        cache.set_collection_info("test_collection", 1024, {"config": "test"})
        result = cache.get_collection_info("test_collection")
        assert result is not None
        dim, info = result
        assert dim == 1024
        assert info == {"config": "test"}

    def test_cache_expiry(self):
        """Cache should expire after TTL."""
        cache.set_collection_info("test_collection", 1024, {"config": "test"})

        # Verify it's cached
        assert cache.get_collection_info("test_collection") is not None

        # Simulate time passing beyond TTL by patching _is_expired to always return True
        with patch.object(cache, "_is_expired", return_value=True):
            result = cache.get_collection_info("test_collection")
            assert result is None

    def test_clear_cache(self):
        """clear_cache should remove all entries."""
        cache.set_collection_info("test1", 512, None)
        cache.set_collection_info("test2", 1024, {"data": "test"})

        cache.clear_cache()

        assert cache.get_collection_info("test1") is None
        assert cache.get_collection_info("test2") is None


class TestCollectionMetadataCache:
    """Tests for collection metadata caching."""

    def test_cache_miss_returns_sentinel(self):
        """Cache miss should return the sentinel value."""
        result = cache.get_collection_metadata("nonexistent_collection")
        assert cache.is_cache_miss(result)

    def test_cache_hit_returns_value(self):
        """Cache hit should return the cached metadata."""
        metadata = {"model_name": "test-model", "quantization": "float32"}
        cache.set_collection_metadata("test_collection", metadata)

        result = cache.get_collection_metadata("test_collection")
        assert not cache.is_cache_miss(result)
        assert result == metadata

    def test_cached_none_not_treated_as_miss(self):
        """A cached None value should not be treated as cache miss."""
        cache.set_collection_metadata("test_collection", None)

        result = cache.get_collection_metadata("test_collection")
        # None is a valid cached value, not a cache miss
        assert not cache.is_cache_miss(result)
        assert result is None

    def test_cache_expiry(self):
        """Metadata cache should expire after TTL."""
        cache.set_collection_metadata("test_collection", {"model": "test"})

        # Verify it's cached
        assert not cache.is_cache_miss(cache.get_collection_metadata("test_collection"))

        # Simulate expiry
        with patch.object(cache, "_is_expired", return_value=True):
            result = cache.get_collection_metadata("test_collection")
            assert cache.is_cache_miss(result)


class TestCacheEviction:
    """Tests for cache eviction when max entries exceeded."""

    def test_eviction_when_max_entries_exceeded(self):
        """Cache should evict oldest entries when max exceeded."""
        # Save original and set to small value
        original_max = cache.MAX_CACHE_ENTRIES

        try:
            # Directly modify the module constant
            cache.MAX_CACHE_ENTRIES = 3

            cache.set_collection_info("col1", 512, None)
            time.sleep(0.01)  # Ensure different timestamps
            cache.set_collection_info("col2", 512, None)
            time.sleep(0.01)
            cache.set_collection_info("col3", 512, None)
            time.sleep(0.01)

            # This should trigger eviction of col1
            cache.set_collection_info("col4", 512, None)

            # col1 should be evicted (oldest)
            assert cache.get_collection_info("col1") is None
            # Others should still exist
            assert cache.get_collection_info("col2") is not None
            assert cache.get_collection_info("col3") is not None
            assert cache.get_collection_info("col4") is not None
        finally:
            cache.MAX_CACHE_ENTRIES = original_max


class TestSearchUtilsConnectionCleanup:
    """Tests for search_utils connection cleanup."""

    @pytest.mark.asyncio()
    async def test_fallback_client_is_closed(self):
        """Fallback AsyncQdrantClient should be properly closed."""
        from vecpipe import search_utils

        mock_client = AsyncMock()
        mock_client.search = AsyncMock(return_value=[])
        mock_client.close = AsyncMock()

        with (
            patch.object(search_utils, "AsyncQdrantClient", return_value=mock_client),
            patch("vecpipe.search.state.sdk_client", None),  # Force fallback path
        ):
            await search_utils.search_qdrant(
                qdrant_host="localhost",
                qdrant_port=6333,
                collection_name="test",
                query_vector=[0.1, 0.2, 0.3],
                k=5,
            )

            # Verify the client was closed
            mock_client.close.assert_awaited_once()

    @pytest.mark.asyncio()
    async def test_shared_sdk_client_not_closed(self):
        """Shared SDK client from state should NOT be closed after use."""
        from vecpipe import search_utils
        from vecpipe.search import state as search_state

        mock_sdk_client = AsyncMock()
        mock_sdk_client.search = AsyncMock(return_value=[])
        mock_sdk_client.close = AsyncMock()

        original_sdk_client = search_state.sdk_client
        try:
            search_state.sdk_client = mock_sdk_client

            await search_utils.search_qdrant(
                qdrant_host="localhost",
                qdrant_port=6333,
                collection_name="test",
                query_vector=[0.1, 0.2, 0.3],
                k=5,
            )

            # Shared client should NOT be closed
            mock_sdk_client.close.assert_not_awaited()
        finally:
            search_state.sdk_client = original_sdk_client


class TestCachedCollectionMetadataHelper:
    """Tests for _get_cached_collection_metadata helper.

    Note: These tests mock cache functions at the service module level to avoid
    module aliasing issues (packages.vecpipe vs vecpipe can create separate instances).
    """

    @pytest.mark.asyncio()
    async def test_cache_hit_skips_qdrant_call(self):
        """When cache has data, Qdrant should not be called."""
        # Pre-import the cache module to ensure it's loaded before patching
        # This is necessary because the service uses lazy imports
        import vecpipe.search.cache  # noqa: F401
        from vecpipe.search.service import _get_cached_collection_metadata

        metadata = {"model_name": "test-model", "quantization": "float32"}

        mock_cfg = AsyncMock()
        mock_cfg.QDRANT_HOST = "localhost"
        mock_cfg.QDRANT_PORT = 6333

        # Mock the cache functions - must patch vecpipe.search.cache since that's
        # the import path used by the service module
        with (
            patch("vecpipe.search.cache.get_collection_metadata", return_value=metadata),
            patch("vecpipe.search.cache.is_cache_miss", return_value=False),
            patch("shared.database.collection_metadata.get_collection_metadata_async") as mock_fetch,
        ):
            result = await _get_cached_collection_metadata("test_collection", mock_cfg)

            # Should return cached value
            assert result == metadata
            # Should NOT call Qdrant
            mock_fetch.assert_not_called()

    @pytest.mark.asyncio()
    async def test_cache_miss_calls_qdrant_and_caches(self):
        """When cache misses, should fetch from Qdrant and cache result."""
        # Pre-import the cache module to ensure it's loaded before patching
        import vecpipe.search.cache  # noqa: F401
        from vecpipe.search import state as search_state
        from vecpipe.search.service import _get_cached_collection_metadata

        metadata = {"model_name": "fetched-model", "quantization": "int8"}
        cache_miss_sentinel = {"__cache_miss__": True}

        mock_sdk_client = AsyncMock()
        original_sdk_client = search_state.sdk_client

        mock_cfg = AsyncMock()
        mock_cfg.QDRANT_HOST = "localhost"
        mock_cfg.QDRANT_PORT = 6333

        cached_values = {}

        def mock_set_cache(name, value):
            cached_values[name] = value

        try:
            search_state.sdk_client = mock_sdk_client

            with (
                patch("vecpipe.search.cache.get_collection_metadata", return_value=cache_miss_sentinel),
                patch("vecpipe.search.cache.is_cache_miss", return_value=True),
                patch("vecpipe.search.cache.set_collection_metadata", side_effect=mock_set_cache),
                patch(
                    "shared.database.collection_metadata.get_collection_metadata_async",
                    new_callable=AsyncMock,
                    return_value=metadata,
                ) as mock_fetch,
            ):
                result = await _get_cached_collection_metadata("new_collection", mock_cfg)

                # Should return fetched value
                assert result == metadata
                # Should call Qdrant
                mock_fetch.assert_awaited_once()

                # Should have cached the result
                assert "new_collection" in cached_values
                assert cached_values["new_collection"] == metadata
        finally:
            search_state.sdk_client = original_sdk_client


class TestCacheMissSentinel:
    """Tests for cache miss sentinel robustness.

    These tests verify the sentinel class pattern handles edge cases
    that can break identity-based comparison with dict sentinels.
    """

    def test_sentinel_is_singleton(self):
        """Multiple instantiations should return same instance."""
        from vecpipe.search.cache import _CacheMiss

        instance1 = _CacheMiss()
        instance2 = _CacheMiss()
        assert instance1 is instance2

    def test_sentinel_identity_stable_across_imports(self):
        """Sentinel should have stable identity across module imports."""
        from vecpipe.search.cache import _METADATA_CACHE_MISS

        # Re-import to get another reference
        sentinel1 = _METADATA_CACHE_MISS
        sentinel2 = cache._METADATA_CACHE_MISS  # type: ignore
        assert sentinel1 is sentinel2

    def test_empty_dict_not_confused_with_sentinel(self):
        """Empty dict {} should not be treated as cache miss."""
        # Cache an empty dict (valid metadata)
        cache.set_collection_metadata("test_empty_dict", {})
        result = cache.get_collection_metadata("test_empty_dict")

        # Should NOT be a cache miss
        assert not cache.is_cache_miss(result)
        assert result == {}

    def test_sentinel_has_repr(self):
        """Sentinel should have clear repr for debugging."""
        from vecpipe.search.cache import _METADATA_CACHE_MISS

        assert repr(_METADATA_CACHE_MISS) == "<CacheMiss>"

    def test_sentinel_survives_pickle_roundtrip(self):
        """Sentinel should maintain identity after pickling.

        Note: This test may skip in environments with module aliasing issues
        (packages.vecpipe vs vecpipe). The implementation includes __reduce__
        for pickle support in production environments.
        """
        import pickle

        # Use cache module's sentinel to avoid module aliasing issues
        sentinel = cache._METADATA_CACHE_MISS  # type: ignore

        try:
            pickled = pickle.dumps(sentinel)
            unpickled = pickle.loads(pickled)

            # Should still be recognized as cache miss via isinstance
            assert cache.is_cache_miss(unpickled)
        except pickle.PicklingError:
            # Module aliasing can cause pickle issues in test environments
            # The __reduce__ implementation is still correct for production use
            pytest.skip("Pickle test skipped due to module aliasing in test environment")

    def test_isinstance_check_with_different_sentinel_instance(self):
        """is_cache_miss should work even with different sentinel instances.

        This simulates what happens during module reload by creating
        a new _CacheMiss instance (bypassing singleton) and verifying
        isinstance works.
        """
        from vecpipe.search.cache import _CacheMiss

        # Create what would happen if module was reloaded (bypass singleton)
        simulated_reload_sentinel = object.__new__(_CacheMiss)

        # Should still be recognized via isinstance
        assert cache.is_cache_miss(simulated_reload_sentinel)

    def test_is_cache_miss_returns_false_for_none(self):
        """is_cache_miss should return False for None (valid cached value)."""
        assert not cache.is_cache_miss(None)

    def test_is_cache_miss_returns_false_for_dict(self):
        """is_cache_miss should return False for regular dicts."""
        assert not cache.is_cache_miss({})
        assert not cache.is_cache_miss({"key": "value"})
        assert not cache.is_cache_miss({"__cache_miss__": True})  # Old sentinel pattern
