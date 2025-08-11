"""
Tests for query optimization and caching functionality.
"""

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import redis.asyncio as aioredis
from sqlalchemy.ext.asyncio import AsyncSession

from packages.shared.database.repositories.chunk_repository import ChunkRepository
from packages.webui.services.cache_manager import CacheManager, QueryMonitor
from packages.webui.services.chunking_service import ChunkingService


@pytest.fixture()
def mock_redis():
    """Create a mock Redis client."""
    redis_mock = AsyncMock(spec=aioredis.Redis)
    redis_mock.get = AsyncMock(return_value=None)
    redis_mock.setex = AsyncMock(return_value=True)
    redis_mock.delete = AsyncMock(return_value=1)
    redis_mock.scan = AsyncMock(return_value=(0, []))
    return redis_mock


@pytest.fixture()
def cache_manager(mock_redis):
    """Create a CacheManager instance with mock Redis."""
    return CacheManager(mock_redis)


@pytest.fixture()
def mock_db_session():
    """Create a mock database session."""
    return AsyncMock(spec=AsyncSession)


@pytest.fixture()
def mock_collection_repo():
    """Create a mock collection repository."""
    repo = AsyncMock()
    repo.get_by_uuid = AsyncMock(return_value={"id": "test-collection"})
    return repo


@pytest.fixture()
def mock_document_repo():
    """Create a mock document repository."""
    return AsyncMock()


@pytest.fixture()
def chunking_service(mock_db_session, mock_collection_repo, mock_document_repo, mock_redis):
    """Create a ChunkingService instance with mocks."""
    return ChunkingService(
        db_session=mock_db_session,
        collection_repo=mock_collection_repo,
        document_repo=mock_document_repo,
        redis_client=mock_redis,
    )


class TestCacheManager:
    """Test cache manager functionality."""

    @pytest.mark.asyncio()
    async def test_cache_key_generation(self, cache_manager):
        """Test cache key generation is deterministic."""
        params1 = {"collection_id": "123", "user_id": 456}
        params2 = {"user_id": 456, "collection_id": "123"}  # Different order

        key1 = cache_manager._generate_cache_key("test", params1)
        key2 = cache_manager._generate_cache_key("test", params2)

        assert key1 == key2  # Should be same despite param order
        assert key1.startswith("cache:test:")

    @pytest.mark.asyncio()
    async def test_cache_get_hit(self, cache_manager, mock_redis):
        """Test cache hit increments counter."""
        cached_data = {"result": "test"}
        mock_redis.get.return_value = json.dumps(cached_data)

        result = await cache_manager.get("test_key")

        assert result == cached_data
        assert cache_manager.hits == 1
        assert cache_manager.misses == 0

    @pytest.mark.asyncio()
    async def test_cache_get_miss(self, cache_manager, mock_redis):
        """Test cache miss increments counter."""
        mock_redis.get.return_value = None

        result = await cache_manager.get("test_key")

        assert result is None
        assert cache_manager.hits == 0
        assert cache_manager.misses == 1

    @pytest.mark.asyncio()
    async def test_cache_set(self, cache_manager, mock_redis):
        """Test setting cache value."""
        data = {"result": "test"}

        await cache_manager.set("test_key", data, ttl=300)

        mock_redis.setex.assert_called_once_with("test_key", 300, json.dumps(data))

    @pytest.mark.asyncio()
    async def test_invalidate_collection(self, cache_manager, mock_redis):
        """Test collection cache invalidation."""
        collection_id = "test-collection"

        await cache_manager.invalidate_collection(collection_id)

        # Should scan and delete matching keys
        assert mock_redis.scan.called

    @pytest.mark.asyncio()
    async def test_cache_stats(self, cache_manager):
        """Test cache statistics calculation."""
        cache_manager.hits = 80
        cache_manager.misses = 20

        stats = cache_manager.get_stats()

        assert stats["hits"] == 80
        assert stats["misses"] == 20
        assert stats["total_requests"] == 100
        assert stats["hit_rate"] == 80.0


class TestQueryOptimization:
    """Test query optimization in chunking service."""

    @pytest.mark.asyncio()
    async def test_statistics_query_optimization(self, chunking_service, mock_db_session):
        """Test that statistics use aggregation queries instead of loading all records."""
        # Mock the aggregation query result
        mock_stats = MagicMock()
        mock_stats.total_operations = 100
        mock_stats.completed_operations = 80
        mock_stats.failed_operations = 10
        mock_stats.processing_operations = 10
        mock_stats.avg_processing_time = 1.5
        mock_stats.last_operation_at = datetime.now(UTC)
        mock_stats.first_operation_at = datetime.now(UTC)

        mock_strategy = MagicMock()
        mock_strategy.strategy = "recursive"

        # Mock execute results
        mock_db_session.execute = AsyncMock()
        mock_db_session.execute.side_effect = [
            MagicMock(one=lambda: mock_stats),  # Stats query
            MagicMock(one_or_none=lambda: mock_strategy),  # Strategy query
        ]

        result = await chunking_service.get_chunking_statistics("test-collection")

        # Verify results
        assert result["total_operations"] == 100
        assert result["completed_operations"] == 80
        assert result["failed_operations"] == 10
        assert result["avg_processing_time"] == 1.5
        assert result["latest_strategy"] == "recursive"

        # Verify we made exactly 2 queries (stats + strategy)
        assert mock_db_session.execute.call_count == 2

    @pytest.mark.asyncio()
    async def test_statistics_caching(self, chunking_service, mock_redis, mock_db_session):
        """Test that statistics are cached after computation."""
        # Setup mock query results
        mock_stats = MagicMock()
        mock_stats.total_operations = 50
        mock_stats.completed_operations = 40
        mock_stats.failed_operations = 5
        mock_stats.processing_operations = 5
        mock_stats.avg_processing_time = 2.0
        mock_stats.last_operation_at = datetime.now(UTC)
        mock_stats.first_operation_at = datetime.now(UTC)

        mock_db_session.execute = AsyncMock()
        mock_db_session.execute.side_effect = [MagicMock(one=lambda: mock_stats), MagicMock(one_or_none=lambda: None)]

        # First call - should query database
        result1 = await chunking_service.get_chunking_statistics("test-collection")

        # Verify cache was set
        assert mock_redis.setex.called
        cache_call_args = mock_redis.setex.call_args
        assert cache_call_args[0][1] == 60  # TTL should be 60 seconds

        # Verify result
        assert result1["total_operations"] == 50

    @pytest.mark.asyncio()
    async def test_cache_invalidation_on_operation_create(self, chunking_service, mock_redis, mock_db_session):
        """Test that cache is invalidated when new operation is created."""
        collection_id = "test-collection"

        # Mock successful operation creation
        mock_db_session.add = MagicMock()
        mock_db_session.commit = AsyncMock()

        await chunking_service.start_chunking_operation(
            collection_id=collection_id, strategy="recursive", config={"chunk_size": 1000}, user_id=1
        )

        # Verify cache invalidation was called
        assert mock_redis.scan.called


class TestChunkRepository:
    """Test chunk repository batch operations."""

    @pytest.mark.asyncio()
    async def test_get_chunks_batch(self, mock_db_session):
        """Test batch fetching of chunks."""
        repo = ChunkRepository(mock_db_session)

        # Mock query result
        mock_chunks = [MagicMock() for _ in range(5)]
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_chunks
        mock_db_session.execute = AsyncMock(return_value=mock_result)

        # Use valid UUIDs
        collection_id = "550e8400-e29b-41d4-a716-446655440000"
        doc_ids = [
            "550e8400-e29b-41d4-a716-446655440001",
            "550e8400-e29b-41d4-a716-446655440002",
            "550e8400-e29b-41d4-a716-446655440003",
        ]

        result = await repo.get_chunks_batch(collection_id=collection_id, document_ids=doc_ids, limit=100)

        assert len(result) == 5
        # Verify IN clause was used (check query construction)
        assert mock_db_session.execute.called

    @pytest.mark.asyncio()
    async def test_get_chunks_paginated(self, mock_db_session):
        """Test paginated chunk retrieval with window function."""
        repo = ChunkRepository(mock_db_session)

        # Mock paginated result with total count
        mock_rows = [(MagicMock(), 100) for _ in range(10)]  # 10 chunks, 100 total
        mock_result = MagicMock()
        mock_result.all.return_value = mock_rows
        mock_db_session.execute = AsyncMock(return_value=mock_result)

        # Use valid UUID
        collection_id = "550e8400-e29b-41d4-a716-446655440000"

        chunks, total = await repo.get_chunks_paginated(collection_id=collection_id, page=1, page_size=10)

        assert len(chunks) == 10
        assert total == 100

    @pytest.mark.asyncio()
    async def test_get_chunk_statistics_optimized(self, mock_db_session):
        """Test optimized statistics query using aggregation."""
        repo = ChunkRepository(mock_db_session)

        # Mock aggregated statistics
        mock_stats = MagicMock()
        mock_stats.total_chunks = 1000
        mock_stats.avg_chunk_size = 512.5
        mock_stats.min_chunk_size = 100
        mock_stats.max_chunk_size = 1024
        mock_stats.unique_documents = 50
        mock_stats.first_chunk_created = datetime.now(UTC)
        mock_stats.last_chunk_created = datetime.now(UTC)

        mock_result = MagicMock()
        mock_result.one.return_value = mock_stats
        mock_db_session.execute = AsyncMock(return_value=mock_result)

        # Use valid UUID
        collection_id = "550e8400-e29b-41d4-a716-446655440000"

        stats = await repo.get_chunk_statistics_optimized(collection_id)

        assert stats["total_chunks"] == 1000
        assert stats["avg_chunk_size"] == 512.5
        assert stats["min_chunk_size"] == 100
        assert stats["max_chunk_size"] == 1024
        assert stats["unique_documents"] == 50

        # Verify single query was made
        assert mock_db_session.execute.call_count == 1


class TestQueryMonitor:
    """Test query performance monitoring."""

    @pytest.mark.asyncio()
    async def test_monitor_decorator(self):
        """Test query monitoring decorator."""

        @QueryMonitor.monitor("test_query")
        async def slow_query():
            import asyncio

            await asyncio.sleep(0.01)  # Simulate slow query
            return "result"

        with patch("packages.webui.services.cache_manager.logger") as mock_logger:
            result = await slow_query()

            assert result == "result"
            # Check that execution was logged
            assert mock_logger.debug.called

    @pytest.mark.asyncio()
    async def test_slow_query_warning(self):
        """Test that slow queries trigger warnings."""

        @QueryMonitor.monitor("slow_test_query")
        async def very_slow_query():
            import asyncio

            await asyncio.sleep(1.1)  # Exceed 1 second threshold
            return "slow_result"

        with patch("packages.webui.services.cache_manager.logger") as mock_logger:
            result = await very_slow_query()

            assert result == "slow_result"
            # Check that warning was logged for slow query
            assert mock_logger.warning.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
