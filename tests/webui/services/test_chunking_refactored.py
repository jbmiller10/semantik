"""
Tests for the refactored chunking services architecture.

This test suite validates the new focused services and ensures
backward compatibility through the adapter.
"""

from unittest.mock import AsyncMock

import pytest

from packages.shared.chunking.infrastructure.exceptions import ValidationError
from packages.webui.services.chunking import (
    ChunkingCache,
    ChunkingConfigManager,
    ChunkingMetrics,
    ChunkingOrchestrator,
    ChunkingProcessor,
    ChunkingServiceAdapter,
    ChunkingValidator,
)


@pytest.fixture()
def mock_redis():
    """Create a mock Redis client."""
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.setex = AsyncMock(return_value=True)
    redis.delete = AsyncMock(return_value=1)
    redis.scan_iter = AsyncMock(return_value=[])
    redis.hincrby = AsyncMock(return_value=1)
    redis.hgetall = AsyncMock(return_value={})
    redis.expire = AsyncMock(return_value=True)
    return redis


@pytest.fixture()
def processor():
    """Create a ChunkingProcessor instance."""
    return ChunkingProcessor()


@pytest.fixture()
def cache(mock_redis):
    """Create a ChunkingCache instance."""
    return ChunkingCache(redis_client=mock_redis)


@pytest.fixture()
def metrics():
    """Create a ChunkingMetrics instance."""
    return ChunkingMetrics()


@pytest.fixture()
def validator():
    """Create a ChunkingValidator instance."""
    return ChunkingValidator()


@pytest.fixture()
def config_manager():
    """Create a ChunkingConfigManager instance."""
    return ChunkingConfigManager()


@pytest.fixture()
def orchestrator(processor, cache, metrics, validator, config_manager):
    """Create a ChunkingOrchestrator instance."""
    return ChunkingOrchestrator(
        processor=processor,
        cache=cache,
        metrics=metrics,
        validator=validator,
        config_manager=config_manager,
    )


@pytest.fixture()
def adapter(orchestrator):
    """Create a ChunkingServiceAdapter instance."""
    return ChunkingServiceAdapter(orchestrator=orchestrator)


class TestChunkingProcessor:
    """Tests for ChunkingProcessor service."""

    async def test_process_document_simple(self, processor):
        """Test basic document processing."""
        content = "This is a test document with some content."
        strategy = "fixed_size"
        config = {"chunk_size": 20, "chunk_overlap": 5}

        chunks = await processor.process_document(content, strategy, config)

        assert len(chunks) > 0
        assert all("content" in chunk for chunk in chunks)
        assert all("index" in chunk for chunk in chunks)
        assert all("strategy" in chunk for chunk in chunks)

    async def test_process_document_with_fallback(self, processor):
        """Test fallback mechanism when strategy fails."""
        content = "Test content"
        strategy = "invalid_strategy"
        config = {}

        # Should use fallback without raising error
        chunks = await processor.process_document(content, strategy, config, use_fallback=True)

        assert len(chunks) > 0
        assert chunks[0]["strategy"] == "fallback"

    def test_calculate_statistics(self, processor):
        """Test statistics calculation."""
        chunks = [
            {"content": "chunk1", "index": 0},
            {"content": "chunk two", "index": 1},
            {"content": "chunk three here", "index": 2},
        ]

        stats = processor.calculate_statistics(chunks)

        assert stats["total_chunks"] == 3
        assert stats["min_chunk_size"] == 6  # "chunk1"
        assert stats["max_chunk_size"] == 16  # "chunk three here"
        assert stats["avg_chunk_size"] > 0


class TestChunkingCache:
    """Tests for ChunkingCache service."""

    async def test_cache_and_retrieve_preview(self, cache):
        """Test caching and retrieving preview data."""
        content_hash = "test_hash"
        strategy = "recursive"
        config = {"chunk_size": 1000}
        preview_data = {"chunks": ["chunk1", "chunk2"], "statistics": {}}

        # Cache the preview
        cache_key = await cache.cache_preview(content_hash, strategy, config, preview_data)
        assert cache_key

        # Mock Redis to return cached data
        cache.redis.get = AsyncMock(return_value='{"chunks": ["chunk1", "chunk2"], "statistics": {}}')

        # Retrieve cached preview
        cached = await cache.get_cached_preview(content_hash, strategy, config)
        assert cached is not None
        assert "chunks" in cached

    async def test_clear_cache(self, cache):
        """Test cache clearing."""

        # Create an async iterator for scan_iter
        async def async_iterator():
            for key in ["key1", "key2"]:
                yield key

        cache.redis.scan_iter = lambda **_kwargs: async_iterator()
        cache.redis.delete = AsyncMock(return_value=2)

        deleted = await cache.clear_cache()
        assert deleted == 2

    def test_generate_content_hash(self, cache):
        """Test content hash generation."""
        content = "Test content for hashing"
        hash1 = cache.generate_content_hash(content)
        hash2 = cache.generate_content_hash(content)

        assert hash1 == hash2  # Same content produces same hash
        assert len(hash1) == 16  # Hash is truncated to 16 chars


class TestChunkingMetrics:
    """Tests for ChunkingMetrics service."""

    async def test_measure_operation_success(self, metrics):
        """Test operation measurement for successful operation."""
        strategy = "recursive"

        async with metrics.measure_operation(strategy) as context:
            context["chunks_produced"] = 5

        stats = metrics.get_statistics()
        assert stats["total_operations"] == 1
        assert stats["successful_operations"] == 1

    async def test_measure_operation_failure(self, metrics):
        """Test operation measurement for failed operation."""
        strategy = "semantic"

        with pytest.raises(ValueError, match="Test error"):
            async with metrics.measure_operation(strategy) as _context:
                raise ValueError("Test error")

        stats = metrics.get_statistics()
        assert stats["total_operations"] == 1
        assert stats["failed_operations"] == 1

    def test_record_chunks_produced(self, metrics):
        """Test recording produced chunks."""
        chunks = [
            {"content": "chunk1"},
            {"content": "chunk2"},
        ]
        metrics.record_chunks_produced("recursive", chunks)

        stats = metrics.get_statistics()
        assert stats["total_chunks_produced"] == 2


class TestChunkingValidator:
    """Tests for ChunkingValidator service."""

    async def test_validate_preview_request_valid(self, validator):
        """Test validation of valid preview request."""
        # Should not raise
        await validator.validate_preview_request(
            content="Test content",
            document_id=None,
            strategy="recursive",
            config={"chunk_size": 1000},
        )

    async def test_validate_preview_request_invalid(self, validator):
        """Test validation of invalid preview request."""
        # No content or document_id
        with pytest.raises(ValidationError):
            await validator.validate_preview_request(
                content=None,
                document_id=None,
                strategy="recursive",
                config={},
            )

    def test_validate_strategy(self, validator):
        """Test strategy validation."""
        # Valid strategy
        validator.validate_strategy("recursive")

        # Invalid strategy
        with pytest.raises(ValidationError):
            validator.validate_strategy("invalid_strategy")

    def test_validate_config(self, validator):
        """Test configuration validation."""
        # Valid config
        validator.validate_config(
            "recursive",
            {"chunk_size": 1000, "chunk_overlap": 200},
        )

        # Invalid chunk size
        with pytest.raises(ValidationError):
            validator.validate_config(
                "recursive",
                {"chunk_size": 10},  # Too small
            )


class TestChunkingConfigManager:
    """Tests for ChunkingConfigManager service."""

    def test_get_default_config(self, config_manager):
        """Test getting default configuration."""
        config = config_manager.get_default_config("recursive")
        assert "chunk_size" in config
        assert "chunk_overlap" in config

    def test_get_all_strategies(self, config_manager):
        """Test getting all strategies."""
        strategies = config_manager.get_all_strategies()
        assert len(strategies) > 0
        assert all("id" in s for s in strategies)
        assert all("name" in s for s in strategies)

    def test_merge_configs(self, config_manager):
        """Test configuration merging."""
        user_config = {"chunk_size": 500}
        merged = config_manager.merge_configs("recursive", user_config)

        assert merged["chunk_size"] == 500  # User value
        assert "chunk_overlap" in merged  # Default value preserved

    def test_recommend_strategy(self, config_manager):
        """Test strategy recommendation."""
        rec = config_manager.recommend_strategy(
            file_type=".md",
            content_length=5000,
        )

        assert rec["strategy"] == "markdown"
        assert rec["confidence"] > 0
        assert len(rec["reasoning"]) > 0


class TestChunkingOrchestrator:
    """Tests for ChunkingOrchestrator service."""

    async def test_preview_chunks(self, orchestrator):
        """Test preview chunks operation."""
        content = "This is test content for chunking preview."
        strategy = "fixed_size"
        config = {"chunk_size": 100}  # Use valid size >= 50

        result = await orchestrator.preview_chunks(
            content=content,
            strategy=strategy,
            config=config,
            use_cache=False,
        )

        assert result.total_chunks > 0
        assert len(result.chunks) > 0
        assert result.strategy == strategy
        assert result.metrics is not None

    async def test_compare_strategies(self, orchestrator):
        """Test strategy comparison."""
        content = "Test content for comparing different chunking strategies."
        strategies = ["fixed_size", "recursive"]

        result = await orchestrator.compare_strategies(
            content=content,
            strategies=strategies,
        )

        assert len(result.comparisons) == 2
        assert result.recommendation is not None
        assert all(hasattr(comp, "total_chunks") for comp in result.comparisons)

    async def test_execute_ingestion_chunking(self, orchestrator):
        """Test chunking for ingestion."""
        content = "Content to be chunked for ingestion."
        strategy = "recursive"
        config = {"chunk_size": 100}

        chunks = await orchestrator.execute_ingestion_chunking(
            content=content,
            strategy=strategy,
            config=config,
        )

        assert len(chunks) > 0
        assert all("content" in chunk for chunk in chunks)


class TestChunkingServiceAdapter:
    """Tests for backward compatibility adapter."""

    async def test_preview_chunks_compatibility(self, adapter):
        """Test preview_chunks method compatibility."""
        content = "Test content"
        strategy = "recursive"

        result = await adapter.preview_chunks(
            content=content,
            strategy=strategy,
        )

        assert isinstance(result, dict)
        assert "chunks" in result
        assert "statistics" in result

    async def test_compare_strategies_compatibility(self, adapter):
        """Test compare_strategies method compatibility."""
        content = "Test content for comparison"

        result = await adapter.compare_strategies(
            content=content,
            strategies=["fixed_size"],
        )

        assert isinstance(result, dict)
        assert "comparisons" in result
        assert "recommendation" in result

    async def test_recommend_strategy_populates_config(self, adapter):
        """Adapter recommend_strategy should surface suggested config."""

        result = await adapter.recommend_strategy(file_type=".md", content_length=2048)
        api_model = result.to_api_model()

        assert api_model.suggested_config.strategy == api_model.recommended_strategy
        assert api_model.suggested_config.chunk_size > 0
        assert 0 <= api_model.suggested_config.chunk_overlap < api_model.suggested_config.chunk_size

    async def test_get_available_strategies_compatibility(self, adapter):
        """Test get_available_strategies compatibility."""
        strategies = await adapter.get_available_strategies()

        assert isinstance(strategies, list)
        assert all(isinstance(s, dict) for s in strategies)
        assert all("id" in s for s in strategies)

    async def test_execute_ingestion_chunking_reports_fallback(self, adapter, orchestrator, monkeypatch):
        """Adapter should surface fallback metadata when orchestrator falls back."""

        async def fake_process(_content, _strategy, _config, use_fallback=False):
            if not use_fallback:
                raise RuntimeError("boom")
            return [
                {
                    "chunk_id": "chunk_0000",
                    "text": "fallback chunk",
                    "strategy": "fallback",
                    "metadata": {},
                }
            ]

        monkeypatch.setattr(orchestrator.processor, "process_document", fake_process)

        result = await adapter.execute_ingestion_chunking(content="payload", strategy="recursive")

        assert result["stats"]["fallback"] is True
        assert result["stats"]["fallback_reason"] == "RuntimeError"
        assert result["stats"]["strategy_used"] == "fallback"

    async def test_execute_ingestion_chunking_legacy_reports_fallback(self, adapter, orchestrator, monkeypatch):
        """Legacy ingestion pathway should also propagate fallback stats."""

        async def fake_process(_content, _strategy, _config, use_fallback=False):
            if not use_fallback:
                raise RuntimeError("legacy boom")
            return [
                {
                    "chunk_id": "chunk_0000",
                    "text": "fallback chunk",
                    "strategy": "fallback",
                    "metadata": {},
                }
            ]

        monkeypatch.setattr(orchestrator.processor, "process_document", fake_process)

        result = await adapter.execute_ingestion_chunking(text="payload", document_id="doc1")

        assert result["stats"]["fallback"] is True
        assert result["stats"]["fallback_reason"] == "RuntimeError"
        assert result["stats"]["strategy_used"] == "fallback"
