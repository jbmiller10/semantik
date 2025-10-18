#!/usr/bin/env python3

"""
Unit tests for ChunkingService.

This module tests the ChunkingService business logic layer.
"""

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from redis import Redis

from packages.shared.chunking.infrastructure.exceptions import DocumentTooLargeError, ValidationError
from packages.webui.api.v2.chunking_schemas import ChunkingStrategy
from packages.webui.services.chunking_service import ChunkingService
from packages.webui.services.dtos.chunking_dtos import ServicePreviewResponse, ServiceStrategyRecommendation


class TestChunkingService:
    """Tests for ChunkingService."""

    @pytest.fixture()
    def mock_redis(self) -> MagicMock:
        """Mock Redis client."""
        redis = MagicMock(spec=Redis)
        redis.get = AsyncMock(return_value=None)
        redis.setex = AsyncMock()
        redis.incr = AsyncMock()
        redis.hincrby = AsyncMock()
        redis.expire = AsyncMock()
        return redis

    @pytest.fixture()
    def mock_repos(self) -> tuple[MagicMock, MagicMock]:
        """Mock repositories."""
        collection_repo = MagicMock()
        collection_repo.get_by_uuid_with_permission_check = AsyncMock()

        document_repo = MagicMock()
        document_repo.list_by_collection = AsyncMock(return_value=([], 0))

        return collection_repo, document_repo

    @pytest.fixture()
    def chunking_service(
        self,
        mock_redis: MagicMock,
        mock_repos: tuple[MagicMock, MagicMock],
    ) -> ChunkingService:
        """Create ChunkingService instance."""
        collection_repo, document_repo = mock_repos
        db_session = MagicMock()

        # Create operation repository mock
        operation_repo = MagicMock()
        operation_repo.get_by_uuid_with_permission_check = AsyncMock()

        service = ChunkingService(
            db_session=db_session,
            collection_repo=collection_repo,
            document_repo=document_repo,
            redis_client=mock_redis,
        )

        # Set the operation_repo attribute for tests that need it
        service.operation_repo = operation_repo

        return service

    async def test_preview_chunking_basic(
        self,
        chunking_service: ChunkingService,
        mock_redis: MagicMock,
    ) -> None:
        """Test basic chunking preview."""
        text = "This is a test document. " * 50  # ~1250 chars

        result = await chunking_service.preview_chunking(
            content=text,
            file_type=".txt",
            max_chunks=3,
        )

        # Verify result is a DTO
        assert isinstance(result, ServicePreviewResponse)
        assert result.strategy == ChunkingStrategy.RECURSIVE
        assert result.total_chunks >= 1  # At least one chunk
        assert len(result.chunks) <= 3  # Respects max_chunks
        assert isinstance(result.processing_time_ms, int)

        # Convert to API model to check additional fields if needed
        api_model = result.to_api_model()
        assert api_model.strategy == ChunkingStrategy.RECURSIVE
        assert api_model.total_chunks >= 1

        # Verify caching
        assert mock_redis.setex.called

    async def test_preview_chunking_code_file(
        self,
        chunking_service: ChunkingService,
    ) -> None:
        """Test chunking preview with code file."""
        code = """
def hello() -> None:
    print("Hello, world!")

class Test:
    pass
"""

        result = await chunking_service.preview_chunking(
            content=code,
            file_type=".py",
        )

        # Verify result is a DTO
        assert isinstance(result, ServicePreviewResponse)
        assert result.strategy == ChunkingStrategy.RECURSIVE

        # Check metadata in config for code file detection
        # The 'is_code_file' flag is typically stored in the config or metadata
        # Based on the service implementation, code detection affects the chunking strategy
        assert result.strategy == ChunkingStrategy.RECURSIVE

    async def test_preview_chunking_markdown(
        self,
        chunking_service: ChunkingService,
    ) -> None:
        """Test chunking preview with markdown file."""
        markdown = """
# Header 1

Content under header 1.

## Header 2

Content under header 2.
"""

        result = await chunking_service.preview_chunking(
            content=markdown,
            file_type=".md",
        )

        # Verify result is a DTO
        assert isinstance(result, ServicePreviewResponse)
        # Verify strategy (markdown files use recursive strategy)
        assert result.strategy == ChunkingStrategy.RECURSIVE

    async def test_preview_chunking_custom_config(
        self,
        chunking_service: ChunkingService,
    ) -> None:
        """Test chunking preview with custom configuration."""
        text = "Test text. " * 100

        config = {
            "params": {"chunk_size": 200, "chunk_overlap": 50},
        }

        result = await chunking_service.preview_chunking(
            content=text,
            strategy="fixed_size",  # character strategy doesn't exist, use fixed_size
            config=config,
        )

        # Verify result is a DTO
        assert isinstance(result, ServicePreviewResponse)
        # Verify custom config was used
        assert result.strategy == "fixed_size"

    async def test_preview_chunking_validation_error(
        self,
        chunking_service: ChunkingService,
    ) -> None:
        """Test chunking preview with invalid parameters."""
        text = "Test text"

        # Invalid chunk size
        config = {
            "strategy": "character",
            "params": {"chunk_size": -100},
        }

        with pytest.raises(ValidationError):
            await chunking_service.preview_chunking(content=text, config=config)

    async def test_preview_chunking_size_limit(
        self,
        chunking_service: ChunkingService,
    ) -> None:
        """Test chunking preview size limit."""
        # Text larger than preview limit (10MB)
        large_text = "x" * (11 * 1024 * 1024)  # 11MB (larger than 10MB limit)

        with pytest.raises(DocumentTooLargeError):
            await chunking_service.preview_chunking(content=large_text)

    async def test_preview_chunking_cached(
        self,
        chunking_service: ChunkingService,
        mock_redis: MagicMock,
    ) -> None:
        """Test chunking preview returns cached result."""

        # Set up cached response
        cached_data = {
            "chunks": [{"chunk_id": "test_0000", "text": "cached", "index": 0}],
            "total_chunks": 1,
            "strategy_used": "recursive",
            "is_code_file": False,
            "performance_metrics": {},
            "recommendations": [],
        }
        # Redis returns JSON string (not bytes in this mock)
        mock_redis.get.return_value = json.dumps(cached_data)

        result = await chunking_service.preview_chunking(content="test")

        # Verify result is a DTO
        assert isinstance(result, ServicePreviewResponse)
        # Should return cached result - check the first chunk has cached text
        assert len(result.chunks) > 0
        # Chunks can be dict or ServiceChunkPreview
        first_chunk = result.chunks[0]
        if isinstance(first_chunk, dict):
            assert first_chunk.get("text") == "cached" or first_chunk.get("content") == "cached"
        else:
            assert first_chunk.text == "cached" or first_chunk.content == "cached"
        assert not mock_redis.setex.called  # Shouldn't cache again

    async def test_recommend_strategy_markdown_majority(
        self,
        chunking_service: ChunkingService,
    ) -> None:
        """Test strategy recommendation with markdown majority."""
        file_paths = [
            "README.md",
            "docs/guide.md",
            "CHANGELOG.md",
            "config.json",
            "main.py",
        ]

        result = await chunking_service.recommend_strategy(file_paths=file_paths)

        # Verify result is a DTO
        assert isinstance(result, ServiceStrategyRecommendation)
        assert result.strategy == ChunkingStrategy.RECURSIVE
        assert "markdown" in result.reasoning.lower()
        # The file_type_breakdown is not directly exposed in the DTO
        # but the reasoning should mention markdown files

    async def test_recommend_strategy_code_files(
        self,
        chunking_service: ChunkingService,
    ) -> None:
        """Test strategy recommendation with significant code files."""
        file_paths = [
            "main.py",
            "utils.js",
            "test.cpp",
            "README.md",
            "data.txt",
            "script.sh",
        ]

        result = await chunking_service.recommend_strategy(file_paths=file_paths)

        # Verify result is a DTO
        assert isinstance(result, ServiceStrategyRecommendation)
        assert result.strategy == ChunkingStrategy.RECURSIVE
        assert result.chunk_size == 500  # Optimized for code
        assert "code" in result.reasoning.lower()

    async def test_recommend_strategy_mixed_content(
        self,
        chunking_service: ChunkingService,
    ) -> None:
        """Test strategy recommendation with mixed content."""
        file_paths = [
            "document1.txt",
            "document2.pdf",
            "notes.txt",
            "config.yaml",
        ]

        result = await chunking_service.recommend_strategy(file_paths=file_paths)

        # Verify result is a DTO
        assert isinstance(result, ServiceStrategyRecommendation)
        assert result.strategy == ChunkingStrategy.RECURSIVE
        assert result.chunk_size == 600  # Default
        assert "general" in result.reasoning.lower() or "mixed" in result.reasoning.lower()

    async def test_get_chunking_statistics(
        self,
        chunking_service: ChunkingService,
    ) -> None:
        """Test getting chunking statistics."""
        # Mock collection
        mock_collection = MagicMock(id="test-collection", uuid="test-collection")
        chunking_service.collection_repo.get_by_uuid = AsyncMock(return_value=mock_collection)

        # Mock aggregated stats query result
        mock_stats = MagicMock()
        mock_stats.total_operations = 3
        mock_stats.completed_operations = 2
        mock_stats.failed_operations = 0
        mock_stats.processing_operations = 1
        mock_stats.avg_processing_time = 10.5
        mock_stats.last_operation_at = datetime.now(tz=UTC)
        mock_stats.first_operation_at = datetime.now(tz=UTC) - timedelta(hours=1)

        # Mock latest strategy query result
        mock_strategy_row = MagicMock()
        mock_strategy_row.strategy = "recursive"

        # Mock the db query results
        mock_stats_result = MagicMock()
        mock_stats_result.one.return_value = mock_stats

        mock_strategy_result = MagicMock()
        mock_strategy_result.one_or_none.return_value = mock_strategy_row

        # Set up execute to return different results for each query
        chunking_service.db_session.execute = AsyncMock(side_effect=[mock_stats_result, mock_strategy_result])

        result = await chunking_service.get_chunking_statistics(collection_id="test-collection")

        assert isinstance(result, dict)
        assert result["collection_id"] == "test-collection"
        assert result["total_operations"] == 3
        assert result["completed_operations"] == 2
        assert result["in_progress_operations"] == 1
        assert result["latest_strategy"] == "recursive"

    async def test_validate_config_for_collection(
        self,
        chunking_service: ChunkingService,
        mock_repos: tuple[MagicMock, MagicMock],
    ) -> None:
        """Test validating config for collection."""
        # Mock collection
        mock_collection = MagicMock(id="test-collection", uuid="test-collection")
        chunking_service.collection_repo.get_by_uuid = AsyncMock(return_value=mock_collection)

        # Mock validator to return valid
        chunking_service.validator.validate_config = MagicMock(return_value=(True, []))

        config = {
            "strategy": "recursive",
            "params": {"chunk_size": 600, "chunk_overlap": 100},
        }

        result = await chunking_service.validate_config_for_collection(
            collection_id="test-collection",
            strategy="recursive",
            config=config,
        )

        assert isinstance(result, dict)
        assert result["valid"] is True
        assert result["errors"] == []
        assert "suggested_config" in result

    async def test_validate_config_invalid_params(
        self,
        chunking_service: ChunkingService,
    ) -> None:
        """Test config validation with invalid parameters."""
        # Mock collection
        mock_collection = MagicMock(id="test-collection", uuid="test-collection")
        chunking_service.collection_repo.get_by_uuid = AsyncMock(return_value=mock_collection)

        # Mock validator to return invalid with errors
        chunking_service.validator.validate_config = MagicMock(return_value=(False, ["chunk_size is too large"]))

        config = {
            "strategy": "recursive",
            "params": {"chunk_size": 100000},  # Too large
        }

        result = await chunking_service.validate_config_for_collection(
            collection_id="test-collection",
            strategy="recursive",
            config=config,
        )

        assert result["valid"] is False
        assert len(result["errors"]) > 0
        assert "suggested_config" in result

    async def test_verify_collection_access(
        self,
        chunking_service: ChunkingService,
        mock_repos: tuple[MagicMock, MagicMock],
    ) -> None:
        """Test collection access verification."""
        collection_repo, _ = mock_repos

        await chunking_service.verify_collection_access(
            collection_id="test-collection",
            user_id=123,
        )

        # Should call repository method
        collection_repo.get_by_uuid_with_permission_check.assert_called_once_with(
            collection_uuid="test-collection",
            user_id=123,
        )

    async def test_track_preview_usage(
        self,
        chunking_service: ChunkingService,
        mock_redis: MagicMock,
    ) -> None:
        """Test preview usage tracking."""
        await chunking_service.track_preview_usage(
            user_id=1,
            strategy="recursive",
            file_type=".py",
        )

        # Should increment counters (user, strategy, and file_type)
        assert mock_redis.incr.call_count == 3
        mock_redis.incr.assert_any_call("chunking:preview:user:1:recursive")
        mock_redis.incr.assert_any_call("chunking:preview:usage:recursive")
        mock_redis.incr.assert_any_call("chunking:preview:file_type:.py")

    def test_calculate_metrics(
        self,
        chunking_service: ChunkingService,
    ) -> None:
        """Test metrics calculation."""
        # Mock chunks
        mock_chunks = [
            MagicMock(text="x" * 100),
            MagicMock(text="x" * 150),
            MagicMock(text="x" * 200),
        ]

        metrics = chunking_service._calculate_metrics(
            chunks=mock_chunks,
            text_length=1000,
            processing_time=0.5,
        )

        assert metrics["total_chunks"] == 3
        assert metrics["average_chunk_size"] == 150  # (100 + 150 + 200) / 3
        assert metrics["min_chunk_size"] == 100
        assert metrics["max_chunk_size"] == 200
        assert metrics["chunks_per_second"] == 6  # 3 / 0.5
        assert "compression_ratio" in metrics

    def test_get_recommendations(
        self,
        chunking_service: ChunkingService,
    ) -> None:
        """Test getting recommendations."""
        # Mock chunks with high variance
        mock_chunks = [
            MagicMock(text="x" * 50),
            MagicMock(text="x" * 500),
            MagicMock(text="x" * 60),
            MagicMock(text="x" * 450),
        ]

        recommendations = chunking_service._get_recommendations(
            chunks=mock_chunks,
            file_type=".py",
        )

        assert len(recommendations) > 0
        assert any("variance" in r for r in recommendations)

        # Test with many small chunks
        small_chunks = [MagicMock(text="x" * 50) for _ in range(10)]
        recommendations = chunking_service._get_recommendations(small_chunks)

        assert any("small chunks" in r for r in recommendations)

    async def test_get_chunking_progress(
        self,
        chunking_service: ChunkingService,
    ) -> None:
        """Test getting chunking progress."""
        # Mock the db query to return an operation
        mock_operation = MagicMock()
        mock_operation.id = "test-operation"
        mock_operation.status = "in_progress"
        mock_operation.started_at = datetime.now(UTC)
        mock_operation.completed_at = None
        mock_operation.meta = {
            "chunks_processed": 50,
            "total_chunks": 100,
        }
        mock_operation.error_message = None

        # Mock the database result
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_operation
        chunking_service.db_session.execute = AsyncMock(return_value=mock_result)

        # Call with just operation_id
        progress = await chunking_service.get_chunking_progress("test-operation")

        assert progress is not None
        assert progress["status"] == "in_progress"
        assert progress["chunks_processed"] == 50
        assert progress["total_chunks"] == 100
        assert progress["progress_percentage"] == 50.0
