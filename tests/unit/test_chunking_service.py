#!/usr/bin/env python3
"""
Unit tests for ChunkingService.

This module tests the ChunkingService business logic layer.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis import Redis

from packages.shared.text_processing.file_type_detector import FileTypeDetector
from packages.webui.services.chunking_security import ValidationError
from packages.webui.services.chunking_service import (
    ChunkingPreviewResponse,
    ChunkingRecommendation,
    ChunkingService,
    ChunkingStatistics,
    ChunkingValidationResult,
)


class TestChunkingService:
    """Tests for ChunkingService."""

    @pytest.fixture
    def mock_redis(self) -> MagicMock:
        """Mock Redis client."""
        redis = MagicMock(spec=Redis)
        redis.get = MagicMock(return_value=None)
        redis.setex = MagicMock()
        redis.incr = MagicMock()
        return redis

    @pytest.fixture
    def mock_repos(self) -> tuple[MagicMock, MagicMock]:
        """Mock repositories."""
        collection_repo = MagicMock()
        collection_repo.get_by_uuid_with_permission_check = AsyncMock()

        document_repo = MagicMock()
        document_repo.list_by_collection = AsyncMock(return_value=([], 0))

        return collection_repo, document_repo

    @pytest.fixture
    def chunking_service(
        self,
        mock_redis: MagicMock,
        mock_repos: tuple[MagicMock, MagicMock],
    ) -> ChunkingService:
        """Create ChunkingService instance."""
        collection_repo, document_repo = mock_repos
        db_session = MagicMock()

        return ChunkingService(
            db_session=db_session,
            collection_repo=collection_repo,
            document_repo=document_repo,
            redis_client=mock_redis,
        )

    async def test_preview_chunking_basic(
        self,
        chunking_service: ChunkingService,
        mock_redis: MagicMock,
    ) -> None:
        """Test basic chunking preview."""
        text = "This is a test document. " * 50  # ~1250 chars

        result = await chunking_service.preview_chunking(
            text=text,
            file_type=".txt",
            max_chunks=3,
        )

        # Verify result
        assert isinstance(result, ChunkingPreviewResponse)
        assert result.strategy_used == "recursive"
        assert result.total_chunks > 1
        assert len(result.chunks) <= 3  # Respects max_chunks
        assert not result.is_code_file
        assert len(result.performance_metrics) > 0
        assert isinstance(result.recommendations, list)

        # Verify caching
        assert mock_redis.setex.called

    async def test_preview_chunking_code_file(
        self,
        chunking_service: ChunkingService,
    ) -> None:
        """Test chunking preview with code file."""
        code = """
def hello():
    print("Hello, world!")
    
class Test:
    pass
"""

        result = await chunking_service.preview_chunking(
            text=code,
            file_type=".py",
        )

        # Verify code file detection
        assert result.is_code_file
        assert result.strategy_used == "recursive"

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
            text=markdown,
            file_type=".md",
        )

        # Verify markdown strategy
        assert result.strategy_used == "markdown"
        assert not result.is_code_file

    async def test_preview_chunking_custom_config(
        self,
        chunking_service: ChunkingService,
    ) -> None:
        """Test chunking preview with custom configuration."""
        text = "Test text. " * 100

        config = {
            "strategy": "character",
            "params": {"chunk_size": 200, "chunk_overlap": 50},
        }

        result = await chunking_service.preview_chunking(
            text=text,
            config=config,
        )

        # Verify custom config was used
        assert result.strategy_used == "character"

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
            await chunking_service.preview_chunking(text=text, config=config)

    async def test_preview_chunking_size_limit(
        self,
        chunking_service: ChunkingService,
    ) -> None:
        """Test chunking preview size limit."""
        # Text larger than preview limit (1MB)
        large_text = "x" * (2 * 1024 * 1024)  # 2MB

        with pytest.raises(ValidationError, match="Document too large"):
            await chunking_service.preview_chunking(text=large_text)

    async def test_preview_chunking_cached(
        self,
        chunking_service: ChunkingService,
        mock_redis: MagicMock,
    ) -> None:
        """Test chunking preview returns cached result."""
        import json

        # Set up cached response
        cached_data = {
            "chunks": [{"chunk_id": "test_0000", "text": "cached"}],
            "total_chunks": 1,
            "strategy_used": "recursive",
            "is_code_file": False,
            "performance_metrics": {},
            "recommendations": [],
        }
        # Redis returns bytes, so encode the JSON string
        mock_redis.get.return_value = json.dumps(cached_data).encode()

        result = await chunking_service.preview_chunking(text="test")

        # Should return cached result
        assert result.chunks[0]["text"] == "cached"
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

        result = await chunking_service.recommend_strategy(file_paths)

        assert isinstance(result, ChunkingRecommendation)
        assert result.recommended_strategy == "markdown"
        assert "markdown" in result.rationale.lower()
        assert result.file_type_breakdown["markdown"] == 3

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

        result = await chunking_service.recommend_strategy(file_paths)

        assert result.recommended_strategy == "recursive"
        assert result.recommended_params["chunk_size"] == 500  # Optimized for code
        assert "code" in result.rationale.lower()

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

        result = await chunking_service.recommend_strategy(file_paths)

        assert result.recommended_strategy == "recursive"
        assert result.recommended_params["chunk_size"] == 600  # Default
        assert "general" in result.rationale.lower() or "mixed" in result.rationale.lower()

    async def test_get_chunking_statistics(
        self,
        chunking_service: ChunkingService,
    ) -> None:
        """Test getting chunking statistics."""
        result = await chunking_service.get_chunking_statistics(
            collection_id="test-collection",
            days=30,
        )

        assert isinstance(result, ChunkingStatistics)
        assert result.total_documents == 100  # Mock data
        assert result.total_chunks == 1000
        assert result.average_chunk_size == 600
        assert "recursive" in result.strategy_breakdown

    async def test_validate_config_for_collection(
        self,
        chunking_service: ChunkingService,
        mock_repos: tuple[MagicMock, MagicMock],
    ) -> None:
        """Test validating config for collection."""
        _, document_repo = mock_repos

        # Mock documents
        mock_docs = [
            MagicMock(file_name="doc1.txt", file_size_bytes=1000),
            MagicMock(file_name="doc2.txt", file_size_bytes=2000),
        ]
        document_repo.list_by_collection.return_value = (mock_docs, 2)

        config = {
            "strategy": "recursive",
            "params": {"chunk_size": 600, "chunk_overlap": 100},
        }

        result = await chunking_service.validate_config_for_collection(
            collection_id="test-collection",
            config=config,
            sample_size=2,
        )

        assert isinstance(result, ChunkingValidationResult)
        assert result.is_valid
        assert len(result.sample_results) == 2
        assert result.estimated_total_chunks > 0
        assert len(result.warnings) == 0

    async def test_validate_config_invalid_params(
        self,
        chunking_service: ChunkingService,
    ) -> None:
        """Test config validation with invalid parameters."""
        config = {
            "strategy": "recursive",
            "params": {"chunk_size": 100000},  # Too large
        }

        with pytest.raises(ValidationError):
            await chunking_service.validate_config_for_collection(
                collection_id="test-collection",
                config=config,
            )

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
            strategy="recursive",
            file_type=".py",
        )

        # Should increment counters
        assert mock_redis.incr.call_count == 2
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
        # This is a generator/async iterator
        progress_gen = chunking_service.get_chunking_progress("test-collection")

        # Get first progress update
        progress = await progress_gen.__anext__()

        assert progress["status"] == "processing"
        assert progress["processed"] == 50
        assert progress["total"] == 100
        assert progress["percentage"] == 50.0
