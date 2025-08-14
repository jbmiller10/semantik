"""
Direct tests for v2 chunking API endpoints.

This module directly tests the endpoint functions to ensure proper coverage.
Similar to test_collections_operations.py approach.
"""

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from packages.shared.chunking.infrastructure.exceptions import (
    DocumentTooLargeError,
    ValidationError,
)
from packages.webui.api.v2.chunking import (
    compare_strategies,
    generate_preview,
    get_cached_preview,
    get_chunking_stats,
    get_operation_progress,
    get_strategy_details,
    list_strategies,
    recommend_strategy,
    start_chunking_operation,
)
from packages.webui.api.v2.chunking_schemas import (
    ChunkingOperationRequest,
    ChunkingStatus,
    ChunkingStrategy,
    CompareRequest,
    PreviewRequest,
)
from packages.webui.services.chunking_service import ChunkingService
from packages.webui.services.collection_service import CollectionService


@pytest.fixture()
def mock_user() -> dict[str, Any]:
    """Mock authenticated user."""
    return {"id": 1, "username": "testuser", "email": "test@example.com"}


@pytest.fixture()
def mock_chunking_service() -> AsyncMock:
    """Mock ChunkingService."""
    service = AsyncMock(spec=ChunkingService)

    # Setup default responses
    service.get_available_strategies_for_api.return_value = [
        {
            "id": "recursive",
            "name": "Recursive",
            "description": "Recursively splits text",
            "config": {"chunk_size": 1000, "chunk_overlap": 100},
            "pros": ["Good for structured text"],
            "cons": ["May split sentences"],
            "use_cases": ["General documents"],
        }
    ]

    service.get_strategy_details.return_value = {
        "id": "recursive",
        "name": "Recursive",
        "description": "Recursively splits text",
        "parameters": [
            {
                "name": "chunk_size",
                "type": "integer",
                "default": 1000,
                "min": 100,
                "max": 10000,
                "description": "Target chunk size",
            }
        ],
        "examples": [],
    }

    service.recommend_strategy.return_value = {
        "strategy": ChunkingStrategy.RECURSIVE,
        "reasoning": "Best for mixed content",
        "params": {"chunk_size": 1000, "chunk_overlap": 100},
        "alternatives": [],
    }

    service.preview_chunking.return_value = {
        "preview_id": "test-preview-id",
        "strategy": "recursive",
        "chunks": [
            {"content": "Test chunk 1", "metadata": {"index": 0}},
            {"content": "Test chunk 2", "metadata": {"index": 1}},
        ],
        "total_chunks": 2,
        "is_cached": False,
        "cached_at": None,
        "performance_metrics": {"processing_time_ms": 50},
        "recommendations": [],
    }

    service.get_cached_preview.return_value = {
        "preview_id": "cached-preview-id",
        "chunks": [{"content": "Cached chunk", "metadata": {}}],
        "total_chunks": 1,
        "is_cached": True,
        "cached_at": datetime.now(UTC).isoformat(),
    }

    service.compare_strategies.return_value = {
        "comparisons": [
            {
                "strategy": "recursive",
                "chunk_count": 10,
                "avg_chunk_size": 500,
                "processing_time_ms": 100,
                "score": 0.85,
                "preview": {"chunks": []},
            }
        ],
        "recommendation": {
            "best_strategy": "recursive",
            "reasoning": "Best balance of chunk size and count",
        },
    }

    service.start_chunking_operation.return_value = {
        "operation_id": "op-123",
        "collection_id": "coll-123",
        "status": "pending",
    }

    service.get_chunking_progress.return_value = {
        "operation_id": "op-123",
        "status": "in_progress",
        "progress_percentage": 50.0,
        "chunks_processed": 50,
        "total_chunks": 100,
        "estimated_time_remaining": 60,
        "started_at": datetime.now(UTC).isoformat(),
    }

    service.get_chunking_statistics.return_value = {
        "collection_id": "coll-123",
        "total_operations": 10,
        "completed_operations": 8,
        "failed_operations": 1,
        "in_progress_operations": 1,
        "latest_strategy": "recursive",
    }

    return service


@pytest.fixture()
def mock_collection_service() -> AsyncMock:
    """Mock CollectionService."""
    return AsyncMock(spec=CollectionService)


@pytest.fixture()
def mock_collection() -> dict[str, Any]:
    """Mock collection object."""
    return {
        "id": "coll-123",
        "uuid": "123e4567-e89b-12d3-a456-426614174000",
        "name": "Test Collection",
        "owner_id": 1,
    }


class TestStrategyEndpoints:
    """Test strategy management endpoints."""

    @pytest.mark.asyncio()
    async def test_list_strategies_success(self, mock_user: dict[str, Any], mock_chunking_service: AsyncMock) -> None:
        """Test successful strategy listing."""
        result = await list_strategies(
            _current_user=mock_user,
            service=mock_chunking_service,
        )

        assert len(result) == 1
        assert result[0].id == "recursive"
        assert result[0].name == "Recursive"
        mock_chunking_service.get_available_strategies_for_api.assert_called_once()

    @pytest.mark.asyncio()
    async def test_list_strategies_service_error(
        self, mock_user: dict[str, Any], mock_chunking_service: AsyncMock
    ) -> None:
        """Test strategy listing with service error."""
        mock_chunking_service.get_available_strategies_for_api.side_effect = Exception("Service error")

        with pytest.raises(Exception, match="Service error"):
            await list_strategies(
                _current_user=mock_user,
                service=mock_chunking_service,
            )

    @pytest.mark.asyncio()
    async def test_get_strategy_details_success(
        self, mock_user: dict[str, Any], mock_chunking_service: AsyncMock
    ) -> None:
        """Test getting strategy details."""
        result = await get_strategy_details(
            strategy="recursive",
            _current_user=mock_user,
            service=mock_chunking_service,
        )

        assert result["id"] == "recursive"
        assert "parameters" in result
        mock_chunking_service.get_strategy_details.assert_called_once_with("recursive")

    @pytest.mark.asyncio()
    async def test_get_strategy_details_not_found(
        self, mock_user: dict[str, Any], mock_chunking_service: AsyncMock
    ) -> None:
        """Test getting details for non-existent strategy."""
        mock_chunking_service.get_strategy_details.return_value = None

        with pytest.raises(HTTPException) as exc_info:
            await get_strategy_details(
                strategy="nonexistent",
                _current_user=mock_user,
                service=mock_chunking_service,
            )
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio()
    async def test_recommend_strategy_success(
        self, mock_user: dict[str, Any], mock_chunking_service: AsyncMock
    ) -> None:
        """Test strategy recommendation."""
        result = await recommend_strategy(
            file_paths=["doc1.txt", "doc2.pdf"],
            _current_user=mock_user,
            service=mock_chunking_service,
        )

        assert result.strategy == ChunkingStrategy.RECURSIVE
        assert result.reasoning == "Best for mixed content"
        mock_chunking_service.recommend_strategy.assert_called_once()


class TestPreviewEndpoints:
    """Test preview operation endpoints."""

    @pytest.mark.asyncio()
    async def test_generate_preview_with_content(
        self, mock_user: dict[str, Any], mock_chunking_service: AsyncMock
    ) -> None:
        """Test preview generation with content."""
        request = PreviewRequest(
            content="Test document content",
            strategy=ChunkingStrategy.RECURSIVE,
            config={"chunk_size": 1000},
        )

        result = await generate_preview(
            request=request,
            _current_user=mock_user,
            service=mock_chunking_service,
        )

        assert result.preview_id == "test-preview-id"
        assert result.total_chunks == 2
        assert len(result.chunks) == 2

        # Verify service was called with correct parameters
        mock_chunking_service.preview_chunking.assert_called_once()
        call_args = mock_chunking_service.preview_chunking.call_args
        assert call_args.kwargs["content"] == "Test document content"

    @pytest.mark.asyncio()
    async def test_generate_preview_validation_error(
        self, mock_user: dict[str, Any], mock_chunking_service: AsyncMock
    ) -> None:
        """Test preview with validation error."""
        mock_chunking_service.preview_chunking.side_effect = ValidationError("Invalid chunk size")

        request = PreviewRequest(
            content="Test content",
            config={"chunk_size": -100},  # Invalid
        )

        with pytest.raises(HTTPException) as exc_info:
            await generate_preview(
                request=request,
                _current_user=mock_user,
                service=mock_chunking_service,
            )
        assert exc_info.value.status_code == 400
        assert "Invalid chunk size" in exc_info.value.detail

    @pytest.mark.asyncio()
    async def test_generate_preview_content_too_large(
        self, mock_user: dict[str, Any], mock_chunking_service: AsyncMock
    ) -> None:
        """Test preview with content too large."""
        mock_chunking_service.preview_chunking.side_effect = DocumentTooLargeError("Content exceeds 10MB limit")

        request = PreviewRequest(content="x" * 11_000_000)  # 11MB

        with pytest.raises(HTTPException) as exc_info:
            await generate_preview(
                request=request,
                _current_user=mock_user,
                service=mock_chunking_service,
            )
        assert exc_info.value.status_code == 413
        assert "Content exceeds 10MB limit" in exc_info.value.detail

    @pytest.mark.asyncio()
    async def test_get_cached_preview_success(
        self, mock_user: dict[str, Any], mock_chunking_service: AsyncMock
    ) -> None:
        """Test getting cached preview."""
        result = await get_cached_preview(
            preview_id="cached-preview-id",
            _current_user=mock_user,
            service=mock_chunking_service,
        )

        assert result["preview_id"] == "cached-preview-id"
        assert result["is_cached"] is True
        assert result["total_chunks"] == 1
        mock_chunking_service.get_cached_preview.assert_called_once_with("cached-preview-id")

    @pytest.mark.asyncio()
    async def test_get_cached_preview_not_found(
        self, mock_user: dict[str, Any], mock_chunking_service: AsyncMock
    ) -> None:
        """Test getting non-existent cached preview."""
        mock_chunking_service.get_cached_preview.return_value = None

        with pytest.raises(HTTPException) as exc_info:
            await get_cached_preview(
                preview_id="nonexistent",
                _current_user=mock_user,
                service=mock_chunking_service,
            )
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio()
    async def test_compare_strategies_success(
        self, mock_user: dict[str, Any], mock_chunking_service: AsyncMock
    ) -> None:
        """Test comparing multiple strategies."""
        request = CompareRequest(
            content="Test content for comparison",
            strategies=[ChunkingStrategy.RECURSIVE, ChunkingStrategy.FIXED_SIZE],
        )

        result = await compare_strategies(
            request=request,
            _current_user=mock_user,
            service=mock_chunking_service,
        )

        assert len(result.comparisons) == 1
        assert result.comparisons[0].strategy == "recursive"
        assert result.recommendation.best_strategy == "recursive"
        mock_chunking_service.compare_strategies.assert_called_once()


class TestOperationEndpoints:
    """Test chunking operation endpoints."""

    @pytest.mark.asyncio()
    async def test_start_chunking_operation_success(
        self,
        mock_user: dict[str, Any],
        mock_chunking_service: AsyncMock,
        mock_collection: dict[str, Any],
    ) -> None:
        """Test starting a chunking operation."""
        request = ChunkingOperationRequest(
            strategy=ChunkingStrategy.RECURSIVE,
            config={"chunk_size": 1000, "chunk_overlap": 100},
        )

        with patch("packages.webui.api.v2.chunking.BackgroundTasks") as mock_bg:
            mock_bg_instance = MagicMock()
            mock_bg.return_value = mock_bg_instance

            result = await start_chunking_operation(
                collection_uuid="coll-123",
                request=request,
                background_tasks=mock_bg_instance,
                _current_user=mock_user,
                _collection=mock_collection,
                service=mock_chunking_service,
            )

            assert result.operation_id == "op-123"
            assert result.status == ChunkingStatus.PENDING
            assert result.collection_id == "coll-123"
            mock_chunking_service.start_chunking_operation.assert_called_once()

    @pytest.mark.asyncio()
    async def test_get_operation_progress_success(
        self, mock_user: dict[str, Any], mock_chunking_service: AsyncMock
    ) -> None:
        """Test getting operation progress."""
        result = await get_operation_progress(
            operation_id="op-123",
            _current_user=mock_user,
            service=mock_chunking_service,
        )

        assert result.operation_id == "op-123"
        assert result.status == ChunkingStatus.IN_PROGRESS
        assert result.progress_percentage == 50.0
        assert result.chunks_processed == 50
        mock_chunking_service.get_chunking_progress.assert_called_once_with("op-123")

    @pytest.mark.asyncio()
    async def test_get_operation_progress_not_found(
        self, mock_user: dict[str, Any], mock_chunking_service: AsyncMock
    ) -> None:
        """Test getting progress for non-existent operation."""
        mock_chunking_service.get_chunking_progress.return_value = None

        with pytest.raises(HTTPException) as exc_info:
            await get_operation_progress(
                operation_id="nonexistent",
                _current_user=mock_user,
                service=mock_chunking_service,
            )
        assert exc_info.value.status_code == 404


class TestStatsEndpoints:
    """Test statistics endpoints."""

    @pytest.mark.asyncio()
    async def test_get_chunking_stats_success(
        self, mock_user: dict[str, Any], mock_chunking_service: AsyncMock
    ) -> None:
        """Test getting chunking statistics."""
        result = await get_chunking_stats(
            collection_uuid="coll-123",
            _current_user=mock_user,
            service=mock_chunking_service,
        )

        assert result.collection_id == "coll-123"
        assert result.total_operations == 10
        assert result.completed_operations == 8
        assert result.failed_operations == 1
        assert result.latest_strategy == "recursive"
        mock_chunking_service.get_chunking_statistics.assert_called_once_with(collection_id="coll-123")


class TestErrorHandling:
    """Test error handling in endpoints."""

    @pytest.mark.asyncio()
    async def test_application_error_handling(
        self, mock_user: dict[str, Any], mock_chunking_service: AsyncMock
    ) -> None:
        """Test that application errors are properly handled."""
        from packages.shared.chunking.infrastructure.exceptions import ApplicationError
        from packages.webui.api.v2.chunking import application_exception_handler

        error = ApplicationError("Test error", correlation_id="test-123")
        request = MagicMock()

        response = await application_exception_handler(request, error)

        assert response.status_code == 500
        response_body = response.body.decode()
        assert "test-123" in response_body
        assert "Test error" in response_body
