"""
Direct tests for v2 chunking API endpoints.

This module directly tests the endpoint functions to ensure proper coverage.
Similar to test_collections_operations.py approach.
"""

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, create_autospec, patch

import pytest
from fastapi import HTTPException, Request

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
    ChunkingConfigBase,
    ChunkingOperationRequest,
    ChunkingStatus,
    ChunkingStrategy,
    CompareRequest,
    PreviewRequest,
)
from packages.webui.services.chunking_service import ChunkingService
from packages.webui.services.collection_service import CollectionService
from packages.webui.services.dtos.chunking_dtos import (
    ServiceChunkingStats,
    ServiceChunkPreview,
    ServiceCompareResponse,
    ServicePreviewResponse,
    ServiceStrategyComparison,
    ServiceStrategyInfo,
    ServiceStrategyRecommendation,
)


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
        ServiceStrategyInfo(
            id="recursive",
            name="Recursive",
            description="Recursively splits text",
            default_config={
                "strategy": ChunkingStrategy.RECURSIVE,
                "chunk_size": 1000,
                "chunk_overlap": 100,
                "preserve_sentences": True,
            },
            best_for=["text", "markdown", "code"],
            pros=["Good for structured text"],
            cons=["May split sentences"],
            performance_characteristics={"speed": "fast", "quality": "good"},
        )
    ]

    service.get_strategy_details.return_value = ServiceStrategyInfo(
        id="recursive",
        name="Recursive",
        description="Recursively splits text",
        default_config={
            "strategy": ChunkingStrategy.RECURSIVE,
            "chunk_size": 1000,
            "chunk_overlap": 100,
            "preserve_sentences": True,
        },
        best_for=["text", "markdown", "code"],
        pros=["Good for structured text"],
        cons=["May split sentences"],
        performance_characteristics={"speed": "fast", "quality": "good"},
    )

    service.recommend_strategy.return_value = ServiceStrategyRecommendation(
        strategy=ChunkingStrategy.RECURSIVE,
        reasoning="Best for mixed content",
        confidence=0.85,
        chunk_size=1000,
        chunk_overlap=100,
        alternatives=[],
    )

    service.preview_chunking.return_value = ServicePreviewResponse(
        preview_id="test-preview-id",
        strategy=ChunkingStrategy.RECURSIVE,
        config={
            "strategy": ChunkingStrategy.RECURSIVE,
            "chunk_size": 1000,
            "chunk_overlap": 100,
        },
        chunks=[
            ServiceChunkPreview(index=0, content="Test chunk 1", metadata={}),
            ServiceChunkPreview(index=1, content="Test chunk 2", metadata={}),
        ],
        total_chunks=2,
        processing_time_ms=50,
        cached=False,
        expires_at=datetime.now(UTC) + timedelta(minutes=15),
    )

    service.get_cached_preview_by_id.return_value = ServicePreviewResponse(
        preview_id="cached-preview-id",
        strategy=ChunkingStrategy.RECURSIVE,
        config={
            "strategy": ChunkingStrategy.RECURSIVE,
            "chunk_size": 1000,
            "chunk_overlap": 100,
        },
        chunks=[ServiceChunkPreview(index=0, content="Cached chunk", metadata={})],
        total_chunks=1,
        processing_time_ms=50,
        cached=True,
        expires_at=datetime.now(UTC) + timedelta(minutes=15),
    )

    service.compare_strategies_for_api.return_value = ServiceCompareResponse(
        comparison_id="comp-123",
        comparisons=[
            ServiceStrategyComparison(
                strategy=ChunkingStrategy.RECURSIVE,
                config={
                    "strategy": ChunkingStrategy.RECURSIVE,
                    "chunk_size": 1000,
                    "chunk_overlap": 100,
                },
                sample_chunks=[],
                total_chunks=10,
                avg_chunk_size=500.0,
                size_variance=0.1,
                quality_score=0.85,
                processing_time_ms=100,
                pros=["Good for structured text"],
                cons=["May split sentences"],
            )
        ],
        recommendation=ServiceStrategyRecommendation(
            strategy=ChunkingStrategy.RECURSIVE,
            confidence=0.9,
            reasoning="Best balance of chunk size and count",
            alternatives=[],
            chunk_size=1000,
            chunk_overlap=100,
        ),
        processing_time_ms=100,
    )

    service.start_chunking_operation.return_value = (
        "chunking:coll-123:op-123",  # websocket_channel
        {"operation_id": "op-123", "status": "pending"},  # operation_info
    )

    service.get_chunking_progress.return_value = {
        "operation_id": "op-123",
        "status": "in_progress",
        "progress_percentage": 50.0,
        "documents_processed": 5,
        "total_documents": 10,
        "chunks_created": 50,
        "current_document": "doc-5.txt",
        "estimated_time_remaining": 60,
        "errors": [],
    }

    service.get_collection_chunk_stats.return_value = ServiceChunkingStats(
        total_chunks=100,
        total_documents=10,
        avg_chunk_size=500,
        min_chunk_size=100,
        max_chunk_size=1000,
        size_variance=0.2,
        strategy_used="fixed_size",
        last_updated=datetime.now(UTC),
        processing_time_seconds=120.0,
        quality_metrics={"quality": 0.85},
    )

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

        with pytest.raises(HTTPException) as exc_info:
            await list_strategies(
                _current_user=mock_user,
                service=mock_chunking_service,
            )
        assert exc_info.value.status_code == 500
        assert "Failed to list strategies" in exc_info.value.detail

    @pytest.mark.asyncio()
    async def test_get_strategy_details_success(
        self, mock_user: dict[str, Any], mock_chunking_service: AsyncMock
    ) -> None:
        """Test getting strategy details."""
        result = await get_strategy_details(
            strategy_id="recursive",
            _current_user=mock_user,
            service=mock_chunking_service,
        )

        assert result.id == "recursive"
        assert result.default_config is not None
        mock_chunking_service.get_strategy_details.assert_called_once_with("recursive")

    @pytest.mark.asyncio()
    async def test_get_strategy_details_not_found(
        self, mock_user: dict[str, Any], mock_chunking_service: AsyncMock
    ) -> None:
        """Test getting details for non-existent strategy."""
        mock_chunking_service.get_strategy_details.return_value = None

        with pytest.raises(HTTPException) as exc_info:
            await get_strategy_details(
                strategy_id="nonexistent",
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
            file_types=["txt", "pdf"],
            _current_user=mock_user,
            service=mock_chunking_service,
        )

        assert result.recommended_strategy == ChunkingStrategy.RECURSIVE
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
            config=ChunkingConfigBase(
                strategy=ChunkingStrategy.RECURSIVE,
                chunk_size=1000,
                chunk_overlap=100,
            ),
        )

        mock_request = create_autospec(Request, spec_set=True)
        result = await generate_preview(
            request=mock_request,
            preview_request=request,
            _current_user=mock_user,
            service=mock_chunking_service,
            correlation_id=None,
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
        mock_chunking_service.validate_preview_content.side_effect = ValidationError(
            field="chunk_size", value=-100, reason="chunk_size must be positive"
        )

        request = PreviewRequest(
            content="Test content",
            strategy=ChunkingStrategy.RECURSIVE,
            config=ChunkingConfigBase(
                strategy=ChunkingStrategy.RECURSIVE,
                chunk_size=100,  # Valid config, error will be raised from validate
                chunk_overlap=10,
            ),
        )

        mock_request = create_autospec(Request, spec_set=True)
        with pytest.raises(HTTPException) as exc_info:
            await generate_preview(
                request=mock_request,
                preview_request=request,
                _current_user=mock_user,
                service=mock_chunking_service,
                correlation_id=None,
            )
        assert exc_info.value.status_code == 400
        assert "chunk_size must be positive" in exc_info.value.detail

    @pytest.mark.asyncio()
    async def test_generate_preview_content_too_large(
        self, mock_user: dict[str, Any], mock_chunking_service: AsyncMock
    ) -> None:
        """Test preview with content too large."""
        mock_chunking_service.validate_preview_content.side_effect = DocumentTooLargeError(
            size=11_000_000, max_size=10_000_000
        )

        request = PreviewRequest(
            content="x" * 11_000_000,  # 11MB
            strategy=ChunkingStrategy.RECURSIVE,
        )

        mock_request = create_autospec(Request, spec_set=True)
        with pytest.raises(HTTPException) as exc_info:
            await generate_preview(
                request=mock_request,
                preview_request=request,
                _current_user=mock_user,
                service=mock_chunking_service,
                correlation_id=None,
            )
        assert exc_info.value.status_code == 507  # API uses 507 for too large
        assert "exceeds maximum" in exc_info.value.detail.lower()

    @pytest.mark.asyncio()
    async def test_get_cached_preview_success(
        self, mock_user: dict[str, Any], mock_chunking_service: AsyncMock
    ) -> None:
        """Test getting cached preview."""
        mock_request = create_autospec(Request, spec_set=True)
        result = await get_cached_preview(
            request=mock_request,
            preview_id="cached-preview-id",
            _current_user=mock_user,
            service=mock_chunking_service,
        )

        assert result.preview_id == "cached-preview-id"
        assert result.cached is True
        assert result.total_chunks == 1
        mock_chunking_service.get_cached_preview_by_id.assert_called_once()

    @pytest.mark.asyncio()
    async def test_get_cached_preview_not_found(
        self, mock_user: dict[str, Any], mock_chunking_service: AsyncMock
    ) -> None:
        """Test getting non-existent cached preview."""
        mock_chunking_service.get_cached_preview_by_id.return_value = None

        mock_request = create_autospec(Request, spec_set=True)
        with pytest.raises(HTTPException) as exc_info:
            await get_cached_preview(
                request=mock_request,
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

        mock_request = create_autospec(Request, spec_set=True)
        result = await compare_strategies(
            request=mock_request,
            compare_request=request,
            _current_user=mock_user,
            service=mock_chunking_service,
        )

        assert result.comparison_id == "comp-123"
        assert len(result.comparisons) == 1
        assert result.comparisons[0].strategy == ChunkingStrategy.RECURSIVE
        assert result.recommendation.recommended_strategy == ChunkingStrategy.RECURSIVE
        mock_chunking_service.compare_strategies_for_api.assert_called_once()


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
            config=ChunkingConfigBase(
                strategy=ChunkingStrategy.RECURSIVE,
                chunk_size=1000,
                chunk_overlap=100,
            ),
        )

        with patch("packages.webui.api.v2.chunking.BackgroundTasks") as mock_bg:
            mock_bg_instance = MagicMock()
            mock_bg.return_value = mock_bg_instance

            mock_request = create_autospec(Request, spec_set=True)
            mock_collection_service = AsyncMock(spec=CollectionService)
            mock_collection_service.create_operation.return_value = {
                "uuid": "op-123",
                "status": "pending",
            }
            mock_chunking_service.validate_config_for_collection.return_value = {
                "valid": True,
                "estimated_time": 60,
            }

            result = await start_chunking_operation(
                request=mock_request,
                collection_uuid="coll-123",
                chunking_request=request,
                background_tasks=mock_bg_instance,
                _current_user=mock_user,
                collection=mock_collection,
                service=mock_chunking_service,
                collection_service=mock_collection_service,
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
        assert result.documents_processed == 5
        assert result.chunks_created == 50
        mock_chunking_service.get_chunking_progress.assert_called_once_with(operation_id="op-123")

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
        mock_collection = {"id": "coll-123", "name": "Test Collection"}
        result = await get_chunking_stats(
            collection_id="coll-123",
            _current_user=mock_user,
            collection=mock_collection,
            service=mock_chunking_service,
        )

        assert result.total_chunks == 100
        assert result.total_documents == 10
        assert result.avg_chunk_size == 500
        assert result.strategy_used == ChunkingStrategy.FIXED_SIZE
        mock_chunking_service.get_collection_chunk_stats.assert_called_once_with(collection_id="coll-123")


class TestErrorHandling:
    """Test error handling in endpoints."""

    @pytest.mark.asyncio()
    async def test_application_error_handling(
        self, mock_user: dict[str, Any], mock_chunking_service: AsyncMock
    ) -> None:
        """Test that application errors are properly handled."""
        from packages.shared.chunking.infrastructure.exceptions import ApplicationError
        from packages.webui.api.v2.chunking import application_exception_handler

        error = ApplicationError(message="Test error", code="TEST_ERROR", correlation_id="test-123")
        request = MagicMock()

        response = await application_exception_handler(request, error)

        assert response.status_code == 500
        response_body = response.body.decode()
        assert "test-123" in response_body
        assert "Test error" in response_body
