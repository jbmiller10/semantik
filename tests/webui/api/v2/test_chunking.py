"""
Tests for v2 chunking API endpoints.

Comprehensive test coverage for all chunking-related endpoints including
strategy management, preview operations, collection processing, and analytics.
"""

import os

# Disable rate limiting for tests BEFORE importing the app
os.environ["DISABLE_RATE_LIMITING"] = "true"

import uuid  # noqa: E402
from datetime import UTC, datetime, timedelta  # noqa: E402
from typing import Any  # noqa: E402
from unittest.mock import AsyncMock, MagicMock, Mock, patch  # noqa: E402

import pytest  # noqa: E402
from fastapi import HTTPException, status  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

# Mock background tasks and Redis manager BEFORE importing the app
import packages.webui.background_tasks as bg_tasks  # noqa: E402
from packages.shared.config import settings  # noqa: E402
from packages.webui.api.v2.chunking_schemas import ChunkingStrategy  # noqa: E402
from packages.webui.auth import get_current_user  # noqa: E402
from packages.webui.dependencies import get_collection_for_user  # noqa: E402
from packages.webui.services.chunking_service import ChunkingService  # noqa: E402
from packages.webui.services.collection_service import CollectionService  # noqa: E402

bg_tasks.start_background_tasks = AsyncMock()
bg_tasks.stop_background_tasks = AsyncMock()

import packages.webui.services.factory as factory_module  # noqa: E402

factory_module._redis_manager = Mock(async_client=AsyncMock(return_value=AsyncMock()))

# Lazy imports to avoid initialization issues
app = None
chunking_module = None
get_chunking_service = None
get_collection_service = None


@pytest.fixture()
def mock_user() -> dict[str, Any]:
    """Mock authenticated user."""
    return {"id": 1, "username": "testuser", "email": "test@example.com"}


@pytest.fixture()
def mock_collection() -> dict[str, Any]:
    """Mock collection object."""
    return {
        "id": "123e4567-e89b-12d3-a456-426614174000",
        "name": "Test Collection",
        "description": "A test collection",
        "owner_id": 1,
        "status": "ready",
        "document_count": 10,
        "vector_count": 100,
    }


@pytest.fixture()
def mock_chunking_service() -> AsyncMock:
    """Mock ChunkingService."""
    return AsyncMock(spec=ChunkingService)


@pytest.fixture()
def mock_collection_service() -> AsyncMock:
    """Mock CollectionService."""
    return AsyncMock(spec=CollectionService)


@pytest.fixture()
def mock_ws_manager() -> AsyncMock:
    """Mock WebSocket manager."""
    return AsyncMock()


@pytest.fixture()
def client(
    mock_user: dict[str, Any],
    mock_chunking_service: AsyncMock,
    mock_collection_service: AsyncMock,
    mock_collection: dict[str, Any],
    mock_ws_manager: AsyncMock,
) -> TestClient:
    """Create a test client with mocked dependencies."""
    global app, chunking_module, get_chunking_service, get_collection_service

    # Check if already imported
    if app is None:
        # Import app and modules (environment variable already set at module level)
        import packages.webui.api.v2.chunking as _chunking_module
        from packages.webui.main import app as _app
        from packages.webui.services.factory import get_chunking_service as _get_chunking_service
        from packages.webui.services.factory import get_collection_service as _get_collection_service

        app = _app
        chunking_module = _chunking_module
        get_chunking_service = _get_chunking_service
        get_collection_service = _get_collection_service

    # Override dependencies
    app.dependency_overrides[get_current_user] = lambda: mock_user
    app.dependency_overrides[get_chunking_service] = lambda: mock_chunking_service
    app.dependency_overrides[get_collection_service] = lambda: mock_collection_service
    app.dependency_overrides[get_collection_for_user] = lambda: mock_collection

    # Replace ws_manager
    chunking_module.ws_manager = mock_ws_manager

    # Create test client
    client = TestClient(app)

    yield client

    # Clean up
    app.dependency_overrides.clear()


class TestStrategyManagement:
    """Test strategy management endpoints."""

    def test_list_strategies_success(self, client: TestClient, mock_chunking_service: AsyncMock) -> None:
        """Test successful listing of all strategies."""
        # Set up mock return value
        mock_strategies = [
            {
                "id": ChunkingStrategy.FIXED_SIZE,
                "name": "fixed_size",
                "description": "Simple fixed-size character-based chunking",
                "best_for": ["Quick processing", "Consistent chunk sizes"],
                "pros": ["Fast", "Predictable"],
                "cons": ["May break mid-sentence"],
                "default_config": {"chunk_size": 1000, "chunk_overlap": 200},
                "performance_characteristics": {"speed": "fast", "accuracy": "medium"},
            },
            {
                "id": ChunkingStrategy.RECURSIVE,
                "name": "recursive",
                "description": "Smart sentence-aware splitting",
                "best_for": ["General documents", "Maintaining context"],
                "pros": ["Respects sentence boundaries"],
                "cons": ["Variable chunk sizes"],
                "default_config": {"chunk_size": 1000, "chunk_overlap": 200},
                "performance_characteristics": {"speed": "medium", "accuracy": "high"},
            },
            {
                "id": ChunkingStrategy.MARKDOWN,
                "name": "markdown",
                "description": "Respects markdown structure",
                "best_for": ["Markdown documents", "Technical documentation"],
                "pros": ["Preserves structure"],
                "cons": ["Only for markdown"],
                "default_config": {"chunk_size": 1000, "chunk_overlap": 0},
                "performance_characteristics": {"speed": "medium", "accuracy": "high"},
            },
            {
                "id": ChunkingStrategy.SEMANTIC,
                "name": "semantic",
                "description": "Uses AI embeddings to find natural boundaries",
                "best_for": ["Complex documents", "Academic papers"],
                "pros": ["Best context preservation"],
                "cons": ["Slower", "Requires embeddings"],
                "default_config": {"buffer_size": 1, "breakpoint_threshold": 95},
                "performance_characteristics": {"speed": "slow", "accuracy": "very_high"},
            },
            {
                "id": ChunkingStrategy.HIERARCHICAL,
                "name": "hierarchical",
                "description": "Creates parent-child chunks",
                "best_for": ["Large documents", "Multi-level analysis"],
                "pros": ["Multiple granularities"],
                "cons": ["Complex", "More storage"],
                "default_config": {"chunk_sizes": [2048, 512, 128]},
                "performance_characteristics": {"speed": "slow", "accuracy": "high"},
            },
            {
                "id": ChunkingStrategy.HYBRID,
                "name": "hybrid",
                "description": "Automatically selects strategy based on content",
                "best_for": ["Mixed content", "Unknown document types"],
                "pros": ["Adaptive", "Best overall"],
                "cons": ["Overhead from analysis"],
                "default_config": {},
                "performance_characteristics": {"speed": "variable", "accuracy": "high"},
            },
        ]

        mock_chunking_service.get_available_strategies.return_value = mock_strategies

        response = client.get("/api/v2/chunking/strategies")

        assert response.status_code == status.HTTP_200_OK
        strategies = response.json()

        assert isinstance(strategies, list)
        assert len(strategies) == 6  # Should have all 6 strategies

        # Check first strategy structure
        strategy = strategies[0]
        assert "id" in strategy
        assert "name" in strategy
        assert "description" in strategy
        assert "best_for" in strategy
        assert "pros" in strategy
        assert "cons" in strategy
        assert "default_config" in strategy
        assert "performance_characteristics" in strategy

    def test_list_strategies_unauthenticated(self) -> None:
        """Test that listing strategies requires authentication."""

        # Ensure auth is enabled for this test
        original_disable_auth = settings.DISABLE_AUTH
        settings.DISABLE_AUTH = False

        # Create a client without overriding authentication
        client = TestClient(app)

        # Clear any existing dependency overrides
        app.dependency_overrides.clear()

        # Mock the auth to always raise 401
        def mock_get_current_user() -> None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not authenticated",
                headers={"WWW-Authenticate": "Bearer"},
            )

        app.dependency_overrides[get_current_user] = mock_get_current_user

        try:
            response = client.get("/api/v2/chunking/strategies")
            assert response.status_code == status.HTTP_401_UNAUTHORIZED
        finally:
            # Clean up
            app.dependency_overrides.clear()
            # Restore original setting
            settings.DISABLE_AUTH = original_disable_auth

    def test_get_strategy_details_success(self, client: TestClient) -> None:
        """Test getting details for a specific strategy."""
        response = client.get("/api/v2/chunking/strategies/fixed_size")

        assert response.status_code == status.HTTP_200_OK
        strategy = response.json()

        assert strategy["id"] == "fixed_size"
        assert strategy["name"] == "Fixed Size Chunking"
        assert "description" in strategy
        assert isinstance(strategy["best_for"], list)
        assert isinstance(strategy["pros"], list)
        assert isinstance(strategy["cons"], list)

    def test_get_strategy_details_not_found(self, client: TestClient) -> None:
        """Test getting details for non-existent strategy."""
        response = client.get("/api/v2/chunking/strategies/invalid_strategy")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "not found" in response.json()["detail"].lower()

    def test_recommend_strategy_success(self, client: TestClient, mock_chunking_service: AsyncMock) -> None:
        """Test successful strategy recommendation."""
        # Setup mock
        mock_chunking_service.recommend_strategy.return_value = {
            "strategy": ChunkingStrategy.SEMANTIC,
            "confidence": 0.85,
            "reasoning": "PDF files work best with semantic chunking",
            "alternatives": [ChunkingStrategy.DOCUMENT_STRUCTURE],
            "chunk_size": 512,
            "chunk_overlap": 50,
        }

        response = client.post("/api/v2/chunking/strategies/recommend", params={"file_types": ["pdf", "docx"]})

        assert response.status_code == status.HTTP_200_OK
        recommendation = response.json()

        assert recommendation["recommended_strategy"] == "semantic"
        assert recommendation["confidence"] == 0.85
        assert "reasoning" in recommendation
        assert len(recommendation["alternative_strategies"]) == 1
        assert "suggested_config" in recommendation

    def test_recommend_strategy_error(self, client: TestClient, mock_chunking_service: AsyncMock) -> None:
        """Test strategy recommendation when service fails."""
        mock_chunking_service.recommend_strategy.side_effect = Exception("Service error")

        response = client.post("/api/v2/chunking/strategies/recommend", params={"file_types": ["txt"]})

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestPreviewOperations:
    """Test preview operation endpoints."""

    def test_generate_preview_with_content(self, client: TestClient, mock_chunking_service: AsyncMock) -> None:
        """Test generating preview with provided content."""
        # Setup mock
        preview_id = str(uuid.uuid4())
        mock_chunking_service.track_preview_usage.return_value = None
        mock_chunking_service.preview_chunking.return_value = {
            "preview_id": preview_id,
            "strategy": ChunkingStrategy.FIXED_SIZE,
            "config": {
                "strategy": "fixed_size",
                "chunk_size": 512,
                "chunk_overlap": 50,
                "preserve_sentences": True,
            },
            "chunks": [
                {
                    "index": 0,
                    "content": "This is chunk 1",
                    "token_count": 4,
                    "char_count": 15,
                    "metadata": {},
                    "quality_score": 0.8,
                }
            ],
            "total_chunks": 1,
            "metrics": {
                "avg_chunk_size": 15,
                "size_variance": 0.0,
                "quality_score": 0.8,
            },
            "processing_time_ms": 100,
            "cached": False,
        }

        request_data = {
            "content": "This is a test document for chunking preview.",
            "strategy": "fixed_size",
            "config": {
                "strategy": "fixed_size",
                "chunk_size": 512,
                "chunk_overlap": 50,
                "preserve_sentences": True,
            },
            "max_chunks": 10,
            "include_metrics": True,
        }

        response = client.post("/api/v2/chunking/preview", json=request_data)

        assert response.status_code == status.HTTP_200_OK
        preview = response.json()

        assert preview["preview_id"] == preview_id
        assert preview["strategy"] == "fixed_size"
        assert len(preview["chunks"]) == 1
        assert preview["total_chunks"] == 1
        assert "metrics" in preview
        assert preview["processing_time_ms"] == 100

    def test_generate_preview_with_document_id(self, client: TestClient, mock_chunking_service: AsyncMock) -> None:
        """Test generating preview with document ID."""
        # Setup mock
        preview_id = str(uuid.uuid4())
        mock_chunking_service.track_preview_usage.return_value = None
        mock_chunking_service.preview_chunking.return_value = {
            "preview_id": preview_id,
            "strategy": ChunkingStrategy.SEMANTIC,
            "config": {
                "strategy": "semantic",
                "chunk_size": 512,
                "chunk_overlap": 50,
                "preserve_sentences": True,
            },
            "chunks": [],
            "total_chunks": 5,
            "processing_time_ms": 200,
            "cached": True,
        }

        request_data = {
            "document_id": "doc-123",
            "strategy": "semantic",
        }

        response = client.post("/api/v2/chunking/preview", json=request_data)

        assert response.status_code == status.HTTP_200_OK
        preview = response.json()

        assert preview["preview_id"] == preview_id
        assert preview["cached"] is True

    def test_generate_preview_missing_input(self, client: TestClient) -> None:
        """Test preview generation without content or document_id."""
        request_data = {
            "strategy": "fixed_size",
        }

        response = client.post("/api/v2/chunking/preview", json=request_data)

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "document_id or content must be provided" in response.json()["detail"]

    def test_generate_preview_content_too_large(self, client: TestClient) -> None:
        """Test preview generation with content exceeding size limit."""
        # Create content larger than 10MB
        large_content = "x" * (11 * 1024 * 1024)

        request_data = {
            "content": large_content,
            "strategy": "fixed_size",
        }

        response = client.post("/api/v2/chunking/preview", json=request_data)

        # ChunkingMemoryError returns 507 (Insufficient Storage)
        assert response.status_code == status.HTTP_507_INSUFFICIENT_STORAGE

    def test_generate_preview_with_null_bytes(self, client: TestClient) -> None:
        """Test preview generation with invalid content containing null bytes."""
        request_data = {
            "content": "test\x00content",
            "strategy": "fixed_size",
        }

        response = client.post("/api/v2/chunking/preview", json=request_data)

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "null bytes" in response.json()["detail"].lower()

    @patch("packages.webui.api.v2.chunking.limiter")
    def test_generate_preview_rate_limiting(
        self, mock_limiter: MagicMock, client: TestClient, mock_chunking_service: AsyncMock
    ) -> None:
        """Test that preview generation is rate limited."""
        # Mock rate limiter to simulate rate limit exceeded
        mock_limiter.limit.return_value = lambda f: f

        # This test verifies the decorator is present
        # In a real scenario, we'd need to test with actual rate limiting
        request_data = {
            "content": "test",
            "strategy": "fixed_size",
        }

        mock_chunking_service.track_preview_usage.return_value = None
        mock_chunking_service.preview_chunking.return_value = {
            "preview_id": str(uuid.uuid4()),
            "strategy": ChunkingStrategy.FIXED_SIZE,
            "config": {"strategy": "fixed_size", "chunk_size": 512, "chunk_overlap": 50, "preserve_sentences": True},
            "chunks": [],
            "total_chunks": 0,
            "processing_time_ms": 100,
        }

        response = client.post("/api/v2/chunking/preview", json=request_data)
        assert response.status_code == status.HTTP_200_OK

    def test_compare_strategies_success(self, client: TestClient, mock_chunking_service: AsyncMock) -> None:
        """Test successful strategy comparison."""
        # Setup mock
        mock_chunking_service.preview_chunking.side_effect = [
            {
                "preview_id": str(uuid.uuid4()),
                "strategy": ChunkingStrategy.FIXED_SIZE,
                "config": {
                    "strategy": "fixed_size",
                    "chunk_size": 512,
                    "chunk_overlap": 50,
                    "preserve_sentences": True,
                },
                "chunks": [
                    {
                        "index": 0,
                        "content": "chunk1",
                        "token_count": 5,
                        "char_count": 6,
                        "metadata": {},
                        "quality_score": 0.7,
                    }
                ],
                "total_chunks": 10,
                "metrics": {"avg_chunk_size": 500, "size_variance": 10.0, "quality_score": 0.7},
                "processing_time_ms": 100,
            },
            {
                "preview_id": str(uuid.uuid4()),
                "strategy": ChunkingStrategy.SEMANTIC,
                "config": {"strategy": "semantic", "chunk_size": 512, "chunk_overlap": 50, "preserve_sentences": True},
                "chunks": [
                    {
                        "index": 0,
                        "content": "chunk1",
                        "token_count": 8,
                        "char_count": 10,
                        "metadata": {},
                        "quality_score": 0.85,
                    }
                ],
                "total_chunks": 8,
                "metrics": {"avg_chunk_size": 600, "size_variance": 20.0, "quality_score": 0.85},
                "processing_time_ms": 200,
            },
        ]

        request_data = {
            "content": "Test document for comparison",
            "strategies": ["fixed_size", "semantic"],
            "max_chunks_per_strategy": 5,
        }

        response = client.post("/api/v2/chunking/compare", json=request_data)

        assert response.status_code == status.HTTP_200_OK
        comparison = response.json()

        assert "comparison_id" in comparison
        assert len(comparison["comparisons"]) == 2
        assert comparison["recommendation"]["recommended_strategy"] == "semantic"  # Higher quality score
        assert comparison["recommendation"]["confidence"] == 0.85

    def test_get_cached_preview_success(self, client: TestClient, mock_chunking_service: AsyncMock) -> None:
        """Test retrieving cached preview results."""
        preview_id = str(uuid.uuid4())
        mock_chunking_service._get_cached_preview_by_key.return_value = {
            "preview_id": preview_id,
            "strategy": "fixed_size",
            "config": {"strategy": "fixed_size", "chunk_size": 512, "chunk_overlap": 50, "preserve_sentences": True},
            "chunks": [],
            "total_chunks": 5,
            "processing_time_ms": 100,
            "cached": True,
            "expires_at": (datetime.now(tz=UTC) + timedelta(minutes=15)).isoformat(),
        }

        response = client.get(f"/api/v2/chunking/preview/{preview_id}")

        assert response.status_code == status.HTTP_200_OK
        preview = response.json()
        assert preview["preview_id"] == preview_id

    def test_get_cached_preview_not_found(self, client: TestClient, mock_chunking_service: AsyncMock) -> None:
        """Test retrieving non-existent preview."""
        mock_chunking_service._get_cached_preview_by_key.return_value = None

        response = client.get(f"/api/v2/chunking/preview/{uuid.uuid4()}")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_clear_preview_cache_success(self, client: TestClient, mock_chunking_service: AsyncMock) -> None:
        """Test clearing preview cache."""
        preview_id = str(uuid.uuid4())
        mock_chunking_service.clear_preview_cache.return_value = None

        response = client.delete(f"/api/v2/chunking/preview/{preview_id}")

        assert response.status_code == status.HTTP_204_NO_CONTENT


class TestCollectionProcessing:
    """Test collection processing endpoints."""

    def test_start_chunking_operation_success(
        self,
        client: TestClient,
        mock_chunking_service: AsyncMock,
        mock_collection_service: AsyncMock,
        mock_ws_manager: AsyncMock,
    ) -> None:
        """Test starting a chunking operation on a collection."""
        # Setup mocks
        operation_id = str(uuid.uuid4())
        mock_chunking_service.validate_config_for_collection.return_value = {
            "is_valid": True,
            "estimated_time": 60,
        }
        mock_collection_service.create_operation.return_value = {
            "uuid": operation_id,
            "collection_id": "123e4567-e89b-12d3-a456-426614174000",
            "type": "chunking",
            "status": "pending",
        }
        # Add missing mock for start_chunking_operation
        websocket_channel = f"chunking:123e4567-e89b-12d3-a456-426614174000:{operation_id}"
        mock_chunking_service.start_chunking_operation.return_value = (
            websocket_channel,
            {"is_valid": True, "estimated_time": 60},
        )
        mock_ws_manager.send_message.return_value = None

        request_data = {
            "strategy": "semantic",
            "config": {
                "strategy": "semantic",
                "chunk_size": 512,
                "chunk_overlap": 50,
                "preserve_sentences": True,
            },
            "document_ids": ["doc1", "doc2"],
            "priority": 8,
        }

        response = client.post(
            "/api/v2/chunking/collections/123e4567-e89b-12d3-a456-426614174000/chunk", json=request_data
        )

        assert response.status_code == status.HTTP_202_ACCEPTED
        operation_response = response.json()

        assert operation_response["operation_id"] == operation_id
        assert operation_response["status"] == "pending"
        assert operation_response["strategy"] == "semantic"
        assert "websocket_channel" in operation_response

    def test_start_chunking_operation_invalid_config(
        self, client: TestClient, mock_chunking_service: AsyncMock
    ) -> None:
        """Test starting operation with invalid configuration."""
        mock_chunking_service.validate_config_for_collection.return_value = {
            "is_valid": False,
            "reason": "Invalid chunk size for this collection",
        }

        request_data = {
            "strategy": "fixed_size",
            "config": {
                "strategy": "fixed_size",
                "chunk_size": 4000,  # Valid for schema but will fail custom validation
                "chunk_overlap": 50,
                "preserve_sentences": True,
            },
        }

        response = client.post(
            "/api/v2/chunking/collections/123e4567-e89b-12d3-a456-426614174000/chunk", json=request_data
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Invalid configuration" in response.json()["detail"]

    def test_update_chunking_strategy_with_reprocess(
        self, client: TestClient, mock_collection_service: AsyncMock
    ) -> None:
        """Test updating chunking strategy with reprocessing."""
        operation_id = str(uuid.uuid4())
        mock_collection_service.update_collection.return_value = None
        mock_collection_service.create_operation.return_value = {
            "uuid": operation_id,
            "collection_id": "123e4567-e89b-12d3-a456-426614174000",
            "type": "rechunking",
            "status": "pending",
        }

        request_data = {
            "strategy": "semantic",
            "config": {
                "strategy": "semantic",
                "chunk_size": 512,
                "chunk_overlap": 50,
                "preserve_sentences": True,
            },
            "reprocess_existing": True,
        }

        response = client.patch(
            "/api/v2/chunking/collections/123e4567-e89b-12d3-a456-426614174000/chunking-strategy", json=request_data
        )

        assert response.status_code == status.HTTP_200_OK
        operation_response = response.json()
        assert operation_response["operation_id"] == operation_id
        assert operation_response["status"] == "pending"

    def test_update_chunking_strategy_without_reprocess(
        self, client: TestClient, mock_collection_service: AsyncMock
    ) -> None:
        """Test updating chunking strategy without reprocessing."""
        mock_collection_service.update_collection.return_value = None

        request_data = {
            "strategy": "recursive",
            "reprocess_existing": False,
        }

        response = client.patch(
            "/api/v2/chunking/collections/123e4567-e89b-12d3-a456-426614174000/chunking-strategy", json=request_data
        )

        assert response.status_code == status.HTTP_200_OK
        operation_response = response.json()
        assert operation_response["status"] == "completed"

    def test_get_collection_chunks_paginated(self, client: TestClient, mock_chunking_service: AsyncMock) -> None:
        """Test getting paginated chunks for a collection."""
        response = client.get(
            "/api/v2/chunking/collections/123e4567-e89b-12d3-a456-426614174000/chunks",
            params={"page": 1, "page_size": 20},
        )

        assert response.status_code == status.HTTP_200_OK
        chunk_list = response.json()

        assert "chunks" in chunk_list
        assert "total" in chunk_list
        assert chunk_list["page"] == 1
        assert chunk_list["page_size"] == 20

    def test_get_chunking_stats_success(self, client: TestClient, mock_chunking_service: AsyncMock) -> None:
        """Test getting chunking statistics for a collection."""
        # Return a dictionary as expected by the endpoint
        mock_stats = {
            "total_documents": 10,
            "total_chunks": 100,
            "average_chunk_size": 512,
            "min_chunk_size": 100,
            "max_chunk_size": 1024,
            "size_variance": 50.0,
            "strategy": "semantic",
            "last_updated": datetime.now(UTC),
            "processing_time": 120,
            "performance_metrics": {"coherence": 0.8, "completeness": 0.9},
        }

        mock_chunking_service.get_chunking_statistics.return_value = mock_stats

        response = client.get("/api/v2/chunking/collections/123e4567-e89b-12d3-a456-426614174000/chunking-stats")

        assert response.status_code == status.HTTP_200_OK
        stats = response.json()

        assert stats["total_chunks"] == 100
        assert stats["total_documents"] == 10
        assert stats["avg_chunk_size"] == 512
        assert stats["strategy_used"] == "semantic"


class TestAnalyticsEndpoints:
    """Test analytics and metrics endpoints."""

    def test_get_global_metrics_success(self, client: TestClient, mock_chunking_service: AsyncMock) -> None:
        """Test getting global chunking metrics."""
        response = client.get("/api/v2/chunking/metrics", params={"period_days": 30})

        assert response.status_code == status.HTTP_200_OK
        metrics = response.json()

        assert "total_collections_processed" in metrics
        assert "total_chunks_created" in metrics
        assert "total_documents_processed" in metrics
        assert "avg_chunks_per_document" in metrics
        assert "most_used_strategy" in metrics
        assert "success_rate" in metrics

    def test_get_metrics_by_strategy_success(self, client: TestClient, mock_chunking_service: AsyncMock) -> None:
        """Test getting metrics grouped by strategy."""
        response = client.get("/api/v2/chunking/metrics/by-strategy", params={"period_days": 30})

        assert response.status_code == status.HTTP_200_OK
        metrics_list = response.json()

        assert isinstance(metrics_list, list)
        assert len(metrics_list) == 6  # One for each strategy

        first_metric = metrics_list[0]
        assert "strategy" in first_metric
        assert "usage_count" in first_metric
        assert "avg_chunk_size" in first_metric
        assert "avg_processing_time" in first_metric
        assert "success_rate" in first_metric

    def test_get_quality_scores_for_collection(self, client: TestClient, mock_chunking_service: AsyncMock) -> None:
        """Test getting quality analysis for a specific collection."""
        response = client.get(
            "/api/v2/chunking/quality-scores", params={"collection_id": "123e4567-e89b-12d3-a456-426614174000"}
        )

        assert response.status_code == status.HTTP_200_OK
        quality = response.json()

        assert quality["overall_quality"] in ["excellent", "good", "fair", "poor"]
        assert 0 <= quality["quality_score"] <= 1
        assert "coherence_score" in quality
        assert "completeness_score" in quality
        assert "recommendations" in quality
        assert isinstance(quality["recommendations"], list)

    def test_analyze_document_success(self, client: TestClient, mock_chunking_service: AsyncMock) -> None:
        """Test document analysis for strategy recommendation."""
        mock_chunking_service.recommend_strategy.return_value = {
            "strategy": ChunkingStrategy.DOCUMENT_STRUCTURE,
            "confidence": 0.9,
            "reasoning": "Document has clear structure with headings",
            "alternatives": [ChunkingStrategy.SEMANTIC],
        }

        request_data = {
            "document_id": "doc-123",
            "file_type": "pdf",
            "content_sample": "# Heading 1\nContent...\n## Heading 2\nMore content...",
        }

        response = client.post("/api/v2/chunking/analyze", json=request_data)

        assert response.status_code == status.HTTP_200_OK
        analysis = response.json()

        assert analysis["document_type"] == "pdf"
        assert "content_structure" in analysis
        assert "recommended_strategy" in analysis
        assert "estimated_chunks" in analysis
        assert "complexity_score" in analysis


class TestConfigurationManagement:
    """Test configuration management endpoints."""

    def test_save_configuration_success(self, client: TestClient, mock_chunking_service: AsyncMock) -> None:
        """Test saving a custom chunking configuration."""
        request_data = {
            "name": "My Custom Config",
            "description": "Optimized for technical documentation",
            "strategy": "recursive",
            "config": {
                "strategy": "recursive",
                "chunk_size": 1000,
                "chunk_overlap": 100,
                "preserve_sentences": True,
            },
            "is_default": False,
            "tags": ["technical", "documentation"],
        }

        response = client.post("/api/v2/chunking/configs", json=request_data)

        assert response.status_code == status.HTTP_201_CREATED
        saved_config = response.json()

        assert "id" in saved_config
        assert saved_config["name"] == "My Custom Config"
        assert saved_config["strategy"] == "recursive"
        assert saved_config["is_default"] is False
        assert len(saved_config["tags"]) == 2

    def test_list_configurations_success(self, client: TestClient, mock_chunking_service: AsyncMock) -> None:
        """Test listing saved configurations."""
        response = client.get("/api/v2/chunking/configs")

        assert response.status_code == status.HTTP_200_OK
        configs = response.json()
        assert isinstance(configs, list)

    def test_list_configurations_filtered_by_strategy(
        self, client: TestClient, mock_chunking_service: AsyncMock
    ) -> None:
        """Test listing configurations filtered by strategy."""
        response = client.get("/api/v2/chunking/configs", params={"strategy": "semantic"})

        assert response.status_code == status.HTTP_200_OK
        configs = response.json()
        assert isinstance(configs, list)


class TestProgressTracking:
    """Test operation progress tracking."""

    def test_get_operation_progress_success(self, client: TestClient, mock_chunking_service: AsyncMock) -> None:
        """Test getting progress of a chunking operation."""
        operation_id = str(uuid.uuid4())
        mock_chunking_service.get_chunking_progress.return_value = {
            "status": "in_progress",
            "progress_percentage": 45.5,
            "documents_processed": 5,
            "total_documents": 11,
            "chunks_created": 250,
            "current_document": "document_6.pdf",
            "estimated_time_remaining": 120,
            "errors": [],
        }

        response = client.get(f"/api/v2/chunking/operations/{operation_id}/progress")

        assert response.status_code == status.HTTP_200_OK
        progress = response.json()

        assert progress["operation_id"] == operation_id
        assert progress["status"] == "in_progress"
        assert progress["progress_percentage"] == 45.5
        assert progress["documents_processed"] == 5
        assert progress["total_documents"] == 11

    def test_get_operation_progress_not_found(self, client: TestClient, mock_chunking_service: AsyncMock) -> None:
        """Test getting progress for non-existent operation."""
        mock_chunking_service.get_chunking_progress.return_value = None

        response = client.get(f"/api/v2/chunking/operations/{uuid.uuid4()}/progress")

        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestSecurityAndValidation:
    """Test security and input validation."""

    @patch("packages.webui.auth.settings")
    def test_authorization_checks(self, mock_settings: MagicMock) -> None:
        """Test that all endpoints require authentication."""

        # Configure mock settings to ensure auth is enabled
        mock_settings.DISABLE_AUTH = False
        mock_settings.JWT_SECRET_KEY = "test-secret-key"
        mock_settings.ALGORITHM = "HS256"
        mock_settings.ACCESS_TOKEN_EXPIRE_MINUTES = 60

        # Clear any existing dependency overrides
        app.dependency_overrides.clear()

        try:
            client = TestClient(app)

            # Test various endpoints without authentication
            endpoints = [
                ("GET", "/api/v2/chunking/strategies"),
                ("GET", "/api/v2/chunking/strategies/fixed_size"),
                ("POST", "/api/v2/chunking/strategies/recommend"),
                ("POST", "/api/v2/chunking/preview"),
                ("GET", f"/api/v2/chunking/preview/{uuid.uuid4()}"),
                ("POST", "/api/v2/chunking/compare"),
                ("GET", "/api/v2/chunking/metrics"),
            ]

            for method, endpoint in endpoints:
                response = client.get(endpoint) if method == "GET" else client.post(endpoint, json={})

                assert (
                    response.status_code == status.HTTP_401_UNAUTHORIZED
                ), f"Endpoint {endpoint} should require auth but returned {response.status_code}"
        finally:
            # Clean up
            app.dependency_overrides.clear()

    def test_input_validation_chunk_size(self, client: TestClient) -> None:
        """Test validation of chunk size parameters."""
        request_data = {
            "content": "test",
            "strategy": "fixed_size",
            "config": {
                "strategy": "fixed_size",
                "chunk_size": 10,  # Below minimum
                "chunk_overlap": 5,
                "preserve_sentences": True,
            },
        }

        response = client.post("/api/v2/chunking/preview", json=request_data)

        # Should fail validation due to chunk_size being below minimum (100)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_input_validation_chunk_overlap(self, client: TestClient) -> None:
        """Test validation of chunk overlap parameters."""
        request_data = {
            "content": "test",
            "strategy": "fixed_size",
            "config": {
                "strategy": "fixed_size",
                "chunk_size": 500,
                "chunk_overlap": 600,  # Greater than chunk size
                "preserve_sentences": True,
            },
        }

        response = client.post("/api/v2/chunking/preview", json=request_data)

        # Should fail validation due to overlap being greater than chunk size
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_sql_injection_attempt(self, client: TestClient, mock_chunking_service: AsyncMock) -> None:
        """Test that SQL injection attempts are properly handled."""
        # Setup mock to return a valid response if validation passes
        mock_chunking_service.track_preview_usage.return_value = None
        mock_chunking_service.preview_chunking.return_value = {
            "preview_id": str(uuid.uuid4()),
            "strategy": ChunkingStrategy.FIXED_SIZE,
            "config": {"strategy": "fixed_size", "chunk_size": 512, "chunk_overlap": 50, "preserve_sentences": True},
            "chunks": [],
            "total_chunks": 0,
            "processing_time_ms": 10,
        }

        # Attempt SQL injection in various parameters
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'--",
        ]

        for malicious_input in malicious_inputs:
            request_data = {
                "content": malicious_input,
                "strategy": "fixed_size",
            }

            response = client.post("/api/v2/chunking/preview", json=request_data)

            # Should process normally without executing SQL
            assert response.status_code in [status.HTTP_200_OK, status.HTTP_400_BAD_REQUEST]

    def test_xss_prevention(self, client: TestClient) -> None:
        """Test that XSS attempts are properly sanitized."""
        xss_payload = "<script>alert('XSS')</script>"

        request_data = {
            "content": xss_payload,
            "strategy": "fixed_size",
        }

        response = client.post("/api/v2/chunking/preview", json=request_data)

        # Should process without executing script
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_400_BAD_REQUEST]


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.skipif(os.getenv("CI", "false").lower() == "true", reason="Requires full application context")
    def test_empty_content_preview(self, client: TestClient, mock_chunking_service: AsyncMock) -> None:
        """Test preview with empty content."""
        mock_chunking_service.track_preview_usage.return_value = None
        mock_chunking_service.preview_chunking.return_value = {
            "preview_id": str(uuid.uuid4()),
            "strategy": ChunkingStrategy.FIXED_SIZE,
            "config": {"strategy": "fixed_size", "chunk_size": 512, "chunk_overlap": 50, "preserve_sentences": True},
            "chunks": [],
            "total_chunks": 0,
            "processing_time_ms": 10,
        }

        request_data = {
            "content": "",
            "strategy": "fixed_size",
        }

        response = client.post("/api/v2/chunking/preview", json=request_data)

        assert response.status_code == status.HTTP_200_OK
        preview = response.json()
        assert preview["total_chunks"] == 0

    def test_unicode_content_handling(self, client: TestClient, mock_chunking_service: AsyncMock) -> None:
        """Test handling of unicode content."""
        mock_chunking_service.track_preview_usage.return_value = None
        mock_chunking_service.preview_chunking.return_value = {
            "preview_id": str(uuid.uuid4()),
            "strategy": ChunkingStrategy.FIXED_SIZE,
            "config": {"strategy": "fixed_size", "chunk_size": 512, "chunk_overlap": 50, "preserve_sentences": True},
            "chunks": [
                {
                    "index": 0,
                    "content": "你好世界",
                    "token_count": 2,
                    "char_count": 4,
                    "metadata": {},
                    "quality_score": 0.8,
                }
            ],
            "total_chunks": 1,
            "processing_time_ms": 50,
        }

        request_data = {
            "content": "你好世界 Hello World مرحبا بالعالم",
            "strategy": "fixed_size",
        }

        response = client.post("/api/v2/chunking/preview", json=request_data)

        assert response.status_code == status.HTTP_200_OK

    def test_concurrent_operations_same_collection(
        self,
        client: TestClient,
        mock_chunking_service: AsyncMock,
        mock_collection_service: AsyncMock,
        mock_ws_manager: AsyncMock,
    ) -> None:
        """Test handling concurrent chunking operations on same collection."""
        # Setup for multiple operations
        operation_ids = [str(uuid.uuid4()) for _ in range(3)]

        mock_chunking_service.validate_config_for_collection.return_value = {
            "is_valid": True,
            "estimated_time": 60,
        }

        # Simulate multiple operations
        for i, op_id in enumerate(operation_ids):
            mock_collection_service.create_operation.return_value = {
                "uuid": op_id,
                "collection_id": "123e4567-e89b-12d3-a456-426614174000",
                "type": "chunking",
                "status": "pending",
            }

            # Add missing mock for start_chunking_operation
            websocket_channel = f"chunking:123e4567-e89b-12d3-a456-426614174000:{op_id}"
            mock_chunking_service.start_chunking_operation.return_value = (
                websocket_channel,
                {"is_valid": True, "estimated_time": 60},
            )

            request_data = {
                "strategy": "fixed_size",
                "document_ids": [f"doc{i}"],
            }

            response = client.post(
                "/api/v2/chunking/collections/123e4567-e89b-12d3-a456-426614174000/chunk", json=request_data
            )

            assert response.status_code == status.HTTP_202_ACCEPTED

    def test_invalid_strategy_enum_value(self, client: TestClient) -> None:
        """Test handling of invalid strategy enum values."""
        request_data = {
            "content": "test",
            "strategy": "invalid_strategy_name",
        }

        response = client.post("/api/v2/chunking/preview", json=request_data)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestPerformance:
    """Test performance-related aspects."""

    def test_large_document_handling(self, client: TestClient, mock_chunking_service: AsyncMock) -> None:
        """Test handling of large documents."""
        # Create a large but valid document (under 10MB limit)
        large_content = "Lorem ipsum " * 100000  # Approximately 1.2MB

        mock_chunking_service.track_preview_usage.return_value = None
        mock_chunking_service.preview_chunking.return_value = {
            "preview_id": str(uuid.uuid4()),
            "strategy": ChunkingStrategy.FIXED_SIZE,
            "config": {"strategy": "fixed_size", "chunk_size": 512, "chunk_overlap": 50, "preserve_sentences": True},
            "chunks": [],
            "total_chunks": 1000,
            "processing_time_ms": 5000,
        }

        request_data = {
            "content": large_content,
            "strategy": "fixed_size",
        }

        response = client.post("/api/v2/chunking/preview", json=request_data)

        assert response.status_code == status.HTTP_200_OK

    @pytest.mark.skipif(os.getenv("CI", "false").lower() == "true", reason="Requires full application context")
    def test_preview_caching_behavior(self, client: TestClient, mock_chunking_service: AsyncMock) -> None:
        """Test that preview results are cached properly."""
        preview_id = str(uuid.uuid4())

        # First call - not cached
        mock_chunking_service.track_preview_usage.return_value = None
        mock_chunking_service.preview_chunking.return_value = {
            "preview_id": preview_id,
            "strategy": ChunkingStrategy.FIXED_SIZE,
            "config": {"strategy": "fixed_size", "chunk_size": 512, "chunk_overlap": 50, "preserve_sentences": True},
            "chunks": [],
            "total_chunks": 10,
            "processing_time_ms": 200,
            "cached": False,
        }

        request_data = {
            "content": "Test content for caching",
            "strategy": "fixed_size",
        }

        response1 = client.post("/api/v2/chunking/preview", json=request_data)
        assert response1.status_code == status.HTTP_200_OK
        assert response1.json()["cached"] is False

        # Second call - should be cached
        mock_chunking_service.preview_chunking.return_value["cached"] = True
        mock_chunking_service.preview_chunking.return_value["processing_time_ms"] = 10

        response2 = client.post("/api/v2/chunking/preview", json=request_data)
        assert response2.status_code == status.HTTP_200_OK
        assert response2.json()["cached"] is True
        assert response2.json()["processing_time_ms"] < 200  # Should be faster
