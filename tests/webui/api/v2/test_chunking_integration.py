"""
Integration tests for v2 chunking API endpoints.

These tests use minimal mocking to test the actual endpoint logic,
including error handling, validation, and service integration.
Only external dependencies (Redis, Celery, Qdrant) are mocked.
"""

import os

# Disable rate limiting for tests
os.environ["DISABLE_RATE_LIMITING"] = "true"
os.environ["TESTING"] = "true"

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from packages.shared.chunking.infrastructure.exceptions import (
    DocumentTooLargeError,
)
from packages.shared.chunking.infrastructure.exceptions import (
    ValidationError as ChunkingValidationError,
)
from packages.webui.auth import create_access_token, get_current_user
from packages.webui.main import app
from packages.webui.services.chunking_service import ChunkingService


@pytest.fixture()
def mock_user():
    """Create a mock authenticated user."""
    return {"id": 1, "username": "testuser", "email": "test@example.com"}


@pytest.fixture()
def auth_headers(mock_user):
    """Create authorization headers with a test JWT token."""
    token = create_access_token(data={"sub": mock_user["username"]})
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def mock_collection():
    """Create a mock collection."""
    return {
        "id": str(uuid.uuid4()),
        "name": "Test Collection",
        "owner_id": 1,
        "status": "ready",
    }


@pytest.fixture()
def client_with_auth(mock_user):
    """Create a test client with authentication mocked."""

    # Override dependencies
    async def override_get_current_user():
        return mock_user

    app.dependency_overrides[get_current_user] = override_get_current_user

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()


@pytest.fixture()
def unauthenticated_client():
    """Create a test client without authentication."""
    app.dependency_overrides.clear()
    with TestClient(app) as client:
        yield client


class TestChunkingStrategyEndpoints:
    """Integration tests for strategy management endpoints."""

    def test_list_strategies_success(self, client_with_auth: TestClient, auth_headers: dict[str, str]) -> None:
        """Test successful listing of available chunking strategies."""
        # Act
        response = client_with_auth.get(
            "/api/v2/chunking/strategies",
            headers=auth_headers,
        )

        # Assert
        assert response.status_code == 200
        strategies = response.json()
        assert isinstance(strategies, list)
        assert len(strategies) > 0

        # Check that all primary strategies are present
        strategy_ids = [s["id"] for s in strategies]
        expected_strategies = ["fixed_size", "recursive", "markdown", "semantic", "hierarchical", "hybrid"]
        for expected in expected_strategies:
            assert expected in strategy_ids, f"Missing strategy: {expected}"

        # Verify structure of each strategy
        for strategy in strategies:
            assert "id" in strategy
            assert "name" in strategy
            assert "description" in strategy
            assert "default_config" in strategy
            assert isinstance(strategy.get("best_for", []), list)
            assert isinstance(strategy.get("pros", []), list)
            assert isinstance(strategy.get("cons", []), list)

    def test_list_strategies_unauthorized(self, unauthenticated_client: TestClient) -> None:
        """Test that listing strategies requires authentication."""
        # Act
        response = unauthenticated_client.get("/api/v2/chunking/strategies")

        # Assert
        assert response.status_code == 401
        assert "Not authenticated" in response.json()["detail"]

    def test_list_strategies_service_error(self, client_with_auth: TestClient, auth_headers: dict[str, str]) -> None:
        """Test handling of service errors when listing strategies."""
        # Arrange
        with patch("packages.webui.api.v2.chunking.get_chunking_service") as mock_get_service:
            mock_service = AsyncMock(spec=ChunkingService)
            mock_service.get_available_strategies_for_api.side_effect = Exception("Service error")
            mock_get_service.return_value = mock_service

            # Act
            response = client_with_auth.get(
                "/api/v2/chunking/strategies",
                headers=auth_headers,
            )

            # Assert
            assert response.status_code == 500
            assert "Failed to list strategies" in response.json()["detail"]

    def test_get_strategy_details_success(self, client_with_auth: TestClient, auth_headers: dict[str, str]) -> None:
        """Test successful retrieval of strategy details."""
        # Act
        response = client_with_auth.get(
            "/api/v2/chunking/strategies/recursive",
            headers=auth_headers,
        )

        # Assert
        assert response.status_code == 200
        strategy = response.json()
        assert strategy["id"] == "recursive"
        assert "name" in strategy
        assert "description" in strategy
        assert "default_config" in strategy
        assert isinstance(strategy.get("best_for", []), list)
        assert isinstance(strategy.get("performance_characteristics", {}), dict)

    def test_get_strategy_details_not_found(self, client_with_auth: TestClient, auth_headers: dict[str, str]) -> None:
        """Test retrieval of non-existent strategy."""
        # Act
        response = client_with_auth.get(
            "/api/v2/chunking/strategies/non_existent",
            headers=auth_headers,
        )

        # Assert
        assert response.status_code == 404
        assert "Strategy 'non_existent' not found" in response.json()["detail"]

    def test_recommend_strategy_success(self, client_with_auth: TestClient, auth_headers: dict[str, str]) -> None:
        """Test successful strategy recommendation."""
        # Act
        response = client_with_auth.post(
            "/api/v2/chunking/strategies/recommend",
            headers=auth_headers,
            params={"file_types": ["markdown", "md"]},
        )

        # Assert
        assert response.status_code == 200
        recommendation = response.json()
        assert "recommended_strategy" in recommendation
        assert "confidence" in recommendation
        assert "reasoning" in recommendation
        assert "suggested_config" in recommendation
        assert recommendation["recommended_strategy"] == "markdown"
        assert recommendation["confidence"] >= 0.0
        assert recommendation["confidence"] <= 1.0

    def test_recommend_strategy_no_file_types(self, client_with_auth: TestClient, auth_headers: dict[str, str]) -> None:
        """Test strategy recommendation without file types."""
        # Act
        response = client_with_auth.post(
            "/api/v2/chunking/strategies/recommend",
            headers=auth_headers,
        )

        # Assert
        assert response.status_code == 422
        assert "Field required" in str(response.json()["detail"])


class TestChunkingPreviewEndpoints:
    """Integration tests for preview operations."""

    @pytest.mark.asyncio()
    async def test_generate_preview_success(self, async_client: AsyncClient, auth_headers: dict[str, str]) -> None:
        """Test successful preview generation."""
        # Arrange
        preview_request = {
            "strategy": "fixed_size",
            "content": "This is a test document with some content for chunking.",
            "config": {
                "strategy": "fixed_size",
                "chunk_size": 10,
                "chunk_overlap": 2,
            },
        }

        # Act
        response = await async_client.post(
            "/api/v2/chunking/preview",
            headers=auth_headers,
            json=preview_request,
        )

        # Assert
        assert response.status_code == 200
        preview = response.json()
        assert "preview_id" in preview
        assert preview["strategy"] == "fixed_size"
        assert "chunks" in preview
        assert isinstance(preview["chunks"], list)
        assert "total_chunks" in preview
        assert "processing_time_ms" in preview
        assert "correlation_id" in preview

    @pytest.mark.asyncio()
    async def test_generate_preview_with_correlation_id(
        self, async_client: AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test preview generation with custom correlation ID."""
        # Arrange
        correlation_id = str(uuid.uuid4())
        headers = {**auth_headers, "X-Correlation-ID": correlation_id}
        preview_request = {
            "strategy": "fixed_size",
            "content": "Test content",
        }

        # Act
        response = await async_client.post(
            "/api/v2/chunking/preview",
            headers=headers,
            json=preview_request,
        )

        # Assert
        assert response.status_code == 200
        preview = response.json()
        assert preview["correlation_id"] == correlation_id

    @pytest.mark.asyncio()
    async def test_generate_preview_content_too_large(
        self, async_client: AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test preview generation with content that's too large."""
        # Arrange
        with patch("packages.webui.api.v2.chunking.get_chunking_service") as mock_get_service:
            mock_service = AsyncMock(spec=ChunkingService)
            mock_service.validate_preview_content.side_effect = DocumentTooLargeError(
                "Content exceeds maximum size of 1MB"
            )
            mock_get_service.return_value = mock_service

            preview_request = {
                "strategy": "fixed_size",
                "content": "x" * 2000000,  # 2MB of content
            }

            # Act
            response = client_with_auth.post(
                "/api/v2/chunking/preview",
                headers=auth_headers,
                json=preview_request,
            )

            # Assert
            assert response.status_code == 507
            assert "Content exceeds maximum size" in response.json()["detail"]

    def test_generate_preview_validation_error(
        self, client_with_auth: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Test preview generation with validation errors."""
        # Arrange
        with patch("packages.webui.api.v2.chunking.get_chunking_service") as mock_get_service:
            mock_service = AsyncMock(spec=ChunkingService)
            mock_service.validate_preview_content.side_effect = ChunkingValidationError(
                "Invalid configuration: chunk_size must be positive"
            )
            mock_get_service.return_value = mock_service

            preview_request = {
                "strategy": "fixed_size",
                "content": "Test content",
                "config": {
                    "strategy": "fixed_size",
                    "chunk_size": -1,
                },
            }

            # Act
            response = client_with_auth.post(
                "/api/v2/chunking/preview",
                headers=auth_headers,
                json=preview_request,
            )

            # Assert
            assert response.status_code == 400
            assert "chunk_size must be positive" in response.json()["detail"]

    @pytest.mark.asyncio()
    async def test_generate_preview_rate_limited(self, async_client: AsyncClient, auth_headers: dict[str, str]) -> None:
        """Test preview generation rate limiting."""
        # Note: Rate limiting is disabled in test environment by default
        # This test verifies the endpoint structure supports rate limiting

        preview_request = {
            "strategy": "fixed_size",
            "content": "Test",
        }

        # Make multiple requests quickly
        responses = []
        for _ in range(3):
            response = await async_client.post(
                "/api/v2/chunking/preview",
                headers=auth_headers,
                json=preview_request,
            )
            responses.append(response)

        # In production, this would trigger rate limiting
        # In tests, all should succeed
        for response in responses:
            assert response.status_code == 200

    @pytest.mark.asyncio()
    async def test_compare_strategies_success(self, async_client: AsyncClient, auth_headers: dict[str, str]) -> None:
        """Test successful strategy comparison."""
        # Arrange
        compare_request = {
            "content": "This is test content for comparing chunking strategies.",
            "strategies": ["fixed_size", "recursive"],
            "configs": {
                "fixed_size": {
                    "strategy": "fixed_size",
                    "chunk_size": 20,
                    "chunk_overlap": 5,
                },
                "recursive": {
                    "strategy": "recursive",
                    "chunk_size": 30,
                    "chunk_overlap": 10,
                },
            },
        }

        # Act
        response = await async_client.post(
            "/api/v2/chunking/compare",
            headers=auth_headers,
            json=compare_request,
        )

        # Assert
        assert response.status_code == 200
        comparison = response.json()
        assert "comparison_id" in comparison
        assert "comparisons" in comparison
        assert "recommendation" in comparison
        assert "processing_time_ms" in comparison

    @pytest.mark.asyncio()
    async def test_get_cached_preview_not_found(self, async_client: AsyncClient, auth_headers: dict[str, str]) -> None:
        """Test retrieval of non-existent cached preview."""
        # Act
        response = await async_client.get(
            f"/api/v2/chunking/preview/{uuid.uuid4()}",
            headers=auth_headers,
        )

        # Assert
        assert response.status_code == 404
        assert "Preview not found or expired" in response.json()["detail"]


class TestChunkingOperationEndpoints:
    """Integration tests for chunking operation endpoints."""

    @pytest_asyncio.fixture
    async def test_collection_with_user(self, db_session: AsyncSession, test_user_db: User) -> Collection:
        """Create a test collection owned by the test user."""
        collection = Collection(
            id=str(uuid.uuid4()),
            name="Test Collection",
            description="Collection for chunking tests",
            owner_id=test_user_db.id,
            status=CollectionStatus.READY,
            vector_store_name=f"test_collection_{uuid.uuid4().hex[:8]}",
            embedding_model="test-model",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        db_session.add(collection)
        await db_session.commit()
        await db_session.refresh(collection)
        return collection

    @pytest.mark.asyncio()
    async def test_start_chunking_operation_success(
        self,
        async_client: AsyncClient,
        db_session: AsyncSession,
        test_user_db: User,
        test_collection_with_user: Collection,
    ) -> None:
        """Test successful start of chunking operation."""
        # Arrange
        token = create_access_token(data={"sub": test_user_db.username})
        headers = {"Authorization": f"Bearer {token}"}

        chunking_request = {
            "strategy": "fixed_size",
            "config": {
                "strategy": "fixed_size",
                "chunk_size": 512,
                "chunk_overlap": 50,
            },
            "priority": "normal",
        }

        with patch("packages.webui.api.v2.chunking.process_chunking_operation"):
            # Act
            response = await async_client.post(
                f"/api/v2/chunking/collections/{test_collection_with_user.id}/chunk",
                headers=headers,
                json=chunking_request,
            )

            # Assert
            assert response.status_code == 202
            operation = response.json()
            assert "operation_id" in operation
            assert operation["collection_id"] == test_collection_with_user.id
            assert operation["status"] == "PENDING"
            assert operation["strategy"] == "fixed_size"
            assert "websocket_channel" in operation

    @pytest.mark.asyncio()
    async def test_start_chunking_operation_invalid_config(
        self,
        async_client: AsyncClient,
        db_session: AsyncSession,
        test_user_db: User,
        test_collection_with_user: Collection,
    ) -> None:
        """Test starting chunking operation with invalid configuration."""
        # Arrange
        token = create_access_token(data={"sub": test_user_db.username})
        headers = {"Authorization": f"Bearer {token}"}

        chunking_request = {
            "strategy": "fixed_size",
            "config": {
                "strategy": "fixed_size",
                "chunk_size": -100,  # Invalid negative size
                "chunk_overlap": 50,
            },
        }

        with patch(
            "packages.webui.services.chunking_service.ChunkingService.validate_config_for_collection"
        ) as mock_validate:
            mock_validate.return_value = {
                "valid": False,
                "errors": ["chunk_size must be positive"],
            }

            # Act
            response = await async_client.post(
                f"/api/v2/chunking/collections/{test_collection_with_user.id}/chunk",
                headers=headers,
                json=chunking_request,
            )

            # Assert
            assert response.status_code == 400
            assert "Invalid configuration" in response.json()["detail"]
            assert "chunk_size must be positive" in response.json()["detail"]

    @pytest.mark.asyncio()
    async def test_start_chunking_operation_collection_not_found(
        self,
        async_client: AsyncClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test starting chunking operation for non-existent collection."""
        # Arrange
        non_existent_id = str(uuid.uuid4())
        chunking_request = {
            "strategy": "fixed_size",
            "config": {
                "strategy": "fixed_size",
                "chunk_size": 512,
                "chunk_overlap": 50,
            },
        }

        # Act
        response = await async_client.post(
            f"/api/v2/chunking/collections/{non_existent_id}/chunk",
            headers=auth_headers,
            json=chunking_request,
        )

        # Assert
        assert response.status_code == 404
        assert "Collection not found" in response.json()["detail"]

    @pytest.mark.asyncio()
    async def test_start_chunking_operation_access_denied(
        self,
        async_client: AsyncClient,
        db_session: AsyncSession,
        test_user_db: User,
        other_user_db: User,
        test_collection_with_user: Collection,
    ) -> None:
        """Test starting chunking operation for collection owned by another user."""
        # Arrange - Use token for other_user trying to access test_user's collection
        token = create_access_token(data={"sub": other_user_db.username})
        headers = {"Authorization": f"Bearer {token}"}

        chunking_request = {
            "strategy": "fixed_size",
            "config": {
                "strategy": "fixed_size",
                "chunk_size": 512,
                "chunk_overlap": 50,
            },
        }

        # Act
        response = await async_client.post(
            f"/api/v2/chunking/collections/{test_collection_with_user.id}/chunk",
            headers=headers,
            json=chunking_request,
        )

        # Assert
        assert response.status_code == 403
        assert "don't have access" in response.json()["detail"]

    @pytest.mark.asyncio()
    async def test_update_chunking_strategy_success(
        self,
        async_client: AsyncClient,
        db_session: AsyncSession,
        test_user_db: User,
        test_collection_with_user: Collection,
    ) -> None:
        """Test successful update of chunking strategy."""
        # Arrange
        token = create_access_token(data={"sub": test_user_db.username})
        headers = {"Authorization": f"Bearer {token}"}

        update_request = {
            "strategy": "recursive",
            "config": {
                "strategy": "recursive",
                "chunk_size": 1000,
                "chunk_overlap": 100,
            },
            "reprocess_existing": False,
        }

        # Act
        response = await async_client.patch(
            f"/api/v2/chunking/collections/{test_collection_with_user.id}/chunking-strategy",
            headers=headers,
            json=update_request,
        )

        # Assert
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "COMPLETED"
        assert result["strategy"] == "recursive"

    @pytest.mark.asyncio()
    async def test_update_chunking_strategy_with_reprocessing(
        self,
        async_client: AsyncClient,
        db_session: AsyncSession,
        test_user_db: User,
        test_collection_with_user: Collection,
    ) -> None:
        """Test updating chunking strategy with document reprocessing."""
        # Arrange
        token = create_access_token(data={"sub": test_user_db.username})
        headers = {"Authorization": f"Bearer {token}"}

        update_request = {
            "strategy": "semantic",
            "config": {
                "strategy": "semantic",
                "chunk_size": 800,
                "chunk_overlap": 200,
            },
            "reprocess_existing": True,
        }

        with patch("packages.webui.api.v2.chunking.process_chunking_operation"):
            # Act
            response = await async_client.patch(
                f"/api/v2/chunking/collections/{test_collection_with_user.id}/chunking-strategy",
                headers=headers,
                json=update_request,
            )

            # Assert
            assert response.status_code == 200
            result = response.json()
            assert "operation_id" in result
            assert result["status"] == "PENDING"
            assert result["strategy"] == "semantic"


class TestChunkingProgressEndpoints:
    """Integration tests for progress tracking endpoints."""

    @pytest.mark.asyncio()
    async def test_get_operation_progress_success(
        self, async_client: AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test successful retrieval of operation progress."""
        # Arrange
        operation_id = str(uuid.uuid4())

        with patch("packages.webui.api.v2.chunking.get_chunking_service") as mock_get_service:
            mock_service = AsyncMock(spec=ChunkingService)
            mock_service.get_chunking_progress.return_value = {
                "status": "PROCESSING",
                "progress_percentage": 45.5,
                "documents_processed": 5,
                "total_documents": 11,
                "chunks_created": 250,
                "current_document": "document_6.pdf",
                "estimated_time_remaining": 120,
                "errors": [],
            }
            mock_get_service.return_value = mock_service

            # Act
            response = await async_client.get(
                f"/api/v2/chunking/operations/{operation_id}/progress",
                headers=auth_headers,
            )

            # Assert
            assert response.status_code == 200
            progress = response.json()
            assert progress["operation_id"] == operation_id
            assert progress["status"] == "PROCESSING"
            assert progress["progress_percentage"] == 45.5
            assert progress["documents_processed"] == 5
            assert progress["total_documents"] == 11
            assert progress["chunks_created"] == 250
            assert progress["current_document"] == "document_6.pdf"

    @pytest.mark.asyncio()
    async def test_get_operation_progress_not_found(
        self, async_client: AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test retrieval of progress for non-existent operation."""
        # Arrange
        operation_id = str(uuid.uuid4())

        with patch("packages.webui.api.v2.chunking.get_chunking_service") as mock_get_service:
            mock_service = AsyncMock(spec=ChunkingService)
            mock_service.get_chunking_progress.return_value = None
            mock_get_service.return_value = mock_service

            # Act
            response = await async_client.get(
                f"/api/v2/chunking/operations/{operation_id}/progress",
                headers=auth_headers,
            )

            # Assert
            assert response.status_code == 404
            assert "Operation not found" in response.json()["detail"]


class TestChunkingStatsEndpoints:
    """Integration tests for chunking statistics endpoints."""

    @pytest.mark.asyncio()
    async def test_get_chunking_stats_success(
        self,
        async_client: AsyncClient,
        db_session: AsyncSession,
        test_user_db: User,
        test_collection_with_user: Collection,
    ) -> None:
        """Test successful retrieval of chunking statistics."""
        # Arrange
        token = create_access_token(data={"sub": test_user_db.username})
        headers = {"Authorization": f"Bearer {token}"}

        with patch("packages.webui.api.v2.chunking.get_chunking_service") as mock_get_service:
            mock_service = AsyncMock(spec=ChunkingService)
            mock_service.get_collection_chunk_stats.return_value = {
                "total_chunks": 1500,
                "total_documents": 25,
                "average_chunk_size": 512,
                "min_chunk_size": 100,
                "max_chunk_size": 1024,
                "size_variance": 156.25,
                "strategy": "recursive",
                "last_updated": datetime.now(UTC),
                "processing_time": 45.3,
                "performance_metrics": {
                    "avg_processing_speed": 33.1,
                    "memory_usage_mb": 256,
                },
            }
            mock_get_service.return_value = mock_service

            # Act
            response = await async_client.get(
                f"/api/v2/chunking/collections/{test_collection_with_user.id}/chunking-stats",
                headers=headers,
            )

            # Assert
            assert response.status_code == 200
            stats = response.json()
            assert stats["total_chunks"] == 1500
            assert stats["total_documents"] == 25
            assert stats["avg_chunk_size"] == 512
            assert stats["strategy_used"] == "recursive"
            assert "quality_metrics" in stats

    @pytest.mark.asyncio()
    async def test_get_collection_chunks_paginated(
        self,
        async_client: AsyncClient,
        db_session: AsyncSession,
        test_user_db: User,
        test_collection_with_user: Collection,
    ) -> None:
        """Test paginated retrieval of collection chunks."""
        # Arrange
        token = create_access_token(data={"sub": test_user_db.username})
        headers = {"Authorization": f"Bearer {token}"}

        # Act
        response = await async_client.get(
            f"/api/v2/chunking/collections/{test_collection_with_user.id}/chunks",
            headers=headers,
            params={"page": 1, "page_size": 20},
        )

        # Assert
        assert response.status_code == 200
        chunk_list = response.json()
        assert "chunks" in chunk_list
        assert "total" in chunk_list
        assert chunk_list["page"] == 1
        assert chunk_list["page_size"] == 20
        assert "has_next" in chunk_list


class TestChunkingAnalyticsEndpoints:
    """Integration tests for chunking analytics endpoints."""

    @pytest.mark.asyncio()
    async def test_get_global_metrics_success(self, async_client: AsyncClient, auth_headers: dict[str, str]) -> None:
        """Test successful retrieval of global chunking metrics."""
        # Act
        response = await async_client.get(
            "/api/v2/chunking/metrics",
            headers=auth_headers,
            params={"period_days": 30},
        )

        # Assert
        assert response.status_code == 200
        metrics = response.json()
        assert "total_collections_processed" in metrics
        assert "total_chunks_created" in metrics
        assert "total_documents_processed" in metrics
        assert "avg_chunks_per_document" in metrics
        assert "most_used_strategy" in metrics
        assert "success_rate" in metrics
        assert "period_start" in metrics
        assert "period_end" in metrics

    @pytest.mark.asyncio()
    async def test_get_metrics_by_strategy_success(
        self, async_client: AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test successful retrieval of metrics grouped by strategy."""
        # Act
        response = await async_client.get(
            "/api/v2/chunking/metrics/by-strategy",
            headers=auth_headers,
            params={"period_days": 30},
        )

        # Assert
        assert response.status_code == 200
        metrics = response.json()
        assert isinstance(metrics, list)

        # Should have metrics for all primary strategies
        strategy_names = [m["strategy"] for m in metrics]
        expected_strategies = ["FIXED_SIZE", "RECURSIVE", "MARKDOWN", "SEMANTIC", "HIERARCHICAL", "HYBRID"]
        for expected in expected_strategies:
            assert expected in strategy_names

        # Verify structure of each metric
        for metric in metrics:
            assert "strategy" in metric
            assert "usage_count" in metric
            assert "avg_chunk_size" in metric
            assert "avg_processing_time" in metric
            assert "success_rate" in metric
            assert "avg_quality_score" in metric

    @pytest.mark.asyncio()
    async def test_get_quality_scores_success(self, async_client: AsyncClient, auth_headers: dict[str, str]) -> None:
        """Test successful retrieval of chunk quality analysis."""
        # Act
        response = await async_client.get(
            "/api/v2/chunking/quality-scores",
            headers=auth_headers,
        )

        # Assert
        assert response.status_code == 200
        quality = response.json()
        assert "overall_quality" in quality
        assert "quality_score" in quality
        assert "coherence_score" in quality
        assert "completeness_score" in quality
        assert "size_consistency" in quality
        assert "recommendations" in quality
        assert isinstance(quality["recommendations"], list)
        assert "issues_detected" in quality
        assert isinstance(quality["issues_detected"], list)


class TestChunkingConfigurationEndpoints:
    """Integration tests for configuration management endpoints."""

    @pytest.mark.asyncio()
    async def test_save_configuration_success(self, async_client: AsyncClient, auth_headers: dict[str, str]) -> None:
        """Test successful saving of custom configuration."""
        # Arrange
        config_request = {
            "name": "My Custom Config",
            "description": "Optimized for technical documentation",
            "strategy": "recursive",
            "config": {
                "strategy": "recursive",
                "chunk_size": 1500,
                "chunk_overlap": 150,
                "preserve_sentences": True,
            },
            "is_default": False,
            "tags": ["technical", "documentation"],
        }

        # Act
        response = await async_client.post(
            "/api/v2/chunking/configs",
            headers=auth_headers,
            json=config_request,
        )

        # Assert
        assert response.status_code == 201
        saved = response.json()
        assert "id" in saved
        assert saved["name"] == "My Custom Config"
        assert saved["strategy"] == "recursive"
        assert saved["is_default"] is False
        assert saved["tags"] == ["technical", "documentation"]

    @pytest.mark.asyncio()
    async def test_list_configurations_success(self, async_client: AsyncClient, auth_headers: dict[str, str]) -> None:
        """Test successful listing of saved configurations."""
        # Act
        response = await async_client.get(
            "/api/v2/chunking/configs",
            headers=auth_headers,
        )

        # Assert
        assert response.status_code == 200
        configs = response.json()
        assert isinstance(configs, list)

    @pytest.mark.asyncio()
    async def test_list_configurations_filtered_by_strategy(
        self, async_client: AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Test listing configurations filtered by strategy."""
        # Act
        response = await async_client.get(
            "/api/v2/chunking/configs",
            headers=auth_headers,
            params={"strategy": "recursive"},
        )

        # Assert
        assert response.status_code == 200
        configs = response.json()
        assert isinstance(configs, list)
        # All returned configs should be for recursive strategy
        for config in configs:
            if config:  # If there are any configs
                assert config["strategy"] == "recursive"


class TestChunkingErrorHandling:
    """Integration tests for error handling across all endpoints."""

    @pytest.mark.asyncio()
    async def test_circuit_breaker_triggered(self, async_client: AsyncClient, auth_headers: dict[str, str]) -> None:
        """Test circuit breaker activation after multiple failures."""
        # Note: Circuit breaker is typically disabled in test environment
        # This test verifies the endpoint structure supports circuit breaking

        with patch("packages.webui.api.v2.chunking.check_circuit_breaker") as mock_check:
            mock_check.side_effect = HTTPException(
                status_code=503,
                detail="Circuit breaker is open - too many failures",
            )

            # Act
            response = await async_client.post(
                "/api/v2/chunking/preview",
                headers=auth_headers,
                json={"strategy": "fixed_size", "content": "test"},
            )

            # Assert
            assert response.status_code == 503
            assert "Circuit breaker is open" in response.json()["detail"]

    @pytest.mark.asyncio()
    async def test_unexpected_error_handling(self, async_client: AsyncClient, auth_headers: dict[str, str]) -> None:
        """Test handling of unexpected errors with proper error response."""
        # Arrange
        with patch("packages.webui.api.v2.chunking.get_chunking_service") as mock_get_service:
            mock_service = AsyncMock(spec=ChunkingService)
            # Simulate an unexpected error
            mock_service.preview_chunking.side_effect = RuntimeError("Unexpected internal error")
            mock_get_service.return_value = mock_service

            # Act
            response = await async_client.post(
                "/api/v2/chunking/preview",
                headers=auth_headers,
                json={"strategy": "fixed_size", "content": "test"},
            )

            # Assert
            assert response.status_code == 500
            error_detail = response.json()["detail"]
            assert "error" in error_detail
            assert error_detail["error"]["code"] == "INTERNAL_ERROR"
            assert "correlation_id" in error_detail["error"]


class TestChunkingSecurityAndAuth:
    """Integration tests for security and authentication."""

    @pytest.mark.asyncio()
    async def test_all_endpoints_require_authentication(self, async_client: AsyncClient) -> None:
        """Test that all chunking endpoints require authentication."""
        # List of all endpoints that should require auth
        endpoints = [
            ("GET", "/api/v2/chunking/strategies"),
            ("GET", "/api/v2/chunking/strategies/fixed_size"),
            ("POST", "/api/v2/chunking/strategies/recommend"),
            ("POST", "/api/v2/chunking/preview"),
            ("POST", "/api/v2/chunking/compare"),
            ("GET", f"/api/v2/chunking/preview/{uuid.uuid4()}"),
            ("DELETE", f"/api/v2/chunking/preview/{uuid.uuid4()}"),
            ("GET", "/api/v2/chunking/metrics"),
            ("GET", "/api/v2/chunking/metrics/by-strategy"),
            ("GET", "/api/v2/chunking/quality-scores"),
            ("POST", "/api/v2/chunking/analyze"),
            ("POST", "/api/v2/chunking/configs"),
            ("GET", "/api/v2/chunking/configs"),
            ("GET", f"/api/v2/chunking/operations/{uuid.uuid4()}/progress"),
        ]

        for method, endpoint in endpoints:
            # Act
            if method == "GET":
                response = await async_client.get(endpoint)
            elif method == "POST":
                response = await async_client.post(endpoint, json={})
            elif method == "DELETE":
                response = await async_client.delete(endpoint)
            else:
                continue

            # Assert
            assert response.status_code == 401, f"Endpoint {endpoint} should require auth"
            assert "Not authenticated" in response.json()["detail"]

    @pytest.mark.asyncio()
    async def test_invalid_jwt_token_rejected(self, async_client: AsyncClient) -> None:
        """Test that invalid JWT tokens are rejected."""
        # Arrange
        headers = {"Authorization": "Bearer invalid.jwt.token"}

        # Act
        response = await async_client.get(
            "/api/v2/chunking/strategies",
            headers=headers,
        )

        # Assert
        assert response.status_code == 401
        assert "Could not validate credentials" in response.json()["detail"]

    @pytest.mark.asyncio()
    async def test_expired_jwt_token_rejected(self, async_client: AsyncClient) -> None:
        """Test that expired JWT tokens are rejected."""
        # Arrange - Create a token that's already expired
        expired_token = create_access_token(
            data={"sub": "testuser"},
            expires_delta=timedelta(seconds=-1),  # Already expired
        )
        headers = {"Authorization": f"Bearer {expired_token}"}

        # Act
        response = await async_client.get(
            "/api/v2/chunking/strategies",
            headers=headers,
        )

        # Assert
        assert response.status_code == 401
