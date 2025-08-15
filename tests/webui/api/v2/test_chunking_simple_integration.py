"""
Simple integration tests for v2 chunking API endpoints.

These tests focus on the most critical endpoints with minimal mocking
to improve coverage of the actual endpoint code.
"""

import os
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from packages.shared.chunking.infrastructure.exceptions import (
    DocumentTooLargeError,
)
from packages.shared.chunking.infrastructure.exceptions import (
    ValidationError as ChunkingValidationError,
)
from packages.webui.auth import create_access_token, get_current_user
from packages.webui.dependencies import get_collection_for_user
from packages.webui.main import app
from packages.webui.services.chunking_service import ChunkingService
from packages.webui.services.collection_service import CollectionService
from packages.webui.services.factory import get_chunking_service, get_collection_service

# Disable rate limiting for tests
os.environ["DISABLE_RATE_LIMITING"] = "true"
os.environ["TESTING"] = "true"


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
def mock_chunking_service():
    """Create a mock ChunkingService."""
    service = AsyncMock(spec=ChunkingService, name="MockChunkingService")

    # Mock get_available_strategies_for_api
    service.get_available_strategies_for_api.return_value = [
        {
            "id": "fixed_size",
            "name": "Fixed Size",
            "description": "Splits text into fixed-size chunks",
            "best_for": ["general text"],
            "pros": ["Simple", "Predictable"],
            "cons": ["May break sentences"],
            "default_config": {"strategy": "fixed_size", "chunk_size": 512, "chunk_overlap": 50},
            "performance_characteristics": {"speed": "fast"},
        },
        {
            "id": "recursive",
            "name": "Recursive",
            "description": "Recursively splits text",
            "best_for": ["structured documents"],
            "pros": ["Preserves structure"],
            "cons": ["More complex"],
            "default_config": {"strategy": "recursive", "chunk_size": 1000, "chunk_overlap": 100},
            "performance_characteristics": {"speed": "medium"},
        },
        {
            "id": "markdown",
            "name": "Markdown",
            "description": "Splits markdown documents",
            "best_for": ["markdown files"],
            "pros": ["Preserves markdown structure"],
            "cons": ["Only for markdown"],
            "default_config": {"strategy": "markdown", "chunk_size": 800, "chunk_overlap": 100},
            "performance_characteristics": {"speed": "fast"},
        },
        {
            "id": "semantic",
            "name": "Semantic",
            "description": "Semantic chunking",
            "best_for": ["complex documents"],
            "pros": ["Better context"],
            "cons": ["Slower"],
            "default_config": {"strategy": "fixed_size", "chunk_size": 512, "chunk_overlap": 50},
            "performance_characteristics": {"speed": "slow"},
        },
        {
            "id": "hierarchical",
            "name": "Hierarchical",
            "description": "Hierarchical chunking",
            "best_for": ["nested documents"],
            "pros": ["Preserves hierarchy"],
            "cons": ["Complex"],
            "default_config": {"strategy": "hierarchical", "chunk_size": 1000, "chunk_overlap": 200},
            "performance_characteristics": {"speed": "medium"},
        },
        {
            "id": "hybrid",
            "name": "Hybrid",
            "description": "Hybrid chunking approach",
            "best_for": ["mixed content"],
            "pros": ["Flexible"],
            "cons": ["Requires tuning"],
            "default_config": {"strategy": "hybrid", "chunk_size": 750, "chunk_overlap": 150},
            "performance_characteristics": {"speed": "medium"},
        },
    ]

    # Mock get_strategy_details
    service.get_strategy_details.return_value = {
        "id": "recursive",
        "name": "Recursive",
        "description": "Recursively splits text",
        "best_for": ["structured documents"],
        "pros": ["Preserves structure"],
        "cons": ["More complex"],
        "default_config": {"strategy": "recursive", "chunk_size": 1000, "chunk_overlap": 100},
        "performance_characteristics": {"speed": "medium"},
    }

    # Mock recommend_strategy
    service.recommend_strategy.return_value = {
        "strategy": "markdown",
        "confidence": 0.95,
        "reasoning": "Markdown files are best handled with markdown-specific chunking",
        "alternatives": ["recursive", "fixed_size"],
        "chunk_size": 800,
        "chunk_overlap": 100,
    }

    # Mock preview_chunking
    service.preview_chunking.return_value = {
        "preview_id": str(uuid.uuid4()),
        "strategy": "fixed_size",
        "config": {"strategy": "fixed_size", "chunk_size": 100, "chunk_overlap": 10},
        "chunks": [
            {
                "index": 0,
                "content": "This is a ",
                "text": "This is a ",
                "token_count": 3,
                "metadata": {},
                "quality_score": 0.8,
            },
            {
                "index": 1,
                "content": "a test doc",
                "text": "a test doc",
                "token_count": 3,
                "metadata": {},
                "quality_score": 0.8,
            },
        ],
        "total_chunks": 2,
        "metrics": {"avg_chunk_size": 100},
        "processing_time_ms": 50,
        "cached": False,
        "expires_at": datetime.now(UTC) + timedelta(minutes=15),
    }

    # Mock validate_preview_content
    service.validate_preview_content.return_value = None

    # Mock compare_strategies_for_api
    service.compare_strategies_for_api.return_value = {
        "comparison_id": str(uuid.uuid4()),
        "comparisons": [
            {
                "strategy": "fixed_size",
                "config": {"strategy": "fixed_size", "chunk_size": 200, "chunk_overlap": 50},
                "sample_chunks": [
                    {
                        "index": 0,
                        "content": "Sample chunk 1",
                        "text": "Sample chunk 1",
                        "token_count": 3,
                        "char_count": 14,
                        "metadata": {},
                        "quality_score": 0.75,
                    }
                ],
                "total_chunks": 5,
                "avg_chunk_size": 20,
                "size_variance": 2.5,
                "quality_score": 0.75,
                "processing_time_ms": 50,
                "pros": ["Fast", "Predictable"],
                "cons": ["May break context"],
            },
            {
                "strategy": "recursive",
                "config": {"strategy": "recursive", "chunk_size": 300, "chunk_overlap": 100},
                "sample_chunks": [
                    {
                        "index": 0,
                        "content": "Sample chunk 2",
                        "text": "Sample chunk 2",
                        "token_count": 3,
                        "char_count": 14,
                        "metadata": {},
                        "quality_score": 0.85,
                    }
                ],
                "total_chunks": 4,
                "avg_chunk_size": 25,
                "size_variance": 3.0,
                "quality_score": 0.85,
                "processing_time_ms": 50,
                "pros": ["Preserves structure"],
                "cons": ["More complex"],
            },
        ],
        "recommendation": {
            "recommended_strategy": "recursive",
            "confidence": 0.85,
            "reasoning": "Better quality score with fewer chunks",
            "alternative_strategies": ["fixed_size"],
            "suggested_config": {"strategy": "recursive", "chunk_size": 300, "chunk_overlap": 100},
        },
        "processing_time_ms": 100,
    }

    # Mock get_cached_preview_by_id
    service.get_cached_preview_by_id.return_value = None

    # Mock clear_preview_cache
    service.clear_preview_cache.return_value = None

    # Mock validate_config_for_collection
    service.validate_config_for_collection.return_value = {
        "valid": True,
        "estimated_time": 60,
    }

    # Mock start_chunking_operation
    service.start_chunking_operation.return_value = (
        f"chunking:collection:{uuid.uuid4()}",
        str(uuid.uuid4()),
    )

    # Mock get_collection_chunk_stats
    service.get_collection_chunk_stats.return_value = {
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

    # Mock get_chunking_progress
    service.get_chunking_progress.return_value = {
        "status": "in_progress",
        "progress_percentage": 45.5,
        "documents_processed": 5,
        "total_documents": 11,
        "chunks_created": 250,
        "current_document": "document_6.pdf",
        "estimated_time_remaining": 120,
        "errors": [],
    }

    # Mock process_chunking_operation
    service.process_chunking_operation.return_value = None

    # Mock get_collection_chunks if method exists
    if hasattr(service, "get_collection_chunks"):
        service.get_collection_chunks.return_value = {
            "chunks": [],
            "total": 0,
            "page": 1,
            "page_size": 20,
            "has_next": False,
        }

    return service


@pytest.fixture()
def mock_collection_service():
    """Create a mock CollectionService."""
    service = AsyncMock(spec=CollectionService)

    # Mock create_operation
    service.create_operation.return_value = {
        "uuid": str(uuid.uuid4()),
        "collection_id": str(uuid.uuid4()),
        "type": "chunking",
        "status": "pending",  # Use lowercase to match the enum
    }

    # Mock update_collection
    service.update_collection.return_value = None

    return service


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
def client_with_mocked_services(mock_user, mock_chunking_service, mock_collection_service, mock_collection):
    """Create a test client with mocked services."""

    # Override dependencies
    async def override_get_current_user():
        return mock_user

    async def override_get_collection_for_user(
        collection_uuid: str = None,  # noqa: ARG001
        current_user: dict = None,  # noqa: ARG001
        db: Any = None,  # noqa: ARG001
    ):
        return mock_collection

    app.dependency_overrides[get_current_user] = override_get_current_user
    app.dependency_overrides[get_collection_for_user] = override_get_collection_for_user
    app.dependency_overrides[get_chunking_service] = lambda: mock_chunking_service
    app.dependency_overrides[get_collection_service] = lambda: mock_collection_service

    # Mock the lifespan events to prevent real database connections
    with (
        patch("packages.webui.main.pg_connection_manager") as mock_pg,
        patch("packages.webui.main.ws_manager") as mock_ws,
    ):
        # Mock the async methods
        mock_pg.initialize = AsyncMock()
        mock_ws.startup = AsyncMock()
        mock_ws.shutdown = AsyncMock()

        with TestClient(app) as client:
            yield client

    app.dependency_overrides.clear()


@pytest.fixture()
def unauthenticated_client():
    """Create a test client without authentication."""
    app.dependency_overrides.clear()

    # Mock the lifespan events to prevent real database connections
    with (
        patch("packages.webui.main.pg_connection_manager") as mock_pg,
        patch("packages.webui.main.ws_manager") as mock_ws,
    ):
        # Mock the async methods
        mock_pg.initialize = AsyncMock()
        mock_ws.startup = AsyncMock()
        mock_ws.shutdown = AsyncMock()

        with TestClient(app) as client:
            yield client


class TestStrategyEndpoints:
    """Test strategy management endpoints."""

    def test_list_strategies_success(
        self, client_with_mocked_services: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Test successful listing of available chunking strategies."""
        # Act
        response = client_with_mocked_services.get(
            "/api/v2/chunking/strategies",
            headers=auth_headers,
        )

        # Assert
        assert response.status_code == 200
        strategies = response.json()
        assert isinstance(strategies, list)
        assert len(strategies) == 6

        # Check that all primary strategies are present
        strategy_ids = [s["id"] for s in strategies]
        expected_strategies = ["fixed_size", "recursive", "markdown", "semantic", "hierarchical", "hybrid"]
        for expected in expected_strategies:
            assert expected in strategy_ids

        # Verify structure
        for strategy in strategies:
            assert "id" in strategy
            assert "name" in strategy
            assert "description" in strategy
            assert "default_config" in strategy

    def test_list_strategies_unauthorized(self, unauthenticated_client: TestClient) -> None:
        """Test that listing strategies requires authentication."""
        # Act
        response = unauthenticated_client.get("/api/v2/chunking/strategies")

        # Assert
        assert response.status_code == 401
        assert "Not authenticated" in response.json()["detail"]

    def test_get_strategy_details(self, client_with_mocked_services: TestClient, auth_headers: dict[str, str]) -> None:
        """Test getting strategy details."""
        # Act
        response = client_with_mocked_services.get(
            "/api/v2/chunking/strategies/recursive",
            headers=auth_headers,
        )

        # Assert
        assert response.status_code == 200
        strategy = response.json()
        assert strategy["id"] == "recursive"
        assert strategy["name"] == "Recursive"

    def test_recommend_strategy(self, client_with_mocked_services: TestClient, auth_headers: dict[str, str]) -> None:
        """Test strategy recommendation."""
        # Act
        response = client_with_mocked_services.post(
            "/api/v2/chunking/strategies/recommend",
            headers=auth_headers,
            params={"file_types": ["markdown", "md"]},
        )

        # Assert
        assert response.status_code == 200
        recommendation = response.json()
        assert recommendation["recommended_strategy"] == "markdown"
        assert recommendation["confidence"] == 0.95


class TestPreviewEndpoints:
    """Test preview endpoints."""

    def test_generate_preview_success(
        self, client_with_mocked_services: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Test successful preview generation."""
        # Arrange
        preview_request = {
            "strategy": "fixed_size",
            "content": "This is a test document with some content for chunking.",
            "config": {
                "strategy": "fixed_size",
                "chunk_size": 100,
                "chunk_overlap": 10,
            },
        }

        # Act
        response = client_with_mocked_services.post(
            "/api/v2/chunking/preview",
            headers=auth_headers,
            json=preview_request,
        )

        # Assert
        if response.status_code != 200:
            print(f"Error response: {response.json()}")
        assert response.status_code == 200
        preview = response.json()
        assert "preview_id" in preview
        assert preview["strategy"] == "fixed_size"
        assert "chunks" in preview
        assert len(preview["chunks"]) == 2
        assert preview["total_chunks"] == 2

    def test_generate_preview_validation_error(
        self, client_with_mocked_services: TestClient, auth_headers: dict[str, str], mock_chunking_service: AsyncMock
    ) -> None:
        """Test preview generation with validation error."""
        # Arrange
        mock_chunking_service.validate_preview_content.side_effect = ChunkingValidationError(
            field="content", value="", reason="Content cannot be empty"
        )

        preview_request = {
            "strategy": "fixed_size",
            "content": "Test content",
            "config": {
                "strategy": "fixed_size",
                "chunk_size": 100,  # Valid chunk size
                "chunk_overlap": 10,
            },
        }

        # Act
        response = client_with_mocked_services.post(
            "/api/v2/chunking/preview",
            headers=auth_headers,
            json=preview_request,
        )

        # Assert
        assert response.status_code == 400
        assert "Content cannot be empty" in response.json()["detail"]

    def test_compare_strategies(self, client_with_mocked_services: TestClient, auth_headers: dict[str, str]) -> None:
        """Test strategy comparison."""
        # Arrange
        compare_request = {
            "content": "This is test content for comparing chunking strategies.",
            "strategies": ["fixed_size", "recursive"],
            "configs": {
                "fixed_size": {
                    "strategy": "fixed_size",
                    "chunk_size": 200,
                    "chunk_overlap": 50,
                },
                "recursive": {
                    "strategy": "recursive",
                    "chunk_size": 300,
                    "chunk_overlap": 100,
                },
            },
        }

        # Act
        response = client_with_mocked_services.post(
            "/api/v2/chunking/compare",
            headers=auth_headers,
            json=compare_request,
        )

        # Assert
        assert response.status_code == 200
        comparison = response.json()
        assert "comparison_id" in comparison
        assert "comparisons" in comparison
        assert len(comparison["comparisons"]) == 2
        assert "recommendation" in comparison


class TestOperationEndpoints:
    """Test chunking operation endpoints."""

    def test_start_chunking_operation(
        self, client_with_mocked_services: TestClient, auth_headers: dict[str, str], mock_collection: dict[str, Any]
    ) -> None:
        """Test starting a chunking operation."""
        # Arrange
        collection_id = mock_collection["id"]
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
            response = client_with_mocked_services.post(
                f"/api/v2/chunking/collections/{collection_id}/chunk",
                headers=auth_headers,
                json=chunking_request,
            )

            # Assert
            assert response.status_code == 202
            operation = response.json()
            assert "operation_id" in operation
            assert operation["collection_id"] == collection_id
            assert operation["status"] == "pending"
            assert operation["strategy"] == "fixed_size"

    def test_start_chunking_invalid_config(
        self,
        client_with_mocked_services: TestClient,
        auth_headers: dict[str, str],
        mock_collection: dict[str, Any],
        mock_chunking_service: AsyncMock,
    ) -> None:
        """Test starting chunking with invalid config."""
        # Arrange
        mock_chunking_service.validate_config_for_collection.return_value = {
            "valid": False,
            "errors": ["chunk_size must be positive"],
        }

        collection_id = mock_collection["id"]
        chunking_request = {
            "strategy": "fixed_size",
            "config": {
                "strategy": "fixed_size",
                "chunk_size": -100,
                "chunk_overlap": 50,
            },
        }

        # Act
        response = client_with_mocked_services.post(
            f"/api/v2/chunking/collections/{collection_id}/chunk",
            headers=auth_headers,
            json=chunking_request,
        )

        # Assert
        # Pydantic validation catches the negative chunk_size and returns 422
        assert response.status_code == 422
        # Check that the error is about validation
        assert "detail" in response.json()
        # The validation error will be about chunk_size being out of range

    def test_get_operation_progress(
        self, client_with_mocked_services: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Test getting operation progress."""
        # Arrange
        operation_id = str(uuid.uuid4())

        # Act
        response = client_with_mocked_services.get(
            f"/api/v2/chunking/operations/{operation_id}/progress",
            headers=auth_headers,
        )

        # Assert
        assert response.status_code == 200
        progress = response.json()
        assert progress["operation_id"] == operation_id
        assert progress["status"] == "in_progress"
        assert progress["progress_percentage"] == 45.5
        assert progress["documents_processed"] == 5

    def test_get_operation_progress_not_found(
        self, client_with_mocked_services: TestClient, auth_headers: dict[str, str], mock_chunking_service: AsyncMock
    ) -> None:
        """Test getting progress for non-existent operation."""
        # Arrange
        mock_chunking_service.get_chunking_progress.return_value = None
        operation_id = str(uuid.uuid4())

        # Act
        response = client_with_mocked_services.get(
            f"/api/v2/chunking/operations/{operation_id}/progress",
            headers=auth_headers,
        )

        # Assert
        assert response.status_code == 404
        assert "Operation not found" in response.json()["detail"]


class TestStatsEndpoints:
    """Test statistics endpoints."""

    def test_get_chunking_stats(
        self, client_with_mocked_services: TestClient, auth_headers: dict[str, str], mock_collection: dict[str, Any]
    ) -> None:
        """Test getting chunking statistics."""
        # Arrange
        collection_id = mock_collection["id"]

        # Act
        response = client_with_mocked_services.get(
            f"/api/v2/chunking/collections/{collection_id}/chunking-stats",
            headers=auth_headers,
        )

        # Assert
        assert response.status_code == 200
        stats = response.json()
        assert stats["total_chunks"] == 1500
        assert stats["total_documents"] == 25
        assert stats["avg_chunk_size"] == 512
        assert stats["strategy_used"] == "recursive"


class TestErrorHandling:
    """Test error handling across endpoints."""

    def test_service_error_handling(
        self, client_with_mocked_services: TestClient, auth_headers: dict[str, str], mock_chunking_service: AsyncMock
    ) -> None:
        """Test handling of service errors."""
        # Arrange
        mock_chunking_service.get_available_strategies_for_api.side_effect = Exception("Database connection error")

        # Act
        response = client_with_mocked_services.get(
            "/api/v2/chunking/strategies",
            headers=auth_headers,
        )

        # Assert
        assert response.status_code == 500
        assert "Failed to list strategies" in response.json()["detail"]

    @pytest.mark.xfail(
        reason="Test passes in isolation but fails when run with other tests due to fixture state pollution"
    )
    def test_content_too_large_error(
        self, client_with_mocked_services: TestClient, auth_headers: dict[str, str], mock_chunking_service: AsyncMock
    ) -> None:
        """Test handling of content too large error."""
        from unittest.mock import patch

        # Use patch to ensure clean mock state
        with patch.object(
            mock_chunking_service,
            "validate_preview_content",
            side_effect=DocumentTooLargeError(size=2_000_000, max_size=1_000_000),
        ):
            preview_request = {
                "strategy": "fixed_size",
                "content": "x" * 2000000,
            }

            # Act
            response = client_with_mocked_services.post(
                "/api/v2/chunking/preview",
                headers=auth_headers,
                json=preview_request,
            )

            # Assert
            assert response.status_code == 507
            # Check that the error message mentions the size limit was exceeded
            error_detail = response.json()["detail"]
            assert "exceeds maximum" in error_detail or "Content exceeds maximum size" in error_detail


class TestAuthenticationSecurity:
    """Test authentication and security."""

    def test_all_endpoints_require_auth(self, unauthenticated_client: TestClient) -> None:
        """Test that all endpoints require authentication."""
        # List of endpoints to test
        endpoints = [
            ("GET", "/api/v2/chunking/strategies"),
            ("GET", "/api/v2/chunking/strategies/fixed_size"),
            ("POST", "/api/v2/chunking/strategies/recommend"),
            ("POST", "/api/v2/chunking/preview"),
            ("POST", "/api/v2/chunking/compare"),
        ]

        for method, endpoint in endpoints:
            # Act
            if method == "GET":
                response = unauthenticated_client.get(endpoint)
            elif method == "POST":
                response = unauthenticated_client.post(endpoint, json={})

            # Assert
            assert response.status_code == 401, f"Endpoint {endpoint} should require auth"
            assert "Not authenticated" in response.json()["detail"]

    def test_invalid_jwt_token(self, unauthenticated_client: TestClient) -> None:
        """Test that invalid JWT tokens are rejected."""
        # Arrange
        headers = {"Authorization": "Bearer invalid.jwt.token"}

        # Act
        response = unauthenticated_client.get(
            "/api/v2/chunking/strategies",
            headers=headers,
        )

        # Assert
        assert response.status_code == 401
        assert "Invalid authentication credentials" in response.json()["detail"]
