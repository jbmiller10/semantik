#!/usr/bin/env python3

"""
Tests for chunking exception handlers.
"""

import json
import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from webui.api.chunking_exception_handlers import (
    _create_error_response,
    _sanitize_error_detail,
    register_chunking_exception_handlers,
)
from shared.chunking.exceptions import (
    ChunkingConfigurationError,
    ChunkingDependencyError,
    ChunkingError,
    ChunkingMemoryError,
    ChunkingPartialFailureError,
    ChunkingResourceLimitError,
    ChunkingStrategyError,
    ChunkingTimeoutError,
    ChunkingValidationError,
    ResourceType,
)
from webui.middleware.correlation import CorrelationMiddleware


@pytest.fixture()
def app() -> FastAPI:
    """Create a test FastAPI app with chunking exception handlers."""
    test_app = FastAPI()
    test_app.add_middleware(CorrelationMiddleware)
    register_chunking_exception_handlers(test_app)

    @test_app.post("/test/memory-error")
    async def memory_error_endpoint() -> None:
        """Endpoint that raises memory error."""
        raise ChunkingMemoryError(
            detail="Out of memory while processing document",
            correlation_id=str(uuid.uuid4()),
            operation_id="op-123",
            memory_used=2_147_483_648,  # 2GB
            memory_limit=1_073_741_824,  # 1GB
        )

    @test_app.post("/test/timeout-error")
    async def timeout_error_endpoint() -> None:
        """Endpoint that raises timeout error."""
        raise ChunkingTimeoutError(
            detail="Processing exceeded time limit",
            correlation_id=str(uuid.uuid4()),
            operation_id="op-456",
            elapsed_time=120.5,
            timeout_limit=60.0,
            estimated_completion=180.0,
        )

    @test_app.post("/test/validation-error")
    async def validation_error_endpoint() -> None:
        """Endpoint that raises validation error."""
        raise ChunkingValidationError(
            detail="Invalid chunking parameters",
            correlation_id=str(uuid.uuid4()),
            field_errors={
                "chunk_size": ["Must be between 100 and 10000"],
                "chunk_overlap": ["Cannot exceed chunk_size"],
            },
        )

    @test_app.post("/test/strategy-error")
    async def strategy_error_endpoint() -> None:
        """Endpoint that raises strategy error."""
        raise ChunkingStrategyError(
            detail="Strategy 'advanced-nlp' not implemented",
            correlation_id=str(uuid.uuid4()),
            strategy="advanced-nlp",
            fallback_strategy="semantic",
        )

    @test_app.post("/test/resource-limit-error")
    async def resource_limit_error_endpoint() -> None:
        """Endpoint that raises resource limit error."""
        raise ChunkingResourceLimitError(
            detail="Too many concurrent operations",
            correlation_id=str(uuid.uuid4()),
            resource_type=ResourceType.THREADS,
            current_usage=100,
            limit=50,
        )

    @test_app.post("/test/partial-failure-error")
    async def partial_failure_error_endpoint() -> None:
        """Endpoint that raises partial failure error."""
        raise ChunkingPartialFailureError(
            detail="Some documents failed to process",
            correlation_id=str(uuid.uuid4()),
            operation_id="op-789",
            total_documents=100,
            failed_documents=["doc-1", "doc-2", "doc-3"],
            failure_reasons={
                "doc-1": "Invalid format",
                "doc-2": "Corrupted content",
                "doc-3": "Encoding error",
            },
            successful_chunks=450,
        )

    @test_app.post("/test/configuration-error")
    async def configuration_error_endpoint() -> None:
        """Endpoint that raises configuration error."""
        raise ChunkingConfigurationError(
            detail="Invalid chunking configuration",
            correlation_id=str(uuid.uuid4()),
            config_errors=[
                "Embedding model not supported",
                "Conflicting strategy parameters",
            ],
        )

    @test_app.post("/test/dependency-error")
    async def dependency_error_endpoint() -> None:
        """Endpoint that raises dependency error."""
        raise ChunkingDependencyError(
            detail="Embedding service unavailable",
            correlation_id=str(uuid.uuid4()),
            dependency="embedding-service",
            dependency_error="Connection timeout",
        )

    @test_app.post("/test/base-error")
    async def base_error_endpoint() -> None:
        """Endpoint that raises base chunking error."""
        raise ChunkingError(
            detail="Generic chunking error",
            correlation_id=str(uuid.uuid4()),
        )

    return test_app


class TestErrorSanitization:
    """Test error message sanitization."""

    def test_sanitize_error_detail_production(self) -> None:
        """Test sanitizing error details in production mode."""
        # Test file path sanitization
        detail = "Error in /home/user/app/config.env file"
        sanitized = _sanitize_error_detail(detail, is_production=True)
        assert "[REDACTED]" in sanitized
        assert "/home/user/app/config.env" not in sanitized

        # Test database connection string sanitization
        detail = "Failed to connect to postgres://user:pass@localhost/db"
        sanitized = _sanitize_error_detail(detail, is_production=True)
        assert "[REDACTED]" in sanitized
        assert "user:pass" not in sanitized

        # Test API key sanitization
        detail = "Invalid api_key='sk-1234567890abcdef'"
        sanitized = _sanitize_error_detail(detail, is_production=True)
        assert "[REDACTED]" in sanitized
        assert "sk-1234567890abcdef" not in sanitized

    def test_sanitize_error_detail_development(self) -> None:
        """Test error details are not sanitized in development mode."""
        sensitive_detail = "Error in /home/user/app/config.env with api_key='secret'"
        sanitized = _sanitize_error_detail(sensitive_detail, is_production=False)
        assert sanitized == sensitive_detail


class TestExceptionHandlers:
    """Test individual exception handlers."""

    def test_memory_error_handler(self, app: FastAPI) -> None:
        """Test memory error handler returns correct response."""
        client = TestClient(app)
        response = client.post("/test/memory-error")

        assert response.status_code == 507  # Insufficient Storage
        data = response.json()

        assert data["error_code"] == "CHUNKING_MEMORY_EXCEEDED"
        assert "memory_used_mb" in data
        assert "memory_limit_mb" in data
        assert data["memory_used_mb"] == 2048.0
        assert data["memory_limit_mb"] == 1024.0
        assert "recovery_hint" in data
        assert "x-correlation-id" in response.headers

    def test_timeout_error_handler(self, app: FastAPI) -> None:
        """Test timeout error handler returns correct response."""
        client = TestClient(app)
        response = client.post("/test/timeout-error")

        assert response.status_code == 504  # Gateway Timeout
        data = response.json()

        assert data["error_code"] == "CHUNKING_TIMEOUT"
        assert data["elapsed_seconds"] == 120.5
        assert data["timeout_seconds"] == 60.0
        assert data["estimated_completion_seconds"] == 180.0
        assert "recovery_hint" in data

    def test_validation_error_handler(self, app: FastAPI) -> None:
        """Test validation error handler returns correct response."""
        client = TestClient(app)
        response = client.post("/test/validation-error")

        assert response.status_code == 422  # Unprocessable Entity
        data = response.json()

        assert data["error_code"] == "CHUNKING_VALIDATION_FAILED"
        assert "field_errors" in data
        assert "chunk_size" in data["field_errors"]
        assert "chunk_overlap" in data["field_errors"]

    def test_strategy_error_handler(self, app: FastAPI) -> None:
        """Test strategy error handler returns correct response."""
        client = TestClient(app)
        response = client.post("/test/strategy-error")

        assert response.status_code == 501  # Not Implemented
        data = response.json()

        assert data["error_code"] == "CHUNKING_STRATEGY_FAILED"
        assert data["strategy"] == "advanced-nlp"
        assert data["fallback_strategy"] == "semantic"
        assert "recovery_hint" in data

    def test_resource_limit_error_handler(self, app: FastAPI) -> None:
        """Test resource limit error handler returns correct response."""
        client = TestClient(app)
        response = client.post("/test/resource-limit-error")

        assert response.status_code == 503  # Service Unavailable
        assert response.headers["retry-after"] == "30"
        data = response.json()

        assert data["error_code"] == "CHUNKING_RESOURCE_LIMIT"
        assert data["resource_type"] == "threads"
        assert data["current_usage"] == 100
        assert data["limit"] == 50
        assert "recovery_hint" in data

    def test_partial_failure_error_handler(self, app: FastAPI) -> None:
        """Test partial failure error handler returns correct response."""
        client = TestClient(app)
        response = client.post("/test/partial-failure-error")

        assert response.status_code == 207  # Multi-Status
        data = response.json()

        assert data["error_code"] == "CHUNKING_PARTIAL_FAILURE"
        assert data["total_documents"] == 100
        assert data["failed_count"] == 3
        assert data["success_count"] == 97
        assert len(data["failed_documents"]) == 3
        assert "failure_reasons" in data
        assert data["successful_chunks"] == 450

    def test_configuration_error_handler(self, app: FastAPI) -> None:
        """Test configuration error handler returns correct response."""
        client = TestClient(app)
        response = client.post("/test/configuration-error")

        assert response.status_code == 500  # Internal Server Error
        data = response.json()

        assert data["error_code"] == "CHUNKING_CONFIG_ERROR"
        assert "config_errors" in data
        assert len(data["config_errors"]) == 2

    def test_dependency_error_handler(self, app: FastAPI) -> None:
        """Test dependency error handler returns correct response."""
        client = TestClient(app)
        response = client.post("/test/dependency-error")

        assert response.status_code == 503  # Service Unavailable
        assert response.headers["retry-after"] == "60"
        data = response.json()

        assert data["error_code"] == "CHUNKING_DEPENDENCY_FAILED"
        assert data["dependency"] == "embedding-service"
        assert data["dependency_error"] == "Connection timeout"
        assert "recovery_hint" in data

    def test_base_error_handler(self, app: FastAPI) -> None:
        """Test base error handler as fallback."""
        client = TestClient(app)
        response = client.post("/test/base-error")

        assert response.status_code == 500  # Internal Server Error
        data = response.json()

        assert data["error_code"] == "ChunkingError"
        assert data["type"] == "ChunkingError"


class TestErrorResponseCreation:
    """Test error response creation logic."""

    @patch("webui.api.chunking_exception_handlers.logger")
    def test_create_error_response_logging(self, mock_logger: MagicMock) -> None:
        """Test error response creation logs appropriately."""
        request = MagicMock(spec=Request)
        request.method = "POST"
        request.url.path = "/test"
        request.query_params = {}

        # Test 5xx error logging
        exc = ChunkingError("Server error", correlation_id=str(uuid.uuid4()))
        _create_error_response(request, exc, 500)
        mock_logger.error.assert_called_once()

        # Reset mock
        mock_logger.reset_mock()

        # Test 4xx error logging
        exc = ChunkingValidationError("Client error", correlation_id=str(uuid.uuid4()))
        _create_error_response(request, exc, 422)
        mock_logger.warning.assert_called_once()

    def test_create_error_response_structure(self) -> None:
        """Test error response has correct structure."""
        request = MagicMock(spec=Request)
        request.method = "POST"
        request.url.path = "/test"
        request.query_params = {"param": "value"}

        exc = ChunkingError(
            detail="Test error",
            correlation_id=str(uuid.uuid4()),
            operation_id="op-123",
        )

        response = _create_error_response(request, exc, 500, is_production=False)

        assert response.status_code == 500
        assert "X-Correlation-ID" in response.headers

        # Parse response body
        data = json.loads(response.body)
        assert data["detail"] == "Test error"
        assert data["correlation_id"] == exc.correlation_id
        assert data["operation_id"] == "op-123"
        assert data["error_code"] == "ChunkingError"
        assert data["request"]["method"] == "POST"
        assert data["request"]["path"] == "/test"
        assert data["request"]["query_params"] == {"param": "value"}

    def test_create_error_response_production_mode(self) -> None:
        """Test error response in production mode."""
        request = MagicMock(spec=Request)
        request.method = "POST"
        request.url.path = "/test"
        request.query_params = {"secret": "value"}

        exc = ChunkingError(
            detail="Error with /etc/passwd file",
            correlation_id=str(uuid.uuid4()),
        )

        response = _create_error_response(request, exc, 500, is_production=True)

        data = json.loads(response.body)
        # Query params should be None in production
        assert data["request"]["query_params"] is None
        # Detail should be sanitized
        assert "[REDACTED]" in data["detail"]


class TestHandlerRegistration:
    """Test exception handler registration."""

    def test_register_chunking_exception_handlers(self) -> None:
        """Test all handlers are registered correctly."""
        app = FastAPI()
        register_chunking_exception_handlers(app)

        # Check that all exception types have handlers registered
        exception_types = [
            ChunkingMemoryError,
            ChunkingTimeoutError,
            ChunkingValidationError,
            ChunkingStrategyError,
            ChunkingResourceLimitError,
            ChunkingPartialFailureError,
            ChunkingConfigurationError,
            ChunkingDependencyError,
            ChunkingError,  # Base class
        ]

        for exc_type in exception_types:
            assert exc_type in app.exception_handlers


class TestIntegrationScenarios:
    """Test real-world integration scenarios."""

    def test_error_with_custom_correlation_id(self, app: FastAPI) -> None:
        """Test error handling preserves custom correlation ID."""
        client = TestClient(app)
        custom_id = str(uuid.uuid4())

        response = client.post(
            "/test/memory-error",
            headers={"X-Correlation-ID": custom_id},
        )

        assert response.status_code == 507
        # Response should have the same correlation ID
        assert response.headers["x-correlation-id"] == custom_id

    def test_multiple_errors_different_correlation_ids(self, app: FastAPI) -> None:
        """Test multiple errors have different correlation IDs."""
        client = TestClient(app)

        response1 = client.post("/test/memory-error")
        response2 = client.post("/test/timeout-error")

        id1 = response1.headers["x-correlation-id"]
        id2 = response2.headers["x-correlation-id"]

        assert id1 != id2
        assert uuid.UUID(id1)  # Valid UUID
        assert uuid.UUID(id2)  # Valid UUID
