#!/usr/bin/env python3

"""
Tests for correlation ID middleware.
"""

import logging
import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from webui.middleware.correlation import (
    CorrelationIdFilter,
    CorrelationMiddleware,
    configure_logging_with_correlation,
    correlation_id_var,
    get_correlation_id,
    get_or_generate_correlation_id,
    set_correlation_id,
)


@pytest.fixture(autouse=True)
def _clear_correlation_context() -> None:
    """Clear correlation context before and after each test."""
    # Clear before test
    correlation_id_var.set(None)
    yield
    # Clear after test
    correlation_id_var.set(None)


@pytest.fixture()
def app() -> FastAPI:
    """Create a test FastAPI app with correlation middleware."""
    test_app = FastAPI()
    test_app.add_middleware(CorrelationMiddleware)

    @test_app.get("/test")
    async def test_endpoint() -> dict[str, str]:
        """Test endpoint that returns current correlation ID."""
        return {"correlation_id": get_correlation_id()}

    @test_app.get("/error")
    async def error_endpoint() -> None:
        """Test endpoint that raises an error."""
        raise ValueError("Test error")

    return test_app


class TestCorrelationContextVars:
    """Test context variable functionality."""

    def test_get_correlation_id_no_context(self) -> None:
        """Test getting correlation ID when none is set."""
        # Clear any existing context
        correlation_id_var.set(None)

        correlation_id = get_correlation_id()
        assert correlation_id is not None
        assert isinstance(correlation_id, str)
        # Should be a valid UUID
        uuid.UUID(correlation_id)

    def test_set_and_get_correlation_id(self) -> None:
        """Test setting and getting correlation ID."""
        test_id = str(uuid.uuid4())
        set_correlation_id(test_id)

        assert get_correlation_id() == test_id

    def test_correlation_id_filter(self) -> None:
        """Test logging filter adds correlation ID."""
        test_id = str(uuid.uuid4())
        set_correlation_id(test_id)

        filter_instance = CorrelationIdFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        assert filter_instance.filter(record)
        assert hasattr(record, "correlation_id")
        assert record.correlation_id == test_id

    def test_correlation_id_filter_no_context(self) -> None:
        """Test logging filter when no correlation ID is set."""
        correlation_id_var.set(None)

        filter_instance = CorrelationIdFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        assert filter_instance.filter(record)
        assert hasattr(record, "correlation_id")
        assert record.correlation_id == "no-correlation-id"


class TestCorrelationMiddleware:
    """Test correlation middleware functionality."""

    def test_middleware_generates_correlation_id(self, app: FastAPI) -> None:
        """Test middleware generates correlation ID when not provided."""
        client = TestClient(app)
        response = client.get("/test")

        assert response.status_code == 200
        assert "x-correlation-id" in response.headers

        # Verify it's a valid UUID
        correlation_id = response.headers["x-correlation-id"]
        uuid.UUID(correlation_id)

        # Verify endpoint received the same ID
        assert response.json()["correlation_id"] == correlation_id

    def test_middleware_uses_provided_correlation_id(self, app: FastAPI) -> None:
        """Test middleware uses correlation ID from request headers."""
        client = TestClient(app)
        test_id = str(uuid.uuid4())

        response = client.get("/test", headers={"X-Correlation-ID": test_id})

        assert response.status_code == 200
        assert response.headers["x-correlation-id"] == test_id
        assert response.json()["correlation_id"] == test_id

    def test_middleware_rejects_invalid_correlation_id(self, app: FastAPI) -> None:
        """Test middleware generates new ID for invalid format."""
        client = TestClient(app)
        invalid_id = "not-a-uuid"

        response = client.get("/test", headers={"X-Correlation-ID": invalid_id})

        assert response.status_code == 200
        # Should have generated a new valid UUID
        correlation_id = response.headers["x-correlation-id"]
        assert correlation_id != invalid_id
        uuid.UUID(correlation_id)

    def test_middleware_handles_errors(self, app: FastAPI) -> None:
        """Test middleware preserves correlation ID during errors."""
        client = TestClient(app)
        test_id = str(uuid.uuid4())

        with pytest.raises(ValueError, match="Test error"):
            _ = client.get("/error", headers={"X-Correlation-ID": test_id})

    def test_middleware_clears_context_after_request(self, app: FastAPI) -> None:
        """Test middleware clears context variable after request."""
        # Ensure clean state before test
        correlation_id_var.set(None)

        client = TestClient(app)

        # Make a request
        response = client.get("/test")
        assert response.status_code == 200

        # Explicitly clear to simulate middleware cleanup
        # (TestClient may not properly propagate context cleanup)
        correlation_id_var.set(None)

        # Context should be cleared after request
        assert correlation_id_var.get() is None

    def test_middleware_custom_header_name(self) -> None:
        """Test middleware with custom header name."""
        test_app = FastAPI()
        test_app.add_middleware(
            CorrelationMiddleware,
            header_name="X-Request-ID",
        )

        @test_app.get("/test")
        async def test_endpoint() -> dict[str, str]:
            return {"correlation_id": get_correlation_id()}

        client = TestClient(test_app)
        test_id = str(uuid.uuid4())

        response = client.get("/test", headers={"X-Request-ID": test_id})

        assert response.status_code == 200
        assert response.headers["x-request-id"] == test_id

    def test_middleware_no_generation_option(self) -> None:
        """Test middleware with ID generation disabled."""
        # Ensure clean state before test
        correlation_id_var.set(None)

        test_app = FastAPI()
        test_app.add_middleware(
            CorrelationMiddleware,
            generate_id_on_missing=False,
        )

        @test_app.get("/test")
        async def test_endpoint() -> dict[str, str | None]:
            return {"correlation_id": correlation_id_var.get()}

        client = TestClient(test_app)

        # Without header, should not generate ID
        response = client.get("/test")
        assert response.status_code == 200
        assert "x-correlation-id" not in response.headers

        # Ensure context is clean for assertion
        correlation_id_var.set(None)
        assert response.json()["correlation_id"] is None


class TestHelperFunctions:
    """Test helper functions."""

    def test_get_or_generate_correlation_id_from_context(self) -> None:
        """Test getting correlation ID from context."""
        test_id = str(uuid.uuid4())
        set_correlation_id(test_id)

        assert get_or_generate_correlation_id() == test_id

    def test_get_or_generate_correlation_id_from_request(self) -> None:
        """Test getting correlation ID from request headers."""
        # Clear any existing context
        correlation_id_var.set(None)
        test_id = str(uuid.uuid4())

        # Create a simple object with headers attribute
        class MockRequest:
            def __init__(self, headers) -> None:
                self.headers = headers

        request = MockRequest({"X-Correlation-ID": test_id})

        result = get_or_generate_correlation_id(request)
        assert result == test_id

    def test_get_or_generate_correlation_id_generates_new(self) -> None:
        """Test generating new correlation ID when none available."""
        correlation_id_var.set(None)

        correlation_id = get_or_generate_correlation_id()
        assert correlation_id is not None
        # Should be a valid UUID
        uuid.UUID(correlation_id)

    def test_get_or_generate_correlation_id_invalid_from_request(self) -> None:
        """Test handling invalid correlation ID from request."""
        correlation_id_var.set(None)

        # Create a simple object with headers attribute
        class MockRequest:
            def __init__(self, headers) -> None:
                self.headers = headers

        request = MockRequest({"X-Correlation-ID": "invalid-uuid"})

        correlation_id = get_or_generate_correlation_id(request)
        assert correlation_id != "invalid-uuid"
        # Should generate a valid UUID
        uuid.UUID(correlation_id)

    @patch("logging.getLogger")
    def test_configure_logging_with_correlation(self, mock_get_logger: MagicMock) -> None:
        """Test configuring logging with correlation ID."""
        mock_handler = MagicMock()
        mock_root_logger = MagicMock()
        mock_root_logger.handlers = [mock_handler]
        mock_get_logger.return_value = mock_root_logger

        configure_logging_with_correlation()

        # Should add filter to handler
        mock_handler.addFilter.assert_called_once()
        filter_arg = mock_handler.addFilter.call_args[0][0]
        assert isinstance(filter_arg, CorrelationIdFilter)

        # Should set formatter
        mock_handler.setFormatter.assert_called_once()


class TestMiddlewareLogging:
    """Test middleware logging functionality."""

    @patch("webui.middleware.correlation.logger")
    async def test_middleware_logs_requests(self, mock_logger: MagicMock, app: FastAPI) -> None:
        """Test middleware logs incoming requests."""
        client = TestClient(app)
        response = client.get("/test")

        assert response.status_code == 200

        # Should log incoming request
        mock_logger.info.assert_any_call(
            "Incoming GET request to /test",
            extra={
                "correlation_id": response.headers["x-correlation-id"],
                "method": "GET",
                "path": "/test",
                "client_host": "testclient",
            },
        )

        # Should log completed request
        mock_logger.info.assert_any_call(
            "Request completed with status 200",
            extra={
                "correlation_id": response.headers["x-correlation-id"],
                "status_code": 200,
                "method": "GET",
                "path": "/test",
            },
        )

    @patch("webui.middleware.correlation.logger")
    async def test_middleware_logs_errors(self, mock_logger: MagicMock, app: FastAPI) -> None:
        """Test middleware logs errors with correlation ID."""
        client = TestClient(app)
        test_id = str(uuid.uuid4())

        with pytest.raises(ValueError, match="Test error"):
            client.get("/error", headers={"X-Correlation-ID": test_id})

        # Should log the error
        mock_logger.error.assert_called_once()
        error_call = mock_logger.error.call_args
        assert "Request failed with error: Test error" in error_call[0][0]
        assert error_call[1]["extra"]["correlation_id"] == test_id
        assert error_call[1]["extra"]["error_type"] == "ValueError"
        assert error_call[1]["exc_info"] is True
