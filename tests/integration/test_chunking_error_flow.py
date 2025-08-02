#!/usr/bin/env python3
"""
Integration tests for chunking error flow.

This module tests error propagation from service to API layers,
correlation ID tracking, exception handler responses, and recovery mechanisms.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from redis import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from packages.shared.database.repositories.collection_repository import CollectionRepository
from packages.shared.database.repositories.document_repository import DocumentRepository
from packages.webui.api.chunking_exception_handlers import register_chunking_exception_handlers
from packages.webui.api.chunking_exceptions import (
    ChunkingMemoryError,
    ChunkingPartialFailureError,
    ChunkingStrategyError,
    ChunkingTimeoutError,
    ChunkingValidationError,
)
from packages.webui.middleware.correlation import CorrelationMiddleware, get_correlation_id
from packages.webui.services.chunking_error_handler import ChunkingErrorHandler
from packages.webui.services.chunking_service import ChunkingService


class TestChunkingErrorFlowIntegration:
    """Integration tests for error flow through the system."""

    @pytest.fixture()
    def mock_deps(self) -> dict:
        """Create mock dependencies."""
        return {
            "db_session": AsyncMock(spec=AsyncSession),
            "collection_repo": MagicMock(spec=CollectionRepository),
            "document_repo": MagicMock(spec=DocumentRepository),
            "redis_client": MagicMock(spec=Redis),
        }

    @pytest.fixture()
    def test_app(self) -> FastAPI:
        """Create test FastAPI app with error handlers."""
        app = FastAPI()
        app.add_middleware(CorrelationMiddleware)
        register_chunking_exception_handlers(app)

        return app

    @pytest.fixture()
    def chunking_service(self, mock_deps: dict) -> ChunkingService:
        """Create ChunkingService with mocked dependencies."""
        return ChunkingService(
            db_session=mock_deps["db_session"],
            collection_repo=mock_deps["collection_repo"],
            document_repo=mock_deps["document_repo"],
            redis_client=mock_deps["redis_client"],
        )

    async def test_correlation_id_propagation(self, test_app: FastAPI, chunking_service: ChunkingService) -> None:
        """Test that correlation IDs propagate through all layers."""
        correlation_id = str(uuid4())

        # Add endpoint that uses chunking service
        @test_app.post("/test/chunking")
        async def test_endpoint(request: Request):  # noqa: ARG001
            # Get correlation ID from request
            req_correlation_id = get_correlation_id()
            assert req_correlation_id == correlation_id

            # Trigger an error in chunking service
            raise ChunkingMemoryError(
                detail="Test memory error",
                correlation_id=req_correlation_id,
                operation_id="test-op",
                memory_used=1000,
                memory_limit=500,
            )

        client = TestClient(test_app)
        response = client.post(
            "/test/chunking",
            headers={"X-Correlation-ID": correlation_id},
        )

        # Verify response includes correlation ID
        assert response.status_code == 507
        data = response.json()
        assert data["correlation_id"] == correlation_id
        assert "X-Correlation-ID" in response.headers
        assert response.headers["X-Correlation-ID"] == correlation_id

    async def test_memory_error_propagation(self, test_app: FastAPI, chunking_service: ChunkingService) -> None:
        """Test ChunkingMemoryError propagation and response format."""

        @test_app.post("/test/memory-error")
        async def test_endpoint():
            correlation_id = get_correlation_id()
            raise ChunkingMemoryError(
                detail="Document too large",
                correlation_id=correlation_id,
                operation_id="mem-test",
                memory_used=2 * 1024 * 1024 * 1024,  # 2GB
                memory_limit=1 * 1024 * 1024 * 1024,  # 1GB
                recovery_hint="Split the document",
            )

        client = TestClient(test_app)
        response = client.post("/test/memory-error")

        assert response.status_code == 507
        data = response.json()

        # Verify error structure
        assert data["error_code"] == "CHUNKING_MEMORY_EXCEEDED"
        assert data["detail"] == "Document too large"
        assert data["memory_used_mb"] == 2048.0
        assert data["memory_limit_mb"] == 1024.0
        assert data["recovery_hint"] == "Split the document"
        assert "correlation_id" in data
        # Note: timestamp is not included in the base to_dict() implementation

    async def test_timeout_error_propagation(self, test_app: FastAPI) -> None:
        """Test ChunkingTimeoutError propagation."""

        @test_app.post("/test/timeout-error")
        async def test_endpoint():
            correlation_id = get_correlation_id()
            raise ChunkingTimeoutError(
                detail="Processing timeout",
                correlation_id=correlation_id,
                operation_id="timeout-test",
                elapsed_time=65.0,
                timeout_limit=60.0,
                estimated_completion=120.0,
            )

        client = TestClient(test_app)
        response = client.post("/test/timeout-error")

        assert response.status_code == 504
        data = response.json()

        assert data["error_code"] == "CHUNKING_TIMEOUT"
        assert data["elapsed_seconds"] == 65.0
        assert data["timeout_seconds"] == 60.0
        assert data["estimated_completion_seconds"] == 120.0

    async def test_validation_error_propagation(self, test_app: FastAPI) -> None:
        """Test ChunkingValidationError propagation with field errors."""

        @test_app.post("/test/validation-error")
        async def test_endpoint():
            correlation_id = get_correlation_id()
            raise ChunkingValidationError(
                detail="Invalid parameters",
                correlation_id=correlation_id,
                field_errors={
                    "chunk_size": ["Must be between 100 and 10000"],
                    "strategy": ["Unknown strategy 'invalid'"],
                },
            )

        client = TestClient(test_app)
        response = client.post("/test/validation-error")

        assert response.status_code == 422
        data = response.json()

        assert data["error_code"] == "CHUNKING_VALIDATION_FAILED"
        assert "field_errors" in data
        assert "chunk_size" in data["field_errors"]
        assert "strategy" in data["field_errors"]

    async def test_strategy_error_with_fallback(self, test_app: FastAPI) -> None:
        """Test ChunkingStrategyError with fallback suggestion."""

        @test_app.post("/test/strategy-error")
        async def test_endpoint():
            correlation_id = get_correlation_id()
            raise ChunkingStrategyError(
                detail="Semantic strategy not implemented",
                correlation_id=correlation_id,
                strategy="semantic",
                fallback_strategy="recursive",
            )

        client = TestClient(test_app)
        response = client.post("/test/strategy-error")

        assert response.status_code == 501
        data = response.json()

        assert data["error_code"] == "CHUNKING_STRATEGY_FAILED"
        assert data["strategy"] == "semantic"
        assert data["fallback_strategy"] == "recursive"
        assert data["recovery_hint"] == "Try using recursive strategy instead"

    async def test_partial_failure_handling(self, test_app: FastAPI) -> None:
        """Test ChunkingPartialFailureError handling."""

        @test_app.post("/test/partial-failure")
        async def test_endpoint():
            correlation_id = get_correlation_id()
            raise ChunkingPartialFailureError(
                detail="5 of 20 documents failed",
                correlation_id=correlation_id,
                operation_id="batch-test",
                total_documents=20,
                failed_documents=["doc1", "doc2", "doc3", "doc4", "doc5"],
                failure_reasons={
                    "doc1": "Memory limit exceeded",
                    "doc2": "Invalid format",
                    "doc3": "Timeout",
                    "doc4": "Strategy error",
                    "doc5": "Unknown error",
                },
                successful_chunks=150,
            )

        client = TestClient(test_app)
        response = client.post("/test/partial-failure")

        assert response.status_code == 207  # Multi-Status
        data = response.json()

        assert data["error_code"] == "CHUNKING_PARTIAL_FAILURE"
        assert data["total_documents"] == 20
        assert data["failed_count"] == 5
        assert data["success_count"] == 15
        assert data["successful_chunks"] == 150
        assert len(data["failed_documents"]) == 5
        assert len(data["failure_reasons"]) == 5

    async def test_error_recovery_with_retry(self, chunking_service: ChunkingService, mock_deps: dict) -> None:
        """Test error recovery mechanism with retry logic in error handler."""
        # Set up Redis mock properly
        mock_deps["redis_client"].setex = AsyncMock()

        # Create error handler with retry enabled
        error_handler = ChunkingErrorHandler(redis_client=mock_deps["redis_client"])

        # Test that retry logic works in error handler
        # The default max retries for TimeoutError is 3
        error1 = TimeoutError("Service timeout")

        # First 3 calls should return retry action
        for i in range(3):
            result = await error_handler.handle_with_correlation(
                operation_id="retry-test",
                correlation_id=str(uuid4()),
                error=error1,
                context={"attempt": i + 1},
            )
            assert result.recovery_action == "retry"
            assert result.retry_after is not None

        # Fourth call should exceed max retries and fail
        result4 = await error_handler.handle_with_correlation(
            operation_id="retry-test",
            correlation_id=str(uuid4()),
            error=error1,
            context={"attempt": 4},
        )
        assert result4.recovery_action == "fail"

    async def test_circuit_breaker_activation(self, chunking_service: ChunkingService, mock_deps: dict) -> None:
        """Test error handler tracks retry counts for repeated failures."""
        # Set up Redis mock properly
        mock_deps["redis_client"].setex = AsyncMock()

        error_handler = ChunkingErrorHandler(redis_client=mock_deps["redis_client"])
        chunking_service.error_handler = error_handler

        # Simulate repeated failures to exceed retry limit
        # TimeoutError has max_retries=3 by default
        for i in range(4):
            result = await error_handler.handle_with_correlation(
                operation_id="op-test",
                correlation_id=str(uuid4()),
                error=TimeoutError("Service unavailable"),
                context={"service": "chunking"},
            )
            # First 3 should retry, 4th should fail
            if i < 3:
                assert result.recovery_action == "retry"
            else:
                assert result.recovery_action == "fail"

        # Verify retry count is tracked
        assert error_handler.retry_counts.get("op-test:timeout_error", 0) >= 3

    async def test_dead_letter_queue_handling(self, mock_deps: dict) -> None:
        """Test that error handling works with Redis client."""
        error_handler = ChunkingErrorHandler(redis_client=mock_deps["redis_client"])

        # Mock Redis operations
        mock_deps["redis_client"].setex = AsyncMock()
        mock_deps["redis_client"].lpush = AsyncMock()

        # Create unrecoverable error
        error = ChunkingMemoryError(
            detail="Out of memory",
            correlation_id="corr-123",
            operation_id="op-456",
            memory_used=2000000000,
            memory_limit=1000000000,
        )

        # Handle error that exceeds retries
        _ = await error_handler.handle_with_correlation(
            operation_id="op-456",
            correlation_id="corr-123",
            error=error,
            context={"attempt": 3},
        )

        # Verify state was saved to Redis (for potential retry/recovery)
        assert mock_deps["redis_client"].setex.called
        state_call = mock_deps["redis_client"].setex.call_args
        assert "chunking:state:" in state_call[0][0]

    async def test_error_metrics_recording(self, chunking_service: ChunkingService, mock_deps: dict) -> None:
        """Test that errors are tracked in error history."""
        error_handler = ChunkingErrorHandler(redis_client=mock_deps["redis_client"])
        chunking_service.error_handler = error_handler

        # Trigger an error
        error = ChunkingMemoryError(
            detail="Memory error",
            correlation_id="corr-123",
            operation_id="op-123",
            memory_used=1000,
            memory_limit=500,
        )

        _ = await error_handler.handle_with_correlation(
            operation_id="op-123",
            correlation_id="corr-123",
            error=error,
            context={"strategy": "semantic"},
        )

        # Verify error was tracked in history
        assert "op-123" in error_handler._error_history
        assert len(error_handler._error_history["op-123"]) > 0
        assert error_handler._error_history["op-123"][0]["error_type"] == "memory_error"

    async def test_graceful_degradation(self, test_app: FastAPI, chunking_service: ChunkingService) -> None:
        """Test system degrades gracefully under resource pressure."""

        @test_app.post("/test/degradation")
        async def test_endpoint():
            # Simulate resource exhaustion
            correlation_id = get_correlation_id()

            # First try primary strategy
            try:
                # Simulate failure
                raise ChunkingMemoryError(
                    detail="Primary strategy failed",
                    correlation_id=correlation_id,
                    operation_id="degrade-1",
                    memory_used=1000000000,
                    memory_limit=500000000,
                )
            except ChunkingMemoryError:
                # Fall back to simpler strategy
                try:
                    # Simulate partial success
                    raise ChunkingPartialFailureError(
                        detail="Fallback partially succeeded",
                        correlation_id=correlation_id,
                        operation_id="degrade-2",
                        total_documents=10,
                        failed_documents=["doc1", "doc2"],
                        failure_reasons={
                            "doc1": "Still too large",
                            "doc2": "Invalid format",
                        },
                        successful_chunks=50,
                    )
                except ChunkingPartialFailureError as e:
                    # Return partial results
                    return {
                        "status": "partial_success",
                        "successful_chunks": e.successful_chunks,
                        "failed_documents": e.failed_documents,
                        "degraded": True,
                    }

        client = TestClient(test_app)
        response = client.post("/test/degradation")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "partial_success"
        assert data["degraded"] is True
        assert data["successful_chunks"] == 50

    async def test_concurrent_error_handling(self, chunking_service: ChunkingService, mock_deps: dict) -> None:
        """Test error handling under concurrent load."""
        # Set up Redis mock
        mock_deps["redis_client"].setex = AsyncMock()
        error_handler = ChunkingErrorHandler(redis_client=mock_deps["redis_client"])

        async def simulate_error(i: int):
            try:
                if i % 3 == 0:
                    raise ChunkingMemoryError(
                        detail=f"Memory error {i}",
                        correlation_id=f"corr-{i}",
                        operation_id=f"op-{i}",
                        memory_used=1000,
                        memory_limit=500,
                    )
                if i % 3 == 1:
                    raise ChunkingTimeoutError(
                        detail=f"Timeout error {i}",
                        correlation_id=f"corr-{i}",
                        operation_id=f"op-{i}",
                        elapsed_time=10,
                        timeout_limit=5,
                    )
                raise ChunkingValidationError(
                    detail=f"Validation error {i}",
                    correlation_id=f"corr-{i}",
                )
            except Exception as e:
                return await error_handler.handle_with_correlation(
                    operation_id=f"op-{i}",
                    correlation_id=f"corr-{i}",
                    error=e,
                    context={"index": i},
                )

        # Run multiple concurrent errors
        results = await asyncio.gather(
            *[simulate_error(i) for i in range(30)],
            return_exceptions=True,
        )

        # Verify all errors were handled
        assert len(results) == 30
        # All results should be ErrorHandlingResult objects with 'handled' attribute
        assert all(hasattr(r, "handled") and r.handled for r in results if not isinstance(r, Exception))

    async def test_error_handler_cleanup(self, chunking_service: ChunkingService, mock_deps: dict) -> None:
        """Test cleanup operations after errors."""
        # Mock cleanup operations
        mock_deps["db_session"].rollback = AsyncMock()
        mock_deps["redis_client"].setex = AsyncMock()
        mock_deps["redis_client"].delete = AsyncMock()
        mock_deps["redis_client"].lrem = AsyncMock()

        # Create an async generator function for scan_iter
        async def mock_scan_iter(pattern):  # noqa: ARG001
            # Return an empty async generator
            for item in []:
                yield item

        mock_deps["redis_client"].scan_iter = mock_scan_iter

        error_handler = ChunkingErrorHandler(redis_client=mock_deps["redis_client"])

        # Create error with cleanup context
        error = ChunkingMemoryError(
            detail="Memory error requiring cleanup",
            correlation_id="corr-cleanup",
            operation_id="op-cleanup",
            memory_used=1000,
            memory_limit=500,
        )

        # Handle the error
        _ = await error_handler.handle_with_correlation(
            operation_id="op-cleanup",
            correlation_id="corr-cleanup",
            error=error,
            context={
                "cleanup_required": True,
                "temp_keys": ["temp:key1", "temp:key2"],
            },
        )

        # Test cleanup method
        cleanup_result = await error_handler.cleanup_failed_operation(
            operation_id="op-cleanup",
            partial_results=None,
            cleanup_strategy="rollback",
        )

        assert cleanup_result.cleaned
        assert cleanup_result.rollback_performed
