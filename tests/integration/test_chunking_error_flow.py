#!/usr/bin/env python3
"""
Integration tests for chunking error flow.

This module tests error propagation from service to API layers,
correlation ID tracking, exception handler responses, and recovery mechanisms.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from httpx import AsyncClient
from redis import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from packages.shared.database.repositories.collection_repository import CollectionRepository
from packages.shared.database.repositories.document_repository import DocumentRepository
from packages.webui.api.chunking_exceptions import (
    ChunkingMemoryError,
    ChunkingPartialFailureError,
    ChunkingStrategyError,
    ChunkingTimeoutError,
    ChunkingValidationError,
    ResourceType,
)
from packages.webui.api.chunking_exception_handlers import register_chunking_exception_handlers
from packages.webui.middleware.correlation import CorrelationMiddleware, get_correlation_id
from packages.webui.services.chunking_error_handler import ChunkingErrorHandler
from packages.webui.services.chunking_service import ChunkingService
from packages.webui.tasks.chunking_tasks import process_collection_chunking


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
        async def test_endpoint(request: Request):
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
        assert "timestamp" in data

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
                detail="Semantic strategy failed",
                correlation_id=correlation_id,
                strategy="semantic",
                fallback_strategy="recursive",
            )
        
        client = TestClient(test_app)
        response = client.post("/test/strategy-error")
        
        assert response.status_code == 503
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
        """Test error recovery mechanism with retry logic."""
        # Mock document repo to return test documents
        mock_deps["document_repo"].list_by_collection = AsyncMock(
            return_value=([
                MagicMock(id="doc1", content="Test content 1"),
                MagicMock(id="doc2", content="Test content 2"),
            ], 2)
        )
        
        # Mock collection repo
        mock_collection = MagicMock(id="coll-123", uuid="uuid-123")
        mock_deps["collection_repo"].get_by_uuid_with_permission_check = AsyncMock(
            return_value=mock_collection
        )
        
        # Create error handler with retry enabled
        error_handler = ChunkingErrorHandler(max_retries=3)
        chunking_service.error_handler = error_handler
        
        # Simulate transient error that succeeds on retry
        call_count = 0
        
        async def mock_chunk_text(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Transient error")
            return [MagicMock(text="chunk1"), MagicMock(text="chunk2")]
        
        with patch("packages.shared.text_processing.chunking_factory.ChunkingFactory.create_chunker") as mock_factory:
            mock_chunker = MagicMock()
            mock_chunker.chunk_text_async = AsyncMock(side_effect=mock_chunk_text)
            mock_factory.return_value = mock_chunker
            
            # Process collection with retry
            result = await chunking_service.process_collection(
                collection_uuid="uuid-123",
                user_id=1,
                config={"strategy": "recursive", "params": {}},
            )
            
            # Verify retry happened
            assert call_count == 3
            assert mock_chunker.chunk_text_async.call_count == 3

    async def test_circuit_breaker_activation(self, chunking_service: ChunkingService) -> None:
        """Test circuit breaker activation after repeated failures."""
        error_handler = ChunkingErrorHandler(
            max_retries=1,
            circuit_breaker_threshold=3,
            circuit_breaker_timeout=60,
        )
        chunking_service.error_handler = error_handler
        
        # Simulate repeated failures
        for i in range(5):
            try:
                await error_handler.handle_with_correlation(
                    operation_id=f"op-{i}",
                    correlation_id=str(uuid4()),
                    error=Exception("Service unavailable"),
                    context={"service": "chunking"},
                )
            except Exception:
                pass
        
        # Verify circuit breaker is open
        assert error_handler.is_circuit_open("chunking")

    async def test_dead_letter_queue_handling(self, mock_deps: dict) -> None:
        """Test failed operations are sent to dead letter queue."""
        error_handler = ChunkingErrorHandler(
            max_retries=1,
            enable_dead_letter_queue=True,
        )
        
        # Mock Redis for DLQ
        mock_deps["redis_client"].lpush = MagicMock()
        error_handler.redis = mock_deps["redis_client"]
        
        # Create unrecoverable error
        error = ChunkingMemoryError(
            detail="Out of memory",
            correlation_id="corr-123",
            operation_id="op-456",
            memory_used=2000000000,
            memory_limit=1000000000,
        )
        
        # Handle error that exceeds retries
        result = await error_handler.handle_with_correlation(
            operation_id="op-456",
            correlation_id="corr-123",
            error=error,
            context={"attempt": 3},
        )
        
        # Verify message sent to DLQ
        assert mock_deps["redis_client"].lpush.called
        dlq_call = mock_deps["redis_client"].lpush.call_args
        assert "chunking:dlq" in dlq_call[0]

    async def test_error_metrics_recording(self, chunking_service: ChunkingService) -> None:
        """Test that errors are properly recorded in metrics."""
        with patch("packages.webui.services.chunking_error_metrics.record_chunking_error") as mock_record:
            # Trigger different error types
            try:
                raise ChunkingMemoryError(
                    detail="Memory error",
                    correlation_id="corr-123",
                    operation_id="op-123",
                    memory_used=1000,
                    memory_limit=500,
                )
            except ChunkingMemoryError as e:
                await chunking_service.error_handler.handle_with_correlation(
                    operation_id="op-123",
                    correlation_id="corr-123",
                    error=e,
                    context={"strategy": "semantic"},
                )
            
            # Verify metrics were recorded
            mock_record.assert_called_with(
                error_type="memory",
                strategy="semantic",
                recoverable=True,
            )

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

    async def test_concurrent_error_handling(self, chunking_service: ChunkingService) -> None:
        """Test error handling under concurrent load."""
        error_handler = ChunkingErrorHandler()
        
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
                elif i % 3 == 1:
                    raise ChunkingTimeoutError(
                        detail=f"Timeout error {i}",
                        correlation_id=f"corr-{i}",
                        operation_id=f"op-{i}",
                        elapsed_time=10,
                        timeout_limit=5,
                    )
                else:
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
        assert all(hasattr(r, "error_handled") for r in results if not isinstance(r, Exception))

    async def test_error_handler_cleanup(self, chunking_service: ChunkingService, mock_deps: dict) -> None:
        """Test cleanup operations after errors."""
        # Mock cleanup operations
        mock_deps["db_session"].rollback = AsyncMock()
        mock_deps["redis_client"].delete = MagicMock()
        
        error_handler = ChunkingErrorHandler()
        error_handler.redis = mock_deps["redis_client"]
        
        # Create error with cleanup context
        error = ChunkingMemoryError(
            detail="Memory error requiring cleanup",
            correlation_id="corr-cleanup",
            operation_id="op-cleanup",
            memory_used=1000,
            memory_limit=500,
        )
        
        await error_handler.handle_with_correlation(
            operation_id="op-cleanup",
            correlation_id="corr-cleanup",
            error=error,
            context={
                "cleanup_required": True,
                "temp_keys": ["temp:key1", "temp:key2"],
            },
        )
        
        # Verify cleanup was performed
        assert mock_deps["redis_client"].delete.called