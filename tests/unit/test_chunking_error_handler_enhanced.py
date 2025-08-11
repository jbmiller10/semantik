#!/usr/bin/env python3

"""
Unit tests for enhanced ChunkingErrorHandler functionality.

Tests the new production features including correlation ID support,
state management, resource tracking, and advanced recovery strategies.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis.asyncio import Redis

from packages.shared.text_processing.base_chunker import ChunkResult
from packages.shared.chunking.infrastructure.exceptions import ResourceType
from packages.webui.services.chunking_error_handler import (
    ChunkingErrorHandler,
    ChunkingErrorType,
    CleanupResult,
    ErrorHandlingResult,
    ErrorReport,
    ResourceRecoveryAction,
)


@pytest.fixture()
async def mock_redis() -> None:
    """Create a mock Redis client."""
    redis = AsyncMock(spec=Redis)
    redis.setex = AsyncMock()
    redis.delete = AsyncMock()
    redis.lpos = AsyncMock(return_value=None)
    redis.rpush = AsyncMock()
    redis.lrem = AsyncMock()

    # Mock scan_iter to return an async iterator
    class AsyncIteratorMock:
        def __init__(self, items) -> None:
            self.items = items

        def __aiter__(self) -> None:
            return self

        async def __anext__(self) -> None:
            if self.items:
                return self.items.pop(0)
            raise StopAsyncIteration

    redis.scan_iter = MagicMock(return_value=AsyncIteratorMock([]))
    return redis


@pytest.fixture()
def error_handler(mock_redis) -> None:
    """Create an error handler with mock Redis."""
    return ChunkingErrorHandler(redis_client=mock_redis)


class TestHandleWithCorrelation:
    """Test correlation ID handling functionality."""

    @pytest.mark.asyncio()
    async def test_handle_with_correlation_success(self, error_handler, mock_redis) -> None:
        """Test successful error handling with correlation ID."""
        operation_id = "test_op_123"
        correlation_id = "corr_456"
        error = MemoryError("Out of memory")
        context = {
            "collection_id": "coll_789",
            "document_ids": ["doc_1", "doc_2"],
            "strategy": "semantic",
        }

        result = await error_handler.handle_with_correlation(
            operation_id=operation_id,
            correlation_id=correlation_id,
            error=error,
            context=context,
        )

        assert isinstance(result, ErrorHandlingResult)
        assert result.handled is True
        assert result.correlation_id == correlation_id
        assert result.operation_id == operation_id
        assert result.error_type == ChunkingErrorType.MEMORY_ERROR
        assert result.recovery_action in ["retry", "fail"]

        # Verify Redis was called to save state
        if result.recovery_action == "retry":
            mock_redis.setex.assert_called()

    @pytest.mark.asyncio()
    async def test_handle_with_correlation_no_retry(self, error_handler) -> None:
        """Test error handling when retry limit exceeded."""
        operation_id = "test_op_123"
        correlation_id = "corr_456"
        error = ValueError("Invalid input")

        # Simulate max retries reached
        error_handler.retry_counts[f"{operation_id}:validation_error"] = 10

        result = await error_handler.handle_with_correlation(
            operation_id=operation_id,
            correlation_id=correlation_id,
            error=error,
            context={},
        )

        assert result.recovery_action == "fail"
        assert result.retry_after is None


class TestHandleResourceExhaustion:
    """Test resource exhaustion handling."""

    @pytest.mark.asyncio()
    async def test_handle_memory_exhaustion(self, error_handler) -> None:
        """Test handling memory resource exhaustion."""
        operation_id = "test_op_123"

        with patch("psutil.virtual_memory") as mock_memory:
            mock_memory.return_value = MagicMock(
                total=16 * 1024**3,  # 16 GB
                available=2 * 1024**3,  # 2 GB
                percent=87.5,
            )

            result = await error_handler.handle_resource_exhaustion(
                operation_id=operation_id,
                resource_type=ResourceType.MEMORY,
                current_usage=14.0,
                limit=16.0,
            )

            assert isinstance(result, ResourceRecoveryAction)
            assert result.action == "reduce_batch"
            assert result.new_batch_size == 8  # High usage -> small batch
            assert result.alternative_strategy == "streaming"

    @pytest.mark.asyncio()
    async def test_handle_cpu_exhaustion(self, error_handler) -> None:
        """Test handling CPU resource exhaustion."""
        with patch("psutil.cpu_percent", return_value=85.0):
            result = await error_handler.handle_resource_exhaustion(
                operation_id="test_op",
                resource_type=ResourceType.CPU,
                current_usage=85.0,
                limit=100.0,
            )

            assert result.action == "wait_and_retry"
            assert result.wait_time == 60
            assert result.alternative_strategy == "character"

    @pytest.mark.asyncio()
    async def test_queue_operation_on_exhaustion(self, error_handler, mock_redis) -> None:
        """Test queueing operation when resources exhausted."""
        mock_redis.lpos = AsyncMock(side_effect=[None, 5])  # Not queued, then position 5

        result = await error_handler.handle_resource_exhaustion(
            operation_id="test_op",
            resource_type=ResourceType.MEMORY,
            current_usage=15.0,
            limit=16.0,
        )

        assert result.action == "queue"
        assert result.queue_position == 5
        assert result.wait_time == 150  # 5 * 30 seconds


class TestCleanupFailedOperation:
    """Test cleanup functionality."""

    @pytest.mark.asyncio()
    async def test_cleanup_save_partial(self, error_handler, mock_redis) -> None:
        """Test cleanup with partial results saved."""
        operation_id = "test_op_123"
        partial_results = [
            ChunkResult(
                chunk_id="chunk_1",
                text="chunk1",
                metadata={"doc_id": "doc1"},
                start_offset=0,
                end_offset=100,
            ),
        ]

        # Mock the save_partial_results method
        error_handler.save_partial_results = AsyncMock()

        result = await error_handler.cleanup_failed_operation(
            operation_id=operation_id,
            partial_results=partial_results,
            cleanup_strategy="save_partial",
        )

        assert isinstance(result, CleanupResult)
        assert result.cleaned is True
        assert result.partial_results_saved is True
        assert "redis_keys" in result.resources_freed

        # Verify Redis cleanup
        mock_redis.delete.assert_called()
        error_handler.save_partial_results.assert_called_once()

    @pytest.mark.asyncio()
    async def test_cleanup_rollback(self, error_handler) -> None:
        """Test cleanup with rollback strategy."""
        result = await error_handler.cleanup_failed_operation(
            operation_id="test_op",
            partial_results=None,
            cleanup_strategy="rollback",
        )

        assert result.rollback_performed is True
        assert result.partial_results_saved is False


class TestCreateErrorReport:
    """Test error report generation."""

    def test_create_error_report_with_history(self, error_handler) -> None:
        """Test creating error report from history."""
        operation_id = "test_op_123"

        # Add some error history
        error_handler._error_history[operation_id] = [
            {
                "timestamp": datetime.now(tz=UTC).isoformat(),
                "correlation_id": "corr_1",
                "error_type": "memory_error",
                "error_message": "Out of memory",
                "error_class": "MemoryError",
            },
            {
                "timestamp": datetime.now(tz=UTC).isoformat(),
                "correlation_id": "corr_2",
                "error_type": "timeout_error",
                "error_message": "Operation timed out",
                "error_class": "TimeoutError",
            },
        ]

        # Add retry counts
        error_handler.retry_counts[f"{operation_id}:memory_error"] = 2

        with patch("psutil.virtual_memory") as mock_memory, patch("psutil.cpu_percent", return_value=45.0):
            mock_memory.return_value = MagicMock(
                total=16 * 1024**3,
                available=8 * 1024**3,
                percent=50.0,
            )

            report = error_handler.create_error_report(operation_id)

        assert isinstance(report, ErrorReport)
        assert report.operation_id == operation_id
        assert report.total_errors == 2
        assert report.error_breakdown["memory_error"] == 1
        assert report.error_breakdown["timeout_error"] == 1
        assert len(report.recovery_attempts) == 1
        assert report.recovery_attempts[0]["retry_count"] == 2
        assert len(report.recommendations) > 0

    def test_create_error_report_with_provided_errors(self, error_handler) -> None:
        """Test creating error report with provided errors."""
        operation_id = "test_op_123"
        errors = [
            MemoryError("Out of memory"),
            TimeoutError("Timed out"),
            ValueError("Bad value"),
        ]

        report = error_handler.create_error_report(operation_id, errors)

        assert report.total_errors == 3
        assert "memory_error" in report.error_breakdown
        assert "timeout_error" in report.error_breakdown
        assert "unknown_error" in report.error_breakdown


class TestHelperMethods:
    """Test helper methods."""

    def test_calculate_adaptive_batch_size(self, error_handler) -> None:
        """Test adaptive batch size calculation."""
        # Very high usage
        assert error_handler._calculate_adaptive_batch_size(9.5, 10.0) == 4

        # High usage
        assert error_handler._calculate_adaptive_batch_size(8.5, 10.0) == 8

        # Moderate usage
        assert error_handler._calculate_adaptive_batch_size(7.5, 10.0) == 16

        # Low usage
        assert error_handler._calculate_adaptive_batch_size(5.0, 10.0) == 32

    def test_create_operation_fingerprint(self, error_handler) -> None:
        """Test operation fingerprint creation."""
        context = {
            "collection_id": "coll_123",
            "document_ids": ["doc_3", "doc_1", "doc_2"],
            "strategy": "semantic",
            "params": {"chunk_size": 1000},
        }

        fingerprint = error_handler._create_operation_fingerprint("op_123", context)

        assert isinstance(fingerprint, str)
        assert len(fingerprint) == 64  # SHA256 hex digest length

        # Same input should produce same fingerprint
        fingerprint2 = error_handler._create_operation_fingerprint("op_123", context)
        assert fingerprint == fingerprint2

    @pytest.mark.asyncio()
    async def test_save_operation_state(self, error_handler, mock_redis) -> None:
        """Test saving operation state to Redis."""
        operation_id = "test_op"
        correlation_id = "corr_123"
        context = {
            "collection_id": "coll_456",
            "checkpoint": {"processed": 10, "total": 100},
        }

        state = await error_handler._save_operation_state(
            operation_id,
            correlation_id,
            context,
            ChunkingErrorType.MEMORY_ERROR,
        )

        assert state is not None
        assert state["operation_id"] == operation_id
        assert state["correlation_id"] == correlation_id
        assert "fingerprint" in state

        # Verify Redis calls
        assert mock_redis.setex.call_count == 2  # State + checkpoint
