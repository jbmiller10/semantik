#!/usr/bin/env python3

"""
Unit tests for ChunkingErrorHandler.

This module tests the error handling framework for chunking operations.
"""

from typing import Any

import pytest

from packages.shared.text_processing.base_chunker import ChunkResult
from packages.webui.api.chunking_exceptions import (
    ChunkingDependencyError,
    ChunkingResourceLimitError,
    ResourceType,
)
from packages.webui.services.chunking_error_handler import (
    ChunkingErrorHandler,
    ChunkingErrorType,
    ChunkingOperationResult,
    StreamRecoveryAction,
)


class TestChunkingErrorHandler:
    """Tests for ChunkingErrorHandler."""

    @pytest.fixture()
    def error_handler(self) -> ChunkingErrorHandler:
        """Create error handler instance."""
        return ChunkingErrorHandler()

    def test_classify_error_memory(self, error_handler: ChunkingErrorHandler) -> None:
        """Test classification of memory errors."""
        # Direct MemoryError
        assert error_handler.classify_error(MemoryError()) == ChunkingErrorType.MEMORY_ERROR

        # String contains "memory"
        error = Exception("Out of memory while processing")
        assert error_handler.classify_error(error) == ChunkingErrorType.MEMORY_ERROR

    def test_classify_error_timeout(self, error_handler: ChunkingErrorHandler) -> None:
        """Test classification of timeout errors."""
        # Direct TimeoutError
        assert error_handler.classify_error(TimeoutError()) == ChunkingErrorType.TIMEOUT_ERROR

        # String contains "timeout"
        error = Exception("Operation timeout exceeded")
        assert error_handler.classify_error(error) == ChunkingErrorType.TIMEOUT_ERROR

    def test_classify_error_encoding(self, error_handler: ChunkingErrorHandler) -> None:
        """Test classification of encoding errors."""
        # UnicodeDecodeError
        with pytest.raises(UnicodeDecodeError) as exc_info:
            b"\xff".decode("utf-8")
        assert error_handler.classify_error(exc_info.value) == ChunkingErrorType.INVALID_ENCODING

        # String contains "encoding"
        error = Exception("Invalid encoding detected")
        assert error_handler.classify_error(error) == ChunkingErrorType.INVALID_ENCODING

    def test_classify_error_permission(self, error_handler: ChunkingErrorHandler) -> None:
        """Test classification of permission errors."""
        error = Exception("Permission denied")
        assert error_handler.classify_error(error) == ChunkingErrorType.PERMISSION_ERROR

        error = Exception("Access denied to resource")
        assert error_handler.classify_error(error) == ChunkingErrorType.PERMISSION_ERROR

    def test_classify_error_network(self, error_handler: ChunkingErrorHandler) -> None:
        """Test classification of network errors."""
        error = Exception("Connection refused")
        assert error_handler.classify_error(error) == ChunkingErrorType.NETWORK_ERROR

        error = Exception("Network timeout")
        assert error_handler.classify_error(error) == ChunkingErrorType.NETWORK_ERROR

    def test_classify_error_validation(self, error_handler: ChunkingErrorHandler) -> None:
        """Test classification of validation errors."""
        error = Exception("Validation failed: invalid parameters")
        assert error_handler.classify_error(error) == ChunkingErrorType.VALIDATION_ERROR

    def test_classify_error_strategy(self, error_handler: ChunkingErrorHandler) -> None:
        """Test classification of strategy errors."""
        error = Exception("Strategy initialization failed")
        assert error_handler.classify_error(error) == ChunkingErrorType.STRATEGY_ERROR

        error = Exception("Chunker not available")
        assert error_handler.classify_error(error) == ChunkingErrorType.STRATEGY_ERROR

    def test_classify_error_unknown(self, error_handler: ChunkingErrorHandler) -> None:
        """Test classification of unknown errors."""
        error = Exception("Some random error")
        assert error_handler.classify_error(error) == ChunkingErrorType.UNKNOWN_ERROR

    def test_classify_error_chunking_specific_exceptions(
        self, error_handler: ChunkingErrorHandler
    ) -> None:
        """Chunking domain exceptions map to specialised error types and codes."""

        dependency_error = ChunkingDependencyError("Vector store unavailable", dependency="qdrant")
        dependency_result = error_handler.classify_error_detailed(dependency_error)
        assert dependency_result.error_type == ChunkingErrorType.DEPENDENCY_ERROR
        assert dependency_result.code == "dependency_error"

        resource_error = ChunkingResourceLimitError("Worker pool exhausted", resource_type=ResourceType.CPU)
        resource_result = error_handler.classify_error_detailed(resource_error)
        assert resource_result.error_type == ChunkingErrorType.RESOURCE_LIMIT_ERROR
        assert resource_result.code == "resource_limit_error"

        network_result = error_handler.classify_error_detailed(ConnectionError("Connection reset"))
        assert network_result.error_type == ChunkingErrorType.NETWORK_ERROR
        assert network_result.code == "connection_error"

    def test_get_retry_strategy(self, error_handler: ChunkingErrorHandler) -> None:
        """Test getting retry strategy for error types."""
        # Memory error strategy
        strategy = error_handler.get_retry_strategy(ChunkingErrorType.MEMORY_ERROR)
        assert strategy.action == "retry"
        assert strategy.max_retries == 2
        assert strategy.backoff_type == "exponential"
        assert strategy.fallback_strategy == "character"
        assert len(strategy.recommendations) > 0

        # Timeout error strategy
        strategy = error_handler.get_retry_strategy(ChunkingErrorType.TIMEOUT_ERROR)
        assert strategy.action == "retry"
        assert strategy.max_retries == 3
        assert strategy.backoff_type == "linear"

        # Partial failure strategy
        strategy = error_handler.get_retry_strategy(ChunkingErrorType.PARTIAL_FAILURE)
        assert strategy.action == "partial_save"
        assert strategy.max_retries == 0

        # Unknown error (no predefined strategy)
        strategy = error_handler.get_retry_strategy(ChunkingErrorType.UNKNOWN_ERROR)
        assert strategy.action == "fail"
        assert strategy.max_retries == 0

    def test_should_retry(self, error_handler: ChunkingErrorHandler) -> None:
        """Test retry decision logic."""
        operation_id = "test_op_123"

        # First retry should be allowed
        assert error_handler.should_retry(operation_id, ChunkingErrorType.MEMORY_ERROR)

        # Second retry should be allowed (max is 2)
        assert error_handler.should_retry(operation_id, ChunkingErrorType.MEMORY_ERROR)

        # Third retry should be denied
        assert not error_handler.should_retry(operation_id, ChunkingErrorType.MEMORY_ERROR)

        # Different error type should have its own counter
        assert error_handler.should_retry(operation_id, ChunkingErrorType.TIMEOUT_ERROR)

    def test_calculate_retry_delay_exponential(
        self,
        error_handler: ChunkingErrorHandler,
    ) -> None:
        """Test exponential backoff calculation."""
        operation_id = "test_op"
        error_type = ChunkingErrorType.MEMORY_ERROR

        # Simulate retries
        error_handler.retry_counts[f"{operation_id}:{error_type.value}"] = 1
        assert error_handler.calculate_retry_delay(operation_id, error_type) == 10

        error_handler.retry_counts[f"{operation_id}:{error_type.value}"] = 2
        assert error_handler.calculate_retry_delay(operation_id, error_type) == 20

        error_handler.retry_counts[f"{operation_id}:{error_type.value}"] = 3
        assert error_handler.calculate_retry_delay(operation_id, error_type) == 40

        # Should cap at 300 seconds
        error_handler.retry_counts[f"{operation_id}:{error_type.value}"] = 10
        assert error_handler.calculate_retry_delay(operation_id, error_type) == 300

    def test_calculate_retry_delay_linear(
        self,
        error_handler: ChunkingErrorHandler,
    ) -> None:
        """Test linear backoff calculation."""
        operation_id = "test_op"
        error_type = ChunkingErrorType.TIMEOUT_ERROR

        # Simulate retries
        error_handler.retry_counts[f"{operation_id}:{error_type.value}"] = 1
        assert error_handler.calculate_retry_delay(operation_id, error_type) == 10

        error_handler.retry_counts[f"{operation_id}:{error_type.value}"] = 2
        assert error_handler.calculate_retry_delay(operation_id, error_type) == 20

        error_handler.retry_counts[f"{operation_id}:{error_type.value}"] = 3
        assert error_handler.calculate_retry_delay(operation_id, error_type) == 30

    async def test_handle_streaming_failure_memory(
        self,
        error_handler: ChunkingErrorHandler,
    ) -> None:
        """Test handling streaming failure due to memory error."""
        result = await error_handler.handle_streaming_failure(
            document_id="doc123",
            bytes_processed=1000000,
            error=MemoryError("Out of memory"),
        )

        assert isinstance(result, StreamRecoveryAction)
        assert result.action == "retry_from_checkpoint"
        assert result.checkpoint == 1000000
        assert result.new_batch_size == 16  # Reduced from default

    async def test_handle_streaming_failure_timeout(
        self,
        error_handler: ChunkingErrorHandler,
    ) -> None:
        """Test handling streaming failure due to timeout."""
        result = await error_handler.handle_streaming_failure(
            document_id="doc123",
            bytes_processed=500000,
            error=TimeoutError("Operation timed out"),
        )

        assert result.action == "retry_with_extended_timeout"
        assert result.checkpoint == 500000
        assert result.new_timeout == 450  # Extended from default

    async def test_handle_streaming_failure_unrecoverable(
        self,
        error_handler: ChunkingErrorHandler,
    ) -> None:
        """Test handling unrecoverable streaming failure."""
        result = await error_handler.handle_streaming_failure(
            document_id="doc123",
            bytes_processed=100000,
            error=Exception("Critical system error"),
        )

        assert result.action == "mark_failed"
        assert result.error_details == "Critical system error"

    async def test_handle_partial_failure(
        self,
        error_handler: ChunkingErrorHandler,
    ) -> None:
        """Test handling partial chunking failures."""
        # Mock successful chunks
        processed_chunks = [
            ChunkResult(
                chunk_id="chunk_001",
                text="Success 1",
                start_offset=0,
                end_offset=100,
                metadata={},
            ),
            ChunkResult(
                chunk_id="chunk_002",
                text="Success 2",
                start_offset=100,
                end_offset=200,
                metadata={},
            ),
        ]

        # Failed documents and errors
        failed_documents = ["doc3", "doc4", "doc5"]
        errors = [
            MemoryError("Out of memory"),
            MemoryError("Out of memory"),
            TimeoutError("Timeout"),
        ]

        result = await error_handler.handle_partial_failure(
            operation_id="op123",
            processed_chunks=processed_chunks,
            failed_documents=failed_documents,
            errors=errors,
        )

        assert isinstance(result, ChunkingOperationResult)
        assert result.status == "partial_success"
        assert result.processed_count == 2
        assert result.failed_count == 3
        assert result.recovery_operation_id is not None
        assert result.recommendations is not None
        assert len(result.recommendations) > 0
        assert result.error_details is not None
        assert result.error_details["failure_analysis"]["most_common"] == "memory_error"

    def test_analyze_failures(self, error_handler: ChunkingErrorHandler) -> None:
        """Test failure analysis."""
        errors = [
            MemoryError(),
            MemoryError(),
            TimeoutError(),
            Exception("Unknown"),
        ]

        analysis = error_handler.analyze_failures(errors)

        assert analysis["total_errors"] == 4
        assert analysis["error_breakdown"]["memory_error"] == 2
        assert analysis["error_breakdown"]["timeout_error"] == 1
        assert analysis["error_breakdown"]["unknown_error"] == 1
        assert analysis["most_common"] == "memory_error"

    def test_create_recovery_strategy(
        self,
        error_handler: ChunkingErrorHandler,
    ) -> None:
        """Test recovery strategy creation."""
        # With most common error
        failure_analysis = {
            "most_common": "memory_error",
            "error_breakdown": {"memory_error": 5, "timeout_error": 2},
        }

        strategy = error_handler.create_recovery_strategy(
            failure_analysis,
            ["doc1", "doc2"],
        )

        assert strategy.action == "retry"
        assert strategy.max_retries == 2
        assert strategy.fallback_strategy == "character"

        # Without most common error
        failure_analysis_no_common: dict[str, Any] = {"most_common": None}

        strategy = error_handler.create_recovery_strategy(
            failure_analysis_no_common,
            ["doc1", "doc2"],
        )

        assert strategy.action == "partial_save"
        assert len(strategy.recommendations) > 0
