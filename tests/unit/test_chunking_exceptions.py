#!/usr/bin/env python3
"""
Unit tests for chunking custom exceptions.

This module tests all custom exception classes, their serialization,
inheritance hierarchy, and attribute handling.
"""

import pytest

from packages.webui.api.chunking_exceptions import (
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


class TestChunkingExceptions:
    """Test suite for chunking exception classes."""

    def test_base_chunking_error(self) -> None:
        """Test base ChunkingError class."""
        error = ChunkingError(
            detail="Test error",
            correlation_id="corr-123",
            operation_id="op-456",
            error_code="TEST_ERROR",
        )

        assert error.detail == "Test error"
        assert error.correlation_id == "corr-123"
        assert error.operation_id == "op-456"
        assert error.error_code == "TEST_ERROR"
        assert str(error) == "Test error"

    def test_base_chunking_error_default_error_code(self) -> None:
        """Test that error_code defaults to class name."""
        error = ChunkingError(
            detail="Test error",
            correlation_id="corr-123",
        )

        assert error.error_code == "ChunkingError"

    def test_base_chunking_error_serialization(self) -> None:
        """Test base error serialization to dict."""
        error = ChunkingError(
            detail="Test error",
            correlation_id="corr-123",
            operation_id="op-456",
            error_code="TEST_ERROR",
        )

        result = error.to_dict()

        assert result == {
            "error_code": "TEST_ERROR",
            "detail": "Test error",
            "correlation_id": "corr-123",
            "operation_id": "op-456",
            "type": "ChunkingError",
        }

    def test_chunking_memory_error(self) -> None:
        """Test ChunkingMemoryError with all attributes."""
        error = ChunkingMemoryError(
            detail="Memory limit exceeded",
            correlation_id="corr-123",
            operation_id="op-456",
            memory_used=1024 * 1024 * 1024,  # 1GB
            memory_limit=512 * 1024 * 1024,  # 512MB
            recovery_hint="Use smaller chunks",
        )

        assert error.detail == "Memory limit exceeded"
        assert error.memory_used == 1024 * 1024 * 1024
        assert error.memory_limit == 512 * 1024 * 1024
        assert error.recovery_hint == "Use smaller chunks"
        assert error.error_code == "CHUNKING_MEMORY_EXCEEDED"

    def test_chunking_memory_error_default_recovery_hint(self) -> None:
        """Test ChunkingMemoryError with default recovery hint."""
        error = ChunkingMemoryError(
            detail="Memory limit exceeded",
            correlation_id="corr-123",
            operation_id="op-456",
            memory_used=1024 * 1024 * 1024,
            memory_limit=512 * 1024 * 1024,
        )

        assert error.recovery_hint == "Try processing smaller documents or use a more memory-efficient strategy"

    def test_chunking_memory_error_serialization(self) -> None:
        """Test ChunkingMemoryError serialization includes MB values."""
        error = ChunkingMemoryError(
            detail="Memory limit exceeded",
            correlation_id="corr-123",
            operation_id="op-456",
            memory_used=1024 * 1024 * 1024,  # 1GB
            memory_limit=512 * 1024 * 1024,  # 512MB
        )

        result = error.to_dict()

        assert result["memory_used_mb"] == 1024.0
        assert result["memory_limit_mb"] == 512.0
        assert "recovery_hint" in result
        assert result["error_code"] == "CHUNKING_MEMORY_EXCEEDED"

    def test_chunking_timeout_error(self) -> None:
        """Test ChunkingTimeoutError with all attributes."""
        error = ChunkingTimeoutError(
            detail="Operation timed out",
            correlation_id="corr-123",
            operation_id="op-456",
            elapsed_time=65.5,
            timeout_limit=60.0,
            estimated_completion=120.0,
        )

        assert error.elapsed_time == 65.5
        assert error.timeout_limit == 60.0
        assert error.estimated_completion == 120.0
        assert error.error_code == "CHUNKING_TIMEOUT"

    def test_chunking_timeout_error_serialization(self) -> None:
        """Test ChunkingTimeoutError serialization with rounded values."""
        error = ChunkingTimeoutError(
            detail="Operation timed out",
            correlation_id="corr-123",
            operation_id="op-456",
            elapsed_time=65.5678,
            timeout_limit=60.0,
            estimated_completion=120.456,
        )

        result = error.to_dict()

        assert result["elapsed_seconds"] == 65.57
        assert result["timeout_seconds"] == 60.0
        assert result["estimated_completion_seconds"] == 120.46
        assert result["recovery_hint"] == "Consider using a faster strategy or processing in smaller batches"

    def test_chunking_timeout_error_no_estimated_completion(self) -> None:
        """Test ChunkingTimeoutError without estimated completion."""
        error = ChunkingTimeoutError(
            detail="Operation timed out",
            correlation_id="corr-123",
            operation_id="op-456",
            elapsed_time=65.5,
            timeout_limit=60.0,
        )

        result = error.to_dict()

        assert result["estimated_completion_seconds"] is None

    def test_chunking_validation_error(self) -> None:
        """Test ChunkingValidationError with field errors."""
        field_errors = {
            "chunk_size": ["Must be positive", "Must be less than 10000"],
            "strategy": ["Invalid strategy name"],
        }

        error = ChunkingValidationError(
            detail="Validation failed",
            correlation_id="corr-123",
            field_errors=field_errors,
            operation_id="op-456",
        )

        assert error.field_errors == field_errors
        assert error.error_code == "CHUNKING_VALIDATION_FAILED"

    def test_chunking_validation_error_empty_fields(self) -> None:
        """Test ChunkingValidationError with no field errors."""
        error = ChunkingValidationError(
            detail="Validation failed",
            correlation_id="corr-123",
        )

        assert error.field_errors == {}

    def test_chunking_validation_error_serialization(self) -> None:
        """Test ChunkingValidationError serialization."""
        field_errors = {"chunk_size": ["Must be positive"]}

        error = ChunkingValidationError(
            detail="Validation failed",
            correlation_id="corr-123",
            field_errors=field_errors,
        )

        result = error.to_dict()

        assert result["field_errors"] == field_errors

    def test_chunking_strategy_error(self) -> None:
        """Test ChunkingStrategyError with fallback."""
        error = ChunkingStrategyError(
            detail="Strategy failed",
            correlation_id="corr-123",
            strategy="semantic",
            fallback_strategy="recursive",
            operation_id="op-456",
        )

        assert error.strategy == "semantic"
        assert error.fallback_strategy == "recursive"
        assert error.error_code == "CHUNKING_STRATEGY_FAILED"

    def test_chunking_strategy_error_serialization(self) -> None:
        """Test ChunkingStrategyError serialization with recovery hint."""
        error = ChunkingStrategyError(
            detail="Strategy failed",
            correlation_id="corr-123",
            strategy="semantic",
            fallback_strategy="recursive",
        )

        result = error.to_dict()

        assert result["strategy"] == "semantic"
        assert result["fallback_strategy"] == "recursive"
        assert result["recovery_hint"] == "Try using recursive strategy instead"

    def test_chunking_strategy_error_no_fallback(self) -> None:
        """Test ChunkingStrategyError without fallback."""
        error = ChunkingStrategyError(
            detail="Strategy failed",
            correlation_id="corr-123",
            strategy="semantic",
        )

        result = error.to_dict()

        assert result["fallback_strategy"] is None
        assert result["recovery_hint"] is None

    def test_resource_type_enum(self) -> None:
        """Test ResourceType enum values."""
        assert ResourceType.MEMORY.value == "memory"
        assert ResourceType.CPU.value == "cpu"
        assert ResourceType.DISK.value == "disk"
        assert ResourceType.CONNECTIONS.value == "connections"
        assert ResourceType.THREADS.value == "threads"

    def test_chunking_resource_limit_error(self) -> None:
        """Test ChunkingResourceLimitError with different resource types."""
        error = ChunkingResourceLimitError(
            detail="Too many connections",
            correlation_id="corr-123",
            resource_type=ResourceType.CONNECTIONS,
            current_usage=150,
            limit=100,
            operation_id="op-456",
        )

        assert error.resource_type == ResourceType.CONNECTIONS
        assert error.current_usage == 150
        assert error.limit == 100
        assert error.error_code == "CHUNKING_RESOURCE_LIMIT"

    def test_chunking_resource_limit_error_serialization(self) -> None:
        """Test ChunkingResourceLimitError serialization."""
        error = ChunkingResourceLimitError(
            detail="CPU limit exceeded",
            correlation_id="corr-123",
            resource_type=ResourceType.CPU,
            current_usage=95.5,
            limit=80.0,
        )

        result = error.to_dict()

        assert result["resource_type"] == "cpu"
        assert result["current_usage"] == 95.5
        assert result["limit"] == 80.0
        assert result["recovery_hint"] == "Wait for other operations to complete or increase cpu limit"

    def test_chunking_partial_failure_error(self) -> None:
        """Test ChunkingPartialFailureError with multiple failures."""
        failed_docs = ["doc1", "doc2", "doc3"]
        failure_reasons = {
            "doc1": "Invalid format",
            "doc2": "Memory error",
            "doc3": "Timeout",
        }

        error = ChunkingPartialFailureError(
            detail="3 of 10 documents failed",
            correlation_id="corr-123",
            operation_id="op-456",
            total_documents=10,
            failed_documents=failed_docs,
            failure_reasons=failure_reasons,
            successful_chunks=50,
        )

        assert error.total_documents == 10
        assert error.failed_documents == failed_docs
        assert error.failure_reasons == failure_reasons
        assert error.successful_chunks == 50
        assert error.error_code == "CHUNKING_PARTIAL_FAILURE"

    def test_chunking_partial_failure_error_serialization(self) -> None:
        """Test ChunkingPartialFailureError serialization with truncation."""
        # Create more than 10 failed documents
        failed_docs = [f"doc{i}" for i in range(15)]
        failure_reasons = {f"doc{i}": f"Error {i}" for i in range(15)}

        error = ChunkingPartialFailureError(
            detail="15 of 20 documents failed",
            correlation_id="corr-123",
            operation_id="op-456",
            total_documents=20,
            failed_documents=failed_docs,
            failure_reasons=failure_reasons,
            successful_chunks=25,
        )

        result = error.to_dict()

        assert result["total_documents"] == 20
        assert result["failed_count"] == 15
        assert result["success_count"] == 5
        assert len(result["failed_documents"]) == 10  # Truncated
        assert len(result["failure_reasons"]) == 10  # Truncated
        assert result["successful_chunks"] == 25
        assert result["recovery_hint"] == "Retry processing for failed documents only"

    def test_chunking_configuration_error(self) -> None:
        """Test ChunkingConfigurationError with config errors."""
        config_errors = [
            "chunk_size must be less than chunk_overlap",
            "strategy 'invalid' is not supported",
            "max_tokens is required for this strategy",
        ]

        error = ChunkingConfigurationError(
            detail="Invalid configuration",
            correlation_id="corr-123",
            config_errors=config_errors,
            operation_id="op-456",
        )

        assert error.config_errors == config_errors
        assert error.error_code == "CHUNKING_CONFIG_ERROR"

    def test_chunking_configuration_error_serialization(self) -> None:
        """Test ChunkingConfigurationError serialization."""
        config_errors = ["Error 1", "Error 2"]

        error = ChunkingConfigurationError(
            detail="Configuration invalid",
            correlation_id="corr-123",
            config_errors=config_errors,
        )

        result = error.to_dict()

        assert result["config_errors"] == config_errors

    def test_chunking_dependency_error(self) -> None:
        """Test ChunkingDependencyError with dependency details."""
        error = ChunkingDependencyError(
            detail="Embedding service unavailable",
            correlation_id="corr-123",
            dependency="embedding_service",
            dependency_error="Connection timeout",
            operation_id="op-456",
        )

        assert error.dependency == "embedding_service"
        assert error.dependency_error == "Connection timeout"
        assert error.error_code == "CHUNKING_DEPENDENCY_FAILED"

    def test_chunking_dependency_error_serialization(self) -> None:
        """Test ChunkingDependencyError serialization."""
        error = ChunkingDependencyError(
            detail="Storage service error",
            correlation_id="corr-123",
            dependency="qdrant",
            dependency_error="Connection refused",
        )

        result = error.to_dict()

        assert result["dependency"] == "qdrant"
        assert result["dependency_error"] == "Connection refused"
        assert result["recovery_hint"] == "Check qdrant service status and retry"

    def test_exception_inheritance(self) -> None:
        """Test that all exceptions inherit from ChunkingError."""
        exceptions = [
            ChunkingMemoryError(
                detail="Test",
                correlation_id="123",
                operation_id="456",
                memory_used=100,
                memory_limit=50,
            ),
            ChunkingTimeoutError(
                detail="Test",
                correlation_id="123",
                operation_id="456",
                elapsed_time=10,
                timeout_limit=5,
            ),
            ChunkingValidationError(
                detail="Test",
                correlation_id="123",
            ),
            ChunkingStrategyError(
                detail="Test",
                correlation_id="123",
                strategy="test",
            ),
            ChunkingResourceLimitError(
                detail="Test",
                correlation_id="123",
                resource_type=ResourceType.CPU,
                current_usage=10,
                limit=5,
            ),
            ChunkingPartialFailureError(
                detail="Test",
                correlation_id="123",
                operation_id="456",
                total_documents=10,
                failed_documents=[],
                failure_reasons={},
            ),
            ChunkingConfigurationError(
                detail="Test",
                correlation_id="123",
                config_errors=[],
            ),
            ChunkingDependencyError(
                detail="Test",
                correlation_id="123",
                dependency="test",
            ),
        ]

        for exc in exceptions:
            assert isinstance(exc, ChunkingError)
            assert isinstance(exc, Exception)

    def test_exception_str_representation(self) -> None:
        """Test string representation of exceptions."""
        exceptions = [
            (
                ChunkingMemoryError(
                    detail="Memory error details",
                    correlation_id="123",
                    operation_id="456",
                    memory_used=100,
                    memory_limit=50,
                ),
                "Memory error details",
            ),
            (
                ChunkingTimeoutError(
                    detail="Timeout error details",
                    correlation_id="123",
                    operation_id="456",
                    elapsed_time=10,
                    timeout_limit=5,
                ),
                "Timeout error details",
            ),
        ]

        for exc, expected_str in exceptions:
            assert str(exc) == expected_str

    def test_all_exceptions_have_to_dict(self) -> None:
        """Test that all exception classes implement to_dict."""
        exceptions = [
            ChunkingError("Test", "123"),
            ChunkingMemoryError("Test", "123", "456", 100, 50),
            ChunkingTimeoutError("Test", "123", "456", 10, 5),
            ChunkingValidationError("Test", "123"),
            ChunkingStrategyError("Test", "123", "test"),
            ChunkingResourceLimitError("Test", "123", ResourceType.CPU, 10, 5),
            ChunkingPartialFailureError("Test", "123", "456", 10, [], {}),
            ChunkingConfigurationError("Test", "123", []),
            ChunkingDependencyError("Test", "123", "test"),
        ]

        for exc in exceptions:
            result = exc.to_dict()
            assert isinstance(result, dict)
            assert "error_code" in result
            assert "detail" in result
            assert "correlation_id" in result
            assert "type" in result