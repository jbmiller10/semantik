#!/usr/bin/env python3

"""
Custom exceptions for chunking operations.

This module defines a comprehensive exception hierarchy for better error handling,
debugging, and recovery in the chunking pipeline.
"""

from enum import Enum
from typing import Any


class ResourceType(str, Enum):
    """Types of resources that can be exhausted."""

    MEMORY = "memory"
    CPU = "cpu"
    DISK = "disk"
    CONNECTIONS = "connections"
    THREADS = "threads"


class ChunkingError(Exception):
    """Base exception for all chunking-related errors."""

    def __init__(
        self,
        detail: str,
        correlation_id: str | None = None,
        operation_id: str | None = None,
        error_code: str | None = None,
    ) -> None:
        """Initialize base chunking error.

        Args:
            detail: Error description
            correlation_id: Request correlation ID for tracing
            operation_id: Operation ID for tracking
            error_code: Specific error code for categorization
        """
        super().__init__(detail)
        self.detail = detail
        self.correlation_id = correlation_id
        self.operation_id = operation_id
        self.error_code = error_code or self.__class__.__name__

    def __str__(self) -> str:
        """Return string representation of the error."""
        return self.detail

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for serialization.

        Returns:
            Dictionary representation of the exception
        """
        result = {
            "error_code": self.error_code,
            "detail": self.detail,
            "correlation_id": self.correlation_id,
            "type": self.__class__.__name__,
        }
        if self.operation_id:
            result["operation_id"] = self.operation_id
        return result


class ChunkingMemoryError(ChunkingError):
    """Raised when memory limits are exceeded during chunking."""

    def __init__(
        self,
        detail: str,
        correlation_id: str | None = None,
        operation_id: str | None = None,
        memory_used: int | None = None,
        memory_limit: int | None = None,
        recovery_hint: str | None = None,
    ) -> None:
        """Initialize memory error.

        Args:
            detail: Error description
            correlation_id: Request correlation ID
            operation_id: Operation ID
            memory_used: Memory used in bytes
            memory_limit: Memory limit in bytes
            recovery_hint: Suggestion for recovery
        """
        super().__init__(detail, correlation_id, operation_id, "CHUNKING_MEMORY_EXCEEDED")
        self.memory_used = memory_used
        self.memory_limit = memory_limit
        self.recovery_hint = recovery_hint or "Try processing smaller documents or use a more memory-efficient strategy"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with memory details."""
        result = super().to_dict()
        if self.memory_used is not None:
            result["memory_used_mb"] = round(self.memory_used / (1024 * 1024), 2)
        if self.memory_limit is not None:
            result["memory_limit_mb"] = round(self.memory_limit / (1024 * 1024), 2)
        result["recovery_hint"] = self.recovery_hint
        return result


class ChunkingTimeoutError(ChunkingError):
    """Raised when chunking operation exceeds time limit."""

    def __init__(
        self,
        detail: str,
        correlation_id: str | None = None,
        operation_id: str | None = None,
        elapsed_time: float | None = None,
        timeout_limit: float | None = None,
        estimated_completion: float | None = None,
    ) -> None:
        """Initialize timeout error.

        Args:
            detail: Error description
            correlation_id: Request correlation ID
            operation_id: Operation ID
            elapsed_time: Time elapsed in seconds
            timeout_limit: Timeout limit in seconds
            estimated_completion: Estimated time to complete in seconds
        """
        super().__init__(detail, correlation_id, operation_id, "CHUNKING_TIMEOUT")
        self.elapsed_time = elapsed_time
        self.timeout_limit = timeout_limit
        self.estimated_completion = estimated_completion

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with timing details."""
        result = super().to_dict()
        if self.elapsed_time is not None:
            result["elapsed_seconds"] = round(self.elapsed_time, 2)
        if self.timeout_limit is not None:
            result["timeout_seconds"] = round(self.timeout_limit, 2)
        if self.estimated_completion is not None:
            result["estimated_completion_seconds"] = round(self.estimated_completion, 2)
        else:
            result["estimated_completion_seconds"] = None
        result["recovery_hint"] = "Consider using a faster strategy or processing in smaller batches"
        return result


class ChunkingValidationError(ChunkingError):
    """Raised when input validation fails."""

    def __init__(
        self,
        detail: str,
        correlation_id: str | None = None,
        field_errors: dict[str, list[str]] | None = None,
        operation_id: str | None = None,
    ) -> None:
        """Initialize validation error.

        Args:
            detail: Error description
            correlation_id: Request correlation ID
            field_errors: Field-specific validation errors
            operation_id: Operation ID
        """
        super().__init__(detail, correlation_id, operation_id, "CHUNKING_VALIDATION_FAILED")
        self.field_errors = field_errors or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with field errors."""
        result = super().to_dict()
        result["field_errors"] = self.field_errors
        return result


class ChunkingStrategyError(ChunkingError):
    """Raised when a chunking strategy fails or is unavailable."""

    def __init__(
        self,
        detail: str,
        correlation_id: str | None = None,
        strategy: str | None = None,
        fallback_strategy: str | None = None,
        operation_id: str | None = None,
    ) -> None:
        """Initialize strategy error.

        Args:
            detail: Error description
            correlation_id: Request correlation ID
            strategy: Failed strategy name
            fallback_strategy: Suggested fallback strategy
            operation_id: Operation ID
        """
        super().__init__(detail, correlation_id, operation_id, "CHUNKING_STRATEGY_FAILED")
        self.strategy = strategy
        self.fallback_strategy = fallback_strategy

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with strategy details."""
        result = super().to_dict()
        result["strategy"] = self.strategy
        result["fallback_strategy"] = self.fallback_strategy
        if self.fallback_strategy:
            result["recovery_hint"] = f"Try using {self.fallback_strategy} strategy instead"
        else:
            result["recovery_hint"] = None
        return result


class ChunkingResourceLimitError(ChunkingError):
    """Raised when resource limits are exceeded."""

    def __init__(
        self,
        detail: str,
        correlation_id: str | None = None,
        resource_type: ResourceType | None = None,
        current_usage: float | None = None,
        limit: float | None = None,
        operation_id: str | None = None,
    ) -> None:
        """Initialize resource limit error.

        Args:
            detail: Error description
            correlation_id: Request correlation ID
            resource_type: Type of resource exhausted
            current_usage: Current usage value
            limit: Resource limit value
            operation_id: Operation ID
        """
        super().__init__(detail, correlation_id, operation_id, "CHUNKING_RESOURCE_LIMIT")
        self.resource_type = resource_type
        self.current_usage = current_usage
        self.limit = limit

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with resource details."""
        result = super().to_dict()
        if self.resource_type:
            result["resource_type"] = self.resource_type.value
        result["current_usage"] = self.current_usage
        result["limit"] = self.limit
        if self.resource_type:
            result[
                "recovery_hint"
            ] = f"Wait for other operations to complete or increase {self.resource_type.value} limit"
        return result


class ChunkingPartialFailureError(ChunkingError):
    """Raised when some documents in a batch fail to process."""

    def __init__(
        self,
        detail: str,
        correlation_id: str | None = None,
        operation_id: str | None = None,
        total_documents: int = 0,
        failed_documents: list[str] | None = None,
        failure_reasons: dict[str, str] | None = None,
        successful_chunks: int = 0,
    ) -> None:
        """Initialize partial failure error.

        Args:
            detail: Error description
            correlation_id: Request correlation ID
            operation_id: Operation ID
            total_documents: Total number of documents
            failed_documents: List of failed document IDs
            failure_reasons: Mapping of document ID to failure reason
            successful_chunks: Number of successfully created chunks
        """
        super().__init__(detail, correlation_id, operation_id, "CHUNKING_PARTIAL_FAILURE")
        self.total_documents = total_documents
        self.failed_documents = failed_documents or []
        self.failure_reasons = failure_reasons or {}
        self.successful_chunks = successful_chunks

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with failure details."""
        result = super().to_dict()
        result["total_documents"] = self.total_documents
        result["failed_count"] = len(self.failed_documents)
        result["success_count"] = self.total_documents - len(self.failed_documents)
        # Limit the number of failed documents shown to prevent huge responses
        result["failed_documents"] = self.failed_documents[:10]
        result["failure_reasons"] = dict(list(self.failure_reasons.items())[:10])
        result["successful_chunks"] = self.successful_chunks
        result["recovery_hint"] = "Retry processing for failed documents only"
        return result


class ChunkingConfigurationError(ChunkingError):
    """Raised when chunking configuration is invalid."""

    def __init__(
        self,
        detail: str,
        correlation_id: str | None = None,
        config_errors: list[str] | None = None,
        operation_id: str | None = None,
    ) -> None:
        """Initialize configuration error.

        Args:
            detail: Error description
            correlation_id: Request correlation ID
            config_errors: List of configuration errors
            operation_id: Operation ID
        """
        super().__init__(detail, correlation_id, operation_id, "CHUNKING_CONFIG_ERROR")
        self.config_errors = config_errors or []

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with config errors."""
        result = super().to_dict()
        result["config_errors"] = self.config_errors
        return result


class ChunkingDependencyError(ChunkingError):
    """Raised when a required dependency is unavailable."""

    def __init__(
        self,
        detail: str,
        correlation_id: str | None = None,
        dependency: str | None = None,
        dependency_error: str | None = None,
        operation_id: str | None = None,
    ) -> None:
        """Initialize dependency error.

        Args:
            detail: Error description
            correlation_id: Request correlation ID
            dependency: Name of the failed dependency
            dependency_error: Error from the dependency
            operation_id: Operation ID
        """
        super().__init__(detail, correlation_id, operation_id, "CHUNKING_DEPENDENCY_FAILED")
        self.dependency = dependency
        self.dependency_error = dependency_error

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with dependency details."""
        result = super().to_dict()
        result["dependency"] = self.dependency
        result["dependency_error"] = self.dependency_error
        if self.dependency:
            result["recovery_hint"] = f"Check {self.dependency} service status and retry"
        return result
