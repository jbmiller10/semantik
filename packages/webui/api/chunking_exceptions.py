#!/usr/bin/env python3
"""
Custom exceptions for chunking operations.

This module defines a hierarchy of exceptions for the chunking system,
providing structured error information with correlation IDs for tracing
and recovery hints for better error handling.
"""

from enum import Enum
from typing import Any


class ResourceType(Enum):
    """Types of resources that can be exhausted."""

    MEMORY = "memory"
    CPU = "cpu"
    DISK = "disk"
    CONNECTIONS = "connections"
    THREADS = "threads"


class ChunkingError(Exception):
    """Base exception for all chunking-related errors.

    Attributes:
        detail: Human-readable error description
        correlation_id: UUID for tracing the error across services
        operation_id: Optional operation identifier
        error_code: Machine-readable error code for client handling
    """

    def __init__(
        self,
        detail: str,
        correlation_id: str,
        operation_id: str | None = None,
        error_code: str | None = None,
    ) -> None:
        """Initialize the chunking error.

        Args:
            detail: Human-readable error description
            correlation_id: UUID for tracing
            operation_id: Optional operation identifier
            error_code: Optional machine-readable error code
        """
        self.detail = detail
        self.correlation_id = correlation_id
        self.operation_id = operation_id
        self.error_code = error_code or self.__class__.__name__
        super().__init__(detail)

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for JSON serialization."""
        return {
            "error_code": self.error_code,
            "detail": self.detail,
            "correlation_id": self.correlation_id,
            "operation_id": self.operation_id,
            "type": self.__class__.__name__,
        }


class ChunkingMemoryError(ChunkingError):
    """Raised when a chunking operation exceeds memory limits.

    This error indicates that the operation consumed more memory than allowed,
    which could be due to large documents, inefficient strategies, or system
    resource constraints.
    """

    def __init__(
        self,
        detail: str,
        correlation_id: str,
        operation_id: str,
        memory_used: int,
        memory_limit: int,
        recovery_hint: str | None = None,
    ) -> None:
        """Initialize memory error with resource details.

        Args:
            detail: Error description
            correlation_id: Tracing ID
            operation_id: Operation that failed
            memory_used: Bytes of memory used
            memory_limit: Byte limit that was exceeded
            recovery_hint: Suggestion for recovery
        """
        super().__init__(detail, correlation_id, operation_id, "CHUNKING_MEMORY_EXCEEDED")
        self.memory_used = memory_used
        self.memory_limit = memory_limit
        self.recovery_hint = recovery_hint or "Try processing smaller documents or use a more memory-efficient strategy"

    def to_dict(self) -> dict[str, Any]:
        """Include memory details in serialization."""
        data = super().to_dict()
        data.update({
            "memory_used_mb": round(self.memory_used / (1024 * 1024), 2),
            "memory_limit_mb": round(self.memory_limit / (1024 * 1024), 2),
            "recovery_hint": self.recovery_hint,
        })
        return data


class ChunkingTimeoutError(ChunkingError):
    """Raised when a chunking operation exceeds time limits.

    This error occurs when processing takes longer than the configured timeout,
    which might indicate complex documents, slow strategies, or system issues.
    """

    def __init__(
        self,
        detail: str,
        correlation_id: str,
        operation_id: str,
        elapsed_time: float,
        timeout_limit: float,
        estimated_completion: float | None = None,
    ) -> None:
        """Initialize timeout error with timing details.

        Args:
            detail: Error description
            correlation_id: Tracing ID
            operation_id: Operation that timed out
            elapsed_time: Seconds elapsed before timeout
            timeout_limit: Timeout limit in seconds
            estimated_completion: Estimated seconds to complete (if known)
        """
        super().__init__(detail, correlation_id, operation_id, "CHUNKING_TIMEOUT")
        self.elapsed_time = elapsed_time
        self.timeout_limit = timeout_limit
        self.estimated_completion = estimated_completion

    def to_dict(self) -> dict[str, Any]:
        """Include timing details in serialization."""
        data = super().to_dict()
        data.update({
            "elapsed_seconds": round(self.elapsed_time, 2),
            "timeout_seconds": round(self.timeout_limit, 2),
            "estimated_completion_seconds": round(self.estimated_completion, 2) if self.estimated_completion else None,
            "recovery_hint": "Consider using a faster strategy or processing in smaller batches",
        })
        return data


class ChunkingValidationError(ChunkingError):
    """Raised when input validation fails for chunking operations.

    This error indicates invalid parameters, malformed documents, or
    configuration issues that prevent processing.
    """

    def __init__(
        self,
        detail: str,
        correlation_id: str,
        field_errors: dict[str, list[str]] | None = None,
        operation_id: str | None = None,
    ) -> None:
        """Initialize validation error with field details.

        Args:
            detail: Error description
            correlation_id: Tracing ID
            field_errors: Dictionary of field names to error messages
            operation_id: Optional operation identifier
        """
        super().__init__(detail, correlation_id, operation_id, "CHUNKING_VALIDATION_FAILED")
        self.field_errors = field_errors or {}

    def to_dict(self) -> dict[str, Any]:
        """Include validation details in serialization."""
        data = super().to_dict()
        data["field_errors"] = self.field_errors
        return data


class ChunkingStrategyError(ChunkingError):
    """Raised when a chunking strategy fails or is unavailable.

    This error occurs when the requested strategy cannot be initialized,
    encounters an error during processing, or is incompatible with the input.
    """

    def __init__(
        self,
        detail: str,
        correlation_id: str,
        strategy: str,
        fallback_strategy: str | None = None,
        operation_id: str | None = None,
    ) -> None:
        """Initialize strategy error with strategy details.

        Args:
            detail: Error description
            correlation_id: Tracing ID
            strategy: Strategy that failed
            fallback_strategy: Suggested alternative strategy
            operation_id: Optional operation identifier
        """
        super().__init__(detail, correlation_id, operation_id, "CHUNKING_STRATEGY_FAILED")
        self.strategy = strategy
        self.fallback_strategy = fallback_strategy

    def to_dict(self) -> dict[str, Any]:
        """Include strategy details in serialization."""
        data = super().to_dict()
        data.update({
            "strategy": self.strategy,
            "fallback_strategy": self.fallback_strategy,
            "recovery_hint": f"Try using {self.fallback_strategy} strategy instead" if self.fallback_strategy else None,
        })
        return data


class ChunkingResourceLimitError(ChunkingError):
    """Raised when a resource limit is exceeded during chunking.

    This is a general resource exhaustion error for limits other than
    memory and time (e.g., concurrent operations, file handles).
    """

    def __init__(
        self,
        detail: str,
        correlation_id: str,
        resource_type: ResourceType,
        current_usage: int | float,
        limit: int | float,
        operation_id: str | None = None,
    ) -> None:
        """Initialize resource limit error.

        Args:
            detail: Error description
            correlation_id: Tracing ID
            resource_type: Type of resource exhausted
            current_usage: Current resource usage
            limit: Resource limit that was exceeded
            operation_id: Optional operation identifier
        """
        super().__init__(detail, correlation_id, operation_id, "CHUNKING_RESOURCE_LIMIT")
        self.resource_type = resource_type
        self.current_usage = current_usage
        self.limit = limit

    def to_dict(self) -> dict[str, Any]:
        """Include resource details in serialization."""
        data = super().to_dict()
        data.update({
            "resource_type": self.resource_type.value,
            "current_usage": self.current_usage,
            "limit": self.limit,
            "recovery_hint": f"Wait for other operations to complete or increase {self.resource_type.value} limit",
        })
        return data


class ChunkingPartialFailureError(ChunkingError):
    """Raised when some documents in a batch fail while others succeed.

    This error provides details about which documents failed and why,
    allowing for partial recovery or retry of failed items only.
    """

    def __init__(
        self,
        detail: str,
        correlation_id: str,
        operation_id: str,
        total_documents: int,
        failed_documents: list[str],
        failure_reasons: dict[str, str],
        successful_chunks: int = 0,
    ) -> None:
        """Initialize partial failure error.

        Args:
            detail: Error description
            correlation_id: Tracing ID
            operation_id: Operation with partial failure
            total_documents: Total number of documents processed
            failed_documents: List of document IDs that failed
            failure_reasons: Map of document ID to failure reason
            successful_chunks: Number of successfully created chunks
        """
        super().__init__(detail, correlation_id, operation_id, "CHUNKING_PARTIAL_FAILURE")
        self.total_documents = total_documents
        self.failed_documents = failed_documents
        self.failure_reasons = failure_reasons
        self.successful_chunks = successful_chunks

    def to_dict(self) -> dict[str, Any]:
        """Include failure details in serialization."""
        data = super().to_dict()
        data.update({
            "total_documents": self.total_documents,
            "failed_count": len(self.failed_documents),
            "success_count": self.total_documents - len(self.failed_documents),
            "failed_documents": self.failed_documents[:10],  # Limit to first 10
            "failure_reasons": dict(list(self.failure_reasons.items())[:10]),  # Limit to first 10
            "successful_chunks": self.successful_chunks,
            "recovery_hint": "Retry processing for failed documents only",
        })
        return data


class ChunkingConfigurationError(ChunkingError):
    """Raised when chunking configuration is invalid or incompatible.

    This error indicates issues with strategy parameters, missing dependencies,
    or incompatible configuration combinations.
    """

    def __init__(
        self,
        detail: str,
        correlation_id: str,
        config_errors: list[str],
        operation_id: str | None = None,
    ) -> None:
        """Initialize configuration error.

        Args:
            detail: Error description
            correlation_id: Tracing ID
            config_errors: List of specific configuration issues
            operation_id: Optional operation identifier
        """
        super().__init__(detail, correlation_id, operation_id, "CHUNKING_CONFIG_ERROR")
        self.config_errors = config_errors

    def to_dict(self) -> dict[str, Any]:
        """Include configuration errors in serialization."""
        data = super().to_dict()
        data["config_errors"] = self.config_errors
        return data


class ChunkingDependencyError(ChunkingError):
    """Raised when an external dependency fails during chunking.

    This error occurs when external services (embeddings, storage) are
    unavailable or return errors during the chunking process.
    """

    def __init__(
        self,
        detail: str,
        correlation_id: str,
        dependency: str,
        dependency_error: str | None = None,
        operation_id: str | None = None,
    ) -> None:
        """Initialize dependency error.

        Args:
            detail: Error description
            correlation_id: Tracing ID
            dependency: Name of the failed dependency
            dependency_error: Error message from the dependency
            operation_id: Optional operation identifier
        """
        super().__init__(detail, correlation_id, operation_id, "CHUNKING_DEPENDENCY_FAILED")
        self.dependency = dependency
        self.dependency_error = dependency_error

    def to_dict(self) -> dict[str, Any]:
        """Include dependency details in serialization."""
        data = super().to_dict()
        data.update({
            "dependency": self.dependency,
            "dependency_error": self.dependency_error,
            "recovery_hint": f"Check {self.dependency} service status and retry",
        })
        return data