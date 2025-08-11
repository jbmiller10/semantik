#!/usr/bin/env python3
"""
Infrastructure exception hierarchy for chunking operations.

Provides comprehensive exception types with context preservation,
correlation tracking, and exception chaining for debugging.
"""

import traceback
import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class ResourceType(Enum):
    """Types of resources that can be exhausted."""

    MEMORY = "memory"
    CPU = "cpu"
    DISK = "disk"
    CONNECTIONS = "connections"
    THREADS = "threads"


class BaseChunkingError(Exception):
    """Base exception with context preservation for chunking operations."""

    def __init__(
        self,
        message: str,
        code: str,
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize base exception with full context.

        Args:
            message: Human-readable error description
            code: Machine-readable error code
            details: Additional context information
            correlation_id: Request correlation ID for tracing
            cause: Original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.cause = cause
        self.timestamp = datetime.now(UTC)
        # Only capture stack trace if there's an actual cause (optimization)
        self.stack_trace = traceback.format_exc() if cause else None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/API responses."""
        return {
            "error": {
                "message": self.message,
                "code": self.code,
                "correlation_id": self.correlation_id,
                "timestamp": self.timestamp.isoformat(),
                "details": self.details,
                "cause": str(self.cause) if self.cause else None,
            }
        }


# Domain Layer Exceptions
class DomainError(BaseChunkingError):
    """Base for all domain exceptions."""


class DocumentTooLargeError(DomainError):
    """Raised when document exceeds size limits."""

    def __init__(
        self,
        size: int,
        max_size: int,
        correlation_id: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize document too large error.

        Args:
            size: Actual document size in bytes
            max_size: Maximum allowed size in bytes
            correlation_id: Request correlation ID
            cause: Original exception
        """
        super().__init__(
            message=f"Document size {size} exceeds maximum {max_size}",
            code="DOCUMENT_TOO_LARGE",
            details={"size": size, "max_size": max_size},
            correlation_id=correlation_id,
            cause=cause,
        )
        self.size = size
        self.max_size = max_size


class InvalidStateTransitionError(DomainError):
    """Raised when invalid state transition is attempted."""

    def __init__(
        self,
        current_state: str,
        attempted_state: str,
        correlation_id: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize invalid state transition error.

        Args:
            current_state: Current state of the entity
            attempted_state: State transition that was attempted
            correlation_id: Request correlation ID
            cause: Original exception
        """
        super().__init__(
            message=f"Cannot transition from {current_state} to {attempted_state}",
            code="INVALID_STATE_TRANSITION",
            details={"current": current_state, "attempted": attempted_state},
            correlation_id=correlation_id,
            cause=cause,
        )
        self.current_state = current_state
        self.attempted_state = attempted_state


class ChunkingStrategyError(DomainError):
    """Raised when chunking strategy fails."""

    def __init__(
        self,
        strategy: str,
        reason: str,
        correlation_id: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize chunking strategy error.

        Args:
            strategy: Name of the strategy that failed
            reason: Reason for the failure
            correlation_id: Request correlation ID
            cause: Original exception
        """
        super().__init__(
            message=f"Strategy {strategy} failed: {reason}",
            code="CHUNKING_STRATEGY_ERROR",
            details={"strategy": strategy, "reason": reason},
            correlation_id=correlation_id,
            cause=cause,
        )
        self.strategy = strategy
        self.reason = reason


# Application Layer Exceptions
class ApplicationError(BaseChunkingError):
    """Base for application layer exceptions."""


class ValidationError(ApplicationError):
    """Raised when validation fails."""

    def __init__(
        self,
        field: str,
        value: Any,
        reason: str,
        correlation_id: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize validation exception.

        Args:
            field: Field that failed validation
            value: Value that was invalid
            reason: Reason for validation failure
            correlation_id: Request correlation ID
            cause: Original exception
        """
        super().__init__(
            message=f"Validation failed for {field}: {reason}",
            code="VALIDATION_ERROR",
            details={"field": field, "value": str(value), "reason": reason},
            correlation_id=correlation_id,
            cause=cause,
        )
        self.field = field
        self.value = value
        self.reason = reason


class ResourceNotFoundError(ApplicationError):
    """Raised when requested resource is not found."""

    def __init__(
        self,
        resource_type: str,
        resource_id: str,
        correlation_id: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize resource not found exception.

        Args:
            resource_type: Type of resource (e.g., 'Document', 'Collection')
            resource_id: ID of the resource that was not found
            correlation_id: Request correlation ID
            cause: Original exception
        """
        super().__init__(
            message=f"{resource_type} not found: {resource_id}",
            code="RESOURCE_NOT_FOUND",
            details={"resource_type": resource_type, "resource_id": resource_id},
            correlation_id=correlation_id,
            cause=cause,
        )
        self.resource_type = resource_type
        self.resource_id = resource_id


class PermissionDeniedError(ApplicationError):
    """Raised when user lacks permission for operation."""

    def __init__(
        self,
        user_id: str,
        resource: str,
        action: str,
        correlation_id: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize permission denied exception.

        Args:
            user_id: ID of the user
            resource: Resource being accessed
            action: Action that was denied
            correlation_id: Request correlation ID
            cause: Original exception
        """
        super().__init__(
            message=f"User {user_id} denied {action} on {resource}",
            code="PERMISSION_DENIED",
            details={"user_id": user_id, "resource": resource, "action": action},
            correlation_id=correlation_id,
            cause=cause,
        )
        self.user_id = user_id
        self.resource = resource
        self.action = action


# Infrastructure Layer Exceptions
class InfrastructureError(BaseChunkingError):
    """Base for infrastructure exceptions."""


class DatabaseError(InfrastructureError):
    """Raised when database operation fails."""

    def __init__(
        self,
        operation: str,
        table: str,
        error: str,
        correlation_id: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize database exception.

        Args:
            operation: Database operation that failed
            table: Table involved in the operation
            error: Error message from database
            correlation_id: Request correlation ID
            cause: Original exception
        """
        super().__init__(
            message=f"Database error during {operation} on {table}: {error}",
            code="DATABASE_ERROR",
            details={"operation": operation, "table": table, "error": error},
            correlation_id=correlation_id,
            cause=cause,
        )
        self.operation = operation
        self.table = table
        self.error = error


class ExternalServiceError(InfrastructureError):
    """Raised when external service fails."""

    def __init__(
        self,
        service: str,
        operation: str,
        error: str,
        correlation_id: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize external service exception.

        Args:
            service: Name of the external service
            operation: Operation being performed
            error: Error message from service
            correlation_id: Request correlation ID
            cause: Original exception
        """
        super().__init__(
            message=f"External service {service} failed during {operation}: {error}",
            code="EXTERNAL_SERVICE_ERROR",
            details={"service": service, "operation": operation, "error": error},
            correlation_id=correlation_id,
            cause=cause,
        )
        self.service = service
        self.operation = operation
        self.error = error


class StreamingError(InfrastructureError):
    """Raised when streaming operation fails."""

    def __init__(
        self,
        message: str,
        processor_state: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize streaming exception.

        Args:
            message: Error message
            processor_state: Current state of the streaming processor
            correlation_id: Request correlation ID
            cause: Original exception
        """
        super().__init__(
            message=message,
            code="STREAMING_ERROR",
            details={"processor_state": processor_state or {}},
            correlation_id=correlation_id,
            cause=cause,
        )
        self.processor_state = processor_state

# ---------------------------------------------------------------------------
# Backwards-compatibility aliases for tests and external callers expecting
# "Exception"-suffixed class names rather than "Error"-suffixed ones.
# ---------------------------------------------------------------------------

# Base classes
BaseChunkingException = BaseChunkingError
DomainException = DomainError
ApplicationException = ApplicationError
InfrastructureException = InfrastructureError

# Domain-level
DocumentTooLargeException = DocumentTooLargeError
InvalidStateTransition = InvalidStateTransitionError
ChunkingStrategyException = ChunkingStrategyError

# Application-level
ValidationException = ValidationError
ResourceNotFoundException = ResourceNotFoundError
PermissionDeniedException = PermissionDeniedError

# Infrastructure-level
DatabaseException = DatabaseError
ExternalServiceException = ExternalServiceError
StreamingException = StreamingError
