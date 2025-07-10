"""Standard error response contracts."""

from typing import Any

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    """Detailed error information."""

    field: str | None = Field(None, description="Field that caused the error")
    message: str = Field(description="Detailed error message")
    code: str | None = Field(None, description="Error code")


class ErrorResponse(BaseModel):
    """Standard error response for all services."""

    error: str = Field(description="Error type or category")
    message: str = Field(description="Human-readable error message")
    details: dict[str, Any] | list[ErrorDetail] | None = Field(None, description="Additional error details")
    request_id: str | None = Field(None, description="Request ID for tracking")
    timestamp: str | None = Field(None, description="Error timestamp")
    status_code: int | None = Field(None, description="HTTP status code")

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "error": "ValidationError",
                    "message": "Invalid request parameters",
                    "details": {"field": "query", "message": "Field required"},
                    "status_code": 400,
                },
                {
                    "error": "NotFoundError",
                    "message": "Collection not found",
                    "details": {"collection": "job_123", "suggestion": "Check if the job has completed"},
                    "status_code": 404,
                },
                {
                    "error": "InsufficientMemoryError",
                    "message": "Insufficient GPU memory for operation",
                    "details": {
                        "required_memory": "4GB",
                        "available_memory": "2GB",
                        "suggestion": "Try using a smaller model or different quantization",
                    },
                    "status_code": 507,
                },
            ]
        }


class ValidationErrorResponse(ErrorResponse):
    """Validation error response with field-specific errors."""

    error: str = Field(default="ValidationError")
    details: list[ErrorDetail] = Field(description="List of validation errors")


class AuthenticationErrorResponse(ErrorResponse):
    """Authentication error response."""

    error: str = Field(default="AuthenticationError")
    message: str = Field(default="Authentication failed")


class AuthorizationErrorResponse(ErrorResponse):
    """Authorization error response."""

    error: str = Field(default="AuthorizationError")
    message: str = Field(default="Access denied")


class NotFoundErrorResponse(ErrorResponse):
    """Resource not found error response."""

    error: str = Field(default="NotFoundError")
    resource_type: str | None = Field(None, description="Type of resource not found")
    resource_id: str | None = Field(None, description="ID of resource not found")


class InsufficientResourcesError(ErrorResponse):
    """Insufficient resources error (memory, disk, etc.)."""

    error: str = Field(default="InsufficientResourcesError")
    resource_type: str = Field(description="Type of resource (memory, disk, gpu)")
    required: str | None = Field(None, description="Required amount")
    available: str | None = Field(None, description="Available amount")
    suggestion: str | None = Field(None, description="Suggestion to resolve the issue")


class ServiceUnavailableError(ErrorResponse):
    """Service unavailable error."""

    error: str = Field(default="ServiceUnavailableError")
    service: str | None = Field(None, description="Name of unavailable service")
    retry_after: int | None = Field(None, description="Seconds to wait before retry")


class RateLimitError(ErrorResponse):
    """Rate limit exceeded error."""

    error: str = Field(default="RateLimitError")
    limit: int | None = Field(None, description="Rate limit")
    window: str | None = Field(None, description="Time window (e.g., '1 hour')")
    retry_after: int | None = Field(None, description="Seconds to wait before retry")


# Helper functions for creating error responses


def create_validation_error(errors: list[tuple[str, str]]) -> ValidationErrorResponse:
    """Create a validation error response from a list of field errors."""
    details = [ErrorDetail(field=field, message=message) for field, message in errors]
    return ValidationErrorResponse(
        error="ValidationError", message="Validation failed", details=details, status_code=400
    )


def create_not_found_error(resource_type: str, resource_id: str) -> NotFoundErrorResponse:
    """Create a not found error response."""
    return NotFoundErrorResponse(
        error="NotFoundError",
        message=f"{resource_type} not found",
        resource_type=resource_type,
        resource_id=resource_id,
        status_code=404,
    )


def create_insufficient_memory_error(required: str, available: str, suggestion: str) -> InsufficientResourcesError:
    """Create an insufficient memory error response."""
    return InsufficientResourcesError(
        error="InsufficientResourcesError",
        message="Insufficient GPU memory for operation",
        resource_type="gpu_memory",
        required=required,
        available=available,
        suggestion=suggestion,
        status_code=507,
    )
