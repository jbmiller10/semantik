"""Centralized error handling decorator for API endpoints.

This module provides a decorator that handles common service layer exceptions
and converts them to appropriate HTTP responses with consistent formatting,
logging, and security (sanitized error messages).

Usage:
    from webui.utils.service_error_handler import handle_service_errors

    @router.get("/{id}")
    @handle_service_errors
    async def get_item(id: str):
        return await service.get(id)

    # With customization:
    @router.post("/")
    @handle_service_errors(exclude_exceptions={CustomError})
    async def create_item():
        ...
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from fastapi import HTTPException

from shared.database.exceptions import (
    AccessDeniedError,
    EntityAlreadyExistsError,
    EntityNotFoundError,
    InvalidStateError,
    ValidationError,
)
from webui.middleware.correlation import get_correlation_id

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Awaitable[Any]])


@dataclass(frozen=True)
class ExceptionMapping:
    """Maps a service exception type to an HTTP response configuration.

    Attributes:
        exception_type: The exception class to handle.
        status_code: HTTP status code to return.
        detail_factory: Optional function to generate custom error detail.
                       If None, uses sanitized default message.
        log_level: Logging level ('warning', 'error', 'info').
        sanitize: If True, uses generic messages instead of exception str.
        include_entity_type: If True, includes entity type in error detail.
    """

    exception_type: type[Exception]
    status_code: int
    detail_factory: Callable[[Exception], str] | None = None
    log_level: str = "warning"
    sanitize: bool = True
    include_entity_type: bool = False


# Sanitized error messages for common exception types.
# These messages avoid leaking sensitive information like IDs, paths, or usernames.
_SANITIZED_MESSAGES: dict[str, str] = {
    "EntityNotFoundError": "The requested resource was not found",
    "EntityAlreadyExistsError": "A resource with this identifier already exists",
    "AccessDeniedError": "You do not have permission to perform this action",
    "ValidationError": "The request data is invalid",
    "InvalidStateError": "The operation cannot be performed in the current state",
    "ValueError": "Invalid value provided",
    "FileNotFoundError": "The requested file or directory was not found",
    "PermissionError": "Permission denied to access this resource",
}


def _entity_not_found_detail(exc: Exception) -> str:
    """Generate detail for EntityNotFoundError including entity type."""
    if hasattr(exc, "entity_type"):
        entity_type = exc.entity_type
        # Humanize the entity type (e.g., "Collection" -> "Collection")
        return f"{entity_type} not found"
    return _SANITIZED_MESSAGES["EntityNotFoundError"]


def _entity_exists_detail(exc: Exception) -> str:
    """Generate detail for EntityAlreadyExistsError including entity type."""
    if hasattr(exc, "entity_type"):
        entity_type = exc.entity_type
        return f"A {entity_type.lower()} with this identifier already exists"
    return _SANITIZED_MESSAGES["EntityAlreadyExistsError"]


def _validation_error_detail(exc: Exception) -> str:
    """Generate detail for ValidationError including field if available."""
    if hasattr(exc, "field") and exc.field:
        return f"Invalid value for field '{exc.field}'"
    if hasattr(exc, "message"):
        # Return the message but not the raw exception string
        return str(exc.message)
    return _SANITIZED_MESSAGES["ValidationError"]


def _invalid_state_detail(exc: Exception) -> str:
    """Generate detail for InvalidStateError with state info if available."""
    if hasattr(exc, "current_state") and hasattr(exc, "allowed_states"):
        current = exc.current_state
        allowed = exc.allowed_states
        if current and allowed:
            return f"Operation not allowed in state '{current}'. Allowed states: {', '.join(allowed)}"
    return _SANITIZED_MESSAGES["InvalidStateError"]


# Default exception mappings ordered from most specific to least specific.
# Order matters for isinstance matching.
DEFAULT_EXCEPTION_MAPPINGS: tuple[ExceptionMapping, ...] = (
    ExceptionMapping(
        exception_type=EntityNotFoundError,
        status_code=404,
        detail_factory=_entity_not_found_detail,
        log_level="warning",
        include_entity_type=True,
    ),
    ExceptionMapping(
        exception_type=EntityAlreadyExistsError,
        status_code=409,
        detail_factory=_entity_exists_detail,
        log_level="warning",
        include_entity_type=True,
    ),
    ExceptionMapping(
        exception_type=AccessDeniedError,
        status_code=403,
        log_level="warning",
        sanitize=True,
    ),
    ExceptionMapping(
        exception_type=ValidationError,
        status_code=400,
        detail_factory=_validation_error_detail,
        log_level="warning",
    ),
    ExceptionMapping(
        exception_type=InvalidStateError,
        status_code=409,
        detail_factory=_invalid_state_detail,
        log_level="warning",
    ),
    # OS-level exceptions
    ExceptionMapping(
        exception_type=FileNotFoundError,
        status_code=404,
        log_level="warning",
        sanitize=True,
    ),
    ExceptionMapping(
        exception_type=PermissionError,
        status_code=403,
        log_level="warning",
        sanitize=True,
    ),
    # Generic ValueError - often from invalid parameters
    ExceptionMapping(
        exception_type=ValueError,
        status_code=400,
        log_level="warning",
        sanitize=True,
    ),
)


def _find_mapping(
    exc: Exception,
    extra_mappings: tuple[ExceptionMapping, ...] | None = None,
) -> ExceptionMapping | None:
    """Find the first matching exception mapping for an exception.

    Checks extra_mappings first, then default mappings. Uses isinstance
    to support exception inheritance.

    Args:
        exc: The exception to find a mapping for.
        extra_mappings: Additional mappings to check before defaults.

    Returns:
        The matching ExceptionMapping, or None if no match found.
    """
    # Check extra mappings first (allows overriding defaults)
    if extra_mappings:
        for mapping in extra_mappings:
            if isinstance(exc, mapping.exception_type):
                return mapping

    # Check default mappings
    for mapping in DEFAULT_EXCEPTION_MAPPINGS:
        if isinstance(exc, mapping.exception_type):
            return mapping

    return None


def _build_detail(mapping: ExceptionMapping, exc: Exception) -> str:
    """Build the error detail string for an exception.

    Args:
        mapping: The exception mapping configuration.
        exc: The exception instance.

    Returns:
        The error detail string, sanitized if configured.
    """
    # Use custom detail factory if provided
    if mapping.detail_factory:
        return mapping.detail_factory(exc)

    # Use sanitized message if configured
    if mapping.sanitize:
        exc_name = type(exc).__name__
        return _SANITIZED_MESSAGES.get(exc_name, "An error occurred")

    # Fall back to exception string (not recommended for security)
    return str(exc)


def _log_exception(
    mapping: ExceptionMapping,
    exc: Exception,
    correlation_id: str | None,
) -> None:
    """Log an exception with appropriate level and context.

    Args:
        mapping: The exception mapping with log level.
        exc: The exception to log.
        correlation_id: Request correlation ID for tracing.
    """
    extra: dict[str, Any] = {
        "correlation_id": correlation_id,
        "exception_type": type(exc).__name__,
    }

    # Add entity info if available
    if hasattr(exc, "entity_type"):
        extra["entity_type"] = exc.entity_type
    if hasattr(exc, "entity_id"):
        extra["entity_id"] = exc.entity_id
    if hasattr(exc, "user_id"):
        extra["user_id"] = exc.user_id
    if hasattr(exc, "resource_type"):
        extra["resource_type"] = exc.resource_type

    message = f"Service exception: {type(exc).__name__}"

    if mapping.log_level == "error":
        logger.error(message, extra=extra, exc_info=True)
    elif mapping.log_level == "info":
        logger.info(message, extra=extra)
    else:
        # Default to warning
        logger.warning(message, extra=extra)


@dataclass
class HandleServiceErrorsConfig:
    """Configuration for the handle_service_errors decorator.

    Attributes:
        extra_mappings: Additional exception mappings for this endpoint.
        exclude_exceptions: Exception types to pass through without handling.
        log_unhandled: Whether to log unhandled exceptions.
    """

    extra_mappings: tuple[ExceptionMapping, ...] = field(default_factory=tuple)
    exclude_exceptions: frozenset[type[Exception]] = field(default_factory=frozenset)
    log_unhandled: bool = True


def handle_service_errors(
    func: F | None = None,
    *,
    extra_mappings: list[ExceptionMapping] | tuple[ExceptionMapping, ...] | None = None,
    exclude_exceptions: set[type[Exception]] | frozenset[type[Exception]] | None = None,
    log_unhandled: bool = True,
) -> F | Callable[[F], F]:
    """Decorator to handle service layer exceptions consistently.

    Catches known service exceptions and converts them to appropriate
    HTTPExceptions with consistent status codes, sanitized messages,
    and proper logging.

    Can be used with or without parentheses:
        @handle_service_errors
        async def endpoint(): ...

        @handle_service_errors()
        async def endpoint(): ...

        @handle_service_errors(exclude_exceptions={CustomError})
        async def endpoint(): ...

    Args:
        func: The endpoint function (when used without parentheses).
        extra_mappings: Additional exception-to-HTTP mappings for this endpoint.
                       These are checked before default mappings.
        exclude_exceptions: Exception types to pass through without handling.
                           Useful for endpoints with custom exception handling.
        log_unhandled: Whether to log exceptions that don't match any mapping.
                      Default True. Unhandled exceptions are re-raised for
                      the global exception handler.

    Returns:
        The decorated function or a decorator function.

    Example:
        @router.get("/{id}")
        @handle_service_errors
        async def get_item(id: str):
            # EntityNotFoundError -> 404
            # ValidationError -> 400
            # AccessDeniedError -> 403
            return await service.get(id)

        @router.post("/special")
        @handle_service_errors(exclude_exceptions={SpecialError})
        async def special_endpoint():
            try:
                result = await service.special_operation()
            except SpecialError:
                # Handle specially - decorator won't catch this
                return custom_response()
            return result
    """
    # Normalize arguments
    config = HandleServiceErrorsConfig(
        extra_mappings=tuple(extra_mappings) if extra_mappings else (),
        exclude_exceptions=frozenset(exclude_exceptions) if exclude_exceptions else frozenset(),
        log_unhandled=log_unhandled,
    )

    def decorator(endpoint_func: F) -> F:
        @functools.wraps(endpoint_func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get correlation ID for logging and error responses
            correlation_id = get_correlation_id()

            try:
                return await endpoint_func(*args, **kwargs)
            except HTTPException:
                # HTTPExceptions are already formatted - pass through
                raise
            except Exception as exc:
                # Check if this exception type should be excluded
                if config.exclude_exceptions and type(exc) in config.exclude_exceptions:
                    raise

                # Find matching exception mapping
                mapping = _find_mapping(exc, config.extra_mappings)

                if mapping:
                    # Log the exception with context
                    _log_exception(mapping, exc, correlation_id)

                    # Build sanitized error detail
                    detail = _build_detail(mapping, exc)

                    # Raise HTTPException with proper status and detail
                    raise HTTPException(
                        status_code=mapping.status_code,
                        detail=detail,
                    ) from exc

                # No mapping found - log and re-raise for global handler
                if config.log_unhandled:
                    logger.exception(
                        "Unhandled exception in endpoint",
                        extra={
                            "correlation_id": correlation_id,
                            "exception_type": type(exc).__name__,
                        },
                    )
                raise

        return wrapper  # type: ignore[return-value]

    # Support both @handle_service_errors and @handle_service_errors()
    if func is not None:
        return decorator(func)
    return decorator


# Re-export for convenience
__all__ = [
    "ExceptionMapping",
    "HandleServiceErrorsConfig",
    "handle_service_errors",
    "DEFAULT_EXCEPTION_MAPPINGS",
]
