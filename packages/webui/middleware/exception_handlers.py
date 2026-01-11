"""FastAPI exception handlers for core application errors.

This module provides global exception handlers that catch service-layer
exceptions and return properly formatted HTTP responses with:
- Sanitized error messages (no sensitive data leakage)
- Correlation IDs for request tracing
- Consistent error response format

These handlers work in conjunction with the @handle_service_errors decorator
from webui.utils.service_error_handler for endpoints that need more control.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from fastapi import HTTPException
from fastapi.responses import JSONResponse
from starlette.status import (
    HTTP_400_BAD_REQUEST,
    HTTP_403_FORBIDDEN,
    HTTP_404_NOT_FOUND,
    HTTP_409_CONFLICT,
    HTTP_500_INTERNAL_SERVER_ERROR,
)

from shared.database.exceptions import (
    AccessDeniedError as PackagesAccessDeniedError,
    DatabaseOperationError,
    EntityAlreadyExistsError,
    EntityNotFoundError,
    InvalidStateError,
    ValidationError,
)
from shared.utils.encryption import EncryptionNotConfiguredError

from .correlation import get_or_generate_correlation_id

if TYPE_CHECKING:
    from fastapi import FastAPI, Request

SharedAccessDeniedError: type[Exception] | None = None
try:  # pragma: no cover - shared may not be importable outside runtime
    from shared.database.exceptions import AccessDeniedError as _SharedAccessDeniedError
except Exception:  # pragma: no cover
    _SharedAccessDeniedError = None
else:
    SharedAccessDeniedError = _SharedAccessDeniedError

logger = logging.getLogger(__name__)

# Sanitized error messages - avoid exposing internal details
_ACCESS_DENIED_MESSAGE = "You do not have permission to perform this action."
_NOT_FOUND_MESSAGE = "The requested resource was not found."
_ALREADY_EXISTS_MESSAGE = "A resource with this identifier already exists."
_VALIDATION_ERROR_MESSAGE = "The request data is invalid."
_INVALID_STATE_MESSAGE = "The operation cannot be performed in the current state."


async def handle_access_denied_error(request: Request, exc: Exception) -> JSONResponse:
    """Translate AccessDeniedError into a sanitized 403 response."""

    correlation_id = getattr(request.state, "correlation_id", None)
    if not correlation_id:
        correlation_id = get_or_generate_correlation_id(request)
    logger.warning(
        "Access denied",
        extra={
            "correlation_id": correlation_id,
            "user_id": getattr(exc, "user_id", None),
            "resource_type": getattr(exc, "resource_type", None),
            "resource_id": getattr(exc, "resource_id", None),
            "path": request.url.path,
            "method": request.method,
        },
    )

    response = JSONResponse(
        status_code=HTTP_403_FORBIDDEN,
        content={"detail": _ACCESS_DENIED_MESSAGE},
    )
    response.headers["X-Correlation-ID"] = correlation_id
    return response


async def handle_unexpected_exception(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler for unexpected exceptions.

    Logs the full exception with stack trace server-side but returns
    a sanitized error message to the client to prevent information leakage.
    """
    correlation_id = getattr(request.state, "correlation_id", None)
    if not correlation_id:
        correlation_id = get_or_generate_correlation_id(request)

    # Log the full exception with stack trace for debugging
    logger.error(
        f"Unexpected error processing {request.method} {request.url.path}",
        exc_info=True,
        extra={
            "correlation_id": correlation_id,
            "path": request.url.path,
            "method": request.method,
            "exception_type": type(exc).__name__,
        },
    )

    # In development, include more details; in production, keep it minimal
    is_development = os.getenv("ENVIRONMENT", "production").lower() in ("development", "dev", "local")

    content = {
        "detail": "An unexpected error occurred. Please try again later.",
        "correlation_id": correlation_id,
    }

    # Include exception type in development mode only
    if is_development:
        content["exception_type"] = type(exc).__name__
        content["message"] = str(exc)

    response = JSONResponse(
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        content=content,
    )
    response.headers["X-Correlation-ID"] = correlation_id
    return response


async def handle_http_exception(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle HTTP exceptions and attach correlation reference."""
    correlation_id = getattr(request.state, "correlation_id", None)
    if not correlation_id:
        correlation_id = get_or_generate_correlation_id(request)

    content = {"detail": exc.detail, "reference": f"ERR-{correlation_id}"}
    response = JSONResponse(status_code=exc.status_code, content=content, headers=exc.headers)
    response.headers["X-Correlation-ID"] = correlation_id
    return response


async def handle_entity_not_found_error(request: Request, exc: EntityNotFoundError) -> JSONResponse:
    """Translate EntityNotFoundError into a sanitized 404 response.

    Includes entity type for context but not the entity ID to prevent
    information disclosure about valid/invalid resource identifiers.
    """
    correlation_id = getattr(request.state, "correlation_id", None)
    if not correlation_id:
        correlation_id = get_or_generate_correlation_id(request)

    logger.warning(
        "Entity not found",
        extra={
            "correlation_id": correlation_id,
            "entity_type": exc.entity_type,
            "entity_id": exc.entity_id,
            "path": request.url.path,
            "method": request.method,
        },
    )

    # Include entity type for helpful error messages, but not IDs
    detail = f"{exc.entity_type} not found" if exc.entity_type else _NOT_FOUND_MESSAGE

    response = JSONResponse(
        status_code=HTTP_404_NOT_FOUND,
        content={"detail": detail, "reference": f"ERR-{correlation_id}"},
    )
    response.headers["X-Correlation-ID"] = correlation_id
    return response


async def handle_entity_already_exists_error(request: Request, exc: EntityAlreadyExistsError) -> JSONResponse:
    """Translate EntityAlreadyExistsError into a sanitized 409 response."""
    correlation_id = getattr(request.state, "correlation_id", None)
    if not correlation_id:
        correlation_id = get_or_generate_correlation_id(request)

    logger.warning(
        "Entity already exists",
        extra={
            "correlation_id": correlation_id,
            "entity_type": exc.entity_type,
            "identifier": exc.identifier,
            "path": request.url.path,
            "method": request.method,
        },
    )

    # Include entity type but not the identifier
    detail = (
        f"A {exc.entity_type.lower()} with this identifier already exists"
        if exc.entity_type
        else _ALREADY_EXISTS_MESSAGE
    )

    response = JSONResponse(
        status_code=HTTP_409_CONFLICT,
        content={"detail": detail, "reference": f"ERR-{correlation_id}"},
    )
    response.headers["X-Correlation-ID"] = correlation_id
    return response


async def handle_validation_error(request: Request, exc: ValidationError) -> JSONResponse:
    """Translate ValidationError into a sanitized 400 response.

    May include field name for helpful error messages, but keeps
    error messages generic to avoid information disclosure.
    """
    correlation_id = getattr(request.state, "correlation_id", None)
    if not correlation_id:
        correlation_id = get_or_generate_correlation_id(request)

    logger.warning(
        "Validation error",
        extra={
            "correlation_id": correlation_id,
            "field": exc.field,
            "message": exc.message,
            "path": request.url.path,
            "method": request.method,
        },
    )

    # Include field name if available, but use generic message
    detail = f"Invalid value for field '{exc.field}'" if exc.field else _VALIDATION_ERROR_MESSAGE

    response = JSONResponse(
        status_code=HTTP_400_BAD_REQUEST,
        content={"detail": detail, "reference": f"ERR-{correlation_id}"},
    )
    response.headers["X-Correlation-ID"] = correlation_id
    return response


async def handle_invalid_state_error(request: Request, exc: InvalidStateError) -> JSONResponse:
    """Translate InvalidStateError into a sanitized 409 response.

    May include state information for helpful error messages.
    """
    correlation_id = getattr(request.state, "correlation_id", None)
    if not correlation_id:
        correlation_id = get_or_generate_correlation_id(request)

    logger.warning(
        "Invalid state error",
        extra={
            "correlation_id": correlation_id,
            "current_state": exc.current_state,
            "allowed_states": exc.allowed_states,
            "path": request.url.path,
            "method": request.method,
        },
    )

    # Include state info if available for helpful errors
    if exc.current_state and exc.allowed_states:
        detail = (
            f"Operation not allowed in state '{exc.current_state}'. " f"Allowed states: {', '.join(exc.allowed_states)}"
        )
    else:
        detail = _INVALID_STATE_MESSAGE

    response = JSONResponse(
        status_code=HTTP_409_CONFLICT,
        content={"detail": detail, "reference": f"ERR-{correlation_id}"},
    )
    response.headers["X-Correlation-ID"] = correlation_id
    return response


async def handle_encryption_not_configured_error(
    request: Request, exc: EncryptionNotConfiguredError  # noqa: ARG001
) -> JSONResponse:
    """Translate EncryptionNotConfiguredError into a 400 response.

    This occurs when encryption features are used without proper configuration.
    """
    correlation_id = getattr(request.state, "correlation_id", None)
    if not correlation_id:
        correlation_id = get_or_generate_correlation_id(request)

    logger.warning(
        "Encryption not configured",
        extra={
            "correlation_id": correlation_id,
            "path": request.url.path,
            "method": request.method,
        },
    )

    response = JSONResponse(
        status_code=HTTP_400_BAD_REQUEST,
        content={
            "detail": "Encryption is not configured. Set CONNECTOR_SECRETS_KEY environment variable.",
            "reference": f"ERR-{correlation_id}",
        },
    )
    response.headers["X-Correlation-ID"] = correlation_id
    return response


async def handle_database_operation_error(request: Request, exc: DatabaseOperationError) -> JSONResponse:
    """Translate DatabaseOperationError into a 500 response.

    This occurs when a database operation fails (e.g., constraint violations).
    """
    correlation_id = getattr(request.state, "correlation_id", None)
    if not correlation_id:
        correlation_id = get_or_generate_correlation_id(request)

    logger.error(
        "Database operation error",
        extra={
            "correlation_id": correlation_id,
            "operation": exc.operation,
            "entity_type": exc.entity_type,
            "details": exc.details,
            "path": request.url.path,
            "method": request.method,
        },
    )

    # Use the exception's built-in message format: "Failed to {operation} {entity_type}"
    detail = f"Failed to {exc.operation} {exc.entity_type}"

    response = JSONResponse(
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": detail, "reference": f"ERR-{correlation_id}"},
    )
    response.headers["X-Correlation-ID"] = correlation_id
    return response


def register_global_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers on the FastAPI application.

    Handlers are registered from most specific to least specific.
    The catch-all Exception handler must be registered last.
    """
    # Database/service layer exceptions - most specific first
    app.add_exception_handler(EntityNotFoundError, handle_entity_not_found_error)
    app.add_exception_handler(EntityAlreadyExistsError, handle_entity_already_exists_error)
    app.add_exception_handler(ValidationError, handle_validation_error)
    app.add_exception_handler(InvalidStateError, handle_invalid_state_error)
    app.add_exception_handler(EncryptionNotConfiguredError, handle_encryption_not_configured_error)
    app.add_exception_handler(DatabaseOperationError, handle_database_operation_error)

    # Access control exceptions
    app.add_exception_handler(PackagesAccessDeniedError, handle_access_denied_error)
    if SharedAccessDeniedError is not None and SharedAccessDeniedError is not PackagesAccessDeniedError:
        app.add_exception_handler(SharedAccessDeniedError, handle_access_denied_error)

    # HTTP exceptions
    app.add_exception_handler(HTTPException, handle_http_exception)

    # Catch-all for unexpected exceptions - must be registered last
    # to ensure more specific handlers take precedence
    app.add_exception_handler(Exception, handle_unexpected_exception)
