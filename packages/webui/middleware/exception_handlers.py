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
    """Translate AccessDeniedError into a 403 response.

    Uses the exception's message since service-layer exceptions already
    contain user-appropriate messages.
    """

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

    # Use original exception message - service layer provides appropriate messages
    detail = str(exc) if str(exc) else _ACCESS_DENIED_MESSAGE

    response = JSONResponse(
        status_code=HTTP_403_FORBIDDEN,
        content={"detail": detail},
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


async def handle_http_exception(request: Request, exc: Exception) -> JSONResponse:
    """Handle HTTP exceptions and attach correlation reference."""
    # Type assertion for HTTPException attributes
    http_exc = exc if isinstance(exc, HTTPException) else None
    if http_exc is None:
        # Fallback - shouldn't happen but handle gracefully
        return JSONResponse(
            status_code=500,
            content={"detail": str(exc)},
        )

    correlation_id = getattr(request.state, "correlation_id", None)
    if not correlation_id:
        correlation_id = get_or_generate_correlation_id(request)

    content = {"detail": http_exc.detail, "reference": f"ERR-{correlation_id}"}
    response = JSONResponse(status_code=http_exc.status_code, content=content, headers=http_exc.headers)
    response.headers["X-Correlation-ID"] = correlation_id
    return response


async def handle_entity_not_found_error(request: Request, exc: Exception) -> JSONResponse:
    """Translate EntityNotFoundError into a 404 response.

    Uses the exception's message since service-layer exceptions already
    contain user-appropriate messages.
    """
    correlation_id = getattr(request.state, "correlation_id", None)
    if not correlation_id:
        correlation_id = get_or_generate_correlation_id(request)

    logger.warning(
        "Entity not found",
        extra={
            "correlation_id": correlation_id,
            "entity_type": getattr(exc, "entity_type", None),
            "entity_id": getattr(exc, "entity_id", None),
            "path": request.url.path,
            "method": request.method,
        },
    )

    # Use original exception message - service layer provides appropriate messages
    detail = str(exc) if str(exc) else _NOT_FOUND_MESSAGE

    response = JSONResponse(
        status_code=HTTP_404_NOT_FOUND,
        content={"detail": detail, "reference": f"ERR-{correlation_id}"},
    )
    response.headers["X-Correlation-ID"] = correlation_id
    return response


async def handle_entity_already_exists_error(request: Request, exc: Exception) -> JSONResponse:
    """Translate EntityAlreadyExistsError into a 409 response.

    Uses the exception's message since service-layer exceptions already
    contain user-appropriate messages.
    """
    correlation_id = getattr(request.state, "correlation_id", None)
    if not correlation_id:
        correlation_id = get_or_generate_correlation_id(request)

    logger.warning(
        "Entity already exists",
        extra={
            "correlation_id": correlation_id,
            "entity_type": getattr(exc, "entity_type", None),
            "identifier": getattr(exc, "identifier", None),
            "path": request.url.path,
            "method": request.method,
        },
    )

    # Use original exception message - service layer provides appropriate messages
    detail = str(exc) if str(exc) else _ALREADY_EXISTS_MESSAGE

    response = JSONResponse(
        status_code=HTTP_409_CONFLICT,
        content={"detail": detail, "reference": f"ERR-{correlation_id}"},
    )
    response.headers["X-Correlation-ID"] = correlation_id
    return response


async def handle_validation_error(request: Request, exc: Exception) -> JSONResponse:
    """Translate ValidationError into a 400 response.

    Uses the exception's message since service-layer exceptions already
    contain user-appropriate messages.
    """
    correlation_id = getattr(request.state, "correlation_id", None)
    if not correlation_id:
        correlation_id = get_or_generate_correlation_id(request)

    logger.warning(
        "Validation error",
        extra={
            "correlation_id": correlation_id,
            "field": getattr(exc, "field", None),
            "validation_message": getattr(exc, "message", None),  # Avoid conflict with LogRecord.message
            "path": request.url.path,
            "method": request.method,
        },
    )

    # Use original exception message - service layer provides appropriate messages
    detail = str(exc) if str(exc) else _VALIDATION_ERROR_MESSAGE

    response = JSONResponse(
        status_code=HTTP_400_BAD_REQUEST,
        content={"detail": detail, "reference": f"ERR-{correlation_id}"},
    )
    response.headers["X-Correlation-ID"] = correlation_id
    return response


async def handle_invalid_state_error(request: Request, exc: Exception) -> JSONResponse:
    """Translate InvalidStateError into a 409 response.

    Uses the exception's message since service-layer exceptions already
    contain user-appropriate messages.
    """
    correlation_id = getattr(request.state, "correlation_id", None)
    if not correlation_id:
        correlation_id = get_or_generate_correlation_id(request)

    logger.warning(
        "Invalid state error",
        extra={
            "correlation_id": correlation_id,
            "current_state": getattr(exc, "current_state", None),
            "allowed_states": getattr(exc, "allowed_states", None),
            "path": request.url.path,
            "method": request.method,
        },
    )

    # Use original exception message - service layer provides appropriate messages
    detail = str(exc) if str(exc) else _INVALID_STATE_MESSAGE

    response = JSONResponse(
        status_code=HTTP_409_CONFLICT,
        content={"detail": detail, "reference": f"ERR-{correlation_id}"},
    )
    response.headers["X-Correlation-ID"] = correlation_id
    return response


async def handle_encryption_not_configured_error(request: Request, exc: Exception) -> JSONResponse:  # noqa: ARG001
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


async def handle_database_operation_error(request: Request, exc: Exception) -> JSONResponse:
    """Translate DatabaseOperationError into a 500 response.

    Uses the exception's message since service-layer exceptions already
    contain user-appropriate messages.
    """
    correlation_id = getattr(request.state, "correlation_id", None)
    if not correlation_id:
        correlation_id = get_or_generate_correlation_id(request)

    logger.error(
        "Database operation error",
        extra={
            "correlation_id": correlation_id,
            "operation": getattr(exc, "operation", None),
            "entity_type": getattr(exc, "entity_type", None),
            "details": getattr(exc, "details", None),
            "path": request.url.path,
            "method": request.method,
        },
    )

    # Use original exception message - service layer provides appropriate messages
    detail = str(exc) if str(exc) else "A database operation failed"

    response = JSONResponse(
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": detail, "reference": f"ERR-{correlation_id}"},
    )
    response.headers["X-Correlation-ID"] = correlation_id
    return response


async def handle_value_error(request: Request, exc: Exception) -> JSONResponse:
    """Translate ValueError into a 400 response.

    ValueError is commonly used for validation in Python code, so we treat it
    as a bad request and return the exception's message.
    """
    correlation_id = getattr(request.state, "correlation_id", None)
    if not correlation_id:
        correlation_id = get_or_generate_correlation_id(request)

    logger.warning(
        "Value error (validation)",
        extra={
            "correlation_id": correlation_id,
            "path": request.url.path,
            "method": request.method,
        },
    )

    # Use original exception message - service layer provides appropriate messages
    detail = str(exc) if str(exc) else "Invalid value provided"

    response = JSONResponse(
        status_code=HTTP_400_BAD_REQUEST,
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

    # Python built-in validation errors
    app.add_exception_handler(ValueError, handle_value_error)

    # Access control exceptions
    app.add_exception_handler(PackagesAccessDeniedError, handle_access_denied_error)
    if SharedAccessDeniedError is not None and SharedAccessDeniedError is not PackagesAccessDeniedError:
        app.add_exception_handler(SharedAccessDeniedError, handle_access_denied_error)

    # HTTP exceptions
    app.add_exception_handler(HTTPException, handle_http_exception)

    # Catch-all for unexpected exceptions - must be registered last
    # to ensure more specific handlers take precedence
    app.add_exception_handler(Exception, handle_unexpected_exception)
