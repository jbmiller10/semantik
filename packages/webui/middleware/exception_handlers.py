"""FastAPI exception handlers for core application errors."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from fastapi import HTTPException
from fastapi.responses import JSONResponse
from starlette.status import HTTP_403_FORBIDDEN, HTTP_500_INTERNAL_SERVER_ERROR

from shared.database.exceptions import AccessDeniedError as PackagesAccessDeniedError

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

_ACCESS_DENIED_MESSAGE = "You do not have permission to perform this action."


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


def register_global_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers on the FastAPI application."""

    app.add_exception_handler(PackagesAccessDeniedError, handle_access_denied_error)
    if SharedAccessDeniedError is not None and SharedAccessDeniedError is not PackagesAccessDeniedError:
        app.add_exception_handler(SharedAccessDeniedError, handle_access_denied_error)

    app.add_exception_handler(HTTPException, handle_http_exception)

    # Catch-all for unexpected exceptions - must be registered last
    # to ensure more specific handlers take precedence
    app.add_exception_handler(Exception, handle_unexpected_exception)
