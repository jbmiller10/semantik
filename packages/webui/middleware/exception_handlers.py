"""FastAPI exception handlers for core application errors."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi.responses import JSONResponse
from starlette.status import HTTP_403_FORBIDDEN

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


def register_global_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers on the FastAPI application."""

    app.add_exception_handler(PackagesAccessDeniedError, handle_access_denied_error)
    if SharedAccessDeniedError is not None and SharedAccessDeniedError is not PackagesAccessDeniedError:
        app.add_exception_handler(SharedAccessDeniedError, handle_access_denied_error)
