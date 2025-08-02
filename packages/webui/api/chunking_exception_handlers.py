#!/usr/bin/env python3
"""
Exception handlers for chunking-related errors.

This module provides FastAPI exception handlers that convert chunking exceptions
into structured JSON responses with appropriate HTTP status codes, correlation IDs,
and recovery hints.
"""

import logging
from typing import Any

from fastapi import Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

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
)
from packages.webui.middleware.correlation import get_or_generate_correlation_id

logger = logging.getLogger(__name__)


def _sanitize_error_detail(detail: str, is_production: bool = True) -> str:
    """Sanitize error details to prevent information leakage.

    Args:
        detail: The raw error detail message.
        is_production: Whether the application is in production mode.

    Returns:
        Sanitized error message safe for client consumption.
    """
    if not is_production:
        # In development, return full details
        return detail

    # In production, remove sensitive information
    # This is a basic implementation - customize based on your security needs
    sensitive_patterns = [
        # System files and paths
        r"\/etc\/[a-zA-Z0-9_\-\/]+",
        # File system paths
        r"\/[a-zA-Z0-9_\-\/]+\.(py|conf|env|key|pem)",
        # Database connection strings
        r"(postgres|mysql|mongodb):\/\/[^@]+@[^\/]+",
        # API keys or tokens
        r"(api_key|token|secret|password)[\s]*[=:]\s*['\"]?[a-zA-Z0-9\-_]+['\"]?",
        # Internal service names
        r"(service|host|server)[\s]*[=:]\s*[a-zA-Z0-9\-._]+",
    ]

    sanitized = detail
    for pattern in sensitive_patterns:
        import re

        sanitized = re.sub(pattern, "[REDACTED]", sanitized, flags=re.IGNORECASE)

    return sanitized


def _create_error_response(
    request: Request,
    exc: ChunkingError,
    status_code: int,
    is_production: bool = True,
) -> JSONResponse:
    """Create a standardized error response from a chunking exception.

    Args:
        request: The incoming request.
        exc: The chunking exception.
        status_code: HTTP status code for the response.
        is_production: Whether to sanitize error messages.

    Returns:
        JSON response with error details.
    """
    # Get or ensure correlation ID
    correlation_id = exc.correlation_id or get_or_generate_correlation_id(request)

    # Build error response
    error_data = exc.to_dict()

    # Sanitize the detail message
    error_data["detail"] = _sanitize_error_detail(error_data["detail"], is_production)

    # Add request context
    error_data["request"] = {
        "method": request.method,
        "path": str(request.url.path),
        "query_params": dict(request.query_params) if not is_production else None,
    }

    # Add correlation ID to response headers
    headers = {"X-Correlation-ID": correlation_id}

    # Log the error with appropriate severity
    log_data = {
        "correlation_id": correlation_id,
        "error_type": type(exc).__name__,
        "status_code": status_code,
        "path": str(request.url.path),
        "method": request.method,
    }

    if status_code >= 500:
        logger.error(f"Server error in chunking operation: {exc}", extra=log_data, exc_info=True)
    elif status_code >= 400:
        logger.warning(f"Client error in chunking operation: {exc}", extra=log_data)
    else:
        logger.info(f"Chunking operation issue: {exc}", extra=log_data)

    return JSONResponse(
        status_code=status_code,
        content=jsonable_encoder(error_data),
        headers=headers,
    )


async def chunking_error_handler(request: Request, exc: ChunkingError) -> JSONResponse:
    """Handle base ChunkingError exceptions.

    This is the fallback handler for any chunking errors not caught by
    more specific handlers.

    Args:
        request: The incoming request.
        exc: The chunking error.

    Returns:
        JSON response with status 500.
    """
    return _create_error_response(
        request=request,
        exc=exc,
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    )


async def chunking_memory_error_handler(
    request: Request,
    exc: ChunkingMemoryError,
) -> JSONResponse:
    """Handle memory limit exceeded errors.

    Args:
        request: The incoming request.
        exc: The memory error.

    Returns:
        JSON response with status 507 (Insufficient Storage).
    """
    return _create_error_response(
        request=request,
        exc=exc,
        status_code=status.HTTP_507_INSUFFICIENT_STORAGE,
    )


async def chunking_timeout_error_handler(
    request: Request,
    exc: ChunkingTimeoutError,
) -> JSONResponse:
    """Handle timeout errors.

    Args:
        request: The incoming request.
        exc: The timeout error.

    Returns:
        JSON response with status 504 (Gateway Timeout).
    """
    return _create_error_response(
        request=request,
        exc=exc,
        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
    )


async def chunking_validation_error_handler(
    request: Request,
    exc: ChunkingValidationError,
) -> JSONResponse:
    """Handle validation errors.

    Args:
        request: The incoming request.
        exc: The validation error.

    Returns:
        JSON response with status 422 (Unprocessable Entity).
    """
    return _create_error_response(
        request=request,
        exc=exc,
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
    )


async def chunking_strategy_error_handler(
    request: Request,
    exc: ChunkingStrategyError,
) -> JSONResponse:
    """Handle strategy-related errors.

    Args:
        request: The incoming request.
        exc: The strategy error.

    Returns:
        JSON response with status 501 (Not Implemented) or 500.
    """
    # Use 501 if strategy is not implemented, 500 for other strategy errors
    status_code = (
        status.HTTP_501_NOT_IMPLEMENTED
        if "not implemented" in exc.detail.lower() or "unsupported" in exc.detail.lower()
        else status.HTTP_500_INTERNAL_SERVER_ERROR
    )

    return _create_error_response(
        request=request,
        exc=exc,
        status_code=status_code,
    )


async def chunking_resource_limit_error_handler(
    request: Request,
    exc: ChunkingResourceLimitError,
) -> JSONResponse:
    """Handle resource limit errors.

    Args:
        request: The incoming request.
        exc: The resource limit error.

    Returns:
        JSON response with status 503 (Service Unavailable).
    """
    response = _create_error_response(
        request=request,
        exc=exc,
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
    )

    # Add Retry-After header if appropriate
    # Suggest retry after 30 seconds for resource limits
    response.headers["Retry-After"] = "30"

    return response


async def chunking_partial_failure_error_handler(
    request: Request,
    exc: ChunkingPartialFailureError,
) -> JSONResponse:
    """Handle partial failure errors.

    Args:
        request: The incoming request.
        exc: The partial failure error.

    Returns:
        JSON response with status 207 (Multi-Status).
    """
    return _create_error_response(
        request=request,
        exc=exc,
        status_code=status.HTTP_207_MULTI_STATUS,
    )


async def chunking_configuration_error_handler(
    request: Request,
    exc: ChunkingConfigurationError,
) -> JSONResponse:
    """Handle configuration errors.

    Args:
        request: The incoming request.
        exc: The configuration error.

    Returns:
        JSON response with status 500 (Internal Server Error).
    """
    return _create_error_response(
        request=request,
        exc=exc,
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    )


async def chunking_dependency_error_handler(
    request: Request,
    exc: ChunkingDependencyError,
) -> JSONResponse:
    """Handle dependency errors.

    Args:
        request: The incoming request.
        exc: The dependency error.

    Returns:
        JSON response with status 503 (Service Unavailable).
    """
    response = _create_error_response(
        request=request,
        exc=exc,
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
    )

    # Add Retry-After header for temporary dependency issues
    response.headers["Retry-After"] = "60"  # Retry after 1 minute

    return response


def register_chunking_exception_handlers(app: Any) -> None:
    """Register all chunking exception handlers with a FastAPI app.

    Args:
        app: The FastAPI application instance.
    """
    # Register specific handlers
    app.add_exception_handler(ChunkingMemoryError, chunking_memory_error_handler)
    app.add_exception_handler(ChunkingTimeoutError, chunking_timeout_error_handler)
    app.add_exception_handler(ChunkingValidationError, chunking_validation_error_handler)
    app.add_exception_handler(ChunkingStrategyError, chunking_strategy_error_handler)
    app.add_exception_handler(ChunkingResourceLimitError, chunking_resource_limit_error_handler)
    app.add_exception_handler(ChunkingPartialFailureError, chunking_partial_failure_error_handler)
    app.add_exception_handler(ChunkingConfigurationError, chunking_configuration_error_handler)
    app.add_exception_handler(ChunkingDependencyError, chunking_dependency_error_handler)

    # Register base handler last (as fallback)
    app.add_exception_handler(ChunkingError, chunking_error_handler)

    logger.info("Registered all chunking exception handlers")
