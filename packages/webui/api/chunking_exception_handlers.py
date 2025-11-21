#!/usr/bin/env python3
"""
Exception handlers for chunking-related errors.

This module provides FastAPI exception handlers for chunking exceptions,
including error sanitization and structured JSON responses.
"""

import logging
import os
import re
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from webui.api.chunking_exceptions import (
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

try:
    from shared.chunking.infrastructure.exception_translator import exception_translator
    from shared.chunking.infrastructure.exceptions import (
        ApplicationException,
        BaseChunkingException,
        DomainException,
        InfrastructureException,
    )

    INFRASTRUCTURE_AVAILABLE = True
except ImportError:
    INFRASTRUCTURE_AVAILABLE = False

try:
    from webui.middleware.correlation import get_or_generate_correlation_id
except ImportError:
    # Define fallback with proper typing
    def get_or_generate_correlation_id(request: Request | None = None) -> str:
        """Fallback correlation ID generator."""
        import uuid

        if request is None:
            return str(uuid.uuid4())
        return request.headers.get("X-Correlation-ID", str(uuid.uuid4()))


logger = logging.getLogger(__name__)


def _sanitize_error_detail(detail: str, is_production: bool = False) -> str:
    """Sanitize error details to prevent information leakage.

    Args:
        detail: The error detail message
        is_production: Whether running in production mode

    Returns:
        Sanitized error detail
    """
    if not is_production:
        return detail

    # Patterns to sanitize
    sanitize_patterns = [
        (r"/[^\s]+\.(env|config|key|pem)", "[REDACTED]"),  # File paths with sensitive extensions
        (r"/etc/[^\s]+", "[REDACTED]"),  # System config files
        (r"/home/[^\s]+", "[REDACTED]"),  # User home directories
        (r"(postgres|mysql|mongodb)://[^\s]+", "[REDACTED]"),  # Connection strings
        (r'(api_key|token|secret|password)\s*=\s*[\'"][^\'"]+[\'"]', "[REDACTED]"),  # Credentials
        (r"\b(sk-[a-zA-Z0-9]+|key-[a-zA-Z0-9]+)\b", "[REDACTED]"),  # API keys
    ]

    sanitized = detail
    for pattern, replacement in sanitize_patterns:
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)

    return sanitized


def _create_error_response(
    request: Request,
    exc: ChunkingError,
    status_code: int,
    is_production: bool | None = None,
) -> JSONResponse:
    """Create a structured error response.

    Args:
        request: The FastAPI request
        exc: The chunking exception
        status_code: HTTP status code
        is_production: Whether in production mode (defaults to checking ENV)

    Returns:
        JSON response with error details
    """
    if is_production is None:
        is_production = os.getenv("ENV", "development").lower() == "production"

    # Log based on status code
    if status_code >= 500:
        logger.error(
            f"Server error: {exc.__class__.__name__}",
            extra={"error": str(exc), "correlation_id": exc.correlation_id},
        )
    else:
        logger.warning(
            f"Client error: {exc.__class__.__name__}",
            extra={"error": str(exc), "correlation_id": exc.correlation_id},
        )

    # Build error response
    error_dict = exc.to_dict()
    error_dict["detail"] = _sanitize_error_detail(error_dict.get("detail", ""), is_production)

    # Add request context
    error_dict["request"] = {
        "method": request.method,
        "path": str(request.url.path),
        "query_params": dict(request.query_params) if not is_production else None,
    }

    response = JSONResponse(
        status_code=status_code,
        content=error_dict,
    )

    # Add correlation ID header
    if exc.correlation_id:
        response.headers["X-Correlation-ID"] = exc.correlation_id

    return response


if INFRASTRUCTURE_AVAILABLE:

    async def handle_chunking_exception(request: Request, exc: BaseChunkingException) -> JSONResponse:
        """Handle all chunking exceptions using the infrastructure translator.

        Args:
            request: The FastAPI request object
            exc: The chunking exception to handle

        Returns:
            JSON response with structured error information
        """
        correlation_id = get_or_generate_correlation_id(request)

        # Log the exception
        logger.error(
            f"Chunking exception occurred: {exc.__class__.__name__}",
            extra={
                "correlation_id": correlation_id,
                "exception_type": exc.__class__.__name__,
                "details": exc.to_dict() if hasattr(exc, "to_dict") else str(exc),
            },
        )

        # Use the exception translator to create the response
        return exception_translator.create_error_response(exc, correlation_id)


async def handle_application_exception(request: Request, exc: Any) -> JSONResponse:
    """Handle application-level exceptions.

    Args:
        request: The FastAPI request object
        exc: The application exception to handle

    Returns:
        JSON response with structured error information
    """
    get_or_generate_correlation_id(request)

    # Translate to HTTP exception and get status code
    http_exc = exception_translator.translate_application_to_api(exc)

    return JSONResponse(
        status_code=http_exc.status_code,
        content=http_exc.detail,
    )


async def handle_domain_exception(request: Request, exc: Any) -> JSONResponse:
    """Handle domain-level exceptions.

    Args:
        request: The FastAPI request object
        exc: The domain exception to handle

    Returns:
        JSON response with structured error information
    """
    correlation_id = get_or_generate_correlation_id(request)

    # Translate domain to application exception first
    app_exc = exception_translator.translate_domain_to_application(exc, correlation_id)

    # Then translate to HTTP
    http_exc = exception_translator.translate_application_to_api(app_exc)

    return JSONResponse(
        status_code=http_exc.status_code,
        content=http_exc.detail,
    )


async def handle_infrastructure_exception(request: Request, exc: Any) -> JSONResponse:
    """Handle infrastructure-level exceptions.

    Args:
        request: The FastAPI request object
        exc: The infrastructure exception to handle

    Returns:
        JSON response with structured error information
    """
    get_or_generate_correlation_id(request)

    # Translate infrastructure to application exception
    app_exc = exception_translator.translate_infrastructure_to_application(exc, {"request_path": str(request.url)})

    # Then translate to HTTP
    http_exc = exception_translator.translate_application_to_api(app_exc)

    return JSONResponse(
        status_code=http_exc.status_code,
        content=http_exc.detail,
    )


# Individual exception handlers for specific chunking exceptions
async def handle_chunking_memory_error(request: Request, exc: ChunkingMemoryError) -> JSONResponse:
    """Handle memory exhaustion errors."""
    response = _create_error_response(request, exc, 507)  # Insufficient Storage
    response.headers["Retry-After"] = "60"
    return response


async def handle_chunking_timeout_error(request: Request, exc: ChunkingTimeoutError) -> JSONResponse:
    """Handle timeout errors."""
    return _create_error_response(request, exc, 504)  # Gateway Timeout


async def handle_chunking_validation_error(request: Request, exc: ChunkingValidationError) -> JSONResponse:
    """Handle validation errors."""
    return _create_error_response(request, exc, 422)  # Unprocessable Entity


async def handle_chunking_strategy_error(request: Request, exc: ChunkingStrategyError) -> JSONResponse:
    """Handle strategy errors."""
    return _create_error_response(request, exc, 501)  # Not Implemented


async def handle_chunking_resource_limit_error(request: Request, exc: ChunkingResourceLimitError) -> JSONResponse:
    """Handle resource limit errors."""
    response = _create_error_response(request, exc, 503)  # Service Unavailable
    response.headers["Retry-After"] = "30"
    return response


async def handle_chunking_partial_failure_error(request: Request, exc: ChunkingPartialFailureError) -> JSONResponse:
    """Handle partial failure errors."""
    return _create_error_response(request, exc, 207)  # Multi-Status


async def handle_chunking_configuration_error(request: Request, exc: ChunkingConfigurationError) -> JSONResponse:
    """Handle configuration errors."""
    return _create_error_response(request, exc, 500)  # Internal Server Error


async def handle_chunking_dependency_error(request: Request, exc: ChunkingDependencyError) -> JSONResponse:
    """Handle dependency errors."""
    response = _create_error_response(request, exc, 503)  # Service Unavailable
    response.headers["Retry-After"] = "60"
    return response


async def handle_base_chunking_error(request: Request, exc: ChunkingError) -> JSONResponse:
    """Handle base chunking errors as fallback."""
    return _create_error_response(request, exc, 500)  # Internal Server Error


def register_chunking_exception_handlers(app: FastAPI) -> None:
    """Register all chunking exception handlers with the FastAPI app.

    Args:
        app: The FastAPI application instance
    """
    # Type-cast handlers to match FastAPI's expected signature
    # FastAPI expects: Callable[[Request, Exception], Response]
    app.add_exception_handler(ChunkingMemoryError, handle_chunking_memory_error)  # type: ignore[arg-type]
    app.add_exception_handler(ChunkingTimeoutError, handle_chunking_timeout_error)  # type: ignore[arg-type]
    app.add_exception_handler(ChunkingValidationError, handle_chunking_validation_error)  # type: ignore[arg-type]
    app.add_exception_handler(ChunkingStrategyError, handle_chunking_strategy_error)  # type: ignore[arg-type]
    app.add_exception_handler(ChunkingResourceLimitError, handle_chunking_resource_limit_error)  # type: ignore[arg-type]
    app.add_exception_handler(ChunkingPartialFailureError, handle_chunking_partial_failure_error)  # type: ignore[arg-type]
    app.add_exception_handler(ChunkingConfigurationError, handle_chunking_configuration_error)  # type: ignore[arg-type]
    app.add_exception_handler(ChunkingDependencyError, handle_chunking_dependency_error)  # type: ignore[arg-type]
    app.add_exception_handler(ChunkingError, handle_base_chunking_error)  # type: ignore[arg-type]  # Base class as fallback

    # Register infrastructure handlers if available
    if INFRASTRUCTURE_AVAILABLE:
        app.add_exception_handler(BaseChunkingException, handle_chunking_exception)  # type: ignore[arg-type]
        app.add_exception_handler(ApplicationException, handle_application_exception)
        app.add_exception_handler(DomainException, handle_domain_exception)
        app.add_exception_handler(InfrastructureException, handle_infrastructure_exception)

    logger.info("Chunking exception handlers registered")


# Alias for backward compatibility
register_exception_handlers = register_chunking_exception_handlers
