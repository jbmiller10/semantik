#!/usr/bin/env python3
"""
Exception handlers for chunking-related errors using the new infrastructure.

This module provides FastAPI exception handlers that use the infrastructure
exception translator to convert exceptions into structured JSON responses.
"""

import logging

from fastapi import Request
from fastapi.responses import JSONResponse

from packages.shared.chunking.infrastructure.exception_translator import (
    exception_translator,
)
from packages.shared.chunking.infrastructure.exceptions import (
    ApplicationException,
    BaseChunkingException,
    DomainException,
    InfrastructureException,
)
from packages.webui.middleware.correlation import get_or_generate_correlation_id

logger = logging.getLogger(__name__)


async def handle_chunking_exception(
    request: Request, exc: BaseChunkingException
) -> JSONResponse:
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
            "details": exc.to_dict() if hasattr(exc, 'to_dict') else str(exc),
        },
    )

    # Use the exception translator to create the response
    return exception_translator.create_error_response(exc, correlation_id)


async def handle_application_exception(
    request: Request, exc: ApplicationException
) -> JSONResponse:
    """Handle application-level exceptions.

    Args:
        request: The FastAPI request object
        exc: The application exception to handle

    Returns:
        JSON response with structured error information
    """
    correlation_id = get_or_generate_correlation_id(request)

    # Translate to HTTP exception and get status code
    http_exc = exception_translator.translate_application_to_api(exc)

    return JSONResponse(
        status_code=http_exc.status_code,
        content=http_exc.detail,
    )


async def handle_domain_exception(
    request: Request, exc: DomainException
) -> JSONResponse:
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


async def handle_infrastructure_exception(
    request: Request, exc: InfrastructureException
) -> JSONResponse:
    """Handle infrastructure-level exceptions.

    Args:
        request: The FastAPI request object
        exc: The infrastructure exception to handle

    Returns:
        JSON response with structured error information
    """
    correlation_id = get_or_generate_correlation_id(request)

    # Translate infrastructure to application exception
    app_exc = exception_translator.translate_infrastructure_to_application(
        exc, {"request_path": str(request.url)}
    )

    # Then translate to HTTP
    http_exc = exception_translator.translate_application_to_api(app_exc)

    return JSONResponse(
        status_code=http_exc.status_code,
        content=http_exc.detail,
    )


def register_exception_handlers(app):
    """Register all chunking exception handlers with the FastAPI app.

    Args:
        app: The FastAPI application instance
    """
    # Register handlers for different exception types
    app.add_exception_handler(BaseChunkingException, handle_chunking_exception)
    app.add_exception_handler(ApplicationException, handle_application_exception)
    app.add_exception_handler(DomainException, handle_domain_exception)
    app.add_exception_handler(InfrastructureException, handle_infrastructure_exception)

    logger.info("Chunking exception handlers registered")


# Alias for backward compatibility
register_chunking_exception_handlers = register_exception_handlers
