#!/usr/bin/env python3
"""
Correlation ID middleware for request tracing.

This middleware generates unique correlation IDs for each incoming request,
stores them in context variables for access throughout the request lifecycle,
and includes them in response headers for end-to-end tracing.
"""

import logging
import uuid
from contextvars import ContextVar

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

# Context variable to store correlation ID for the current request
correlation_id_var: ContextVar[str | None] = ContextVar("correlation_id", default=None)

logger = logging.getLogger(__name__)


class CorrelationIdFilter(logging.Filter):
    """Logging filter to inject correlation ID into log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation_id to the log record."""
        record.correlation_id = correlation_id_var.get() or "no-correlation-id"
        return True


def get_correlation_id() -> str:
    """Get the current correlation ID from context.

    Returns:
        The current correlation ID or a new one if none exists.
    """
    correlation_id = correlation_id_var.get()
    if not correlation_id:
        correlation_id = str(uuid.uuid4())
        correlation_id_var.set(correlation_id)
    return correlation_id


def set_correlation_id(correlation_id: str) -> None:
    """Set the correlation ID in the current context.

    Args:
        correlation_id: The correlation ID to set.
    """
    correlation_id_var.set(correlation_id)


class CorrelationMiddleware(BaseHTTPMiddleware):
    """Middleware to manage correlation IDs for request tracing.

    This middleware:
    1. Extracts correlation ID from incoming request headers (if present)
    2. Generates a new UUID if no correlation ID is provided
    3. Stores the correlation ID in context variables
    4. Adds the correlation ID to response headers
    5. Ensures the correlation ID is available throughout the request lifecycle
    """

    def __init__(
        self,
        app: ASGIApp,
        header_name: str = "X-Correlation-ID",
        generate_id_on_missing: bool = True,
    ) -> None:
        """Initialize the correlation middleware.

        Args:
            app: The ASGI application.
            header_name: The header name for correlation ID.
            generate_id_on_missing: Whether to generate ID if missing from request.
        """
        super().__init__(app)
        self.header_name = header_name
        self.generate_id_on_missing = generate_id_on_missing

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Process the request and add correlation ID handling.

        Args:
            request: The incoming request.
            call_next: The next middleware or endpoint to call.

        Returns:
            The response with correlation ID header added.
        """
        # Extract correlation ID from request headers
        correlation_id = request.headers.get(self.header_name)

        # Validate correlation ID format if provided
        if correlation_id:
            try:
                # Ensure it's a valid UUID format
                uuid.UUID(correlation_id)
            except ValueError:
                logger.warning(f"Invalid correlation ID format received: {correlation_id}. Generating new ID.")
                correlation_id = None

        # Generate new correlation ID if missing or invalid
        if not correlation_id and self.generate_id_on_missing:
            correlation_id = str(uuid.uuid4())
            logger.debug(f"Generated new correlation ID: {correlation_id}")

        # Set correlation ID in context for the duration of the request
        if correlation_id:
            correlation_id_var.set(correlation_id)

        try:
            # Log the incoming request with correlation ID
            logger.info(
                f"Incoming {request.method} request to {request.url.path}",
                extra={
                    "correlation_id": correlation_id,
                    "method": request.method,
                    "path": request.url.path,
                    "client_host": request.client.host if request.client else None,
                },
            )

            # Process the request
            response = await call_next(request)

            # Add correlation ID to response headers
            if correlation_id:
                response.headers[self.header_name] = correlation_id

            # Log the response
            logger.info(
                f"Request completed with status {response.status_code}",
                extra={
                    "correlation_id": correlation_id,
                    "status_code": response.status_code,
                    "method": request.method,
                    "path": request.url.path,
                },
            )

            return response

        except Exception as e:
            # Log the error with correlation ID
            logger.error(
                f"Request failed with error: {str(e)}",
                extra={
                    "correlation_id": correlation_id,
                    "method": request.method,
                    "path": request.url.path,
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            # Re-raise the exception to be handled by exception handlers
            raise

        finally:
            # Clear the context variable
            correlation_id_var.set(None)


def configure_logging_with_correlation() -> None:
    """Configure logging to include correlation IDs in all log messages.

    This function should be called during application startup to ensure
    all log messages include correlation IDs when available.
    """
    # Add correlation ID filter to all handlers
    correlation_filter = CorrelationIdFilter()

    # Add to root logger to affect all loggers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.addFilter(correlation_filter)

    # Update formatter to include correlation ID
    # This is a basic example - adjust based on your logging configuration
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - [%(correlation_id)s] - %(message)s")
    for handler in root_logger.handlers:
        handler.setFormatter(formatter)


def get_or_generate_correlation_id(request: Request | None = None) -> str:
    """Get correlation ID from request or context, generating if needed.

    Args:
        request: Optional request object to extract correlation ID from.

    Returns:
        A correlation ID string.
    """
    # First try to get from context
    correlation_id = correlation_id_var.get()
    if correlation_id:
        return correlation_id

    # If request provided, try to get from headers
    if request:
        correlation_id = request.headers.get("X-Correlation-ID")
        if correlation_id:
            try:
                # Validate it's a proper UUID
                uuid.UUID(correlation_id)
                return correlation_id
            except ValueError:
                pass

    # Generate new one if all else fails
    return str(uuid.uuid4())
