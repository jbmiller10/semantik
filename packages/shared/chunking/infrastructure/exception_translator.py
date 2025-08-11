#!/usr/bin/env python3
"""
Exception translator for converting exceptions between architectural layers.

Handles translation of domain exceptions to application exceptions,
and application exceptions to HTTP exceptions with proper status codes.
"""

import logging
from typing import Any

from fastapi import HTTPException
from fastapi.responses import JSONResponse

from .exceptions import (
    ApplicationException,
    ChunkingStrategyError,
    DatabaseException,
    DocumentTooLargeError,
    DomainException,
    ExternalServiceException,
    InfrastructureException,
    InvalidStateTransition,
    PermissionDeniedException,
    ResourceNotFoundException,
    StreamingException,
    ValidationException,
)

logger = logging.getLogger(__name__)


class ExceptionTranslator:
    """Translates exceptions between architectural layers."""

    def __init__(self) -> None:
        """Initialize exception translator with mappings."""
        from collections.abc import Callable

        # Domain to Application mappings
        self.domain_to_app_map: dict[type[Exception], Callable] = {
            DocumentTooLargeError: self._document_too_large_to_app,
            InvalidStateTransition: self._invalid_state_to_app,
            ChunkingStrategyError: self._strategy_error_to_app,
        }

        # Application to API status code mappings
        self.app_to_api_map: dict[type[Exception], int] = {
            ValidationException: 400,
            ResourceNotFoundException: 404,
            PermissionDeniedException: 403,
            DocumentTooLargeError: 413,
            InvalidStateTransition: 409,
            ChunkingStrategyError: 422,  # Unprocessable Entity for strategy errors
            ApplicationException: 500,  # Base application exception
            StreamingException: 500,  # Streaming errors are server-side
        }

    def translate_domain_to_application(
        self, exc: DomainException, correlation_id: str | None = None
    ) -> ApplicationException:
        """Translate domain exception to application exception.

        Args:
            exc: Domain exception to translate
            correlation_id: Optional correlation ID to use

        Returns:
            Translated application exception
        """
        # Log original exception with full context
        logger.error(
            "Domain exception occurred",
            extra={
                "exception_type": type(exc).__name__,
                "correlation_id": exc.correlation_id,
                "details": exc.to_dict(),
            },
        )

        # Get specific translator or use default
        translator = self.domain_to_app_map.get(type(exc), self._default_domain_to_app)

        return translator(exc, correlation_id)

    def translate_application_to_api(self, exc: ApplicationException) -> HTTPException:
        """Translate application exception to HTTP exception.

        Args:
            exc: Application exception to translate

        Returns:
            HTTP exception with appropriate status code
        """
        # Get HTTP status code
        status_code = self.app_to_api_map.get(type(exc), 500)

        # Create structured error response
        detail = {
            "error": {
                "message": exc.message,
                "code": exc.code,
                "correlation_id": exc.correlation_id,
                "details": exc.details,
            }
        }

        # Log API error
        logger.warning(
            "API error response",
            extra={
                "status_code": status_code,
                "correlation_id": exc.correlation_id,
                "error_code": exc.code,
            },
        )

        return HTTPException(status_code=status_code, detail=detail)

    def translate_infrastructure_to_application(
        self, exc: Exception, context: dict[str, Any]
    ) -> ApplicationException:
        """Translate infrastructure exception to application exception.

        Args:
            exc: Infrastructure exception to translate
            context: Additional context for translation

        Returns:
            Translated application exception
        """
        if isinstance(exc, DatabaseException):
            # Check if it's a not found error
            if "does not exist" in str(exc).lower() or "not found" in str(exc).lower():
                return ResourceNotFoundException(
                    resource_type=context.get("resource_type", "Resource"),
                    resource_id=context.get("resource_id", "Unknown"),
                    correlation_id=getattr(exc, "correlation_id", None),
                    cause=exc,
                )
            # Generic database error
            return ApplicationException(
                message="Database operation failed",
                code="DATABASE_ERROR",
                details={"original_error": str(exc)},
                correlation_id=getattr(exc, "correlation_id", None),
                cause=exc,
            )

        if isinstance(exc, ExternalServiceException):
            return ApplicationException(
                message=f"External service unavailable: {exc.service}",
                code="SERVICE_UNAVAILABLE",
                details=exc.details,
                correlation_id=exc.correlation_id,
                cause=exc,
            )

        if isinstance(exc, StreamingException):
            return ApplicationException(
                message="Streaming operation failed",
                code="STREAMING_ERROR",
                details=exc.details,
                correlation_id=exc.correlation_id,
                cause=exc,
            )

        if isinstance(exc, InfrastructureException):
            # Generic infrastructure error
            return ApplicationException(
                message="An infrastructure error occurred",
                code="INFRASTRUCTURE_ERROR",
                details=getattr(exc, "details", {"error": str(exc)}),
                correlation_id=getattr(exc, "correlation_id", None),
                cause=exc,
            )

        # Unknown infrastructure error
        return ApplicationException(
            message="An unexpected infrastructure error occurred",
            code="INFRASTRUCTURE_ERROR",
            details={"error": str(exc), "type": type(exc).__name__},
            cause=exc,
        )

    def create_error_response(self, exc: Exception, correlation_id: str) -> JSONResponse:
        """Create a structured error response from any exception.

        Args:
            exc: Exception to convert to response
            correlation_id: Request correlation ID

        Returns:
            JSON response with error details
        """
        if isinstance(exc, ApplicationException):
            status_code = self.app_to_api_map.get(type(exc), 500)
            content = exc.to_dict()
        elif isinstance(exc, DomainException):
            # Translate domain to application first
            app_exc = self.translate_domain_to_application(exc, correlation_id)
            status_code = self.app_to_api_map.get(type(app_exc), 500)
            content = app_exc.to_dict()
        elif isinstance(exc, InfrastructureException):
            # Translate infrastructure to application
            app_exc = self.translate_infrastructure_to_application(exc, {})
            status_code = self.app_to_api_map.get(type(app_exc), 500)
            content = app_exc.to_dict()
        else:
            # Unknown exception type
            status_code = 500
            content = {
                "error": {
                    "message": "An unexpected error occurred",
                    "code": "INTERNAL_ERROR",
                    "correlation_id": correlation_id,
                    "details": {"error": str(exc), "type": type(exc).__name__},
                }
            }

        return JSONResponse(status_code=status_code, content=content)

    # Specific translator methods
    def _document_too_large_to_app(
        self, exc: DocumentTooLargeError, correlation_id: str | None
    ) -> ApplicationException:
        """Translate document too large error to application exception."""
        return ValidationException(
            field="document",
            value=f"{exc.size} bytes",
            reason=f"Exceeds maximum size of {exc.max_size} bytes",
            correlation_id=correlation_id or exc.correlation_id,
            cause=exc,
        )

    def _invalid_state_to_app(
        self, exc: InvalidStateTransition, correlation_id: str | None
    ) -> ApplicationException:
        """Translate invalid state transition to application exception."""
        return ApplicationException(
            message=exc.message,
            code="INVALID_OPERATION",
            details=exc.details,
            correlation_id=correlation_id or exc.correlation_id,
            cause=exc,
        )

    def _strategy_error_to_app(
        self, exc: ChunkingStrategyError, correlation_id: str | None
    ) -> ApplicationException:
        """Translate strategy error to application exception."""
        return ApplicationException(
            message=f"Chunking failed: {exc.reason}",
            code="CHUNKING_FAILED",
            details=exc.details,
            correlation_id=correlation_id or exc.correlation_id,
            cause=exc,
        )

    def _default_domain_to_app(
        self, exc: DomainException, correlation_id: str | None
    ) -> ApplicationException:
        """Default translator for unmapped domain exceptions."""
        return ApplicationException(
            message=exc.message,
            code=exc.code,
            details=exc.details,
            correlation_id=correlation_id or exc.correlation_id,
            cause=exc,
        )


# Global translator instance
exception_translator = ExceptionTranslator()
