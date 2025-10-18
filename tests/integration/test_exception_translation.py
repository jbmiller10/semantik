#!/usr/bin/env python3
"""
Integration tests for exception translation layer.

Tests the proper translation and context preservation of exceptions
across architectural layers.
"""

import uuid
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from packages.shared.chunking.infrastructure.exception_translator import ExceptionTranslator
from packages.shared.chunking.infrastructure.exceptions import (
    ApplicationException,
    ChunkingStrategyError,
    DatabaseException,
    DocumentTooLargeError,
    DomainException,
    ExternalServiceException,
    InvalidStateTransition,
    PermissionDeniedException,
    ResourceNotFoundException,
    ValidationException,
)


class TestExceptionTranslator:
    """Test exception translator functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.translator = ExceptionTranslator()
        self.correlation_id = str(uuid.uuid4())

    def test_document_too_large_translation(self):
        """Test translation of DocumentTooLargeError to ApplicationException."""
        # Create domain exception
        domain_exc = DocumentTooLargeError(
            size=20_000_000,
            max_size=10_000_000,
            correlation_id=self.correlation_id,
        )

        # Translate to application exception
        app_exc = self.translator.translate_domain_to_application(
            domain_exc,
            self.correlation_id,
        )

        # Verify translation
        assert isinstance(app_exc, ValidationException)
        assert app_exc.field == "document"
        assert "20000000 bytes" in app_exc.value
        assert "Exceeds maximum size" in app_exc.reason
        assert app_exc.correlation_id == self.correlation_id
        assert app_exc.cause == domain_exc

    def test_invalid_state_transition_translation(self):
        """Test translation of InvalidStateTransition to ApplicationException."""
        # Create domain exception
        domain_exc = InvalidStateTransition(
            current_state="processing",
            attempted_state="completed",
            correlation_id=self.correlation_id,
        )

        # Translate to application exception
        app_exc = self.translator.translate_domain_to_application(
            domain_exc,
            self.correlation_id,
        )

        # Verify translation
        assert isinstance(app_exc, ApplicationException)
        assert app_exc.code == "INVALID_OPERATION"
        assert app_exc.details["current"] == "processing"
        assert app_exc.details["attempted"] == "completed"
        assert app_exc.correlation_id == self.correlation_id
        assert app_exc.cause == domain_exc

    def test_chunking_strategy_error_translation(self):
        """Test translation of ChunkingStrategyError to ApplicationException."""
        # Create domain exception
        domain_exc = ChunkingStrategyError(
            strategy="semantic",
            reason="Embedding service unavailable",
            correlation_id=self.correlation_id,
        )

        # Translate to application exception
        app_exc = self.translator.translate_domain_to_application(
            domain_exc,
            self.correlation_id,
        )

        # Verify translation
        assert isinstance(app_exc, ApplicationException)
        assert app_exc.code == "CHUNKING_FAILED"
        assert "Chunking failed" in app_exc.message
        assert app_exc.details["strategy"] == "semantic"
        assert app_exc.correlation_id == self.correlation_id
        assert app_exc.cause == domain_exc

    def test_application_to_api_translation(self):
        """Test translation of ApplicationException to HTTPException."""
        # Create application exception
        app_exc = ValidationException(
            field="chunk_size",
            value="invalid",
            reason="Must be a positive integer",
            correlation_id=self.correlation_id,
        )

        # Translate to HTTP exception
        http_exc = self.translator.translate_application_to_api(app_exc)

        # Verify translation
        assert isinstance(http_exc, HTTPException)
        assert http_exc.status_code == 400
        assert http_exc.detail["error"]["code"] == "VALIDATION_ERROR"
        assert http_exc.detail["error"]["correlation_id"] == self.correlation_id
        assert http_exc.detail["error"]["details"]["field"] == "chunk_size"

    def test_resource_not_found_translation(self):
        """Test translation of ResourceNotFoundException to HTTP 404."""
        # Create application exception
        app_exc = ResourceNotFoundException(
            resource_type="Document",
            resource_id="doc-123",
            correlation_id=self.correlation_id,
        )

        # Translate to HTTP exception
        http_exc = self.translator.translate_application_to_api(app_exc)

        # Verify translation
        assert http_exc.status_code == 404
        assert http_exc.detail["error"]["code"] == "RESOURCE_NOT_FOUND"

    def test_permission_denied_translation(self):
        """Test translation of PermissionDeniedException to HTTP 403."""
        # Create application exception
        app_exc = PermissionDeniedException(
            user_id="user-123",
            resource="collection:456",
            action="write",
            correlation_id=self.correlation_id,
        )

        # Translate to HTTP exception
        http_exc = self.translator.translate_application_to_api(app_exc)

        # Verify translation
        assert http_exc.status_code == 403
        assert http_exc.detail["error"]["code"] == "PERMISSION_DENIED"

    def test_database_exception_to_application(self):
        """Test translation of DatabaseException to ApplicationException."""
        # Test not found case
        db_exc = DatabaseException(
            operation="SELECT",
            table="documents",
            error="Record does not exist",
            correlation_id=self.correlation_id,
        )

        app_exc = self.translator.translate_infrastructure_to_application(
            db_exc,
            {"resource_type": "Document", "resource_id": "doc-123"},
        )

        assert isinstance(app_exc, ResourceNotFoundException)
        assert app_exc.resource_type == "Document"
        assert app_exc.resource_id == "doc-123"
        assert app_exc.cause == db_exc

        # Test generic database error
        db_exc2 = DatabaseException(
            operation="INSERT",
            table="chunks",
            error="Connection timeout",
            correlation_id=self.correlation_id,
        )

        app_exc2 = self.translator.translate_infrastructure_to_application(
            db_exc2,
            {},
        )

        assert isinstance(app_exc2, ApplicationException)
        assert app_exc2.code == "DATABASE_ERROR"
        assert app_exc2.cause == db_exc2

    def test_external_service_exception_translation(self):
        """Test translation of ExternalServiceException to ApplicationException."""
        ext_exc = ExternalServiceException(
            service="embedding-api",
            operation="generate_embeddings",
            error="Service unavailable",
            correlation_id=self.correlation_id,
        )

        app_exc = self.translator.translate_infrastructure_to_application(
            ext_exc,
            {},
        )

        assert isinstance(app_exc, ApplicationException)
        assert app_exc.code == "SERVICE_UNAVAILABLE"
        assert "embedding-api" in app_exc.message
        assert app_exc.cause == ext_exc

    def test_context_preservation(self):
        """Test that exception context is properly preserved through translation."""
        # Create a chain of exceptions
        original_error = ValueError("Original error")

        domain_exc = DocumentTooLargeError(
            size=1000,
            max_size=500,
            correlation_id=self.correlation_id,
            cause=original_error,
        )

        app_exc = self.translator.translate_domain_to_application(
            domain_exc,
            self.correlation_id,
        )

        # Verify chain is preserved
        assert app_exc.cause == domain_exc
        assert domain_exc.cause == original_error
        assert app_exc.correlation_id == self.correlation_id

        # Verify to_dict includes context
        exc_dict = app_exc.to_dict()
        assert exc_dict["error"]["correlation_id"] == self.correlation_id
        assert exc_dict["error"]["code"] == "VALIDATION_ERROR"
        assert exc_dict["error"]["cause"] is not None

    def test_unmapped_exception_handling(self):
        """Test handling of unmapped domain exceptions."""
        # Create custom domain exception
        custom_exc = DomainException(
            message="Custom domain error",
            code="CUSTOM_ERROR",
            details={"custom": "data"},
            correlation_id=self.correlation_id,
        )

        # Translate using default handler
        app_exc = self.translator.translate_domain_to_application(
            custom_exc,
            self.correlation_id,
        )

        # Verify default translation
        assert isinstance(app_exc, ApplicationException)
        assert app_exc.message == custom_exc.message
        assert app_exc.code == custom_exc.code
        assert app_exc.details == custom_exc.details
        assert app_exc.correlation_id == self.correlation_id
        assert app_exc.cause == custom_exc

    def test_create_error_response(self):
        """Test creation of error response from various exception types."""
        # Test with ApplicationException
        app_exc = ValidationException(
            field="test",
            value="invalid",
            reason="Test reason",
            correlation_id=self.correlation_id,
        )

        response = self.translator.create_error_response(
            app_exc,
            self.correlation_id,
        )

        assert response.status_code == 400
        content = response.body.decode()
        assert self.correlation_id in content
        assert "VALIDATION_ERROR" in content

        # Test with DomainException
        domain_exc = DocumentTooLargeError(
            size=1000,
            max_size=500,
            correlation_id=self.correlation_id,
        )

        response = self.translator.create_error_response(
            domain_exc,
            self.correlation_id,
        )

        assert response.status_code == 400  # ValidationException maps to 400
        content = response.body.decode()
        assert self.correlation_id in content

        # Test with unknown exception
        unknown_exc = RuntimeError("Unknown error")

        response = self.translator.create_error_response(
            unknown_exc,
            self.correlation_id,
        )

        assert response.status_code == 500
        content = response.body.decode()
        assert "INTERNAL_ERROR" in content
        assert self.correlation_id in content


@pytest.mark.asyncio()
class TestServiceExceptionHandling:
    """Test exception handling in the service layer."""

    async def test_preview_chunks_with_validation_error(self):
        """Test preview_chunks handles validation errors properly."""
        from packages.webui.services.chunking_service import ChunkingService

        # Mock dependencies
        mock_db = AsyncMock()
        mock_collection_repo = AsyncMock()
        mock_document_repo = AsyncMock()
        mock_redis = AsyncMock()

        service = ChunkingService(
            db_session=mock_db,
            collection_repo=mock_collection_repo,
            document_repo=mock_document_repo,
            redis_client=mock_redis,
        )

        # Test missing input
        with pytest.raises(ApplicationException) as exc_info:
            await service.preview_chunks(
                strategy="recursive",
                content=None,
                document_id=None,
            )

        assert exc_info.value.code == "VALIDATION_ERROR"
        assert "Either content or document_id must be provided" in exc_info.value.reason

    async def test_preview_chunks_with_document_too_large(self):
        """Test preview_chunks handles document size limits."""
        from packages.webui.services.chunking_service import ChunkingService

        # Mock dependencies
        mock_db = AsyncMock()
        mock_collection_repo = AsyncMock()
        mock_document_repo = AsyncMock()
        mock_redis = AsyncMock()

        service = ChunkingService(
            db_session=mock_db,
            collection_repo=mock_collection_repo,
            document_repo=mock_document_repo,
            redis_client=mock_redis,
        )

        # Test with oversized content
        large_content = "x" * 11_000_000  # 11MB

        with pytest.raises(DocumentTooLargeError) as exc_info:
            await service.preview_chunks(
                strategy="recursive",
                content=large_content,
            )

        assert exc_info.value.size == len(large_content)
        assert exc_info.value.max_size == 10_000_000

    async def test_preview_chunks_with_permission_denied(self):
        """Test preview_chunks handles permission errors."""
        from packages.webui.services.chunking_service import ChunkingService

        # Mock dependencies
        mock_db = AsyncMock()
        mock_collection_repo = AsyncMock()
        mock_document_repo = AsyncMock()
        mock_redis = AsyncMock()

        service = ChunkingService(
            db_session=mock_db,
            collection_repo=mock_collection_repo,
            document_repo=mock_document_repo,
            redis_client=mock_redis,
        )

        # Test document access without user_id
        with pytest.raises(PermissionDeniedException) as exc_info:
            await service.preview_chunks(
                strategy="recursive",
                document_id="doc-123",
                user_id=None,
            )

        assert exc_info.value.resource == "document:doc-123"
        assert exc_info.value.action == "read"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
