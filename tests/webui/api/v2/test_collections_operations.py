"""
Tests for v2 collections operations and documents API endpoints.

This module provides tests for listing operations and documents within collections.
"""

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from packages.shared.database.exceptions import AccessDeniedError, EntityNotFoundError
from packages.shared.database.models import Document, DocumentStatus, Operation, OperationStatus, OperationType
from packages.webui.api.schemas import DocumentListResponse, OperationResponse
from packages.webui.api.v2.collections import list_collection_documents, list_collection_operations
from packages.webui.services.collection_service import CollectionService


@pytest.fixture()
def mock_user() -> dict[str, Any]:
    """Mock authenticated user."""
    return {"id": 1, "username": "testuser"}


@pytest.fixture()
def mock_collection_service() -> AsyncMock:
    """Mock CollectionService."""
    return AsyncMock(spec=CollectionService)


@pytest.fixture()
def mock_request() -> MagicMock:
    """Mock FastAPI Request object."""
    request = MagicMock()
    request.headers = {}
    request.client = MagicMock()
    request.client.host = "127.0.0.1"
    return request


class TestListCollectionOperations:
    """Test list_collection_operations endpoint."""

    @pytest.mark.asyncio()
    async def test_list_operations_success(
        self,
        mock_user: dict[str, Any],
        mock_collection_service: AsyncMock) -> None:
        """Test successful operation listing."""
        collection_uuid = "123e4567-e89b-12d3-a456-426614174000"

        # Create mock operations with proper collection_id
        operations = []
        for i in range(3):
            op = MagicMock(spec=Operation)
            op.uuid = f"op-{i}"
            op.collection_id = collection_uuid  # Fix: Use proper string value
            op.type = OperationType.INDEX
            op.status = OperationStatus.COMPLETED
            op.config = {}
            op.error_message = None
            op.created_at = datetime.now(UTC)
            op.started_at = datetime.now(UTC)
            op.completed_at = datetime.now(UTC)
            operations.append(op)

        mock_collection_service.list_operations.return_value = (operations, 3)

        # Execute
        result = await list_collection_operations(
            collection_uuid=collection_uuid,
            status=None,
            operation_type=None,
            page=1,
            per_page=50,
            current_user=mock_user,
            service=mock_collection_service)

        # Verify
        assert len(result) == 3
        assert all(isinstance(r, OperationResponse) for r in result)
        assert result[0].collection_id == collection_uuid
        assert result[1].collection_id == collection_uuid
        assert result[2].collection_id == collection_uuid

    @pytest.mark.asyncio()
    async def test_list_operations_with_filters(
        self,
        mock_user: dict[str, Any],
        mock_collection_service: AsyncMock) -> None:
        """Test listing operations with status and type filters."""
        collection_uuid = "123e4567-e89b-12d3-a456-426614174000"

        # Create mock operations
        operation1 = MagicMock(spec=Operation)
        operation1.uuid = "op-1"
        operation1.collection_id = collection_uuid  # Fix: Use proper string value
        operation1.type = OperationType.INDEX
        operation1.status = OperationStatus.COMPLETED
        operation1.config = {}
        operation1.error_message = None
        operation1.created_at = datetime.now(UTC)
        operation1.started_at = datetime.now(UTC)
        operation1.completed_at = datetime.now(UTC)

        operation2 = MagicMock(spec=Operation)
        operation2.uuid = "op-2"
        operation2.collection_id = collection_uuid  # Fix: Use proper string value
        operation2.type = OperationType.REINDEX
        operation2.status = OperationStatus.PROCESSING
        operation2.config = {}
        operation2.error_message = None
        operation2.created_at = datetime.now(UTC)
        operation2.started_at = datetime.now(UTC)
        operation2.completed_at = None

        mock_collection_service.list_operations.return_value = ([operation1, operation2], 2)

        # Test with status filter
        result = await list_collection_operations(
            collection_uuid=collection_uuid,
            status="completed",
            operation_type=None,
            page=1,
            per_page=50,
            current_user=mock_user,
            service=mock_collection_service)

        assert len(result) == 1
        assert result[0].status == "completed"

        # Test with type filter
        result = await list_collection_operations(
            collection_uuid=collection_uuid,
            status=None,
            operation_type="reindex",
            page=1,
            per_page=50,
            current_user=mock_user,
            service=mock_collection_service)

        assert len(result) == 1
        assert result[0].type == "reindex"

    @pytest.mark.asyncio()
    async def test_list_operations_invalid_status(
        self,
        mock_user: dict[str, Any],
        mock_collection_service: AsyncMock) -> None:
        """Test listing operations with invalid status."""
        collection_uuid = "123e4567-e89b-12d3-a456-426614174000"

        # Execute & Verify
        with pytest.raises(HTTPException) as exc_info:
            await list_collection_operations(
                collection_uuid=collection_uuid,
                status="invalid_status",
                operation_type=None,
                page=1,
                per_page=50,
                current_user=mock_user,
                service=mock_collection_service)

        assert exc_info.value.status_code == 400
        assert "Invalid status: invalid_status" in str(exc_info.value.detail)

    @pytest.mark.asyncio()
    async def test_list_operations_invalid_type(
        self,
        mock_user: dict[str, Any],
        mock_collection_service: AsyncMock) -> None:
        """Test listing operations with invalid type."""
        collection_uuid = "123e4567-e89b-12d3-a456-426614174000"

        # Execute & Verify
        with pytest.raises(HTTPException) as exc_info:
            await list_collection_operations(
                collection_uuid=collection_uuid,
                status=None,
                operation_type="invalid_type",
                page=1,
                per_page=50,
                current_user=mock_user,
                service=mock_collection_service)

        assert exc_info.value.status_code == 400
        assert "Invalid operation type: invalid_type" in str(exc_info.value.detail)

    @pytest.mark.asyncio()
    async def test_list_operations_collection_not_found(
        self,
        mock_user: dict[str, Any],
        mock_collection_service: AsyncMock) -> None:
        """Test 404 error when collection not found."""
        collection_uuid = "non-existent-uuid"

        mock_collection_service.list_operations.side_effect = EntityNotFoundError("Collection", collection_uuid)

        with pytest.raises(HTTPException) as exc_info:
            await list_collection_operations(
                collection_uuid=collection_uuid,
                status=None,
                operation_type=None,
                page=1,
                per_page=50,
                current_user=mock_user,
                service=mock_collection_service)

        assert exc_info.value.status_code == 404
        assert f"Collection '{collection_uuid}' not found" in str(exc_info.value.detail)

    @pytest.mark.asyncio()
    async def test_list_operations_access_denied(
        self,
        mock_user: dict[str, Any],
        mock_collection_service: AsyncMock) -> None:
        """Test 403 error when user lacks access."""
        collection_uuid = "123e4567-e89b-12d3-a456-426614174000"

        mock_collection_service.list_operations.side_effect = AccessDeniedError(
            str(mock_user["id"]), "collection", collection_uuid
        )

        with pytest.raises(HTTPException) as exc_info:
            await list_collection_operations(
                collection_uuid=collection_uuid,
                status=None,
                operation_type=None,
                page=1,
                per_page=50,
                current_user=mock_user,
                service=mock_collection_service)

        assert exc_info.value.status_code == 403
        assert "You don't have access to this collection" in str(exc_info.value.detail)


class TestListCollectionDocuments:
    """Test list_collection_documents endpoint."""

    @pytest.mark.asyncio()
    async def test_list_documents_success(
        self,
        mock_user: dict[str, Any],
        mock_collection_service: AsyncMock) -> None:
        """Test successful document listing."""
        collection_uuid = "123e4567-e89b-12d3-a456-426614174000"

        # Create mock documents
        documents = []
        for i in range(3):
            doc = MagicMock(spec=Document)
            doc.id = f"doc-{i+1}"
            doc.collection_id = collection_uuid
            doc.file_name = f"document{i}.txt"
            doc.file_path = f"/data/document{i}.txt"
            doc.file_size = 1024 * (i + 1)
            doc.mime_type = "text/plain"
            doc.content_hash = f"hash{i}"
            doc.status = DocumentStatus.COMPLETED
            doc.error_message = None
            doc.chunk_count = 10 * (i + 1)
            doc.meta = {"test": f"metadata{i}"}
            doc.created_at = datetime.now(UTC)
            doc.updated_at = datetime.now(UTC)
            documents.append(doc)

        mock_collection_service.list_documents.return_value = (documents, 3)

        # Execute
        result = await list_collection_documents(
            collection_uuid=collection_uuid,
            page=1,
            per_page=50,
            status=None,
            current_user=mock_user,
            service=mock_collection_service)

        # Verify
        assert isinstance(result, DocumentListResponse)
        assert len(result.documents) == 3
        assert result.total == 3
        assert result.page == 1
        assert result.per_page == 50

    @pytest.mark.asyncio()
    async def test_list_documents_with_status_filter(
        self,
        mock_user: dict[str, Any],
        mock_collection_service: AsyncMock) -> None:
        """Test listing documents with status filter."""
        collection_uuid = "123e4567-e89b-12d3-a456-426614174000"

        # Create mock documents with different statuses
        doc1 = MagicMock(spec=Document)
        doc1.id = "doc-1"
        doc1.collection_id = collection_uuid
        doc1.file_name = "document1.txt"
        doc1.file_path = "/data/document1.txt"
        doc1.file_size = 1024
        doc1.mime_type = "text/plain"
        doc1.content_hash = "hash1"
        doc1.status = DocumentStatus.COMPLETED
        doc1.error_message = None
        doc1.chunk_count = 10
        doc1.meta = {}
        doc1.created_at = datetime.now(UTC)
        doc1.updated_at = datetime.now(UTC)

        doc2 = MagicMock(spec=Document)
        doc2.id = "doc-2"
        doc2.collection_id = collection_uuid
        doc2.file_name = "document2.txt"
        doc2.file_path = "/data/document2.txt"
        doc2.file_size = 2048
        doc2.mime_type = "text/plain"
        doc2.content_hash = "hash2"
        doc2.status = DocumentStatus.PENDING
        doc2.error_message = None
        doc2.chunk_count = 0
        doc2.meta = {}
        doc2.created_at = datetime.now(UTC)
        doc2.updated_at = datetime.now(UTC)

        mock_collection_service.list_documents.return_value = ([doc1, doc2], 2)

        # Test with status filter
        result = await list_collection_documents(
            collection_uuid=collection_uuid,
            page=1,
            per_page=50,
            status="completed",
            current_user=mock_user,
            service=mock_collection_service)

        assert len(result.documents) == 1
        assert result.documents[0].status == "completed"
        assert result.total == 1  # Total should be updated after filtering

    @pytest.mark.asyncio()
    async def test_list_documents_invalid_status(
        self,
        mock_user: dict[str, Any],
        mock_collection_service: AsyncMock) -> None:
        """Test listing documents with invalid status."""
        collection_uuid = "123e4567-e89b-12d3-a456-426614174000"

        # Execute & Verify
        with pytest.raises(HTTPException) as exc_info:
            await list_collection_documents(
                collection_uuid=collection_uuid,
                page=1,
                per_page=50,
                status="invalid_status",
                current_user=mock_user,
                service=mock_collection_service)

        assert exc_info.value.status_code == 400
        assert "Invalid status: invalid_status" in str(exc_info.value.detail)

    @pytest.mark.asyncio()
    async def test_list_documents_collection_not_found(
        self,
        mock_user: dict[str, Any],
        mock_collection_service: AsyncMock) -> None:
        """Test 404 error when collection not found."""
        collection_uuid = "non-existent-uuid"

        mock_collection_service.list_documents.side_effect = EntityNotFoundError("Collection", collection_uuid)

        with pytest.raises(HTTPException) as exc_info:
            await list_collection_documents(
                collection_uuid=collection_uuid,
                page=1,
                per_page=50,
                status=None,
                current_user=mock_user,
                service=mock_collection_service)

        assert exc_info.value.status_code == 404
        assert f"Collection '{collection_uuid}' not found" in str(exc_info.value.detail)

    @pytest.mark.asyncio()
    async def test_list_documents_access_denied(
        self,
        mock_user: dict[str, Any],
        mock_collection_service: AsyncMock) -> None:
        """Test 403 error when user lacks access."""
        collection_uuid = "123e4567-e89b-12d3-a456-426614174000"

        mock_collection_service.list_documents.side_effect = AccessDeniedError(
            str(mock_user["id"]), "collection", collection_uuid
        )

        with pytest.raises(HTTPException) as exc_info:
            await list_collection_documents(
                collection_uuid=collection_uuid,
                page=1,
                per_page=50,
                status=None,
                current_user=mock_user,
                service=mock_collection_service)

        assert exc_info.value.status_code == 403
        assert "You don't have access to this collection" in str(exc_info.value.detail)

    @pytest.mark.asyncio()
    async def test_list_documents_pagination(
        self,
        mock_user: dict[str, Any],
        mock_collection_service: AsyncMock) -> None:
        """Test document listing with pagination."""
        collection_uuid = "123e4567-e89b-12d3-a456-426614174000"

        # Return empty list for page 2
        mock_collection_service.list_documents.return_value = ([], 50)

        # Execute
        result = await list_collection_documents(
            collection_uuid=collection_uuid,
            page=2,
            per_page=20,
            status=None,
            current_user=mock_user,
            service=mock_collection_service)

        # Verify
        assert len(result.documents) == 0
        assert result.total == 50
        assert result.page == 2
        assert result.per_page == 20

        # Verify offset calculation
        mock_collection_service.list_documents.assert_called_once_with(
            collection_id=collection_uuid,
            user_id=1,
            offset=20,  # (page-1) * per_page = (2-1) * 20
            limit=20)
