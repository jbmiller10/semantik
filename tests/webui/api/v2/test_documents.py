"""
Tests for v2 document content API endpoints.

Tests focus on security aspects including authentication, authorization,
and path traversal protection.
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from packages.shared.database.exceptions import EntityNotFoundError
from packages.shared.database.models import Collection, Document, DocumentStatus
from packages.webui.api.v2.documents import get_document_content


@pytest.fixture()
def mock_user():
    """Mock authenticated user."""
    return {"id": 1, "username": "testuser"}


@pytest.fixture()
def mock_collection():
    """Mock collection object."""
    collection = MagicMock(spec=Collection)
    collection.id = "123e4567-e89b-12d3-a456-426614174000"
    collection.name = "Test Collection"
    collection.owner_id = 1
    return collection


@pytest.fixture()
def mock_document():
    """Mock document object."""
    document = MagicMock(spec=Document)
    document.id = "456e7890-e89b-12d3-a456-426614174001"
    document.collection_id = "123e4567-e89b-12d3-a456-426614174000"
    document.file_name = "test_document.pdf"
    document.file_path = "/tmp/test_document.pdf"
    document.mime_type = "application/pdf"
    document.status = DocumentStatus.COMPLETED
    return document


@pytest.fixture()
def temp_file():
    """Create a temporary test file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdf", delete=False) as f:
        f.write("Test PDF content")
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


class TestGetDocumentContent:
    """Test document content retrieval endpoint."""

    @pytest.mark.asyncio()
    async def test_get_document_content_success(
        self, mock_user, mock_collection, mock_document, temp_file
    ):
        """Test successful document content retrieval."""
        # Update document to use temp file path
        mock_document.file_path = temp_file

        mock_db = AsyncMock(spec=AsyncSession)
        mock_document_repo = AsyncMock()
        mock_document_repo.get_by_id.return_value = mock_document

        with patch("packages.webui.api.v2.documents.create_document_repository", return_value=mock_document_repo):
            result = await get_document_content(
                collection_uuid=mock_collection.id,
                document_uuid=mock_document.id,
                collection=mock_collection,
                current_user=mock_user,
                db=mock_db,
            )

        # Verify result is a FileResponse
        assert isinstance(result, FileResponse)
        assert result.path == temp_file
        assert result.media_type == "application/pdf"
        assert result.filename == "test_document.pdf"
        assert result.headers["Content-Disposition"] == 'inline; filename="test_document.pdf"'

    @pytest.mark.asyncio()
    async def test_get_document_content_document_not_found(
        self, mock_user, mock_collection
    ):
        """Test 404 when document doesn't exist."""
        mock_db = AsyncMock(spec=AsyncSession)
        mock_document_repo = AsyncMock()
        mock_document_repo.get_by_id.side_effect = EntityNotFoundError("Document not found")

        with patch("packages.webui.api.v2.documents.create_document_repository", return_value=mock_document_repo):
            with pytest.raises(HTTPException) as exc_info:
                await get_document_content(
                    collection_uuid=mock_collection.id,
                    document_uuid="nonexistent-doc-id",
                    collection=mock_collection,
                    current_user=mock_user,
                    db=mock_db,
                )

        assert exc_info.value.status_code == 404
        assert "Document nonexistent-doc-id not found" in str(exc_info.value.detail)

    @pytest.mark.asyncio()
    async def test_get_document_content_cross_collection_access(
        self, mock_user, mock_collection, mock_document
    ):
        """Test 403 when document belongs to different collection."""
        # Set document to belong to a different collection
        mock_document.collection_id = "different-collection-id"

        mock_db = AsyncMock(spec=AsyncSession)
        mock_document_repo = AsyncMock()
        mock_document_repo.get_by_id.return_value = mock_document

        with patch("packages.webui.api.v2.documents.create_document_repository", return_value=mock_document_repo):
            with pytest.raises(HTTPException) as exc_info:
                await get_document_content(
                    collection_uuid=mock_collection.id,
                    document_uuid=mock_document.id,
                    collection=mock_collection,
                    current_user=mock_user,
                    db=mock_db,
                )

        assert exc_info.value.status_code == 403
        assert "Document does not belong to the specified collection" in str(exc_info.value.detail)

    @pytest.mark.asyncio()
    async def test_get_document_content_file_not_found(
        self, mock_user, mock_collection, mock_document
    ):
        """Test 404 when document file doesn't exist on disk."""
        # Use a non-existent file path
        mock_document.file_path = "/tmp/nonexistent_file.pdf"

        mock_db = AsyncMock(spec=AsyncSession)
        mock_document_repo = AsyncMock()
        mock_document_repo.get_by_id.return_value = mock_document

        with patch("packages.webui.api.v2.documents.create_document_repository", return_value=mock_document_repo):
            with pytest.raises(HTTPException) as exc_info:
                await get_document_content(
                    collection_uuid=mock_collection.id,
                    document_uuid=mock_document.id,
                    collection=mock_collection,
                    current_user=mock_user,
                    db=mock_db,
                )

        assert exc_info.value.status_code == 404
        assert "Document file not found on disk" in str(exc_info.value.detail)

    @pytest.mark.asyncio()
    async def test_get_document_content_path_is_directory(
        self, mock_user, mock_collection, mock_document
    ):
        """Test 500 when document path points to a directory."""
        # Use temp directory instead of file
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_document.file_path = temp_dir

            mock_db = AsyncMock(spec=AsyncSession)
            mock_document_repo = AsyncMock()
            mock_document_repo.get_by_id.return_value = mock_document

            with patch("packages.webui.api.v2.documents.create_document_repository", return_value=mock_document_repo):
                with pytest.raises(HTTPException) as exc_info:
                    await get_document_content(
                        collection_uuid=mock_collection.id,
                        document_uuid=mock_document.id,
                        collection=mock_collection,
                        current_user=mock_user,
                        db=mock_db,
                    )

            assert exc_info.value.status_code == 500
            assert "Invalid document path" in str(exc_info.value.detail)

    @pytest.mark.asyncio()
    async def test_get_document_content_path_traversal_attempt(
        self, mock_user, mock_collection, mock_document
    ):
        """Test that path traversal attempts are properly handled."""
        # Try various path traversal patterns
        path_traversal_attempts = [
            "../../../etc/passwd",
            "../../sensitive_file.txt",
            "/etc/../etc/passwd",
            "test/../../etc/hosts",
        ]

        mock_db = AsyncMock(spec=AsyncSession)
        mock_document_repo = AsyncMock()

        for malicious_path in path_traversal_attempts:
            mock_document.file_path = malicious_path
            mock_document_repo.get_by_id.return_value = mock_document

            with patch("packages.webui.api.v2.documents.create_document_repository", return_value=mock_document_repo):
                # The endpoint should either:
                # 1. Return 404 if the resolved path doesn't exist
                # 2. Return 500 for invalid paths
                # But should never serve files outside allowed directories
                with pytest.raises(HTTPException) as exc_info:
                    await get_document_content(
                        collection_uuid=mock_collection.id,
                        document_uuid=mock_document.id,
                        collection=mock_collection,
                        current_user=mock_user,
                        db=mock_db,
                    )

                # Should get 404 (file not found) or 500 (error), never 200
                assert exc_info.value.status_code in (404, 500)

    @pytest.mark.asyncio()
    async def test_get_document_content_default_mime_type(
        self, mock_user, mock_collection, mock_document, temp_file
    ):
        """Test that default mime type is used when document has no mime type."""
        mock_document.file_path = temp_file
        mock_document.mime_type = None  # No mime type set

        mock_db = AsyncMock(spec=AsyncSession)
        mock_document_repo = AsyncMock()
        mock_document_repo.get_by_id.return_value = mock_document

        with patch("packages.webui.api.v2.documents.create_document_repository", return_value=mock_document_repo):
            result = await get_document_content(
                collection_uuid=mock_collection.id,
                document_uuid=mock_document.id,
                collection=mock_collection,
                current_user=mock_user,
                db=mock_db,
            )

        # Should use default octet-stream mime type
        assert result.media_type == "application/octet-stream"

    @pytest.mark.asyncio()
    async def test_get_document_content_cache_headers(
        self, mock_user, mock_collection, mock_document, temp_file
    ):
        """Test that appropriate cache headers are set."""
        mock_document.file_path = temp_file

        mock_db = AsyncMock(spec=AsyncSession)
        mock_document_repo = AsyncMock()
        mock_document_repo.get_by_id.return_value = mock_document

        with patch("packages.webui.api.v2.documents.create_document_repository", return_value=mock_document_repo):
            result = await get_document_content(
                collection_uuid=mock_collection.id,
                document_uuid=mock_document.id,
                collection=mock_collection,
                current_user=mock_user,
                db=mock_db,
            )

        # Check cache control header
        assert result.headers["Cache-Control"] == "private, max-age=3600"

