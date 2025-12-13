"""Security and behaviour tests for the v2 document content endpoint."""

from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from shared.config import settings
from shared.database.exceptions import EntityNotFoundError
from shared.database.models import Collection, Document, DocumentStatus
from webui.api.v2.documents import get_document_content


@pytest.fixture()
def mock_artifact_repo() -> AsyncMock:
    """Mock artifact repository that returns None (no artifact, use file)."""
    repo = AsyncMock()
    repo.get_content.return_value = None
    return repo


@pytest.fixture()
def mock_user() -> dict[str, Any]:
    """Mock authenticated user."""

    return {"id": 1, "username": "testuser"}


@pytest.fixture()
def mock_collection() -> MagicMock:
    """Mock collection object."""

    collection = MagicMock(spec=Collection)
    collection.id = "123e4567-e89b-12d3-a456-426614174000"
    collection.name = "Test Collection"
    collection.owner_id = 1
    return collection


@pytest.fixture()
def document_root(tmp_path, monkeypatch) -> Path:
    """Provision an isolated document root for each test."""

    root = tmp_path / "documents"
    root.mkdir()
    monkeypatch.setattr(settings, "_document_root", root, raising=False)
    monkeypatch.setattr(settings, "_document_allowed_roots", (), raising=False)
    return root


@pytest.fixture()
def mock_document() -> MagicMock:
    """Mock document model with a default path inside the root."""

    document = MagicMock(spec=Document)
    document.id = "456e7890-e89b-12d3-a456-426614174001"
    document.collection_id = "123e4567-e89b-12d3-a456-426614174000"
    document.file_name = "test_document.pdf"
    document.file_path = "/tmp/test_document.pdf"
    document.mime_type = "application/pdf"
    document.status = DocumentStatus.COMPLETED
    return document


@pytest.fixture()
def temp_file(document_root: Path) -> Generator[Path, None, None]:
    """Create a temporary test file inside the document root."""

    file_path = document_root / "safe" / "test_document.pdf"
    file_path.parent.mkdir(exist_ok=True)
    file_path.write_text("Test PDF content")

    yield file_path

    file_path.unlink(missing_ok=True)


class TestGetDocumentContent:
    """Test document content retrieval endpoint."""

    @pytest.mark.asyncio()
    async def test_get_document_content_success(
        self,
        mock_user: dict[str, Any],
        mock_collection: MagicMock,
        mock_document: MagicMock,
        mock_artifact_repo: AsyncMock,
        temp_file: Path,
    ) -> None:
        """Test successful document content retrieval."""
        mock_document.file_path = str(temp_file)
        mock_document.file_name = temp_file.name

        mock_db = AsyncMock(spec=AsyncSession)
        mock_document_repo = AsyncMock()
        mock_document_repo.get_by_id.return_value = mock_document

        with (
            patch("webui.api.v2.documents.create_document_repository", return_value=mock_document_repo),
            patch("webui.api.v2.documents.DocumentArtifactRepository", return_value=mock_artifact_repo),
        ):
            result = await get_document_content(
                collection_uuid=mock_collection.id,
                document_uuid=mock_document.id,
                collection=mock_collection,
                current_user=mock_user,
                db=mock_db,
            )

        assert isinstance(result, FileResponse)
        assert result.path == str(temp_file)
        assert result.media_type == "application/pdf"
        assert result.filename == temp_file.name
        assert result.headers["Content-Disposition"] == f'inline; filename="{temp_file.name}"'

    @pytest.mark.asyncio()
    async def test_get_document_content_document_not_found(
        self, mock_user: dict[str, Any], mock_collection: MagicMock, document_root: Path
    ) -> None:
        """Test 404 when document doesn't exist."""
        mock_db = AsyncMock(spec=AsyncSession)
        mock_document_repo = AsyncMock()
        mock_document_repo.get_by_id.side_effect = EntityNotFoundError("Document", "nonexistent-doc-id")

        with (
            patch("webui.api.v2.documents.create_document_repository", return_value=mock_document_repo),
            pytest.raises(HTTPException) as exc_info,
        ):
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
        self,
        mock_user: dict[str, Any],
        mock_collection: MagicMock,
        mock_document: MagicMock,
    ) -> None:
        """Test 403 when document belongs to different collection."""
        # Set document to belong to a different collection
        mock_document.collection_id = "different-collection-id"

        mock_db = AsyncMock(spec=AsyncSession)
        mock_document_repo = AsyncMock()
        mock_document_repo.get_by_id.return_value = mock_document

        with (
            patch("webui.api.v2.documents.create_document_repository", return_value=mock_document_repo),
            pytest.raises(HTTPException) as exc_info,
        ):
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
        self,
        mock_user: dict[str, Any],
        mock_collection: MagicMock,
        mock_document: MagicMock,
        mock_artifact_repo: AsyncMock,
        document_root: Path,
    ) -> None:
        """Test 404 when document file doesn't exist on disk."""
        mock_document.file_path = str(document_root / "nonexistent_file.pdf")

        mock_db = AsyncMock(spec=AsyncSession)
        mock_document_repo = AsyncMock()
        mock_document_repo.get_by_id.return_value = mock_document

        with (
            patch("webui.api.v2.documents.create_document_repository", return_value=mock_document_repo),
            patch("webui.api.v2.documents.DocumentArtifactRepository", return_value=mock_artifact_repo),
            pytest.raises(HTTPException) as exc_info,
        ):
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
        self,
        mock_user: dict[str, Any],
        mock_collection: MagicMock,
        mock_document: MagicMock,
        mock_artifact_repo: AsyncMock,
        document_root: Path,
    ) -> None:
        """Test 400 when document path points to a directory."""
        subdir = document_root / "nested"
        subdir.mkdir()
        mock_document.file_path = str(subdir)

        mock_db = AsyncMock(spec=AsyncSession)
        mock_document_repo = AsyncMock()
        mock_document_repo.get_by_id.return_value = mock_document

        with (
            patch("webui.api.v2.documents.create_document_repository", return_value=mock_document_repo),
            patch("webui.api.v2.documents.DocumentArtifactRepository", return_value=mock_artifact_repo),
            pytest.raises(HTTPException) as exc_info,
        ):
            await get_document_content(
                collection_uuid=mock_collection.id,
                document_uuid=mock_document.id,
                collection=mock_collection,
                current_user=mock_user,
                db=mock_db,
            )

        assert exc_info.value.status_code == 400
        assert "Invalid document path" in str(exc_info.value.detail)

    @pytest.mark.asyncio()
    async def test_get_document_content_path_traversal_blocked(
        self,
        mock_user: dict[str, Any],
        mock_collection: MagicMock,
        mock_document: MagicMock,
        mock_artifact_repo: AsyncMock,
        document_root: Path,
    ) -> None:
        """Ensure attempts to escape the document root are rejected."""

        outside_file = (document_root.parent / "outside.txt").resolve()
        outside_file.write_text("classified")

        traversal_paths = [
            str(outside_file),
            "../../etc/passwd",
            str(document_root / ".." / outside_file.name),
        ]

        mock_db = AsyncMock(spec=AsyncSession)
        mock_document_repo = AsyncMock()

        for malicious_path in traversal_paths:
            mock_document.file_path = malicious_path
            mock_document_repo.get_by_id.return_value = mock_document

            with (
                patch("webui.api.v2.documents.create_document_repository", return_value=mock_document_repo),
                patch("webui.api.v2.documents.DocumentArtifactRepository", return_value=mock_artifact_repo),
                pytest.raises(HTTPException) as exc_info,
            ):
                await get_document_content(
                    collection_uuid=mock_collection.id,
                    document_uuid=mock_document.id,
                    collection=mock_collection,
                    current_user=mock_user,
                    db=mock_db,
                )

            assert exc_info.value.status_code == 403
            assert "Access to the requested document is forbidden" in str(exc_info.value.detail)

        outside_file.unlink(missing_ok=True)

    @pytest.mark.asyncio()
    async def test_get_document_content_without_document_restrictions_allows_outside_paths(
        self,
        mock_user: dict[str, Any],
        mock_collection: MagicMock,
        mock_document: MagicMock,
        mock_artifact_repo: AsyncMock,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        """When no roots are configured and no default mount exists, allow legacy paths."""

        monkeypatch.setattr(settings, "_document_root", None, raising=False)
        monkeypatch.setattr(settings, "_document_allowed_roots", (), raising=False)
        monkeypatch.setattr(settings, "_default_document_mounts", (), raising=False)

        outside_file = tmp_path / "legacy.pdf"
        outside_file.write_text("legacy content")

        mock_document.file_path = str(outside_file)
        mock_document.file_name = outside_file.name

        mock_db = AsyncMock(spec=AsyncSession)
        mock_document_repo = AsyncMock()
        mock_document_repo.get_by_id.return_value = mock_document

        with (
            patch("webui.api.v2.documents.create_document_repository", return_value=mock_document_repo),
            patch("webui.api.v2.documents.DocumentArtifactRepository", return_value=mock_artifact_repo),
        ):
            result = await get_document_content(
                collection_uuid=mock_collection.id,
                document_uuid=mock_document.id,
                collection=mock_collection,
                current_user=mock_user,
                db=mock_db,
            )

        assert result.path == str(outside_file)
        outside_file.unlink(missing_ok=True)

    @pytest.mark.asyncio()
    async def test_get_document_content_allows_default_mount_without_explicit_config(
        self,
        mock_user: dict[str, Any],
        mock_collection: MagicMock,
        mock_document: MagicMock,
        mock_artifact_repo: AsyncMock,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        """Default docker mount remains accessible when no roots are configured."""

        default_mount = (tmp_path / "default_mount").resolve()
        default_mount.mkdir()
        allowed_file = default_mount / "docker.pdf"
        allowed_file.write_text("docker content")

        monkeypatch.setattr(settings, "_document_root", None, raising=False)
        monkeypatch.setattr(settings, "_document_allowed_roots", (), raising=False)
        monkeypatch.setattr(settings, "_default_document_mounts", (default_mount,), raising=False)

        mock_document.file_path = str(allowed_file)
        mock_document.file_name = allowed_file.name

        mock_db = AsyncMock(spec=AsyncSession)
        mock_document_repo = AsyncMock()
        mock_document_repo.get_by_id.return_value = mock_document

        with (
            patch("webui.api.v2.documents.create_document_repository", return_value=mock_document_repo),
            patch("webui.api.v2.documents.DocumentArtifactRepository", return_value=mock_artifact_repo),
        ):
            result = await get_document_content(
                collection_uuid=mock_collection.id,
                document_uuid=mock_document.id,
                collection=mock_collection,
                current_user=mock_user,
                db=mock_db,
            )

        assert result.path == str(allowed_file)
        allowed_file.unlink(missing_ok=True)

    @pytest.mark.asyncio()
    async def test_get_document_content_respects_additional_allowed_roots(
        self,
        mock_user: dict[str, Any],
        mock_collection: MagicMock,
        mock_document: MagicMock,
        mock_artifact_repo: AsyncMock,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        """Ensure DOCUMENT_ALLOWED_ROOTS entries permit additional locations."""

        extra_root = (tmp_path / "external").resolve()
        extra_root.mkdir()
        allowed_file = extra_root / "external.pdf"
        allowed_file.write_text("extra content")

        monkeypatch.setattr(settings, "_document_root", None, raising=False)
        monkeypatch.setattr(settings, "_document_allowed_roots", (extra_root,), raising=False)

        mock_document.file_path = str(allowed_file)
        mock_document.file_name = allowed_file.name

        mock_db = AsyncMock(spec=AsyncSession)
        mock_document_repo = AsyncMock()
        mock_document_repo.get_by_id.return_value = mock_document

        with (
            patch("webui.api.v2.documents.create_document_repository", return_value=mock_document_repo),
            patch("webui.api.v2.documents.DocumentArtifactRepository", return_value=mock_artifact_repo),
        ):
            result = await get_document_content(
                collection_uuid=mock_collection.id,
                document_uuid=mock_document.id,
                collection=mock_collection,
                current_user=mock_user,
                db=mock_db,
            )

        assert result.path == str(allowed_file)
        allowed_file.unlink(missing_ok=True)

    @pytest.mark.asyncio()
    async def test_get_document_content_default_mime_type(
        self,
        mock_user: dict[str, Any],
        mock_collection: MagicMock,
        mock_document: MagicMock,
        mock_artifact_repo: AsyncMock,
        temp_file: Path,
    ) -> None:
        """Test that default mime type is used when document has no mime type."""
        mock_document.file_path = str(temp_file)
        mock_document.mime_type = None  # No mime type set

        mock_db = AsyncMock(spec=AsyncSession)
        mock_document_repo = AsyncMock()
        mock_document_repo.get_by_id.return_value = mock_document

        with (
            patch("webui.api.v2.documents.create_document_repository", return_value=mock_document_repo),
            patch("webui.api.v2.documents.DocumentArtifactRepository", return_value=mock_artifact_repo),
        ):
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
        self,
        mock_user: dict[str, Any],
        mock_collection: MagicMock,
        mock_document: MagicMock,
        mock_artifact_repo: AsyncMock,
        temp_file: Path,
    ) -> None:
        """Test that appropriate cache headers are set."""
        mock_document.file_path = str(temp_file)

        mock_db = AsyncMock(spec=AsyncSession)
        mock_document_repo = AsyncMock()
        mock_document_repo.get_by_id.return_value = mock_document

        with (
            patch("webui.api.v2.documents.create_document_repository", return_value=mock_document_repo),
            patch("webui.api.v2.documents.DocumentArtifactRepository", return_value=mock_artifact_repo),
        ):
            result = await get_document_content(
                collection_uuid=mock_collection.id,
                document_uuid=mock_document.id,
                collection=mock_collection,
                current_user=mock_user,
                db=mock_db,
            )

        # Check cache control header
        assert result.headers["Cache-Control"] == "private, max-age=3600"
