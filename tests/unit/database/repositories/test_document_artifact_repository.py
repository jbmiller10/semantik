"""Unit tests for DocumentArtifactRepository."""

from unittest.mock import MagicMock

import pytest

from shared.database.exceptions import DatabaseOperationError, ValidationError
from shared.database.repositories.document_artifact_repository import (
    DEFAULT_MAX_ARTIFACT_BYTES,
    DocumentArtifactRepository,
)
from shared.utils.hashing import compute_content_hash


class MockSession:
    """Mock async session for testing."""

    def __init__(self):
        self.added = []
        self.executed = []
        self.flushed = False
        self._execute_results = []

    async def execute(self, stmt):
        self.executed.append(stmt)
        if self._execute_results:
            return self._execute_results.pop(0)
        return MagicMock(scalar_one_or_none=lambda: None, rowcount=0, one=lambda: MagicMock())

    async def flush(self):
        self.flushed = True

    def add(self, obj):
        self.added.append(obj)

    def set_execute_result(self, result):
        self._execute_results.append(result)


class TestDocumentArtifactRepository:
    """Test cases for DocumentArtifactRepository."""

    @pytest.fixture()
    def session(self):
        return MockSession()

    @pytest.fixture()
    def repo(self, session):
        return DocumentArtifactRepository(session)

    @pytest.fixture()
    def small_repo(self, session):
        """Repository with small max size for truncation tests."""
        return DocumentArtifactRepository(session, max_artifact_bytes=10)

    # --- Initialization tests ---

    def test_default_max_bytes(self, repo):
        """Test default max artifact bytes is set correctly."""
        assert repo._max_bytes == DEFAULT_MAX_ARTIFACT_BYTES

    def test_custom_max_bytes(self, session):
        """Test custom max artifact bytes."""
        repo = DocumentArtifactRepository(session, max_artifact_bytes=1000)
        assert repo._max_bytes == 1000

    # --- create_or_replace tests ---

    @pytest.mark.asyncio()
    async def test_create_or_replace_text_content(self, repo, session):
        """Test creating artifact with text content."""
        artifact = await repo.create_or_replace(
            document_id="doc-123",
            collection_id="col-456",
            content="Hello, World!",
            mime_type="text/plain",
            content_hash="abc123",
        )

        assert artifact.document_id == "doc-123"
        assert artifact.collection_id == "col-456"
        assert artifact.content_text == "Hello, World!"
        assert artifact.content_bytes is None
        assert artifact.mime_type == "text/plain"
        assert artifact.charset == "utf-8"
        assert artifact.is_truncated is False
        assert len(session.added) == 1
        assert session.flushed

    @pytest.mark.asyncio()
    async def test_create_or_replace_binary_content(self, repo, session):
        """Test creating artifact with binary content."""
        content = b"\x00\x01\x02\x03\x04"

        artifact = await repo.create_or_replace(
            document_id="doc-123",
            collection_id="col-456",
            content=content,
            mime_type="application/octet-stream",
            content_hash="def456",
        )

        assert artifact.content_text is None
        assert artifact.content_bytes == content
        assert artifact.charset is None
        assert artifact.is_truncated is False

    @pytest.mark.asyncio()
    async def test_create_or_replace_with_charset(self, repo, session):
        """Test creating artifact with custom charset."""
        artifact = await repo.create_or_replace(
            document_id="doc-123",
            collection_id="col-456",
            content="Hello",
            mime_type="text/plain",
            content_hash="abc123",
            charset="latin-1",
        )

        assert artifact.charset == "latin-1"

    @pytest.mark.asyncio()
    async def test_create_or_replace_invalid_kind(self, repo):
        """Test create_or_replace fails with invalid artifact_kind."""
        with pytest.raises(ValidationError) as exc_info:
            await repo.create_or_replace(
                document_id="doc-123",
                collection_id="col-456",
                content="Hello",
                mime_type="text/plain",
                content_hash="abc123",
                artifact_kind="invalid",
            )

        assert "artifact_kind" in str(exc_info.value)

    @pytest.mark.asyncio()
    async def test_create_or_replace_valid_kinds(self, repo, session):
        """Test create_or_replace accepts all valid artifact kinds."""
        for kind in ["primary", "preview", "thumbnail"]:
            artifact = await repo.create_or_replace(
                document_id="doc-123",
                collection_id="col-456",
                content="Hello",
                mime_type="text/plain",
                content_hash="abc123",
                artifact_kind=kind,
            )
            assert artifact.artifact_kind == kind

    @pytest.mark.asyncio()
    async def test_create_or_replace_truncates_text(self, small_repo, session):
        """Test text content truncation."""
        original = "Hello, World!"  # 13 bytes, exceeds max of 10

        artifact = await small_repo.create_or_replace(
            document_id="doc-123",
            collection_id="col-456",
            content=original,
            mime_type="text/plain",
            content_hash=compute_content_hash(original),
        )

        assert artifact.is_truncated is True
        assert len(artifact.content_text.encode("utf-8")) <= 10
        assert artifact.content_hash == compute_content_hash(artifact.content_text)

    @pytest.mark.asyncio()
    async def test_create_or_replace_truncates_bytes(self, small_repo, session):
        """Test binary content truncation."""
        original = b"0123456789ABCDEF"  # 16 bytes, exceeds max of 10

        artifact = await small_repo.create_or_replace(
            document_id="doc-123",
            collection_id="col-456",
            content=original,
            mime_type="application/octet-stream",
            content_hash=compute_content_hash(original),
        )

        assert artifact.is_truncated is True
        assert artifact.content_bytes == b"0123456789"
        assert artifact.size_bytes == 10
        assert artifact.content_hash == compute_content_hash(b"0123456789")

    @pytest.mark.asyncio()
    async def test_create_or_replace_handles_utf8_boundary(self, session):
        """Test truncation respects UTF-8 character boundaries."""
        repo = DocumentArtifactRepository(session, max_artifact_bytes=5)
        # "ab" + 4-byte emoji + "c" = 7 bytes total
        original = "ab\U0001f642c"

        artifact = await repo.create_or_replace(
            document_id="doc-123",
            collection_id="col-456",
            content=original,
            mime_type="text/plain",
            content_hash=compute_content_hash(original),
        )

        # Should truncate to "ab" (2 bytes) since emoji would exceed limit
        assert artifact.is_truncated is True
        assert artifact.content_text == "ab"

    # --- get_primary tests ---

    @pytest.mark.asyncio()
    async def test_get_primary_found(self, repo, session):
        """Test retrieving primary artifact."""
        mock_artifact = MagicMock()
        mock_artifact.artifact_kind = "primary"

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_artifact
        session.set_execute_result(mock_result)

        result = await repo.get_primary("doc-123")

        assert result == mock_artifact

    @pytest.mark.asyncio()
    async def test_get_primary_not_found(self, repo, session):
        """Test get_primary returns None when not found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        session.set_execute_result(mock_result)

        result = await repo.get_primary("doc-123")

        assert result is None

    # --- get_by_kind tests ---

    @pytest.mark.asyncio()
    async def test_get_by_kind_found(self, repo, session):
        """Test retrieving artifact by kind."""
        mock_artifact = MagicMock()
        mock_artifact.artifact_kind = "preview"

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_artifact
        session.set_execute_result(mock_result)

        result = await repo.get_by_kind("doc-123", "preview")

        assert result == mock_artifact

    @pytest.mark.asyncio()
    async def test_get_by_kind_not_found(self, repo, session):
        """Test get_by_kind returns None when not found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        session.set_execute_result(mock_result)

        result = await repo.get_by_kind("doc-123", "thumbnail")

        assert result is None

    # --- get_content tests ---

    @pytest.mark.asyncio()
    async def test_get_content_text(self, repo, session):
        """Test get_content returns text content."""
        mock_artifact = MagicMock()
        mock_artifact.content_text = "Hello, World!"
        mock_artifact.content_bytes = None
        mock_artifact.mime_type = "text/plain"
        mock_artifact.charset = "utf-8"

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_artifact
        session.set_execute_result(mock_result)

        result = await repo.get_content("doc-123")

        assert result == ("Hello, World!", "text/plain", "utf-8")

    @pytest.mark.asyncio()
    async def test_get_content_bytes(self, repo, session):
        """Test get_content returns binary content."""
        mock_artifact = MagicMock()
        mock_artifact.content_text = None
        mock_artifact.content_bytes = b"\x00\x01\x02"
        mock_artifact.mime_type = "application/octet-stream"
        mock_artifact.charset = None

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_artifact
        session.set_execute_result(mock_result)

        result = await repo.get_content("doc-123")

        assert result == (b"\x00\x01\x02", "application/octet-stream", None)

    @pytest.mark.asyncio()
    async def test_get_content_not_found(self, repo, session):
        """Test get_content returns None when not found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        session.set_execute_result(mock_result)

        result = await repo.get_content("doc-123")

        assert result is None

    @pytest.mark.asyncio()
    async def test_get_content_custom_kind(self, repo, session):
        """Test get_content with custom artifact kind."""
        mock_artifact = MagicMock()
        mock_artifact.content_text = "Preview content"
        mock_artifact.content_bytes = None
        mock_artifact.mime_type = "text/html"
        mock_artifact.charset = "utf-8"

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_artifact
        session.set_execute_result(mock_result)

        result = await repo.get_content("doc-123", artifact_kind="preview")

        assert result[0] == "Preview content"

    # --- has_artifact tests ---

    @pytest.mark.asyncio()
    async def test_has_artifact_true(self, repo, session):
        """Test has_artifact returns True when artifact exists."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = 1  # Artifact ID
        session.set_execute_result(mock_result)

        result = await repo.has_artifact("doc-123")

        assert result is True

    @pytest.mark.asyncio()
    async def test_has_artifact_false(self, repo, session):
        """Test has_artifact returns False when artifact doesn't exist."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        session.set_execute_result(mock_result)

        result = await repo.has_artifact("doc-123")

        assert result is False

    @pytest.mark.asyncio()
    async def test_has_artifact_custom_kind(self, repo, session):
        """Test has_artifact with custom kind."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = 1
        session.set_execute_result(mock_result)

        result = await repo.has_artifact("doc-123", artifact_kind="thumbnail")

        assert result is True

    # --- delete_for_document tests ---

    @pytest.mark.asyncio()
    async def test_delete_for_document_success(self, repo, session):
        """Test deleting all artifacts for a document."""
        mock_result = MagicMock()
        mock_result.rowcount = 3
        session.set_execute_result(mock_result)

        result = await repo.delete_for_document("doc-123")

        assert result == 3

    @pytest.mark.asyncio()
    async def test_delete_for_document_none_exist(self, repo, session):
        """Test delete_for_document when no artifacts exist."""
        mock_result = MagicMock()
        mock_result.rowcount = 0
        session.set_execute_result(mock_result)

        result = await repo.delete_for_document("doc-123")

        assert result == 0

    # --- delete_for_collection tests ---

    @pytest.mark.asyncio()
    async def test_delete_for_collection_success(self, repo, session):
        """Test deleting all artifacts for a collection."""
        mock_result = MagicMock()
        mock_result.rowcount = 10
        session.set_execute_result(mock_result)

        result = await repo.delete_for_collection("col-456")

        assert result == 10

    @pytest.mark.asyncio()
    async def test_delete_for_collection_none_exist(self, repo, session):
        """Test delete_for_collection when no artifacts exist."""
        mock_result = MagicMock()
        mock_result.rowcount = 0
        session.set_execute_result(mock_result)

        result = await repo.delete_for_collection("col-456")

        assert result == 0

    # --- get_stats_for_collection tests ---

    @pytest.mark.asyncio()
    async def test_get_stats_for_collection(self, repo, session):
        """Test getting collection artifact statistics."""
        mock_row = MagicMock()
        mock_row.count = 5
        mock_row.total_bytes = 10000
        mock_row.truncated_count = 1

        mock_result = MagicMock()
        mock_result.one.return_value = mock_row
        session.set_execute_result(mock_result)

        result = await repo.get_stats_for_collection("col-456")

        assert result["artifact_count"] == 5
        assert result["total_bytes"] == 10000
        assert result["truncated_count"] == 1

    @pytest.mark.asyncio()
    async def test_get_stats_for_collection_empty(self, repo, session):
        """Test getting stats for collection with no artifacts."""
        mock_row = MagicMock()
        mock_row.count = 0
        mock_row.total_bytes = None
        mock_row.truncated_count = 0

        mock_result = MagicMock()
        mock_result.one.return_value = mock_row
        session.set_execute_result(mock_result)

        result = await repo.get_stats_for_collection("col-456")

        assert result["artifact_count"] == 0
        assert result["total_bytes"] == 0
        assert result["truncated_count"] == 0

    # --- Error handling tests ---

    @pytest.mark.asyncio()
    async def test_create_or_replace_database_error(self, repo, session):
        """Test create_or_replace wraps database errors."""

        async def raise_error(_stmt):
            raise Exception("DB error")

        session.execute = raise_error

        with pytest.raises(DatabaseOperationError):
            await repo.create_or_replace(
                document_id="doc-123",
                collection_id="col-456",
                content="Hello",
                mime_type="text/plain",
                content_hash="abc123",
            )

    @pytest.mark.asyncio()
    async def test_get_primary_database_error(self, repo, session):
        """Test get_primary wraps database errors."""

        async def raise_error(_stmt):
            raise Exception("DB error")

        session.execute = raise_error

        with pytest.raises(DatabaseOperationError):
            await repo.get_primary("doc-123")

    @pytest.mark.asyncio()
    async def test_get_by_kind_database_error(self, repo, session):
        """Test get_by_kind wraps database errors."""

        async def raise_error(_stmt):
            raise Exception("DB error")

        session.execute = raise_error

        with pytest.raises(DatabaseOperationError):
            await repo.get_by_kind("doc-123", "primary")

    @pytest.mark.asyncio()
    async def test_has_artifact_database_error(self, repo, session):
        """Test has_artifact wraps database errors."""

        async def raise_error(_stmt):
            raise Exception("DB error")

        session.execute = raise_error

        with pytest.raises(DatabaseOperationError):
            await repo.has_artifact("doc-123")

    @pytest.mark.asyncio()
    async def test_delete_for_document_database_error(self, repo, session):
        """Test delete_for_document wraps database errors."""

        async def raise_error(_stmt):
            raise Exception("DB error")

        session.execute = raise_error

        with pytest.raises(DatabaseOperationError):
            await repo.delete_for_document("doc-123")

    @pytest.mark.asyncio()
    async def test_delete_for_collection_database_error(self, repo, session):
        """Test delete_for_collection wraps database errors."""

        async def raise_error(_stmt):
            raise Exception("DB error")

        session.execute = raise_error

        with pytest.raises(DatabaseOperationError):
            await repo.delete_for_collection("col-456")

    @pytest.mark.asyncio()
    async def test_get_stats_database_error(self, repo, session):
        """Test get_stats_for_collection wraps database errors."""

        async def raise_error(_stmt):
            raise Exception("DB error")

        session.execute = raise_error

        with pytest.raises(DatabaseOperationError):
            await repo.get_stats_for_collection("col-456")
