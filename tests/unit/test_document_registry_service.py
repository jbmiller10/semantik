"""Unit tests for DocumentRegistryService."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from shared.dtos.ingestion import IngestedDocument
from shared.utils.hashing import compute_content_hash
from webui.services.document_registry_service import DocumentRegistryService


@pytest.fixture()
def mock_db_session() -> AsyncMock:
    """Create mock database session."""
    return AsyncMock()


@pytest.fixture()
def mock_document_repo() -> MagicMock:
    """Create mock document repository."""
    repo = MagicMock()
    repo.get_by_content_hash = AsyncMock(return_value=None)
    repo.create = AsyncMock()
    repo.session = AsyncMock()
    repo.session.refresh = AsyncMock()
    return repo


@pytest.fixture()
def registry_service(mock_db_session: AsyncMock, mock_document_repo: MagicMock) -> DocumentRegistryService:
    """Create DocumentRegistryService with mocks."""
    return DocumentRegistryService(mock_db_session, mock_document_repo)


@pytest.fixture()
def sample_ingested_document() -> IngestedDocument:
    """Create a sample IngestedDocument for testing."""
    content = "Test document content"
    return IngestedDocument(
        content=content,
        unique_id="file:///path/to/document.txt",
        source_type="directory",
        metadata={"file_size": 1234, "mime_type": "text/plain"},
        content_hash=compute_content_hash(content),
        file_path="/path/to/document.txt",
    )


class TestDocumentRegistryService:
    """Tests for DocumentRegistryService."""

    @pytest.mark.asyncio()
    async def test_register_new_document(
        self,
        registry_service: DocumentRegistryService,
        mock_document_repo: MagicMock,
        sample_ingested_document: IngestedDocument,
    ) -> None:
        """Test registering a new document returns is_new=True."""
        # Setup mock document returned by create
        mock_document = MagicMock()
        mock_document.id = "doc-uuid-123"
        mock_document_repo.create.return_value = mock_document

        result = await registry_service.register(
            collection_id="collection-uuid",
            ingested=sample_ingested_document,
            source_id=1,
        )

        assert result["is_new"] is True
        assert result["document_id"] == "doc-uuid-123"
        assert result["file_size"] == 1234

        # Verify repository was called correctly
        mock_document_repo.get_by_content_hash.assert_called_once_with(
            "collection-uuid", sample_ingested_document.content_hash
        )
        mock_document_repo.create.assert_called_once()
        create_kwargs = mock_document_repo.create.call_args.kwargs
        assert create_kwargs["collection_id"] == "collection-uuid"
        assert create_kwargs["content_hash"] == sample_ingested_document.content_hash
        assert create_kwargs["source_id"] == 1

    @pytest.mark.asyncio()
    async def test_register_duplicate_document(
        self,
        registry_service: DocumentRegistryService,
        mock_document_repo: MagicMock,
        sample_ingested_document: IngestedDocument,
    ) -> None:
        """Test registering a duplicate document returns is_new=False."""
        # Setup mock existing document
        mock_existing = MagicMock()
        mock_existing.id = "existing-doc-uuid"
        mock_existing.file_size = 5678
        mock_document_repo.get_by_content_hash.return_value = mock_existing

        result = await registry_service.register(
            collection_id="collection-uuid",
            ingested=sample_ingested_document,
        )

        assert result["is_new"] is False
        assert result["document_id"] == "existing-doc-uuid"
        assert result["file_size"] == 5678

        # Verify create was NOT called
        mock_document_repo.create.assert_not_called()

    @pytest.mark.asyncio()
    async def test_file_size_from_metadata(
        self,
        registry_service: DocumentRegistryService,
        mock_document_repo: MagicMock,
    ) -> None:
        """Test file size is extracted from metadata."""
        mock_document = MagicMock()
        mock_document.id = "doc-uuid"
        mock_document_repo.create.return_value = mock_document

        ingested = IngestedDocument(
            content="Short",
            unique_id="test-id",
            source_type="test",
            metadata={"file_size": 99999},
            content_hash=compute_content_hash("Short"),
        )

        result = await registry_service.register(
            collection_id="collection-uuid",
            ingested=ingested,
        )

        assert result["file_size"] == 99999
        create_kwargs = mock_document_repo.create.call_args.kwargs
        assert create_kwargs["file_size"] == 99999

    @pytest.mark.asyncio()
    async def test_file_size_estimated_from_content(
        self,
        registry_service: DocumentRegistryService,
        mock_document_repo: MagicMock,
    ) -> None:
        """Test file size is estimated from content when not in metadata."""
        mock_document = MagicMock()
        mock_document.id = "doc-uuid"
        mock_document_repo.create.return_value = mock_document

        content = "Test content here"
        ingested = IngestedDocument(
            content=content,
            unique_id="test-id",
            source_type="test",
            metadata={},  # No file_size
            content_hash=compute_content_hash(content),
        )

        result = await registry_service.register(
            collection_id="collection-uuid",
            ingested=ingested,
        )

        # Should be UTF-8 byte length
        expected_size = len(content.encode("utf-8"))
        assert result["file_size"] == expected_size

    @pytest.mark.asyncio()
    async def test_file_size_from_size_key(
        self,
        registry_service: DocumentRegistryService,
        mock_document_repo: MagicMock,
    ) -> None:
        """Test file size from 'size' metadata key."""
        mock_document = MagicMock()
        mock_document.id = "doc-uuid"
        mock_document_repo.create.return_value = mock_document

        ingested = IngestedDocument(
            content="Short",
            unique_id="test-id",
            source_type="test",
            metadata={"size": 12345},
            content_hash=compute_content_hash("Short"),
        )

        result = await registry_service.register(
            collection_id="collection-uuid",
            ingested=ingested,
        )

        assert result["file_size"] == 12345

    @pytest.mark.asyncio()
    async def test_file_size_from_content_length_key(
        self,
        registry_service: DocumentRegistryService,
        mock_document_repo: MagicMock,
    ) -> None:
        """Test file size from 'content_length' metadata key (HTTP responses)."""
        mock_document = MagicMock()
        mock_document.id = "doc-uuid"
        mock_document_repo.create.return_value = mock_document

        ingested = IngestedDocument(
            content="Short",
            unique_id="https://example.com/page",
            source_type="web",
            metadata={"content_length": 54321},
            content_hash=compute_content_hash("Short"),
        )

        result = await registry_service.register(
            collection_id="collection-uuid",
            ingested=ingested,
        )

        assert result["file_size"] == 54321

    @pytest.mark.asyncio()
    async def test_mime_type_from_metadata(
        self,
        registry_service: DocumentRegistryService,
        mock_document_repo: MagicMock,
    ) -> None:
        """Test MIME type is extracted from metadata."""
        mock_document = MagicMock()
        mock_document.id = "doc-uuid"
        mock_document_repo.create.return_value = mock_document

        ingested = IngestedDocument(
            content="Content",
            unique_id="test-id",
            source_type="test",
            metadata={"mime_type": "application/pdf"},
            content_hash=compute_content_hash("Content"),
            file_path="/path/to/file.txt",  # Would infer text/plain, but metadata takes precedence
        )

        await registry_service.register(
            collection_id="collection-uuid",
            ingested=ingested,
        )

        create_kwargs = mock_document_repo.create.call_args.kwargs
        assert create_kwargs["mime_type"] == "application/pdf"

    @pytest.mark.asyncio()
    async def test_mime_type_from_content_type_key(
        self,
        registry_service: DocumentRegistryService,
        mock_document_repo: MagicMock,
    ) -> None:
        """Test MIME type from 'content_type' metadata key."""
        mock_document = MagicMock()
        mock_document.id = "doc-uuid"
        mock_document_repo.create.return_value = mock_document

        ingested = IngestedDocument(
            content="Content",
            unique_id="https://example.com",
            source_type="web",
            metadata={"content_type": "text/html"},
            content_hash=compute_content_hash("Content"),
        )

        await registry_service.register(
            collection_id="collection-uuid",
            ingested=ingested,
        )

        create_kwargs = mock_document_repo.create.call_args.kwargs
        assert create_kwargs["mime_type"] == "text/html"

    @pytest.mark.asyncio()
    async def test_mime_type_inferred_from_file_path(
        self,
        registry_service: DocumentRegistryService,
        mock_document_repo: MagicMock,
    ) -> None:
        """Test MIME type is inferred from file_path when not in metadata."""
        mock_document = MagicMock()
        mock_document.id = "doc-uuid"
        mock_document_repo.create.return_value = mock_document

        ingested = IngestedDocument(
            content="Content",
            unique_id="test-id",
            source_type="test",
            metadata={},  # No mime_type
            content_hash=compute_content_hash("Content"),
            file_path="/path/to/document.pdf",
        )

        await registry_service.register(
            collection_id="collection-uuid",
            ingested=ingested,
        )

        create_kwargs = mock_document_repo.create.call_args.kwargs
        assert create_kwargs["mime_type"] == "application/pdf"

    @pytest.mark.asyncio()
    async def test_file_name_derived_from_file_path(
        self,
        registry_service: DocumentRegistryService,
        mock_document_repo: MagicMock,
    ) -> None:
        """Test file name is derived from file_path."""
        mock_document = MagicMock()
        mock_document.id = "doc-uuid"
        mock_document_repo.create.return_value = mock_document

        ingested = IngestedDocument(
            content="Content",
            unique_id="file:///path/to/document.txt",
            source_type="directory",
            metadata={},
            content_hash=compute_content_hash("Content"),
            file_path="/path/to/document.txt",
        )

        await registry_service.register(
            collection_id="collection-uuid",
            ingested=ingested,
        )

        create_kwargs = mock_document_repo.create.call_args.kwargs
        assert create_kwargs["file_name"] == "document.txt"

    @pytest.mark.asyncio()
    async def test_file_name_derived_from_unique_id_url(
        self,
        registry_service: DocumentRegistryService,
        mock_document_repo: MagicMock,
    ) -> None:
        """Test file name is derived from unique_id when no file_path."""
        mock_document = MagicMock()
        mock_document.id = "doc-uuid"
        mock_document_repo.create.return_value = mock_document

        ingested = IngestedDocument(
            content="Content",
            unique_id="https://example.com/docs/readme.html",
            source_type="web",
            metadata={},
            content_hash=compute_content_hash("Content"),
            # No file_path
        )

        await registry_service.register(
            collection_id="collection-uuid",
            ingested=ingested,
        )

        create_kwargs = mock_document_repo.create.call_args.kwargs
        assert create_kwargs["file_name"] == "readme.html"

    @pytest.mark.asyncio()
    async def test_file_path_uses_unique_id_when_no_file_path(
        self,
        registry_service: DocumentRegistryService,
        mock_document_repo: MagicMock,
    ) -> None:
        """Test file_path falls back to unique_id when file_path is None."""
        mock_document = MagicMock()
        mock_document.id = "doc-uuid"
        mock_document_repo.create.return_value = mock_document

        ingested = IngestedDocument(
            content="Content",
            unique_id="https://example.com/page",
            source_type="web",
            metadata={},
            content_hash=compute_content_hash("Content"),
            # No file_path
        )

        await registry_service.register(
            collection_id="collection-uuid",
            ingested=ingested,
        )

        create_kwargs = mock_document_repo.create.call_args.kwargs
        assert create_kwargs["file_path"] == "https://example.com/page"

    @pytest.mark.asyncio()
    async def test_uri_set_to_unique_id(
        self,
        registry_service: DocumentRegistryService,
        mock_document_repo: MagicMock,
        sample_ingested_document: IngestedDocument,
    ) -> None:
        """Test uri is set to unique_id."""
        mock_document = MagicMock()
        mock_document.id = "doc-uuid"
        mock_document_repo.create.return_value = mock_document

        await registry_service.register(
            collection_id="collection-uuid",
            ingested=sample_ingested_document,
        )

        create_kwargs = mock_document_repo.create.call_args.kwargs
        assert create_kwargs["uri"] == sample_ingested_document.unique_id

    @pytest.mark.asyncio()
    async def test_metadata_passed_to_create(
        self,
        registry_service: DocumentRegistryService,
        mock_document_repo: MagicMock,
    ) -> None:
        """Test metadata is passed to document repository."""
        mock_document = MagicMock()
        mock_document.id = "doc-uuid"
        mock_document_repo.create.return_value = mock_document

        custom_metadata = {"custom_key": "custom_value", "author": "test"}
        ingested = IngestedDocument(
            content="Content",
            unique_id="test-id",
            source_type="test",
            metadata=custom_metadata,
            content_hash=compute_content_hash("Content"),
        )

        await registry_service.register(
            collection_id="collection-uuid",
            ingested=ingested,
        )

        create_kwargs = mock_document_repo.create.call_args.kwargs
        assert create_kwargs["meta"] == custom_metadata
        assert create_kwargs["source_metadata"] == custom_metadata

    @pytest.mark.asyncio()
    async def test_web_source_example(
        self,
        registry_service: DocumentRegistryService,
        mock_document_repo: MagicMock,
    ) -> None:
        """Test registering a web source document."""
        mock_document = MagicMock()
        mock_document.id = "doc-uuid"
        mock_document_repo.create.return_value = mock_document

        content = "<html><body>Page content</body></html>"
        ingested = IngestedDocument(
            content=content,
            unique_id="https://example.com/article",
            source_type="web",
            metadata={
                "url": "https://example.com/article",
                "content_type": "text/html",
                "status_code": 200,
            },
            content_hash=compute_content_hash(content),
        )

        result = await registry_service.register(
            collection_id="collection-uuid",
            ingested=ingested,
        )

        assert result["is_new"] is True
        create_kwargs = mock_document_repo.create.call_args.kwargs
        assert create_kwargs["file_path"] == "https://example.com/article"
        assert create_kwargs["file_name"] == "article"
        assert create_kwargs["mime_type"] == "text/html"

    @pytest.mark.asyncio()
    async def test_slack_source_example(
        self,
        registry_service: DocumentRegistryService,
        mock_document_repo: MagicMock,
    ) -> None:
        """Test registering a Slack source document."""
        mock_document = MagicMock()
        mock_document.id = "doc-uuid"
        mock_document_repo.create.return_value = mock_document

        content = "Hello team, here's an update..."
        ingested = IngestedDocument(
            content=content,
            unique_id="slack://C12345/p1234567890",
            source_type="slack",
            metadata={
                "channel_id": "C12345",
                "message_ts": "1234567890.123456",
                "user_id": "U12345",
            },
            content_hash=compute_content_hash(content),
        )

        result = await registry_service.register(
            collection_id="collection-uuid",
            ingested=ingested,
        )

        assert result["is_new"] is True
        create_kwargs = mock_document_repo.create.call_args.kwargs
        assert create_kwargs["uri"] == "slack://C12345/p1234567890"


class TestRegisterOrUpdate:
    """Tests for register_or_update method (sync-aware document management)."""

    @pytest.fixture()
    def mock_chunk_repo(self) -> MagicMock:
        """Create mock chunk repository."""
        repo = MagicMock()
        repo.delete_chunks_by_document = AsyncMock(return_value=5)
        return repo

    @pytest.fixture()
    def registry_with_chunk_repo(
        self, mock_db_session: AsyncMock, mock_document_repo: MagicMock, mock_chunk_repo: MagicMock
    ) -> DocumentRegistryService:
        """Create DocumentRegistryService with chunk repo for sync operations."""
        return DocumentRegistryService(mock_db_session, mock_document_repo, mock_chunk_repo)

    @pytest.mark.asyncio()
    async def test_register_or_update_new_document(
        self,
        registry_with_chunk_repo: DocumentRegistryService,
        mock_document_repo: MagicMock,
        sample_ingested_document: IngestedDocument,
    ) -> None:
        """Test register_or_update creates new document when not found by URI."""
        # Setup: no existing document by URI
        mock_document_repo.get_by_uri = AsyncMock(return_value=None)
        mock_document_repo.update_last_seen = AsyncMock()

        # Setup mock document returned by create
        mock_document = MagicMock()
        mock_document.id = "new-doc-uuid"
        mock_document_repo.create.return_value = mock_document

        result = await registry_with_chunk_repo.register_or_update(
            collection_id="collection-uuid",
            ingested=sample_ingested_document,
            source_id=1,
        )

        assert result["is_new"] is True
        assert result["is_updated"] is False
        assert result["document_id"] == "new-doc-uuid"

        # Verify URI lookup was called
        mock_document_repo.get_by_uri.assert_called_once_with("collection-uuid", sample_ingested_document.unique_id)
        # Verify create was called (via register)
        mock_document_repo.create.assert_called_once()
        # Verify last_seen was updated for new document
        mock_document_repo.update_last_seen.assert_called_once_with("new-doc-uuid")

    @pytest.mark.asyncio()
    async def test_register_or_update_unchanged_document(
        self,
        registry_with_chunk_repo: DocumentRegistryService,
        mock_document_repo: MagicMock,
        sample_ingested_document: IngestedDocument,
    ) -> None:
        """Test register_or_update skips processing when content unchanged."""
        # Setup: existing document with same content hash
        mock_existing = MagicMock()
        mock_existing.id = "existing-doc-uuid"
        mock_existing.content_hash = sample_ingested_document.content_hash  # Same hash
        mock_existing.file_size = 1234
        mock_document_repo.get_by_uri = AsyncMock(return_value=mock_existing)
        mock_document_repo.update_last_seen = AsyncMock()

        result = await registry_with_chunk_repo.register_or_update(
            collection_id="collection-uuid",
            ingested=sample_ingested_document,
        )

        assert result["is_new"] is False
        assert result["is_updated"] is False
        assert result["document_id"] == "existing-doc-uuid"
        assert result["file_size"] == 1234

        # Verify last_seen was updated
        mock_document_repo.update_last_seen.assert_called_once_with("existing-doc-uuid")
        # Verify create was NOT called (skipped)
        mock_document_repo.create.assert_not_called()
        # Verify update_content was NOT called
        mock_document_repo.update_content = AsyncMock()
        mock_document_repo.update_content.assert_not_called()

    @pytest.mark.asyncio()
    async def test_register_or_update_content_changed(
        self,
        registry_with_chunk_repo: DocumentRegistryService,
        mock_document_repo: MagicMock,
        mock_chunk_repo: MagicMock,
        sample_ingested_document: IngestedDocument,
    ) -> None:
        """Test register_or_update updates document when content changed."""
        # Setup: existing document with different content hash
        mock_existing = MagicMock()
        mock_existing.id = "existing-doc-uuid"
        mock_existing.content_hash = "old_hash_" + "a" * 55  # Different hash
        mock_existing.file_size = 500
        mock_document_repo.get_by_uri = AsyncMock(return_value=mock_existing)
        mock_document_repo.update_last_seen = AsyncMock()
        mock_document_repo.update_content = AsyncMock()

        result = await registry_with_chunk_repo.register_or_update(
            collection_id="collection-uuid",
            ingested=sample_ingested_document,
        )

        assert result["is_new"] is False
        assert result["is_updated"] is True
        assert result["document_id"] == "existing-doc-uuid"

        # Verify last_seen was updated
        mock_document_repo.update_last_seen.assert_called_once_with("existing-doc-uuid")
        # Verify update_content was called with new hash
        mock_document_repo.update_content.assert_called_once()
        update_kwargs = mock_document_repo.update_content.call_args.kwargs
        assert update_kwargs["document_id"] == "existing-doc-uuid"
        assert update_kwargs["content_hash"] == sample_ingested_document.content_hash
        # Verify old chunks were deleted
        mock_chunk_repo.delete_chunks_by_document.assert_called_once_with(
            document_id="existing-doc-uuid",
            collection_id="collection-uuid",
        )

    @pytest.mark.asyncio()
    async def test_register_or_update_content_changed_no_chunk_repo(
        self,
        registry_service: DocumentRegistryService,
        mock_document_repo: MagicMock,
        sample_ingested_document: IngestedDocument,
    ) -> None:
        """Test register_or_update without chunk_repo skips chunk deletion."""
        # Setup: existing document with different content hash
        mock_existing = MagicMock()
        mock_existing.id = "existing-doc-uuid"
        mock_existing.content_hash = "old_hash_" + "a" * 55  # Different hash
        mock_existing.file_size = 500
        mock_document_repo.get_by_uri = AsyncMock(return_value=mock_existing)
        mock_document_repo.update_last_seen = AsyncMock()
        mock_document_repo.update_content = AsyncMock()

        # Use registry_service (no chunk_repo)
        result = await registry_service.register_or_update(
            collection_id="collection-uuid",
            ingested=sample_ingested_document,
        )

        assert result["is_new"] is False
        assert result["is_updated"] is True
        # Verify update_content was called
        mock_document_repo.update_content.assert_called_once()
        # No assertion on chunk_repo since it's None

    @pytest.mark.asyncio()
    async def test_register_or_update_updates_file_path_and_mime_type(
        self,
        registry_with_chunk_repo: DocumentRegistryService,
        mock_document_repo: MagicMock,
        mock_chunk_repo: MagicMock,
    ) -> None:
        """Test register_or_update passes file_path and mime_type to update_content."""
        # Setup: existing document with different content
        mock_existing = MagicMock()
        mock_existing.id = "existing-doc-uuid"
        mock_existing.content_hash = "old_hash_" + "a" * 55
        mock_existing.file_size = 500
        mock_document_repo.get_by_uri = AsyncMock(return_value=mock_existing)
        mock_document_repo.update_last_seen = AsyncMock()
        mock_document_repo.update_content = AsyncMock()

        new_content = "Updated document content"
        ingested = IngestedDocument(
            content=new_content,
            unique_id="file:///path/to/updated.pdf",
            source_type="directory",
            metadata={"mime_type": "application/pdf", "file_size": 2048},
            content_hash=compute_content_hash(new_content),
            file_path="/path/to/updated.pdf",
        )

        await registry_with_chunk_repo.register_or_update(
            collection_id="collection-uuid",
            ingested=ingested,
        )

        # Verify update_content was called with all fields
        update_kwargs = mock_document_repo.update_content.call_args.kwargs
        assert update_kwargs["file_path"] == "/path/to/updated.pdf"
        assert update_kwargs["mime_type"] == "application/pdf"
        assert update_kwargs["file_size"] == 2048
        assert update_kwargs["source_metadata"] == ingested.metadata

    @pytest.mark.asyncio()
    async def test_register_or_update_handles_zero_file_size(
        self,
        registry_with_chunk_repo: DocumentRegistryService,
        mock_document_repo: MagicMock,
    ) -> None:
        """Test register_or_update returns 0 for missing file_size on unchanged doc."""
        # Setup: existing document with None file_size
        mock_existing = MagicMock()
        mock_existing.id = "existing-doc-uuid"
        mock_existing.content_hash = compute_content_hash("Test")
        mock_existing.file_size = None  # Missing size
        mock_document_repo.get_by_uri = AsyncMock(return_value=mock_existing)
        mock_document_repo.update_last_seen = AsyncMock()

        ingested = IngestedDocument(
            content="Test",
            unique_id="test-uri",
            source_type="test",
            metadata={},
            content_hash=compute_content_hash("Test"),  # Same hash
        )

        result = await registry_with_chunk_repo.register_or_update(
            collection_id="collection-uuid",
            ingested=ingested,
        )

        assert result["file_size"] == 0  # Falls back to 0 when None

    @pytest.mark.asyncio()
    async def test_register_or_update_logs_chunk_deletion_count(
        self,
        registry_with_chunk_repo: DocumentRegistryService,
        mock_document_repo: MagicMock,
        mock_chunk_repo: MagicMock,
        caplog,
    ) -> None:
        """Test register_or_update logs chunk deletion when content changes."""
        import logging

        # Setup: existing document with different content
        mock_existing = MagicMock()
        mock_existing.id = "existing-doc-uuid"
        mock_existing.content_hash = "old_hash_" + "a" * 55
        mock_existing.file_size = 500
        mock_document_repo.get_by_uri = AsyncMock(return_value=mock_existing)
        mock_document_repo.update_last_seen = AsyncMock()
        mock_document_repo.update_content = AsyncMock()
        mock_chunk_repo.delete_chunks_by_document.return_value = 10

        ingested = IngestedDocument(
            content="New content",
            unique_id="test-uri",
            source_type="test",
            metadata={},
            content_hash=compute_content_hash("New content"),
        )

        with caplog.at_level(logging.DEBUG):
            await registry_with_chunk_repo.register_or_update(
                collection_id="collection-uuid",
                ingested=ingested,
            )

        # Verify chunk deletion was logged
        assert "Deleted 10 old chunks" in caplog.text
