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
