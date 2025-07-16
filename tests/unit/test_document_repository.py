"""Unit tests for DocumentRepository."""

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from shared.database.exceptions import DatabaseOperationError, EntityNotFoundError, ValidationError
from shared.database.models import Collection, Document, DocumentStatus
from shared.database.repositories.document_repository import DocumentRepository


class TestDocumentRepository:
    """Test cases for DocumentRepository."""

    @pytest.fixture()
    def mock_session(self):
        """Create a mock async session."""
        return AsyncMock()

    @pytest.fixture()
    def repository(self, mock_session):
        """Create a DocumentRepository instance with mocked session."""
        return DocumentRepository(mock_session)

    @pytest.fixture()
    def sample_collection(self):
        """Create a sample collection for testing."""
        return Collection(id=str(uuid4()), name="test-collection", owner_id=1, is_public=False)

    @pytest.mark.asyncio()
    async def test_create_document_validation_errors(self, repository):
        """Test validation errors during document creation."""
        # Test missing file path
        with pytest.raises(ValidationError) as exc_info:
            await repository.create(
                collection_id=str(uuid4()),
                file_path="",
                file_name="file.txt",
                file_size=1024,
                content_hash="hash",
            )
        assert "File path and name are required" in str(exc_info.value)

        # Test negative file size
        with pytest.raises(ValidationError) as exc_info:
            await repository.create(
                collection_id=str(uuid4()),
                file_path="/test/file.txt",
                file_name="file.txt",
                file_size=-1,
                content_hash="hash",
            )
        assert "File size cannot be negative" in str(exc_info.value)

        # Test missing content hash
        with pytest.raises(ValidationError) as exc_info:
            await repository.create(
                collection_id=str(uuid4()),
                file_path="/test/file.txt",
                file_name="file.txt",
                file_size=1024,
                content_hash="",
            )
        assert "Content hash is required" in str(exc_info.value)

        # Test invalid SHA-256 hash format
        with pytest.raises(ValidationError) as exc_info:
            await repository.create(
                collection_id=str(uuid4()),
                file_path="/test/file.txt",
                file_name="file.txt",
                file_size=1024,
                content_hash="invalid_hash",
            )
        assert "Invalid SHA-256 hash format" in str(exc_info.value)

        # Test hash too short
        with pytest.raises(ValidationError) as exc_info:
            await repository.create(
                collection_id=str(uuid4()),
                file_path="/test/file.txt",
                file_name="file.txt",
                file_size=1024,
                content_hash="abc123",
            )
        assert "Invalid SHA-256 hash format" in str(exc_info.value)

    @pytest.mark.asyncio()
    async def test_create_document_deduplication_logic(self, repository, mock_session, sample_collection):
        """Test that deduplication logic is called correctly."""
        # Setup
        collection_id = str(uuid4())
        content_hash = "a" * 64  # Valid SHA-256 hash format

        # Mock collection exists
        collection_result = AsyncMock()
        collection_result.scalar_one_or_none.return_value = sample_collection

        # Mock existing document found
        existing_doc = Document(
            id=str(uuid4()),
            collection_id=collection_id,
            source_id=None,
            file_path="/existing.txt",
            file_name="existing.txt",
            file_size=100,
            content_hash=content_hash,
            status=DocumentStatus.COMPLETED,
        )

        # Mock get_by_content_hash to return existing document
        repository.get_by_content_hash = AsyncMock(return_value=existing_doc)
        
        # Configure mock session for collection check
        mock_session.execute.return_value = collection_result

        # Act
        result = await repository.create(
            collection_id=collection_id,
            file_path="/test.txt",
            file_name="test.txt",
            file_size=200,
            content_hash=content_hash,
        )

        # Assert - should return existing document without creating new one
        assert result == existing_doc
        repository.get_by_content_hash.assert_called_once_with(collection_id, content_hash)
        mock_session.add.assert_not_called()
        mock_session.flush.assert_not_called()

    @pytest.mark.asyncio()
    async def test_get_by_id_calls_correct_query(self, repository, mock_session):
        """Test that get_by_id executes the correct query."""
        # Setup
        doc_id = str(uuid4())

        # Act
        await repository.get_by_id(doc_id)

        # Assert
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio()
    async def test_update_status_validation(self, repository, mock_session):
        """Test update_status with document not found."""
        # Setup
        doc_id = str(uuid4())
        
        # Mock get_by_id to return None
        repository.get_by_id = AsyncMock(return_value=None)

        # Act & Assert
        with pytest.raises(EntityNotFoundError) as exc_info:
            await repository.update_status(document_id=doc_id, status=DocumentStatus.COMPLETED)
        assert "document" in str(exc_info.value)

    @pytest.mark.asyncio()
    async def test_bulk_update_status_executes_update(self, repository, mock_session):
        """Test that bulk_update_status executes update query."""
        # Setup
        doc_ids = [str(uuid4()) for _ in range(3)]
        mock_result = AsyncMock()
        mock_result.rowcount = 3
        mock_session.execute.return_value = mock_result

        # Act
        count = await repository.bulk_update_status(
            document_ids=doc_ids, status=DocumentStatus.FAILED, error_message="Test error"
        )

        # Assert
        assert count == 3
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio()
    async def test_delete_document_not_found(self, repository, mock_session):
        """Test deletion when document doesn't exist."""
        # Setup
        doc_id = str(uuid4())
        
        # Mock get_by_id to return None
        repository.get_by_id = AsyncMock(return_value=None)

        # Act & Assert
        with pytest.raises(EntityNotFoundError) as exc_info:
            await repository.delete(doc_id)
        assert "document" in str(exc_info.value)

    @pytest.mark.asyncio()
    async def test_get_stats_structure(self, repository, mock_session):
        """Test that get_stats_by_collection returns correct structure."""
        # Setup
        collection_id = str(uuid4())

        # Mock the status counts query
        status_counts = [
            (DocumentStatus.PENDING, 5),
            (DocumentStatus.COMPLETED, 10),
            (DocumentStatus.FAILED, 2),
        ]

        # Create a mock result that returns status counts when iterated
        mock_result = AsyncMock()
        mock_result.__iter__.return_value = iter(status_counts)

        # Configure scalar responses for aggregates
        mock_session.scalar.side_effect = [
            1048576,  # total_size
            150,  # total_chunks
            3,  # duplicate_groups
        ]

        mock_session.execute.return_value = mock_result

        # Act
        stats = await repository.get_stats_by_collection(collection_id)

        # Assert structure
        assert "total_documents" in stats
        assert "by_status" in stats
        assert "total_size_bytes" in stats
        assert "total_chunks" in stats
        assert "duplicate_groups" in stats

    @pytest.mark.asyncio()
    async def test_database_operation_error_handling(self, repository, mock_session):
        """Test handling of unexpected database errors."""
        # Setup
        mock_session.execute.side_effect = Exception("Database connection lost")

        # Act & Assert
        with pytest.raises(DatabaseOperationError) as exc_info:
            await repository.get_by_id(str(uuid4()))
        assert "Database connection lost" in str(exc_info.value)
