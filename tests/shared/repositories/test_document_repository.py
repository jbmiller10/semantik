"""Tests for DocumentRepository.

This module tests document CRUD operations, deduplication, and sync tracking.
"""

from datetime import UTC, datetime, timedelta

import pytest

from shared.database.exceptions import EntityNotFoundError, ValidationError
from shared.database.models import DocumentStatus
from shared.database.repositories.document_repository import DocumentRepository


class TestDocumentRepositoryCreate:
    """Tests for document creation."""

    @pytest.mark.asyncio()
    async def test_create_validates_empty_file_path(self, db_session):
        """Test create() raises ValidationError for empty file_path."""
        repo = DocumentRepository(db_session)

        with pytest.raises(ValidationError, match="File path and name are required"):
            await repo.create(
                collection_id="test-col-id",
                file_path="",
                file_name="test.txt",
                file_size=1024,
                content_hash="a" * 64,
            )

    @pytest.mark.asyncio()
    async def test_create_validates_empty_file_name(self, db_session):
        """Test create() raises ValidationError for empty file_name."""
        repo = DocumentRepository(db_session)

        with pytest.raises(ValidationError, match="File path and name are required"):
            await repo.create(
                collection_id="test-col-id",
                file_path="/path/to/file",
                file_name="",
                file_size=1024,
                content_hash="a" * 64,
            )

    @pytest.mark.asyncio()
    async def test_create_validates_negative_file_size(self, db_session):
        """Test create() raises ValidationError for negative file_size."""
        repo = DocumentRepository(db_session)

        with pytest.raises(ValidationError, match="File size cannot be negative"):
            await repo.create(
                collection_id="test-col-id",
                file_path="/path/to/file",
                file_name="test.txt",
                file_size=-1,
                content_hash="a" * 64,
            )

    @pytest.mark.asyncio()
    async def test_create_validates_empty_content_hash(self, db_session):
        """Test create() raises ValidationError for empty content_hash."""
        repo = DocumentRepository(db_session)

        with pytest.raises(ValidationError, match="Content hash is required"):
            await repo.create(
                collection_id="test-col-id",
                file_path="/path/to/file",
                file_name="test.txt",
                file_size=1024,
                content_hash="",
            )

    @pytest.mark.asyncio()
    async def test_create_validates_invalid_content_hash_format(self, db_session):
        """Test create() raises ValidationError for invalid SHA-256 hash format."""
        repo = DocumentRepository(db_session)

        with pytest.raises(ValidationError, match="Invalid SHA-256 hash format"):
            await repo.create(
                collection_id="test-col-id",
                file_path="/path/to/file",
                file_name="test.txt",
                file_size=1024,
                content_hash="not-a-valid-sha256-hash",
            )

    @pytest.mark.asyncio()
    async def test_create_validates_short_content_hash(self, db_session):
        """Test create() raises ValidationError for too short hash."""
        repo = DocumentRepository(db_session)

        with pytest.raises(ValidationError, match="Invalid SHA-256 hash format"):
            await repo.create(
                collection_id="test-col-id",
                file_path="/path/to/file",
                file_name="test.txt",
                file_size=1024,
                content_hash="abc123",  # Too short
            )

    @pytest.mark.asyncio()
    async def test_create_raises_entity_not_found_for_missing_collection(self, db_session):
        """Test create() raises EntityNotFoundError when collection doesn't exist."""
        repo = DocumentRepository(db_session)

        with pytest.raises(EntityNotFoundError, match="collection"):
            await repo.create(
                collection_id="nonexistent-collection-id",
                file_path="/path/to/file",
                file_name="test.txt",
                file_size=1024,
                content_hash="a" * 64,
            )

    @pytest.mark.asyncio()
    async def test_create_returns_existing_document_on_duplicate_hash(
        self, db_session, test_user_db, collection_factory, document_factory
    ):
        """Test create() returns existing document if content_hash already exists."""
        user = test_user_db
        collection = await collection_factory(owner_id=user.id)

        # Create first document
        existing = await document_factory(
            collection_id=collection.id,
            content_hash="a" * 64,
        )

        repo = DocumentRepository(db_session)

        # Try to create duplicate
        result = await repo.create(
            collection_id=collection.id,
            file_path="/different/path",
            file_name="different.txt",
            file_size=2048,
            content_hash="a" * 64,  # Same hash
        )

        # Should return existing document
        assert result.id == existing.id


class TestDocumentRepositoryUpdateStatus:
    """Tests for status update operations."""

    @pytest.mark.asyncio()
    async def test_update_status_raises_entity_not_found(self, db_session):
        """Test update_status() raises EntityNotFoundError for missing document."""
        repo = DocumentRepository(db_session)

        with pytest.raises(EntityNotFoundError, match="document"):
            await repo.update_status(
                document_id="nonexistent-doc-id",
                status=DocumentStatus.COMPLETED,
            )

    @pytest.mark.asyncio()
    async def test_update_status_updates_chunk_count(
        self, db_session, test_user_db, collection_factory, document_factory
    ):
        """Test update_status() correctly updates chunk_count when provided."""
        user = test_user_db
        collection = await collection_factory(owner_id=user.id)
        document = await document_factory(collection_id=collection.id)

        repo = DocumentRepository(db_session)

        result = await repo.update_status(
            document_id=document.id,
            status=DocumentStatus.COMPLETED,
            chunk_count=42,
        )

        assert result.chunk_count == 42
        assert result.status == DocumentStatus.COMPLETED.value


class TestDocumentRepositoryBulkUpdateStatus:
    """Tests for bulk status updates."""

    @pytest.mark.asyncio()
    async def test_bulk_update_status_returns_zero_for_empty_list(self, db_session):
        """Test bulk_update_status() returns 0 when document_ids is empty."""
        repo = DocumentRepository(db_session)

        result = await repo.bulk_update_status(
            document_ids=[],
            status=DocumentStatus.COMPLETED,
        )

        assert result == 0

    @pytest.mark.asyncio()
    async def test_bulk_update_status_updates_multiple_documents(
        self, db_session, test_user_db, collection_factory, document_factory
    ):
        """Test bulk_update_status() updates multiple documents."""
        user = test_user_db
        collection = await collection_factory(owner_id=user.id)
        doc1 = await document_factory(collection_id=collection.id)
        doc2 = await document_factory(collection_id=collection.id)

        repo = DocumentRepository(db_session)

        result = await repo.bulk_update_status(
            document_ids=[doc1.id, doc2.id],
            status=DocumentStatus.FAILED,
            error_message="Test error",
        )

        assert result == 2


class TestDocumentRepositorySyncTracking:
    """Tests for sync tracking methods."""

    @pytest.mark.asyncio()
    async def test_mark_unseen_as_stale_updates_correct_documents(
        self, db_session, test_user_db, collection_factory, document_factory, source_factory
    ):
        """Test mark_unseen_as_stale() marks documents with last_seen_at < since."""
        user = test_user_db
        collection = await collection_factory(owner_id=user.id)

        # Create a source first
        source = await source_factory(collection_id=collection.id)

        # Create document with old last_seen_at
        old_doc = await document_factory(collection_id=collection.id, source_id=source.id)
        old_doc.last_seen_at = datetime.now(UTC) - timedelta(hours=2)
        old_doc.is_stale = False
        await db_session.flush()

        # Create document with recent last_seen_at
        recent_doc = await document_factory(collection_id=collection.id, source_id=source.id)
        recent_doc.last_seen_at = datetime.now(UTC)
        recent_doc.is_stale = False
        await db_session.flush()

        repo = DocumentRepository(db_session)

        # Mark documents not seen in last hour as stale
        since = datetime.now(UTC) - timedelta(hours=1)
        count = await repo.mark_unseen_as_stale(
            collection_id=collection.id,
            source_id=source.id,
            since=since,
        )

        assert count == 1

    @pytest.mark.asyncio()
    async def test_mark_unseen_as_stale_handles_null_last_seen_at(
        self, db_session, test_user_db, collection_factory, document_factory, source_factory
    ):
        """Test mark_unseen_as_stale() also marks documents with NULL last_seen_at."""
        user = test_user_db
        collection = await collection_factory(owner_id=user.id)

        # Create a source first
        source = await source_factory(collection_id=collection.id)

        # Create document with NULL last_seen_at
        doc = await document_factory(collection_id=collection.id, source_id=source.id)
        doc.last_seen_at = None
        doc.is_stale = False
        await db_session.flush()

        repo = DocumentRepository(db_session)

        count = await repo.mark_unseen_as_stale(
            collection_id=collection.id,
            source_id=source.id,
            since=datetime.now(UTC),
        )

        assert count == 1

    @pytest.mark.asyncio()
    async def test_get_stale_documents_returns_stale_only(
        self, db_session, test_user_db, collection_factory, document_factory
    ):
        """Test get_stale_documents() only returns stale documents."""
        user = test_user_db
        collection = await collection_factory(owner_id=user.id)

        # Create stale document
        stale_doc = await document_factory(collection_id=collection.id)
        stale_doc.is_stale = True
        await db_session.flush()

        # Create non-stale document
        fresh_doc = await document_factory(collection_id=collection.id)
        fresh_doc.is_stale = False
        await db_session.flush()

        repo = DocumentRepository(db_session)

        docs, total = await repo.get_stale_documents(collection_id=collection.id)

        assert total == 1
        assert len(docs) == 1
        assert docs[0].id == stale_doc.id


class TestDocumentRepositoryValidation:
    """Tests for validation edge cases."""

    @pytest.mark.asyncio()
    async def test_create_accepts_uppercase_hash(self, db_session, test_user_db, collection_factory):
        """Test create() accepts uppercase SHA-256 hash."""
        user = test_user_db
        collection = await collection_factory(owner_id=user.id)

        repo = DocumentRepository(db_session)

        # Uppercase hash should be valid
        result = await repo.create(
            collection_id=collection.id,
            file_path="/path/to/file",
            file_name="test.txt",
            file_size=1024,
            content_hash="A" * 64,  # Uppercase
        )

        assert result is not None
        assert result.content_hash == "A" * 64
