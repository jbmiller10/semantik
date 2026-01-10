"""Tests for DocumentRepository.

This module tests document CRUD operations, deduplication, and sync tracking.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

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


class TestDocumentRepositoryListing:
    """Tests for list and duplicate utilities."""

    @pytest.mark.asyncio()
    async def test_list_by_collection_filters_status(
        self, db_session, test_user_db, collection_factory, document_factory
    ):
        """list_by_collection() should filter by status and return total count."""
        user = test_user_db
        collection = await collection_factory(owner_id=user.id)

        completed_doc = await document_factory(
            collection_id=collection.id,
            status=DocumentStatus.COMPLETED,
        )
        failed_doc_1 = await document_factory(
            collection_id=collection.id,
            status=DocumentStatus.FAILED,
        )
        failed_doc_2 = await document_factory(
            collection_id=collection.id,
            status=DocumentStatus.FAILED,
        )

        repo = DocumentRepository(db_session)
        docs, total = await repo.list_by_collection(collection.id, status=DocumentStatus.FAILED)

        assert total == 2
        assert {doc.id for doc in docs} == {failed_doc_1.id, failed_doc_2.id}
        assert completed_doc.id not in {doc.id for doc in docs}

    @pytest.mark.asyncio()
    async def test_list_by_source_id_filters(
        self, db_session, test_user_db, collection_factory, document_factory, source_factory
    ):
        """list_by_source_id() should filter by source and status."""
        user = test_user_db
        collection = await collection_factory(owner_id=user.id)
        source = await source_factory(collection_id=collection.id)

        matching_doc = await document_factory(
            collection_id=collection.id,
            source_id=source.id,
            status=DocumentStatus.COMPLETED,
        )
        await document_factory(
            collection_id=collection.id,
            source_id=source.id,
            status=DocumentStatus.FAILED,
        )
        await document_factory(
            collection_id=collection.id,
            source_id=None,
            status=DocumentStatus.COMPLETED,
        )

        repo = DocumentRepository(db_session)
        docs = await repo.list_by_source_id(collection.id, source.id, status=DocumentStatus.COMPLETED)

        assert [doc.id for doc in docs] == [matching_doc.id]

    @pytest.mark.asyncio()
    async def test_list_and_delete_duplicates(self):
        """list_duplicates() and delete_duplicates() should identify and remove duplicates."""
        dup_hash = "a" * 64
        older = MagicMock()
        older.id = "doc-1"
        older.content_hash = dup_hash
        older.created_at = datetime.now(UTC) - timedelta(minutes=5)
        newer = MagicMock()
        newer.id = "doc-2"
        newer.content_hash = dup_hash
        newer.created_at = datetime.now(UTC)

        result = MagicMock()
        result.scalars.return_value.all.return_value = [older, newer]

        session = AsyncMock()
        session.execute.return_value = result

        repo = DocumentRepository(session)

        duplicates = await repo.list_duplicates("collection-1")
        assert len(duplicates) == 1
        content_hash, count, docs = duplicates[0]
        assert content_hash == dup_hash
        assert count == 2
        assert {doc.id for doc in docs} == {older.id, newer.id}

        delete_result = MagicMock()
        delete_result.rowcount = 1
        session.execute.return_value = delete_result
        repo.list_duplicates = AsyncMock(return_value=[(dup_hash, 2, [older, newer])])  # type: ignore[assignment]

        deleted = await repo.delete_duplicates("collection-1", keep_oldest=True)
        assert deleted == 1


class TestDocumentRepositoryStatsAndSync:
    """Tests for stats and sync-related helpers."""

    @pytest.mark.asyncio()
    async def test_get_stats_by_collection(self, db_session, test_user_db, collection_factory, document_factory):
        """get_stats_by_collection() should aggregate counts and sizes."""
        user = test_user_db
        collection = await collection_factory(owner_id=user.id)

        # Each document needs a unique content_hash due to unique constraint
        await document_factory(
            collection_id=collection.id,
            status=DocumentStatus.COMPLETED,
            file_size=100,
            chunk_count=2,
            content_hash="a" * 64,
        )
        await document_factory(
            collection_id=collection.id,
            status=DocumentStatus.FAILED,
            file_size=200,
            chunk_count=3,
            content_hash="b" * 64,
        )
        await document_factory(
            collection_id=collection.id,
            status=DocumentStatus.COMPLETED,
            file_size=50,
            chunk_count=1,
            content_hash="c" * 64,
        )

        repo = DocumentRepository(db_session)
        stats = await repo.get_stats_by_collection(collection.id)

        assert stats["total_documents"] == 3
        assert stats["by_status"]["completed"] == 2
        assert stats["by_status"]["failed"] == 1
        assert stats["total_size_bytes"] == 350
        assert stats["total_chunks"] == 6
        # No duplicates possible due to unique constraint on (collection_id, content_hash)
        assert stats["duplicate_groups"] == 0

    @pytest.mark.asyncio()
    async def test_update_last_seen_and_clear_stale_flag(
        self, db_session, test_user_db, collection_factory, document_factory
    ):
        """update_last_seen() and clear_stale_flag() should clear stale state."""
        user = test_user_db
        collection = await collection_factory(owner_id=user.id)
        document = await document_factory(collection_id=collection.id)
        document.is_stale = True
        await db_session.flush()

        repo = DocumentRepository(db_session)

        updated = await repo.update_last_seen(document.id)
        assert updated.last_seen_at is not None
        assert updated.is_stale is False

        document.is_stale = True
        await db_session.flush()

        cleared = await repo.clear_stale_flag(document.id)
        assert cleared.is_stale is False

    @pytest.mark.asyncio()
    async def test_update_content_resets_status_and_chunking_fields(
        self, db_session, test_user_db, collection_factory, document_factory
    ):
        """update_content() should reset processing fields for re-ingestion."""
        user = test_user_db
        collection = await collection_factory(owner_id=user.id)
        document = await document_factory(
            collection_id=collection.id,
            status=DocumentStatus.FAILED,
            error_message="boom",
            chunk_count=5,
            chunks_count=5,
        )
        document.chunking_started_at = datetime.now(UTC)
        document.chunking_completed_at = datetime.now(UTC)
        document.is_stale = True
        await db_session.flush()

        repo = DocumentRepository(db_session)
        updated = await repo.update_content(
            document_id=document.id,
            content_hash="e" * 64,
            file_size=999,
            file_path="/new/path/file.txt",
            mime_type="text/plain",
            source_metadata={"source": "test"},
        )

        assert updated.content_hash == "e" * 64
        assert updated.file_size == 999
        assert updated.file_path == "/new/path/file.txt"
        assert updated.mime_type == "text/plain"
        assert updated.source_metadata == {"source": "test"}
        assert updated.status == DocumentStatus.PENDING.value
        assert updated.error_message is None
        assert updated.chunk_count == 0
        assert updated.chunks_count == 0
        assert updated.chunking_started_at is None
        assert updated.chunking_completed_at is None
        assert updated.is_stale is False


class TestDocumentRepositoryRetryUtilities:
    """Tests for retry-related repository helpers."""

    @pytest.mark.asyncio()
    async def test_reset_for_retry_success(self, db_session, test_user_db, collection_factory, document_factory):
        """reset_for_retry() should reset failed docs and increment retry count."""
        user = test_user_db
        collection = await collection_factory(owner_id=user.id)
        document = await document_factory(
            collection_id=collection.id,
            status=DocumentStatus.FAILED,
            retry_count=1,
            error_message="boom",
            error_category="transient",
            chunk_count=3,
        )

        repo = DocumentRepository(db_session)
        updated = await repo.reset_for_retry(document.id)

        assert updated.status == DocumentStatus.PENDING.value
        assert updated.retry_count == 2
        assert updated.last_retry_at is not None
        assert updated.error_message is None
        assert updated.error_category is None
        assert updated.chunk_count == 0

    @pytest.mark.asyncio()
    async def test_reset_for_retry_requires_failed_status(
        self, db_session, test_user_db, collection_factory, document_factory
    ):
        """reset_for_retry() should reject non-failed docs."""
        user = test_user_db
        collection = await collection_factory(owner_id=user.id)
        document = await document_factory(collection_id=collection.id, status=DocumentStatus.COMPLETED)

        repo = DocumentRepository(db_session)
        with pytest.raises(ValidationError, match="not in FAILED status"):
            await repo.reset_for_retry(document.id)

    @pytest.mark.asyncio()
    async def test_bulk_reset_failed_for_retry_filters_docs(
        self, db_session, test_user_db, collection_factory, document_factory
    ):
        """bulk_reset_failed_for_retry() should only reset retryable docs."""
        user = test_user_db
        collection = await collection_factory(owner_id=user.id)

        doc_retryable = await document_factory(
            collection_id=collection.id,
            status=DocumentStatus.FAILED,
            retry_count=0,
            error_category="transient",
        )
        await document_factory(
            collection_id=collection.id,
            status=DocumentStatus.FAILED,
            retry_count=3,
            error_category="transient",
        )
        await document_factory(
            collection_id=collection.id,
            status=DocumentStatus.FAILED,
            retry_count=0,
            error_category="permanent",
        )
        doc_null_category = await document_factory(
            collection_id=collection.id,
            status=DocumentStatus.FAILED,
            retry_count=0,
            error_category=None,
        )

        repo = DocumentRepository(db_session)
        count = await repo.bulk_reset_failed_for_retry(collection.id, max_retry_count=3)

        assert count == 2

        updated_retryable = await repo.get_by_id(doc_retryable.id)
        updated_null = await repo.get_by_id(doc_null_category.id)
        assert updated_retryable.status == DocumentStatus.PENDING.value
        assert updated_retryable.retry_count == 1
        assert updated_null.status == DocumentStatus.PENDING.value
        assert updated_null.retry_count == 1

    @pytest.mark.asyncio()
    async def test_list_failed_documents_and_counts(
        self, db_session, test_user_db, collection_factory, document_factory
    ):
        """list_failed_documents() and get_failed_document_count() should respect filters."""
        user = test_user_db
        collection = await collection_factory(owner_id=user.id)

        retryable = await document_factory(
            collection_id=collection.id,
            status=DocumentStatus.FAILED,
            retry_count=0,
            error_category="transient",
        )
        permanent = await document_factory(
            collection_id=collection.id,
            status=DocumentStatus.FAILED,
            retry_count=0,
            error_category="permanent",
        )
        await document_factory(
            collection_id=collection.id,
            status=DocumentStatus.FAILED,
            retry_count=3,
            error_category="transient",
        )
        await document_factory(
            collection_id=collection.id,
            status=DocumentStatus.COMPLETED,
        )

        repo = DocumentRepository(db_session)

        docs, total = await repo.list_failed_documents(
            collection_id=collection.id,
            retryable_only=True,
            max_retry_count=3,
        )
        assert total == 1
        assert [doc.id for doc in docs] == [retryable.id]

        docs, total = await repo.list_failed_documents(
            collection_id=collection.id,
            error_category="permanent",
        )
        assert total == 1
        assert [doc.id for doc in docs] == [permanent.id]

        counts = await repo.get_failed_document_count(collection.id, retryable_only=False)
        assert counts["transient"] == 2
        assert counts["permanent"] == 1
        assert counts["total"] == 3
