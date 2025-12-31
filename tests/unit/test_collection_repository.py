"""Unit tests for CollectionRepository using mocks."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from shared.database.exceptions import (
    DatabaseOperationError,
    EntityAlreadyExistsError,
    EntityNotFoundError,
    ValidationError,
)
from shared.database.models import Collection, CollectionStatus
from shared.database.repositories.collection_repository import CollectionRepository


class TestCollectionRepository:
    """Unit tests for CollectionRepository."""

    @pytest.fixture()
    def mock_session(self) -> AsyncMock:
        """Create a mock async session."""
        return AsyncMock()

    @pytest.fixture()
    def repository(self, mock_session) -> CollectionRepository:
        """Create a CollectionRepository instance with mocked session."""
        return CollectionRepository(mock_session)

    @pytest.fixture()
    def sample_collection(self) -> Collection:
        """Create a sample collection for testing."""
        return Collection(
            id=str(uuid4()),
            name="test-collection",
            owner_id=1,
            is_public=False,
            status=CollectionStatus.READY,
            embedding_model="test-model",
            chunk_size=1000,
            chunk_overlap=200,
        )

    # --- get_by_uuid Tests ---

    @pytest.mark.asyncio()
    async def test_get_by_uuid_found(self, repository, mock_session, sample_collection) -> None:
        """Test get_by_uuid returns collection when found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_collection
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repository.get_by_uuid(sample_collection.id)

        assert result == sample_collection
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio()
    async def test_get_by_uuid_not_found(self, repository, mock_session) -> None:
        """Test get_by_uuid returns None when not found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repository.get_by_uuid("nonexistent-id")

        assert result is None

    @pytest.mark.asyncio()
    async def test_get_by_uuid_database_error(self, repository, mock_session) -> None:
        """Test get_by_uuid raises DatabaseOperationError on failure."""
        mock_session.execute = AsyncMock(side_effect=Exception("Connection error"))

        with pytest.raises(DatabaseOperationError):
            await repository.get_by_uuid(str(uuid4()))

    # --- get_by_name Tests ---

    @pytest.mark.asyncio()
    async def test_get_by_name_found(self, repository, mock_session, sample_collection) -> None:
        """Test get_by_name returns collection when found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_collection
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repository.get_by_name("test-collection")

        assert result == sample_collection

    @pytest.mark.asyncio()
    async def test_get_by_name_not_found(self, repository, mock_session) -> None:
        """Test get_by_name returns None when not found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repository.get_by_name("nonexistent")

        assert result is None

    @pytest.mark.asyncio()
    async def test_get_by_name_database_error(self, repository, mock_session) -> None:
        """Test get_by_name raises DatabaseOperationError on failure."""
        mock_session.execute = AsyncMock(side_effect=Exception("Connection error"))

        with pytest.raises(DatabaseOperationError):
            await repository.get_by_name("test-collection")

    # --- update_status Tests ---

    @pytest.mark.asyncio()
    async def test_update_status_success(self, repository, mock_session, sample_collection) -> None:
        """Test update_status updates collection status."""
        repository.get_by_uuid = AsyncMock(return_value=sample_collection)
        mock_session.flush = AsyncMock()

        result = await repository.update_status(sample_collection.id, CollectionStatus.PROCESSING, "Processing started")

        assert result.status == CollectionStatus.PROCESSING
        assert result.status_message == "Processing started"
        mock_session.flush.assert_called_once()

    @pytest.mark.asyncio()
    async def test_update_status_not_found(self, repository, mock_session) -> None:
        """Test update_status raises EntityNotFoundError for missing collection."""
        repository.get_by_uuid = AsyncMock(return_value=None)

        with pytest.raises(EntityNotFoundError):
            await repository.update_status(str(uuid4()), CollectionStatus.READY)

    @pytest.mark.asyncio()
    async def test_update_status_database_error(self, repository, mock_session, sample_collection) -> None:
        """Test update_status raises DatabaseOperationError on failure."""
        repository.get_by_uuid = AsyncMock(return_value=sample_collection)
        mock_session.flush = AsyncMock(side_effect=Exception("Connection error"))

        with pytest.raises(DatabaseOperationError):
            await repository.update_status(sample_collection.id, CollectionStatus.READY)

    # --- update_stats Tests ---

    @pytest.mark.asyncio()
    async def test_update_stats_success(self, repository, mock_session, sample_collection) -> None:
        """Test update_stats updates collection statistics."""
        repository.get_by_uuid = AsyncMock(return_value=sample_collection)
        mock_session.flush = AsyncMock()

        result = await repository.update_stats(
            sample_collection.id,
            document_count=10,
            vector_count=100,
            total_size_bytes=1024,
        )

        assert result.document_count == 10
        assert result.vector_count == 100
        assert result.total_size_bytes == 1024

    @pytest.mark.asyncio()
    async def test_update_stats_negative_document_count_error(self, repository, mock_session) -> None:
        """Test update_stats raises error for negative document count."""
        # Validation happens before DB access, but error is wrapped in DatabaseOperationError
        with pytest.raises((ValidationError, DatabaseOperationError)):
            await repository.update_stats(str(uuid4()), document_count=-1)

    @pytest.mark.asyncio()
    async def test_update_stats_negative_vector_count_error(self, repository, mock_session) -> None:
        """Test update_stats raises error for negative vector count."""
        with pytest.raises((ValidationError, DatabaseOperationError)):
            await repository.update_stats(str(uuid4()), vector_count=-1)

    @pytest.mark.asyncio()
    async def test_update_stats_negative_size_error(self, repository, mock_session) -> None:
        """Test update_stats raises error for negative total size."""
        with pytest.raises((ValidationError, DatabaseOperationError)):
            await repository.update_stats(str(uuid4()), total_size_bytes=-1)

    @pytest.mark.asyncio()
    async def test_update_stats_not_found(self, repository, mock_session) -> None:
        """Test update_stats raises EntityNotFoundError for missing collection."""
        repository.get_by_uuid = AsyncMock(return_value=None)

        with pytest.raises(EntityNotFoundError):
            await repository.update_stats(str(uuid4()), document_count=10)

    # --- update Tests ---

    @pytest.mark.asyncio()
    async def test_update_success(self, repository, mock_session, sample_collection) -> None:
        """Test update applies multiple field updates."""
        repository.get_by_uuid = AsyncMock(return_value=sample_collection)
        mock_session.flush = AsyncMock()

        result = await repository.update(
            sample_collection.id,
            {"name": "updated-name", "description": "Updated description"},
        )

        assert result.name == "updated-name"
        assert result.description == "Updated description"

    @pytest.mark.asyncio()
    async def test_update_invalid_fields_error(self, repository, mock_session, sample_collection) -> None:
        """Test update raises ValidationError for invalid fields."""
        repository.get_by_uuid = AsyncMock(return_value=sample_collection)

        with pytest.raises(ValidationError, match="Invalid fields"):
            await repository.update(sample_collection.id, {"invalid_field": "value"})

    @pytest.mark.asyncio()
    async def test_update_negative_chunk_size_error(self, repository, mock_session, sample_collection) -> None:
        """Test update raises ValidationError for non-positive chunk size."""
        repository.get_by_uuid = AsyncMock(return_value=sample_collection)

        with pytest.raises(ValidationError, match="Chunk size must be positive"):
            await repository.update(sample_collection.id, {"chunk_size": 0})

    @pytest.mark.asyncio()
    async def test_update_negative_chunk_overlap_error(self, repository, mock_session, sample_collection) -> None:
        """Test update raises ValidationError for negative chunk overlap."""
        repository.get_by_uuid = AsyncMock(return_value=sample_collection)

        with pytest.raises(ValidationError, match="Chunk overlap cannot be negative"):
            await repository.update(sample_collection.id, {"chunk_overlap": -1})

    @pytest.mark.asyncio()
    async def test_update_chunk_overlap_exceeds_size_error(self, repository, mock_session, sample_collection) -> None:
        """Test update raises ValidationError when overlap >= chunk_size."""
        repository.get_by_uuid = AsyncMock(return_value=sample_collection)
        sample_collection.chunk_size = 100

        with pytest.raises(ValidationError, match="Chunk overlap must be less than chunk size"):
            await repository.update(sample_collection.id, {"chunk_overlap": 100})

    @pytest.mark.asyncio()
    async def test_update_not_found(self, repository, mock_session) -> None:
        """Test update raises EntityNotFoundError for missing collection."""
        repository.get_by_uuid = AsyncMock(return_value=None)

        with pytest.raises(EntityNotFoundError):
            await repository.update(str(uuid4()), {"name": "test"})

    # --- get_document_count Tests ---

    @pytest.mark.asyncio()
    async def test_get_document_count_success(self, repository, mock_session) -> None:
        """Test get_document_count returns correct count."""
        mock_session.scalar = AsyncMock(return_value=10)

        result = await repository.get_document_count(str(uuid4()))

        assert result == 10

    @pytest.mark.asyncio()
    async def test_get_document_count_returns_zero_when_none(self, repository, mock_session) -> None:
        """Test get_document_count returns 0 when scalar returns None."""
        mock_session.scalar = AsyncMock(return_value=None)

        result = await repository.get_document_count(str(uuid4()))

        assert result == 0

    @pytest.mark.asyncio()
    async def test_get_document_count_database_error(self, repository, mock_session) -> None:
        """Test get_document_count raises DatabaseOperationError on failure."""
        mock_session.scalar = AsyncMock(side_effect=Exception("Connection error"))

        with pytest.raises(DatabaseOperationError):
            await repository.get_document_count(str(uuid4()))

    # --- get_due_for_sync Tests ---

    @pytest.mark.asyncio()
    async def test_get_due_for_sync_returns_collections(self, repository, mock_session, sample_collection) -> None:
        """Test get_due_for_sync returns collections due for sync."""
        sample_collection.sync_mode = "continuous"
        sample_collection.sync_paused_at = None
        sample_collection.sync_next_run_at = datetime.now(UTC) - timedelta(hours=1)
        sample_collection.status = CollectionStatus.READY

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_collection]
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repository.get_due_for_sync()

        assert len(result) == 1
        assert result[0].sync_mode == "continuous"

    @pytest.mark.asyncio()
    async def test_get_due_for_sync_database_error(self, repository, mock_session) -> None:
        """Test get_due_for_sync raises DatabaseOperationError on failure."""
        mock_session.execute = AsyncMock(side_effect=Exception("Connection error"))

        with pytest.raises(DatabaseOperationError):
            await repository.get_due_for_sync()

    # --- update_sync_status Tests ---

    @pytest.mark.asyncio()
    async def test_update_sync_status_success(self, repository, mock_session, sample_collection) -> None:
        """Test update_sync_status updates sync tracking fields."""
        repository.get_by_uuid = AsyncMock(return_value=sample_collection)
        mock_session.flush = AsyncMock()

        now = datetime.now(UTC)
        result = await repository.update_sync_status(
            sample_collection.id,
            status="success",
            started_at=now - timedelta(hours=1),
            completed_at=now,
        )

        assert result.sync_last_run_status == "success"
        assert result.sync_last_run_started_at is not None
        assert result.sync_last_run_completed_at is not None

    @pytest.mark.asyncio()
    async def test_update_sync_status_with_error(self, repository, mock_session, sample_collection) -> None:
        """Test update_sync_status records error message."""
        repository.get_by_uuid = AsyncMock(return_value=sample_collection)
        mock_session.flush = AsyncMock()

        result = await repository.update_sync_status(
            sample_collection.id,
            status="failed",
            error="Connection timeout",
        )

        assert result.sync_last_run_status == "failed"
        assert result.sync_last_error == "Connection timeout"

    @pytest.mark.asyncio()
    async def test_update_sync_status_not_found(self, repository, mock_session) -> None:
        """Test update_sync_status raises EntityNotFoundError for missing collection."""
        repository.get_by_uuid = AsyncMock(return_value=None)

        with pytest.raises(EntityNotFoundError):
            await repository.update_sync_status(str(uuid4()), status="success")

    # --- set_next_sync_run Tests ---

    @pytest.mark.asyncio()
    async def test_set_next_sync_run_explicit_time(self, repository, mock_session, sample_collection) -> None:
        """Test set_next_sync_run with explicit next run time."""
        repository.get_by_uuid = AsyncMock(return_value=sample_collection)
        mock_session.flush = AsyncMock()

        next_run = datetime.now(UTC) + timedelta(hours=2)
        result = await repository.set_next_sync_run(sample_collection.id, next_run)

        assert result.sync_next_run_at == next_run

    @pytest.mark.asyncio()
    async def test_set_next_sync_run_calculated_time(self, repository, mock_session, sample_collection) -> None:
        """Test set_next_sync_run calculates time from interval."""
        sample_collection.sync_interval_minutes = 30
        repository.get_by_uuid = AsyncMock(return_value=sample_collection)
        mock_session.flush = AsyncMock()

        before = datetime.now(UTC)
        result = await repository.set_next_sync_run(sample_collection.id)
        after = datetime.now(UTC) + timedelta(minutes=30)

        assert result.sync_next_run_at is not None
        assert result.sync_next_run_at >= before + timedelta(minutes=30)
        assert result.sync_next_run_at <= after + timedelta(seconds=1)

    @pytest.mark.asyncio()
    async def test_set_next_sync_run_not_found(self, repository, mock_session) -> None:
        """Test set_next_sync_run raises EntityNotFoundError for missing collection."""
        repository.get_by_uuid = AsyncMock(return_value=None)

        with pytest.raises(EntityNotFoundError):
            await repository.set_next_sync_run(str(uuid4()))

    # --- create Tests (validation) ---

    @pytest.mark.asyncio()
    async def test_create_duplicate_name_error(self, repository, mock_session) -> None:
        """Test create raises EntityAlreadyExistsError for duplicate name."""
        existing_collection = Collection(id=str(uuid4()), name="existing", owner_id=1)

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = existing_collection
        mock_session.execute = AsyncMock(return_value=mock_result)

        with pytest.raises(EntityAlreadyExistsError):
            await repository.create(name="existing", owner_id=1)

    @pytest.mark.asyncio()
    async def test_create_success(self, repository, mock_session) -> None:
        """Test create successfully creates a collection."""
        # Mock that no existing collection found
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.add = MagicMock()
        mock_session.flush = AsyncMock()

        result = await repository.create(name="new-collection", owner_id=1)

        assert result.name == "new-collection"
        assert result.owner_id == 1
        assert result.status == CollectionStatus.PENDING
        mock_session.add.assert_called_once()

    # --- rename Tests (additional) ---

    @pytest.mark.asyncio()
    async def test_rename_non_owner_error(self, repository, mock_session, sample_collection) -> None:
        """Test rename raises AccessDeniedError when user is not the owner."""
        from shared.database.exceptions import AccessDeniedError

        sample_collection.owner_id = 1
        repository.get_by_name = AsyncMock(return_value=None)
        repository.get_by_uuid_with_permission_check = AsyncMock(return_value=sample_collection)
        mock_session.flush = AsyncMock()

        with pytest.raises(AccessDeniedError):
            await repository.rename(sample_collection.id, "new-name", 999)  # Different user

    # --- delete Tests (additional) ---

    @pytest.mark.asyncio()
    async def test_delete_success(self, repository, mock_session, sample_collection) -> None:
        """Test delete successfully removes a collection."""
        sample_collection.owner_id = 1
        repository.get_by_uuid = AsyncMock(return_value=sample_collection)
        mock_session.execute = AsyncMock()
        mock_session.flush = AsyncMock()

        await repository.delete(sample_collection.id, 1)

        # Verify delete was called
        mock_session.execute.assert_called_once()
        mock_session.flush.assert_called_once()

    # --- list_for_user Tests ---

    @pytest.mark.asyncio()
    async def test_list_for_user_with_pagination(self, repository, mock_session, sample_collection) -> None:
        """Test list_for_user returns paginated results."""
        mock_session.scalar = AsyncMock(return_value=10)

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_collection]
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute = AsyncMock(return_value=mock_result)

        collections, total = await repository.list_for_user(1, offset=0, limit=5)

        assert total == 10
        assert len(collections) == 1

    @pytest.mark.asyncio()
    async def test_list_for_user_database_error(self, repository, mock_session) -> None:
        """Test list_for_user raises DatabaseOperationError on failure."""
        mock_session.scalar = AsyncMock(side_effect=Exception("Connection error"))

        with pytest.raises(DatabaseOperationError):
            await repository.list_for_user(1)
