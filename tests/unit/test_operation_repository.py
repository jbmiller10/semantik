"""Unit tests for OperationRepository using mocks."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from shared.database.exceptions import (
    AccessDeniedError,
    DatabaseOperationError,
    EntityNotFoundError,
    ValidationError,
)
from shared.database.models import Collection, CollectionStatus, Operation, OperationStatus, OperationType
from shared.database.repositories.operation_repository import OperationRepository


class TestOperationRepository:
    """Unit tests for OperationRepository."""

    @pytest.fixture()
    def mock_session(self) -> AsyncMock:
        """Create a mock async session."""
        return AsyncMock()

    @pytest.fixture()
    def repository(self, mock_session) -> OperationRepository:
        """Create an OperationRepository instance with mocked session."""
        return OperationRepository(mock_session)

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

    @pytest.fixture()
    def sample_operation(self, sample_collection) -> Operation:
        """Create a sample operation for testing."""
        return Operation(
            uuid=str(uuid4()),
            collection_id=sample_collection.id,
            user_id=1,
            type=OperationType.INDEX,
            status=OperationStatus.PENDING,
            config={"source": "test"},
            meta={},
        )

    # --- create Tests ---

    @pytest.mark.asyncio()
    async def test_create_success(self, repository, mock_session, sample_collection) -> None:
        """Test create successfully creates an operation."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_collection
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.add = MagicMock()
        mock_session.flush = AsyncMock()

        result = await repository.create(
            collection_id=sample_collection.id,
            user_id=1,
            operation_type=OperationType.INDEX,
            config={"source": "test"},
        )

        assert result.collection_id == sample_collection.id
        assert result.user_id == 1
        assert result.type == OperationType.INDEX
        assert result.status == OperationStatus.PENDING
        mock_session.add.assert_called_once()

    @pytest.mark.asyncio()
    async def test_create_empty_config_error(self, repository, mock_session) -> None:
        """Test create raises ValidationError for empty config."""
        with pytest.raises(ValidationError, match="Operation config cannot be empty"):
            await repository.create(
                collection_id=str(uuid4()),
                user_id=1,
                operation_type=OperationType.INDEX,
                config={},
            )

    @pytest.mark.asyncio()
    async def test_create_collection_not_found(self, repository, mock_session) -> None:
        """Test create raises EntityNotFoundError for missing collection."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)

        with pytest.raises(EntityNotFoundError, match="collection"):
            await repository.create(
                collection_id=str(uuid4()),
                user_id=1,
                operation_type=OperationType.INDEX,
                config={"source": "test"},
            )

    @pytest.mark.asyncio()
    async def test_create_access_denied(self, repository, mock_session, sample_collection) -> None:
        """Test create raises AccessDeniedError for non-owner of private collection."""
        sample_collection.owner_id = 1
        sample_collection.is_public = False

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_collection
        mock_session.execute = AsyncMock(return_value=mock_result)

        with pytest.raises(AccessDeniedError):
            await repository.create(
                collection_id=sample_collection.id,
                user_id=999,  # Different user
                operation_type=OperationType.INDEX,
                config={"source": "test"},
            )

    @pytest.mark.asyncio()
    async def test_create_database_error(self, repository, mock_session, sample_collection) -> None:
        """Test create raises DatabaseOperationError on failure."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_collection
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.add = MagicMock()
        mock_session.flush = AsyncMock(side_effect=Exception("Connection error"))

        with pytest.raises(DatabaseOperationError):
            await repository.create(
                collection_id=sample_collection.id,
                user_id=1,
                operation_type=OperationType.INDEX,
                config={"source": "test"},
            )

    # --- get_by_uuid Tests ---

    @pytest.mark.asyncio()
    async def test_get_by_uuid_found(self, repository, mock_session, sample_operation) -> None:
        """Test get_by_uuid returns operation when found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_operation
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repository.get_by_uuid(sample_operation.uuid)

        assert result == sample_operation
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

    # --- get_by_uuid_with_permission_check Tests ---

    @pytest.mark.asyncio()
    async def test_get_by_uuid_with_permission_check_owner(
        self, repository, mock_session, sample_operation, sample_collection
    ) -> None:
        """Test get_by_uuid_with_permission_check allows operation owner."""
        sample_operation.collection = sample_collection
        repository.get_by_uuid = AsyncMock(return_value=sample_operation)
        mock_session.refresh = AsyncMock()

        result = await repository.get_by_uuid_with_permission_check(sample_operation.uuid, 1)

        assert result == sample_operation

    @pytest.mark.asyncio()
    async def test_get_by_uuid_with_permission_check_not_found(self, repository, mock_session) -> None:
        """Test get_by_uuid_with_permission_check raises EntityNotFoundError for missing operation."""
        repository.get_by_uuid = AsyncMock(return_value=None)

        with pytest.raises(EntityNotFoundError, match="operation"):
            await repository.get_by_uuid_with_permission_check(str(uuid4()), 1)

    @pytest.mark.asyncio()
    async def test_get_by_uuid_with_permission_check_access_denied(
        self, repository, mock_session, sample_operation, sample_collection
    ) -> None:
        """Test get_by_uuid_with_permission_check raises AccessDeniedError for non-owner."""
        sample_operation.user_id = 1
        sample_collection.owner_id = 1
        sample_collection.is_public = False
        sample_operation.collection = sample_collection

        repository.get_by_uuid = AsyncMock(return_value=sample_operation)
        mock_session.refresh = AsyncMock()

        with pytest.raises(AccessDeniedError):
            await repository.get_by_uuid_with_permission_check(sample_operation.uuid, 999)

    # --- set_task_id Tests ---

    @pytest.mark.asyncio()
    async def test_set_task_id_success(self, repository, mock_session, sample_operation) -> None:
        """Test set_task_id updates task_id."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_operation
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.flush = AsyncMock()

        result = await repository.set_task_id(sample_operation.uuid, "task-123")

        assert result == sample_operation
        mock_session.flush.assert_called_once()

    @pytest.mark.asyncio()
    async def test_set_task_id_not_found(self, repository, mock_session) -> None:
        """Test set_task_id raises EntityNotFoundError for missing operation."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)

        with pytest.raises(EntityNotFoundError, match="operation"):
            await repository.set_task_id(str(uuid4()), "task-123")

    @pytest.mark.asyncio()
    async def test_set_task_id_database_error(self, repository, mock_session, sample_operation) -> None:
        """Test set_task_id raises DatabaseOperationError on failure."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_operation
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.flush = AsyncMock(side_effect=Exception("Connection error"))

        with pytest.raises(DatabaseOperationError):
            await repository.set_task_id(sample_operation.uuid, "task-123")

    # --- update_status Tests ---

    @pytest.mark.asyncio()
    async def test_update_status_success(self, repository, mock_session, sample_operation) -> None:
        """Test update_status updates operation status."""
        repository.get_by_uuid = AsyncMock(return_value=sample_operation)
        mock_session.flush = AsyncMock()

        result = await repository.update_status(sample_operation.uuid, OperationStatus.PROCESSING)

        assert result.status == OperationStatus.PROCESSING
        assert result.started_at is not None

    @pytest.mark.asyncio()
    async def test_update_status_completed(self, repository, mock_session, sample_operation) -> None:
        """Test update_status sets completed_at for terminal statuses."""
        sample_operation.started_at = datetime.now(UTC)
        repository.get_by_uuid = AsyncMock(return_value=sample_operation)
        mock_session.flush = AsyncMock()

        result = await repository.update_status(sample_operation.uuid, OperationStatus.COMPLETED)

        assert result.status == OperationStatus.COMPLETED
        assert result.completed_at is not None

    @pytest.mark.asyncio()
    async def test_update_status_with_error_message(self, repository, mock_session, sample_operation) -> None:
        """Test update_status sets error_message."""
        sample_operation.started_at = datetime.now(UTC)
        repository.get_by_uuid = AsyncMock(return_value=sample_operation)
        mock_session.flush = AsyncMock()

        result = await repository.update_status(
            sample_operation.uuid, OperationStatus.FAILED, error_message="Test error"
        )

        assert result.status == OperationStatus.FAILED
        assert result.error_message == "Test error"
        assert result.completed_at is not None

    @pytest.mark.asyncio()
    async def test_update_status_not_found(self, repository, mock_session) -> None:
        """Test update_status raises EntityNotFoundError for missing operation."""
        repository.get_by_uuid = AsyncMock(return_value=None)

        with pytest.raises(EntityNotFoundError, match="operation"):
            await repository.update_status(str(uuid4()), OperationStatus.PROCESSING)

    @pytest.mark.asyncio()
    async def test_update_status_database_error(self, repository, mock_session, sample_operation) -> None:
        """Test update_status raises DatabaseOperationError on failure."""
        repository.get_by_uuid = AsyncMock(return_value=sample_operation)
        mock_session.flush = AsyncMock(side_effect=Exception("Connection error"))

        with pytest.raises(DatabaseOperationError):
            await repository.update_status(sample_operation.uuid, OperationStatus.PROCESSING)

    # --- list_for_collection Tests ---

    @pytest.mark.asyncio()
    async def test_list_for_collection_success(
        self, repository, mock_session, sample_operation, sample_collection
    ) -> None:
        """Test list_for_collection returns operations."""
        # Mock collection lookup
        collection_result = MagicMock()
        collection_result.scalar_one_or_none.return_value = sample_collection
        mock_session.execute = AsyncMock(return_value=collection_result)

        # Mock count
        mock_session.scalar = AsyncMock(return_value=1)

        # Mock operations result
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_operation]
        ops_result = MagicMock()
        ops_result.scalars.return_value = mock_scalars

        # Set up execute to return different values
        mock_session.execute = AsyncMock(side_effect=[collection_result, ops_result])

        operations, total = await repository.list_for_collection(sample_collection.id, 1)

        assert total == 1
        assert len(operations) == 1

    @pytest.mark.asyncio()
    async def test_list_for_collection_not_found(self, repository, mock_session) -> None:
        """Test list_for_collection raises EntityNotFoundError for missing collection."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)

        with pytest.raises(EntityNotFoundError, match="collection"):
            await repository.list_for_collection(str(uuid4()), 1)

    @pytest.mark.asyncio()
    async def test_list_for_collection_access_denied(self, repository, mock_session, sample_collection) -> None:
        """Test list_for_collection raises AccessDeniedError for non-owner."""
        sample_collection.owner_id = 1
        sample_collection.is_public = False

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_collection
        mock_session.execute = AsyncMock(return_value=mock_result)

        with pytest.raises(AccessDeniedError):
            await repository.list_for_collection(sample_collection.id, 999)

    # --- list_for_user Tests ---

    @pytest.mark.asyncio()
    async def test_list_for_user_success(self, repository, mock_session, sample_operation) -> None:
        """Test list_for_user returns operations for user."""
        mock_session.scalar = AsyncMock(return_value=1)

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_operation]
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute = AsyncMock(return_value=mock_result)

        operations, total = await repository.list_for_user(1)

        assert total == 1
        assert len(operations) == 1

    @pytest.mark.asyncio()
    async def test_list_for_user_with_status_filter(self, repository, mock_session, sample_operation) -> None:
        """Test list_for_user filters by status."""
        mock_session.scalar = AsyncMock(return_value=1)

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_operation]
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute = AsyncMock(return_value=mock_result)

        operations, total = await repository.list_for_user(1, status=OperationStatus.PENDING)

        assert total == 1

    @pytest.mark.asyncio()
    async def test_list_for_user_with_status_list(self, repository, mock_session, sample_operation) -> None:
        """Test list_for_user filters by status list."""
        mock_session.scalar = AsyncMock(return_value=2)

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_operation]
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute = AsyncMock(return_value=mock_result)

        operations, total = await repository.list_for_user(
            1, status_list=[OperationStatus.PENDING, OperationStatus.PROCESSING]
        )

        assert total == 2

    @pytest.mark.asyncio()
    async def test_list_for_user_database_error(self, repository, mock_session) -> None:
        """Test list_for_user raises DatabaseOperationError on failure."""
        mock_session.scalar = AsyncMock(side_effect=Exception("Connection error"))

        with pytest.raises(DatabaseOperationError):
            await repository.list_for_user(1)

    # --- cancel Tests ---

    @pytest.mark.asyncio()
    async def test_cancel_success(self, repository, mock_session, sample_operation, sample_collection) -> None:
        """Test cancel updates operation to cancelled status."""
        sample_operation.status = OperationStatus.PENDING
        sample_operation.collection = sample_collection
        repository.get_by_uuid_with_permission_check = AsyncMock(return_value=sample_operation)
        mock_session.flush = AsyncMock()

        result = await repository.cancel(sample_operation.uuid, 1)

        assert result.status == OperationStatus.CANCELLED
        assert result.completed_at is not None

    @pytest.mark.asyncio()
    async def test_cancel_processing_operation(
        self, repository, mock_session, sample_operation, sample_collection
    ) -> None:
        """Test cancel can cancel processing operation."""
        sample_operation.status = OperationStatus.PROCESSING
        sample_operation.collection = sample_collection
        repository.get_by_uuid_with_permission_check = AsyncMock(return_value=sample_operation)
        mock_session.flush = AsyncMock()

        result = await repository.cancel(sample_operation.uuid, 1)

        assert result.status == OperationStatus.CANCELLED

    @pytest.mark.asyncio()
    async def test_cancel_completed_operation_error(
        self, repository, mock_session, sample_operation, sample_collection
    ) -> None:
        """Test cancel raises ValidationError for completed operation."""
        sample_operation.status = OperationStatus.COMPLETED
        sample_operation.collection = sample_collection
        repository.get_by_uuid_with_permission_check = AsyncMock(return_value=sample_operation)

        with pytest.raises(ValidationError, match="Cannot cancel operation"):
            await repository.cancel(sample_operation.uuid, 1)

    @pytest.mark.asyncio()
    async def test_cancel_not_found(self, repository, mock_session) -> None:
        """Test cancel raises EntityNotFoundError for missing operation."""
        repository.get_by_uuid_with_permission_check = AsyncMock(
            side_effect=EntityNotFoundError("operation", "test-id")
        )

        with pytest.raises(EntityNotFoundError):
            await repository.cancel(str(uuid4()), 1)

    # --- get_active_operations_count Tests ---

    @pytest.mark.asyncio()
    async def test_get_active_operations_count_success(self, repository, mock_session) -> None:
        """Test get_active_operations_count returns count."""
        mock_session.scalar = AsyncMock(return_value=5)

        result = await repository.get_active_operations_count(str(uuid4()))

        assert result == 5

    @pytest.mark.asyncio()
    async def test_get_active_operations_count_returns_zero_when_none(self, repository, mock_session) -> None:
        """Test get_active_operations_count returns 0 when scalar returns None."""
        mock_session.scalar = AsyncMock(return_value=None)

        result = await repository.get_active_operations_count(str(uuid4()))

        assert result == 0

    @pytest.mark.asyncio()
    async def test_get_active_operations_count_database_error(self, repository, mock_session) -> None:
        """Test get_active_operations_count raises DatabaseOperationError on failure."""
        mock_session.scalar = AsyncMock(side_effect=Exception("Connection error"))

        with pytest.raises(DatabaseOperationError):
            await repository.get_active_operations_count(str(uuid4()))

    # --- get_active_operations Tests ---

    @pytest.mark.asyncio()
    async def test_get_active_operations_success(self, repository, mock_session, sample_operation) -> None:
        """Test get_active_operations returns active operations."""
        sample_operation.status = OperationStatus.PROCESSING

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [sample_operation]
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repository.get_active_operations(str(uuid4()))

        assert len(result) == 1
        assert result[0].status == OperationStatus.PROCESSING

    @pytest.mark.asyncio()
    async def test_get_active_operations_empty(self, repository, mock_session) -> None:
        """Test get_active_operations returns empty list when none active."""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await repository.get_active_operations(str(uuid4()))

        assert len(result) == 0

    @pytest.mark.asyncio()
    async def test_get_active_operations_database_error(self, repository, mock_session) -> None:
        """Test get_active_operations raises DatabaseOperationError on failure."""
        mock_session.execute = AsyncMock(side_effect=Exception("Connection error"))

        with pytest.raises(DatabaseOperationError):
            await repository.get_active_operations(str(uuid4()))
