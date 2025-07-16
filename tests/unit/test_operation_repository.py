"""Unit tests for OperationRepository."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from shared.database.exceptions import (
    AccessDeniedError,
    DatabaseOperationError,
    EntityNotFoundError,
    ValidationError,
)
from shared.database.models import Collection, Operation, OperationStatus, OperationType
from shared.database.repositories.operation_repository import OperationRepository


class TestOperationRepository:
    """Test cases for OperationRepository."""

    @pytest.fixture()
    def mock_session(self):
        """Create a mock async session."""
        session = AsyncMock()
        # Make execute return completed coroutines immediately
        session.execute = AsyncMock()
        session.scalar = AsyncMock()
        session.add = MagicMock()
        session.delete = MagicMock()
        session.flush = AsyncMock()
        session.refresh = AsyncMock()
        return session

    @pytest.fixture()
    def repository(self, mock_session):
        """Create an OperationRepository instance with mocked session."""
        return OperationRepository(mock_session)

    @pytest.fixture()
    def sample_collection(self):
        """Create a sample collection for testing."""
        return Collection(
            id=str(uuid4()),
            name="test-collection",
            owner_id=1,
            is_public=False,
            vector_store_name="vec_store_1",
            embedding_model="model1",
            chunk_size=1000,
            chunk_overlap=200,
        )

    @pytest.fixture()
    def sample_operation(self, sample_collection):
        """Create a sample operation for testing."""
        operation = Operation(
            id=1,
            uuid=str(uuid4()),
            collection_id=sample_collection.id,
            user_id=1,
            type=OperationType.INDEX,
            status=OperationStatus.PENDING,
            config={"source_path": "/data/test"},
            created_at=datetime.now(UTC),
        )
        operation.collection = sample_collection
        return operation

    @pytest.mark.asyncio()
    async def test_create_operation_success(self, repository, mock_session, sample_collection):
        """Test successful operation creation."""
        # Setup
        user_id = 1
        operation_type = OperationType.INDEX
        config = {"source_path": "/data/test"}

        # Mock collection exists
        collection_result = MagicMock()
        collection_result.scalar_one_or_none.return_value = sample_collection
        mock_session.execute.return_value = collection_result

        # Mock UUID generation
        with patch("shared.database.repositories.operation_repository.uuid4") as mock_uuid:
            mock_uuid.return_value = "test-operation-uuid"

            # Act
            operation = await repository.create(
                collection_id=sample_collection.id,
                user_id=user_id,
                operation_type=operation_type,
                config=config,
            )

        # Assert
        assert operation.uuid == "test-operation-uuid"
        assert operation.collection_id == sample_collection.id
        assert operation.user_id == user_id
        assert operation.type == operation_type
        assert operation.status == OperationStatus.PENDING
        assert operation.config == config
        mock_session.add.assert_called_once()
        mock_session.flush.assert_called_once()

    @pytest.mark.asyncio()
    async def test_create_operation_empty_config(self, repository, mock_session):
        """Test operation creation with empty config."""
        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            await repository.create(
                collection_id="test-id",
                user_id=1,
                operation_type=OperationType.INDEX,
                config={},
            )
        assert "Operation config cannot be empty" in str(exc_info.value)

    @pytest.mark.asyncio()
    async def test_create_operation_collection_not_found(self, repository, mock_session):
        """Test operation creation with non-existent collection."""
        # Setup
        collection_result = MagicMock()
        collection_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = collection_result

        # Act & Assert
        with pytest.raises(EntityNotFoundError) as exc_info:
            await repository.create(
                collection_id="nonexistent",
                user_id=1,
                operation_type=OperationType.INDEX,
                config={"path": "/test"},
            )
        assert "collection" in str(exc_info.value)

    @pytest.mark.asyncio()
    async def test_create_operation_access_denied(self, repository, mock_session, sample_collection):
        """Test operation creation without access to collection."""
        # Setup - private collection with different owner
        sample_collection.owner_id = 999
        sample_collection.is_public = False

        collection_result = MagicMock()
        collection_result.scalar_one_or_none.return_value = sample_collection
        mock_session.execute.return_value = collection_result

        # Act & Assert
        with pytest.raises(AccessDeniedError) as exc_info:
            await repository.create(
                collection_id=sample_collection.id,
                user_id=1,
                operation_type=OperationType.INDEX,
                config={"path": "/test"},
            )
        assert "1" in str(exc_info.value)
        assert sample_collection.id in str(exc_info.value)

    @pytest.mark.asyncio()
    async def test_create_operation_public_collection(self, repository, mock_session, sample_collection):
        """Test operation creation on public collection."""
        # Setup - public collection with different owner
        sample_collection.owner_id = 999
        sample_collection.is_public = True

        collection_result = MagicMock()
        collection_result.scalar_one_or_none.return_value = sample_collection
        mock_session.execute.return_value = collection_result

        # Act
        operation = await repository.create(
            collection_id=sample_collection.id,
            user_id=1,
            operation_type=OperationType.INDEX,
            config={"path": "/test"},
        )

        # Assert
        assert operation is not None
        mock_session.add.assert_called_once()

    @pytest.mark.asyncio()
    async def test_get_by_uuid(self, repository, mock_session, sample_operation):
        """Test getting operation by UUID."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_operation
        mock_session.execute.return_value = mock_result

        # Act
        result = await repository.get_by_uuid(sample_operation.uuid)

        # Assert
        assert result == sample_operation

    @pytest.mark.asyncio()
    async def test_get_by_uuid_not_found(self, repository, mock_session):
        """Test getting non-existent operation by UUID."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        # Act
        result = await repository.get_by_uuid("nonexistent")

        # Assert
        assert result is None

    @pytest.mark.asyncio()
    async def test_get_by_uuid_with_permission_check_owner(self, repository, mock_session, sample_operation):
        """Test getting operation with permission check as owner."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_operation
        mock_session.execute.return_value = mock_result

        # Act
        result = await repository.get_by_uuid_with_permission_check(sample_operation.uuid, sample_operation.user_id)

        # Assert
        assert result == sample_operation

    @pytest.mark.asyncio()
    async def test_get_by_uuid_with_permission_check_collection_owner(self, repository, mock_session, sample_operation):
        """Test getting operation with permission check as collection owner."""
        # Setup - different user but owns the collection
        sample_operation.user_id = 999
        sample_operation.collection.owner_id = 1

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_operation
        mock_session.execute.return_value = mock_result

        # Act
        result = await repository.get_by_uuid_with_permission_check(sample_operation.uuid, 1)

        # Assert
        assert result == sample_operation

    @pytest.mark.asyncio()
    async def test_get_by_uuid_with_permission_check_public_collection(
        self, repository, mock_session, sample_operation
    ):
        """Test getting operation on public collection."""
        # Setup
        sample_operation.user_id = 999
        sample_operation.collection.owner_id = 999
        sample_operation.collection.is_public = True

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_operation
        mock_session.execute.return_value = mock_result

        # Act
        result = await repository.get_by_uuid_with_permission_check(sample_operation.uuid, 1)

        # Assert
        assert result == sample_operation

    @pytest.mark.asyncio()
    async def test_get_by_uuid_with_permission_check_denied(self, repository, mock_session, sample_operation):
        """Test permission denied for operation."""
        # Setup
        sample_operation.user_id = 999
        sample_operation.collection.owner_id = 999
        sample_operation.collection.is_public = False

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_operation
        mock_session.execute.return_value = mock_result

        # Act & Assert
        with pytest.raises(AccessDeniedError):
            await repository.get_by_uuid_with_permission_check(sample_operation.uuid, 1)

    @pytest.mark.asyncio()
    async def test_set_task_id(self, repository, mock_session, sample_operation):
        """Test setting task ID for an operation."""
        # Setup
        task_id = "celery-task-123"
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_operation
        mock_session.execute.return_value = mock_result

        # Act
        result = await repository.set_task_id(sample_operation.uuid, task_id)

        # Assert
        assert result == sample_operation
        mock_session.execute.assert_called_once()
        mock_session.flush.assert_called_once()

    @pytest.mark.asyncio()
    async def test_set_task_id_not_found(self, repository, mock_session):
        """Test setting task ID for non-existent operation."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        # Act & Assert
        with pytest.raises(EntityNotFoundError):
            await repository.set_task_id("nonexistent", "task-id")

    @pytest.mark.asyncio()
    async def test_update_status(self, repository, mock_session, sample_operation):
        """Test updating operation status."""
        # Setup
        # Mock get_by_uuid to return the operation
        repository.get_by_uuid = AsyncMock(return_value=sample_operation)

        # Act
        result = await repository.update_status(
            sample_operation.uuid,
            OperationStatus.PROCESSING,
            error_message=None,
            started_at=datetime.now(UTC),
        )

        # Assert
        assert result.status == OperationStatus.PROCESSING
        assert result.started_at is not None
        mock_session.flush.assert_called_once()

    @pytest.mark.asyncio()
    async def test_update_status_auto_timestamps(self, repository, mock_session, sample_operation):
        """Test automatic timestamp setting based on status."""
        # Setup for PROCESSING status
        sample_operation.started_at = None
        repository.get_by_uuid = AsyncMock(return_value=sample_operation)

        # Act
        await repository.update_status(sample_operation.uuid, OperationStatus.PROCESSING)

        # Assert - should auto-set started_at
        assert sample_operation.started_at is not None

        # Setup for COMPLETED status
        sample_operation.completed_at = None

        # Act
        await repository.update_status(sample_operation.uuid, OperationStatus.COMPLETED)

        # Assert - should auto-set completed_at
        assert sample_operation.completed_at is not None

    @pytest.mark.asyncio()
    async def test_list_for_collection(self, repository, mock_session, sample_collection):
        """Test listing operations for a collection."""
        # Setup
        operations = [
            Operation(
                id=i,
                uuid=str(uuid4()),
                collection_id=sample_collection.id,
                user_id=1,
                type=OperationType.INDEX,
                status=OperationStatus.COMPLETED,
                config={},
            )
            for i in range(3)
        ]

        # Mock collection exists
        collection_result = MagicMock()
        collection_result.scalar_one_or_none.return_value = sample_collection

        # Mock count
        mock_session.scalar.return_value = 3

        # Mock operations query
        operations_result = MagicMock()
        operations_result.scalars.return_value.all.return_value = operations

        mock_session.execute.side_effect = [collection_result, operations_result]

        # Act
        result_operations, total = await repository.list_for_collection(sample_collection.id, user_id=1)

        # Assert
        assert len(result_operations) == 3
        assert total == 3

    @pytest.mark.asyncio()
    async def test_list_for_collection_with_filters(self, repository, mock_session, sample_collection):
        """Test listing operations with status and type filters."""
        # Setup
        operations = [
            Operation(
                id=1,
                uuid=str(uuid4()),
                collection_id=sample_collection.id,
                user_id=1,
                type=OperationType.REINDEX,
                status=OperationStatus.PROCESSING,
                config={},
            )
        ]

        # Mock collection
        collection_result = MagicMock()
        collection_result.scalar_one_or_none.return_value = sample_collection

        # Mock count
        mock_session.scalar.return_value = 1

        # Mock filtered operations
        operations_result = MagicMock()
        operations_result.scalars.return_value.all.return_value = operations

        mock_session.execute.side_effect = [collection_result, operations_result]

        # Act
        result_operations, total = await repository.list_for_collection(
            sample_collection.id,
            user_id=1,
            status=OperationStatus.PROCESSING,
            operation_type=OperationType.REINDEX,
        )

        # Assert
        assert len(result_operations) == 1
        assert total == 1

    @pytest.mark.asyncio()
    async def test_list_for_collection_access_denied(self, repository, mock_session, sample_collection):
        """Test listing operations without access to collection."""
        # Setup
        sample_collection.owner_id = 999
        sample_collection.is_public = False

        collection_result = MagicMock()
        collection_result.scalar_one_or_none.return_value = sample_collection
        mock_session.execute.return_value = collection_result

        # Act & Assert
        with pytest.raises(AccessDeniedError):
            await repository.list_for_collection(sample_collection.id, user_id=1)

    @pytest.mark.asyncio()
    async def test_list_for_user(self, repository, mock_session):
        """Test listing operations for a user."""
        # Setup
        user_id = 1
        operations = [
            Operation(
                id=i,
                uuid=str(uuid4()),
                collection_id=str(uuid4()),
                user_id=user_id,
                type=OperationType.INDEX,
                status=OperationStatus.COMPLETED,
                config={},
            )
            for i in range(5)
        ]

        # Mock count
        mock_session.scalar.return_value = 5

        # Mock operations
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = operations
        mock_session.execute.return_value = mock_result

        # Act
        result_operations, total = await repository.list_for_user(user_id)

        # Assert
        assert len(result_operations) == 5
        assert total == 5

    @pytest.mark.asyncio()
    async def test_cancel_pending_operation(self, repository, mock_session, sample_operation):
        """Test cancelling a pending operation."""
        # Setup
        sample_operation.status = OperationStatus.PENDING
        repository.get_by_uuid_with_permission_check = AsyncMock(return_value=sample_operation)

        # Act
        result = await repository.cancel(sample_operation.uuid, sample_operation.user_id)

        # Assert
        assert result.status == OperationStatus.CANCELLED
        assert result.completed_at is not None
        mock_session.flush.assert_called_once()

    @pytest.mark.asyncio()
    async def test_cancel_processing_operation(self, repository, mock_session, sample_operation):
        """Test cancelling a processing operation."""
        # Setup
        sample_operation.status = OperationStatus.PROCESSING
        repository.get_by_uuid_with_permission_check = AsyncMock(return_value=sample_operation)

        # Act
        result = await repository.cancel(sample_operation.uuid, sample_operation.user_id)

        # Assert
        assert result.status == OperationStatus.CANCELLED

    @pytest.mark.asyncio()
    async def test_cancel_completed_operation(self, repository, mock_session, sample_operation):
        """Test cancelling an already completed operation."""
        # Setup
        sample_operation.status = OperationStatus.COMPLETED
        repository.get_by_uuid_with_permission_check = AsyncMock(return_value=sample_operation)

        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            await repository.cancel(sample_operation.uuid, sample_operation.user_id)
        assert "Cannot cancel operation in completed status" in str(exc_info.value)

    @pytest.mark.asyncio()
    async def test_get_active_operations_count(self, repository, mock_session):
        """Test getting count of active operations."""
        # Setup
        mock_session.scalar.return_value = 5

        # Act
        count = await repository.get_active_operations_count("collection-id")

        # Assert
        assert count == 5
        mock_session.scalar.assert_called_once()

    @pytest.mark.asyncio()
    async def test_database_operation_error_handling(self, repository, mock_session):
        """Test handling of unexpected database errors."""
        # Setup
        mock_session.execute.side_effect = Exception("Database connection lost")

        # Act & Assert
        with pytest.raises(DatabaseOperationError) as exc_info:
            await repository.get_by_uuid("test-id")
        assert "Database connection lost" in str(exc_info.value)
