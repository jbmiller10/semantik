"""Unit tests for CollectionRepository."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from shared.database.exceptions import (
    AccessDeniedError,
    DatabaseOperationError,
    EntityAlreadyExistsError,
    EntityNotFoundError,
    ValidationError,
)
from shared.database.models import Collection, CollectionStatus
from shared.database.repositories.collection_repository import CollectionRepository
from sqlalchemy.exc import IntegrityError


class TestCollectionRepository:
    """Test cases for CollectionRepository."""

    @pytest.fixture()
    def mock_session(self) -> None:
        """Create a mock async session."""
        session = AsyncMock()
        # Make execute return completed coroutines immediately
        session.execute = AsyncMock()
        session.scalar = AsyncMock()
        session.add = MagicMock()
        session.delete = MagicMock()
        session.flush = AsyncMock()
        return session

    @pytest.fixture()
    def repository(self, mock_session) -> None:
        """Create a CollectionRepository instance with mocked session."""
        return CollectionRepository(mock_session)

    @pytest.fixture()
    def sample_collection(self) -> None:
        """Create a sample collection for testing."""
        return Collection(
            id=str(uuid4()),
            name="test-collection",
            owner_id=1,
            description="Test collection",
            embedding_model="model1",
            chunk_size=1000,
            chunk_overlap=200,
            is_public=False,
            vector_store_name="vec_store_1",
            status=CollectionStatus.READY,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

    @pytest.mark.asyncio()
    async def test_create_collection_success(self, repository, mock_session):
        """Test successful collection creation."""
        # Setup
        user_id = 1
        name = "new-collection"
        description = "Test description"

        # Mock no existing collection
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        # Mock UUID generation
        with patch("shared.database.repositories.collection_repository.uuid4") as mock_uuid:
            mock_uuid.return_value = "test-uuid-1234"

            # Act
            collection = await repository.create(
                name=name,
                owner_id=user_id,
                description=description,
            )

        # Assert
        assert collection.name == name
        assert collection.owner_id == user_id
        assert collection.description == description
        assert collection.status == CollectionStatus.PENDING
        assert collection.vector_store_name == "col_test_uuid_1234"
        mock_session.add.assert_called_once()
        mock_session.flush.assert_called_once()

    @pytest.mark.asyncio()
    async def test_create_collection_name_already_exists(self, repository, mock_session, sample_collection):
        """Test collection creation with duplicate name."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_collection
        mock_session.execute.return_value = mock_result

        # Act & Assert
        with pytest.raises(DatabaseOperationError) as exc_info:
            await repository.create(name="test-collection", owner_id=1)
        # The EntityAlreadyExistsError gets wrapped in DatabaseOperationError
        assert "Failed to create collection" in str(exc_info.value)

    @pytest.mark.asyncio()
    async def test_create_collection_validation_errors(self, repository, mock_session):
        """Test validation errors during collection creation."""
        # Mock no existing collection
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        # Test invalid chunk size
        with pytest.raises(DatabaseOperationError) as exc_info:
            await repository.create(name="test", owner_id=1, chunk_size=0)
        # The ValidationError gets wrapped in DatabaseOperationError
        assert "Failed to create collection" in str(exc_info.value)
        assert "Chunk size must be positive" in str(exc_info.value)

        # Test negative chunk overlap
        with pytest.raises(DatabaseOperationError) as exc_info:
            await repository.create(name="test", owner_id=1, chunk_overlap=-1)
        assert "Failed to create collection" in str(exc_info.value)
        assert "Chunk overlap cannot be negative" in str(exc_info.value)

        # Test chunk overlap >= chunk size
        with pytest.raises(DatabaseOperationError) as exc_info:
            await repository.create(name="test", owner_id=1, chunk_size=100, chunk_overlap=100)
        assert "Failed to create collection" in str(exc_info.value)
        assert "Chunk overlap must be less than chunk size" in str(exc_info.value)

    @pytest.mark.asyncio()
    async def test_create_collection_integrity_error(self, repository, mock_session):
        """Test handling of integrity errors."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result
        # Mock flush to raise IntegrityError (not add)
        mock_session.flush.side_effect = IntegrityError("statement", "params", "orig")

        # Act & Assert
        with pytest.raises(EntityAlreadyExistsError):
            await repository.create(name="test", owner_id=1)

    @pytest.mark.asyncio()
    async def test_get_by_uuid(self, repository, mock_session, sample_collection):
        """Test getting collection by UUID."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_collection
        mock_session.execute.return_value = mock_result

        # Act
        result = await repository.get_by_uuid(sample_collection.id)

        # Assert
        assert result == sample_collection
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio()
    async def test_get_by_uuid_not_found(self, repository, mock_session):
        """Test getting non-existent collection by UUID."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        # Act
        result = await repository.get_by_uuid("nonexistent")

        # Assert
        assert result is None

    @pytest.mark.asyncio()
    async def test_get_by_name(self, repository, mock_session, sample_collection):
        """Test getting collection by name."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_collection
        mock_session.execute.return_value = mock_result

        # Act
        result = await repository.get_by_name("test-collection")

        # Assert
        assert result == sample_collection

    @pytest.mark.asyncio()
    async def test_get_by_uuid_with_permission_check_owner(self, repository, mock_session, sample_collection):
        """Test getting collection with permission check as owner."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_collection
        mock_session.execute.return_value = mock_result

        # Act
        result = await repository.get_by_uuid_with_permission_check(sample_collection.id, sample_collection.owner_id)

        # Assert
        assert result == sample_collection

    @pytest.mark.asyncio()
    async def test_get_by_uuid_with_permission_check_public(self, repository, mock_session, sample_collection):
        """Test getting public collection with permission check."""
        # Setup
        sample_collection.is_public = True
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_collection
        mock_session.execute.return_value = mock_result

        # Act
        result = await repository.get_by_uuid_with_permission_check(sample_collection.id, 999)  # Different user

        # Assert
        assert result == sample_collection

    @pytest.mark.asyncio()
    async def test_get_by_uuid_with_permission_check_denied(self, repository, mock_session, sample_collection):
        """Test permission denied for private collection."""
        # Setup
        sample_collection.is_public = False
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_collection
        mock_session.execute.return_value = mock_result

        # Act & Assert
        with pytest.raises(AccessDeniedError) as exc_info:
            await repository.get_by_uuid_with_permission_check(sample_collection.id, 999)
        assert "999" in str(exc_info.value)
        assert sample_collection.id in str(exc_info.value)

    @pytest.mark.asyncio()
    async def test_get_by_uuid_with_permission_check_not_found(self, repository, mock_session):
        """Test permission check for non-existent collection."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        # Act & Assert
        with pytest.raises(EntityNotFoundError) as exc_info:
            await repository.get_by_uuid_with_permission_check("nonexistent", 1)
        assert "collection" in str(exc_info.value)

    @pytest.mark.asyncio()
    async def test_list_for_user(self, repository, mock_session):
        """Test listing collections for a user."""
        # Setup
        user_id = 1
        collections = [Collection(id=str(uuid4()), name=f"collection-{i}", owner_id=user_id) for i in range(3)]

        # Mock count query
        mock_session.scalar.return_value = 3

        # Mock collection query
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = collections
        mock_session.execute.return_value = mock_result

        # Act
        result_collections, total = await repository.list_for_user(user_id)

        # Assert
        assert len(result_collections) == 3
        assert total == 3

    @pytest.mark.asyncio()
    async def test_list_for_user_with_pagination(self, repository, mock_session):
        """Test listing collections with pagination."""
        # Setup
        user_id = 1
        collections = [Collection(id=str(uuid4()), name="collection-1", owner_id=user_id)]

        # Mock count
        mock_session.scalar.return_value = 10

        # Mock paginated results
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = collections
        mock_session.execute.return_value = mock_result

        # Act
        result_collections, total = await repository.list_for_user(user_id, offset=5, limit=1)

        # Assert
        assert len(result_collections) == 1
        assert total == 10

    @pytest.mark.asyncio()
    async def test_update_status(self, repository, mock_session, sample_collection):
        """Test updating collection status."""
        # Setup - first call returns collection, second call for update
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_collection
        mock_session.execute.return_value = mock_result

        # Act
        result = await repository.update_status(
            sample_collection.id, CollectionStatus.PROCESSING, "Starting processing"
        )

        # Assert
        assert result.status == CollectionStatus.PROCESSING
        assert result.status_message == "Starting processing"
        mock_session.flush.assert_called_once()

    @pytest.mark.asyncio()
    async def test_update_status_not_found(self, repository, mock_session):
        """Test updating status of non-existent collection."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        # Act & Assert
        with pytest.raises(EntityNotFoundError):
            await repository.update_status("nonexistent", CollectionStatus.READY)

    @pytest.mark.asyncio()
    async def test_update_stats(self, repository, mock_session, sample_collection):
        """Test updating collection statistics."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_collection
        mock_session.execute.return_value = mock_result

        # Act
        result = await repository.update_stats(
            sample_collection.id, document_count=100, vector_count=5000, total_size_bytes=1048576
        )

        # Assert
        assert result.document_count == 100
        assert result.vector_count == 5000
        assert result.total_size_bytes == 1048576
        mock_session.flush.assert_called_once()

    @pytest.mark.asyncio()
    async def test_update_stats_validation_errors(self, repository, mock_session):
        """Test validation errors when updating stats."""
        # Mock get_by_uuid to return None (not needed as validation happens first)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result
        
        # Test negative document count
        with pytest.raises(DatabaseOperationError) as exc_info:
            await repository.update_stats("test-id", document_count=-1)
        assert "Document count cannot be negative" in str(exc_info.value)

        # Test negative vector count
        with pytest.raises(DatabaseOperationError) as exc_info:
            await repository.update_stats("test-id", vector_count=-1)
        assert "Vector count cannot be negative" in str(exc_info.value)

        # Test negative size
        with pytest.raises(DatabaseOperationError) as exc_info:
            await repository.update_stats("test-id", total_size_bytes=-1)
        assert "Total size cannot be negative" in str(exc_info.value)

    @pytest.mark.asyncio()
    async def test_rename(self, repository, mock_session, sample_collection):
        """Test renaming a collection."""
        # Setup
        # First call checks for existing name
        mock_result1 = MagicMock()
        mock_result1.scalar_one_or_none.return_value = None

        # Second call gets the collection to rename
        mock_result2 = MagicMock()
        mock_result2.scalar_one_or_none.return_value = sample_collection

        mock_session.execute.side_effect = [mock_result1, mock_result2]

        # Act
        result = await repository.rename(sample_collection.id, "new-name", sample_collection.owner_id)

        # Assert
        assert result.name == "new-name"
        mock_session.flush.assert_called_once()

    @pytest.mark.asyncio()
    async def test_rename_validation_errors(self, repository):
        """Test validation errors when renaming."""
        # Test empty name
        with pytest.raises(ValidationError) as exc_info:
            await repository.rename("test-id", "", 1)
        assert "Collection name cannot be empty" in str(exc_info.value)

        # Test whitespace-only name
        with pytest.raises(ValidationError) as exc_info:
            await repository.rename("test-id", "   ", 1)
        assert "Collection name cannot be empty" in str(exc_info.value)

    @pytest.mark.asyncio()
    async def test_rename_duplicate_name(self, repository, mock_session, sample_collection):
        """Test renaming to an existing name."""
        # Setup - existing collection found
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_collection
        mock_session.execute.return_value = mock_result

        # Act & Assert
        with pytest.raises(EntityAlreadyExistsError):
            await repository.rename("test-id", "existing-name", 1)

    @pytest.mark.asyncio()
    async def test_rename_not_owner(self, repository, mock_session, sample_collection):
        """Test renaming collection by non-owner."""
        # Setup
        # First call - no existing collection with new name
        mock_result1 = MagicMock()
        mock_result1.scalar_one_or_none.return_value = None

        # Second call - get collection (different owner)
        mock_result2 = MagicMock()
        mock_result2.scalar_one_or_none.return_value = sample_collection

        mock_session.execute.side_effect = [mock_result1, mock_result2]

        # Act & Assert
        with pytest.raises(AccessDeniedError):
            await repository.rename(sample_collection.id, "new-name", 999)

    @pytest.mark.asyncio()
    async def test_delete(self, repository, mock_session, sample_collection):
        """Test deleting a collection."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_collection
        mock_session.execute.return_value = mock_result

        # Act
        await repository.delete(sample_collection.id, sample_collection.owner_id)

        # Assert
        mock_session.delete.assert_called_once_with(sample_collection)
        mock_session.flush.assert_called_once()

    @pytest.mark.asyncio()
    async def test_delete_not_owner(self, repository, mock_session, sample_collection):
        """Test deleting collection by non-owner."""
        # Setup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_collection
        mock_session.execute.return_value = mock_result

        # Act & Assert
        with pytest.raises(AccessDeniedError):
            await repository.delete(sample_collection.id, 999)

    @pytest.mark.asyncio()
    async def test_get_document_count(self, repository, mock_session):
        """Test getting document count for a collection."""
        # Setup
        mock_session.scalar.return_value = 42

        # Act
        count = await repository.get_document_count("test-collection-id")

        # Assert
        assert count == 42
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
