"""Mock integration tests for CollectionService to verify proper instantiation."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.database.exceptions import AccessDeniedError, InvalidStateError
from webui.services.factory import create_collection_service


class TestCollectionServiceMockIntegration:
    """Test CollectionService with mocked dependencies."""

    @pytest.mark.asyncio()
    async def test_collection_deletion_via_service_commits_transaction(self) -> None:
        """Test that CollectionService.delete_collection properly commits the transaction."""
        # Mock database session
        db_session = AsyncMock()

        # Mock the qdrant_connection_manager.get_client() in factory to return a mock client
        with patch("webui.services.factory.qdrant_connection_manager") as mock_connection_manager:
            mock_qdrant_client = MagicMock()
            mock_connection_manager.get_client.return_value = mock_qdrant_client

            # Create service using factory (which will now get our mock client)
            service = create_collection_service(db_session)

            # Mock collection
            mock_collection = MagicMock()
            mock_collection.id = "test-collection-id"
            mock_collection.owner_id = 1
            mock_collection.vector_store_name = "test_vector_store"

            # Mock repository methods
            service.collection_repo.get_by_uuid_with_permission_check = AsyncMock(return_value=mock_collection)
            service.operation_repo.get_active_operations_count = AsyncMock(return_value=0)
            service.collection_repo.delete = AsyncMock()

            # Mock qdrant_manager methods (service.qdrant_manager exists now)
            service.qdrant_manager.list_collections = MagicMock(return_value=["test_vector_store"])
            service.qdrant_manager.client.delete_collection = MagicMock()

            # Act
            await service.delete_collection("test-collection-id", 1)

            # Assert - verify all expected calls
            service.collection_repo.get_by_uuid_with_permission_check.assert_called_once_with(
                collection_uuid="test-collection-id", user_id=1
            )
            service.operation_repo.get_active_operations_count.assert_called_once_with("test-collection-id")
            service.collection_repo.delete.assert_called_once_with("test-collection-id", 1)
            db_session.commit.assert_called_once()
            service.qdrant_manager.client.delete_collection.assert_called_once_with("test_vector_store")

    @pytest.mark.asyncio()
    async def test_collection_deletion_handles_missing_qdrant_collection(self) -> None:
        """Test that deletion succeeds even if Qdrant collection doesn't exist."""
        # Mock database session
        db_session = AsyncMock()

        # Mock the qdrant_connection_manager.get_client() in factory to return a mock client
        with patch("webui.services.factory.qdrant_connection_manager") as mock_connection_manager:
            mock_qdrant_client = MagicMock()
            mock_connection_manager.get_client.return_value = mock_qdrant_client

            # Create service using factory
            service = create_collection_service(db_session)

            # Mock collection
            mock_collection = MagicMock()
            mock_collection.id = "test-collection-id"
            mock_collection.owner_id = 1
            mock_collection.vector_store_name = "col_test_123"

            # Mock repository methods
            service.collection_repo.get_by_uuid_with_permission_check = AsyncMock(return_value=mock_collection)
            service.operation_repo.get_active_operations_count = AsyncMock(return_value=0)
            service.collection_repo.delete = AsyncMock()

            # Mock qdrant_manager methods - list shows no collections, and delete throws exception
            service.qdrant_manager.list_collections = MagicMock(return_value=[])  # Collection doesn't exist
            service.qdrant_manager.client.delete_collection.side_effect = Exception("Collection not found")

            # Act - should not raise exception
            await service.delete_collection("test-collection-id", 1)

            # Assert - PostgreSQL collection should still be deleted
            service.collection_repo.delete.assert_called_once_with("test-collection-id", 1)
            db_session.commit.assert_called_once()

    @pytest.mark.asyncio()
    async def test_collection_deletion_fails_with_active_operations(self) -> None:
        """Test that collection cannot be deleted while operations are active."""
        # Mock database session
        db_session = AsyncMock()

        # Create service using factory
        service = create_collection_service(db_session)

        # Mock collection
        mock_collection = MagicMock()
        mock_collection.id = "test-collection-id"
        mock_collection.owner_id = 1
        mock_collection.vector_store_name = "test_vector_store"

        # Mock repository methods - active operations exist
        service.collection_repo.get_by_uuid_with_permission_check = AsyncMock(return_value=mock_collection)
        service.operation_repo.get_active_operations_count = AsyncMock(return_value=1)  # Active operation exists

        # Act & Assert
        with pytest.raises(InvalidStateError) as exc_info:
            await service.delete_collection("test-collection-id", 1)

        assert "operations are in progress" in str(exc_info.value)

        # Verify delete was not called
        assert not hasattr(service.collection_repo.delete, "called") or not service.collection_repo.delete.called
        db_session.commit.assert_not_called()

    @pytest.mark.asyncio()
    async def test_collection_deletion_requires_owner_permission(self) -> None:
        """Test that only the owner can delete a collection."""
        # Mock database session
        db_session = AsyncMock()

        # Create service using factory
        service = create_collection_service(db_session)

        # Mock collection owned by different user
        mock_collection = MagicMock()
        mock_collection.id = "test-collection-id"
        mock_collection.owner_id = 1  # Owner is user 1
        mock_collection.vector_store_name = "test_vector_store"

        # Mock repository methods
        service.collection_repo.get_by_uuid_with_permission_check = AsyncMock(return_value=mock_collection)

        # Act & Assert - try to delete as user 2
        with pytest.raises(AccessDeniedError) as exc_info:
            await service.delete_collection("test-collection-id", 2)  # Different user

        assert "test-collection-id" in str(exc_info.value)

        # Verify delete was not called
        assert not hasattr(service.collection_repo.delete, "called") or not service.collection_repo.delete.called
        db_session.commit.assert_not_called()
