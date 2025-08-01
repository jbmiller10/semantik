"""Mock integration tests for CollectionService to verify proper instantiation."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from packages.shared.database.exceptions import AccessDeniedError, InvalidStateError
from packages.webui.services.factory import create_collection_service


class TestCollectionServiceMockIntegration:
    """Test CollectionService with mocked dependencies."""

    @pytest.mark.asyncio()
    async def test_collection_deletion_via_service_commits_transaction(self):
        """Test that CollectionService.delete_collection properly commits the transaction."""
        # Mock database session
        db_session = AsyncMock()

        # Mock Qdrant manager at the module level where it's imported
        with patch("packages.webui.services.collection_service.qdrant_manager") as mock_qdrant_manager:
            mock_qdrant_client = MagicMock()
            # Create a proper collection object with name attribute
            mock_collection_obj = MagicMock()
            mock_collection_obj.name = "test_vector_store"
            mock_collections_response = MagicMock()
            mock_collections_response.collections = [mock_collection_obj]
            mock_qdrant_client.get_collections.return_value = mock_collections_response
            mock_qdrant_client.delete_collection = MagicMock()
            mock_qdrant_manager.get_client.return_value = mock_qdrant_client

            # Create service using factory
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

            # Act
            await service.delete_collection("test-collection-id", 1)

            # Assert - verify all expected calls
            service.collection_repo.get_by_uuid_with_permission_check.assert_called_once_with(
                collection_uuid="test-collection-id", user_id=1
            )
            service.operation_repo.get_active_operations_count.assert_called_once_with("test-collection-id")
            service.collection_repo.delete.assert_called_once_with("test-collection-id", 1)
            db_session.commit.assert_called_once()
            mock_qdrant_client.delete_collection.assert_called_once_with("test_vector_store")

    @pytest.mark.asyncio()
    async def test_collection_deletion_handles_missing_qdrant_collection(self):
        """Test that deletion succeeds even if Qdrant collection doesn't exist."""
        # Mock database session
        db_session = AsyncMock()

        # Mock Qdrant manager to throw exception
        with patch("packages.webui.services.collection_service.qdrant_manager") as mock_qdrant_manager:
            mock_qdrant_client = MagicMock()
            mock_collections_response = MagicMock()
            mock_collections_response.collections = []  # No collections
            mock_qdrant_client.get_collections.return_value = mock_collections_response
            mock_qdrant_client.delete_collection.side_effect = Exception("Collection not found")
            mock_qdrant_manager.get_client.return_value = mock_qdrant_client

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

            # Act - should not raise exception
            await service.delete_collection("test-collection-id", 1)

            # Assert - PostgreSQL collection should still be deleted
            service.collection_repo.delete.assert_called_once_with("test-collection-id", 1)
            db_session.commit.assert_called_once()

    @pytest.mark.asyncio()
    async def test_collection_deletion_fails_with_active_operations(self):
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
    async def test_collection_deletion_requires_owner_permission(self):
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
