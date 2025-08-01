"""Unit tests for CollectionService instantiation."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from packages.webui.services.collection_service import CollectionService
from packages.webui.services.factory import create_collection_service


class TestCollectionServiceInstantiation:
    """Test proper instantiation of CollectionService."""

    def test_collection_service_requires_repositories(self):
        """Test that CollectionService cannot be instantiated without repositories."""
        db_session = AsyncMock()

        # This should fail with TypeError
        with pytest.raises(TypeError) as exc_info:
            CollectionService(db_session)

        assert "missing 3 required positional arguments" in str(exc_info.value)

    def test_collection_service_instantiation_with_all_args(self):
        """Test that CollectionService can be instantiated with all required arguments."""
        # Mock dependencies
        db_session = AsyncMock()
        collection_repo = MagicMock()
        operation_repo = MagicMock()
        document_repo = MagicMock()

        # This should work
        service = CollectionService(
            db_session=db_session,
            collection_repo=collection_repo,
            operation_repo=operation_repo,
            document_repo=document_repo,
        )

        assert service.db_session == db_session
        assert service.collection_repo == collection_repo
        assert service.operation_repo == operation_repo
        assert service.document_repo == document_repo

    def test_create_collection_service_factory(self):
        """Test that create_collection_service factory creates service properly."""
        # Mock database session
        db_session = AsyncMock()

        # Create service using factory
        service = create_collection_service(db_session)

        # Verify service was created
        assert isinstance(service, CollectionService)
        assert service.db_session == db_session
        assert service.collection_repo is not None
        assert service.operation_repo is not None
        assert service.document_repo is not None

    @pytest.mark.asyncio()
    async def test_collection_service_delete_with_factory(self):
        """Test that CollectionService created with factory can call delete_collection."""
        # Mock database session
        db_session = AsyncMock()

        # Mock Qdrant manager
        from packages.webui.utils.qdrant_manager import qdrant_manager

        original_get_client = qdrant_manager.get_client

        mock_qdrant = MagicMock()
        mock_collections_response = MagicMock()
        mock_collections_response.collections = []
        mock_qdrant.get_collections.return_value = mock_collections_response
        mock_qdrant.delete_collection = MagicMock()

        qdrant_manager.get_client = lambda: mock_qdrant

        try:
            # Create service using factory
            service = create_collection_service(db_session)

            # Mock repository methods
            mock_collection = MagicMock()
            mock_collection.id = "test-collection-id"
            mock_collection.owner_id = 1
            mock_collection.vector_store_name = "test_vector_store"

            service.collection_repo.get_by_uuid_with_permission_check = AsyncMock(return_value=mock_collection)
            service.operation_repo.get_active_operations_count = AsyncMock(return_value=0)
            service.collection_repo.delete = AsyncMock()

            # Call delete_collection
            await service.delete_collection("test-collection-id", 1)

            # Verify methods were called
            service.collection_repo.get_by_uuid_with_permission_check.assert_called_once_with(
                collection_uuid="test-collection-id", user_id=1
            )
            service.collection_repo.delete.assert_called_once_with("test-collection-id", 1)
            db_session.commit.assert_called_once()

        finally:
            # Restore original
            qdrant_manager.get_client = original_get_client
