"""Integration tests for collection persistence in Qdrant."""

import asyncio
from unittest.mock import Mock, patch

import pytest
from qdrant_client import QdrantClient
from shared.database.repositories.collection_repository import CollectionRepository
from sqlalchemy.ext.asyncio import AsyncSession


@pytest.fixture()
def mock_qdrant_client():
    """Mock Qdrant client."""
    client = Mock(spec=QdrantClient)
    # Mock successful collection creation
    client.create_collection.return_value = True
    # Mock get_collection to return a collection info object
    collection_info = Mock()
    collection_info.vectors_count = 0
    client.get_collection.return_value = collection_info
    # Mock get_collections for cleanup tests
    collections_result = Mock()
    collections_result.collections = []
    client.get_collections.return_value = collections_result
    return client


@pytest.mark.asyncio()
async def test_collection_creation_persists_in_qdrant(async_session: AsyncSession, mock_qdrant_client):
    """Test that collections created in tasks persist in Qdrant."""
    from webui.tasks import _process_index_operation
    from webui.utils.qdrant_manager import qdrant_manager

    # Mock the qdrant manager to return our mock client
    with patch.object(qdrant_manager, "get_client", return_value=mock_qdrant_client):
        # Create a collection in the database
        collection_repo = CollectionRepository(async_session)
        collection = await collection_repo.create(
            name="Test Collection",
            owner_id=1,
            description="Test",
            embedding_model="test-model",
        )
        await async_session.commit()

        # Prepare test data
        operation = {
            "id": "test-operation-123",
            "collection_id": collection.id,
            "user_id": 1,
        }
        collection_dict = {
            "id": collection.id,
            "uuid": collection.id,
            "name": collection.name,
            "vector_store_name": collection.vector_store_name,
            "config": {"vector_dim": 768},
        }

        # Mock updater
        updater = Mock()
        updater.send_update = Mock(return_value=asyncio.Future())
        updater.send_update.return_value.set_result(None)

        # Process INDEX operation
        result = await _process_index_operation(operation, collection_dict, collection_repo, Mock(), updater)

        # Verify Qdrant collection was created
        assert mock_qdrant_client.create_collection.called
        call_args = mock_qdrant_client.create_collection.call_args
        assert call_args[1]["collection_name"] == collection.vector_store_name
        assert call_args[1]["vectors_config"].size == 768

        # Verify collection was verified after creation
        assert mock_qdrant_client.get_collection.called
        assert mock_qdrant_client.get_collection.call_args[0][0] == collection.vector_store_name

        # Verify result
        assert result["success"] is True
        assert result["qdrant_collection"] == collection.vector_store_name
        assert result["vector_dim"] == 768


@pytest.mark.asyncio()
async def test_collection_creation_rollback_on_db_failure(async_session: AsyncSession, mock_qdrant_client):
    """Test that Qdrant collection is deleted if database update fails."""
    from webui.tasks import _process_index_operation
    from webui.utils.qdrant_manager import qdrant_manager

    # Mock the qdrant manager to return our mock client
    with patch.object(qdrant_manager, "get_client", return_value=mock_qdrant_client):
        # Create a collection in the database
        collection_repo = CollectionRepository(async_session)
        collection = await collection_repo.create(
            name="Test Collection 2",
            owner_id=1,
            description="Test",
            embedding_model="test-model",
        )
        await async_session.commit()

        # Mock repository update to fail
        mock_collection_repo = Mock(spec=CollectionRepository)
        mock_collection_repo.update.side_effect = Exception("Database error")

        # Prepare test data
        operation = {
            "id": "test-operation-456",
            "collection_id": collection.id,
            "user_id": 1,
        }
        collection_dict = {
            "id": collection.id,
            "uuid": collection.id,
            "name": collection.name,
            "vector_store_name": collection.vector_store_name,
            "config": {"vector_dim": 768},
        }

        # Mock updater
        updater = Mock()
        updater.send_update = Mock(return_value=asyncio.Future())
        updater.send_update.return_value.set_result(None)

        # Process INDEX operation - should raise exception
        with pytest.raises(Exception, match="Failed to update collection") as exc_info:
            await _process_index_operation(operation, collection_dict, mock_collection_repo, Mock(), updater)

        assert "Failed to update collection" in str(exc_info.value)

        # Verify Qdrant collection was created
        assert mock_qdrant_client.create_collection.called

        # Verify cleanup was attempted
        assert mock_qdrant_client.delete_collection.called
        assert mock_qdrant_client.delete_collection.call_args[0][0] == collection.vector_store_name


@pytest.mark.asyncio()
async def test_maintenance_script_preserves_active_collections(async_session: AsyncSession):
    """Test that maintenance script doesn't delete active collections."""
    from vecpipe.maintenance import QdrantMaintenanceService

    # Create test collections in database
    collection_repo = CollectionRepository(async_session)
    collections = []
    for i in range(3):
        collection = await collection_repo.create(
            name=f"Active Collection {i}",
            owner_id=1,
            description="Test",
            embedding_model="test-model",
        )
        collections.append(collection)
    await async_session.commit()

    # Mock the webui API response
    with patch("httpx.get") as mock_get:
        # Mock response for vector-store-names endpoint
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [c.vector_store_name for c in collections]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Create maintenance service
        service = QdrantMaintenanceService()

        # Get active collections
        active_collections = service.get_active_collections()

        # Verify all collections are recognized as active
        for collection in collections:
            assert collection.vector_store_name in active_collections

        # Mock Qdrant to have our collections plus some orphaned ones
        mock_qdrant_collections = [Mock(name=c.vector_store_name) for c in collections]
        # Add some orphaned collections
        mock_qdrant_collections.extend(
            [
                Mock(name="col_orphaned_1"),
                Mock(name="col_orphaned_2"),
                Mock(name="job_old_job_123"),
            ]
        )

        with patch.object(service.client, "get_collections") as mock_get_collections:
            mock_collections_result = Mock()
            mock_collections_result.collections = mock_qdrant_collections
            mock_get_collections.return_value = mock_collections_result

            with patch.object(service.client, "delete_collection") as mock_delete:
                # Run cleanup in dry run mode
                result = service.cleanup_orphaned_collections(dry_run=True)

                # Verify no deletions in dry run
                assert not mock_delete.called

                # Verify correct orphaned collections identified
                assert len(result["orphaned_collections"]) == 3
                assert "col_orphaned_1" in result["orphaned_collections"]
                assert "col_orphaned_2" in result["orphaned_collections"]
                assert "job_old_job_123" in result["orphaned_collections"]

                # Verify active collections are NOT in orphaned list
                for collection in collections:
                    assert collection.vector_store_name not in result["orphaned_collections"]


@pytest.mark.asyncio()
async def test_collection_naming_convention():
    """Test that collection names follow the expected format."""
    collection_id = "550e8400-e29b-41d4-a716-446655440000"
    expected_name = "col_550e8400_e29b_41d4_a716_446655440000"

    # Test the transformation
    vector_store_name = f"col_{collection_id.replace('-', '_')}"
    assert vector_store_name == expected_name

    # Verify no hyphens in the final name
    assert "-" not in vector_store_name

    # Verify it starts with col_
    assert vector_store_name.startswith("col_")

