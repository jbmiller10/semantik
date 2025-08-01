"""Integration tests for collection persistence in Qdrant."""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest
from qdrant_client import QdrantClient

from packages.shared.database.repositories.collection_repository import CollectionRepository


@pytest.fixture()
def mock_qdrant_client() -> None:
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
async def test_collection_creation_persists_in_qdrant(mock_qdrant_client):
    """Test that collections created in tasks persist in Qdrant."""
    from packages.webui.tasks import _process_index_operation

    # Mock the qdrant manager to return our mock client
    with patch("packages.webui.tasks.qdrant_manager") as mock_qdrant_manager:
        mock_qdrant_manager.get_client.return_value = mock_qdrant_client
        # Create a mock collection
        collection = Mock()
        collection.id = "test-collection-id"
        collection.name = "Test Collection"
        collection.vector_store_name = "col_test_collection_id"

        # Mock collection repository
        collection_repo = Mock(spec=CollectionRepository)
        collection_repo.update = AsyncMock(return_value=None)

        # Prepare test data
        operation = {
            "id": "test-operation-123",
            "collection_id": collection.id,
            "user_id": 1,
            "type": "INDEX",  # Add operation type
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
async def test_collection_creation_rollback_on_db_failure(mock_qdrant_client):
    """Test that Qdrant collection is deleted if database update fails."""
    from packages.webui.tasks import _process_index_operation

    # Mock the qdrant manager to return our mock client
    with patch("packages.webui.tasks.qdrant_manager") as mock_qdrant_manager:
        mock_qdrant_manager.get_client.return_value = mock_qdrant_client
        # Create a mock collection
        collection = Mock()
        collection.id = "test-collection-id-2"
        collection.name = "Test Collection 2"
        collection.vector_store_name = "col_test_collection_id_2"

        # Mock repository update to fail
        mock_collection_repo = Mock(spec=CollectionRepository)
        mock_collection_repo.update = AsyncMock(side_effect=Exception("Database error"))

        # Prepare test data
        operation = {
            "id": "test-operation-456",
            "collection_id": collection.id,
            "user_id": 1,
            "type": "INDEX",  # Add operation type
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
