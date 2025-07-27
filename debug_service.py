"""Temporary debug version of delete_collection to understand the issue."""

import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from packages.webui.services.collection_service import CollectionService
from packages.shared.database.models import CollectionStatus

# Monkey patch the delete_collection method to add debug output
original_delete_collection = CollectionService.delete_collection

async def debug_delete_collection(self, collection_id: str, user_id: int) -> None:
    """Debug version of delete_collection."""
    print("=== DEBUG: delete_collection called ===")
    
    # Get collection with permission check
    collection = await self.collection_repo.get_by_uuid_with_permission_check(
        collection_uuid=collection_id, user_id=user_id
    )
    print(f"DEBUG: Got collection: {collection}")
    print(f"DEBUG: collection.owner_id = {collection.owner_id}, user_id = {user_id}")
    
    # Only owner can delete
    if collection.owner_id != user_id:
        print("DEBUG: User is not owner, raising AccessDeniedError")
        from packages.shared.database.exceptions import AccessDeniedError
        raise AccessDeniedError(user_id=str(user_id), resource_type="Collection", resource_id=collection_id)
    
    # Check if there's an active operation
    active_ops = await self.operation_repo.get_active_operations_count(collection.id)
    print(f"DEBUG: active_ops = {active_ops}")
    if active_ops > 0:
        print("DEBUG: Active operations exist, raising InvalidStateError")
        from packages.shared.database.exceptions import InvalidStateError
        raise InvalidStateError(
            "Cannot delete collection while operations are in progress. "
            "Please cancel or wait for operations to complete."
        )
    
    try:
        # Delete from Qdrant if collection exists
        print(f"DEBUG: collection.vector_store_name = {collection.vector_store_name}")
        if collection.vector_store_name:
            try:
                print("DEBUG: Getting qdrant_manager...")
                from packages.webui.utils.qdrant_manager import qdrant_manager
                print(f"DEBUG: qdrant_manager = {qdrant_manager}")
                qdrant_client = qdrant_manager.get_client()
                print(f"DEBUG: Got qdrant_client = {qdrant_client}")
                # Check if collection exists in Qdrant
                collections_response = qdrant_client.get_collections()
                print(f"DEBUG: collections_response = {collections_response}")
                collections = collections_response.collections
                print(f"DEBUG: collections = {collections}")
                collection_names = [c.name for c in collections]
                print(f"DEBUG: collection_names = {collection_names}")
                if collection.vector_store_name in collection_names:
                    print(f"DEBUG: Calling delete_collection({collection.vector_store_name})")
                    qdrant_client.delete_collection(collection.vector_store_name)
                    print(f"DEBUG: delete_collection called successfully")
                else:
                    print(f"DEBUG: Collection {collection.vector_store_name} not found in Qdrant")
            except Exception as e:
                print(f"DEBUG: Exception in Qdrant deletion: {e}")
                import traceback
                traceback.print_exc()
                # Continue with database deletion even if Qdrant deletion fails
        
        # Delete from database
        print("DEBUG: Deleting from database...")
        await self.collection_repo.delete(collection.id, user_id)
        print("DEBUG: Database deletion complete")
        
    except Exception as e:
        print(f"DEBUG: Exception in delete_collection: {e}")
        import traceback
        traceback.print_exc()
        raise

# Apply the monkey patch
CollectionService.delete_collection = debug_delete_collection

# Now run the test
async def test_delete():
    # Create mocks
    mock_db_session = AsyncMock()
    mock_collection_repo = AsyncMock()
    mock_operation_repo = AsyncMock()
    mock_document_repo = AsyncMock()

    # Create service
    service = CollectionService(
        db_session=mock_db_session,
        collection_repo=mock_collection_repo,
        operation_repo=mock_operation_repo,
        document_repo=mock_document_repo,
    )

    # Create mock collection
    mock_collection = MagicMock()
    mock_collection.id = '123e4567-e89b-12d3-a456-426614174000'
    mock_collection.uuid = '123e4567-e89b-12d3-a456-426614174000'
    mock_collection.owner_id = 1
    mock_collection.vector_store_name = 'col_123e4567_e89b_12d3_a456_426614174000'
    mock_collection.status = CollectionStatus.READY

    # Setup repo mocks
    mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection
    mock_operation_repo.get_active_operations_count.return_value = 0
    mock_collection_repo.delete = AsyncMock()

    print('Testing delete_collection...')
    
    with patch('packages.webui.utils.qdrant_manager.qdrant_manager') as mock_qdrant_manager:
        # Setup the mock
        mock_qdrant_client = MagicMock()
        mock_qdrant_manager.get_client.return_value = mock_qdrant_client
        
        # Mock the collections response  
        mock_collections_response = MagicMock()
        mock_collections_response.collections = [
            MagicMock(name='col_123e4567_e89b_12d3_a456_426614174000'),
            MagicMock(name='other_collection'),
        ]
        mock_qdrant_client.get_collections.return_value = mock_collections_response
        
        # Add delete_collection mock
        mock_qdrant_client.delete_collection = MagicMock()
        
        try:
            await service.delete_collection(
                collection_id='123e4567-e89b-12d3-a456-426614174000',
                user_id=1,
            )
            print(f'\\nSUCCESS! delete_collection called: {mock_qdrant_client.delete_collection.called}')
            print(f'delete_collection call count: {mock_qdrant_client.delete_collection.call_count}')
            print(f'delete_collection call args: {mock_qdrant_client.delete_collection.call_args}')
        except Exception as e:
            print(f'\\nERROR: {e}')
            import traceback
            traceback.print_exc()

asyncio.run(test_delete())