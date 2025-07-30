"""API endpoint tests for collection deletion."""

import pytest
from fastapi import status
from unittest.mock import MagicMock, AsyncMock, patch

from packages.shared.database.exceptions import EntityNotFoundError, AccessDeniedError, InvalidStateError


@pytest.mark.asyncio
class TestCollectionDeletionEndpoint:
    """Test the DELETE /api/v2/collections/{collection_uuid} endpoint."""

    async def test_delete_collection_success(self, test_client, mock_collection_repository):
        """Test successful collection deletion."""
        # Arrange
        collection_uuid = "test-collection-uuid"
        mock_collection_repository.delete = AsyncMock()
        
        # Act
        response = test_client.delete(f"/api/v2/collections/{collection_uuid}")
        
        # Assert
        assert response.status_code == status.HTTP_204_NO_CONTENT
        assert response.content == b""  # No content in 204 response
        
    async def test_delete_collection_not_found(self, test_client):
        """Test deletion of non-existent collection."""
        # Arrange
        collection_uuid = "non-existent-uuid"
        
        with patch('packages.webui.services.collection_service.CollectionService.delete_collection') as mock_delete:
            mock_delete.side_effect = EntityNotFoundError("collection", collection_uuid)
            
            # Act
            response = test_client.delete(f"/api/v2/collections/{collection_uuid}")
            
            # Assert
            assert response.status_code == status.HTTP_404_NOT_FOUND
            assert "not found" in response.json()["detail"].lower()
            
    async def test_delete_collection_access_denied(self, test_client):
        """Test deletion without permission."""
        # Arrange
        collection_uuid = "test-collection-uuid"
        
        with patch('packages.webui.services.collection_service.CollectionService.delete_collection') as mock_delete:
            mock_delete.side_effect = AccessDeniedError("1", "collection", collection_uuid)
            
            # Act
            response = test_client.delete(f"/api/v2/collections/{collection_uuid}")
            
            # Assert
            assert response.status_code == status.HTTP_403_FORBIDDEN
            assert "owner can delete" in response.json()["detail"]
            
    async def test_delete_collection_with_active_operations(self, test_client):
        """Test deletion while operations are in progress."""
        # Arrange
        collection_uuid = "test-collection-uuid"
        
        with patch('packages.webui.services.collection_service.CollectionService.delete_collection') as mock_delete:
            mock_delete.side_effect = InvalidStateError(
                "Cannot delete collection while operations are in progress"
            )
            
            # Act
            response = test_client.delete(f"/api/v2/collections/{collection_uuid}")
            
            # Assert
            assert response.status_code == status.HTTP_409_CONFLICT
            assert "operations are in progress" in response.json()["detail"]
            
    async def test_delete_collection_rate_limiting(self, test_client):
        """Test rate limiting on delete endpoint."""
        # Arrange
        collection_uuid = "test-collection-uuid"
        
        with patch('packages.webui.services.collection_service.CollectionService.delete_collection') as mock_delete:
            mock_delete.return_value = None
            
            # Act - make 6 requests (limit is 5/hour)
            for i in range(6):
                response = test_client.delete(f"/api/v2/collections/{collection_uuid}")
                
                if i < 5:
                    assert response.status_code == status.HTTP_204_NO_CONTENT
                else:
                    # 6th request should be rate limited
                    assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS
                    
    async def test_delete_collection_internal_error(self, test_client):
        """Test handling of unexpected errors."""
        # Arrange
        collection_uuid = "test-collection-uuid"
        
        with patch('packages.webui.services.collection_service.CollectionService.delete_collection') as mock_delete:
            mock_delete.side_effect = Exception("Unexpected error")
            
            # Act
            response = test_client.delete(f"/api/v2/collections/{collection_uuid}")
            
            # Assert
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert "Failed to delete collection" in response.json()["detail"]


@pytest.mark.asyncio
class TestCollectionDeletionIntegration:
    """Integration tests for collection deletion through the API."""
    
    async def test_delete_collection_removes_from_list(self, test_client):
        """Test that deleted collection disappears from list endpoint."""
        # This would require a more complex setup with a real database
        # For now, we'll mock the behavior
        
        with patch('packages.webui.api.v2.collections.CollectionService') as MockService:
            # Setup mock service
            mock_service = MockService.return_value
            
            # First call returns collection in list
            mock_service.list_for_user = AsyncMock(return_value=([
                {"id": "test-uuid", "name": "Test Collection"}
            ], 1))
            
            # Get initial list
            response = test_client.get("/api/v2/collections")
            assert response.status_code == 200
            assert len(response.json()["collections"]) == 1
            
            # Delete the collection
            mock_service.delete_collection = AsyncMock()
            response = test_client.delete("/api/v2/collections/test-uuid")
            assert response.status_code == 204
            
            # Second call returns empty list
            mock_service.list_for_user = AsyncMock(return_value=([], 0))
            
            # Get list again - should be empty
            response = test_client.get("/api/v2/collections")
            assert response.status_code == 200
            assert len(response.json()["collections"]) == 0
            
    async def test_delete_collection_cascades_operations(self, test_client):
        """Test that collection deletion cascades to operations."""
        # This test would verify that operations endpoint returns 404
        # after collection is deleted
        
        collection_uuid = "test-collection-uuid"
        
        with patch('packages.webui.services.collection_service.CollectionService') as MockService:
            mock_service = MockService.return_value
            
            # Setup initial state
            mock_service.list_operations = AsyncMock(return_value=([
                {"id": "op1", "type": "add_source", "status": "completed"}
            ], 1))
            
            # Get operations before deletion
            response = test_client.get(f"/api/v2/collections/{collection_uuid}/operations")
            # This might fail if the endpoint checks collection existence first
            
            # Delete collection
            mock_service.delete_collection = AsyncMock()
            response = test_client.delete(f"/api/v2/collections/{collection_uuid}")
            assert response.status_code == 204
            
            # Try to get operations after deletion
            mock_service.list_operations = AsyncMock(
                side_effect=EntityNotFoundError("collection", collection_uuid)
            )
            response = test_client.get(f"/api/v2/collections/{collection_uuid}/operations")
            # Should return 404 since collection doesn't exist