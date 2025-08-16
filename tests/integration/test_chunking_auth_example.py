"""
Example integration test demonstrating proper authentication fixture usage.

This test file shows how to use the authenticated fixtures from conftest.py
to avoid authentication issues when testing collection-related endpoints.
"""

import pytest
from httpx import AsyncClient


class TestChunkingWithProperAuth:
    """Test chunking endpoints with proper authentication."""

    @pytest.mark.asyncio
    async def test_chunking_with_authenticated_collection(
        self,
        authenticated_async_client: AsyncClient,
        authenticated_collection: dict,
        auth_headers: dict,
    ):
        """Test chunking endpoint with properly authenticated collection.
        
        This test demonstrates:
        1. Using authenticated_async_client which has get_current_user overridden
        2. Using authenticated_collection which has the correct owner_id
        3. No AccessDeniedError because user.id matches collection.owner_id
        """
        # The authenticated_async_client already has authentication set up
        # The authenticated_collection is owned by the authenticated user
        
        # Example: Get collection details (should work without AccessDeniedError)
        response = await authenticated_async_client.get(
            f"/api/v2/collections/{authenticated_collection['id']}",
            headers=auth_headers,
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == authenticated_collection["id"]
        assert data["owner_id"] == authenticated_collection["owner_id"]

    @pytest.mark.asyncio 
    async def test_chunking_strategies_list(
        self,
        authenticated_async_client: AsyncClient,
        auth_headers: dict,
    ):
        """Test listing chunking strategies with authentication."""
        response = await authenticated_async_client.get(
            "/api/v2/chunking/strategies",
            headers=auth_headers,
        )
        
        # This should work because we're using authenticated_async_client
        # Note: This will fail if the endpoint isn't mocked or implemented
        # but won't fail due to authentication issues
        
        # The actual response depends on your implementation
        # This is just an example of the pattern

    @pytest.mark.asyncio
    async def test_apply_chunking_to_collection(
        self,
        authenticated_async_client: AsyncClient,
        authenticated_collection: dict,
        auth_headers: dict,
    ):
        """Test applying chunking strategy to an authenticated collection."""
        # Example request to apply chunking
        chunking_config = {
            "strategy": "fixed_size",
            "chunk_size": 512,
            "chunk_overlap": 50,
        }
        
        response = await authenticated_async_client.post(
            f"/api/v2/collections/{authenticated_collection['id']}/chunking/apply",
            json=chunking_config,
            headers=auth_headers,
        )
        
        # This won't fail with AccessDeniedError because:
        # 1. authenticated_async_client returns a user with the correct ID
        # 2. authenticated_collection is owned by that user
        
        # The actual status depends on your implementation
        # This demonstrates the authentication pattern


class TestChunkingWithSyncClient:
    """Test using synchronous authenticated client."""

    def test_chunking_with_sync_authenticated_client(
        self,
        authenticated_client,  # This is the sync TestClient
        authenticated_user: dict,
        auth_headers: dict,
    ):
        """Test using the synchronous authenticated client."""
        # The authenticated_client already has authentication set up
        
        response = authenticated_client.get(
            "/api/v2/chunking/strategies",
            headers=auth_headers,
        )
        
        # This demonstrates using the sync client
        # No authentication errors should occur


# Key points for migrating existing tests:
# 
# 1. Replace custom mock_user fixtures with authenticated_user from conftest
# 2. Replace custom async_client with authenticated_async_client 
# 3. Replace custom collections with authenticated_collection
# 4. Remove manual dependency overrides for get_current_user
# 5. The fixtures ensure user.id matches collection.owner_id
#
# Migration example:
#
# OLD:
#   @pytest.fixture
#   def mock_user():
#       return {"id": 1, ...}
#   
#   @pytest.fixture
#   def mock_collection():
#       return {"owner_id": 1, ...}
#
#   async def test_something(mock_user, mock_collection):
#       app.dependency_overrides[get_current_user] = lambda: mock_user
#       ...
#
# NEW:
#   async def test_something(
#       authenticated_async_client,
#       authenticated_collection,
#       auth_headers
#   ):
#       # No manual overrides needed!
#       response = await authenticated_async_client.get(...)