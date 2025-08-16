"""Integration tests for chunking validation through the API layer."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, status
from httpx import AsyncClient

from packages.shared.database.models import Collection

# The test_app and authenticated_client fixtures are now provided by conftest.py
# which uses the centralized auth mocking infrastructure


@pytest.mark.asyncio()
async def test_create_collection_with_invalid_chunking_strategy(test_app: FastAPI, authenticated_client: AsyncClient) -> None:
    """Test that API returns 400 for invalid chunking strategy."""
    from packages.webui.services.factory import get_collection_service
    
    # Mock service to raise ValueError
    mock_service = MagicMock()
    mock_service.create_collection = AsyncMock(
        side_effect=ValueError(
            "Invalid chunking_strategy: Strategy invalid_strategy failed: "
            "Unknown strategy: invalid_strategy. Available: fixed_size, semantic, recursive"
        )
    )
    
    # Override the service dependency for this test
    test_app.dependency_overrides[get_collection_service] = lambda: mock_service
    
    response = await authenticated_client.post(
        "/api/v2/collections",
        json={
            "name": "Test Collection",
            "description": "Test",
            "chunking_strategy": "invalid_strategy",
        },
    )
    
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    data = response.json()
    assert "Invalid chunking_strategy" in data["detail"]
    assert "Available:" in data["detail"]


@pytest.mark.asyncio()
async def test_create_collection_with_invalid_chunking_config(test_app: FastAPI, authenticated_client: AsyncClient) -> None:
    """Test that API returns 400 for invalid chunking config."""
    from packages.webui.services.factory import get_collection_service
    
    # Mock service to raise ValueError
    mock_service = MagicMock()
    mock_service.create_collection = AsyncMock(
        side_effect=ValueError("Invalid chunking_config for strategy 'recursive': chunk_size must be at least 10")
    )
    
    # Override the service dependency for this test
    test_app.dependency_overrides[get_collection_service] = lambda: mock_service
    
    response = await authenticated_client.post(
        "/api/v2/collections",
        json={
            "name": "Test Collection",
            "description": "Test",
            "chunking_strategy": "recursive",
            "chunking_config": {"chunk_size": -100},
        },
    )
    
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    data = response.json()
    assert "Invalid chunking_config" in data["detail"]
    assert "chunk_size must be at least 10" in data["detail"]


@pytest.mark.asyncio()
async def test_create_collection_with_config_but_no_strategy(test_app: FastAPI, authenticated_client: AsyncClient) -> None:
    """Test that API returns 400 when config is provided without strategy."""
    from packages.webui.services.factory import get_collection_service
    
    # Mock service to raise ValueError
    mock_service = MagicMock()
    mock_service.create_collection = AsyncMock(
        side_effect=ValueError("chunking_config requires chunking_strategy to be specified")
    )
    
    # Override the service dependency for this test
    test_app.dependency_overrides[get_collection_service] = lambda: mock_service
    
    response = await authenticated_client.post(
        "/api/v2/collections",
        json={
            "name": "Test Collection",
            "description": "Test",
            "chunking_config": {"chunk_size": 500},
        },
    )
    
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    data = response.json()
    assert "chunking_config requires chunking_strategy" in data["detail"]


@pytest.mark.asyncio()
async def test_update_collection_with_invalid_chunking_strategy(test_app: FastAPI, authenticated_client: AsyncClient) -> None:
    """Test that API returns 400 for invalid chunking strategy on update."""
    from packages.webui.services.factory import get_collection_service
    
    collection_id = "test-uuid-123"
    
    # Mock service to raise ValueError
    mock_service = MagicMock()
    mock_service.update = AsyncMock(
        side_effect=ValueError(
            "Invalid chunking_strategy: Strategy bad_strategy failed: "
            "Unknown strategy: bad_strategy. Available: fixed_size, semantic, recursive"
        )
    )
    
    # Override the service dependency for this test
    test_app.dependency_overrides[get_collection_service] = lambda: mock_service

    response = await authenticated_client.put(
        f"/api/v2/collections/{collection_id}", json={"chunking_strategy": "bad_strategy"}
    )
    
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    data = response.json()
    assert "Invalid chunking_strategy" in data["detail"]


@pytest.mark.asyncio()
async def test_update_collection_with_invalid_config_for_strategy(test_app: FastAPI, authenticated_client: AsyncClient) -> None:
    """Test that API returns 400 for invalid config on update."""
    from packages.webui.services.factory import get_collection_service
    
    collection_id = "test-uuid-123"
    
    # Mock service to raise ValueError
    mock_service = MagicMock()
    mock_service.update = AsyncMock(
        side_effect=ValueError(
            "Invalid chunking_config for strategy 'semantic': similarity_threshold must be between 0 and 1"
        )
    )
    
    # Override the service dependency for this test
    test_app.dependency_overrides[get_collection_service] = lambda: mock_service

    response = await authenticated_client.put(
        f"/api/v2/collections/{collection_id}",
        json={"chunking_strategy": "semantic", "chunking_config": {"similarity_threshold": 2.0}},
    )
    
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    data = response.json()
    assert "Invalid chunking_config" in data["detail"]
    assert "similarity_threshold must be between 0 and 1" in data["detail"]


@pytest.mark.asyncio()
async def test_successful_collection_creation_with_valid_chunking(test_app: FastAPI, authenticated_client: AsyncClient) -> None:
    """Test that valid chunking configuration is accepted."""
    from packages.webui.services.factory import get_collection_service
    
    # Mock successful creation - return complete collection dict
    mock_collection = {
        "id": "new-uuid-123",  # CollectionResponse expects 'id' not 'uuid'
        "uuid": "new-uuid-123",
        "name": "Test Collection",
        "description": "Test",
        "owner_id": 1,
        "vector_store_name": "test_vector_store",
        "embedding_model": "text-embedding-ada-002",
        "quantization": "scalar",
        "chunk_size": 500,
        "chunk_overlap": 50,
        "chunking_strategy": "recursive",
        "chunking_config": {"chunk_size": 500, "chunk_overlap": 50},
        "is_public": False,
        "metadata": {},
        "created_at": "2024-01-01T00:00:00",
        "updated_at": "2024-01-01T00:00:00",
        "document_count": 0,
        "vector_count": 0,
        "status": "pending",
        "status_message": None,
    }
    
    mock_operation = {
        "uuid": "op-uuid-123",
        "collection_id": "new-uuid-123",
        "type": "index",
        "status": "pending",
        "config": {},
        "created_at": "2024-01-01T00:00:00",
    }
    
    mock_service = MagicMock()
    mock_service.create_collection = AsyncMock(
        return_value=(mock_collection, mock_operation)
    )
    
    # Override the service dependency for this test
    test_app.dependency_overrides[get_collection_service] = lambda: mock_service

    response = await authenticated_client.post(
        "/api/v2/collections",
        json={
            "name": "Test Collection",
            "description": "Test",
            "chunking_strategy": "recursive",
            "chunking_config": {"chunk_size": 500, "chunk_overlap": 50},
        },
    )
    
    assert response.status_code == status.HTTP_201_CREATED
    data = response.json()
    assert data["chunking_strategy"] == "recursive"
    assert data["chunking_config"]["chunk_size"] == 500
