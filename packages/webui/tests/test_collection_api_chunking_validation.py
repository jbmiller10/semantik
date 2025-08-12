"""Integration tests for chunking validation through the API layer."""

import pytest
from fastapi import status
from httpx import AsyncClient
from unittest.mock import AsyncMock, MagicMock, patch

from packages.webui.main import app
from packages.shared.database.models import User, Collection


@pytest.fixture
async def authenticated_client():
    """Create an authenticated test client."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        # Mock authentication
        with patch("packages.webui.dependencies.get_current_user") as mock_get_user:
            mock_user = MagicMock(spec=User)
            mock_user.id = 1
            mock_user.username = "testuser"
            mock_get_user.return_value = mock_user
            yield client


@pytest.mark.asyncio
async def test_create_collection_with_invalid_chunking_strategy(authenticated_client):
    """Test that API returns 400 for invalid chunking strategy."""
    with patch("packages.webui.api.v2.collections.get_collection_service") as mock_get_service:
        # Mock service to raise ValueError
        mock_service = MagicMock()
        mock_service.create_collection = AsyncMock(
            side_effect=ValueError(
                "Invalid chunking_strategy: Strategy invalid_strategy failed: "
                "Unknown strategy: invalid_strategy. Available: fixed_size, semantic, recursive"
            )
        )
        mock_get_service.return_value = mock_service

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


@pytest.mark.asyncio
async def test_create_collection_with_invalid_chunking_config(authenticated_client):
    """Test that API returns 400 for invalid chunking config."""
    with patch("packages.webui.api.v2.collections.get_collection_service") as mock_get_service:
        # Mock service to raise ValueError
        mock_service = MagicMock()
        mock_service.create_collection = AsyncMock(
            side_effect=ValueError(
                "Invalid chunking_config for strategy 'recursive': " "chunk_size must be at least 10"
            )
        )
        mock_get_service.return_value = mock_service

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


@pytest.mark.asyncio
async def test_create_collection_with_config_but_no_strategy(authenticated_client):
    """Test that API returns 400 when config is provided without strategy."""
    with patch("packages.webui.api.v2.collections.get_collection_service") as mock_get_service:
        # Mock service to raise ValueError
        mock_service = MagicMock()
        mock_service.create_collection = AsyncMock(
            side_effect=ValueError("chunking_config requires chunking_strategy to be specified")
        )
        mock_get_service.return_value = mock_service

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


@pytest.mark.asyncio
async def test_update_collection_with_invalid_chunking_strategy(authenticated_client):
    """Test that API returns 400 for invalid chunking strategy on update."""
    collection_id = "test-uuid-123"

    with patch("packages.webui.api.v2.collections.get_collection_service") as mock_get_service:
        # Mock service to raise ValueError
        mock_service = MagicMock()
        mock_service.update_collection = AsyncMock(
            side_effect=ValueError(
                "Invalid chunking_strategy: Strategy bad_strategy failed: "
                "Unknown strategy: bad_strategy. Available: fixed_size, semantic, recursive"
            )
        )
        mock_get_service.return_value = mock_service

        response = await authenticated_client.patch(
            f"/api/v2/collections/{collection_id}", json={"chunking_strategy": "bad_strategy"}
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "Invalid chunking_strategy" in data["detail"]


@pytest.mark.asyncio
async def test_update_collection_with_invalid_config_for_strategy(authenticated_client):
    """Test that API returns 400 for invalid config on update."""
    collection_id = "test-uuid-123"

    with patch("packages.webui.api.v2.collections.get_collection_service") as mock_get_service:
        # Mock service to raise ValueError
        mock_service = MagicMock()
        mock_service.update_collection = AsyncMock(
            side_effect=ValueError(
                "Invalid chunking_config for strategy 'semantic': " "similarity_threshold must be between 0 and 1"
            )
        )
        mock_get_service.return_value = mock_service

        response = await authenticated_client.patch(
            f"/api/v2/collections/{collection_id}",
            json={"chunking_strategy": "semantic", "chunking_config": {"similarity_threshold": 2.0}},
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "Invalid chunking_config" in data["detail"]
        assert "similarity_threshold must be between 0 and 1" in data["detail"]


@pytest.mark.asyncio
async def test_successful_collection_creation_with_valid_chunking(authenticated_client):
    """Test that valid chunking configuration is accepted."""
    with patch("packages.webui.api.v2.collections.get_collection_service") as mock_get_service:
        # Mock successful creation
        mock_collection = MagicMock(spec=Collection)
        mock_collection.uuid = "new-uuid-123"
        mock_collection.name = "Test Collection"
        mock_collection.chunking_strategy = "recursive"
        mock_collection.chunking_config = {"chunk_size": 500, "chunk_overlap": 50}

        mock_operation = {
            "uuid": "op-uuid-123",
            "collection_id": "new-uuid-123",
            "type": "index",
            "status": "pending",
            "meta": None,
            "created_at": "2024-01-01T00:00:00",
        }

        mock_service = MagicMock()
        mock_service.create_collection = AsyncMock(
            return_value=(
                {
                    "uuid": mock_collection.uuid,
                    "name": mock_collection.name,
                    "chunking_strategy": mock_collection.chunking_strategy,
                    "chunking_config": mock_collection.chunking_config,
                },
                mock_operation,
            )
        )
        mock_get_service.return_value = mock_service

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
        assert data["collection"]["chunking_strategy"] == "recursive"
        assert data["collection"]["chunking_config"]["chunk_size"] == 500
