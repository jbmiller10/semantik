"""Unit tests for system API endpoints."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from webui.main import app


@pytest_asyncio.fixture
async def system_api_client():
    """Provide an AsyncClient for system endpoint tests."""
    # Clear any existing overrides
    app.dependency_overrides.clear()

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    app.dependency_overrides.clear()


class TestSystemInfo:
    """Tests for GET /api/v2/system/info endpoint."""

    @pytest.mark.asyncio()
    async def test_get_system_info_returns_expected_structure(self, system_api_client):
        """Test that system info returns the expected structure."""
        response = await system_api_client.get("/api/v2/system/info")

        assert response.status_code == 200
        data = response.json()

        # Verify top-level fields
        assert "version" in data
        assert "environment" in data
        assert "python_version" in data
        assert "limits" in data
        assert "rate_limits" in data

        # Verify limits structure
        assert "max_collections_per_user" in data["limits"]
        assert "max_storage_gb_per_user" in data["limits"]
        assert isinstance(data["limits"]["max_collections_per_user"], int)
        assert isinstance(data["limits"]["max_storage_gb_per_user"], int | float)

        # Verify rate_limits structure
        assert "chunking_preview" in data["rate_limits"]
        assert "plugin_install" in data["rate_limits"]
        assert "llm_test" in data["rate_limits"]

    @pytest.mark.asyncio()
    async def test_get_system_info_no_auth_required(self, system_api_client):
        """Test that system info does not require authentication."""
        response = await system_api_client.get("/api/v2/system/info")
        # Should not return 401 (no auth required)
        assert response.status_code == 200


class TestSystemHealth:
    """Tests for GET /api/v2/system/health endpoint."""

    @pytest.mark.asyncio()
    async def test_get_system_health_returns_service_statuses(self, system_api_client):
        """Test that system health returns status for all services."""
        with (
            patch("webui.api.v2.system._check_database_health") as mock_db,
            patch("webui.api.v2.system._check_redis_health") as mock_redis,
            patch("webui.api.v2.system._check_qdrant_health") as mock_qdrant,
            patch("webui.api.v2.system._check_search_api_health") as mock_search,
        ):
            mock_db.return_value = {"status": "healthy", "message": "Database connection successful"}
            mock_redis.return_value = {"status": "healthy", "message": "Redis connection successful"}
            mock_qdrant.return_value = {
                "status": "healthy",
                "message": "Qdrant connection successful",
                "collections_count": 5,
            }
            mock_search.return_value = {"status": "healthy", "message": "Search API connection successful"}

            response = await system_api_client.get("/api/v2/system/health")

        assert response.status_code == 200
        data = response.json()

        # Verify all services are present
        assert "postgres" in data
        assert "redis" in data
        assert "qdrant" in data
        assert "vecpipe" in data

        # Verify status structure
        assert data["postgres"]["status"] == "healthy"
        assert data["redis"]["status"] == "healthy"
        assert data["qdrant"]["status"] == "healthy"
        assert data["vecpipe"]["status"] == "healthy"

    @pytest.mark.asyncio()
    async def test_get_system_health_handles_partial_failures(self, system_api_client):
        """Test that system health returns 200 even with partial failures."""
        with (
            patch("webui.api.v2.system._check_database_health") as mock_db,
            patch("webui.api.v2.system._check_redis_health") as mock_redis,
            patch("webui.api.v2.system._check_qdrant_health") as mock_qdrant,
            patch("webui.api.v2.system._check_search_api_health") as mock_search,
        ):
            mock_db.return_value = {"status": "healthy", "message": "Database connection successful"}
            mock_redis.return_value = {"status": "unhealthy", "message": "Redis connection failed"}
            mock_qdrant.return_value = {"status": "healthy", "message": "Qdrant connection successful"}
            mock_search.return_value = {"status": "degraded", "message": "Search API is degraded"}

            response = await system_api_client.get("/api/v2/system/health")

        # Should still return 200 for partial results
        assert response.status_code == 200
        data = response.json()

        assert data["postgres"]["status"] == "healthy"
        assert data["redis"]["status"] == "unhealthy"
        assert data["qdrant"]["status"] == "healthy"
        assert data["vecpipe"]["status"] == "degraded"

    @pytest.mark.asyncio()
    async def test_get_system_health_handles_exceptions(self, system_api_client):
        """Test that system health handles exceptions gracefully."""
        with (
            patch("webui.api.v2.system._check_database_health") as mock_db,
            patch("webui.api.v2.system._check_redis_health") as mock_redis,
            patch("webui.api.v2.system._check_qdrant_health") as mock_qdrant,
            patch("webui.api.v2.system._check_search_api_health") as mock_search,
        ):
            mock_db.side_effect = Exception("Database error")
            mock_redis.return_value = {"status": "healthy", "message": "Redis connection successful"}
            mock_qdrant.return_value = {"status": "healthy", "message": "Qdrant connection successful"}
            mock_search.return_value = {"status": "healthy", "message": "Search API connection successful"}

            response = await system_api_client.get("/api/v2/system/health")

        # Should still return 200
        assert response.status_code == 200
        data = response.json()

        # Exception should be converted to unhealthy status
        assert data["postgres"]["status"] == "unhealthy"
        assert "Database error" in data["postgres"]["message"]

    @pytest.mark.asyncio()
    async def test_get_system_health_no_auth_required(self, system_api_client):
        """Test that system health does not require authentication."""
        with (
            patch("webui.api.v2.system._check_database_health") as mock_db,
            patch("webui.api.v2.system._check_redis_health") as mock_redis,
            patch("webui.api.v2.system._check_qdrant_health") as mock_qdrant,
            patch("webui.api.v2.system._check_search_api_health") as mock_search,
        ):
            mock_db.return_value = {"status": "healthy", "message": "OK"}
            mock_redis.return_value = {"status": "healthy", "message": "OK"}
            mock_qdrant.return_value = {"status": "healthy", "message": "OK"}
            mock_search.return_value = {"status": "healthy", "message": "OK"}

            response = await system_api_client.get("/api/v2/system/health")

        # Should not return 401 (no auth required)
        assert response.status_code == 200
