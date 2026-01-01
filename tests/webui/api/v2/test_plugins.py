"""Integration tests for Plugin API endpoints."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from shared.plugins.manifest import PluginManifest
from shared.plugins.registry import PluginRecord, PluginSource, plugin_registry
from webui.api.v2.plugins import _get_plugin_service
from webui.main import app
from webui.services.plugin_service import PluginService


@pytest.fixture(autouse=True)
def _clear_registry():
    """Clear plugin registry before and after each test."""
    plugin_registry.reset()
    yield
    plugin_registry.reset()


def _make_manifest(
    plugin_id: str = "test-plugin",
    plugin_type: str = "embedding",
) -> PluginManifest:
    """Create a test plugin manifest."""
    return PluginManifest(
        id=plugin_id,
        type=plugin_type,
        version="1.0.0",
        display_name="Test Plugin",
        description="A test plugin",
    )


def _make_record(
    plugin_id: str = "test-plugin",
    plugin_type: str = "embedding",
    source: PluginSource = PluginSource.EXTERNAL,
) -> PluginRecord:
    """Create a test plugin record."""
    manifest = _make_manifest(plugin_id, plugin_type)
    return PluginRecord(
        plugin_type=plugin_type,
        plugin_id=plugin_id,
        plugin_version="1.0.0",
        manifest=manifest,
        plugin_class=MagicMock,
        source=source,
    )


@pytest_asyncio.fixture
async def mock_plugin_service():
    """Create a mock PluginService."""
    service = MagicMock(spec=PluginService)
    service.list_plugins = AsyncMock(return_value=[])
    service.get_plugin = AsyncMock(return_value=None)
    service.get_manifest = AsyncMock(return_value=None)
    service.get_config_schema = AsyncMock(return_value=None)
    service.set_enabled = AsyncMock(return_value=None)
    service.update_config = AsyncMock(return_value=None)
    service.check_health = AsyncMock(return_value=None)
    return service


@pytest_asyncio.fixture
async def api_client_with_plugin(mock_plugin_service):
    """Provide an AsyncClient with plugin-related dependencies mocked."""
    from webui.auth import get_current_user

    mock_user = {
        "id": 1,
        "username": "testuser",
        "email": "test@example.com",
        "full_name": "Test User",
    }

    async def override_get_current_user() -> dict[str, Any]:
        return mock_user

    async def override_get_plugin_service() -> PluginService:
        return mock_plugin_service

    app.dependency_overrides[get_current_user] = override_get_current_user
    app.dependency_overrides[_get_plugin_service] = override_get_plugin_service

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, mock_plugin_service

    app.dependency_overrides.clear()


class TestListPlugins:
    """Tests for GET /api/v2/plugins."""

    @pytest.mark.asyncio()
    async def test_list_plugins_empty(self, api_client_with_plugin):
        """Test list_plugins returns empty list when no external plugins."""
        client, mock_service = api_client_with_plugin
        mock_service.list_plugins.return_value = []

        response = await client.get("/api/v2/plugins")
        assert response.status_code == 200
        data = response.json()
        assert "plugins" in data
        assert data["plugins"] == []

    @pytest.mark.asyncio()
    async def test_list_plugins_with_plugins(self, api_client_with_plugin):
        """Test list_plugins returns external plugins."""
        client, mock_service = api_client_with_plugin
        mock_service.list_plugins.return_value = [
            {
                "id": "test-plugin",
                "type": "embedding",
                "version": "1.0.0",
                "manifest": {
                    "id": "test-plugin",
                    "type": "embedding",
                    "version": "1.0.0",
                    "display_name": "Test Plugin",
                    "description": "A test plugin",
                },
                "enabled": True,
                "health_status": "unknown",
            }
        ]

        response = await client.get("/api/v2/plugins")
        assert response.status_code == 200
        data = response.json()
        assert len(data["plugins"]) == 1
        assert data["plugins"][0]["id"] == "test-plugin"

    @pytest.mark.asyncio()
    async def test_list_plugins_filter_by_type(self, api_client_with_plugin):
        """Test list_plugins filters by plugin_type."""
        client, mock_service = api_client_with_plugin
        mock_service.list_plugins.return_value = [
            {
                "id": "embed-plugin",
                "type": "embedding",
                "version": "1.0.0",
                "manifest": {
                    "id": "embed-plugin",
                    "type": "embedding",
                    "version": "1.0.0",
                    "display_name": "Embed Plugin",
                    "description": "An embedding plugin",
                },
                "enabled": True,
                "health_status": "unknown",
            }
        ]

        response = await client.get("/api/v2/plugins", params={"plugin_type": "embedding"})
        assert response.status_code == 200
        data = response.json()
        assert len(data["plugins"]) == 1
        assert data["plugins"][0]["type"] == "embedding"
        mock_service.list_plugins.assert_called_once_with(plugin_type="embedding", enabled=None, include_health=False)


class TestGetPlugin:
    """Tests for GET /api/v2/plugins/{plugin_id}."""

    @pytest.mark.asyncio()
    async def test_get_plugin_found(self, api_client_with_plugin):
        """Test get_plugin returns plugin when found."""
        client, mock_service = api_client_with_plugin
        mock_service.get_plugin.return_value = {
            "id": "test-plugin",
            "type": "embedding",
            "version": "1.0.0",
            "manifest": {
                "id": "test-plugin",
                "type": "embedding",
                "version": "1.0.0",
                "display_name": "Test Plugin",
                "description": "A test plugin",
            },
            "enabled": True,
            "health_status": "unknown",
        }

        response = await client.get("/api/v2/plugins/test-plugin")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "test-plugin"
        assert data["type"] == "embedding"

    @pytest.mark.asyncio()
    async def test_get_plugin_not_found(self, api_client_with_plugin):
        """Test get_plugin returns 404 when not found."""
        client, mock_service = api_client_with_plugin
        mock_service.get_plugin.return_value = None

        response = await client.get("/api/v2/plugins/nonexistent")
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()


class TestGetPluginManifest:
    """Tests for GET /api/v2/plugins/{plugin_id}/manifest."""

    @pytest.mark.asyncio()
    async def test_get_manifest_found(self, api_client_with_plugin):
        """Test get_manifest returns manifest."""
        client, mock_service = api_client_with_plugin
        mock_service.get_manifest.return_value = {
            "id": "test-plugin",
            "type": "embedding",
            "version": "1.0.0",
            "display_name": "Test Plugin",
            "description": "A test plugin",
        }

        response = await client.get("/api/v2/plugins/test-plugin/manifest")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "test-plugin"
        assert data["type"] == "embedding"
        assert data["display_name"] == "Test Plugin"

    @pytest.mark.asyncio()
    async def test_get_manifest_not_found(self, api_client_with_plugin):
        """Test get_manifest returns 404 when not found."""
        client, mock_service = api_client_with_plugin
        mock_service.get_manifest.return_value = None

        response = await client.get("/api/v2/plugins/nonexistent/manifest")
        assert response.status_code == 404


class TestGetPluginConfigSchema:
    """Tests for GET /api/v2/plugins/{plugin_id}/config-schema."""

    @pytest.mark.asyncio()
    async def test_get_config_schema_found(self, api_client_with_plugin):
        """Test get_config_schema returns schema."""
        client, mock_service = api_client_with_plugin
        mock_service.get_plugin.return_value = {
            "id": "test-plugin",
            "type": "embedding",
            "version": "1.0.0",
            "display_name": "Test Plugin",
            "description": "A test plugin",
            "enabled": True,
            "health_status": "unknown",
        }
        mock_service.get_config_schema.return_value = {"type": "object", "properties": {}}

        response = await client.get("/api/v2/plugins/test-plugin/config-schema")
        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "object"

    @pytest.mark.asyncio()
    async def test_get_config_schema_not_found(self, api_client_with_plugin):
        """Test get_config_schema returns 404 when plugin not found."""
        client, mock_service = api_client_with_plugin
        mock_service.get_plugin.return_value = None

        response = await client.get("/api/v2/plugins/nonexistent/config-schema")
        assert response.status_code == 404


class TestEnableDisablePlugin:
    """Tests for POST /api/v2/plugins/{plugin_id}/enable|disable."""

    @pytest.mark.asyncio()
    async def test_enable_plugin(self, api_client_with_plugin):
        """Test enable_plugin enables a plugin."""
        client, mock_service = api_client_with_plugin
        mock_service.set_enabled.return_value = {"id": "test-plugin", "enabled": True}

        response = await client.post("/api/v2/plugins/test-plugin/enable")
        assert response.status_code == 200
        data = response.json()
        assert data["plugin_id"] == "test-plugin"
        assert data["enabled"] is True
        assert data["requires_restart"] is True

    @pytest.mark.asyncio()
    async def test_enable_plugin_not_found(self, api_client_with_plugin):
        """Test enable_plugin returns 404 when not found."""
        client, mock_service = api_client_with_plugin
        mock_service.set_enabled.return_value = None

        response = await client.post("/api/v2/plugins/nonexistent/enable")
        assert response.status_code == 404

    @pytest.mark.asyncio()
    async def test_disable_plugin(self, api_client_with_plugin):
        """Test disable_plugin disables a plugin."""
        client, mock_service = api_client_with_plugin
        mock_service.set_enabled.return_value = {"id": "test-plugin", "enabled": False}

        response = await client.post("/api/v2/plugins/test-plugin/disable")
        assert response.status_code == 200
        data = response.json()
        assert data["plugin_id"] == "test-plugin"
        assert data["enabled"] is False
        assert data["requires_restart"] is True

    @pytest.mark.asyncio()
    async def test_disable_plugin_not_found(self, api_client_with_plugin):
        """Test disable_plugin returns 404 when not found."""
        client, mock_service = api_client_with_plugin
        mock_service.set_enabled.return_value = None

        response = await client.post("/api/v2/plugins/nonexistent/disable")
        assert response.status_code == 404


class TestUpdatePluginConfig:
    """Tests for PUT /api/v2/plugins/{plugin_id}/config."""

    @pytest.mark.asyncio()
    async def test_update_config_valid(self, api_client_with_plugin):
        """Test update_config with valid config."""
        client, mock_service = api_client_with_plugin
        mock_service.update_config.return_value = {
            "id": "test-plugin",
            "type": "embedding",
            "version": "1.0.0",
            "manifest": {
                "id": "test-plugin",
                "type": "embedding",
                "version": "1.0.0",
                "display_name": "Test Plugin",
                "description": "A test plugin",
            },
            "enabled": True,
            "health_status": "unknown",
            "config": {"key": "value"},
        }

        response = await client.put(
            "/api/v2/plugins/test-plugin/config",
            json={"config": {"key": "value"}},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "test-plugin"
        assert data["requires_restart"] is True

    @pytest.mark.asyncio()
    async def test_update_config_invalid(self, api_client_with_plugin):
        """Test update_config with invalid config returns 400."""
        client, mock_service = api_client_with_plugin
        mock_service.update_config.side_effect = ValueError("'count' is a required property")

        response = await client.put(
            "/api/v2/plugins/test-plugin/config",
            json={"config": {}},
        )
        assert response.status_code == 400
        data = response.json()
        assert "required" in data["detail"].lower()

    @pytest.mark.asyncio()
    async def test_update_config_not_found(self, api_client_with_plugin):
        """Test update_config returns 404 when not found."""
        client, mock_service = api_client_with_plugin
        mock_service.update_config.return_value = None

        response = await client.put(
            "/api/v2/plugins/nonexistent/config",
            json={"config": {}},
        )
        assert response.status_code == 404


class TestCheckPluginHealth:
    """Tests for GET /api/v2/plugins/{plugin_id}/health."""

    @pytest.mark.asyncio()
    async def test_check_health_healthy(self, api_client_with_plugin):
        """Test check_health returns health status."""
        client, mock_service = api_client_with_plugin
        mock_service.check_health.return_value = {
            "plugin_id": "healthy-plugin",
            "health_status": "healthy",
            "last_health_check": None,
            "error_message": None,
        }

        response = await client.get("/api/v2/plugins/healthy-plugin/health")
        assert response.status_code == 200
        data = response.json()
        assert data["plugin_id"] == "healthy-plugin"
        assert data["health_status"] == "healthy"

    @pytest.mark.asyncio()
    async def test_check_health_not_found(self, api_client_with_plugin):
        """Test check_health returns 404 when not found."""
        client, mock_service = api_client_with_plugin
        mock_service.check_health.return_value = None

        response = await client.get("/api/v2/plugins/nonexistent/health")
        assert response.status_code == 404


class TestPluginAuthRequired:
    """Tests for authentication requirements."""

    @pytest_asyncio.fixture
    async def unauthenticated_client(self):
        """Provide an AsyncClient without authentication."""
        from fastapi import HTTPException

        from webui.auth import get_current_user

        async def require_auth():
            raise HTTPException(status_code=401, detail="Not authenticated")

        app.dependency_overrides[get_current_user] = require_auth

        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client

        app.dependency_overrides.clear()

    @pytest.mark.asyncio()
    async def test_list_plugins_requires_auth(self, unauthenticated_client):
        """Test list_plugins requires authentication."""
        response = await unauthenticated_client.get("/api/v2/plugins")
        assert response.status_code == 401

    @pytest.mark.asyncio()
    async def test_get_plugin_requires_auth(self, unauthenticated_client):
        """Test get_plugin requires authentication."""
        response = await unauthenticated_client.get("/api/v2/plugins/test")
        assert response.status_code == 401

    @pytest.mark.asyncio()
    async def test_enable_plugin_requires_auth(self, unauthenticated_client):
        """Test enable_plugin requires authentication."""
        response = await unauthenticated_client.post("/api/v2/plugins/test/enable")
        assert response.status_code == 401
