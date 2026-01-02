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

    @pytest.mark.asyncio()
    async def test_get_plugin_rejects_invalid_plugin_id(self, api_client_with_plugin):
        """Invalid plugin_id path params should be rejected by validation."""
        client, _ = api_client_with_plugin

        response = await client.get("/api/v2/plugins/Invalid")
        assert response.status_code == 422


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

    @pytest.mark.asyncio()
    async def test_list_available_requires_auth(self, unauthenticated_client):
        """Test list_available_plugins requires authentication."""
        response = await unauthenticated_client.get("/api/v2/plugins/available")
        assert response.status_code == 401

    @pytest.mark.asyncio()
    async def test_refresh_available_requires_auth(self, unauthenticated_client):
        """Test refresh_available_plugins requires authentication."""
        response = await unauthenticated_client.post("/api/v2/plugins/available/refresh")
        assert response.status_code == 401


class TestListAvailablePlugins:
    """Tests for GET /api/v2/plugins/available."""

    @pytest.mark.asyncio()
    async def test_list_available_returns_plugins(self, api_client_with_plugin):
        """Test list_available_plugins returns available plugins from registry."""
        client, mock_service = api_client_with_plugin

        response = await client.get("/api/v2/plugins/available")
        assert response.status_code == 200
        data = response.json()

        assert "plugins" in data
        assert "semantik_version" in data
        assert isinstance(data["plugins"], list)

    @pytest.mark.asyncio()
    async def test_list_available_filter_by_type(self, api_client_with_plugin):
        """Test list_available_plugins filters by type."""
        client, mock_service = api_client_with_plugin

        response = await client.get("/api/v2/plugins/available", params={"plugin_type": "embedding"})
        assert response.status_code == 200
        data = response.json()

        # All returned plugins should be of the requested type
        for plugin in data["plugins"]:
            assert plugin["type"] == "embedding"

    @pytest.mark.asyncio()
    async def test_list_available_verified_only(self, api_client_with_plugin):
        """Test list_available_plugins filters by verified status."""
        client, mock_service = api_client_with_plugin

        response = await client.get("/api/v2/plugins/available", params={"verified_only": True})
        assert response.status_code == 200
        data = response.json()

        # All returned plugins should be verified
        for plugin in data["plugins"]:
            assert plugin["verified"] is True

    @pytest.mark.asyncio()
    async def test_list_available_includes_compatibility(self, api_client_with_plugin):
        """Test list_available_plugins includes compatibility info."""
        client, mock_service = api_client_with_plugin

        response = await client.get("/api/v2/plugins/available")
        assert response.status_code == 200
        data = response.json()

        # Each plugin should have compatibility fields
        for plugin in data["plugins"]:
            assert "is_compatible" in plugin
            assert "install_command" in plugin

    @pytest.mark.asyncio()
    async def test_list_available_includes_installed_status(self, api_client_with_plugin):
        """Test list_available_plugins marks installed plugins."""
        client, mock_service = api_client_with_plugin

        response = await client.get("/api/v2/plugins/available")
        assert response.status_code == 200
        data = response.json()

        # Each plugin should have is_installed field
        for plugin in data["plugins"]:
            assert "is_installed" in plugin
            assert isinstance(plugin["is_installed"], bool)


class TestRefreshAvailablePlugins:
    """Tests for POST /api/v2/plugins/available/refresh."""

    @pytest.mark.asyncio()
    async def test_refresh_returns_plugins(self, api_client_with_plugin):
        """Test refresh_available_plugins returns fresh plugin list."""
        client, mock_service = api_client_with_plugin

        response = await client.post("/api/v2/plugins/available/refresh")
        assert response.status_code == 200
        data = response.json()

        assert "plugins" in data
        assert "semantik_version" in data
        assert isinstance(data["plugins"], list)

    @pytest.mark.asyncio()
    async def test_refresh_includes_metadata(self, api_client_with_plugin):
        """Test refresh includes registry metadata."""
        client, mock_service = api_client_with_plugin

        response = await client.post("/api/v2/plugins/available/refresh")
        assert response.status_code == 200
        data = response.json()

        # Should include registry metadata
        assert "registry_version" in data
        assert "registry_source" in data


# --- Plugin Installation Tests ---


@pytest_asyncio.fixture
async def admin_client_with_plugin(mock_plugin_service):
    """Provide an AsyncClient with admin user for install/uninstall tests."""
    from webui.auth import get_current_user

    mock_admin_user = {
        "id": 1,
        "username": "admin",
        "email": "admin@example.com",
        "full_name": "Admin User",
        "is_superuser": True,
    }

    async def override_get_current_user() -> dict[str, Any]:
        return mock_admin_user

    async def override_get_plugin_service() -> PluginService:
        return mock_plugin_service

    app.dependency_overrides[get_current_user] = override_get_current_user
    app.dependency_overrides[_get_plugin_service] = override_get_plugin_service

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, mock_plugin_service

    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def non_admin_client_with_plugin(mock_plugin_service):
    """Provide an AsyncClient with non-admin user for install/uninstall tests."""
    from webui.auth import get_current_user

    mock_regular_user = {
        "id": 2,
        "username": "user",
        "email": "user@example.com",
        "full_name": "Regular User",
        "is_superuser": False,
    }

    async def override_get_current_user() -> dict[str, Any]:
        return mock_regular_user

    async def override_get_plugin_service() -> PluginService:
        return mock_plugin_service

    app.dependency_overrides[get_current_user] = override_get_current_user
    app.dependency_overrides[_get_plugin_service] = override_get_plugin_service

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, mock_plugin_service

    app.dependency_overrides.clear()


class TestInstallPlugin:
    """Tests for POST /api/v2/plugins/install."""

    @pytest.mark.asyncio()
    async def test_install_requires_admin(self, non_admin_client_with_plugin):
        """Test install_plugin requires admin access."""
        client, _ = non_admin_client_with_plugin

        response = await client.post(
            "/api/v2/plugins/install",
            json={"plugin_id": "test-plugin"},
        )
        assert response.status_code == 403
        data = response.json()
        assert "admin" in data["detail"].lower()

    @pytest.mark.asyncio()
    async def test_install_plugin_not_in_registry(self, admin_client_with_plugin, monkeypatch):
        """Test install returns 404 for plugin not in registry."""
        client, _ = admin_client_with_plugin

        # Mock fetch_registry to return empty registry
        from shared.plugins import registry_client

        async def mock_fetch_registry(_force_refresh: bool = False):
            return registry_client.PluginRegistry(
                registry_version="1.0.0",
                last_updated="2025-01-01",
                plugins=[],
            )

        monkeypatch.setattr(registry_client, "fetch_registry", mock_fetch_registry)

        response = await client.post(
            "/api/v2/plugins/install",
            json={"plugin_id": "nonexistent-plugin"},
        )
        assert response.status_code == 404
        data = response.json()
        assert "not found in registry" in data["detail"].lower()

    @pytest.mark.asyncio()
    async def test_install_plugin_success(self, admin_client_with_plugin, monkeypatch):
        """Test successful plugin installation."""
        client, _ = admin_client_with_plugin

        # Mock registry to return a plugin
        from shared.plugins import registry_client

        mock_plugin = registry_client.RegistryPlugin(
            id="test-plugin",
            type="embedding",
            name="Test Plugin",
            description="A test plugin",
            author="Test Author",
            repository="https://github.com/test/test-plugin",
            install_command="git+https://github.com/test/test-plugin.git",
            verified=True,
        )

        async def mock_fetch_registry(_force_refresh: bool = False):
            return registry_client.PluginRegistry(
                registry_version="1.0.0",
                last_updated="2025-01-01",
                plugins=[mock_plugin],
            )

        monkeypatch.setattr(registry_client, "fetch_registry", mock_fetch_registry)

        # Mock install_plugin to succeed
        from webui.services import plugin_installer

        def mock_install_plugin(_install_cmd: str, _timeout: int = 300):
            return True, "Successfully installed. Restart required."

        monkeypatch.setattr(plugin_installer, "install_plugin", mock_install_plugin)

        # Mock audit_log
        from shared.plugins import security

        def mock_audit_log(_plugin_id: str, _action: str, _details: dict):
            pass

        monkeypatch.setattr(security, "audit_log", mock_audit_log)

        response = await client.post(
            "/api/v2/plugins/install",
            json={"plugin_id": "test-plugin"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["restart_required"] is True

    @pytest.mark.asyncio()
    async def test_install_plugin_failure(self, admin_client_with_plugin, monkeypatch):
        """Test plugin installation failure."""
        client, _ = admin_client_with_plugin

        # Mock registry
        from shared.plugins import registry_client

        mock_plugin = registry_client.RegistryPlugin(
            id="test-plugin",
            type="embedding",
            name="Test Plugin",
            description="A test plugin",
            author="Test Author",
            repository="https://github.com/test/test-plugin",
            install_command="git+https://github.com/test/test-plugin.git",
            verified=True,
        )

        async def mock_fetch_registry(_force_refresh: bool = False):
            return registry_client.PluginRegistry(
                registry_version="1.0.0",
                last_updated="2025-01-01",
                plugins=[mock_plugin],
            )

        monkeypatch.setattr(registry_client, "fetch_registry", mock_fetch_registry)

        # Mock install_plugin to fail
        from webui.services import plugin_installer

        def mock_install_plugin(_install_cmd: str, _timeout: int = 300):
            return False, "Installation failed: pip error"

        monkeypatch.setattr(plugin_installer, "install_plugin", mock_install_plugin)

        # Mock audit_log
        from shared.plugins import security

        def mock_audit_log(_plugin_id: str, _action: str, _details: dict):
            pass

        monkeypatch.setattr(security, "audit_log", mock_audit_log)

        response = await client.post(
            "/api/v2/plugins/install",
            json={"plugin_id": "test-plugin"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "failed" in data["message"].lower()

    @pytest.mark.asyncio()
    async def test_install_plugin_with_version(self, admin_client_with_plugin, monkeypatch):
        """Test plugin installation with specific version."""
        client, _ = admin_client_with_plugin
        captured_cmd = []

        # Mock registry
        from shared.plugins import registry_client

        mock_plugin = registry_client.RegistryPlugin(
            id="test-plugin",
            type="embedding",
            name="Test Plugin",
            description="A test plugin",
            author="Test Author",
            repository="https://github.com/test/test-plugin",
            install_command="git+https://github.com/test/test-plugin.git",
            verified=True,
        )

        async def mock_fetch_registry(_force_refresh: bool = False):
            return registry_client.PluginRegistry(
                registry_version="1.0.0",
                last_updated="2025-01-01",
                plugins=[mock_plugin],
            )

        monkeypatch.setattr(registry_client, "fetch_registry", mock_fetch_registry)

        # Mock install_plugin to capture command
        from webui.services import plugin_installer

        def mock_install_plugin(install_cmd: str, _timeout: int = 300):
            captured_cmd.append(install_cmd)
            return True, "Successfully installed."

        monkeypatch.setattr(plugin_installer, "install_plugin", mock_install_plugin)

        # Mock audit_log
        from shared.plugins import security

        def mock_audit_log(_plugin_id: str, _action: str, _details: dict):
            pass

        monkeypatch.setattr(security, "audit_log", mock_audit_log)

        response = await client.post(
            "/api/v2/plugins/install",
            json={"plugin_id": "test-plugin", "version": "v1.0.0"},
        )
        assert response.status_code == 200
        # Verify version was appended to install command
        assert len(captured_cmd) == 1
        assert "v1.0.0" in captured_cmd[0]

    @pytest.mark.asyncio()
    async def test_install_strips_pip_install_prefix(self, admin_client_with_plugin, monkeypatch):
        """Test that 'pip install ' prefix is stripped from install_command."""
        client, _ = admin_client_with_plugin
        captured_cmd = []

        # Mock registry with install_command that has "pip install " prefix
        from shared.plugins import registry_client

        mock_plugin = registry_client.RegistryPlugin(
            id="test-plugin",
            type="embedding",
            name="Test Plugin",
            description="A test plugin",
            author="Test Author",
            repository="https://github.com/test/test-plugin",
            install_command="pip install git+https://github.com/test/test-plugin.git",
            verified=True,
        )

        async def mock_fetch_registry(_force_refresh: bool = False):
            return registry_client.PluginRegistry(
                registry_version="1.0.0",
                last_updated="2025-01-01",
                plugins=[mock_plugin],
            )

        monkeypatch.setattr(registry_client, "fetch_registry", mock_fetch_registry)

        # Mock install_plugin to capture command
        from webui.services import plugin_installer

        def mock_install_plugin(install_cmd: str, _timeout: int = 300):
            captured_cmd.append(install_cmd)
            return True, "Successfully installed."

        monkeypatch.setattr(plugin_installer, "install_plugin", mock_install_plugin)

        # Mock audit_log
        from shared.plugins import security

        def mock_audit_log(_plugin_id: str, _action: str, _details: dict):
            pass

        monkeypatch.setattr(security, "audit_log", mock_audit_log)

        response = await client.post(
            "/api/v2/plugins/install",
            json={"plugin_id": "test-plugin"},
        )
        assert response.status_code == 200
        # Verify "pip install " prefix was stripped
        assert len(captured_cmd) == 1
        assert captured_cmd[0] == "git+https://github.com/test/test-plugin.git"
        assert not captured_cmd[0].startswith("pip install ")

    @pytest.mark.asyncio()
    async def test_install_rejects_invalid_version(self, admin_client_with_plugin):
        """install should reject versions/refs with whitespace."""
        client, _ = admin_client_with_plugin

        response = await client.post(
            "/api/v2/plugins/install",
            json={"plugin_id": "test-plugin", "version": "bad ref"},
        )
        assert response.status_code == 422


class TestUninstallPlugin:
    """Tests for DELETE /api/v2/plugins/{plugin_id}/uninstall."""

    @pytest.mark.asyncio()
    async def test_uninstall_requires_admin(self, non_admin_client_with_plugin):
        """Test uninstall_plugin requires admin access."""
        client, _ = non_admin_client_with_plugin

        response = await client.delete("/api/v2/plugins/test-plugin/uninstall")
        assert response.status_code == 403
        data = response.json()
        assert "admin" in data["detail"].lower()

    @pytest.mark.asyncio()
    async def test_uninstall_plugin_not_in_registry(self, admin_client_with_plugin, monkeypatch):
        """Test uninstall returns 404 for plugin not in registry."""
        client, _ = admin_client_with_plugin

        # Mock fetch_registry to return empty registry
        from shared.plugins import registry_client

        async def mock_fetch_registry(_force_refresh: bool = False):
            return registry_client.PluginRegistry(
                registry_version="1.0.0",
                last_updated="2025-01-01",
                plugins=[],
            )

        monkeypatch.setattr(registry_client, "fetch_registry", mock_fetch_registry)

        response = await client.delete("/api/v2/plugins/nonexistent-plugin/uninstall")
        assert response.status_code == 404
        data = response.json()
        assert "not found in registry" in data["detail"].lower()

    @pytest.mark.asyncio()
    async def test_uninstall_plugin_success(self, admin_client_with_plugin, monkeypatch):
        """Test successful plugin uninstallation."""
        client, _ = admin_client_with_plugin

        # Mock registry
        from shared.plugins import registry_client

        mock_plugin = registry_client.RegistryPlugin(
            id="test-plugin",
            type="embedding",
            name="Test Plugin",
            description="A test plugin",
            author="Test Author",
            repository="https://github.com/test/test-plugin",
            pypi="semantik-plugin-test",
            verified=True,
        )

        async def mock_fetch_registry(_force_refresh: bool = False):
            return registry_client.PluginRegistry(
                registry_version="1.0.0",
                last_updated="2025-01-01",
                plugins=[mock_plugin],
            )

        monkeypatch.setattr(registry_client, "fetch_registry", mock_fetch_registry)

        # Mock uninstall_plugin to succeed
        from webui.services import plugin_installer

        def mock_uninstall_plugin(_package_name: str):
            return True, "Uninstalled successfully. Restart required."

        monkeypatch.setattr(plugin_installer, "uninstall_plugin", mock_uninstall_plugin)

        # Mock audit_log
        from shared.plugins import security

        def mock_audit_log(_plugin_id: str, _action: str, _details: dict):
            pass

        monkeypatch.setattr(security, "audit_log", mock_audit_log)

        response = await client.delete("/api/v2/plugins/test-plugin/uninstall")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["restart_required"] is True

    @pytest.mark.asyncio()
    async def test_uninstall_plugin_failure(self, admin_client_with_plugin, monkeypatch):
        """Test plugin uninstallation failure."""
        client, _ = admin_client_with_plugin

        # Mock registry
        from shared.plugins import registry_client

        mock_plugin = registry_client.RegistryPlugin(
            id="test-plugin",
            type="embedding",
            name="Test Plugin",
            description="A test plugin",
            author="Test Author",
            repository="https://github.com/test/test-plugin",
            pypi="semantik-plugin-test",
            verified=True,
        )

        async def mock_fetch_registry(_force_refresh: bool = False):
            return registry_client.PluginRegistry(
                registry_version="1.0.0",
                last_updated="2025-01-01",
                plugins=[mock_plugin],
            )

        monkeypatch.setattr(registry_client, "fetch_registry", mock_fetch_registry)

        # Mock uninstall_plugin to fail
        from webui.services import plugin_installer

        def mock_uninstall_plugin(_package_name: str):
            return False, "Plugin not found in plugins directory"

        monkeypatch.setattr(plugin_installer, "uninstall_plugin", mock_uninstall_plugin)

        # Mock audit_log
        from shared.plugins import security

        def mock_audit_log(_plugin_id: str, _action: str, _details: dict):
            pass

        monkeypatch.setattr(security, "audit_log", mock_audit_log)

        response = await client.delete("/api/v2/plugins/test-plugin/uninstall")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
