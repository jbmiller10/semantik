"""Integration tests for Plugin API endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from shared.plugins.manifest import PluginManifest
from shared.plugins.registry import PluginRecord, PluginSource, plugin_registry
from webui.main import app

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


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
async def api_client_with_plugin(db_session, test_user_db, use_fakeredis, reset_redis_manager):
    """Provide an AsyncClient with plugin-related dependencies mocked."""

    from shared.database import get_db
    from webui.auth import get_current_user

    _ = use_fakeredis
    _ = reset_redis_manager

    async def override_get_db() -> AsyncGenerator[Any, None]:
        yield db_session

    async def override_get_current_user() -> dict[str, Any]:
        return {
            "id": test_user_db.id,
            "username": test_user_db.username,
            "email": test_user_db.email,
            "full_name": test_user_db.full_name,
        }

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = override_get_current_user

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    app.dependency_overrides.clear()


class TestListPlugins:
    """Tests for GET /api/v2/plugins."""

    @pytest.mark.asyncio()
    async def test_list_plugins_empty(self, api_client_with_plugin):
        """Test list_plugins returns empty list when no external plugins."""
        with patch("webui.services.plugin_service.load_plugins"):
            with patch("webui.services.plugin_service.PluginConfigRepository") as mock_repo_cls:
                mock_repo = AsyncMock()
                mock_repo.list_configs = AsyncMock(return_value=[])
                mock_repo_cls.return_value = mock_repo

                response = await api_client_with_plugin.get("/api/v2/plugins")
                assert response.status_code == 200
                data = response.json()
                assert "plugins" in data
                assert data["plugins"] == []

    @pytest.mark.asyncio()
    async def test_list_plugins_with_plugins(self, api_client_with_plugin):
        """Test list_plugins returns external plugins."""
        record = _make_record("test-plugin", "embedding", PluginSource.EXTERNAL)
        plugin_registry.register(record)

        with patch("webui.services.plugin_service.load_plugins"):
            with patch("webui.services.plugin_service.PluginConfigRepository") as mock_repo_cls:
                mock_repo = AsyncMock()
                mock_repo.list_configs = AsyncMock(return_value=[])
                mock_repo_cls.return_value = mock_repo

                response = await api_client_with_plugin.get("/api/v2/plugins")
                assert response.status_code == 200
                data = response.json()
                assert len(data["plugins"]) == 1
                assert data["plugins"][0]["id"] == "test-plugin"

    @pytest.mark.asyncio()
    async def test_list_plugins_filter_by_type(self, api_client_with_plugin):
        """Test list_plugins filters by plugin_type."""
        embedding = _make_record("embed-plugin", "embedding", PluginSource.EXTERNAL)
        chunking = _make_record("chunk-plugin", "chunking", PluginSource.EXTERNAL)
        plugin_registry.register(embedding)
        plugin_registry.register(chunking)

        with patch("webui.services.plugin_service.load_plugins"):
            with patch("webui.services.plugin_service.PluginConfigRepository") as mock_repo_cls:
                mock_repo = AsyncMock()
                mock_repo.list_configs = AsyncMock(return_value=[])
                mock_repo_cls.return_value = mock_repo

                response = await api_client_with_plugin.get("/api/v2/plugins", params={"plugin_type": "embedding"})
                assert response.status_code == 200
                data = response.json()
                assert len(data["plugins"]) == 1
                assert data["plugins"][0]["type"] == "embedding"


class TestGetPlugin:
    """Tests for GET /api/v2/plugins/{plugin_id}."""

    @pytest.mark.asyncio()
    async def test_get_plugin_found(self, api_client_with_plugin):
        """Test get_plugin returns plugin when found."""
        record = _make_record("test-plugin", "embedding", PluginSource.EXTERNAL)
        plugin_registry.register(record)

        with patch("webui.services.plugin_service.load_plugins"):
            with patch("webui.services.plugin_service.PluginConfigRepository") as mock_repo_cls:
                mock_repo = AsyncMock()
                mock_repo.get_config = AsyncMock(return_value=None)
                mock_repo_cls.return_value = mock_repo

                response = await api_client_with_plugin.get("/api/v2/plugins/test-plugin")
                assert response.status_code == 200
                data = response.json()
                assert data["id"] == "test-plugin"
                assert data["type"] == "embedding"

    @pytest.mark.asyncio()
    async def test_get_plugin_not_found(self, api_client_with_plugin):
        """Test get_plugin returns 404 when not found."""
        with patch("webui.services.plugin_service.load_plugins"):
            response = await api_client_with_plugin.get("/api/v2/plugins/nonexistent")
            assert response.status_code == 404
            data = response.json()
            assert "not found" in data["detail"].lower()


class TestGetPluginManifest:
    """Tests for GET /api/v2/plugins/{plugin_id}/manifest."""

    @pytest.mark.asyncio()
    async def test_get_manifest_found(self, api_client_with_plugin):
        """Test get_manifest returns manifest."""
        record = _make_record("test-plugin", "embedding", PluginSource.EXTERNAL)
        plugin_registry.register(record)

        with patch("webui.services.plugin_service.load_plugins"):
            response = await api_client_with_plugin.get("/api/v2/plugins/test-plugin/manifest")
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == "test-plugin"
            assert data["type"] == "embedding"
            assert data["display_name"] == "Test Plugin"

    @pytest.mark.asyncio()
    async def test_get_manifest_not_found(self, api_client_with_plugin):
        """Test get_manifest returns 404 when not found."""
        with patch("webui.services.plugin_service.load_plugins"):
            response = await api_client_with_plugin.get("/api/v2/plugins/nonexistent/manifest")
            assert response.status_code == 404


class TestGetPluginConfigSchema:
    """Tests for GET /api/v2/plugins/{plugin_id}/config-schema."""

    @pytest.mark.asyncio()
    async def test_get_config_schema_found(self, api_client_with_plugin):
        """Test get_config_schema returns schema."""
        record = _make_record("test-plugin", "embedding", PluginSource.EXTERNAL)
        plugin_registry.register(record)

        with patch("webui.services.plugin_service.load_plugins"):
            with patch("webui.services.plugin_service.PluginConfigRepository") as mock_repo_cls:
                mock_repo = AsyncMock()
                mock_repo.get_config = AsyncMock(return_value=None)
                mock_repo_cls.return_value = mock_repo

                with patch("webui.services.plugin_service.get_config_schema") as mock_get_schema:
                    mock_get_schema.return_value = {"type": "object", "properties": {}}

                    response = await api_client_with_plugin.get("/api/v2/plugins/test-plugin/config-schema")
                    assert response.status_code == 200
                    data = response.json()
                    assert data["type"] == "object"

    @pytest.mark.asyncio()
    async def test_get_config_schema_not_found(self, api_client_with_plugin):
        """Test get_config_schema returns 404 when plugin not found."""
        with patch("webui.services.plugin_service.load_plugins"):
            response = await api_client_with_plugin.get("/api/v2/plugins/nonexistent/config-schema")
            assert response.status_code == 404


class TestEnableDisablePlugin:
    """Tests for POST /api/v2/plugins/{plugin_id}/enable|disable."""

    @pytest.mark.asyncio()
    async def test_enable_plugin(self, api_client_with_plugin):
        """Test enable_plugin enables a plugin."""
        record = _make_record("test-plugin", "embedding", PluginSource.EXTERNAL)
        plugin_registry.register(record)

        mock_config = MagicMock()
        mock_config.id = "test-plugin"
        mock_config.enabled = True
        mock_config.config = {}
        mock_config.health_status = "unknown"
        mock_config.last_health_check = None
        mock_config.error_message = None

        with patch("webui.services.plugin_service.load_plugins"):
            with patch("webui.services.plugin_service.PluginConfigRepository") as mock_repo_cls:
                mock_repo = AsyncMock()
                mock_repo.upsert_config = AsyncMock(return_value=mock_config)
                mock_repo_cls.return_value = mock_repo

                response = await api_client_with_plugin.post("/api/v2/plugins/test-plugin/enable")
                assert response.status_code == 200
                data = response.json()
                assert data["plugin_id"] == "test-plugin"
                assert data["enabled"] is True
                assert data["requires_restart"] is True

    @pytest.mark.asyncio()
    async def test_enable_plugin_not_found(self, api_client_with_plugin):
        """Test enable_plugin returns 404 when not found."""
        with patch("webui.services.plugin_service.load_plugins"):
            response = await api_client_with_plugin.post("/api/v2/plugins/nonexistent/enable")
            assert response.status_code == 404

    @pytest.mark.asyncio()
    async def test_disable_plugin(self, api_client_with_plugin):
        """Test disable_plugin disables a plugin."""
        record = _make_record("test-plugin", "embedding", PluginSource.EXTERNAL)
        plugin_registry.register(record)

        mock_config = MagicMock()
        mock_config.id = "test-plugin"
        mock_config.enabled = False
        mock_config.config = {}
        mock_config.health_status = "unknown"
        mock_config.last_health_check = None
        mock_config.error_message = None

        with patch("webui.services.plugin_service.load_plugins"):
            with patch("webui.services.plugin_service.PluginConfigRepository") as mock_repo_cls:
                mock_repo = AsyncMock()
                mock_repo.upsert_config = AsyncMock(return_value=mock_config)
                mock_repo_cls.return_value = mock_repo

                response = await api_client_with_plugin.post("/api/v2/plugins/test-plugin/disable")
                assert response.status_code == 200
                data = response.json()
                assert data["plugin_id"] == "test-plugin"
                assert data["enabled"] is False
                assert data["requires_restart"] is True

    @pytest.mark.asyncio()
    async def test_disable_plugin_not_found(self, api_client_with_plugin):
        """Test disable_plugin returns 404 when not found."""
        with patch("webui.services.plugin_service.load_plugins"):
            response = await api_client_with_plugin.post("/api/v2/plugins/nonexistent/disable")
            assert response.status_code == 404


class TestUpdatePluginConfig:
    """Tests for PUT /api/v2/plugins/{plugin_id}/config."""

    @pytest.mark.asyncio()
    async def test_update_config_valid(self, api_client_with_plugin):
        """Test update_config with valid config."""
        record = _make_record("test-plugin", "embedding", PluginSource.EXTERNAL)
        plugin_registry.register(record)

        mock_config = MagicMock()
        mock_config.id = "test-plugin"
        mock_config.enabled = True
        mock_config.config = {"key": "value"}
        mock_config.health_status = "unknown"
        mock_config.last_health_check = None
        mock_config.error_message = None

        with patch("webui.services.plugin_service.load_plugins"):
            with patch("webui.services.plugin_service.PluginConfigRepository") as mock_repo_cls:
                mock_repo = AsyncMock()
                mock_repo.upsert_config = AsyncMock(return_value=mock_config)
                mock_repo_cls.return_value = mock_repo

                with patch("webui.services.plugin_service.get_config_schema", return_value=None):
                    response = await api_client_with_plugin.put(
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
        record = _make_record("test-plugin", "embedding", PluginSource.EXTERNAL)
        plugin_registry.register(record)

        schema = {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
        }

        with patch("webui.services.plugin_service.load_plugins"):
            with patch("webui.services.plugin_service.get_config_schema", return_value=schema):
                response = await api_client_with_plugin.put(
                    "/api/v2/plugins/test-plugin/config",
                    json={"config": {}},
                )
                assert response.status_code == 400
                data = response.json()
                assert "required" in data["detail"].lower()

    @pytest.mark.asyncio()
    async def test_update_config_not_found(self, api_client_with_plugin):
        """Test update_config returns 404 when not found."""
        with patch("webui.services.plugin_service.load_plugins"):
            response = await api_client_with_plugin.put(
                "/api/v2/plugins/nonexistent/config",
                json={"config": {}},
            )
            assert response.status_code == 404


class TestCheckPluginHealth:
    """Tests for GET /api/v2/plugins/{plugin_id}/health."""

    @pytest.mark.asyncio()
    async def test_check_health_healthy(self, api_client_with_plugin):
        """Test check_health returns health status."""

        class HealthyPlugin:
            @staticmethod
            def health_check():
                return True

        manifest = _make_manifest("healthy-plugin", "embedding")
        record = PluginRecord(
            plugin_type="embedding",
            plugin_id="healthy-plugin",
            plugin_version="1.0.0",
            manifest=manifest,
            plugin_class=HealthyPlugin,
            source=PluginSource.EXTERNAL,
        )
        plugin_registry.register(record)

        mock_config = MagicMock()
        mock_config.id = "healthy-plugin"
        mock_config.health_status = "healthy"
        mock_config.last_health_check = None
        mock_config.error_message = None

        with patch("webui.services.plugin_service.load_plugins"):
            with patch("webui.services.plugin_service.PluginConfigRepository") as mock_repo_cls:
                mock_repo = AsyncMock()
                mock_repo.update_health = AsyncMock(return_value=mock_config)
                mock_repo_cls.return_value = mock_repo

                response = await api_client_with_plugin.get("/api/v2/plugins/healthy-plugin/health")
                assert response.status_code == 200
                data = response.json()
                assert data["plugin_id"] == "healthy-plugin"
                assert data["health_status"] == "healthy"

    @pytest.mark.asyncio()
    async def test_check_health_not_found(self, api_client_with_plugin):
        """Test check_health returns 404 when not found."""
        with patch("webui.services.plugin_service.load_plugins"):
            response = await api_client_with_plugin.get("/api/v2/plugins/nonexistent/health")
            assert response.status_code == 404


class TestPluginAuthRequired:
    """Tests for authentication requirements."""

    @pytest_asyncio.fixture
    async def unauthenticated_client(self, db_session, use_fakeredis, reset_redis_manager):
        """Provide an AsyncClient without authentication."""

        from shared.database import get_db

        _ = use_fakeredis
        _ = reset_redis_manager

        async def override_get_db() -> AsyncGenerator[Any, None]:
            yield db_session

        app.dependency_overrides[get_db] = override_get_db

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
