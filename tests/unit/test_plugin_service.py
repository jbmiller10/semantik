"""Unit tests for PluginService."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.plugins.manifest import PluginManifest
from shared.plugins.registry import PluginRecord, PluginSource, plugin_registry
from webui.services.plugin_service import (
    PluginService,
    _coerce_type,
    _validate_value,
    validate_config_schema,
)


@pytest.fixture(autouse=True)
def _clear_registry():
    """Clear plugin registry before and after each test."""
    plugin_registry.reset()
    yield
    plugin_registry.reset()


def _make_manifest(
    plugin_id: str = "test-plugin",
    plugin_type: str = "embedding",
    version: str = "1.0.0",
) -> PluginManifest:
    """Create a test plugin manifest."""
    return PluginManifest(
        id=plugin_id,
        type=plugin_type,
        version=version,
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


class TestCoerceType:
    """Tests for _coerce_type helper."""

    def test_string_type(self):
        assert _coerce_type("hello", "string") is True
        assert _coerce_type(123, "string") is False

    def test_integer_type(self):
        assert _coerce_type(42, "integer") is True
        assert _coerce_type(3.14, "integer") is False
        assert _coerce_type(True, "integer") is False  # bool is not int for schema

    def test_number_type(self):
        assert _coerce_type(42, "number") is True
        assert _coerce_type(3.14, "number") is True
        assert _coerce_type(True, "number") is False

    def test_boolean_type(self):
        assert _coerce_type(True, "boolean") is True
        assert _coerce_type(False, "boolean") is True
        assert _coerce_type(1, "boolean") is False

    def test_object_type(self):
        assert _coerce_type({}, "object") is True
        assert _coerce_type({"key": "value"}, "object") is True
        assert _coerce_type([], "object") is False

    def test_array_type(self):
        assert _coerce_type([], "array") is True
        assert _coerce_type([1, 2, 3], "array") is True
        assert _coerce_type({}, "array") is False

    def test_unknown_type(self):
        assert _coerce_type("anything", "unknown") is True


class TestValidateValue:
    """Tests for _validate_value helper."""

    def test_type_mismatch(self):
        errors = _validate_value(123, {"type": "string"}, "config.field")
        assert len(errors) == 1
        assert "expected string" in errors[0]

    def test_enum_validation(self):
        errors = _validate_value("invalid", {"enum": ["a", "b", "c"]}, "config.field")
        assert len(errors) == 1
        assert "must be one of" in errors[0]

    def test_enum_valid(self):
        errors = _validate_value("a", {"enum": ["a", "b", "c"]}, "config.field")
        assert len(errors) == 0

    def test_array_items_validation(self):
        schema = {"type": "array", "items": {"type": "string"}}
        errors = _validate_value(["a", 123, "b"], schema, "config.field")
        assert len(errors) == 1
        assert "[1]" in errors[0]

    def test_object_required_fields(self):
        schema = {"type": "object", "required": ["name", "age"]}
        errors = _validate_value({"name": "test"}, schema, "config")
        assert len(errors) == 1
        assert "config.age: field is required" in errors[0]

    def test_object_properties_validation(self):
        schema = {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
        }
        errors = _validate_value({"count": "not-an-int"}, schema, "config")
        assert len(errors) == 1
        assert "config.count" in errors[0]

    def test_additional_properties_false(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "additionalProperties": False,
        }
        errors = _validate_value({"name": "test", "extra": "bad"}, schema, "config")
        assert len(errors) == 1
        assert "additional properties are not allowed" in errors[0]

    def test_nested_object_validation(self):
        schema = {
            "type": "object",
            "properties": {
                "nested": {
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                }
            },
        }
        errors = _validate_value({"nested": {"value": "wrong"}}, schema, "config")
        assert len(errors) == 1
        assert "config.nested.value" in errors[0]


class TestValidateConfigSchema:
    """Tests for validate_config_schema."""

    def test_none_schema_returns_empty(self):
        errors = validate_config_schema({"key": "value"}, None)
        assert errors == []

    def test_non_dict_config(self):
        errors = validate_config_schema("not-a-dict", {"type": "object"})  # type: ignore[arg-type]
        assert "config must be an object" in errors

    def test_non_object_schema_type(self):
        errors = validate_config_schema({}, {"type": "string"})
        assert "schema type must be 'object'" in errors

    def test_valid_config(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        errors = validate_config_schema({"name": "test"}, schema)
        assert errors == []


class TestPluginService:
    """Tests for PluginService."""

    @pytest.fixture()
    def mock_session(self):
        """Create a mock database session."""
        session = AsyncMock()
        session.commit = AsyncMock()
        session.flush = AsyncMock()
        return session

    @pytest.fixture()
    def service(self, mock_session):
        """Create a PluginService with mocked dependencies."""
        return PluginService(mock_session)

    @pytest.mark.asyncio()
    async def test_list_plugins_empty_registry(self, service):
        """Test list_plugins with no external plugins."""
        with patch.object(service.repo, "list_configs", new=AsyncMock(return_value=[])):
            with patch("webui.services.plugin_service.load_plugins"):
                result = await service.list_plugins()
                assert result == []

    @pytest.mark.asyncio()
    async def test_list_plugins_with_external_plugins(self, service):
        """Test list_plugins returns external plugins."""
        record = _make_record("ext-plugin", "embedding", PluginSource.EXTERNAL)
        plugin_registry.register(record)

        with patch.object(service.repo, "list_configs", new=AsyncMock(return_value=[])):
            with patch("webui.services.plugin_service.load_plugins"):
                result = await service.list_plugins()
                assert len(result) == 1
                assert result[0]["id"] == "ext-plugin"

    @pytest.mark.asyncio()
    async def test_list_plugins_excludes_builtin(self, service):
        """Test list_plugins excludes builtin plugins."""
        builtin = _make_record("builtin-plugin", "embedding", PluginSource.BUILTIN)
        external = _make_record("ext-plugin", "embedding", PluginSource.EXTERNAL)
        plugin_registry.register(builtin)
        plugin_registry.register(external)

        with patch.object(service.repo, "list_configs", new=AsyncMock(return_value=[])):
            with patch("webui.services.plugin_service.load_plugins"):
                result = await service.list_plugins()
                assert len(result) == 1
                assert result[0]["id"] == "ext-plugin"

    @pytest.mark.asyncio()
    async def test_list_plugins_filter_by_type(self, service):
        """Test list_plugins filters by plugin type."""
        embedding = _make_record("embedding-plugin", "embedding", PluginSource.EXTERNAL)
        chunking = _make_record("chunking-plugin", "chunking", PluginSource.EXTERNAL)
        plugin_registry.register(embedding)
        plugin_registry.register(chunking)

        with patch.object(service.repo, "list_configs", new=AsyncMock(return_value=[])):
            with patch("webui.services.plugin_service.load_plugins"):
                result = await service.list_plugins(plugin_type="embedding")
                assert len(result) == 1
                assert result[0]["type"] == "embedding"

    @pytest.mark.asyncio()
    async def test_list_plugins_filter_by_enabled(self, service):
        """Test list_plugins filters by enabled status."""
        record = _make_record("test-plugin", "embedding", PluginSource.EXTERNAL)
        plugin_registry.register(record)

        mock_config = MagicMock()
        mock_config.id = "test-plugin"
        mock_config.enabled = False
        mock_config.config = {}
        mock_config.health_status = "unknown"
        mock_config.last_health_check = None
        mock_config.error_message = None

        with patch.object(service.repo, "list_configs", new=AsyncMock(return_value=[mock_config])):
            with patch("webui.services.plugin_service.load_plugins"):
                result = await service.list_plugins(enabled=True)
                assert len(result) == 0

                result = await service.list_plugins(enabled=False)
                assert len(result) == 1

    @pytest.mark.asyncio()
    async def test_get_plugin_found(self, service):
        """Test get_plugin returns plugin when found."""
        record = _make_record("test-plugin", "embedding", PluginSource.EXTERNAL)
        plugin_registry.register(record)

        with patch.object(service.repo, "get_config", new=AsyncMock(return_value=None)):
            with patch("webui.services.plugin_service.load_plugins"):
                result = await service.get_plugin("test-plugin")
                assert result is not None
                assert result["id"] == "test-plugin"

    @pytest.mark.asyncio()
    async def test_get_plugin_not_found(self, service):
        """Test get_plugin returns None when not found."""
        with patch("webui.services.plugin_service.load_plugins"):
            result = await service.get_plugin("nonexistent")
            assert result is None

    @pytest.mark.asyncio()
    async def test_get_plugin_excludes_builtin(self, service):
        """Test get_plugin excludes builtin plugins."""
        record = _make_record("builtin-plugin", "embedding", PluginSource.BUILTIN)
        plugin_registry.register(record)

        with patch("webui.services.plugin_service.load_plugins"):
            result = await service.get_plugin("builtin-plugin")
            assert result is None

    @pytest.mark.asyncio()
    async def test_update_config_valid(self, service):
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

        with patch.object(service.repo, "upsert_config", new=AsyncMock(return_value=mock_config)):
            with patch("webui.services.plugin_service.load_plugins"):
                with patch("webui.services.plugin_service.get_config_schema", return_value=None):
                    result = await service.update_config("test-plugin", {"key": "value"})
                    assert result is not None
                    assert result["config"]["key"] == "value"

    @pytest.mark.asyncio()
    async def test_update_config_invalid(self, service):
        """Test update_config with invalid config raises ValueError."""
        record = _make_record("test-plugin", "embedding", PluginSource.EXTERNAL)
        plugin_registry.register(record)

        schema = {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
        }

        with patch("webui.services.plugin_service.load_plugins"):
            with patch("webui.services.plugin_service.get_config_schema", return_value=schema):
                with pytest.raises(ValueError, match="field is required"):
                    await service.update_config("test-plugin", {})

    @pytest.mark.asyncio()
    async def test_update_config_not_found(self, service):
        """Test update_config returns None for unknown plugin."""
        with patch("webui.services.plugin_service.load_plugins"):
            result = await service.update_config("nonexistent", {})
            assert result is None

    @pytest.mark.asyncio()
    async def test_set_enabled(self, service):
        """Test set_enabled updates enabled status."""
        record = _make_record("test-plugin", "embedding", PluginSource.EXTERNAL)
        plugin_registry.register(record)

        mock_config = MagicMock()
        mock_config.id = "test-plugin"
        mock_config.enabled = False
        mock_config.config = {}
        mock_config.health_status = "unknown"
        mock_config.last_health_check = None
        mock_config.error_message = None

        with patch.object(service.repo, "upsert_config", new=AsyncMock(return_value=mock_config)):
            with patch("webui.services.plugin_service.load_plugins"):
                result = await service.set_enabled("test-plugin", False)
                assert result is not None
                assert result["enabled"] is False
                assert result["requires_restart"] is True

    @pytest.mark.asyncio()
    async def test_set_enabled_not_found(self, service):
        """Test set_enabled returns None for unknown plugin."""
        with patch("webui.services.plugin_service.load_plugins"):
            result = await service.set_enabled("nonexistent", True)
            assert result is None

    @pytest.mark.asyncio()
    async def test_get_manifest(self, service):
        """Test get_manifest returns manifest dict."""
        record = _make_record("test-plugin", "embedding", PluginSource.EXTERNAL)
        plugin_registry.register(record)

        with patch("webui.services.plugin_service.load_plugins"):
            result = await service.get_manifest("test-plugin")
            assert result is not None
            assert result["id"] == "test-plugin"
            assert result["type"] == "embedding"

    @pytest.mark.asyncio()
    async def test_get_manifest_not_found(self, service):
        """Test get_manifest returns None for unknown plugin."""
        with patch("webui.services.plugin_service.load_plugins"):
            result = await service.get_manifest("nonexistent")
            assert result is None

    @pytest.mark.asyncio()
    async def test_get_config_schema(self, service):
        """Test get_config_schema returns schema."""
        record = _make_record("test-plugin", "embedding", PluginSource.EXTERNAL)
        plugin_registry.register(record)

        schema = {"type": "object", "properties": {}}

        with patch("webui.services.plugin_service.load_plugins"):
            with patch("webui.services.plugin_service.get_config_schema", return_value=schema):
                result = await service.get_config_schema("test-plugin")
                assert result == schema

    @pytest.mark.asyncio()
    async def test_get_config_schema_not_found(self, service):
        """Test get_config_schema returns None for unknown plugin."""
        with patch("webui.services.plugin_service.load_plugins"):
            result = await service.get_config_schema("nonexistent")
            assert result is None

    @pytest.mark.asyncio()
    async def test_check_health_healthy(self, service):
        """Test check_health with healthy plugin."""

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

        with patch.object(service.repo, "update_health", new=AsyncMock(return_value=mock_config)):
            with patch("webui.services.plugin_service.load_plugins"):
                result = await service.check_health("healthy-plugin")
                assert result is not None
                assert result["health_status"] == "healthy"

    @pytest.mark.asyncio()
    async def test_check_health_unhealthy_exception(self, service):
        """Test check_health with exception from health_check."""

        class UnhealthyPlugin:
            @staticmethod
            def health_check():
                raise RuntimeError("Connection failed")

        manifest = _make_manifest("unhealthy-plugin", "embedding")
        record = PluginRecord(
            plugin_type="embedding",
            plugin_id="unhealthy-plugin",
            plugin_version="1.0.0",
            manifest=manifest,
            plugin_class=UnhealthyPlugin,
            source=PluginSource.EXTERNAL,
        )
        plugin_registry.register(record)

        mock_config = MagicMock()
        mock_config.id = "unhealthy-plugin"
        mock_config.health_status = "unhealthy"
        mock_config.last_health_check = None
        mock_config.error_message = "Connection failed"

        with patch.object(service.repo, "update_health", new=AsyncMock(return_value=mock_config)):
            with patch("webui.services.plugin_service.load_plugins"):
                result = await service.check_health("unhealthy-plugin")
                assert result is not None
                assert result["health_status"] == "unhealthy"

    @pytest.mark.asyncio()
    async def test_check_health_async(self, service):
        """Test check_health with async health_check method."""

        class AsyncHealthPlugin:
            @staticmethod
            async def health_check():
                return True

        manifest = _make_manifest("async-plugin", "embedding")
        record = PluginRecord(
            plugin_type="embedding",
            plugin_id="async-plugin",
            plugin_version="1.0.0",
            manifest=manifest,
            plugin_class=AsyncHealthPlugin,
            source=PluginSource.EXTERNAL,
        )
        plugin_registry.register(record)

        mock_config = MagicMock()
        mock_config.id = "async-plugin"
        mock_config.health_status = "healthy"
        mock_config.last_health_check = None
        mock_config.error_message = None

        with patch.object(service.repo, "update_health", new=AsyncMock(return_value=mock_config)):
            with patch("webui.services.plugin_service.load_plugins"):
                result = await service.check_health("async-plugin")
                assert result is not None
                assert result["health_status"] == "healthy"

    @pytest.mark.asyncio()
    async def test_check_health_no_health_check_method(self, service):
        """Test check_health when plugin has no health_check method."""

        class NoHealthPlugin:
            pass

        manifest = _make_manifest("no-health-plugin", "embedding")
        record = PluginRecord(
            plugin_type="embedding",
            plugin_id="no-health-plugin",
            plugin_version="1.0.0",
            manifest=manifest,
            plugin_class=NoHealthPlugin,
            source=PluginSource.EXTERNAL,
        )
        plugin_registry.register(record)

        mock_config = MagicMock()
        mock_config.id = "no-health-plugin"
        mock_config.health_status = "unknown"
        mock_config.last_health_check = None
        mock_config.error_message = None

        with patch.object(service.repo, "update_health", new=AsyncMock(return_value=mock_config)):
            with patch("webui.services.plugin_service.load_plugins"):
                result = await service.check_health("no-health-plugin")
                assert result is not None
                assert result["health_status"] == "unknown"

    @pytest.mark.asyncio()
    async def test_check_health_not_found(self, service):
        """Test check_health returns None for unknown plugin."""
        with patch("webui.services.plugin_service.load_plugins"):
            result = await service.check_health("nonexistent")
            assert result is None

    @pytest.mark.asyncio()
    async def test_run_health_check_type_error(self, service):
        """Test _run_health_check handles TypeError gracefully."""

        class BadHealthPlugin:
            @staticmethod
            def health_check(required_arg):  # noqa: ARG004
                return True

        manifest = _make_manifest("bad-plugin", "embedding")
        record = PluginRecord(
            plugin_type="embedding",
            plugin_id="bad-plugin",
            plugin_version="1.0.0",
            manifest=manifest,
            plugin_class=BadHealthPlugin,
            source=PluginSource.EXTERNAL,
        )

        status, error = await service._run_health_check(record)
        assert status == "unknown"
        assert error is None

    @pytest.mark.asyncio()
    async def test_run_health_check_async_exception(self, service):
        """Test _run_health_check handles async exception."""

        class AsyncFailPlugin:
            @staticmethod
            async def health_check():
                raise ValueError("Async failure")

        manifest = _make_manifest("async-fail-plugin", "embedding")
        record = PluginRecord(
            plugin_type="embedding",
            plugin_id="async-fail-plugin",
            plugin_version="1.0.0",
            manifest=manifest,
            plugin_class=AsyncFailPlugin,
            source=PluginSource.EXTERNAL,
        )

        status, error = await service._run_health_check(record)
        assert status == "unhealthy"
        assert "Async failure" in error

    @pytest.mark.asyncio()
    async def test_run_health_check_returns_false(self, service):
        """Test _run_health_check marks unhealthy when returning False."""

        class UnhealthyPlugin:
            @staticmethod
            def health_check():
                return False

        manifest = _make_manifest("unhealthy-plugin", "embedding")
        record = PluginRecord(
            plugin_type="embedding",
            plugin_id="unhealthy-plugin",
            plugin_version="1.0.0",
            manifest=manifest,
            plugin_class=UnhealthyPlugin,
            source=PluginSource.EXTERNAL,
        )

        status, error = await service._run_health_check(record)
        assert status == "unhealthy"
        assert error is None

    @pytest.mark.asyncio()
    async def test_list_plugins_with_health_refresh(self, service):
        """Test list_plugins with include_health=True."""
        record = _make_record("test-plugin", "embedding", PluginSource.EXTERNAL)
        plugin_registry.register(record)

        mock_config = MagicMock()
        mock_config.id = "test-plugin"
        mock_config.enabled = True
        mock_config.config = {}
        mock_config.health_status = "healthy"
        mock_config.last_health_check = None
        mock_config.error_message = None

        with patch.object(service.repo, "list_configs", new=AsyncMock(return_value=[mock_config])):
            with patch.object(service, "_refresh_health", new=AsyncMock()):
                with patch("webui.services.plugin_service.load_plugins"):
                    result = await service.list_plugins(include_health=True)
                    assert len(result) == 1

    @pytest.mark.asyncio()
    async def test_refresh_health_skips_disabled(self, service, mock_session):
        """Test _refresh_health skips disabled plugins."""
        record = _make_record("test-plugin", "embedding", PluginSource.EXTERNAL)

        mock_config = MagicMock()
        mock_config.id = "test-plugin"
        mock_config.enabled = False
        mock_config.config = {}

        config_map = {"test-plugin": mock_config}

        with patch.object(service, "_check_and_update_health", new=AsyncMock()) as mock_check:
            await service._refresh_health([record], config_map)
            mock_check.assert_not_called()

    @pytest.mark.asyncio()
    async def test_refresh_health_empty_tasks(self, service, mock_session):
        """Test _refresh_health with no enabled plugins."""
        await service._refresh_health([], {})
        # Should complete without error

    @pytest.mark.asyncio()
    async def test_build_plugin_payload_no_config(self, service):
        """Test _build_plugin_payload with no config row."""
        record = _make_record("test-plugin", "embedding", PluginSource.EXTERNAL)
        payload = service._build_plugin_payload(record, None)
        assert payload["id"] == "test-plugin"
        assert payload["enabled"] is True
        assert payload["config"] == {}
        assert payload["health_status"] == "unknown"
