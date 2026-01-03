"""End-to-end plugin lifecycle integration tests.

These tests verify the complete plugin lifecycle operations
at the service and registry level.

Phase 4.3 of the plugin system remediation plan.
"""

from __future__ import annotations

from typing import Any, ClassVar
from unittest.mock import MagicMock

import pytest

from shared.plugins.manifest import PluginManifest
from shared.plugins.registry import PluginRecord, PluginRegistry, PluginSource


def _make_test_plugin_class(plugin_id: str, plugin_type: str = "embedding"):
    """Create a mock plugin class with proper attributes."""

    class TestPlugin:
        PLUGIN_TYPE: ClassVar[str] = plugin_type
        PLUGIN_ID: ClassVar[str] = plugin_id
        PLUGIN_VERSION: ClassVar[str] = "1.0.0"

        @classmethod
        def get_manifest(cls) -> PluginManifest:
            return PluginManifest(
                id=cls.PLUGIN_ID,
                type=cls.PLUGIN_TYPE,
                version=cls.PLUGIN_VERSION,
                display_name=f"Test Plugin {plugin_id}",
                description="A test plugin for lifecycle testing",
                author="Test Author",
            )

        @classmethod
        def get_config_schema(cls) -> dict[str, Any]:
            return {
                "type": "object",
                "properties": {
                    "api_key_env": {"type": "string", "description": "API key env var"},
                    "model": {"type": "string", "default": "test-model"},
                },
                "required": ["api_key_env"],
            }

        @classmethod
        async def health_check(cls, _config: dict[str, Any] | None = None) -> bool:
            return True

    return TestPlugin


def _make_manifest(plugin_id: str, plugin_type: str = "embedding") -> PluginManifest:
    """Create a test manifest."""
    return PluginManifest(
        id=plugin_id,
        type=plugin_type,
        version="1.0.0",
        display_name=f"Test Plugin {plugin_id}",
        description="A test plugin for lifecycle testing",
        author="Test Author",
    )


def _make_record(plugin_id: str, plugin_type: str = "embedding") -> PluginRecord:
    """Create a test plugin record."""
    plugin_cls = _make_test_plugin_class(plugin_id, plugin_type)
    return PluginRecord(
        plugin_type=plugin_type,
        plugin_id=plugin_id,
        plugin_version="1.0.0",
        manifest=_make_manifest(plugin_id, plugin_type),
        plugin_class=plugin_cls,
        source=PluginSource.EXTERNAL,
    )


class TestPluginRegistryLifecycle:
    """Test plugin registry lifecycle operations."""

    @pytest.fixture()
    def registry(self):
        """Create a fresh registry for each test."""
        return PluginRegistry()

    def test_full_registry_lifecycle(self, registry):
        """Test register → get → list → disable → check disabled → reset."""
        plugin_id = "lifecycle-test"

        # Step 1: Register plugin
        record = _make_record(plugin_id)
        result = registry.register(record)
        assert result is True

        # Step 2: Get plugin
        retrieved = registry.get("embedding", plugin_id)
        assert retrieved is not None
        assert retrieved.plugin_id == plugin_id
        assert retrieved.plugin_version == "1.0.0"

        # Step 3: List plugins
        all_plugins = registry.get_by_type("embedding")
        assert plugin_id in all_plugins

        # Step 4: Disable plugin
        registry.set_disabled({plugin_id})
        assert registry.is_disabled(plugin_id) is True

        # Step 5: Verify still accessible but marked disabled
        still_there = registry.get("embedding", plugin_id)
        assert still_there is not None
        assert registry.is_disabled(still_there.plugin_id) is True

        # Step 6: Reset (simulates uninstall/restart)
        registry.reset()
        assert registry.get("embedding", plugin_id) is None
        assert registry.is_disabled(plugin_id) is False

    def test_multi_plugin_lifecycle(self, registry):
        """Test lifecycle with multiple plugins of different types."""
        plugins = [
            ("embed-1", "embedding"),
            ("embed-2", "embedding"),
            ("chunk-1", "chunking"),
            ("conn-1", "connector"),
        ]

        # Register all
        for plugin_id, plugin_type in plugins:
            record = _make_record(plugin_id, plugin_type)
            result = registry.register(record)
            assert result is True

        # Verify counts by type
        assert len(registry.get_by_type("embedding")) == 2
        assert len(registry.get_by_type("chunking")) == 1
        assert len(registry.get_by_type("connector")) == 1

        # Disable some
        registry.set_disabled({"embed-1", "chunk-1"})
        assert registry.is_disabled("embed-1") is True
        assert registry.is_disabled("embed-2") is False
        assert registry.is_disabled("chunk-1") is True

        # List all types
        types = registry.list_types()
        assert set(types) == {"embedding", "chunking", "connector"}

        # Find by ID
        found = registry.find_by_id("conn-1")
        assert found is not None
        assert found.plugin_type == "connector"

    def test_enable_disable_cycle(self, registry):
        """Test enable/disable cycling."""
        plugin_id = "toggle-test"
        record = _make_record(plugin_id)
        registry.register(record)

        # Initially not disabled
        assert registry.is_disabled(plugin_id) is False

        # Disable
        registry.set_disabled({plugin_id})
        assert registry.is_disabled(plugin_id) is True

        # Re-enable (set empty disabled set or new set without plugin)
        registry.set_disabled(set())
        assert registry.is_disabled(plugin_id) is False

        # Disable again
        registry.set_disabled({plugin_id})
        assert registry.is_disabled(plugin_id) is True

    def test_loaded_types_tracking(self, registry):
        """Test tracking of loaded plugin types."""
        # Initially none loaded
        assert registry.is_loaded({"embedding"}) is False

        # Mark loaded
        registry.mark_loaded({"embedding"})
        assert registry.is_loaded({"embedding"}) is True
        assert registry.is_loaded({"chunking"}) is False

        # Mark more loaded
        registry.mark_loaded({"chunking", "connector"})
        assert registry.is_loaded({"embedding", "chunking"}) is True
        assert registry.is_loaded({"embedding", "chunking", "connector"}) is True

        # Check subset
        assert registry.is_loaded({"embedding"}) is True


@pytest.mark.asyncio()
class TestPluginHealthLifecycle:
    """Test plugin health check lifecycle."""

    async def test_health_check_success(self):
        """Test health check returns success for healthy plugin."""
        plugin_cls = _make_test_plugin_class("health-test")

        result = await plugin_cls.health_check({})
        assert result is True

    async def test_health_check_with_config(self):
        """Test health check receives config correctly."""
        received_config = None

        class ConfigAwarePlugin:
            PLUGIN_TYPE: ClassVar[str] = "embedding"
            PLUGIN_ID: ClassVar[str] = "config-aware"
            PLUGIN_VERSION: ClassVar[str] = "1.0.0"

            @classmethod
            async def health_check(cls, config: dict[str, Any] | None = None) -> bool:
                nonlocal received_config
                received_config = config
                return config is not None and "api_key" in config

        test_config = {"api_key": "test-key"}
        result = await ConfigAwarePlugin.health_check(test_config)

        assert result is True
        assert received_config == test_config

    async def test_health_check_failure(self):
        """Test health check can return failure."""

        class UnhealthyPlugin:
            PLUGIN_TYPE: ClassVar[str] = "embedding"
            PLUGIN_ID: ClassVar[str] = "unhealthy"
            PLUGIN_VERSION: ClassVar[str] = "1.0.0"

            @classmethod
            async def health_check(cls, _config: dict[str, Any] | None = None) -> bool:
                return False

        result = await UnhealthyPlugin.health_check({})
        assert result is False

    async def test_health_check_exception_handling(self):
        """Test health check exceptions are caught properly."""

        class FailingPlugin:
            PLUGIN_TYPE: ClassVar[str] = "embedding"
            PLUGIN_ID: ClassVar[str] = "failing"
            PLUGIN_VERSION: ClassVar[str] = "1.0.0"

            @classmethod
            async def health_check(cls, _config: dict[str, Any] | None = None) -> bool:
                raise ConnectionError("Cannot connect to API")

        # Should raise, caller needs to handle
        with pytest.raises(ConnectionError):
            await FailingPlugin.health_check({})


class TestPluginManifestLifecycle:
    """Test plugin manifest operations."""

    def test_manifest_retrieval(self):
        """Test manifest retrieval from plugin class."""
        plugin_cls = _make_test_plugin_class("manifest-test")

        manifest = plugin_cls.get_manifest()

        assert manifest.id == "manifest-test"
        assert manifest.type == "embedding"
        assert manifest.version == "1.0.0"
        assert manifest.display_name == "Test Plugin manifest-test"

    def test_manifest_in_record(self):
        """Test manifest is stored in record."""
        record = _make_record("record-manifest-test")

        assert record.manifest.id == "record-manifest-test"
        assert record.manifest.type == "embedding"

    def test_config_schema_retrieval(self):
        """Test config schema retrieval from plugin class."""
        plugin_cls = _make_test_plugin_class("schema-test")

        schema = plugin_cls.get_config_schema()

        assert schema["type"] == "object"
        assert "properties" in schema
        assert "api_key_env" in schema["properties"]
        assert "required" in schema


class TestPluginRecordLifecycle:
    """Test plugin record creation and properties."""

    def test_record_properties(self):
        """Test record has expected properties."""
        record = _make_record("props-test", "chunking")

        assert record.plugin_id == "props-test"
        assert record.plugin_type == "chunking"
        assert record.plugin_version == "1.0.0"
        assert record.source == PluginSource.EXTERNAL
        assert record.manifest is not None
        assert record.plugin_class is not None

    def test_record_source_types(self):
        """Test different record source types."""
        # External plugin
        external = PluginRecord(
            plugin_type="embedding",
            plugin_id="ext-plugin",
            plugin_version="1.0.0",
            manifest=_make_manifest("ext-plugin"),
            plugin_class=MagicMock,
            source=PluginSource.EXTERNAL,
        )
        assert external.source == PluginSource.EXTERNAL

        # Builtin plugin
        builtin = PluginRecord(
            plugin_type="embedding",
            plugin_id="builtin-plugin",
            plugin_version="1.0.0",
            manifest=_make_manifest("builtin-plugin"),
            plugin_class=MagicMock,
            source=PluginSource.BUILTIN,
        )
        assert builtin.source == PluginSource.BUILTIN


class TestPluginListingLifecycle:
    """Test plugin listing operations."""

    @pytest.fixture()
    def populated_registry(self):
        """Create a registry with various plugins."""
        registry = PluginRegistry()

        plugins = [
            ("embed-a", "embedding", PluginSource.BUILTIN),
            ("embed-b", "embedding", PluginSource.EXTERNAL),
            ("embed-c", "embedding", PluginSource.EXTERNAL),
            ("chunk-a", "chunking", PluginSource.BUILTIN),
            ("chunk-b", "chunking", PluginSource.EXTERNAL),
            ("conn-a", "connector", PluginSource.EXTERNAL),
        ]

        for plugin_id, plugin_type, source in plugins:
            record = PluginRecord(
                plugin_type=plugin_type,
                plugin_id=plugin_id,
                plugin_version="1.0.0",
                manifest=_make_manifest(plugin_id, plugin_type),
                plugin_class=_make_test_plugin_class(plugin_id, plugin_type),
                source=source,
            )
            registry.register(record)

        return registry

    def test_list_all_records(self, populated_registry):
        """Test listing all records."""
        records = populated_registry.list_records()
        assert len(records) == 6

    def test_list_records_by_type(self, populated_registry):
        """Test listing records filtered by type."""
        embedding_records = populated_registry.list_records(plugin_type="embedding")
        assert len(embedding_records) == 3

        chunking_records = populated_registry.list_records(plugin_type="chunking")
        assert len(chunking_records) == 2

        connector_records = populated_registry.list_records(plugin_type="connector")
        assert len(connector_records) == 1

    def test_list_records_by_source(self, populated_registry):
        """Test listing records filtered by source."""
        external = populated_registry.list_records(source=PluginSource.EXTERNAL)
        assert len(external) == 4

        builtin = populated_registry.list_records(source=PluginSource.BUILTIN)
        assert len(builtin) == 2

    def test_list_records_by_type_and_source(self, populated_registry):
        """Test listing records filtered by both type and source."""
        external_embedding = populated_registry.list_records(
            plugin_type="embedding",
            source=PluginSource.EXTERNAL,
        )
        assert len(external_embedding) == 2

        builtin_chunking = populated_registry.list_records(
            plugin_type="chunking",
            source=PluginSource.BUILTIN,
        )
        assert len(builtin_chunking) == 1

    def test_list_ids(self, populated_registry):
        """Test listing plugin IDs."""
        all_ids = populated_registry.list_ids()
        assert len(all_ids) == 6

        embedding_ids = populated_registry.list_ids(plugin_type="embedding")
        assert set(embedding_ids) == {"embed-a", "embed-b", "embed-c"}

    def test_get_all_grouped(self, populated_registry):
        """Test getting all plugins grouped by type."""
        grouped = populated_registry.get_all()

        assert "embedding" in grouped
        assert "chunking" in grouped
        assert "connector" in grouped

        assert len(grouped["embedding"]) == 3
        assert len(grouped["chunking"]) == 2
        assert len(grouped["connector"]) == 1
