"""Unit tests for PluginRegistry."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

from shared.plugins.manifest import PluginManifest
from shared.plugins.registry import PluginRecord, PluginRegistry, PluginSource


def _make_manifest(plugin_id: str = "test", plugin_type: str = "embedding") -> PluginManifest:
    """Create a test manifest."""
    return PluginManifest(
        id=plugin_id,
        type=plugin_type,
        version="1.0.0",
        display_name="Test",
        description="Test plugin",
    )


def _make_record(
    plugin_id: str = "test",
    plugin_type: str = "embedding",
    source: PluginSource = PluginSource.EXTERNAL,
    plugin_class: type | None = None,
) -> PluginRecord:
    """Create a test record."""
    return PluginRecord(
        plugin_type=plugin_type,
        plugin_id=plugin_id,
        plugin_version="1.0.0",
        manifest=_make_manifest(plugin_id, plugin_type),
        plugin_class=plugin_class or MagicMock,
        source=source,
    )


class TestPluginRegistry:
    """Tests for PluginRegistry."""

    @pytest.fixture
    def registry(self):
        """Create a fresh registry for each test."""
        return PluginRegistry()

    def test_register_new_plugin(self, registry):
        """Test registering a new plugin returns True."""
        record = _make_record("test-plugin", "embedding")
        result = registry.register(record)
        assert result is True
        assert registry.get("embedding", "test-plugin") is not None

    def test_register_duplicate_same_class(self, registry):
        """Test registering same plugin twice returns False."""
        plugin_cls = MagicMock
        record1 = _make_record("test-plugin", "embedding", plugin_class=plugin_cls)
        record2 = _make_record("test-plugin", "embedding", plugin_class=plugin_cls)

        assert registry.register(record1) is True
        assert registry.register(record2) is False

    def test_register_duplicate_different_class(self, registry):
        """Test registering same ID with different class returns False."""
        record1 = _make_record("test-plugin", "embedding", plugin_class=type("ClassA", (), {}))
        record2 = _make_record("test-plugin", "embedding", plugin_class=type("ClassB", (), {}))

        assert registry.register(record1) is True
        assert registry.register(record2) is False

    def test_register_cross_type_conflict(self, registry):
        """Test registering same ID across types returns False."""
        record1 = _make_record("shared-id", "embedding")
        record2 = _make_record("shared-id", "chunking")

        assert registry.register(record1) is True
        assert registry.register(record2) is False

    def test_get_existing_plugin(self, registry):
        """Test getting an existing plugin."""
        record = _make_record("test", "embedding")
        registry.register(record)

        result = registry.get("embedding", "test")
        assert result is not None
        assert result.plugin_id == "test"

    def test_get_nonexistent_plugin(self, registry):
        """Test getting a nonexistent plugin returns None."""
        result = registry.get("embedding", "nonexistent")
        assert result is None

    def test_get_by_type(self, registry):
        """Test getting all plugins of a type."""
        registry.register(_make_record("plugin1", "embedding"))
        registry.register(_make_record("plugin2", "embedding"))
        registry.register(_make_record("plugin3", "chunking"))

        embedding_plugins = registry.get_by_type("embedding")
        assert len(embedding_plugins) == 2
        assert "plugin1" in embedding_plugins
        assert "plugin2" in embedding_plugins

    def test_get_by_type_empty(self, registry):
        """Test getting plugins of empty type returns empty dict."""
        result = registry.get_by_type("nonexistent")
        assert result == {}

    def test_get_all(self, registry):
        """Test getting all plugins grouped by type."""
        registry.register(_make_record("p1", "embedding"))
        registry.register(_make_record("p2", "chunking"))

        all_plugins = registry.get_all()
        assert "embedding" in all_plugins
        assert "chunking" in all_plugins
        assert len(all_plugins["embedding"]) == 1
        assert len(all_plugins["chunking"]) == 1

    def test_list_types(self, registry):
        """Test listing all plugin types."""
        registry.register(_make_record("p1", "embedding"))
        registry.register(_make_record("p2", "chunking"))
        registry.register(_make_record("p3", "connector"))

        types = registry.list_types()
        assert set(types) == {"embedding", "chunking", "connector"}

    def test_list_records_no_filter(self, registry):
        """Test listing all records without filters."""
        registry.register(_make_record("p1", "embedding", PluginSource.BUILTIN))
        registry.register(_make_record("p2", "embedding", PluginSource.EXTERNAL))
        registry.register(_make_record("p3", "chunking", PluginSource.EXTERNAL))

        records = registry.list_records()
        assert len(records) == 3

    def test_list_records_by_type(self, registry):
        """Test listing records filtered by type."""
        registry.register(_make_record("p1", "embedding"))
        registry.register(_make_record("p2", "chunking"))

        records = registry.list_records(plugin_type="embedding")
        assert len(records) == 1
        assert records[0].plugin_id == "p1"

    def test_list_records_by_source(self, registry):
        """Test listing records filtered by source."""
        registry.register(_make_record("p1", "embedding", PluginSource.BUILTIN))
        registry.register(_make_record("p2", "embedding", PluginSource.EXTERNAL))

        external_records = registry.list_records(source=PluginSource.EXTERNAL)
        assert len(external_records) == 1
        assert external_records[0].plugin_id == "p2"

    def test_list_records_by_type_and_source(self, registry):
        """Test listing records filtered by both type and source."""
        registry.register(_make_record("p1", "embedding", PluginSource.BUILTIN))
        registry.register(_make_record("p2", "embedding", PluginSource.EXTERNAL))
        registry.register(_make_record("p3", "chunking", PluginSource.EXTERNAL))

        records = registry.list_records(plugin_type="embedding", source=PluginSource.EXTERNAL)
        assert len(records) == 1
        assert records[0].plugin_id == "p2"

    def test_list_ids(self, registry):
        """Test listing plugin IDs."""
        registry.register(_make_record("p1", "embedding"))
        registry.register(_make_record("p2", "embedding"))

        ids = registry.list_ids(plugin_type="embedding")
        assert set(ids) == {"p1", "p2"}

    def test_find_by_id_found(self, registry):
        """Test finding a plugin by ID across all types."""
        registry.register(_make_record("unique-id", "embedding"))

        result = registry.find_by_id("unique-id")
        assert result is not None
        assert result.plugin_id == "unique-id"

    def test_find_by_id_not_found(self, registry):
        """Test finding a nonexistent plugin returns None."""
        result = registry.find_by_id("nonexistent")
        assert result is None

    def test_is_loaded(self, registry):
        """Test is_loaded checks loaded types."""
        registry.mark_loaded({"embedding", "chunking"})

        assert registry.is_loaded({"embedding"}) is True
        assert registry.is_loaded({"embedding", "chunking"}) is True
        assert registry.is_loaded({"connector"}) is False
        assert registry.is_loaded({"embedding", "connector"}) is False

    def test_mark_loaded(self, registry):
        """Test mark_loaded adds to loaded types."""
        registry.mark_loaded({"embedding"})
        assert registry.loaded_types() == {"embedding"}

        registry.mark_loaded({"chunking"})
        assert registry.loaded_types() == {"embedding", "chunking"}

    def test_set_disabled(self, registry):
        """Test set_disabled sets disabled IDs."""
        registry.set_disabled({"plugin1", "plugin2"})
        assert registry.disabled_ids() == {"plugin1", "plugin2"}

    def test_is_disabled(self, registry):
        """Test is_disabled checks disabled status."""
        registry.set_disabled({"disabled-plugin"})

        assert registry.is_disabled("disabled-plugin") is True
        assert registry.is_disabled("enabled-plugin") is False

    def test_reset(self, registry):
        """Test reset clears all state."""
        registry.register(_make_record("test", "embedding"))
        registry.mark_loaded({"embedding"})
        registry.set_disabled({"test"})

        registry.reset()

        assert registry.get_all() == {}
        assert registry.loaded_types() == set()
        assert registry.disabled_ids() == set()

    def test_thread_safety_register(self, registry):
        """Test concurrent registration is thread-safe."""
        results = []

        def register_plugin(i):
            record = _make_record(f"plugin-{i}", "embedding", plugin_class=type(f"Cls{i}", (), {}))
            result = registry.register(record)
            results.append(result)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(register_plugin, i) for i in range(100)]
            for f in futures:
                f.result()

        # All 100 should have registered
        assert sum(results) == 100
        assert len(registry.get_by_type("embedding")) == 100

    def test_thread_safety_concurrent_ops(self, registry):
        """Test concurrent read/write operations are thread-safe."""
        errors = []

        def writer(i):
            try:
                record = _make_record(f"w-{i}", "embedding", plugin_class=type(f"W{i}", (), {}))
                registry.register(record)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                registry.get_all()
                registry.list_records()
                registry.find_by_id("w-1")
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            for i in range(50):
                futures.append(executor.submit(writer, i))
                futures.append(executor.submit(reader))
            for f in futures:
                f.result()

        assert len(errors) == 0
