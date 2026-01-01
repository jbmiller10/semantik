#!/usr/bin/env python3
"""Tests for runtime chunking plugin loading."""

from importlib import metadata

from shared.chunking.domain.services.chunking_strategies import (
    STRATEGY_REGISTRY,
    _restore_strategy_registry,
    _snapshot_strategy_registry,
)
from shared.plugins.loader import load_plugins
from shared.plugins.registry import PluginSource, plugin_registry
from webui.services.chunking import strategy_registry


def test_plugin_loader_registers_strategy(monkeypatch):
    """Ensure plugin loader registers strategy, metadata, and factory defaults."""
    plugin_registry.reset()

    class DummyStrategy:
        INTERNAL_NAME = "my_plugin"
        API_ID = "my_plugin"
        METADATA = {
            "display_name": "My Plugin",
            "description": "plugin test",
            "manager_defaults": {"chunk_size": 123},
            "factory_defaults": {"chunk_size": 123, "chunk_overlap": 10},
            "visual_example": {
                "url": "https://example.com/my-plugin.png",
                "caption": "Example chunks rendered by My Plugin",
            },
        }

    class DummyEntryPoint:
        name = "my_plugin"

        def load(self):
            return DummyStrategy

    class DummyEntryPoints:
        def select(self, group):
            assert group == "semantik.plugins"
            return [DummyEntryPoint()]

    monkeypatch.setenv("SEMANTIK_ENABLE_PLUGINS", "true")
    monkeypatch.setenv("SEMANTIK_ENABLE_CHUNKING_PLUGINS", "true")
    monkeypatch.setattr(metadata, "entry_points", lambda: DummyEntryPoints())

    original_strategies, original_factory_defaults = strategy_registry._snapshot_registry_state()
    original_domain_registry = _snapshot_strategy_registry()

    try:
        registry = load_plugins(plugin_types={"chunking"}, include_builtins=False)
        assert "my_plugin" in registry.list_ids(plugin_type="chunking", source=PluginSource.EXTERNAL)

        definition = strategy_registry.get_strategy_definition("my_plugin")
        assert definition is not None
        assert definition.is_plugin is True
        assert definition.display_name == "My Plugin"
        assert strategy_registry._FACTORY_DEFAULTS["my_plugin"]["chunk_size"] == 123

        assert "my_plugin" in STRATEGY_REGISTRY
    finally:
        strategy_registry._restore_registry_state(original_strategies, original_factory_defaults)
        _restore_strategy_registry(original_domain_registry)
        plugin_registry.reset()


def test_plugin_loader_is_idempotent(monkeypatch):
    plugin_registry.reset()
    call_count = {"count": 0}

    class DummyStrategy:
        INTERNAL_NAME = "my_plugin"
        API_ID = "my_plugin"
        METADATA = {
            "display_name": "My Plugin",
            "description": "plugin test",
            "manager_defaults": {"chunk_size": 123},
            "factory_defaults": {"chunk_size": 123, "chunk_overlap": 10},
            "visual_example": {
                "url": "https://example.com/my-plugin.png",
                "caption": "Example chunks rendered by My Plugin",
            },
        }

    class DummyEntryPoint:
        name = "my_plugin"

        def load(self):
            return DummyStrategy

    class DummyEntryPoints:
        def select(self, group):
            assert group == "semantik.plugins"
            call_count["count"] += 1
            return [DummyEntryPoint()]

    monkeypatch.setenv("SEMANTIK_ENABLE_PLUGINS", "true")
    monkeypatch.setenv("SEMANTIK_ENABLE_CHUNKING_PLUGINS", "true")
    monkeypatch.setattr(metadata, "entry_points", lambda: DummyEntryPoints())

    original_strategies, original_factory_defaults = strategy_registry._snapshot_registry_state()
    original_domain_registry = _snapshot_strategy_registry()

    try:
        first_registry = load_plugins(plugin_types={"chunking"}, include_builtins=False)
        second_registry = load_plugins(plugin_types={"chunking"}, include_builtins=False)

        assert "my_plugin" in first_registry.list_ids(plugin_type="chunking", source=PluginSource.EXTERNAL)
        assert "my_plugin" in second_registry.list_ids(plugin_type="chunking", source=PluginSource.EXTERNAL)
        assert call_count["count"] == 1
    finally:
        strategy_registry._restore_registry_state(original_strategies, original_factory_defaults)
        _restore_strategy_registry(original_domain_registry)
        plugin_registry.reset()
