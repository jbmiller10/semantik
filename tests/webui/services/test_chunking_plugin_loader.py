#!/usr/bin/env python3
"""Tests for runtime chunking plugin loading."""

from importlib import metadata

from packages.shared.chunking import plugin_loader
from packages.shared.chunking.domain.services.chunking_strategies import STRATEGY_REGISTRY
from packages.webui.services.chunking import strategy_registry


def test_plugin_loader_registers_strategy(monkeypatch):
    """Ensure plugin loader registers strategy, metadata, and factory defaults."""

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
            assert group == plugin_loader.ENTRYPOINT_GROUP
            return [DummyEntryPoint()]

    # Ensure plugins are enabled
    monkeypatch.setenv(plugin_loader.ENV_FLAG, "true")
    # Monkeypatch entry_points to return our dummy plugin
    monkeypatch.setattr(metadata, "entry_points", lambda: DummyEntryPoints())

    # Snapshot existing registries for cleanup
    original_strategies = dict(strategy_registry._STRATEGIES)
    original_factory_defaults = dict(strategy_registry._FACTORY_DEFAULTS)
    original_domain_registry = dict(STRATEGY_REGISTRY)

    strategy_registry._clear_caches()

    try:
        registered = plugin_loader.load_chunking_plugins()

        assert "my_plugin" in registered

        definition = strategy_registry.get_strategy_definition("my_plugin")
        assert definition is not None
        assert definition.is_plugin is True
        assert definition.display_name == "My Plugin"
        assert strategy_registry._FACTORY_DEFAULTS["my_plugin"]["chunk_size"] == 123

        assert "my_plugin" in STRATEGY_REGISTRY
    finally:
        # Restore registries to avoid cross-test pollution
        strategy_registry._STRATEGIES.clear()
        strategy_registry._STRATEGIES.update(original_strategies)

        strategy_registry._FACTORY_DEFAULTS.clear()
        strategy_registry._FACTORY_DEFAULTS.update(original_factory_defaults)

        STRATEGY_REGISTRY.clear()
        STRATEGY_REGISTRY.update(original_domain_registry)

        strategy_registry._clear_caches()
