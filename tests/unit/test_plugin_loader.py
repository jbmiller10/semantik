"""Unit tests for plugin loader."""

from __future__ import annotations

from importlib import metadata
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from shared.plugins.loader import (
    ENTRYPOINT_GROUP,
    _coerce_class,
    _find_external_plugin,
    _flag_enabled,
    _is_internal_module,
    _plugin_type_enabled,
    _resolve_plugin_type,
    get_plugin_config_schema,
    load_plugins,
)
from shared.plugins.manifest import PluginManifest
from shared.plugins.registry import PluginRecord, PluginSource, plugin_registry


@pytest.fixture(autouse=True)
def _clear_registry():
    """Clear plugin registry before and after each test."""
    plugin_registry.reset()
    yield
    plugin_registry.reset()


class TestFlagEnabled:
    """Tests for _flag_enabled function."""

    def test_none_flag_returns_true(self):
        """Test that None flag returns True."""
        assert _flag_enabled(None) is True

    def test_empty_flag_returns_true(self):
        """Test that empty flag returns True."""
        assert _flag_enabled("") is True

    def test_unset_env_uses_default_true(self, monkeypatch):
        """Test that unset env var uses default 'true'."""
        monkeypatch.delenv("TEST_FLAG", raising=False)
        assert _flag_enabled("TEST_FLAG") is True

    def test_true_values(self, monkeypatch):
        """Test various true values."""
        for value in ["true", "True", "TRUE", "1", "yes", "YES"]:
            monkeypatch.setenv("TEST_FLAG", value)
            assert _flag_enabled("TEST_FLAG") is True

    def test_false_values(self, monkeypatch):
        """Test various false values."""
        for value in ["false", "False", "FALSE", "0", "no", "NO", "off", "OFF"]:
            monkeypatch.setenv("TEST_FLAG", value)
            assert _flag_enabled("TEST_FLAG") is False


class TestPluginTypeEnabled:
    """Tests for _plugin_type_enabled function."""

    def test_global_flag_disabled(self, monkeypatch):
        """Test that global flag disabled returns False."""
        monkeypatch.setenv("SEMANTIK_ENABLE_PLUGINS", "false")
        assert _plugin_type_enabled("embedding") is False

    def test_global_enabled_type_disabled(self, monkeypatch):
        """Test that type-specific flag can disable."""
        monkeypatch.setenv("SEMANTIK_ENABLE_PLUGINS", "true")
        monkeypatch.setenv("SEMANTIK_ENABLE_EMBEDDING_PLUGINS", "false")
        assert _plugin_type_enabled("embedding") is False

    def test_both_enabled(self, monkeypatch):
        """Test that both enabled returns True."""
        monkeypatch.setenv("SEMANTIK_ENABLE_PLUGINS", "true")
        monkeypatch.setenv("SEMANTIK_ENABLE_CHUNKING_PLUGINS", "true")
        assert _plugin_type_enabled("chunking") is True

    def test_unknown_type_uses_global(self, monkeypatch):
        """Test that unknown type just uses global flag."""
        monkeypatch.setenv("SEMANTIK_ENABLE_PLUGINS", "true")
        assert _plugin_type_enabled("unknown_type") is True


class TestCoerceClass:
    """Tests for _coerce_class function."""

    def test_class_returns_class(self):
        """Test that a class is returned as-is."""

        class TestClass:
            pass

        result = _coerce_class(TestClass)
        assert result is TestClass

    def test_callable_returning_class(self):
        """Test callable that returns a class."""

        class TestClass:
            pass

        def factory():
            return TestClass

        result = _coerce_class(factory)
        assert result is TestClass

    def test_callable_returning_instance(self):
        """Test callable that returns an instance."""

        class TestClass:
            pass

        def factory():
            return TestClass()

        result = _coerce_class(factory)
        assert result is TestClass

    def test_non_callable_returns_none(self):
        """Test that non-callable returns None."""
        result = _coerce_class("not a class")
        assert result is None

    def test_callable_returning_none(self):
        """Test callable returning None returns None."""

        def factory():
            return None

        result = _coerce_class(factory)
        assert result is None


class TestIsInternalModule:
    """Tests for _is_internal_module function."""

    def test_shared_module(self):
        """Test shared module is internal."""
        assert _is_internal_module("shared.plugins.loader") is True

    def test_webui_module(self):
        """Test webui module is internal."""
        assert _is_internal_module("webui.services.plugin_service") is True

    def test_vecpipe_module(self):
        """Test vecpipe module is internal."""
        assert _is_internal_module("vecpipe.worker") is True

    def test_external_module(self):
        """Test external module is not internal."""
        assert _is_internal_module("my_plugin.connector") is False
        assert _is_internal_module("third_party.embedding") is False


class TestResolvePluginType:
    """Tests for _resolve_plugin_type function."""

    def test_embedding_plugin(self):
        """Test resolving embedding plugin type."""
        from shared.embedding.plugin_base import BaseEmbeddingPlugin

        class TestEmbed(BaseEmbeddingPlugin):
            @classmethod
            def get_definition(cls):
                return MagicMock()

            @classmethod
            def get_config_schema(cls):
                return None

        result = _resolve_plugin_type(TestEmbed)
        assert result == "embedding"

    def test_connector_plugin(self):
        """Test resolving connector plugin type."""
        from shared.connectors.base import BaseConnector

        class TestConnector(BaseConnector):
            async def authenticate(self):
                return True

            async def load_documents(self):
                return
                yield

        result = _resolve_plugin_type(TestConnector)
        assert result == "connector"

    def test_unknown_type(self):
        """Test resolving unknown plugin type."""

        class RandomClass:
            pass

        result = _resolve_plugin_type(RandomClass)
        assert result is None


class TestLoadPlugins:
    """Tests for load_plugins function."""

    def test_load_plugins_returns_registry(self, monkeypatch):
        """Test load_plugins returns the plugin registry."""
        monkeypatch.setenv("SEMANTIK_ENABLE_PLUGINS", "false")
        result = load_plugins(plugin_types={"embedding"}, include_external=False, include_builtins=False)
        assert result is plugin_registry

    def test_load_plugins_sets_disabled_ids(self, monkeypatch):
        """Test load_plugins sets disabled IDs."""
        monkeypatch.setenv("SEMANTIK_ENABLE_PLUGINS", "false")
        disabled = {"plugin1", "plugin2"}
        load_plugins(
            plugin_types={"embedding"},
            include_external=False,
            include_builtins=False,
            disabled_plugin_ids=disabled,
        )
        assert plugin_registry.disabled_ids() == disabled

    def test_load_plugins_idempotent(self, monkeypatch):
        """Test load_plugins is idempotent for same types."""
        monkeypatch.setenv("SEMANTIK_ENABLE_PLUGINS", "false")

        load_plugins(plugin_types={"embedding"}, include_external=False, include_builtins=False)
        load_plugins(plugin_types={"embedding"}, include_external=False, include_builtins=False)

        # Should have marked as loaded
        assert plugin_registry.is_loaded({"embedding"}) is True

    def test_load_plugins_global_disabled(self, monkeypatch):
        """Test load_plugins skips external when global flag disabled."""
        monkeypatch.setenv("SEMANTIK_ENABLE_PLUGINS", "false")

        class DummyEntryPoints:
            def select(self, group):
                pytest.fail("Should not query entry points when disabled")
                return []

        monkeypatch.setattr(metadata, "entry_points", lambda: DummyEntryPoints())

        load_plugins(plugin_types={"embedding"}, include_builtins=False)


class TestGetPluginConfigSchema:
    """Tests for get_plugin_config_schema function."""

    def test_get_schema_not_found(self):
        """Test getting schema for nonexistent plugin."""
        result = get_plugin_config_schema("nonexistent")
        assert result is None

    def test_get_schema_found(self):
        """Test getting schema for existing plugin."""

        class PluginWithSchema:
            CONFIG_SCHEMA = {"type": "object", "properties": {}}

        manifest = PluginManifest(
            id="schema-plugin",
            type="embedding",
            version="1.0.0",
            display_name="Schema Plugin",
            description="",
        )
        record = PluginRecord(
            plugin_type="embedding",
            plugin_id="schema-plugin",
            plugin_version="1.0.0",
            manifest=manifest,
            plugin_class=PluginWithSchema,
            source=PluginSource.EXTERNAL,
        )
        plugin_registry.register(record)

        result = get_plugin_config_schema("schema-plugin")
        assert result is not None
        assert result["type"] == "object"

    def test_get_schema_builtin_not_found(self):
        """Test getting schema for builtin plugin returns None."""

        class BuiltinPlugin:
            CONFIG_SCHEMA = {"type": "object"}

        manifest = PluginManifest(
            id="builtin-plugin",
            type="embedding",
            version="1.0.0",
            display_name="Builtin",
            description="",
        )
        record = PluginRecord(
            plugin_type="embedding",
            plugin_id="builtin-plugin",
            plugin_version="1.0.0",
            manifest=manifest,
            plugin_class=BuiltinPlugin,
            source=PluginSource.BUILTIN,
        )
        plugin_registry.register(record)

        # _find_external_plugin only finds external plugins
        result = get_plugin_config_schema("builtin-plugin")
        assert result is None


class TestFindExternalPlugin:
    """Tests for _find_external_plugin function."""

    def test_find_external_found(self):
        """Test finding an external plugin."""
        manifest = PluginManifest(
            id="ext-plugin",
            type="embedding",
            version="1.0.0",
            display_name="External",
            description="",
        )
        record = PluginRecord(
            plugin_type="embedding",
            plugin_id="ext-plugin",
            plugin_version="1.0.0",
            manifest=manifest,
            plugin_class=MagicMock,
            source=PluginSource.EXTERNAL,
        )
        plugin_registry.register(record)

        result = _find_external_plugin("ext-plugin")
        assert result is not None
        assert result.plugin_id == "ext-plugin"

    def test_find_external_not_found(self):
        """Test finding nonexistent external plugin."""
        result = _find_external_plugin("nonexistent")
        assert result is None

    def test_find_external_excludes_builtin(self):
        """Test finding excludes builtin plugins."""
        manifest = PluginManifest(
            id="builtin",
            type="embedding",
            version="1.0.0",
            display_name="Builtin",
            description="",
        )
        record = PluginRecord(
            plugin_type="embedding",
            plugin_id="builtin",
            plugin_version="1.0.0",
            manifest=manifest,
            plugin_class=MagicMock,
            source=PluginSource.BUILTIN,
        )
        plugin_registry.register(record)

        result = _find_external_plugin("builtin")
        assert result is None


class TestConnectorPluginRegistration:
    """Tests for connector plugin registration."""

    def test_connector_without_plugin_id_skipped(self, monkeypatch):
        """Test connector without PLUGIN_ID is skipped."""

        class NoIdConnector:
            pass

        class DummyEntryPoint:
            name = "no_id_connector"

            def load(self):
                return NoIdConnector

        class DummyEntryPoints:
            def select(self, group):
                return [DummyEntryPoint()]

        monkeypatch.setenv("SEMANTIK_ENABLE_PLUGINS", "true")
        monkeypatch.setenv("SEMANTIK_ENABLE_CONNECTOR_PLUGINS", "true")
        monkeypatch.setattr(metadata, "entry_points", lambda: DummyEntryPoints())

        # Mock _resolve_plugin_type to return "connector"
        with patch("shared.plugins.loader._resolve_plugin_type", return_value="connector"):
            load_plugins(plugin_types={"connector"}, include_builtins=False)

        # No connector should be registered
        assert len(plugin_registry.list_ids(plugin_type="connector")) == 0


class TestDisabledPluginHandling:
    """Tests for disabled plugin handling."""

    def test_disabled_plugin_registered_but_not_activated(self, monkeypatch):
        """Test disabled plugin is registered in registry but skips activation."""
        from shared.chunking.domain.services.chunking_strategies.base import ChunkingStrategy

        class DisabledStrategy(ChunkingStrategy):
            INTERNAL_NAME = "disabled_plugin"
            API_ID = "disabled_plugin"
            METADATA = {
                "display_name": "Disabled",
                "description": "test",
                "visual_example": {"url": "https://example.com/test.png"},
            }

            def __init__(self):
                super().__init__(self.INTERNAL_NAME)

            def chunk(self, content, config, progress_callback=None):
                return []

            def validate_content(self, content):
                return True, None

            def estimate_chunks(self, content_length, config):
                return 1

        class DummyEntryPoint:
            name = "disabled_plugin"

            def load(self):
                return DisabledStrategy

        class DummyEntryPoints:
            def select(self, group):
                return [DummyEntryPoint()]

        monkeypatch.setenv("SEMANTIK_ENABLE_PLUGINS", "true")
        monkeypatch.setenv("SEMANTIK_ENABLE_CHUNKING_PLUGINS", "true")
        monkeypatch.setattr(metadata, "entry_points", lambda: DummyEntryPoints())

        # Load with plugin disabled
        load_plugins(
            plugin_types={"chunking"},
            include_builtins=False,
            disabled_plugin_ids={"disabled_plugin"},
        )

        # Plugin should be in registry
        assert "disabled_plugin" in plugin_registry.list_ids(plugin_type="chunking")
        # But marked as disabled
        assert plugin_registry.is_disabled("disabled_plugin") is True
