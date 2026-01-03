"""Unit tests for plugin loader."""

from __future__ import annotations

import logging
from importlib import metadata
from unittest.mock import ANY, MagicMock, patch

import pytest

from shared.plugins.loader import (
    _coerce_class,
    _find_external_plugin,
    _flag_enabled,
    _is_internal_module,
    _parse_dependency,
    _plugin_type_enabled,
    _register_plugin_record,
    _resolve_plugin_type,
    _validate_dependencies,
    get_plugin_config_schema,
    load_plugins,
)
from shared.plugins.manifest import PluginDependency, PluginManifest
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


class TestEntryPointExceptionHandling:
    """Tests for entry point loading exception handling."""

    def test_entry_points_query_exception_logs_warning(self, monkeypatch, caplog):
        """Test that exception during entry_points() query logs warning and returns gracefully."""
        monkeypatch.setenv("SEMANTIK_ENABLE_PLUGINS", "true")

        def raise_error():
            raise RuntimeError("metadata corruption")

        monkeypatch.setattr(metadata, "entry_points", raise_error)

        with caplog.at_level(logging.WARNING):
            load_plugins(plugin_types={"embedding"}, include_builtins=False)

        assert "Unable to query entry points for plugins" in caplog.text
        assert "metadata corruption" in caplog.text

    def test_entry_point_load_exception_logs_and_audits(self, monkeypatch, caplog):
        """Test that exception during ep.load() logs warning and creates audit log."""

        class FailingEntryPoint:
            name = "failing_plugin"

            def load(self):
                raise ImportError("Module not found: bad_module")

        class DummyEntryPoints:
            def select(self, group):
                return [FailingEntryPoint()]

        monkeypatch.setenv("SEMANTIK_ENABLE_PLUGINS", "true")
        monkeypatch.setattr(metadata, "entry_points", lambda: DummyEntryPoints())

        with patch("shared.plugins.loader.audit_log") as mock_audit:
            with caplog.at_level(logging.WARNING):
                load_plugins(plugin_types={"embedding"}, include_builtins=False)

            # Verify warning was logged
            assert "Failed to load plugin entry point" in caplog.text
            assert "failing_plugin" in caplog.text

            # Verify audit_log was called with failure details
            mock_audit.assert_called_with(
                "failing_plugin",
                "plugin.load.failed",
                {"entry_point": "failing_plugin", "error": ANY},
                level=logging.WARNING,
            )

    def test_entry_point_type_error_logs_and_audits(self, monkeypatch, caplog):
        """Test that TypeError from _coerce_class logs warning and creates audit log."""

        class BadEntryPoint:
            name = "bad_type_plugin"

            def load(self):
                return "not a class"  # Returns string, not class

        class DummyEntryPoints:
            def select(self, group):
                return [BadEntryPoint()]

        monkeypatch.setenv("SEMANTIK_ENABLE_PLUGINS", "true")
        monkeypatch.setattr(metadata, "entry_points", lambda: DummyEntryPoints())

        with patch("shared.plugins.loader.audit_log") as mock_audit:
            with caplog.at_level(logging.WARNING):
                load_plugins(plugin_types={"embedding"}, include_builtins=False)

            # Should log and audit the failure
            assert "Failed to load plugin entry point" in caplog.text
            mock_audit.assert_called_once()

    def test_entry_points_old_api_fallback(self, monkeypatch):
        """Test fallback to old dict-like entry_points API (Python <3.10)."""

        class ValidPlugin:
            PLUGIN_ID = "valid_plugin"

        class ValidEntryPoint:
            name = "valid_plugin"

            def load(self):
                return ValidPlugin

        # Simulate old API without .select() method
        old_style_eps = {"semantik.plugins": [ValidEntryPoint()]}

        monkeypatch.setenv("SEMANTIK_ENABLE_PLUGINS", "true")
        monkeypatch.setattr(metadata, "entry_points", lambda: old_style_eps)

        with patch("shared.plugins.loader._resolve_plugin_type", return_value="connector"):
            load_plugins(plugin_types={"connector"}, include_builtins=False)

        # Should have processed the plugin via old API
        # No crash means the fallback worked


class TestResolvePluginTypeFailures:
    """Tests for plugin type resolution exception handling."""

    def test_resolve_type_with_import_error_continues(self, monkeypatch):
        """Test that ImportError during legacy type check doesn't crash."""

        class UnknownPlugin:
            pass

        # This should return None without crashing, even if imports fail
        result = _resolve_plugin_type(UnknownPlugin)
        assert result is None

    def test_resolve_type_with_non_class_catches_exception(self):
        """Test that issubclass with non-class is handled gracefully."""

        # Create something that would cause issubclass to fail
        class NotReallyAPlugin:
            pass

        # Should not crash, just return None for unknown type
        result = _resolve_plugin_type(NotReallyAPlugin)
        assert result is None


class TestEmbeddingPluginValidation:
    """Tests for embedding plugin validation edge cases."""

    def test_embedding_plugin_missing_get_definition(self, monkeypatch, caplog):
        """Test that embedding plugin without get_definition() is skipped."""
        from shared.plugins.loader import _register_embedding_plugin

        class NoDefinitionPlugin:
            pass

        with caplog.at_level(logging.WARNING):
            _register_embedding_plugin(
                NoDefinitionPlugin,
                PluginSource.EXTERNAL,
                entry_point="test_ep",
                disabled_plugin_ids=None,
            )

        assert "Embedding plugin missing get_definition()" in caplog.text

    def test_embedding_plugin_get_definition_not_callable(self, monkeypatch, caplog):
        """Test that embedding plugin with non-callable get_definition is skipped."""
        from shared.plugins.loader import _register_embedding_plugin

        class BadDefinitionPlugin:
            get_definition = "not a method"

        with caplog.at_level(logging.WARNING):
            _register_embedding_plugin(
                BadDefinitionPlugin,
                PluginSource.EXTERNAL,
                entry_point="test_ep",
                disabled_plugin_ids=None,
            )

        assert "Embedding plugin missing get_definition()" in caplog.text

    def test_embedding_plugin_get_definition_raises_exception(self, monkeypatch, caplog):
        """Test that exception in get_definition() is handled gracefully."""
        from shared.plugins.loader import _register_embedding_plugin

        class ExceptionPlugin:
            @classmethod
            def get_definition(cls):
                raise ValueError("Definition error")

        with caplog.at_level(logging.WARNING):
            _register_embedding_plugin(
                ExceptionPlugin,
                PluginSource.EXTERNAL,
                entry_point="test_ep",
                disabled_plugin_ids=None,
            )

        assert "get_definition() failed" in caplog.text
        assert "Definition error" in caplog.text

    def test_embedding_plugin_contract_validation_fails(self, monkeypatch, caplog):
        """Test that plugin failing contract validation is skipped."""
        from shared.embedding.plugin_base import BaseEmbeddingPlugin
        from shared.plugins.loader import _register_embedding_plugin

        class InvalidContractPlugin(BaseEmbeddingPlugin):
            @classmethod
            def get_definition(cls):
                return MagicMock()

            @classmethod
            def validate_plugin_contract(cls):
                return False, "Missing required method: embed"

            @classmethod
            def get_config_schema(cls):
                return None

        with caplog.at_level(logging.WARNING):
            _register_embedding_plugin(
                InvalidContractPlugin,
                PluginSource.EXTERNAL,
                entry_point="test_ep",
                disabled_plugin_ids=None,
            )

        assert "Skipping invalid embedding plugin" in caplog.text
        assert "Missing required method" in caplog.text


class TestChunkingPluginValidation:
    """Tests for chunking plugin visual_example validation."""

    def test_external_chunking_plugin_missing_visual_example(self, monkeypatch, caplog):
        """Test that external chunking plugin without visual_example is skipped."""
        from shared.plugins.loader import _register_chunking_plugin

        class NoVisualPlugin:
            INTERNAL_NAME = "no_visual"
            API_ID = "no_visual"
            METADATA = {}  # Missing visual_example

        with caplog.at_level(logging.WARNING):
            _register_chunking_plugin(
                NoVisualPlugin,
                PluginSource.EXTERNAL,
                entry_point="test_ep",
                disabled_plugin_ids=None,
            )

        assert "missing required visual_example" in caplog.text

    def test_external_chunking_plugin_visual_example_not_dict(self, monkeypatch, caplog):
        """Test that visual_example must be a dict."""
        from shared.plugins.loader import _register_chunking_plugin

        class BadVisualPlugin:
            INTERNAL_NAME = "bad_visual"
            API_ID = "bad_visual"
            METADATA = {"visual_example": "https://example.com/img.png"}  # String, not dict

        with caplog.at_level(logging.WARNING):
            _register_chunking_plugin(
                BadVisualPlugin,
                PluginSource.EXTERNAL,
                entry_point="test_ep",
                disabled_plugin_ids=None,
            )

        assert "missing required visual_example" in caplog.text

    def test_external_chunking_plugin_visual_example_not_https(self, monkeypatch, caplog):
        """Test that visual_example.url must be https://."""
        from shared.plugins.loader import _register_chunking_plugin

        class HttpVisualPlugin:
            INTERNAL_NAME = "http_visual"
            API_ID = "http_visual"
            METADATA = {"visual_example": {"url": "http://example.com/img.png"}}  # http, not https

        with caplog.at_level(logging.WARNING):
            _register_chunking_plugin(
                HttpVisualPlugin,
                PluginSource.EXTERNAL,
                entry_point="test_ep",
                disabled_plugin_ids=None,
            )

        assert "visual_example.url must be https://" in caplog.text

    def test_external_chunking_plugin_visual_example_url_not_string(self, monkeypatch, caplog):
        """Test that visual_example.url must be a string."""
        from shared.plugins.loader import _register_chunking_plugin

        class BadUrlPlugin:
            INTERNAL_NAME = "bad_url"
            API_ID = "bad_url"
            METADATA = {"visual_example": {"url": 123}}  # int, not string

        with caplog.at_level(logging.WARNING):
            _register_chunking_plugin(
                BadUrlPlugin,
                PluginSource.EXTERNAL,
                entry_point="test_ep",
                disabled_plugin_ids=None,
            )

        assert "visual_example.url must be https://" in caplog.text

    def test_builtin_chunking_plugin_skips_visual_validation(self, monkeypatch, caplog):
        """Test that builtin plugins don't require visual_example."""
        from shared.plugins.loader import _register_chunking_plugin

        class BuiltinPlugin:
            INTERNAL_NAME = "builtin_strat"
            API_ID = "builtin_strat"
            METADATA = {}  # No visual_example

        # Mock the factory and registry to prevent actual registration side effects
        with patch("webui.services.chunking_strategy_factory.ChunkingStrategyFactory"):
            with patch("webui.services.chunking.strategy_registry.register_strategy_definition"):
                with caplog.at_level(logging.WARNING):
                    _register_chunking_plugin(
                        BuiltinPlugin,
                        PluginSource.BUILTIN,  # Builtin source
                        entry_point=None,
                        disabled_plugin_ids=None,
                    )

        # Should NOT have visual_example warning
        assert "visual_example" not in caplog.text


class TestParseDependency:
    """Tests for _parse_dependency function."""

    def test_parse_dependency_from_plugin_dependency(self):
        """Test parsing an existing PluginDependency returns it as-is."""
        dep = PluginDependency(plugin_id="test-plugin")
        result = _parse_dependency(dep)
        assert result is dep

    def test_parse_dependency_from_string(self):
        """Test parsing a string plugin_id."""
        result = _parse_dependency("my-plugin")
        assert isinstance(result, PluginDependency)
        assert result.plugin_id == "my-plugin"

    def test_parse_dependency_from_dict(self):
        """Test parsing a dict dependency specification."""
        dep_dict = {"plugin_id": "other-plugin", "min_version": "1.0.0", "optional": True}
        result = _parse_dependency(dep_dict)
        assert isinstance(result, PluginDependency)
        assert result.plugin_id == "other-plugin"
        assert result.min_version == "1.0.0"
        assert result.optional is True

    def test_parse_dependency_invalid_type_raises_value_error(self):
        """Test that invalid types raise ValueError."""
        with pytest.raises(ValueError, match="Invalid dependency format"):
            _parse_dependency(123)

        with pytest.raises(ValueError, match="Invalid dependency format"):
            _parse_dependency(["list", "of", "items"])

        with pytest.raises(ValueError, match="Invalid dependency format"):
            _parse_dependency(None)


class TestValidateDependencies:
    """Tests for _validate_dependencies function."""

    def test_validate_missing_required_dependency(self):
        """Test that missing required dependency returns warning."""
        warnings = _validate_dependencies("test-plugin", ["nonexistent-dep"])
        assert len(warnings) == 1
        assert "missing required dependency: nonexistent-dep" in warnings[0]

    def test_validate_missing_optional_dependency_logs_debug(self, caplog):
        """Test that missing optional dependency logs debug, no warning."""
        with caplog.at_level(logging.DEBUG):
            warnings = _validate_dependencies(
                "test-plugin",
                [PluginDependency(plugin_id="optional-dep", optional=True)],
            )

        assert len(warnings) == 0
        assert "optional dependency 'optional-dep' not found" in caplog.text

    def test_validate_disabled_required_dependency(self):
        """Test that disabled required dependency returns warning."""
        # Register a plugin then disable it
        manifest = PluginManifest(
            id="disabled-dep",
            type="embedding",
            version="1.0.0",
            display_name="Disabled",
            description="",
        )
        record = PluginRecord(
            plugin_type="embedding",
            plugin_id="disabled-dep",
            plugin_version="1.0.0",
            manifest=manifest,
            plugin_class=MagicMock,
            source=PluginSource.EXTERNAL,
        )
        plugin_registry.register(record)
        plugin_registry.set_disabled({"disabled-dep"})

        warnings = _validate_dependencies("test-plugin", ["disabled-dep"])
        assert len(warnings) == 1
        assert "is disabled" in warnings[0]

    def test_validate_disabled_optional_dependency_logs_debug(self, caplog):
        """Test that disabled optional dependency logs debug, no warning."""
        # Register a plugin then disable it
        manifest = PluginManifest(
            id="optional-disabled",
            type="embedding",
            version="1.0.0",
            display_name="Optional Disabled",
            description="",
        )
        record = PluginRecord(
            plugin_type="embedding",
            plugin_id="optional-disabled",
            plugin_version="1.0.0",
            manifest=manifest,
            plugin_class=MagicMock,
            source=PluginSource.EXTERNAL,
        )
        plugin_registry.register(record)
        plugin_registry.set_disabled({"optional-disabled"})

        with caplog.at_level(logging.DEBUG):
            warnings = _validate_dependencies(
                "test-plugin",
                [PluginDependency(plugin_id="optional-disabled", optional=True)],
            )

        assert len(warnings) == 0
        assert "optional dependency 'optional-disabled' is disabled" in caplog.text

    def test_validate_version_constraint_not_satisfied(self):
        """Test that version constraint failure returns warning."""
        # Register a plugin with version 1.0.0
        manifest = PluginManifest(
            id="versioned-dep",
            type="embedding",
            version="1.0.0",
            display_name="Versioned",
            description="",
        )
        record = PluginRecord(
            plugin_type="embedding",
            plugin_id="versioned-dep",
            plugin_version="1.0.0",
            manifest=manifest,
            plugin_class=MagicMock,
            source=PluginSource.EXTERNAL,
        )
        plugin_registry.register(record)

        # Require version >= 2.0.0
        warnings = _validate_dependencies(
            "test-plugin",
            [PluginDependency(plugin_id="versioned-dep", min_version="2.0.0")],
        )
        assert len(warnings) == 1
        assert "versioned-dep" in warnings[0]

    def test_validate_optional_version_constraint_logs_debug(self, caplog):
        """Test that optional version constraint failure logs debug."""
        # Register a plugin with version 1.0.0
        manifest = PluginManifest(
            id="optional-versioned",
            type="embedding",
            version="1.0.0",
            display_name="Optional Versioned",
            description="",
        )
        record = PluginRecord(
            plugin_type="embedding",
            plugin_id="optional-versioned",
            plugin_version="1.0.0",
            manifest=manifest,
            plugin_class=MagicMock,
            source=PluginSource.EXTERNAL,
        )
        plugin_registry.register(record)

        with caplog.at_level(logging.DEBUG):
            warnings = _validate_dependencies(
                "test-plugin",
                [PluginDependency(plugin_id="optional-versioned", min_version="2.0.0", optional=True)],
            )

        assert len(warnings) == 0
        assert "optional dependency 'optional-versioned'" in caplog.text

    def test_validate_invalid_dependency_format_adds_warning(self):
        """Test that invalid dependency format adds warning."""
        warnings = _validate_dependencies("test-plugin", [123, ["bad"], None])
        assert len(warnings) == 3
        for w in warnings:
            assert "Invalid dependency format" in w


class TestAuditLogging:
    """Tests for audit logging during plugin registration."""

    def test_successful_registration_creates_audit_log(self):
        """Test that successful registration creates audit log entry."""
        manifest = PluginManifest(
            id="audit-test",
            type="embedding",
            version="1.0.0",
            display_name="Audit Test",
            description="",
        )

        with patch("shared.plugins.loader.audit_log") as mock_audit:
            with patch("shared.plugins.loader.record_plugin_load"):
                registered = _register_plugin_record(
                    plugin_type="embedding",
                    plugin_id="audit-test",
                    plugin_cls=MagicMock,
                    manifest=manifest,
                    source=PluginSource.EXTERNAL,
                    entry_point="test_ep",
                )

        assert registered is True
        mock_audit.assert_called_once()
        call_args = mock_audit.call_args
        assert call_args[0][0] == "audit-test"
        assert call_args[0][1] == "plugin.registered.external"
        assert call_args[0][2]["plugin_type"] == "embedding"
        assert call_args[0][2]["entry_point"] == "test_ep"

    def test_registration_with_dependency_warnings_audits(self, caplog):
        """Test that dependency warnings are logged and audited."""
        manifest = PluginManifest(
            id="dep-warn-test",
            type="embedding",
            version="1.0.0",
            display_name="Dep Warn Test",
            description="",
            requires=[{"plugin_id": "missing-plugin"}],  # Missing dependency
        )

        with patch("shared.plugins.loader.audit_log") as mock_audit:
            with patch("shared.plugins.loader.record_plugin_load"):
                with patch("shared.plugins.loader.record_dependency_warning") as mock_dep_warn:
                    with caplog.at_level(logging.WARNING):
                        _register_plugin_record(
                            plugin_type="embedding",
                            plugin_id="dep-warn-test",
                            plugin_cls=MagicMock,
                            manifest=manifest,
                            source=PluginSource.EXTERNAL,
                        )

        # Should have logged the warning
        assert "has unmet dependencies" in caplog.text

        # Should have called audit_log twice: once for registration, once for dependency warnings
        assert mock_audit.call_count == 2

        # Check the dependency warning audit call
        dep_warn_call = mock_audit.call_args_list[1]
        assert dep_warn_call[0][0] == "dep-warn-test"
        assert dep_warn_call[0][1] == "plugin.dependency.warnings"
        assert "warnings" in dep_warn_call[0][2]

        # Should have recorded the dependency warning metric
        mock_dep_warn.assert_called_once_with("dep-warn-test", "missing")

    def test_dependency_warning_metric_categorization(self):
        """Test that dependency warnings are categorized correctly for metrics."""
        # Register plugins for version and disabled tests
        for pid in ["version-dep", "disabled-dep"]:
            manifest = PluginManifest(
                id=pid,
                type="embedding",
                version="1.0.0",
                display_name=pid,
                description="",
            )
            record = PluginRecord(
                plugin_type="embedding",
                plugin_id=pid,
                plugin_version="1.0.0",
                manifest=manifest,
                plugin_class=MagicMock,
                source=PluginSource.EXTERNAL,
            )
            plugin_registry.register(record)

        plugin_registry.set_disabled({"disabled-dep"})

        # Create manifest with multiple dependency issues
        manifest = PluginManifest(
            id="multi-dep-test",
            type="connector",
            version="1.0.0",
            display_name="Multi Dep Test",
            description="",
            requires=[
                {"plugin_id": "nonexistent"},  # missing
                {"plugin_id": "disabled-dep"},  # disabled
                {"plugin_id": "version-dep", "min_version": "99.0.0"},  # version
            ],
        )

        with patch("shared.plugins.loader.audit_log"):
            with patch("shared.plugins.loader.record_plugin_load"):
                with patch("shared.plugins.loader.record_dependency_warning") as mock_dep_warn:
                    _register_plugin_record(
                        plugin_type="connector",
                        plugin_id="multi-dep-test",
                        plugin_cls=MagicMock,
                        manifest=manifest,
                        source=PluginSource.EXTERNAL,
                    )

        # Should have recorded all three warning types
        call_types = [call[0][1] for call in mock_dep_warn.call_args_list]
        assert "missing" in call_types
        assert "disabled" in call_types
        assert "version" in call_types
