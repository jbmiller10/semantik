"""Regression tests for embedding plugin entry-point discovery.

These tests guard the plugin discovery path that uses Python entry points,
ensuring plugins are correctly discovered and registered through the public API.
"""

from __future__ import annotations

from importlib import metadata
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from shared.embedding import plugin_loader

if TYPE_CHECKING:
    import pytest

from shared.embedding.factory import EmbeddingProviderFactory
from shared.embedding.provider_registry import get_provider_definition, list_provider_metadata

# Import ValidPlugin from existing test module to avoid duplication
from tests.shared.embedding.test_embedding_plugin_loader import ValidPlugin


class TestEntryPointPluginDiscoveryRegression:
    """Regression tests to guard the entry-point plugin discovery path."""

    def test_plugin_discovered_via_entry_points_and_registered_in_factory(
        self,
        monkeypatch: pytest.MonkeyPatch,
        empty_registry: None,
    ) -> None:
        """Regression test: Plugins discovered through entry points are registered.

        This guards against regressions in the plugin discovery path by verifying
        plugins are accessible via the public EmbeddingProviderFactory API.
        """

        class DummyEntryPoint:
            """Fake entry point that returns ValidPlugin."""

            name = "regression_test_plugin"

            def load(self) -> type:
                return ValidPlugin

        class DummyEntryPoints:
            """Fake entry points collection."""

            def select(self, group: str) -> list:
                assert group == plugin_loader.ENTRYPOINT_GROUP
                return [DummyEntryPoint()]

        # Enable plugins via environment variable
        monkeypatch.setenv(plugin_loader.ENV_FLAG, "true")
        monkeypatch.setattr(metadata, "entry_points", lambda: DummyEntryPoints())

        # Reset plugin loader state for fresh discovery
        plugin_loader._reset_plugin_loader_state()

        # Act: Load plugins via entry points
        registered = plugin_loader.load_embedding_plugins()

        # Assert: Plugin was registered
        assert ValidPlugin.INTERNAL_NAME in registered

        # Assert: Plugin accessible via public factory API (not internal state)
        available_providers = EmbeddingProviderFactory.list_available_providers()
        assert ValidPlugin.INTERNAL_NAME in available_providers

        # Assert: Plugin definition registered in provider registry
        definition = get_provider_definition(ValidPlugin.INTERNAL_NAME)
        assert definition is not None
        assert definition.api_id == ValidPlugin.API_ID
        assert definition.internal_id == ValidPlugin.INTERNAL_NAME

        # Assert: Plugin marked as plugin (not built-in)
        assert definition.is_plugin is True

        # Assert: Plugin appears in provider metadata list
        all_metadata = list_provider_metadata()
        plugin_meta = next(
            (m for m in all_metadata if m.get("internal_id") == ValidPlugin.INTERNAL_NAME),
            None,
        )
        assert plugin_meta is not None
        assert plugin_meta.get("is_plugin") is True

    def test_plugin_loading_disabled_when_env_flag_false(
        self,
        monkeypatch: pytest.MonkeyPatch,
        empty_registry: None,
    ) -> None:
        """Regression test: Plugins are not loaded when env flag is false.

        This ensures the SEMANTIK_ENABLE_EMBEDDING_PLUGINS=false branch
        correctly short-circuits plugin loading without querying entry points.
        """
        # Create a spy to verify entry_points is NOT called
        entry_points_spy = MagicMock()
        monkeypatch.setattr(metadata, "entry_points", entry_points_spy)

        # Disable plugins via environment variable
        monkeypatch.setenv(plugin_loader.ENV_FLAG, "false")

        # Reset plugin loader state for fresh check
        plugin_loader._reset_plugin_loader_state()

        # Act: Attempt to load plugins
        registered = plugin_loader.load_embedding_plugins()

        # Assert: No plugins registered
        assert registered == []

        # Assert: entry_points was NOT called (env-flag-off branch)
        entry_points_spy.assert_not_called()

        # Assert: Factory has no providers (empty_registry cleared all)
        available_providers = EmbeddingProviderFactory.list_available_providers()
        assert ValidPlugin.INTERNAL_NAME not in available_providers

    def test_plugin_loading_disabled_preserves_no_side_effects(
        self,
        monkeypatch: pytest.MonkeyPatch,
        empty_registry: None,
    ) -> None:
        """Regression test: Disabling plugins leaves registry unchanged.

        When plugins are disabled, the registry should remain in its
        pre-call state with no side effects.
        """
        # Disable plugins
        monkeypatch.setenv(plugin_loader.ENV_FLAG, "false")
        plugin_loader._reset_plugin_loader_state()

        # Capture state before call
        providers_before = set(EmbeddingProviderFactory.list_available_providers())
        metadata_before = list_provider_metadata()

        # Act: Call load_embedding_plugins (should be no-op)
        plugin_loader.load_embedding_plugins()

        # Assert: State unchanged
        providers_after = set(EmbeddingProviderFactory.list_available_providers())
        metadata_after = list_provider_metadata()

        assert providers_before == providers_after
        assert metadata_before == metadata_after
