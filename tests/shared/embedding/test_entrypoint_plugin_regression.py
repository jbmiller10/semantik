"""Regression tests for embedding plugin entry-point discovery."""

from __future__ import annotations

from importlib import metadata
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from shared.embedding.factory import EmbeddingProviderFactory
from shared.embedding.provider_registry import get_provider_definition, list_provider_metadata
from shared.plugins.loader import load_plugins

# Import ValidPlugin from existing test module to avoid duplication
from tests.shared.embedding.test_embedding_plugin_loader import ValidPlugin

if TYPE_CHECKING:
    import pytest


class TestEntryPointPluginDiscoveryRegression:
    """Regression tests to guard the entry-point plugin discovery path."""

    def test_plugin_discovered_via_entry_points_and_registered_in_factory(
        self,
        monkeypatch: pytest.MonkeyPatch,
        empty_registry: None,
    ) -> None:
        class DummyEntryPoint:
            name = "regression_test_plugin"

            def load(self) -> type:
                return ValidPlugin

        class DummyEntryPoints:
            def select(self, group: str) -> list:
                assert group == "semantik.plugins"
                return [DummyEntryPoint()]

        monkeypatch.setenv("SEMANTIK_ENABLE_PLUGINS", "true")
        monkeypatch.setenv("SEMANTIK_ENABLE_EMBEDDING_PLUGINS", "true")
        monkeypatch.setattr(metadata, "entry_points", lambda: DummyEntryPoints())

        load_plugins(plugin_types={"embedding"}, include_builtins=False)

        available_providers = EmbeddingProviderFactory.list_available_providers()
        assert ValidPlugin.INTERNAL_NAME in available_providers

        definition = get_provider_definition(ValidPlugin.INTERNAL_NAME)
        assert definition is not None
        assert definition.api_id == ValidPlugin.API_ID
        assert definition.internal_id == ValidPlugin.INTERNAL_NAME
        assert definition.is_plugin is True

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
        entry_points_spy = MagicMock()
        monkeypatch.setattr(metadata, "entry_points", entry_points_spy)

        monkeypatch.setenv("SEMANTIK_ENABLE_PLUGINS", "false")

        load_plugins(plugin_types={"embedding"}, include_builtins=False)

        entry_points_spy.assert_not_called()
        available_providers = EmbeddingProviderFactory.list_available_providers()
        assert ValidPlugin.INTERNAL_NAME not in available_providers

    def test_plugin_loading_disabled_preserves_no_side_effects(
        self,
        monkeypatch: pytest.MonkeyPatch,
        empty_registry: None,
    ) -> None:
        monkeypatch.setenv("SEMANTIK_ENABLE_PLUGINS", "false")

        providers_before = set(EmbeddingProviderFactory.list_available_providers())
        metadata_before = list_provider_metadata()

        load_plugins(plugin_types={"embedding"}, include_builtins=False)

        providers_after = set(EmbeddingProviderFactory.list_available_providers())
        metadata_after = list_provider_metadata()

        assert providers_before == providers_after
        assert metadata_before == metadata_after
