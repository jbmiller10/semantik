"""Tests for built-in plugin loader registration."""

from __future__ import annotations

from shared.plugins.loader import load_plugins
from shared.plugins.registry import PluginSource, plugin_registry


def setup_function() -> None:
    plugin_registry.reset()


def teardown_function() -> None:
    plugin_registry.reset()


def test_loader_registers_builtin_extractor() -> None:
    registry = load_plugins(plugin_types={"extractor"}, include_external=False)

    record = registry.get("extractor", "keyword-extractor")
    assert record is not None
    assert record.source == PluginSource.BUILTIN
    assert registry.is_loaded({"extractor"})


def test_loader_registers_builtin_reranker() -> None:
    registry = load_plugins(plugin_types={"reranker"}, include_external=False)

    record = registry.get("reranker", "qwen3-reranker")
    assert record is not None
    assert record.source == PluginSource.BUILTIN
    assert registry.is_loaded({"reranker"})
