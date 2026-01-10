"""Tests for built-in plugin loader registration."""

from __future__ import annotations

import importlib.util

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


def test_loader_registers_builtin_sparse_indexer_bm25() -> None:
    registry = load_plugins(plugin_types={"sparse_indexer"}, include_external=False)

    record = registry.get("sparse_indexer", "bm25-local")
    assert record is not None
    assert record.source == PluginSource.BUILTIN
    assert registry.is_loaded({"sparse_indexer"})


def test_loader_registers_builtin_sparse_indexer_splade_when_available() -> None:
    has_deps = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None

    registry = load_plugins(plugin_types={"sparse_indexer"}, include_external=False)

    record = registry.get("sparse_indexer", "splade-local")
    if has_deps:
        assert record is not None
        assert record.source == PluginSource.BUILTIN
    else:
        assert record is None
