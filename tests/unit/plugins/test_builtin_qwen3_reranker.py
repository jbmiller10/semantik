"""Tests for Qwen3RerankerPlugin."""

from __future__ import annotations

import pytest

from shared.plugins.builtins.qwen3_reranker import Qwen3RerankerPlugin


class StubReranker:
    """Minimal reranker stub for plugin tests."""

    def __init__(self) -> None:
        self.calls = []

    def rerank(self, query: str, documents: list[str], top_k: int, instruction=None, return_scores=True):
        self.calls.append((query, documents, top_k))
        ranked = [(index, float(index)) for index in reversed(range(len(documents)))]
        return ranked[:top_k]

    def unload_model(self) -> None:
        return None


@pytest.mark.asyncio()
async def test_rerank_uses_stubbed_reranker(monkeypatch) -> None:
    plugin = Qwen3RerankerPlugin(config={"device": "cpu"})
    plugin._reranker = StubReranker()

    async def noop() -> None:
        return None

    monkeypatch.setattr(plugin, "_ensure_model_loaded_async", noop)

    documents = ["doc1", "doc2", "doc3"]
    metadata = [{"id": 1}, {"id": 2}, {"id": 3}]
    results = await plugin.rerank("query", documents, top_k=2, metadata=metadata)

    assert len(results) == 2
    assert results[0].document == "doc3"
    assert results[0].metadata == {"id": 3}


@pytest.mark.asyncio()
async def test_rerank_empty_documents() -> None:
    plugin = Qwen3RerankerPlugin()
    results = await plugin.rerank("query", [])
    assert results == []


def test_manifest_and_schema() -> None:
    manifest = Qwen3RerankerPlugin.get_manifest()
    schema = Qwen3RerankerPlugin.get_config_schema()

    assert manifest.id == Qwen3RerankerPlugin.PLUGIN_ID
    assert schema["type"] == "object"
    assert "model_name" in schema["properties"]
