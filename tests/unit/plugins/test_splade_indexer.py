from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def _require_torch_and_transformers():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")


def test_get_recommended_batch_size_tiers() -> None:
    _require_torch_and_transformers()
    from shared.plugins.builtins.splade_indexer import get_recommended_batch_size

    assert get_recommended_batch_size(24000) == 128
    assert get_recommended_batch_size(12000) == 64
    assert get_recommended_batch_size(8000) == 32
    assert get_recommended_batch_size(6000) == 16
    assert get_recommended_batch_size(4000) == 8
    assert get_recommended_batch_size(1000) == 4


@pytest.mark.asyncio()
async def test_splade_initialize_encode_documents_and_query_with_top_k(monkeypatch) -> None:
    _require_torch_and_transformers()
    import torch

    from shared.plugins.builtins.splade_indexer import SPLADESparseIndexerPlugin
    from shared.plugins.types.sparse_indexer import SparseQueryVector, SparseVector

    class DummyBatch(dict):
        def to(self, device: str):
            for key, val in list(self.items()):
                self[key] = val.to(device)
            return self

    class DummyTokenizer:
        def __call__(self, texts, **_kwargs):  # type: ignore[no-untyped-def]
            batch = len(texts)
            seq = 3
            input_ids = torch.ones((batch, seq), dtype=torch.long)
            attention_mask = torch.ones((batch, seq), dtype=torch.long)
            return DummyBatch({"input_ids": input_ids, "attention_mask": attention_mask})

    class DummyModel:
        def __init__(self) -> None:
            self._param = torch.nn.Parameter(torch.zeros(()))

        def to(self, _device: str):
            return self

        def eval(self) -> None:
            return None

        def parameters(self):
            return iter([self._param])

        def __call__(self, **encoded):  # type: ignore[no-untyped-def]
            input_ids = encoded["input_ids"]
            batch, seq = input_ids.shape
            vocab = 6
            logits = torch.zeros((batch, seq, vocab), device=input_ids.device)
            # Make token 2 strongest and token 4 second strongest.
            logits[:, :, 2] = 10.0
            logits[:, :, 4] = 5.0
            return SimpleNamespace(logits=logits)

    monkeypatch.setattr("shared.plugins.builtins.splade_indexer.torch.cuda.is_available", lambda: False)
    monkeypatch.setattr(
        "shared.plugins.builtins.splade_indexer.AutoTokenizer.from_pretrained",
        lambda *_args, **_kwargs: DummyTokenizer(),
    )
    monkeypatch.setattr(
        "shared.plugins.builtins.splade_indexer.AutoModelForMaskedLM.from_pretrained",
        lambda *_args, **_kwargs: DummyModel(),
    )

    plugin = SPLADESparseIndexerPlugin(
        {
            "model_name": "dummy",
            "device": "auto",
            "quantization": "float16",
            "batch_size": 2,
            "max_length": 16,
            "top_k_tokens": 1,
        }
    )
    await plugin.initialize()

    docs = [
        {"content": "doc-1", "chunk_id": "c1", "metadata": {"x": 1}},
        {"content": "doc-2", "chunk_id": "c2", "metadata": {"x": 2}},
        {"content": "doc-3", "chunk_id": "c3", "metadata": {"x": 3}},
    ]
    vectors = await plugin.encode_documents(docs)
    assert len(vectors) == 3
    assert all(isinstance(v, SparseVector) for v in vectors)
    assert vectors[0].chunk_id == "c1"
    assert vectors[0].metadata == {"x": 1}
    assert vectors[0].indices == (2,)
    assert len(vectors[0].values) == 1

    qv = await plugin.encode_query("hello")
    assert isinstance(qv, SparseQueryVector)
    assert qv.indices == (2,)
    assert len(qv.values) == 1

    await plugin.cleanup()


@pytest.mark.asyncio()
async def test_splade_encode_documents_oom_recovers_by_splitting(monkeypatch) -> None:
    _require_torch_and_transformers()
    import torch

    from shared.plugins.builtins.splade_indexer import SPLADESparseIndexerPlugin

    plugin = SPLADESparseIndexerPlugin({"batch_size": 4, "device": "cpu"})
    plugin._model = Mock()
    plugin._tokenizer = Mock()
    plugin._actual_device = "cpu"

    empty_cache = Mock()
    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)

    def fake_encode_single_batch(texts):  # type: ignore[no-untyped-def]
        if len(texts) > 1:
            raise RuntimeError("CUDA out of memory")
        return [((1, 2), (0.1, 0.2))]

    monkeypatch.setattr(plugin, "_encode_single_batch", fake_encode_single_batch)
    monkeypatch.setattr(plugin, "_tokenize_batch", lambda _texts: {"attention_mask": torch.ones((1, 1))})
    vectors = await plugin.encode_documents(
        [
            {"content": "a", "chunk_id": "c1", "metadata": {}},
            {"content": "b", "chunk_id": "c2", "metadata": {}},
        ]
    )

    assert len(vectors) == 2
    assert empty_cache.called is True


@pytest.mark.asyncio()
async def test_splade_encode_requires_initialize() -> None:
    _require_torch_and_transformers()
    from shared.plugins.builtins.splade_indexer import SPLADESparseIndexerPlugin

    plugin = SPLADESparseIndexerPlugin()
    with pytest.raises(RuntimeError, match="Call initialize\\(\\) first"):
        await plugin.encode_documents([{"content": "x", "chunk_id": "c1"}])

    with pytest.raises(RuntimeError, match="Call initialize\\(\\) first"):
        await plugin.encode_query("hello")


@pytest.mark.asyncio()
async def test_splade_encode_query_empty_is_empty_vector(monkeypatch) -> None:
    _require_torch_and_transformers()
    import torch

    from shared.plugins.builtins.splade_indexer import SPLADESparseIndexerPlugin

    plugin = SPLADESparseIndexerPlugin({"device": "cpu"})
    plugin._model = Mock()
    plugin._tokenizer = Mock()
    plugin._actual_device = "cpu"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    vec = await plugin.encode_query("   ")
    assert vec.indices == ()
    assert vec.values == ()
