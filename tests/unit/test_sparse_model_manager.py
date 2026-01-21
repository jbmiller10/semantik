from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest


class DummyOffloader:
    def __init__(self) -> None:
        self._offloaded: set[str] = set()
        self.offload_calls: list[tuple[str, object]] = []
        self.restore_calls: list[str] = []
        self.discard_calls: list[str] = []

    def is_offloaded(self, key: str) -> bool:
        return key in self._offloaded

    def offload_to_cpu(self, key: str, model: object) -> None:
        self._offloaded.add(key)
        self.offload_calls.append((key, model))

    def restore_to_gpu(self, key: str) -> None:
        self._offloaded.discard(key)
        self.restore_calls.append(key)

    def discard(self, key: str) -> None:
        self._offloaded.discard(key)
        self.discard_calls.append(key)


class DummySparsePlugin:
    SPARSE_TYPE = "splade"

    def __init__(self) -> None:
        self._model = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device="cuda:0")]))
        self._actual_device = "cuda:0"
        self.initialized_with: dict | None = None
        self.cleaned_up = False

    async def initialize(self, config):  # type: ignore[no-untyped-def]
        self.initialized_with = dict(config or {})

    async def cleanup(self) -> None:
        self.cleaned_up = True

    async def encode_documents(self, documents):  # type: ignore[no-untyped-def]
        from shared.plugins.types.sparse_indexer import SparseVector

        return [
            SparseVector(indices=(1,), values=(0.1,), chunk_id=doc["chunk_id"], metadata=doc.get("metadata", {}))
            for doc in documents
        ]

    async def encode_query(self, _query: str):  # type: ignore[no-untyped-def]
        from shared.plugins.types.sparse_indexer import SparseQueryVector

        return SparseQueryVector(indices=(1,), values=(0.1,))


def _mock_registry_record(plugin_class=DummySparsePlugin):
    return SimpleNamespace(plugin_type="sparse_indexer", plugin_class=plugin_class)


def test_estimate_splade_memory_varies_by_quantization_and_batch() -> None:
    from vecpipe.sparse_model_manager import _estimate_splade_memory

    default_mb = _estimate_splade_memory({})
    float32_mb = _estimate_splade_memory({"quantization": "float32"})
    int8_mb = _estimate_splade_memory({"quantization": "int8"})
    bigger_batch_mb = _estimate_splade_memory({"batch_size": 64})

    assert float32_mb > default_mb
    assert int8_mb < default_mb
    assert bigger_batch_mb > default_mb


def test_config_hash_is_stable() -> None:
    from vecpipe.sparse_model_manager import _config_hash

    assert _config_hash(None) == "default"
    assert _config_hash({"a": 1, "b": 2}) == _config_hash({"b": 2, "a": 1})


@pytest.mark.asyncio()
async def test_get_or_load_plugin_requests_memory_and_marks_loaded(monkeypatch) -> None:
    from vecpipe.sparse_model_manager import SparseModelManager

    offloader = DummyOffloader()
    monkeypatch.setattr("vecpipe.sparse_model_manager.get_offloader", lambda: offloader)
    monkeypatch.setattr("vecpipe.sparse_model_manager.load_plugins", lambda *_a, **_k: None)
    monkeypatch.setattr("vecpipe.sparse_model_manager.plugin_registry.find_by_id", lambda _pid: _mock_registry_record())

    governor = Mock()
    governor.register_callbacks = Mock()
    governor.request_memory = AsyncMock(return_value=True)
    governor.mark_loaded = AsyncMock()
    governor.mark_unloaded = AsyncMock()
    governor.touch = AsyncMock()
    governor.get_memory_stats = Mock(return_value={"free_mb": 1})

    mgr = SparseModelManager(governor=governor)
    plugin = await mgr.get_or_load_plugin("splade-local", {"batch_size": 32})
    assert isinstance(plugin, DummySparsePlugin)
    assert plugin.initialized_with == {"batch_size": 32}
    assert governor.request_memory.await_count == 1
    assert governor.mark_loaded.await_count == 1

    # Second call hits cache and touches governor.
    plugin2 = await mgr.get_or_load_plugin("splade-local", {"batch_size": 32})
    assert plugin2 is plugin
    assert governor.touch.await_count == 1


@pytest.mark.asyncio()
async def test_get_or_load_plugin_denied_by_governor_raises(monkeypatch) -> None:
    from vecpipe.sparse_model_manager import SparseModelManager

    monkeypatch.setattr("vecpipe.sparse_model_manager.get_offloader", lambda: DummyOffloader())
    monkeypatch.setattr("vecpipe.sparse_model_manager.load_plugins", lambda *_a, **_k: None)
    monkeypatch.setattr("vecpipe.sparse_model_manager.plugin_registry.find_by_id", lambda _pid: _mock_registry_record())

    governor = Mock()
    governor.register_callbacks = Mock()
    governor.request_memory = AsyncMock(return_value=False)
    governor.get_memory_stats = Mock(return_value={"free_mb": 0})

    mgr = SparseModelManager(governor=governor)
    with pytest.raises(RuntimeError, match="Cannot allocate memory"):
        await mgr.get_or_load_plugin("splade-local", {"batch_size": 64})


@pytest.mark.asyncio()
async def test_get_or_load_plugin_restores_offloaded_model(monkeypatch) -> None:
    from vecpipe.sparse_model_manager import SparseModelManager

    offloader = DummyOffloader()
    monkeypatch.setattr("vecpipe.sparse_model_manager.get_offloader", lambda: offloader)
    monkeypatch.setattr("vecpipe.sparse_model_manager.load_plugins", lambda *_a, **_k: None)
    monkeypatch.setattr("vecpipe.sparse_model_manager.plugin_registry.find_by_id", lambda _pid: _mock_registry_record())

    governor = Mock()
    governor.register_callbacks = Mock()
    governor.request_memory = AsyncMock(return_value=True)
    governor.mark_loaded = AsyncMock()
    governor.mark_unloaded = AsyncMock()
    governor.touch = AsyncMock()
    governor.get_memory_stats = Mock(return_value={"free_mb": 1})

    mgr = SparseModelManager(governor=governor)
    plugin = await mgr.get_or_load_plugin("splade-local", {})
    plugin_key = mgr._get_plugin_key("splade-local", {})
    offloader._offloaded.add(f"sparse:{plugin_key}")

    plugin2 = await mgr.get_or_load_plugin("splade-local", {})
    assert plugin2 is plugin
    assert offloader.restore_calls == [f"sparse:{plugin_key}"]


@pytest.mark.asyncio()
async def test_encode_documents_validates_lengths(monkeypatch) -> None:
    from vecpipe.sparse_model_manager import SparseModelManager

    monkeypatch.setattr("vecpipe.sparse_model_manager.get_offloader", lambda: DummyOffloader())
    mgr = SparseModelManager(governor=None)

    with pytest.raises(ValueError, match="must have same length"):
        await mgr.encode_documents("splade-local", ["a"], ["c1", "c2"])

    assert await mgr.encode_documents("splade-local", [], []) == []


@pytest.mark.asyncio()
async def test_governor_callbacks_offload_and_unload(monkeypatch) -> None:
    from vecpipe.sparse_model_manager import SparseModelManager

    offloader = DummyOffloader()
    monkeypatch.setattr("vecpipe.sparse_model_manager.get_offloader", lambda: offloader)

    mgr = SparseModelManager(governor=None)
    plugin = DummySparsePlugin()
    mgr._loaded_plugins["splade-local:default"] = plugin
    mgr._plugin_configs["splade-local:default"] = {}

    await mgr._governor_offload_sparse("splade-local:default", "float16", "cpu")
    assert offloader.offload_calls
    assert plugin._actual_device == "cpu"

    # Pretend it's still offloaded, then restore.
    offloader._offloaded.add("sparse:splade-local:default")
    await mgr._governor_offload_sparse("splade-local:default", "float16", "cuda")
    assert offloader.restore_calls

    await mgr._governor_unload_sparse("splade-local:default", "float16")
    assert offloader.discard_calls == ["sparse:splade-local:default"]
