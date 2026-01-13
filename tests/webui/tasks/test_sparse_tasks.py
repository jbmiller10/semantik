from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest


def _dummy_engine():
    engine = Mock()
    engine.dispose = AsyncMock()
    return engine


class _DummySession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None


class _DummyCollection:
    def __init__(self, vector_store_name: str) -> None:
        self.vector_store_name = vector_store_name


class _DummyIndexer:
    def __init__(self) -> None:
        self.cleaned_up = False
        self.initialized_with: dict | None = None

    async def initialize(self, cfg):  # type: ignore[no-untyped-def]
        self.initialized_with = dict(cfg or {})

    async def cleanup(self) -> None:
        self.cleaned_up = True

    def get_sparse_collection_name(self, vector_store_name: str) -> str:
        return f"{vector_store_name}_sparse_bm25"

    async def encode_documents(self, documents):  # type: ignore[no-untyped-def]
        from shared.plugins.types.sparse_indexer import SparseVector

        return [
            SparseVector(indices=(1, 2), values=(0.1, 0.2), chunk_id=doc["chunk_id"], metadata=doc.get("metadata", {}))
            for doc in documents
        ]


def test_load_sparse_indexer_plugin_errors() -> None:
    from webui import sparse_tasks

    with (
        patch("webui.sparse_tasks.load_plugins"),
        patch("webui.sparse_tasks.plugin_registry.find_by_id", return_value=None),
    ):
        with pytest.raises(ValueError, match="not found"):
            sparse_tasks._load_sparse_indexer_plugin("missing")

    bad_record = SimpleNamespace(plugin_type="embedding", plugin_class=_DummyIndexer)
    with (
        patch("webui.sparse_tasks.load_plugins"),
        patch("webui.sparse_tasks.plugin_registry.find_by_id", return_value=bad_record),
    ):
        with pytest.raises(ValueError, match="not a sparse indexer"):
            sparse_tasks._load_sparse_indexer_plugin("bad")


@pytest.mark.asyncio()
async def test_reindex_collection_async_local_bm25_happy_path() -> None:
    from webui.sparse_tasks import _reindex_collection_async

    task = Mock()
    task.update_state = Mock()

    qdrant = AsyncMock()
    qdrant.close = AsyncMock()
    qdrant.get_collection = AsyncMock(return_value=SimpleNamespace(points_count=2))

    points = [
        SimpleNamespace(id=1, payload={"content": "hello", "chunk_id": "orig-1", "doc_id": "d1"}),
        SimpleNamespace(id=2, payload={"content": "world", "chunk_id": "orig-2", "doc_id": "d2"}),
    ]
    qdrant.scroll = AsyncMock(side_effect=[(points, None), ([], None)])

    existing_config = {"enabled": True, "document_count": 0, "sparse_collection_name": "ignored"}

    with (
        patch("webui.sparse_tasks._load_sparse_indexer_plugin", return_value=_DummyIndexer()),
        patch("webui.sparse_tasks.create_async_engine", return_value=_dummy_engine()),
        patch("webui.sparse_tasks.async_sessionmaker", return_value=lambda *_a, **_k: _DummySession()),
        patch(
            "webui.sparse_tasks.CollectionRepository",
            return_value=AsyncMock(get_by_uuid=AsyncMock(return_value=_DummyCollection("dense"))),
        ),
        patch("webui.sparse_tasks.AsyncQdrantClient", return_value=qdrant),
        patch("webui.sparse_tasks.ensure_sparse_collection", new=AsyncMock()),
        patch("webui.sparse_tasks.upsert_sparse_vectors", new=AsyncMock()),
        patch("webui.sparse_tasks.get_sparse_index_config", new=AsyncMock(return_value=existing_config)),
        patch("webui.sparse_tasks.store_sparse_index_config", new=AsyncMock(return_value=True)),
    ):
        result = await _reindex_collection_async(task, "col-1", "bm25-local", {"k1": 1.9})

    assert result["status"] == "completed"
    assert result["documents_processed"] == 2
    assert task.update_state.call_count >= 1


@pytest.mark.asyncio()
async def test_reindex_collection_async_vecpipe_path_uses_sparse_client() -> None:
    from webui.sparse_tasks import _reindex_collection_async

    task = Mock()
    task.update_state = Mock()

    qdrant = AsyncMock()
    qdrant.close = AsyncMock()
    qdrant.get_collection = AsyncMock(return_value=SimpleNamespace(points_count=1))
    points = [SimpleNamespace(id="uuid-1", payload={"content": "hello", "chunk_id": "orig-1"})]
    qdrant.scroll = AsyncMock(side_effect=[(points, None), ([], None)])

    vecpipe_client = AsyncMock()
    vecpipe_client.encode_documents = AsyncMock(return_value=[{"chunk_id": "uuid-1", "indices": [1], "values": [0.5]}])

    record = SimpleNamespace(plugin_type="sparse_indexer", plugin_class=SimpleNamespace(SPARSE_TYPE="splade"))

    with (
        patch("webui.sparse_tasks.create_async_engine", return_value=_dummy_engine()),
        patch("webui.sparse_tasks.async_sessionmaker", return_value=lambda *_a, **_k: _DummySession()),
        patch(
            "webui.sparse_tasks.CollectionRepository",
            return_value=AsyncMock(get_by_uuid=AsyncMock(return_value=_DummyCollection("dense"))),
        ),
        patch("webui.sparse_tasks.AsyncQdrantClient", return_value=qdrant),
        patch("webui.sparse_tasks.SparseEncodingClient", return_value=vecpipe_client),
        patch("webui.sparse_tasks.plugin_registry.find_by_id", return_value=record),
        patch("webui.sparse_tasks.ensure_sparse_collection", new=AsyncMock()),
        patch("webui.sparse_tasks.upsert_sparse_vectors", new=AsyncMock()),
        patch("webui.sparse_tasks.get_sparse_index_config", new=AsyncMock(return_value={"enabled": True})),
        patch("webui.sparse_tasks.store_sparse_index_config", new=AsyncMock(return_value=True)),
    ):
        result = await _reindex_collection_async(task, "col-1", "splade-local", {"batch_size": 32})

    assert result["status"] == "completed"
    vecpipe_client.encode_documents.assert_awaited_once()
