from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest


def test_load_sparse_indexer_plugin_loads_and_instantiates() -> None:
    from webui.sparse_tasks import _load_sparse_indexer_plugin

    class DummyPlugin:
        pass

    record = SimpleNamespace(plugin_type="sparse_indexer", plugin_class=DummyPlugin)

    with (
        patch("webui.sparse_tasks.load_plugins") as mock_load,
        patch("webui.sparse_tasks.plugin_registry.find_by_id", return_value=record) as mock_find,
    ):
        plugin = _load_sparse_indexer_plugin("dummy")

    mock_load.assert_called_once_with(plugin_types={"sparse_indexer"})
    mock_find.assert_called_once_with("dummy")
    assert isinstance(plugin, DummyPlugin)


def test_load_sparse_indexer_plugin_raises_for_missing_plugin() -> None:
    from webui.sparse_tasks import _load_sparse_indexer_plugin

    with (
        patch("webui.sparse_tasks.load_plugins"),
        patch("webui.sparse_tasks.plugin_registry.find_by_id", return_value=None),
    ):
        with pytest.raises(ValueError, match="not found"):
            _load_sparse_indexer_plugin("missing")


def test_load_sparse_indexer_plugin_raises_for_wrong_type() -> None:
    from webui.sparse_tasks import _load_sparse_indexer_plugin

    record = SimpleNamespace(plugin_type="embedding", plugin_class=object)

    with (
        patch("webui.sparse_tasks.load_plugins"),
        patch("webui.sparse_tasks.plugin_registry.find_by_id", return_value=record),
    ):
        with pytest.raises(ValueError, match="not a sparse indexer"):
            _load_sparse_indexer_plugin("wrong-type")


@pytest.mark.asyncio()
async def test_reindex_collection_async_returns_completed_when_no_chunks() -> None:
    from webui.sparse_tasks import _reindex_collection_async

    dummy_task = Mock()
    dummy_task.update_state = Mock()

    class DummyIndexer:
        initialize = AsyncMock()
        cleanup = AsyncMock()

    collection = SimpleNamespace(vector_store_name="dense_collection")

    collection_repo = AsyncMock()
    collection_repo.get_by_uuid = AsyncMock(return_value=collection)

    chunk_repo = AsyncMock()
    chunk_repo.get_chunks_paginated = AsyncMock(return_value=([], 0))

    class DummySessionContext:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    with (
        patch("webui.sparse_tasks._load_sparse_indexer_plugin", return_value=DummyIndexer()),
        patch("webui.sparse_tasks.AsyncSessionLocal", return_value=DummySessionContext()),
        patch("webui.sparse_tasks.CollectionRepository", return_value=collection_repo),
        patch("webui.sparse_tasks.ChunkRepository", return_value=chunk_repo),
    ):
        result = await _reindex_collection_async(
            task=dummy_task,
            collection_uuid="col-1",
            plugin_id="bm25-local",
            model_config={},
        )

    assert result == {"status": "completed", "documents_processed": 0, "total_documents": 0}
    DummyIndexer.initialize.assert_awaited_once()
    DummyIndexer.cleanup.assert_awaited_once()


def test_reindex_sparse_collection_task_delegates_to_asyncio_run() -> None:
    from webui.sparse_tasks import reindex_sparse_collection

    sentinel = {"status": "ok"}
    coroutine_sentinel = object()

    with (
        patch("webui.sparse_tasks._reindex_collection_async", new=Mock(return_value=coroutine_sentinel)) as mock_async,
        patch("webui.sparse_tasks.asyncio.run", return_value=sentinel) as mock_run,
    ):
        result = reindex_sparse_collection.run("col-1", "bm25-local", model_config={"k1": 1.2})

    assert result == sentinel
    mock_async.assert_called_once()
    task_arg = mock_async.call_args.kwargs["task"]
    assert hasattr(task_arg, "update_state")
    assert mock_async.call_args.kwargs["collection_uuid"] == "col-1"
    assert mock_async.call_args.kwargs["plugin_id"] == "bm25-local"
    assert mock_async.call_args.kwargs["model_config"] == {"k1": 1.2}
    mock_run.assert_called_once_with(coroutine_sentinel)
