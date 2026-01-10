from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def test_ensure_metadata_collection_creates_when_missing() -> None:
    from shared.database.collection_metadata import METADATA_COLLECTION, ensure_metadata_collection

    qdrant = MagicMock()
    qdrant.get_collections.return_value = SimpleNamespace(collections=[SimpleNamespace(name="other")])

    ensure_metadata_collection(qdrant)

    qdrant.create_collection.assert_called_once()
    assert qdrant.create_collection.call_args.kwargs["collection_name"] == METADATA_COLLECTION


def test_ensure_metadata_collection_noop_when_present() -> None:
    from shared.database.collection_metadata import METADATA_COLLECTION, ensure_metadata_collection

    qdrant = MagicMock()
    qdrant.get_collections.return_value = SimpleNamespace(collections=[SimpleNamespace(name=METADATA_COLLECTION)])

    ensure_metadata_collection(qdrant)

    qdrant.create_collection.assert_not_called()


def test_store_collection_metadata_can_skip_ensure() -> None:
    from shared.database.collection_metadata import store_collection_metadata

    qdrant = MagicMock()
    qdrant.upsert = MagicMock()

    store_collection_metadata(
        qdrant=qdrant,
        collection_name="dense_collection",
        model_name="test-model",
        quantization="float16",
        vector_dim=1024,
        chunk_size=1000,
        chunk_overlap=200,
        instruction=None,
        ensure=False,
    )

    qdrant.upsert.assert_called_once()
    point = qdrant.upsert.call_args.kwargs["points"][0]
    assert point.payload["collection_name"] == "dense_collection"
    assert point.payload["id"] == "dense_collection"


def test_get_collection_metadata_scrolls_by_payload_field() -> None:
    from shared.database.collection_metadata import get_collection_metadata

    qdrant = MagicMock()
    qdrant.scroll.return_value = ([SimpleNamespace(payload={"collection_name": "dense", "vector_dim": 10})], None)

    result = get_collection_metadata(qdrant, "dense")

    assert result == {"collection_name": "dense", "vector_dim": 10}
    qdrant.scroll.assert_called_once()


@pytest.mark.asyncio()
async def test_store_sparse_index_config_returns_false_when_missing_metadata_point() -> None:
    from shared.database.collection_metadata import store_sparse_index_config

    qdrant = AsyncMock()
    qdrant.scroll = AsyncMock(return_value=([], None))
    qdrant.upsert = AsyncMock()

    ok = await store_sparse_index_config(qdrant, "dense", {"enabled": True})

    assert ok is False
    qdrant.upsert.assert_not_awaited()


@pytest.mark.asyncio()
async def test_store_sparse_index_config_updates_existing_point_payload() -> None:
    from shared.database.collection_metadata import store_sparse_index_config

    point = SimpleNamespace(id="point-1", payload={"collection_name": "dense", "vector_dim": 10})
    qdrant = AsyncMock()
    qdrant.scroll = AsyncMock(return_value=([point], None))
    qdrant.upsert = AsyncMock()

    sparse_cfg = {"enabled": True, "plugin_id": "bm25-local"}
    ok = await store_sparse_index_config(qdrant, "dense", sparse_cfg)

    assert ok is True
    qdrant.upsert.assert_awaited_once()
    upsert_point = qdrant.upsert.await_args.kwargs["points"][0]
    assert upsert_point.id == "point-1"
    assert upsert_point.payload["sparse_index_config"] == sparse_cfg


@pytest.mark.asyncio()
async def test_delete_sparse_index_config_returns_false_when_missing_metadata_point() -> None:
    from shared.database.collection_metadata import delete_sparse_index_config

    qdrant = AsyncMock()
    qdrant.scroll = AsyncMock(return_value=([], None))
    qdrant.upsert = AsyncMock()

    ok = await delete_sparse_index_config(qdrant, "dense")

    assert ok is False
    qdrant.upsert.assert_not_awaited()


@pytest.mark.asyncio()
async def test_delete_sparse_index_config_returns_true_when_absent() -> None:
    from shared.database.collection_metadata import delete_sparse_index_config

    point = SimpleNamespace(id="point-1", payload={"collection_name": "dense"})
    qdrant = AsyncMock()
    qdrant.scroll = AsyncMock(return_value=([point], None))
    qdrant.upsert = AsyncMock()

    ok = await delete_sparse_index_config(qdrant, "dense")

    assert ok is True
    qdrant.upsert.assert_not_awaited()


@pytest.mark.asyncio()
async def test_delete_sparse_index_config_removes_field_and_upserts() -> None:
    from shared.database.collection_metadata import delete_sparse_index_config

    point = SimpleNamespace(id="point-1", payload={"collection_name": "dense", "sparse_index_config": {"enabled": True}})
    qdrant = AsyncMock()
    qdrant.scroll = AsyncMock(return_value=([point], None))
    qdrant.upsert = AsyncMock()

    ok = await delete_sparse_index_config(qdrant, "dense")

    assert ok is True
    qdrant.upsert.assert_awaited_once()
    upsert_point = qdrant.upsert.await_args.kwargs["points"][0]
    assert upsert_point.id == "point-1"
    assert "sparse_index_config" not in upsert_point.payload


@pytest.mark.asyncio()
async def test_update_sparse_index_stats_returns_false_when_not_enabled(monkeypatch) -> None:
    from shared.database import collection_metadata as module

    qdrant = AsyncMock()
    monkeypatch.setattr(module, "get_sparse_index_config", AsyncMock(return_value=None))
    monkeypatch.setattr(module, "store_sparse_index_config", AsyncMock(return_value=True))
    ok = await module.update_sparse_index_stats(qdrant, "dense", document_count=3, last_indexed_at="2026-01-01")

    assert ok is False


@pytest.mark.asyncio()
async def test_update_sparse_index_stats_updates_and_stores(monkeypatch) -> None:
    from shared.database import collection_metadata as module

    qdrant = AsyncMock()
    sparse_cfg = {"enabled": True, "plugin_id": "bm25-local", "document_count": 0, "last_indexed_at": None}
    store = AsyncMock(return_value=True)

    monkeypatch.setattr(module, "get_sparse_index_config", AsyncMock(return_value=sparse_cfg))
    monkeypatch.setattr(module, "store_sparse_index_config", store)
    ok = await module.update_sparse_index_stats(qdrant, "dense", document_count=7, last_indexed_at="2026-01-01")

    assert ok is True
    stored_cfg = store.await_args.args[2]
    assert stored_cfg["document_count"] == 7
    assert stored_cfg["last_indexed_at"] == "2026-01-01"
