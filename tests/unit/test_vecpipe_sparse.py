from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio()
async def test_ensure_sparse_collection_returns_false_when_exists() -> None:
    from vecpipe.sparse import ensure_sparse_collection

    existing = SimpleNamespace(name="my_sparse")
    mock_client = AsyncMock()
    mock_client.get_collections.return_value = SimpleNamespace(collections=[existing])
    mock_client.create_collection = AsyncMock()

    created = await ensure_sparse_collection("my_sparse", mock_client)

    assert created is False
    mock_client.create_collection.assert_not_awaited()


@pytest.mark.asyncio()
async def test_ensure_sparse_collection_creates_when_missing() -> None:
    from vecpipe.sparse import ensure_sparse_collection

    existing = SimpleNamespace(name="other")
    mock_client = AsyncMock()
    mock_client.get_collections.return_value = SimpleNamespace(collections=[existing])
    mock_client.create_collection = AsyncMock()

    created = await ensure_sparse_collection("my_sparse", mock_client)

    assert created is True
    mock_client.create_collection.assert_awaited_once()
    assert mock_client.create_collection.await_args.kwargs["collection_name"] == "my_sparse"
    assert "sparse_vectors_config" in mock_client.create_collection.await_args.kwargs


@pytest.mark.asyncio()
async def test_upsert_sparse_vectors_noop_for_empty_input() -> None:
    from vecpipe.sparse import upsert_sparse_vectors

    mock_client = AsyncMock()
    mock_client.upsert = AsyncMock()

    count = await upsert_sparse_vectors("sparse_col", [], mock_client)

    assert count == 0
    mock_client.upsert.assert_not_awaited()


@pytest.mark.asyncio()
async def test_upsert_sparse_vectors_includes_chunk_id_in_payload() -> None:
    from vecpipe.sparse import upsert_sparse_vectors

    mock_client = AsyncMock()
    mock_client.upsert = AsyncMock()

    count = await upsert_sparse_vectors(
        sparse_collection_name="sparse_col",
        vectors=[
            {
                "chunk_id": "chunk-1",
                "indices": [1, 3],
                "values": [0.2, 0.4],
                "metadata": {"path": "/a.txt"},
            }
        ],
        qdrant_client=mock_client,
    )

    assert count == 1
    mock_client.upsert.assert_awaited_once()
    points = mock_client.upsert.await_args.kwargs["points"]
    assert len(points) == 1
    assert points[0].payload["chunk_id"] == "chunk-1"
    assert points[0].payload["path"] == "/a.txt"


@pytest.mark.asyncio()
async def test_search_sparse_collection_maps_results() -> None:
    from vecpipe.sparse import search_sparse_collection

    point = SimpleNamespace(id="chunk-1", score=0.42, payload={"chunk_id": "chunk-1", "path": "/a.txt"})
    mock_client = AsyncMock()
    mock_client.search = AsyncMock(return_value=[point])

    results = await search_sparse_collection(
        sparse_collection_name="sparse_col",
        query_indices=[1],
        query_values=[0.1],
        limit=3,
        qdrant_client=mock_client,
    )

    assert results == [{"chunk_id": "chunk-1", "score": 0.42, "payload": point.payload}]


@pytest.mark.asyncio()
async def test_delete_sparse_vectors_noop_for_empty_ids() -> None:
    from vecpipe.sparse import delete_sparse_vectors

    mock_client = AsyncMock()
    mock_client.delete = AsyncMock()

    deleted = await delete_sparse_vectors("sparse_col", [], mock_client)

    assert deleted == 0
    mock_client.delete.assert_not_awaited()


@pytest.mark.asyncio()
async def test_delete_sparse_vectors_passes_point_ids_list() -> None:
    from vecpipe.sparse import delete_sparse_vectors

    mock_client = AsyncMock()
    mock_client.delete = AsyncMock()

    deleted = await delete_sparse_vectors("sparse_col", ["chunk-1", "chunk-2"], mock_client)

    assert deleted == 2
    mock_client.delete.assert_awaited_once()
    selector = mock_client.delete.await_args.kwargs["points_selector"]
    assert list(selector.points) == ["chunk-1", "chunk-2"]


@pytest.mark.asyncio()
async def test_delete_sparse_collection_returns_false_when_missing() -> None:
    from vecpipe.sparse import delete_sparse_collection

    mock_client = AsyncMock()
    mock_client.get_collections.return_value = SimpleNamespace(collections=[SimpleNamespace(name="other")])
    mock_client.delete_collection = AsyncMock()

    deleted = await delete_sparse_collection("sparse_col", mock_client)

    assert deleted is False
    mock_client.delete_collection.assert_not_awaited()


@pytest.mark.asyncio()
async def test_delete_sparse_collection_deletes_when_present() -> None:
    from vecpipe.sparse import delete_sparse_collection

    mock_client = AsyncMock()
    mock_client.get_collections.return_value = SimpleNamespace(collections=[SimpleNamespace(name="sparse_col")])
    mock_client.delete_collection = AsyncMock()

    deleted = await delete_sparse_collection("sparse_col", mock_client)

    assert deleted is True
    mock_client.delete_collection.assert_awaited_once_with(collection_name="sparse_col")


@pytest.mark.asyncio()
async def test_get_sparse_collection_info_returns_none_when_missing() -> None:
    from vecpipe.sparse import get_sparse_collection_info

    mock_client = AsyncMock()
    mock_client.get_collections.return_value = SimpleNamespace(collections=[SimpleNamespace(name="other")])
    mock_client.get_collection = AsyncMock()

    info = await get_sparse_collection_info("sparse_col", mock_client)

    assert info is None
    mock_client.get_collection.assert_not_awaited()


@pytest.mark.asyncio()
async def test_get_sparse_collection_info_fetches_when_present() -> None:
    from vecpipe.sparse import get_sparse_collection_info

    mock_client = AsyncMock()
    mock_client.get_collections.return_value = SimpleNamespace(collections=[SimpleNamespace(name="sparse_col")])
    mock_client.get_collection = AsyncMock(return_value=MagicMock())

    info = await get_sparse_collection_info("sparse_col", mock_client)

    assert info is mock_client.get_collection.return_value
    mock_client.get_collection.assert_awaited_once_with(collection_name="sparse_col")


def test_generate_sparse_collection_name() -> None:
    from vecpipe.sparse import generate_sparse_collection_name

    assert generate_sparse_collection_name("dense", "bm25") == "dense_sparse_bm25"
