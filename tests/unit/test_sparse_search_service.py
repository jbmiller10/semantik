from unittest.mock import AsyncMock, Mock, patch

import pytest


@pytest.mark.asyncio()
async def test_perform_sparse_search_accepts_dataclass_query_vector() -> None:
    """_perform_sparse_search should accept SparseQueryVector (not just dicts)."""
    from shared.plugins.types.sparse_indexer import SparseQueryVector
    from vecpipe.search.service import _perform_sparse_search

    class DummyPlugin:
        SPARSE_TYPE = "bm25"
        init_config: dict | None = None
        cleaned_up: bool = False

        async def initialize(self, config: dict | None = None) -> None:
            DummyPlugin.init_config = dict(config or {})

        async def cleanup(self) -> None:
            DummyPlugin.cleaned_up = True

        async def encode_query(self, _query: str) -> SparseQueryVector:
            assert DummyPlugin.init_config is not None
            return SparseQueryVector(indices=(1, 2, 3), values=(0.1, 0.2, 0.3))

    record = Mock()
    record.plugin_class = DummyPlugin

    mock_qdrant_client = AsyncMock()
    mock_qdrant_client.close = AsyncMock()

    mock_search_sparse_collection = AsyncMock(return_value=[{"chunk_id": "chunk-1", "score": 0.42, "payload": {}}])

    with (
        patch(
            "vecpipe.search.service._get_settings",
            return_value=Mock(QDRANT_HOST="localhost", QDRANT_PORT=6333, QDRANT_API_KEY="qdrant-key"),
        ),
        patch("shared.plugins.load_plugins"),
        patch("shared.plugins.plugin_registry.get", return_value=record),
        patch("qdrant_client.AsyncQdrantClient", return_value=mock_qdrant_client) as mock_async_qdrant_client,
        patch("vecpipe.search.service.search_sparse_collection", mock_search_sparse_collection),
    ):
        results, _time_ms = await _perform_sparse_search(
            collection_name="dense_collection",
            sparse_config={"plugin_id": "bm25-local", "sparse_collection_name": "sparse_collection"},
            query="hello world",
            k=10,
        )

    assert len(results) == 1
    assert results[0]["chunk_id"] == "chunk-1"
    mock_search_sparse_collection.assert_awaited_once()
    call_kwargs = mock_search_sparse_collection.await_args.kwargs
    assert call_kwargs["query_indices"] == [1, 2, 3]
    assert call_kwargs["query_values"] == [0.1, 0.2, 0.3]
    mock_async_qdrant_client.assert_called_once_with(url="http://localhost:6333", api_key="qdrant-key")
    assert DummyPlugin.init_config == {"collection_name": "dense_collection"}
    assert DummyPlugin.cleaned_up is True


@pytest.mark.asyncio()
async def test_get_sparse_config_for_collection_includes_api_key() -> None:
    """_get_sparse_config_for_collection should authenticate when Qdrant API key is set."""
    from vecpipe.search.service import _get_sparse_config_for_collection

    mock_qdrant_client = AsyncMock()
    mock_qdrant_client.close = AsyncMock()

    sparse_config = {"enabled": True, "plugin_id": "bm25-local", "sparse_collection_name": "sparse_collection"}

    with (
        patch(
            "vecpipe.search.service._get_settings",
            return_value=Mock(QDRANT_HOST="localhost", QDRANT_PORT=6333, QDRANT_API_KEY="qdrant-key"),
        ),
        patch("qdrant_client.AsyncQdrantClient", return_value=mock_qdrant_client) as mock_async_qdrant_client,
        patch("vecpipe.search.service.get_sparse_index_config", new=AsyncMock(return_value=sparse_config)),
    ):
        result = await _get_sparse_config_for_collection("dense_collection")

    assert result == sparse_config
    mock_async_qdrant_client.assert_called_once_with(url="http://localhost:6333", api_key="qdrant-key")
