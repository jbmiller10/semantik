from unittest.mock import AsyncMock, Mock, patch

import pytest


@pytest.mark.asyncio()
async def test_perform_sparse_search_accepts_dataclass_query_vector() -> None:
    """_perform_sparse_search should accept SparseQueryVector (not just dicts)."""
    from shared.plugins.types.sparse_indexer import SparseQueryVector
    from vecpipe.search.sparse_search import perform_sparse_search

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
        patch("shared.plugins.load_plugins"),
        patch("shared.plugins.plugin_registry.get", return_value=record),
        patch("qdrant_client.AsyncQdrantClient", return_value=mock_qdrant_client) as mock_async_qdrant_client,
        patch("vecpipe.search.sparse_search.search_sparse_collection", mock_search_sparse_collection),
    ):
        results, _time_ms, warnings = await perform_sparse_search(
            cfg=Mock(QDRANT_HOST="localhost", QDRANT_PORT=6333, QDRANT_API_KEY="qdrant-key"),
            collection_name="dense_collection",
            sparse_config={"plugin_id": "bm25-local", "sparse_collection_name": "sparse_collection"},
            query="hello world",
            k=10,
            sparse_manager=None,
            qdrant_sdk=None,
        )

    assert len(results) == 1
    assert results[0]["chunk_id"] == "chunk-1"
    assert warnings == []
    mock_search_sparse_collection.assert_awaited_once()
    call_kwargs = mock_search_sparse_collection.await_args.kwargs
    assert call_kwargs["query_indices"] == [1, 2, 3]
    assert call_kwargs["query_values"] == [0.1, 0.2, 0.3]
    mock_async_qdrant_client.assert_called_once_with(url="http://localhost:6333", api_key="qdrant-key")
    assert DummyPlugin.init_config == {"collection_name": "dense_collection"}
    assert DummyPlugin.cleaned_up is True


@pytest.mark.asyncio()
async def test_perform_sparse_search_returns_empty_when_no_plugin_id() -> None:
    from vecpipe.search.sparse_search import perform_sparse_search

    mock_fallbacks = Mock()
    mock_fallbacks.labels.return_value.inc = Mock()

    with (patch("vecpipe.search.sparse_search.sparse_search_fallbacks", mock_fallbacks),):
        results, time_ms, warnings = await perform_sparse_search(
            cfg=Mock(QDRANT_HOST="h", QDRANT_PORT=1, QDRANT_API_KEY=None),
            collection_name="dense",
            sparse_config={"enabled": True, "sparse_collection_name": "sparse"},
            query="q",
            k=5,
            sparse_manager=None,
            qdrant_sdk=None,
        )

    assert results == []
    assert time_ms == 0.0
    assert warnings == ["Sparse search skipped: missing sparse plugin_id"]
    mock_fallbacks.labels.assert_called_with(reason="no_plugin_id")


@pytest.mark.asyncio()
async def test_perform_sparse_search_returns_empty_when_plugin_missing() -> None:
    from vecpipe.search.sparse_search import perform_sparse_search

    mock_fallbacks = Mock()
    mock_fallbacks.labels.return_value.inc = Mock()

    with (
        patch("shared.plugins.load_plugins"),
        patch("shared.plugins.plugin_registry.get", return_value=None),
        patch("vecpipe.search.sparse_search.sparse_search_fallbacks", mock_fallbacks),
    ):
        results, time_ms, warnings = await perform_sparse_search(
            cfg=Mock(QDRANT_HOST="h", QDRANT_PORT=1, QDRANT_API_KEY=None),
            collection_name="dense",
            sparse_config={"plugin_id": "missing", "sparse_collection_name": "sparse"},
            query="q",
            k=5,
            sparse_manager=None,
            qdrant_sdk=None,
        )

    assert results == []
    assert time_ms == 0.0
    assert warnings == ["Sparse search skipped: plugin 'missing' not found"]
    mock_fallbacks.labels.assert_called_with(reason="plugin_not_found")


@pytest.mark.asyncio()
async def test_perform_sparse_search_returns_empty_for_invalid_query_vector_type() -> None:
    from vecpipe.search.sparse_search import perform_sparse_search

    class DummyPlugin:
        SPARSE_TYPE = "bm25"

        async def encode_query(self, _query: str):
            return ["not", "supported"]

    record = Mock()
    record.plugin_class = DummyPlugin

    mock_fallbacks = Mock()
    mock_fallbacks.labels.return_value.inc = Mock()

    with (
        patch("shared.plugins.load_plugins"),
        patch("shared.plugins.plugin_registry.get", return_value=record),
        patch("vecpipe.search.sparse_search.sparse_search_fallbacks", mock_fallbacks),
    ):
        results, _time_ms, warnings = await perform_sparse_search(
            cfg=Mock(QDRANT_HOST="h", QDRANT_PORT=1, QDRANT_API_KEY=None),
            collection_name="dense",
            sparse_config={"plugin_id": "bm25-local", "sparse_collection_name": "sparse"},
            query="q",
            k=5,
            sparse_manager=None,
            qdrant_sdk=None,
        )

    assert results == []
    assert warnings == ["Sparse search skipped: sparse query encoder returned unsupported vector type"]
    mock_fallbacks.labels.assert_called_with(reason="invalid_query_vector")


@pytest.mark.asyncio()
async def test_perform_sparse_search_returns_empty_when_indices_values_mismatch() -> None:
    from vecpipe.search.sparse_search import perform_sparse_search

    class DummyPlugin:
        SPARSE_TYPE = "bm25"

        async def encode_query(self, _query: str):
            return {"indices": [1, 2], "values": [0.1]}

    record = Mock()
    record.plugin_class = DummyPlugin

    mock_fallbacks = Mock()
    mock_fallbacks.labels.return_value.inc = Mock()

    with (
        patch("shared.plugins.load_plugins"),
        patch("shared.plugins.plugin_registry.get", return_value=record),
        patch("vecpipe.search.sparse_search.sparse_search_fallbacks", mock_fallbacks),
    ):
        results, _time_ms, warnings = await perform_sparse_search(
            cfg=Mock(QDRANT_HOST="h", QDRANT_PORT=1, QDRANT_API_KEY=None),
            collection_name="dense",
            sparse_config={"plugin_id": "bm25-local", "sparse_collection_name": "sparse"},
            query="q",
            k=5,
            sparse_manager=None,
            qdrant_sdk=None,
        )

    assert results == []
    assert warnings == ["Sparse search skipped: sparse query indices/values mismatch"]
    mock_fallbacks.labels.assert_called_with(reason="invalid_query_vector")


@pytest.mark.asyncio()
async def test_perform_sparse_search_uses_managed_sparse_manager_and_skips_plugin_loading() -> None:
    """Managed sparse_manager path should call encode_query and avoid direct plugin construction."""
    from vecpipe.search.sparse_search import perform_sparse_search

    class ManagedVector:
        _sparse_type = "splade"

        def __init__(self) -> None:
            self.indices = (1, 2)
            self.values = (0.3, 0.7)

    mock_sparse_manager = AsyncMock()
    mock_sparse_manager.encode_query = AsyncMock(return_value=ManagedVector())

    mock_qdrant = AsyncMock()
    mock_qdrant.close = AsyncMock()

    mock_search_sparse_collection = AsyncMock(return_value=[{"chunk_id": "c1", "score": 0.5, "payload": {"doc_id": "d"}}])

    with (
        patch("qdrant_client.AsyncQdrantClient", return_value=mock_qdrant),
        patch("vecpipe.search.sparse_search.search_sparse_collection", mock_search_sparse_collection),
        patch("shared.plugins.load_plugins") as load_plugins,
        patch("shared.plugins.plugin_registry.get") as plugin_get,
    ):
        results, _time_ms, warnings = await perform_sparse_search(
            cfg=Mock(QDRANT_HOST="h", QDRANT_PORT=1, QDRANT_API_KEY=None),
            collection_name="dense",
            sparse_config={"plugin_id": "splade-local", "sparse_collection_name": "sparse", "model_config": "not-a-dict"},
            query="q",
            k=5,
            sparse_manager=mock_sparse_manager,
            qdrant_sdk=None,
        )

    assert warnings == []
    assert results
    assert results[0]["chunk_id"] == "c1"
    mock_sparse_manager.encode_query.assert_awaited_once()
    # sparse_type came from query vector, so plugins should not be consulted
    load_plugins.assert_not_called()
    plugin_get.assert_not_called()


@pytest.mark.asyncio()
async def test_perform_sparse_search_returns_empty_when_sparse_collection_name_missing() -> None:
    from vecpipe.search.sparse_search import perform_sparse_search

    mock_fallbacks = Mock()
    mock_fallbacks.labels.return_value.inc = Mock()

    sparse_manager = AsyncMock()
    sparse_manager.encode_query = AsyncMock(return_value={"indices": [1], "values": [0.1]})

    with patch("vecpipe.search.sparse_search.sparse_search_fallbacks", mock_fallbacks):
        results, time_ms, warnings = await perform_sparse_search(
            cfg=Mock(QDRANT_HOST="h", QDRANT_PORT=1, QDRANT_API_KEY=None),
            collection_name="dense",
            sparse_config={"plugin_id": "bm25-local"},
            query="q",
            k=5,
            sparse_manager=sparse_manager,
            qdrant_sdk=None,
        )

    assert results == []
    assert time_ms == 0.0
    assert warnings == ["Sparse search skipped: missing sparse_collection_name"]
    mock_fallbacks.labels.assert_called_with(reason="no_collection_name")


@pytest.mark.asyncio()
async def test_perform_sparse_search_returns_empty_on_sparse_search_exception() -> None:
    from vecpipe.search.sparse_search import perform_sparse_search

    class Vector:
        indices = (1,)
        values = (0.1,)

    sparse_manager = AsyncMock()
    sparse_manager.encode_query = AsyncMock(return_value=Vector())

    mock_search_sparse_collection = AsyncMock(side_effect=RuntimeError("boom"))

    with patch("vecpipe.search.sparse_search.search_sparse_collection", mock_search_sparse_collection):
        results, _time_ms, warnings = await perform_sparse_search(
            cfg=Mock(QDRANT_HOST="h", QDRANT_PORT=1, QDRANT_API_KEY=None),
            collection_name="dense",
            sparse_config={"plugin_id": "bm25-local", "sparse_collection_name": "sparse"},
            query="q",
            k=5,
            sparse_manager=sparse_manager,
            qdrant_sdk=AsyncMock(),  # provided client, won't be closed
        )

    assert results == []
    assert warnings
    assert "Sparse search failed" in warnings[0]
