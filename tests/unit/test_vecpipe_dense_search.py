from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from vecpipe.search.dense_search import generate_embedding, generate_mock_embedding, search_dense_qdrant


def test_generate_mock_embedding_is_deterministic_and_normalized() -> None:
    emb1 = generate_mock_embedding("hello world", vector_dim=16)
    emb2 = generate_mock_embedding("hello world", vector_dim=16)

    assert emb1 == emb2
    assert len(emb1) == 16

    # Rough normalization check.
    norm = sum(v * v for v in emb1) ** 0.5
    assert 0.99 <= norm <= 1.01


@pytest.mark.asyncio()
async def test_generate_embedding_uses_mock_embeddings_when_enabled() -> None:
    cfg = SimpleNamespace(USE_MOCK_EMBEDDINGS=True)
    out = await generate_embedding(
        cfg=cfg,
        model_manager=None,
        text="hello",
        model_name="any",
        quantization="float32",
        instruction=None,
        mode="query",
        vector_dim=8,
    )
    assert len(out) == 8


@pytest.mark.asyncio()
async def test_generate_embedding_raises_when_model_manager_missing() -> None:
    cfg = SimpleNamespace(USE_MOCK_EMBEDDINGS=False)
    with pytest.raises(RuntimeError, match="Model manager not initialized"):
        await generate_embedding(
            cfg=cfg,
            model_manager=None,
            text="hello",
            model_name="any",
            quantization="float32",
            instruction=None,
            mode="query",
            vector_dim=None,
        )


@pytest.mark.asyncio()
async def test_generate_embedding_raises_when_provider_returns_none() -> None:
    cfg = SimpleNamespace(USE_MOCK_EMBEDDINGS=False)
    model_manager = Mock()
    model_manager.generate_embedding_async = AsyncMock(return_value=None)

    with pytest.raises(RuntimeError, match="Failed to generate embedding"):
        await generate_embedding(
            cfg=cfg,
            model_manager=model_manager,
            text="hello",
            model_name="any",
            quantization="float32",
            instruction=None,
            mode="query",
            vector_dim=None,
        )


@pytest.mark.asyncio()
async def test_search_dense_qdrant_uses_rest_for_filtered_search() -> None:
    qdrant_http = AsyncMock()
    resp = Mock()
    resp.raise_for_status = Mock()
    resp.json.return_value = {"result": [{"id": "1", "score": 0.9, "payload": {"doc_id": "d"}}]}
    qdrant_http.post = AsyncMock(return_value=resp)

    qdrant_sdk = AsyncMock()

    results, used_fallback = await search_dense_qdrant(
        collection_name="col",
        query_vector=[0.1, 0.2],
        limit=3,
        qdrant_http=qdrant_http,
        qdrant_sdk=qdrant_sdk,
        filters={"must": [{"key": "tenant", "match": {"value": "t1"}}]},
    )

    assert used_fallback is False
    assert results[0]["payload"]["doc_id"] == "d"
    qdrant_http.post.assert_awaited_once()
    qdrant_sdk.search.assert_not_awaited()


@pytest.mark.asyncio()
async def test_search_dense_qdrant_converts_sdk_points_dict_and_object() -> None:
    qdrant_http = AsyncMock()
    qdrant_sdk = AsyncMock()
    qdrant_sdk.search = AsyncMock(
        return_value=[
            {"id": "1", "score": 0.1, "payload": {"doc_id": "d1"}},
            SimpleNamespace(id=2, score=0.2, payload={"doc_id": "d2"}),
        ]
    )

    results, used_fallback = await search_dense_qdrant(
        collection_name="col",
        query_vector=[0.1, 0.2],
        limit=2,
        qdrant_http=qdrant_http,
        qdrant_sdk=qdrant_sdk,
        filters=None,
    )

    assert used_fallback is False
    assert results == [
        {"id": "1", "score": 0.1, "payload": {"doc_id": "d1"}},
        {"id": "2", "score": 0.2, "payload": {"doc_id": "d2"}},
    ]


@pytest.mark.asyncio()
async def test_search_dense_qdrant_falls_back_to_rest_when_sdk_fails() -> None:
    qdrant_http = AsyncMock()
    resp = Mock()
    resp.raise_for_status = Mock()
    resp.json.return_value = {"result": [{"id": "1", "score": 0.9, "payload": {"doc_id": "d"}}]}
    qdrant_http.post = AsyncMock(return_value=resp)

    qdrant_sdk = AsyncMock()
    qdrant_sdk.search = AsyncMock(side_effect=RuntimeError("sdk down"))

    fake_fallbacks = Mock()
    fake_fallbacks.labels.return_value.inc = Mock()

    with patch("vecpipe.search.dense_search.dense_search_fallbacks", fake_fallbacks):
        results, used_fallback = await search_dense_qdrant(
            collection_name="col",
            query_vector=[0.1, 0.2],
            limit=3,
            qdrant_http=qdrant_http,
            qdrant_sdk=qdrant_sdk,
            filters=None,
        )

    assert used_fallback is True
    assert results[0]["payload"]["doc_id"] == "d"
    fake_fallbacks.labels.assert_called_with(reason="sdk_error")
