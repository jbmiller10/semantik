from unittest.mock import AsyncMock, Mock, patch

import pytest

from shared.contracts.search import SearchRequest, SearchResult


@pytest.mark.asyncio()
async def test_maybe_rerank_results_fetches_missing_content_before_reranking() -> None:
    from vecpipe.search.rerank import maybe_rerank_results

    request = SearchRequest(query="q", k=2, use_reranker=True)
    results = [
        SearchResult(doc_id="d1", chunk_id="c1", score=0.1, path="/p1", content=None),
        SearchResult(doc_id="d2", chunk_id="c2", score=0.2, path="/p2", content="already"),
    ]

    fetch_payloads = AsyncMock(return_value={"c1": {"content": "fetched"}})

    model_manager = Mock()
    model_manager.rerank_async = AsyncMock(return_value=[(1, 0.9), (0, 0.8)])

    with patch("vecpipe.search.rerank.fetch_payloads_for_chunk_ids", fetch_payloads):
        reranked, model_used, _time_ms = await maybe_rerank_results(
            cfg=Mock(),
            model_manager=model_manager,
            qdrant_http=AsyncMock(),
            collection_name="col",
            request=request,
            results=results,
            embedding_model_name="Qwen/Qwen3-Embedding-0.6B",
            embedding_quantization="float16",
        )

    fetch_payloads.assert_awaited_once()
    fetch_call_kwargs = fetch_payloads.await_args.kwargs
    assert fetch_call_kwargs["chunk_ids"] == ["c1"]

    model_manager.rerank_async.assert_awaited_once()
    rerank_call_kwargs = model_manager.rerank_async.await_args.kwargs
    assert rerank_call_kwargs["documents"] == ["fetched", "already"]
    assert rerank_call_kwargs["top_k"] == 2

    assert model_used is not None
    assert [r.chunk_id for r in reranked] == ["c2", "c1"]
