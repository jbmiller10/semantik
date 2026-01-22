"""Router-level tests for vecpipe search API."""

from unittest.mock import AsyncMock, Mock, patch

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from shared.contracts.search import SearchResponse, SearchResult
from vecpipe.search.router import router


def make_client(runtime: Mock | None = None) -> TestClient:
    """Create test client with patched auth settings."""
    app = FastAPI()
    app.include_router(router)
    # Satisfy VecPipe runtime dependency for endpoints.
    app.state.vecpipe_runtime = runtime or Mock(is_closed=False)
    client = TestClient(app)
    client.headers.update({"X-Internal-Api-Key": "test-internal-key"})
    return client


def test_search_semantic_rerank_off() -> None:
    search_response = SearchResponse(
        query="semantic",
        results=[SearchResult(doc_id="d1", chunk_id="c1", score=0.9, path="/doc")],
        num_results=1,
        search_type="semantic",
        reranking_used=False,
    )

    with (
        patch("vecpipe.search.auth.settings.INTERNAL_API_KEY", "test-internal-key"),
        patch("vecpipe.search.router.service.perform_search", return_value=search_response) as mock_search,
    ):
        client = make_client()
        resp = client.post("/search", json={"query": "semantic", "k": 1})
        assert resp.status_code == 200
        body = resp.json()
        assert body["results"][0]["score"] == 0.9
        assert body["reranking_used"] is False
        mock_search.assert_called_once()


def test_search_semantic_rerank_on() -> None:
    search_response = SearchResponse(
        query="semantic",
        results=[SearchResult(doc_id="d1", chunk_id="c1", score=0.95, path="/doc")],
        num_results=1,
        search_type="semantic",
        reranking_used=True,
        reranker_model="Qwen/Qwen3-Reranker-0.6B",
    )

    with (
        patch("vecpipe.search.auth.settings.INTERNAL_API_KEY", "test-internal-key"),
        patch("vecpipe.search.router.service.perform_search", return_value=search_response),
    ):
        client = make_client()
        resp = client.post("/search", json={"query": "semantic", "k": 1, "use_reranker": True})
        assert resp.status_code == 200
        assert resp.json()["reranker_model"] == "Qwen/Qwen3-Reranker-0.6B"


def test_search_error_mapping() -> None:
    with (
        patch("vecpipe.search.auth.settings.INTERNAL_API_KEY", "test-internal-key"),
        patch("vecpipe.search.router.service.perform_search", side_effect=HTTPException(status_code=502, detail="db")),
    ):
        client = make_client()
        resp = client.post("/search", json={"query": "broken"})
        assert resp.status_code == 502
    assert resp.json()["detail"] == "db"


# NOTE: Legacy hybrid_search endpoint test removed - use search_mode="hybrid" instead


def test_search_get_sets_search_mode_hybrid_when_search_type_is_hybrid() -> None:
    captured: dict[str, object] = {}

    async def _fake_perform_search(request, **_kwargs):  # type: ignore[no-untyped-def]
        captured["search_mode"] = request.search_mode
        return SearchResponse(
            query=request.query,
            results=[SearchResult(doc_id="d1", chunk_id="c1", score=0.9, path="/doc")],
            num_results=1,
            search_type=request.search_type,
            search_mode_used=request.search_mode,
        )

    with (
        patch("vecpipe.search.auth.settings.INTERNAL_API_KEY", "test-internal-key"),
        patch("vecpipe.search.router.service.perform_search", new=_fake_perform_search),
    ):
        client = make_client()
        resp = client.get("/search", params={"q": "hello", "k": 1, "search_type": "hybrid"})
        assert resp.status_code == 200

    assert captured["search_mode"] == "hybrid"


def test_search_post_legacy_hybrid_sets_search_mode_when_missing() -> None:
    captured: dict[str, object] = {}

    async def _fake_perform_search(request, **_kwargs):  # type: ignore[no-untyped-def]
        captured["search_mode"] = request.search_mode
        return SearchResponse(
            query=request.query,
            results=[SearchResult(doc_id="d1", chunk_id="c1", score=0.9, path="/doc")],
            num_results=1,
            search_type=request.search_type,
            search_mode_used=request.search_mode,
        )

    with (
        patch("vecpipe.search.auth.settings.INTERNAL_API_KEY", "test-internal-key"),
        patch("vecpipe.search.router.service.perform_search", new=_fake_perform_search),
    ):
        client = make_client()

        # No explicit search_mode provided: legacy hybrid should map to search_mode="hybrid".
        resp = client.post("/search", json={"query": "hello", "k": 1, "search_type": "hybrid"})
        assert resp.status_code == 200
        assert captured["search_mode"] == "hybrid"

        # Explicit search_mode should be preserved, even if search_type="hybrid".
        resp2 = client.post("/search", json={"query": "hello", "k": 1, "search_type": "hybrid", "search_mode": "dense"})
        assert resp2.status_code == 200
        assert captured["search_mode"] == "dense"


def test_root_and_collection_info_endpoints_return_qdrant_details() -> None:
    cfg = Mock(DEFAULT_COLLECTION="col", USE_MOCK_EMBEDDINGS=False)

    qdrant_response = Mock()
    qdrant_response.raise_for_status = Mock()
    qdrant_response.json.return_value = {
        "result": {
            "points_count": 123,
            "config": {"params": {"vectors": {"size": 1024}}},
        }
    }

    runtime = Mock(is_closed=False)
    runtime.qdrant_http.get = AsyncMock(return_value=qdrant_response)
    runtime.model_manager.get_status.return_value = {
        "current_embedding_model": "m",
        "embedding_provider": "p",
        "provider_info": {"dimension": 1024},
        "is_mock_mode": False,
    }

    with patch("vecpipe.search.router.service._get_settings", return_value=cfg):
        client = make_client(runtime)

        root_resp = client.get("/")
        assert root_resp.status_code == 200
        body = root_resp.json()
        assert body["collection"]["name"] == "col"
        assert body["collection"]["points_count"] == 123
        assert body["collection"]["vector_size"] == 1024

        info_resp = client.get("/collection/info")
        assert info_resp.status_code == 200
        info_body = info_resp.json()
        assert info_body["points_count"] == 123


def test_models_endpoints_proxy_service_calls() -> None:
    with (
        patch("vecpipe.search.auth.settings.INTERNAL_API_KEY", "test-internal-key"),
        patch("vecpipe.search.router.service.list_models", new_callable=AsyncMock, return_value={"models": []}),
        patch("vecpipe.search.router.service.suggest_models", new_callable=AsyncMock, return_value={"gpu_available": False}),
        patch("vecpipe.search.router.service.embedding_info", new_callable=AsyncMock, return_value={"mode": "mock"}),
    ):
        client = make_client()

        resp_models = client.get("/models")
        assert resp_models.status_code == 200
        assert resp_models.json() == {"models": []}

        resp_suggest = client.get("/models/suggest")
        assert resp_suggest.status_code == 200
        assert resp_suggest.json() == {"gpu_available": False}

        resp_info = client.get("/embedding/info")
        assert resp_info.status_code == 200
        assert resp_info.json() == {"mode": "mock"}
