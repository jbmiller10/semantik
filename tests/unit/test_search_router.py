"""Router-level tests for vecpipe search API."""

from unittest.mock import patch

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

import vecpipe.search.router as router_module
from shared.contracts.search import SearchResponse, SearchResult
from vecpipe.search.router import router


def make_client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    router_module.settings.INTERNAL_API_KEY = "test-internal-key"
    client = TestClient(app)
    client.headers.update({"X-Internal-Api-Key": "test-internal-key"})
    return client


def test_search_semantic_rerank_off() -> None:
    client = make_client()
    search_response = SearchResponse(
        query="semantic",
        results=[SearchResult(doc_id="d1", chunk_id="c1", score=0.9, path="/doc")],
        num_results=1,
        search_type="semantic",
        reranking_used=False,
    )

    with patch("vecpipe.search.router.service.perform_search", return_value=search_response) as mock_search:
        resp = client.post("/search", json={"query": "semantic", "k": 1})
        assert resp.status_code == 200
        body = resp.json()
        assert body["results"][0]["score"] == 0.9
        assert body["reranking_used"] is False
        mock_search.assert_called_once()


def test_search_semantic_rerank_on() -> None:
    client = make_client()
    search_response = SearchResponse(
        query="semantic",
        results=[SearchResult(doc_id="d1", chunk_id="c1", score=0.95, path="/doc")],
        num_results=1,
        search_type="semantic",
        reranking_used=True,
        reranker_model="Qwen/Qwen3-Reranker-0.6B",
    )

    with patch("vecpipe.search.router.service.perform_search", return_value=search_response):
        resp = client.post("/search", json={"query": "semantic", "k": 1, "use_reranker": True})
        assert resp.status_code == 200
        assert resp.json()["reranker_model"] == "Qwen/Qwen3-Reranker-0.6B"


def test_search_error_mapping() -> None:
    client = make_client()

    with patch("vecpipe.search.router.service.perform_search", side_effect=HTTPException(status_code=502, detail="db")):
        resp = client.post("/search", json={"query": "broken"})
        assert resp.status_code == 502
        assert resp.json()["detail"] == "db"


# NOTE: Legacy hybrid_search endpoint test removed - use search_mode="hybrid" instead
