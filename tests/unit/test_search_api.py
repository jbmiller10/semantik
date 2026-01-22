"""Unit tests for VecPipe search API wiring with VecpipeRuntime DI."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from shared.contracts.search import SearchResult
from vecpipe.memory_utils import InsufficientMemoryError
from vecpipe.search.router import router as search_router
from vecpipe.search.runtime import VecpipeRuntime

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture()
def mock_settings() -> Mock:
    settings = Mock()
    settings.QDRANT_HOST = "localhost"
    settings.QDRANT_PORT = 6333
    settings.QDRANT_API_KEY = None
    settings.DEFAULT_COLLECTION = "test_collection"
    settings.USE_MOCK_EMBEDDINGS = False
    settings.DEFAULT_EMBEDDING_MODEL = "test-model"
    settings.DEFAULT_QUANTIZATION = "float32"
    settings.INTERNAL_API_KEY = "test-internal-key"
    return settings


@pytest.fixture()
def runtime() -> Generator[VecpipeRuntime, None, None]:
    qdrant_http = AsyncMock()
    qdrant_sdk = AsyncMock()

    model_manager = Mock()
    model_manager.get_status = Mock(
        return_value={
            "embedding_model_loaded": True,
            "current_embedding_model": "test-model_float32",
            "embedding_provider": "dense_local",
            "is_mock_mode": False,
            "provider_info": {"dimension": 1024},
        }
    )
    model_manager.generate_embedding_async = AsyncMock(return_value=[0.1] * 1024)
    model_manager.generate_embeddings_batch_async = AsyncMock(return_value=[[0.1] * 1024])
    model_manager.rerank_async = AsyncMock(return_value=[(0, 0.95)])

    sparse_manager = AsyncMock()
    sparse_manager.get_loaded_plugins = Mock(return_value=[])

    executor = ThreadPoolExecutor(max_workers=1)
    try:
        yield VecpipeRuntime(
            qdrant_http=qdrant_http,
            qdrant_sdk=qdrant_sdk,
            model_manager=model_manager,
            sparse_manager=sparse_manager,
            executor=executor,
            llm_manager=None,
        )
    finally:
        executor.shutdown(wait=False)


@pytest.fixture()
def client(mock_settings: Mock, runtime: VecpipeRuntime) -> Generator[TestClient, None, None]:
    app = FastAPI()
    app.include_router(search_router)
    app.state.vecpipe_runtime = runtime

    with (
        patch("vecpipe.search.service.settings", mock_settings),
        patch("vecpipe.search.auth.settings", mock_settings),
    ):
        client = TestClient(app)
        client.headers.update({"X-Internal-Api-Key": mock_settings.INTERNAL_API_KEY})
        yield client


def test_generate_mock_embedding_deterministic() -> None:
    from vecpipe.search.service import generate_mock_embedding

    emb1 = generate_mock_embedding("hello", 16)
    emb2 = generate_mock_embedding("hello", 16)
    assert emb1 == emb2
    assert len(emb1) == 16


def test_model_status_endpoint(client: TestClient) -> None:
    resp = client.get("/model/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["embedding_provider"] == "dense_local"


def test_root_endpoint(client: TestClient, runtime: VecpipeRuntime, mock_settings: Mock) -> None:
    resp = Mock()
    resp.raise_for_status = Mock()
    resp.json.return_value = {
        "result": {
            "points_count": 100,
            "config": {"params": {"vectors": {"size": 1024}}},
        }
    }
    runtime.qdrant_http.get = AsyncMock(return_value=resp)

    out = client.get("/")
    assert out.status_code == 200
    body = out.json()
    assert body["status"] == "healthy"
    assert body["collection"]["name"] == mock_settings.DEFAULT_COLLECTION
    assert body["collection"]["points_count"] == 100


def test_health_endpoint(client: TestClient, runtime: VecpipeRuntime) -> None:
    resp = Mock()
    resp.status_code = 200
    resp.json.return_value = {"result": {"collections": [{"name": "col1"}, {"name": "col2"}]}}
    runtime.qdrant_http.get = AsyncMock(return_value=resp)

    out = client.get("/health")
    assert out.status_code == 200
    body = out.json()
    assert body["components"]["qdrant"]["collections_count"] == 2
    assert body["components"]["embedding"]["status"] == "healthy"


def test_search_post_basic(client: TestClient) -> None:
    dense_payload = {
        "path": "/test/file1.txt",
        "chunk_id": "chunk-1",
        "doc_id": "doc-1",
        "content": "hello",
    }

    with (
        patch(
            "vecpipe.search.service.get_collection_info",
            new_callable=AsyncMock,
            return_value=(1024, {"config": {"params": {"vectors": {"size": 1024}}}}),
        ),
        patch("vecpipe.search.service.get_cached_collection_metadata", new_callable=AsyncMock, return_value=None),
        patch("vecpipe.search.service.generate_embedding", new_callable=AsyncMock, return_value=[0.1] * 1024),
        patch(
            "vecpipe.search.service.search_dense_qdrant",
            new_callable=AsyncMock,
            return_value=([{"id": "1", "score": 0.95, "payload": dense_payload}], False),
        ),
    ):
        out = client.post("/search", json={"query": "test", "k": 1})
        assert out.status_code == 200
        body = out.json()
        assert body["num_results"] == 1
        assert body["results"][0]["doc_id"] == "doc-1"
        assert body["results"][0]["chunk_id"] == "chunk-1"


def test_search_dimension_mismatch_returns_400(client: TestClient) -> None:
    dense_payload = {
        "path": "/test/file1.txt",
        "chunk_id": "chunk-1",
        "doc_id": "doc-1",
        "content": "hello",
    }

    with (
        patch(
            "vecpipe.search.service.get_collection_info",
            new_callable=AsyncMock,
            return_value=(1024, {"config": {"params": {"vectors": {"size": 1024}}}}),
        ),
        patch("vecpipe.search.service.get_cached_collection_metadata", new_callable=AsyncMock, return_value=None),
        patch("vecpipe.search.service.generate_embedding", new_callable=AsyncMock, return_value=[0.1] * 768),
        patch(
            "vecpipe.search.service.search_dense_qdrant",
            new_callable=AsyncMock,
            return_value=([{"id": "1", "score": 0.95, "payload": dense_payload}], False),
        ),
    ):
        out = client.post("/search", json={"query": "test", "k": 1})
        assert out.status_code == 400
        detail = out.json()["detail"]
        assert detail["error"] == "dimension_mismatch"
        assert detail["expected_dimension"] == 1024
        assert detail["actual_dimension"] == 768


def test_search_qdrant_error_returns_502(client: TestClient) -> None:
    request = httpx.Request("POST", "http://qdrant.local")
    response = httpx.Response(500, request=request)
    err = httpx.HTTPStatusError("boom", request=request, response=response)

    with (
        patch(
            "vecpipe.search.service.get_collection_info",
            new_callable=AsyncMock,
            return_value=(1024, {"config": {"params": {"vectors": {"size": 1024}}}}),
        ),
        patch("vecpipe.search.service.get_cached_collection_metadata", new_callable=AsyncMock, return_value=None),
        patch("vecpipe.search.service.generate_embedding", new_callable=AsyncMock, return_value=[0.1] * 1024),
        patch("vecpipe.search.service.search_dense_qdrant", new_callable=AsyncMock, side_effect=err),
    ):
        out = client.post("/search", json={"query": "test", "k": 1})
        assert out.status_code == 502


def test_embed_endpoint_success(client: TestClient, runtime: VecpipeRuntime) -> None:
    runtime.model_manager.generate_embeddings_batch_async = AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])
    out = client.post(
        "/embed",
        json={"texts": ["a", "b"], "model_name": "m", "quantization": "float32", "batch_size": 2, "mode": "query"},
    )
    assert out.status_code == 200
    body = out.json()
    assert len(body["embeddings"]) == 2


def test_embed_endpoint_insufficient_memory(client: TestClient, runtime: VecpipeRuntime) -> None:
    runtime.model_manager.generate_embeddings_batch_async = AsyncMock(side_effect=InsufficientMemoryError("oom"))
    out = client.post(
        "/embed",
        json={"texts": ["a"], "model_name": "m", "quantization": "float32", "batch_size": 1},
    )
    assert out.status_code == 507
    assert out.json()["detail"]["error"] == "insufficient_memory"


def test_upsert_wait_false_omits_query_parameter(client: TestClient, runtime: VecpipeRuntime) -> None:
    resp = Mock()
    resp.raise_for_status = Mock()
    runtime.qdrant_http.put = AsyncMock(return_value=resp)

    payload = {
        "id": "p1",
        "vector": [0.1, 0.2],
        "payload": {"doc_id": "d", "chunk_id": "c", "path": "/x"},
    }
    out = client.post(
        "/upsert",
        json={"collection_name": "col", "wait": False, "points": [payload]},
    )
    assert out.status_code == 200
    url = runtime.qdrant_http.put.await_args.args[0]
    assert url == "/collections/col/points"


def test_search_response_model_compatible() -> None:
    """SearchResult still validates with required fields."""
    r = SearchResult(doc_id="d", chunk_id="c", score=0.1, path="/p")
    assert r.doc_id == "d"


def test_search_rerank_failure_adds_warning_and_marks_reranking_unused(client: TestClient) -> None:
    dense_payload = {
        "path": "/test/file1.txt",
        "chunk_id": "chunk-1",
        "doc_id": "doc-1",
        "content": "hello",
    }

    async def _fake_maybe_rerank_results(**kwargs):  # type: ignore[no-untyped-def]
        results = kwargs["results"]
        request = kwargs["request"]
        return results[: request.k], None, 12.3

    with (
        patch(
            "vecpipe.search.service.get_collection_info",
            new_callable=AsyncMock,
            return_value=(1024, {"config": {"params": {"vectors": {"size": 1024}}}}),
        ),
        patch("vecpipe.search.service.get_cached_collection_metadata", new_callable=AsyncMock, return_value=None),
        patch("vecpipe.search.service.generate_embedding", new_callable=AsyncMock, return_value=[0.1] * 1024),
        patch(
            "vecpipe.search.service.search_dense_qdrant",
            new_callable=AsyncMock,
            return_value=([{"id": "1", "score": 0.95, "payload": dense_payload}], False),
        ),
        patch("vecpipe.search.service.maybe_rerank_results", new=_fake_maybe_rerank_results),
    ):
        out = client.post("/search", json={"query": "test", "k": 1, "use_reranker": True})
        assert out.status_code == 200
        body = out.json()
        assert body["reranking_used"] is False
        assert body["reranker_model"] is None
        assert "Reranking failed; returning un-reranked results" in body["warnings"]


def test_search_dense_sdk_fallback_adds_warning(client: TestClient) -> None:
    dense_payload = {
        "path": "/test/file1.txt",
        "chunk_id": "chunk-1",
        "doc_id": "doc-1",
        "content": "hello",
    }

    with (
        patch(
            "vecpipe.search.service.get_collection_info",
            new_callable=AsyncMock,
            return_value=(1024, {"config": {"params": {"vectors": {"size": 1024}}}}),
        ),
        patch("vecpipe.search.service.get_cached_collection_metadata", new_callable=AsyncMock, return_value=None),
        patch("vecpipe.search.service.generate_embedding", new_callable=AsyncMock, return_value=[0.1] * 1024),
        patch(
            "vecpipe.search.service.search_dense_qdrant",
            new_callable=AsyncMock,
            return_value=([{"id": "1", "score": 0.95, "payload": dense_payload}], True),
        ),
    ):
        out = client.post("/search", json={"query": "test", "k": 1})
        assert out.status_code == 200
        assert "Dense search SDK failed; used REST fallback" in out.json()["warnings"]


def test_batch_search_sets_per_query_warnings_for_dense_sdk_fallback(client: TestClient) -> None:
    dense_payload_1 = {
        "path": "/test/file1.txt",
        "chunk_id": "chunk-1",
        "doc_id": "doc-1",
    }
    dense_payload_2 = {
        "path": "/test/file2.txt",
        "chunk_id": "chunk-2",
        "doc_id": "doc-2",
    }

    with (
        patch("vecpipe.search.service.generate_embedding", new_callable=AsyncMock, return_value=[0.1] * 1024),
        patch(
            "vecpipe.search.service.search_dense_qdrant",
            new_callable=AsyncMock,
            side_effect=[
                ([{"id": "1", "score": 0.95, "payload": dense_payload_1}], False),
                ([{"id": "2", "score": 0.85, "payload": dense_payload_2}], True),
            ],
        ),
    ):
        out = client.post(
            "/search/batch",
            json={
                "queries": ["q1", "q2"],
                "k": 1,
                "search_type": "semantic",
                "collection": "test_collection",
            },
        )

    assert out.status_code == 200
    body = out.json()
    assert len(body["responses"]) == 2
    assert body["responses"][0]["results"][0]["doc_id"] == "doc-1"
    assert body["responses"][0]["warnings"] == []
    assert body["responses"][1]["results"][0]["doc_id"] == "doc-2"
    assert "Dense search SDK failed; used REST fallback" in body["responses"][1]["warnings"]
