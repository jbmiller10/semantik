from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import HTTPException

from shared.contracts.search import SearchRequest
from vecpipe.search import service
from vecpipe.search.runtime import VecpipeRuntime
from vecpipe.search.schemas import EmbedRequest, PointPayload, UpsertPoint, UpsertRequest

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
    settings.MODEL_UNLOAD_AFTER_SECONDS = 300
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

    sparse_manager = AsyncMock()

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


@pytest.mark.asyncio()
async def test_perform_search_hybrid_maps_sparse_results_to_original_chunk_ids(
    mock_settings: Mock, runtime: VecpipeRuntime
) -> None:
    request = SearchRequest(query="q", k=3, search_mode="hybrid", search_type="semantic", include_content=True)

    dense_payload = {"doc_id": "doc-dense", "chunk_id": "orig-1", "path": "/dense", "content": "dense"}
    dense_missing_doc = {"chunk_id": "orig-missing", "path": "/broken"}  # should be skipped later

    sparse_results = [
        {"chunk_id": "sparse-1", "score": 1.0, "payload": {"metadata": {"original_chunk_id": "orig-2"}}},
        {"chunk_id": "sparse-2", "score": 0.9, "payload": {"original_chunk_id": "orig-3"}},
    ]

    payloads_by_chunk_id = {
        "orig-2": {"doc_id": "doc-sparse2", "chunk_id": "orig-2", "path": "/s2", "content": "s2"},
        "orig-3": {"doc_id": "doc-sparse3", "chunk_id": "orig-3", "path": "/s3", "content": "s3"},
    }

    async def _fake_rerank(**kwargs):  # type: ignore[no-untyped-def]
        return kwargs["results"], None, None

    with (
        patch("vecpipe.search.service.settings", mock_settings),
        patch("vecpipe.search.service.resolve_collection_name", new_callable=AsyncMock, return_value="col"),
        patch(
            "vecpipe.search.service.get_collection_info",
            new_callable=AsyncMock,
            return_value=(1024, {"config": {"params": {"vectors": {"size": 1024}}}}),
        ),
        patch(
            "vecpipe.search.service.get_cached_collection_metadata",
            new_callable=AsyncMock,
            return_value={
                "sparse_index_config": {
                    "enabled": True,
                    "plugin_id": "bm25-local",
                    "sparse_collection_name": "sparse_col",
                    "model_config": {"bm25": True},
                },
                "model_name": "col-model",
                "quantization": "int8",
                "instruction": "collection-instruction",
            },
        ),
        patch("vecpipe.search.service.generate_embedding", new_callable=AsyncMock, return_value=[0.1] * 1024),
        patch(
            "vecpipe.search.service.search_dense_qdrant",
            new_callable=AsyncMock,
            return_value=(
                [
                    {"id": "dense-1", "score": 0.9, "payload": dense_payload},
                    {"id": "dense-missing", "score": 0.8, "payload": dense_missing_doc},
                ],
                False,
            ),
        ),
        patch(
            "vecpipe.search.service.perform_sparse_search",
            new_callable=AsyncMock,
            return_value=(sparse_results, 12.0, []),
        ),
        patch(
            "vecpipe.search.service.fetch_payloads_for_chunk_ids",
            new_callable=AsyncMock,
            return_value=payloads_by_chunk_id,
        ) as fetch_payloads,
        patch("vecpipe.search.service.maybe_rerank_results", new=_fake_rerank),
    ):
        resp = await service.perform_search(request, runtime=runtime)

    assert resp.search_mode_used == "hybrid"
    assert any(r.chunk_id == "orig-2" for r in resp.results)
    assert not any(str(r.chunk_id).startswith("sparse-") for r in resp.results)

    called_chunk_ids = fetch_payloads.await_args.kwargs["chunk_ids"]
    assert set(called_chunk_ids) == {"orig-2", "orig-3"}


@pytest.mark.asyncio()
async def test_perform_search_sparse_only_skips_dense_embedding(mock_settings: Mock, runtime: VecpipeRuntime) -> None:
    request = SearchRequest(query="q", k=2, search_mode="sparse", search_type="semantic", include_content=False)

    sparse_results = [
        {"chunk_id": "orig-1", "score": 1.0, "payload": {"doc_id": "d1", "chunk_id": "orig-1", "path": "/p"}},
    ]

    async def _fake_rerank(**kwargs):  # type: ignore[no-untyped-def]
        return kwargs["results"], None, None

    with (
        patch("vecpipe.search.service.settings", mock_settings),
        patch("vecpipe.search.service.resolve_collection_name", new_callable=AsyncMock, return_value="col"),
        patch(
            "vecpipe.search.service.get_collection_info",
            new_callable=AsyncMock,
            return_value=(1024, {"config": {"params": {"vectors": {"size": 1024}}}}),
        ),
        patch(
            "vecpipe.search.service.get_cached_collection_metadata",
            new_callable=AsyncMock,
            return_value={
                "sparse_index_config": {
                    "enabled": True,
                    "plugin_id": "bm25-local",
                    "sparse_collection_name": "sparse_col",
                }
            },
        ),
        patch("vecpipe.search.service.generate_embedding", new_callable=AsyncMock) as gen_embed,
        patch("vecpipe.search.service.search_dense_qdrant", new_callable=AsyncMock) as dense_search,
        patch(
            "vecpipe.search.service.perform_sparse_search",
            new_callable=AsyncMock,
            return_value=(sparse_results, 7.0, []),
        ),
        patch("vecpipe.search.service.maybe_rerank_results", new=_fake_rerank),
    ):
        resp = await service.perform_search(request, runtime=runtime)

    assert resp.search_mode_used == "sparse"
    assert resp.search_time_ms == 7.0
    gen_embed.assert_not_awaited()
    dense_search.assert_not_awaited()


@pytest.mark.asyncio()
async def test_perform_search_falls_back_to_dense_when_sparse_index_unavailable(
    mock_settings: Mock, runtime: VecpipeRuntime
) -> None:
    request = SearchRequest(query="q", k=1, search_mode="hybrid", search_type="semantic")

    with (
        patch("vecpipe.search.service.settings", mock_settings),
        patch("vecpipe.search.service.resolve_collection_name", new_callable=AsyncMock, return_value="col"),
        patch(
            "vecpipe.search.service.get_collection_info",
            new_callable=AsyncMock,
            return_value=(1024, {"config": {"params": {"vectors": {"size": 1024}}}}),
        ),
        patch(
            "vecpipe.search.service.get_cached_collection_metadata",
            new_callable=AsyncMock,
            return_value={"sparse_index_config": {"enabled": True, "sparse_collection_name": "missing-plugin"}},
        ),
        patch("vecpipe.search.service.generate_embedding", new_callable=AsyncMock, return_value=[0.1] * 1024),
        patch(
            "vecpipe.search.service.search_dense_qdrant",
            new_callable=AsyncMock,
            return_value=(
                [{"id": "1", "score": 0.9, "payload": {"doc_id": "d", "chunk_id": "c", "path": "/p"}}],
                False,
            ),
        ),
        patch("vecpipe.search.service.perform_sparse_search", new_callable=AsyncMock) as sparse_search,
        patch("vecpipe.search.service.maybe_rerank_results", new_callable=AsyncMock, return_value=([], None, None)),
    ):
        resp = await service.perform_search(request, runtime=runtime)

    assert resp.search_mode_used == "dense"
    assert resp.warnings
    sparse_search.assert_not_awaited()


@pytest.mark.asyncio()
async def test_perform_search_applies_score_threshold_and_includes_content_for_reranker(
    mock_settings: Mock, runtime: VecpipeRuntime
) -> None:
    request = SearchRequest(
        query="q",
        k=2,
        search_mode="dense",
        search_type="semantic",
        include_content=False,
        use_reranker=True,
        score_threshold=0.5,
    )

    dense_results = [
        {"id": "low", "score": 0.4, "payload": {"doc_id": "d0", "chunk_id": "c0", "path": "/p0", "content": "low"}},
        {"id": "hi", "score": 0.6, "payload": {"doc_id": "d1", "chunk_id": "c1", "path": "/p1", "content": "hi"}},
    ]

    async def _fake_rerank(**kwargs):  # type: ignore[no-untyped-def]
        return kwargs["results"], "reranker-model", 5.0

    with (
        patch("vecpipe.search.service.settings", mock_settings),
        patch("vecpipe.search.service.resolve_collection_name", new_callable=AsyncMock, return_value="col"),
        patch(
            "vecpipe.search.service.get_collection_info",
            new_callable=AsyncMock,
            return_value=(1024, {"config": {"params": {"vectors": {"size": 1024}}}}),
        ),
        patch("vecpipe.search.service.get_cached_collection_metadata", new_callable=AsyncMock, return_value=None),
        patch("vecpipe.search.service.generate_embedding", new_callable=AsyncMock, return_value=[0.1] * 1024),
        patch(
            "vecpipe.search.service.search_dense_qdrant", new_callable=AsyncMock, return_value=(dense_results, False)
        ),
        patch("vecpipe.search.service.maybe_rerank_results", new=_fake_rerank),
    ):
        resp = await service.perform_search(request, runtime=runtime)

    assert resp.num_results == 1
    assert resp.results[0].chunk_id == "c1"
    # include_content=False but use_reranker=True should force content to be included.
    assert resp.results[0].content == "hi"
    assert resp.reranking_used is True


@pytest.mark.asyncio()
async def test_embed_texts_treats_class_name_insufficient_memory_error_as_oom(
    mock_settings: Mock, runtime: VecpipeRuntime
) -> None:
    class InsufficientMemoryError(Exception):
        pass

    runtime.model_manager.generate_embeddings_batch_async = AsyncMock(side_effect=InsufficientMemoryError("oom"))

    with patch("vecpipe.search.service.settings", mock_settings):
        with pytest.raises(HTTPException) as exc_info:
            await service.embed_texts(
                EmbedRequest(texts=["a"], model_name="m", quantization="float32", batch_size=1), runtime=runtime
            )

    assert exc_info.value.status_code == 507
    assert exc_info.value.detail["error"] == "insufficient_memory"


class _FakeQdrantHttp:
    def __init__(self, collection_dim: int, *, put_status: int = 200, put_payload: dict | None = None) -> None:
        self.collection_dim = collection_dim
        self.put_status = put_status
        self.put_payload = put_payload or {}
        self.put_urls: list[str] = []

    async def get(self, _url: str):  # type: ignore[no-untyped-def]
        import httpx

        return httpx.Response(
            200,
            json={"result": {"config": {"params": {"vectors": {"size": self.collection_dim}}}}},
            request=httpx.Request("GET", "http://qdrant.local"),
        )

    async def put(self, url: str, *, json: dict):  # type: ignore[no-untyped-def]
        import httpx

        self.put_urls.append(url)
        return httpx.Response(
            self.put_status,
            json=self.put_payload,
            request=httpx.Request("PUT", "http://qdrant.local"),
        )


@pytest.mark.asyncio()
async def test_upsert_points_validates_collection_dimension_and_wait_parameter(
    mock_settings: Mock, runtime: VecpipeRuntime
) -> None:
    runtime.qdrant_http = _FakeQdrantHttp(collection_dim=3)

    req = UpsertRequest(
        collection_name="col",
        wait=True,
        points=[
            UpsertPoint(
                id="p1",
                vector=[0.1, 0.2],  # wrong length (2 != 3)
                payload=PointPayload(doc_id="d", chunk_id="c", path="/p"),
            )
        ],
    )

    with patch("vecpipe.search.service.settings", mock_settings):
        with pytest.raises(HTTPException) as exc_info:
            await service.upsert_points(req, runtime=runtime)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["error"] == "dimension_mismatch"


@pytest.mark.asyncio()
async def test_upsert_points_returns_qdrant_error_detail_when_available(
    mock_settings: Mock, runtime: VecpipeRuntime
) -> None:
    runtime.qdrant_http = _FakeQdrantHttp(collection_dim=2, put_status=500, put_payload={"status": {"error": "bad"}})

    req = UpsertRequest(
        collection_name="col",
        wait=True,
        points=[
            UpsertPoint(
                id="p1",
                vector=[0.1, 0.2],
                payload=PointPayload(doc_id="d", chunk_id="c", path="/p"),
            )
        ],
    )

    with patch("vecpipe.search.service.settings", mock_settings):
        with pytest.raises(HTTPException) as exc_info:
            await service.upsert_points(req, runtime=runtime)

    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == "Vector database error: bad"
    assert runtime.qdrant_http.put_urls == ["/collections/col/points?wait=true"]


@pytest.mark.asyncio()
async def test_list_models_and_embedding_info_format_current_model(
    mock_settings: Mock, runtime: VecpipeRuntime
) -> None:
    fake_models = [
        {"name": "A", "provider": "dense_local", "dimension": 1024, "description": "a", "memory_estimate": {}},
        {"model_name": "B", "provider": "plugin_provider", "dimension": 768, "description": "b", "memory_estimate": {}},
    ]

    runtime.model_manager.get_status = Mock(
        return_value={
            "embedding_model_loaded": True,
            "current_embedding_model": "A_float16",
            "embedding_provider": "dense_local",
            "is_mock_mode": False,
            "provider_info": {"dimension": 1024, "device": "cuda", "quantization": "float16"},
        }
    )

    with (
        patch("vecpipe.search.service.settings", mock_settings),
        patch("shared.embedding.factory.get_all_supported_models", return_value=fake_models),
    ):
        models_out = await service.list_models(runtime=runtime)
        info_out = await service.embedding_info(runtime=runtime)

    assert models_out["current_model"] == "A"
    assert models_out["current_quantization"] == "float16"
    assert any(m["is_plugin"] for m in models_out["models"])
    assert info_out["current_model"] == "A"
    assert info_out["quantization"] == "float16"


@pytest.mark.asyncio()
async def test_load_model_rejects_mock_mode_and_suggest_models_cpu_branch(
    mock_settings: Mock, runtime: VecpipeRuntime
) -> None:
    mock_settings.USE_MOCK_EMBEDDINGS = True

    with patch("vecpipe.search.service.settings", mock_settings):
        with pytest.raises(HTTPException) as exc_info:
            await service.load_model("m", "float32", runtime=runtime)
    assert exc_info.value.status_code == 400

    mock_settings.USE_MOCK_EMBEDDINGS = False

    with (
        patch("vecpipe.search.service.settings", mock_settings),
        patch("vecpipe.memory_utils.get_gpu_memory_info", return_value=(0, 0)),
        patch("vecpipe.memory_utils.suggest_model_configuration", return_value={"embedding_model": "m"}),
    ):
        out = await service.suggest_models(runtime=runtime)

    assert out["gpu_available"] is False


@pytest.mark.asyncio()
async def test_suggest_models_gpu_branch_and_load_model_success(mock_settings: Mock, runtime: VecpipeRuntime) -> None:
    with (
        patch("vecpipe.search.service.settings", mock_settings),
        patch("vecpipe.memory_utils.get_gpu_memory_info", return_value=(6000, 8000)),
        patch("vecpipe.memory_utils.suggest_model_configuration", return_value={"embedding_model": "m"}),
    ):
        out = await service.suggest_models(runtime=runtime)

    assert out["gpu_available"] is True
    assert out["gpu_memory"]["usage_percent"] == 25.0

    runtime.model_manager.generate_embedding_async = AsyncMock(return_value=[0.1])
    runtime.model_manager.get_status = Mock(
        return_value={"embedding_provider": "dense_local", "provider_info": {"d": 1}}
    )

    with patch("vecpipe.search.service.settings", mock_settings):
        loaded = await service.load_model("model-x", "float32", runtime=runtime)

    assert loaded["status"] == "success"
    assert loaded["model"] == "model-x"
