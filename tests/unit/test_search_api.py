"""Comprehensive unit tests for search_api.py to achieve 80%+ coverage.

This module tests all the endpoints, error scenarios, edge cases, and FAISS fallback logic.
"""

from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from prometheus_client import Counter, Histogram

import vecpipe.search_api as search_api_module
from vecpipe.search import state as search_state

# Import InsufficientMemoryError from the service module to ensure class identity
# matches what the exception handler catches (avoids dual-path import issues)
from vecpipe.search.service import InsufficientMemoryError
from vecpipe.search_api import (
    PointPayload,
    UpsertPoint,
    UpsertRequest,
    app,
    generate_mock_embedding,
    get_or_create_metric,
    lifespan,
    upsert_points,
)


@pytest.fixture()
def mock_settings() -> None:
    """Mock settings for testing."""
    mock = Mock()
    mock.QDRANT_HOST = "localhost"
    mock.QDRANT_PORT = 6333
    mock.QDRANT_API_KEY = None
    mock.DEFAULT_COLLECTION = "test_collection"
    mock.USE_MOCK_EMBEDDINGS = False
    mock.DEFAULT_EMBEDDING_MODEL = "test-model"
    mock.DEFAULT_QUANTIZATION = "float32"
    mock.MODEL_UNLOAD_AFTER_SECONDS = 300
    mock.SEARCH_API_PORT = 8088
    mock.METRICS_PORT = 9090
    mock.INTERNAL_API_KEY = "test-internal-key"
    mock.ENVIRONMENT = "test"
    mock.data_dir = Path("/tmp")
    mock.ENABLE_MEMORY_GOVERNOR = False  # Use regular ModelManager in tests
    return mock


@pytest.fixture()
def mock_qdrant_client() -> None:
    """Mock Qdrant client."""
    client = AsyncMock()
    client.get = AsyncMock()
    client.post = AsyncMock()
    client.put = AsyncMock()
    client.aclose = AsyncMock()
    return client


@pytest.fixture()
def mock_model_manager() -> None:
    """Mock model manager.

    Uses spec to limit attributes - excludes shutdown_async to simulate ModelManager
    (not GovernedModelManager which has async shutdown).
    """
    # Use spec to prevent Mock from auto-creating shutdown_async attribute
    manager = Mock(
        spec=[
            "generate_embedding",
            "generate_embedding_async",
            "generate_embeddings_batch_async",
            "rerank_async",
            "get_status",
            "shutdown",
            "current_model_key",
            "current_reranker_key",
        ]
    )
    manager.generate_embedding_async = AsyncMock(return_value=[0.1] * 1024)

    # Make generate_embeddings_batch_async return the correct number of embeddings
    async def mock_batch_embed(texts, *_args, **_kwargs) -> None:
        return [[0.1] * 1024 for _ in texts]

    manager.generate_embeddings_batch_async = AsyncMock(side_effect=mock_batch_embed)
    manager.rerank_async = AsyncMock(return_value=[(0, 0.95), (1, 0.90)])
    # Return proper status data for health and embedding_info endpoints
    manager.get_status = Mock(
        return_value={
            "embedding_model_loaded": True,
            "current_embedding_model": "test-model_float32",
            "embedding_provider": "dense_local",
            "is_mock_mode": False,
            "provider_info": {
                "model_name": "test-model",
                "dimension": 1024,
                "device": "cpu",
                "quantization": "float32",
                "max_sequence_length": 512,
            },
            "loaded_models": [],
            "memory_usage": {},
        }
    )
    manager.shutdown = Mock()
    manager.current_model_key = "test-model_float32"
    manager.current_reranker_key = None
    return manager


@pytest.fixture()
def mock_embedding_service() -> None:
    """Mock embedding service."""
    service = Mock()
    service.is_initialized = True
    service.current_model_name = "test-model"
    service.current_quantization = "float32"
    service.device = "cpu"
    service.mock_mode = False
    service.allow_quantization_fallback = True

    def mock_get_model_info(*_args, **_kwargs) -> None:
        return {"model_name": "test-model", "dimension": 1024, "description": "Test model"}

    service.get_model_info = Mock(side_effect=mock_get_model_info)
    return service


def test_search_api_globals_follow_search_state() -> None:
    """search_api exposes live search_state resources for callers and patchers."""
    sentinel_from_state = object()
    sentinel_from_api = object()

    from vecpipe.search.service import _get_model_manager, _get_qdrant_client

    original = {
        "qdrant_client": search_state.qdrant_client,
        "model_manager": search_state.model_manager,
        "embedding_service": search_state.embedding_service,
        "executor": search_state.executor,
    }

    try:
        search_state.qdrant_client = sentinel_from_state
        search_state.model_manager = sentinel_from_state
        search_state.embedding_service = sentinel_from_state
        search_state.executor = sentinel_from_state

        assert search_api_module.qdrant_client is sentinel_from_state
        assert search_api_module.model_manager is sentinel_from_state
        assert search_api_module.state_model_manager is sentinel_from_state
        assert search_api_module.embedding_service is sentinel_from_state
        assert search_api_module.executor is sentinel_from_state

        search_api_module.qdrant_client = sentinel_from_api
        search_api_module.model_manager = sentinel_from_api
        search_api_module.embedding_service = sentinel_from_api
        search_api_module.executor = sentinel_from_api

        # Helpers in search.service should propagate entrypoint patches back into shared state
        assert _get_qdrant_client() is sentinel_from_api
        assert _get_model_manager() is sentinel_from_api
        assert search_state.qdrant_client is sentinel_from_api
        assert search_state.model_manager is sentinel_from_api
        assert search_state.embedding_service is sentinel_from_api
        assert search_state.executor is sentinel_from_api
        assert search_api_module.embedding_service is sentinel_from_api
        assert search_api_module.executor is sentinel_from_api
    finally:
        search_state.qdrant_client = original["qdrant_client"]
        search_state.model_manager = original["model_manager"]
        search_state.embedding_service = original["embedding_service"]
        search_state.executor = original["executor"]

        for name in ("qdrant_client", "model_manager", "embedding_service", "executor"):
            search_api_module.__dict__.pop(name, None)


@pytest.fixture()
def mock_hybrid_engine() -> Generator[Any, None, None]:
    """Mock hybrid search engine."""
    with patch("vecpipe.search.service.HybridSearchEngine") as mock_class:
        engine = Mock()
        engine.extract_keywords = Mock(return_value=["test", "query"])
        engine.hybrid_search = Mock(
            return_value=[
                {
                    "score": 0.95,
                    "payload": {"path": "/test/file1.txt", "chunk_id": "chunk-1", "doc_id": "doc-1"},
                    "matched_keywords": ["test"],
                    "keyword_score": 0.8,
                    "combined_score": 0.875,
                }
            ]
        )
        engine.search_by_keywords = Mock(
            return_value=[
                {
                    "payload": {"path": "/test/file1.txt", "chunk_id": "chunk-1", "doc_id": "doc-1"},
                    "matched_keywords": ["test", "query"],
                }
            ]
        )
        engine.close = Mock()
        mock_class.return_value = engine
        yield engine


@pytest.fixture()
def test_client_for_search_api(
    mock_settings, mock_qdrant_client, mock_model_manager, mock_embedding_service
) -> Generator[Any, None, None]:
    """Create a test client for the search API with mocked dependencies."""
    original_qdrant = search_state.qdrant_client
    original_model_manager = search_state.model_manager
    original_embedding_service = search_state.embedding_service
    original_executor = search_state.executor

    exec_pool = ThreadPoolExecutor(max_workers=1)

    search_state.qdrant_client = mock_qdrant_client
    search_state.model_manager = mock_model_manager
    search_state.embedding_service = mock_embedding_service
    search_state.executor = exec_pool

    mock_embedding_facade = AsyncMock()
    mock_embedding_facade.initialize = AsyncMock()

    with (
        patch("vecpipe.search.service.settings", mock_settings),
        patch("vecpipe.search.lifespan.settings", mock_settings),
        patch("vecpipe.search.router.settings", mock_settings),
        patch("vecpipe.search_api.settings", mock_settings),
        patch("vecpipe.search.lifespan.httpx.AsyncClient", return_value=mock_qdrant_client),
        patch("vecpipe.search.lifespan.start_metrics_server"),
        # Note: get_embedding_service no longer used in lifespan - ModelManager handles providers
        patch("vecpipe.search.lifespan.ModelManager", return_value=mock_model_manager),
    ):
        client = TestClient(app)
        client.headers.update({"X-Internal-Api-Key": mock_settings.INTERNAL_API_KEY})
        yield client

    search_state.qdrant_client = original_qdrant
    search_state.model_manager = original_model_manager
    search_state.embedding_service = original_embedding_service
    search_state.executor = original_executor
    exec_pool.shutdown(wait=False)


class TestSearchAPI:
    """Test search API endpoints and functionality."""

    def test_generate_mock_embedding(self) -> None:
        """Test mock embedding generation."""
        text = "test query"
        vector_dim = 768

        embedding = generate_mock_embedding(text, vector_dim)

        assert len(embedding) == vector_dim
        assert all(isinstance(x, float) for x in embedding)
        # Check normalization (unit vector)
        norm = sum(x**2 for x in embedding) ** 0.5
        assert abs(norm - 1.0) < 0.01

        # Test with default dimension
        embedding_default = generate_mock_embedding(text)
        assert len(embedding_default) == 1024

        # Test deterministic behavior
        embedding2 = generate_mock_embedding(text, vector_dim)
        assert embedding == embedding2

    def test_get_or_create_metric(self) -> None:
        """Test metric creation and retrieval."""

        # Create a new metric
        metric1 = get_or_create_metric(Counter, "test_counter", "Test counter metric", ["label1", "label2"])
        assert metric1 is not None

        # Try to get the same metric again
        metric2 = get_or_create_metric(Counter, "test_counter", "Test counter metric", ["label1", "label2"])
        # Should return the same instance or handle duplicate gracefully
        assert metric2 is not None

        # Create histogram without labels
        hist = get_or_create_metric(Histogram, "test_histogram", "Test histogram", buckets=(0.1, 0.5, 1.0))
        assert hist is not None

    @pytest.mark.asyncio()
    async def test_lifespan(self, mock_settings, mock_qdrant_client, mock_model_manager) -> None:
        """Test application lifespan management.

        Note: ModelManager now handles embedding providers internally,
        so there's no separate embedding service initialization in lifespan.
        """
        # Create mock for Qdrant SDK client
        mock_qdrant_sdk = AsyncMock()
        mock_qdrant_sdk.close = AsyncMock()

        # Test that lifespan context manager starts and cleans up properly
        with (
            patch("vecpipe.search.lifespan.settings", mock_settings),
            patch("vecpipe.search.lifespan.httpx.AsyncClient") as mock_httpx,
            patch("vecpipe.search.lifespan.AsyncQdrantClient") as mock_qdrant_sdk_class,
            patch("vecpipe.search.lifespan.start_metrics_server") as mock_start_metrics,
            patch("vecpipe.search.lifespan.ModelManager") as mock_mm_class,
        ):
            mock_httpx.return_value = mock_qdrant_client
            mock_qdrant_sdk_class.return_value = mock_qdrant_sdk
            mock_mm_class.return_value = mock_model_manager

            # Test startup and shutdown
            test_app = FastAPI()

            async with lifespan(test_app):
                # Verify initialization
                mock_start_metrics.assert_called_once()
                mock_httpx.assert_called_once()
                mock_qdrant_sdk_class.assert_called_once()
                mock_mm_class.assert_called_once()

            # Verify cleanup
            mock_qdrant_client.aclose.assert_called_once()
            mock_qdrant_sdk.close.assert_called_once()
            mock_model_manager.shutdown.assert_called_once()

    def test_model_status(self, mock_model_manager, test_client_for_search_api) -> None:
        """Test /model/status endpoint."""
        response = test_client_for_search_api.get("/model/status")
        assert response.status_code == 200
        result = response.json()
        # Verify key status fields are present
        assert "embedding_model_loaded" in result
        assert "current_embedding_model" in result
        assert "embedding_provider" in result
        assert "provider_info" in result
        assert result["embedding_model_loaded"] is True
        assert result["current_embedding_model"] == "test-model_float32"

        # Test when model manager is not initialized

        original_manager = search_state.model_manager
        try:
            search_state.model_manager = None
            response = test_client_for_search_api.get("/model/status")
            assert response.status_code == 200
            assert response.json() == {"error": "Model manager not initialized"}
        finally:
            search_state.model_manager = original_manager

    def test_root_endpoint(
        self, mock_settings, mock_qdrant_client, mock_embedding_service, test_client_for_search_api
    ) -> None:
        """Test root health check endpoint."""
        # Mock successful response
        # Create a proper mock response object
        mock_response = Mock()
        mock_response.json.return_value = {
            "result": {"points_count": 100, "config": {"params": {"vectors": {"size": 1024}}}}
        }
        mock_response.raise_for_status = Mock()
        mock_qdrant_client.get.return_value = mock_response

        search_state.qdrant_client = mock_qdrant_client
        search_state.embedding_service = mock_embedding_service

        response = test_client_for_search_api.get("/")
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "healthy"
        assert result["collection"]["points_count"] == 100
        assert result["collection"]["vector_size"] == 1024
        assert result["embedding_mode"] == "real"
        assert "embedding_service" in result

        # Test error handling - need to patch the global variable in the module

        original_client = search_state.qdrant_client
        try:
            search_state.qdrant_client = None
            response = test_client_for_search_api.get("/")
            assert response.status_code == 503
            assert "Qdrant client not initialized" in response.json()["detail"]
        finally:
            search_state.qdrant_client = original_client

    def test_health_endpoint(
        self, mock_settings, mock_qdrant_client, mock_model_manager, test_client_for_search_api
    ) -> None:
        """Test /health endpoint.

        Note: With the plugin-aware provider system, ModelManager handles embedding
        providers. Models are lazy-loaded, so "no model loaded" is considered healthy.
        """
        mock_settings.USE_MOCK_EMBEDDINGS = False

        # Mock successful Qdrant response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": {"collections": [{"name": "col1"}, {"name": "col2"}]}}
        mock_qdrant_client.get.return_value = mock_response

        search_state.qdrant_client = mock_qdrant_client

        response = test_client_for_search_api.get("/health")
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "healthy"
        assert result["components"]["qdrant"]["status"] == "healthy"
        assert result["components"]["qdrant"]["collections_count"] == 2
        assert result["components"]["embedding"]["status"] == "healthy"
        assert result["components"]["embedding"]["model"] == "test-model_float32"
        assert result["components"]["embedding"]["provider"] == "dense_local"
        assert result["components"]["embedding"]["dimension"] == 1024
        assert result["components"]["embedding"]["is_mock_mode"] is False

        # Test with no model loaded - should still be healthy due to lazy loading
        mock_model_manager.get_status.return_value = {
            "embedding_model_loaded": False,
            "current_embedding_model": None,
            "embedding_provider": None,
        }
        response = test_client_for_search_api.get("/health")
        assert response.status_code == 200
        result = response.json()
        # No model loaded is OK because models are lazy-loaded
        assert result["status"] == "healthy"
        assert result["components"]["embedding"]["is_mock_mode"] is False
        assert result["components"]["embedding"]["note"] == "Embedding model loaded on first use"

        # Test with Qdrant error
        mock_qdrant_client.get.side_effect = Exception("Connection error")

        original_manager = search_state.model_manager
        try:
            search_state.model_manager = None
            response = test_client_for_search_api.get("/health")
            assert response.status_code == 503
        finally:
            search_state.model_manager = original_manager

    def test_health_endpoint_mock_mode(
        self, mock_settings, mock_qdrant_client, mock_model_manager, test_client_for_search_api
    ) -> None:
        """Test /health endpoint in mock mode."""
        mock_settings.USE_MOCK_EMBEDDINGS = True

        # Mock successful Qdrant response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": {"collections": [{"name": "col1"}]}}
        mock_qdrant_client.get.return_value = mock_response

        # Mock model manager status for mock mode
        mock_model_manager.get_status.return_value = {
            "embedding_model_loaded": True,
            "current_embedding_model": "mock_float32",
            "embedding_provider": "mock",
            "is_mock_mode": True,
            "provider_info": {
                "dimension": 384,
                "is_mock": True,
            },
        }

        search_state.qdrant_client = mock_qdrant_client

        response = test_client_for_search_api.get("/health")
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "healthy"
        assert result["components"]["embedding"]["status"] == "healthy"
        assert result["components"]["embedding"]["is_mock_mode"] is True
        assert result["components"]["embedding"]["provider"] == "mock"

    def test_search_post_endpoint(
        self, mock_settings, mock_qdrant_client, mock_model_manager, test_client_for_search_api
    ) -> None:
        """Test POST /search endpoint."""
        mock_settings.USE_MOCK_EMBEDDINGS = False
        # Mock collection info
        mock_response = Mock()
        mock_response.json.return_value = {"result": {"config": {"params": {"vectors": {"size": 1024}}}}}
        mock_response.raise_for_status = Mock()
        mock_qdrant_client.get.return_value = mock_response

        # Mock search results
        mock_search_response = Mock()
        mock_search_response.json.return_value = {
            "result": [
                {
                    "id": "1",
                    "score": 0.95,
                    "payload": {
                        "path": "/test/file1.txt",
                        "chunk_id": "chunk-1",
                        "doc_id": "doc-1",
                        "content": "Test content 1",
                    },
                }
            ]
        }
        mock_search_response.raise_for_status = Mock()
        mock_qdrant_client.post.return_value = mock_search_response

        # Mock search_qdrant function and metadata
        with (
            patch("vecpipe.search.service.search_qdrant") as mock_search,
            patch("qdrant_client.QdrantClient"),
            patch("shared.database.collection_metadata.get_collection_metadata") as mock_get_metadata,
        ):
            mock_get_metadata.return_value = None
            mock_search.return_value = [
                {
                    "id": "1",
                    "score": 0.95,
                    "payload": {
                        "path": "/test/file1.txt",
                        "chunk_id": "chunk-1",
                        "doc_id": "doc-1",
                        "content": "Test content 1",
                    },
                }
            ]

            response = test_client_for_search_api.post(
                "/search", json={"query": "test query", "k": 5, "search_type": "semantic"}
            )

            assert response.status_code == 200
            result = response.json()
            assert result["query"] == "test query"
            assert len(result["results"]) == 1
            assert result["results"][0]["score"] == 0.95
            assert result["model_used"] == "test-model/float32"

    def test_search_sparse_only_skips_dense_embedding_and_search(
        self, mock_settings, mock_qdrant_client, mock_model_manager, test_client_for_search_api
    ) -> None:
        """Sparse-only mode should not depend on dense embedding/vector search."""
        mock_settings.USE_MOCK_EMBEDDINGS = False

        # Mock collection info
        mock_response = Mock()
        mock_response.json.return_value = {"result": {"config": {"params": {"vectors": {"size": 1024}}}}}
        mock_response.raise_for_status = Mock()
        mock_qdrant_client.get.return_value = mock_response

        sparse_results = [{"chunk_id": "chunk-1", "score": 0.42}]
        payload_map = {
            "chunk-1": {
                "path": "/test/file1.txt",
                "chunk_id": "chunk-1",
                "doc_id": "doc-1",
                "content": "Test content 1",
            }
        }

        with (
            patch(
                "vecpipe.search.service._get_sparse_config_for_collection",
                new=AsyncMock(
                    return_value={"enabled": True, "plugin_id": "bm25-local", "sparse_collection_name": "sparse_test"}
                ),
            ),
            patch("vecpipe.search.service._perform_sparse_search", new=AsyncMock(return_value=(sparse_results, 12.34))),
            patch("vecpipe.search.service._fetch_payloads_for_chunk_ids", new=AsyncMock(return_value=payload_map)),
            patch("vecpipe.search.service.search_qdrant") as mock_search,
            patch("vecpipe.search.service.generate_embedding_async") as mock_embed,
            patch("qdrant_client.QdrantClient"),
            patch("shared.database.collection_metadata.get_collection_metadata", return_value=None),
        ):
            response = test_client_for_search_api.post(
                "/search",
                json={"query": "test query", "k": 5, "search_type": "semantic", "search_mode": "sparse"},
            )

            assert response.status_code == 200
            result = response.json()
            assert result["search_mode_used"] == "sparse"
            assert len(result["results"]) == 1
            assert result["results"][0]["chunk_id"] == "chunk-1"
            assert result["results"][0]["doc_id"] == "doc-1"

            assert mock_model_manager.generate_embedding_async.call_count == 0
            assert mock_search.call_count == 0
            assert mock_embed.call_count == 0

    def test_search_with_reranking(
        self, mock_settings, mock_qdrant_client, mock_model_manager, test_client_for_search_api
    ) -> None:
        """Test search with reranking enabled."""
        mock_settings.USE_MOCK_EMBEDDINGS = False

        with patch("vecpipe.search.service.get_reranker_for_embedding_model") as mock_get_reranker:
            mock_get_reranker.return_value = "test-reranker"

            # Mock collection info
            mock_response = Mock()
            mock_response.json.return_value = {"result": {"config": {"params": {"vectors": {"size": 1024}}}}}
            mock_response.raise_for_status = Mock()
            mock_qdrant_client.get.return_value = mock_response

            # Mock search results with more candidates for reranking
            mock_search_response = Mock()
            mock_search_response.json.return_value = {
                "result": [
                    {
                        "id": "1",
                        "score": 0.85,
                        "payload": {
                            "path": "/test/file1.txt",
                            "chunk_id": "chunk-1",
                            "doc_id": "doc-1",
                            "content": "Content 1",
                        },
                    },
                    {
                        "id": "2",
                        "score": 0.80,
                        "payload": {
                            "path": "/test/file2.txt",
                            "chunk_id": "chunk-2",
                            "doc_id": "doc-2",
                            "content": "Content 2",
                        },
                    },
                ]
            }
            mock_search_response.raise_for_status = Mock()
            mock_qdrant_client.post.return_value = mock_search_response

            # Mock metadata and search_qdrant
            with (
                patch("vecpipe.search.service.search_qdrant") as mock_search,
                patch("qdrant_client.QdrantClient"),
                patch("shared.database.collection_metadata.get_collection_metadata") as mock_get_metadata,
            ):
                mock_get_metadata.return_value = None
                mock_search.return_value = [
                    {
                        "id": "1",
                        "score": 0.85,
                        "payload": {
                            "path": "/test/file1.txt",
                            "chunk_id": "chunk-1",
                            "doc_id": "doc-1",
                            "content": "Content 1",
                        },
                    },
                    {
                        "id": "2",
                        "score": 0.80,
                        "payload": {
                            "path": "/test/file2.txt",
                            "chunk_id": "chunk-2",
                            "doc_id": "doc-2",
                            "content": "Content 2",
                        },
                    },
                ]

                response = test_client_for_search_api.post(
                    "/search",
                    json={"query": "test query", "k": 2, "use_reranker": True, "include_content": True},
                )

                assert response.status_code == 200
                result = response.json()
                assert result["reranking_used"] is True
                assert result["reranker_model"] == "test-reranker/float32"
                assert result["reranking_time_ms"] is not None
                # Reranked results should have updated scores
                assert result["results"][0]["score"] == 0.95
                assert result["results"][1]["score"] == 0.90

    def test_search_with_filters(
        self, mock_settings, mock_qdrant_client, mock_model_manager, test_client_for_search_api
    ) -> None:
        """Test search with metadata filters."""
        mock_settings.USE_MOCK_EMBEDDINGS = False

        # Mock filtered search results
        mock_search_response = Mock()
        mock_search_response.json.return_value = {
            "result": [
                {
                    "id": "1",
                    "score": 0.90,
                    "payload": {"path": "/filtered/file.txt", "chunk_id": "chunk-1", "doc_id": "doc-1"},
                }
            ]
        }
        mock_search_response.raise_for_status = Mock()
        mock_qdrant_client.post.return_value = mock_search_response

        response = test_client_for_search_api.post(
            "/search",
            json={
                "query": "test query",
                "k": 5,
                "filters": {"must": [{"key": "type", "match": {"value": "document"}}]},
            },
        )

        assert response.status_code == 200
        result = response.json()
        assert len(result["results"]) == 1
        assert result["results"][0]["path"] == "/filtered/file.txt"

        # Verify filter was passed to Qdrant
        call_args = mock_qdrant_client.post.call_args
        assert "filter" in call_args[1]["json"]

    def test_search_error_handling(
        self, mock_settings, mock_qdrant_client, mock_model_manager, test_client_for_search_api
    ) -> None:
        """Test search error handling scenarios."""
        mock_settings.USE_MOCK_EMBEDDINGS = False

        # Mock collection info first (successful)
        mock_response = Mock()
        mock_response.json.return_value = {"result": {"config": {"params": {"vectors": {"size": 1024}}}}}
        mock_response.raise_for_status = Mock()
        mock_qdrant_client.get.return_value = mock_response

        # Mock metadata lookup
        with (
            patch("qdrant_client.QdrantClient"),
            patch("shared.database.collection_metadata.get_collection_metadata") as mock_get_metadata,
        ):
            mock_get_metadata.return_value = None

            # Test Qdrant HTTP error during search
            with patch("vecpipe.search.service.search_qdrant") as mock_search:
                mock_search.side_effect = httpx.HTTPStatusError(
                    "Bad request", request=Mock(), response=Mock(status_code=400)
                )

                response = test_client_for_search_api.post("/search", json={"query": "test", "k": 5})

                assert response.status_code == 502
                assert "Vector database error" in response.json()["detail"]

            # Test embedding generation error
            mock_model_manager.generate_embedding_async.side_effect = RuntimeError("Model load failed")

            response = test_client_for_search_api.post("/search", json={"query": "test", "k": 5})

            assert response.status_code == 503
            assert "Embedding service error" in response.json()["detail"]

    # NOTE: test_hybrid_search_endpoint removed - legacy /hybrid_search endpoint deleted

    def test_batch_search_endpoint(
        self, mock_settings, mock_qdrant_client, mock_model_manager, test_client_for_search_api
    ) -> None:
        """Test /search/batch endpoint."""
        mock_settings.USE_MOCK_EMBEDDINGS = False

        with patch("vecpipe.search.service.search_qdrant") as mock_search:
            # Mock search results for each query
            mock_search.return_value = [
                {"score": 0.9, "payload": {"path": "/test/batch.txt", "chunk_id": "chunk-1", "doc_id": "doc-1"}}
            ]

            response = test_client_for_search_api.post(
                "/search/batch", json={"queries": ["query1", "query2", "query3"], "k": 5, "search_type": "semantic"}
            )

            assert response.status_code == 200
            result = response.json()
            assert len(result["responses"]) == 3
            assert all(r["query"] in ["query1", "query2", "query3"] for r in result["responses"])
            assert result["total_time_ms"] is not None

            # Verify embeddings were generated for all queries
            assert mock_model_manager.generate_embedding_async.call_count == 3

    # NOTE: test_keyword_search_endpoint removed - use search_mode="sparse" instead

    def test_collection_info_endpoint(self, mock_qdrant_client, test_client_for_search_api) -> None:
        """Test /collection/info endpoint."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "result": {"name": "test_collection", "points_count": 1000, "indexed_vectors_count": 1000}
        }
        mock_response.raise_for_status = Mock()
        mock_qdrant_client.get.return_value = mock_response

        search_state.qdrant_client = mock_qdrant_client

        response = test_client_for_search_api.get("/collection/info")

        assert response.status_code == 200
        result = response.json()
        assert result["name"] == "test_collection"
        assert result["points_count"] == 1000

    def test_list_models_endpoint(self, test_client_for_search_api, mock_model_manager) -> None:
        """Test /models endpoint returns built-in models with provider info.

        Note: Current model info now comes from ModelManager.get_status() using the
        current_embedding_model key which is in format "model_name_quantization".
        """
        mock_models = [
            {
                "model_name": "test-model",
                "name": "test-model",
                "description": "Test embedding model",
                "dimension": 768,
                "supports_quantization": True,
                "recommended_quantization": "float16",
                "memory_estimate": {"float32": 1024, "float16": 512},
                "provider": "dense_local",
            }
        ]

        with patch(
            "shared.embedding.factory.get_all_supported_models",
            return_value=mock_models,
        ):
            # Mock model manager status with current model key
            mock_model_manager.get_status.return_value = {
                "embedding_model_loaded": True,
                "current_embedding_model": "test-model_float32",
                "embedding_provider": "dense_local",
            }

            response = test_client_for_search_api.get("/models")
            assert response.status_code == 200
            result = response.json()

            assert len(result["models"]) == 1
            model = result["models"][0]

            # Verify existing fields (backward compatibility)
            assert model["name"] == "test-model"
            assert model["dimension"] == 768
            assert model["description"] == "Test embedding model"
            assert model["supports_quantization"] is True
            assert model["recommended_quantization"] == "float16"

            # Verify new plugin-aware fields
            assert model["provider_id"] == "dense_local"
            assert model["is_plugin"] is False

            # Verify current model state
            assert result["current_model"] == "test-model"
            assert result["current_quantization"] == "float32"

    def test_list_models_includes_plugin_models(self, test_client_for_search_api, mock_model_manager) -> None:
        """Test /models endpoint includes plugin models with is_plugin=True."""
        mock_models = [
            {
                "model_name": "builtin-model",
                "name": "builtin-model",
                "description": "Built-in model",
                "dimension": 768,
                "provider": "dense_local",
            },
            {
                "model_name": "plugin-vendor/custom-model",
                "name": "plugin-vendor/custom-model",
                "description": "Custom plugin model",
                "dimension": 1024,
                "provider": "custom_plugin_provider",
            },
        ]

        with patch(
            "shared.embedding.factory.get_all_supported_models",
            return_value=mock_models,
        ):
            mock_model_manager.get_status.return_value = {
                "embedding_model_loaded": False,
                "current_embedding_model": None,
            }

            response = test_client_for_search_api.get("/models")
            assert response.status_code == 200
            result = response.json()

            assert len(result["models"]) == 2

            # Find models by name
            builtin = next(m for m in result["models"] if m["name"] == "builtin-model")
            plugin = next(m for m in result["models"] if m["name"] == "plugin-vendor/custom-model")

            # Verify built-in model
            assert builtin["provider_id"] == "dense_local"
            assert builtin["is_plugin"] is False

            # Verify plugin model
            assert plugin["provider_id"] == "custom_plugin_provider"
            assert plugin["is_plugin"] is True
            assert plugin["dimension"] == 1024

    def test_embed_endpoint(self, mock_model_manager, test_client_for_search_api) -> None:
        """Test /embed endpoint."""
        response = test_client_for_search_api.post(
            "/embed",
            json={
                "texts": ["text1", "text2", "text3"],
                "model_name": "test-model",
                "quantization": "float32",
                "batch_size": 2,
            },
        )

        assert response.status_code == 200
        result = response.json()
        assert len(result["embeddings"]) == 3
        assert result["model_used"] == "test-model/float32"
        assert result["batch_count"] == 2  # 3 texts with batch_size=2
        assert result["embedding_time_ms"] is not None

    def test_embed_memory_error(self, mock_model_manager, test_client_for_search_api) -> None:
        """Test /embed endpoint with memory error."""

        mock_model_manager.generate_embeddings_batch_async.side_effect = InsufficientMemoryError(
            "Not enough GPU memory"
        )

        response = test_client_for_search_api.post(
            "/embed", json={"texts": ["text1"], "model_name": "large-model", "quantization": "float32"}
        )

        assert response.status_code == 507
        # The detail is a dict with 'error' key
        detail = response.json()["detail"]
        assert isinstance(detail, dict)
        assert detail["error"] == "insufficient_memory"

    def test_upsert_endpoint(self, mock_qdrant_client, test_client_for_search_api) -> None:
        """Test /upsert endpoint."""
        # Mock collection info first
        mock_get_response = Mock()
        mock_get_response.json.return_value = {"result": {"config": {"params": {"vectors": {"size": 768}}}}}
        mock_get_response.raise_for_status = Mock()
        mock_qdrant_client.get.return_value = mock_get_response

        # Mock upsert response
        mock_put_response = Mock()
        mock_put_response.raise_for_status = Mock()
        mock_qdrant_client.put.return_value = mock_put_response

        response = test_client_for_search_api.post(
            "/upsert",
            json={
                "collection_name": "test_collection",
                "points": [
                    {
                        "id": "point-1",
                        "vector": [0.1] * 768,
                        "payload": {
                            "doc_id": "doc-1",
                            "chunk_id": "chunk-1",
                            "path": "/test/file.txt",
                            "content": "Test content",
                            "metadata": {"type": "document"},
                        },
                    }
                ],
                "wait": True,
            },
        )

        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "success"
        assert result["points_upserted"] == 1
        assert result["collection_name"] == "test_collection"
        assert result["upsert_time_ms"] is not None

        # Verify the request format
        call_args = mock_qdrant_client.put.call_args
        assert "points" in call_args[1]["json"]
        # wait should be a query parameter, not in the JSON body
        assert "wait" not in call_args[1]["json"]
        # Verify URL contains wait query parameter
        url_called = call_args[0][0]
        assert "?wait=true" in url_called

    @pytest.mark.asyncio()
    async def test_upsert_error_handling(self, mock_qdrant_client) -> None:
        """Test /upsert endpoint error handling."""
        with patch("vecpipe.search.state.qdrant_client", mock_qdrant_client):
            # Mock collection info first
            mock_get_response = Mock()
            mock_get_response.json.return_value = {"result": {"config": {"params": {"vectors": {"size": 768}}}}}
            mock_get_response.raise_for_status = Mock()
            mock_qdrant_client.get.return_value = mock_get_response

            # Mock Qdrant error response for PUT
            error_response = Mock()
            error_response.json.return_value = {"status": {"error": "Collection not found"}}
            mock_qdrant_client.put.side_effect = httpx.HTTPStatusError(
                "Not found", request=Mock(), response=error_response
            )

            request = UpsertRequest(
                collection_name="nonexistent",
                points=[
                    UpsertPoint(
                        id="point-1",
                        vector=[0.1] * 768,
                        payload=PointPayload(doc_id="doc-1", chunk_id="chunk-1", path="/test/file.txt"),
                    )
                ],
            )

            with pytest.raises(HTTPException) as exc_info:
                await upsert_points(request)
            assert exc_info.value.status_code == 502
            assert "Collection not found" in str(exc_info.value.detail)

    def test_suggest_models_endpoint(self, test_client_for_search_api, mock_model_manager) -> None:
        """Test /models/suggest endpoint."""
        with (
            patch("vecpipe.memory_utils.get_gpu_memory_info") as mock_gpu_info,
            patch("vecpipe.memory_utils.suggest_model_configuration") as mock_suggest,
        ):
            # Test with GPU available
            mock_gpu_info.return_value = (8000, 16000)  # 8GB free, 16GB total
            mock_suggest.return_value = {
                "embedding_model": "large-model",
                "embedding_quantization": "float16",
                "reranker_model": "reranker-model",
                "reranker_quantization": "int8",
                "notes": ["GPU detected with sufficient memory"],
            }
            mock_model_manager.current_model_key = "current-model"
            mock_model_manager.current_reranker_key = None

            response = test_client_for_search_api.get("/models/suggest")
            assert response.status_code == 200
            result = response.json()
            assert result["gpu_available"] is True
            assert result["gpu_memory"]["free_mb"] == 8000
            assert result["gpu_memory"]["usage_percent"] == 50.0
            assert result["suggestion"]["embedding_model"] == "large-model"

            # Test without GPU
            mock_gpu_info.return_value = (0, 0)
            response = test_client_for_search_api.get("/models/suggest")
            assert response.status_code == 200
            result = response.json()
            assert result["gpu_available"] is False
            assert "No GPU detected" in result["message"]

    def test_embedding_info_endpoint(self, mock_settings, mock_model_manager, test_client_for_search_api) -> None:
        """Test /embedding/info endpoint with model loaded."""
        mock_settings.USE_MOCK_EMBEDDINGS = False

        response = test_client_for_search_api.get("/embedding/info")
        assert response.status_code == 200
        result = response.json()
        assert result["mode"] == "real"
        assert result["available"] is True
        assert result["is_mock_mode"] is False
        assert result["current_model"] == "test-model"
        assert result["quantization"] == "float32"
        assert result["device"] == "cpu"
        assert result["provider"] == "dense_local"
        assert result["dimension"] == 1024
        assert "model_details" in result
        assert result["model_details"]["dimension"] == 1024

    def test_embedding_info_endpoint_lazy_loading(
        self, mock_settings, mock_model_manager, test_client_for_search_api
    ) -> None:
        """Test /embedding/info endpoint when model not yet loaded (lazy loading)."""
        mock_settings.USE_MOCK_EMBEDDINGS = False

        # Simulate model not loaded yet
        mock_model_manager.get_status.return_value = {
            "embedding_model_loaded": False,
            "current_embedding_model": None,
            "embedding_provider": None,
            "is_mock_mode": False,
        }

        response = test_client_for_search_api.get("/embedding/info")
        assert response.status_code == 200
        result = response.json()
        assert result["mode"] == "real"
        assert result["available"] is True  # Available even if not loaded (capability-based)
        assert result["is_mock_mode"] is False
        assert result["note"] == "Embedding model loaded on first use"
        assert result["default_model"] == "test-model"
        assert result["default_quantization"] == "float32"
        assert "current_model" not in result  # Not loaded yet

    def test_embedding_info_endpoint_mock_mode(
        self, mock_settings, mock_model_manager, test_client_for_search_api
    ) -> None:
        """Test /embedding/info endpoint in mock mode."""
        mock_settings.USE_MOCK_EMBEDDINGS = True

        # Simulate mock mode with mock provider loaded
        mock_model_manager.get_status.return_value = {
            "embedding_model_loaded": True,
            "current_embedding_model": "mock_float32",
            "embedding_provider": "mock",
            "is_mock_mode": True,
            "provider_info": {
                "model_name": "mock",
                "dimension": 384,
                "device": "cpu",
                "quantization": "float32",
                "is_mock": True,
            },
        }

        response = test_client_for_search_api.get("/embedding/info")
        assert response.status_code == 200
        result = response.json()
        assert result["mode"] == "mock"
        assert result["available"] is True
        assert result["is_mock_mode"] is True
        assert result["provider"] == "mock"
        assert result["dimension"] == 384

    def test_embedding_info_endpoint_no_model_manager(self, mock_settings, test_client_for_search_api) -> None:
        """Test /embedding/info endpoint when ModelManager is not initialized."""
        mock_settings.USE_MOCK_EMBEDDINGS = False

        original_manager = search_state.model_manager
        search_state.model_manager = None

        try:
            response = test_client_for_search_api.get("/embedding/info")
            assert response.status_code == 200
            result = response.json()
            assert result["mode"] == "real"
            assert result["available"] is False  # Not available without ModelManager
            assert result["is_mock_mode"] is False
        finally:
            search_state.model_manager = original_manager

    def test_search_with_collection_metadata(
        self, mock_settings, mock_qdrant_client, mock_model_manager, test_client_for_search_api
    ) -> None:
        """Test search with collection metadata for model selection."""
        from vecpipe.search.cache import clear_cache

        mock_settings.USE_MOCK_EMBEDDINGS = False

        # Clear cache to ensure metadata fetch happens
        clear_cache()

        with patch(
            "shared.database.collection_metadata.get_collection_metadata_async",
            new_callable=AsyncMock,
        ) as mock_get_metadata:
            # Mock collection metadata (AsyncMock provides awaitable return)
            mock_get_metadata.return_value = {
                "model_name": "collection-model",
                "quantization": "float16",
                "instruction": "Custom instruction",
            }

            # Mock collection info
            mock_response = Mock()
            mock_response.json.return_value = {"result": {"config": {"params": {"vectors": {"size": 768}}}}}
            mock_response.raise_for_status = Mock()
            mock_qdrant_client.get.return_value = mock_response

            # Make sure model manager returns embeddings with correct dimension (768)
            mock_model_manager.generate_embedding.return_value = [0.1] * 768
            mock_model_manager.generate_embedding_async.return_value = [0.1] * 768

            # Mock search results
            mock_search_response = Mock()
            mock_search_response.json.return_value = {"result": []}
            mock_search_response.raise_for_status = Mock()
            mock_qdrant_client.post.return_value = mock_search_response

            # Mock search_qdrant
            with patch("vecpipe.search.service.search_qdrant") as mock_search:
                mock_search.return_value = []

                response = test_client_for_search_api.post("/search", json={"query": "test", "k": 5})

                assert response.status_code == 200

                # Verify collection metadata was fetched (now uses cache + shared client)
                mock_get_metadata.assert_called()
                # And that the response was successful
                assert response.status_code == 200

    def test_search_get_endpoint(
        self, mock_settings, mock_qdrant_client, mock_model_manager, test_client_for_search_api
    ) -> None:
        """Test GET /search endpoint (compatibility)."""
        mock_settings.USE_MOCK_EMBEDDINGS = False

        # Mock collection info
        mock_response = Mock()
        mock_response.json.return_value = {"result": {"config": {"params": {"vectors": {"size": 1024}}}}}
        mock_response.raise_for_status = Mock()
        mock_qdrant_client.get.return_value = mock_response

        # Mock search results
        mock_search_response = Mock()
        mock_search_response.json.return_value = {"result": []}
        mock_search_response.raise_for_status = Mock()
        mock_qdrant_client.post.return_value = mock_search_response

        # Mock metadata and search_qdrant
        with (
            patch("qdrant_client.QdrantClient"),
            patch("shared.database.collection_metadata.get_collection_metadata") as mock_get_metadata,
        ):
            mock_get_metadata.return_value = None

            with patch("vecpipe.search.service.search_qdrant") as mock_search:
                mock_search.return_value = []

                response = test_client_for_search_api.get(
                    "/search",
                    params={
                        "q": "test query",
                        "k": 10,
                        "collection": "custom_collection",
                        "search_type": "question",
                        "model_name": "custom-model",
                        "quantization": "int8",
                    },
                )

                assert response.status_code == 200
                result = response.json()
                assert result["query"] == "test query"

                # Verify the parameters were passed through
                call_args = mock_model_manager.generate_embedding_async.call_args
                assert call_args[0][1] == "custom-model"  # model_name
                assert call_args[0][2] == "int8"  # quantization

    def test_load_model_endpoint_mock_mode(self, mock_settings, test_client_for_search_api) -> None:
        """Test /models/load endpoint in mock mode."""
        mock_settings.USE_MOCK_EMBEDDINGS = True

        response = test_client_for_search_api.post(
            "/models/load", json={"model_name": "test-model", "quantization": "float32"}
        )

        assert response.status_code == 400
        assert "Cannot load models when using mock embeddings" in response.json()["detail"]

    @pytest.mark.asyncio()
    async def test_generate_embedding_async_mock_mode(self, mock_settings) -> None:
        """Test generate_embedding_async in mock mode."""
        mock_settings.USE_MOCK_EMBEDDINGS = True

        # Call the function directly with proper mock settings
        with patch("vecpipe.search.service.settings", mock_settings):
            embedding = await search_api_module.generate_embedding_async("test text")

        assert len(embedding) == 1024
        assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.skip(reason="Model manager patching is complex due to module state")
    @pytest.mark.asyncio()
    async def test_generate_embedding_async_error(self, mock_settings, mock_model_manager) -> None:
        """Test generate_embedding_async error handling."""
        mock_settings.USE_MOCK_EMBEDDINGS = False

        # Mock the generate_embedding_async function to return None
        with patch("vecpipe.search_api.model_manager", mock_model_manager):
            mock_model_manager.generate_embedding_async.return_value = None

            with pytest.raises(RuntimeError) as exc_info:
                await search_api_module.generate_embedding_async("test text")
            assert "Failed to generate embedding" in str(exc_info.value)

    @pytest.mark.skip(reason="Reranking error path requires complex mocking")
    def test_reranking_memory_error(
        self, mock_settings, mock_qdrant_client, mock_model_manager, test_client_for_search_api
    ) -> None:
        """Test reranking with insufficient memory."""

        mock_settings.USE_MOCK_EMBEDDINGS = False

        with patch("vecpipe.search.service.get_reranker_for_embedding_model") as mock_get_reranker:
            mock_get_reranker.return_value = "test-reranker"

            # Mock collection info
            mock_response = Mock()
            mock_response.json.return_value = {"result": {"config": {"params": {"vectors": {"size": 1024}}}}}
            mock_response.raise_for_status = Mock()
            mock_qdrant_client.get.return_value = mock_response

            # Mock search results
            mock_search_response = Mock()
            mock_search_response.json.return_value = {
                "result": [
                    {
                        "id": "1",
                        "score": 0.85,
                        "payload": {
                            "path": "/test/file.txt",
                            "chunk_id": "chunk-1",
                            "doc_id": "doc-1",
                            "content": "Content",
                        },
                    }
                ]
            }
            mock_search_response.raise_for_status = Mock()
            mock_qdrant_client.post.return_value = mock_search_response

            # Mock metadata and search_qdrant
            with (
                patch("qdrant_client.QdrantClient"),
                patch("shared.database.collection_metadata.get_collection_metadata") as mock_get_metadata,
            ):
                mock_get_metadata.return_value = None

                with patch("vecpipe.search.service.search_qdrant") as mock_search:
                    mock_search.return_value = [
                        {
                            "id": "1",
                            "score": 0.85,
                            "payload": {
                                "path": "/test/file.txt",
                                "chunk_id": "chunk-1",
                                "doc_id": "doc-1",
                                "content": "Content",
                            },
                        }
                    ]

                    # Mock reranking to raise memory error
                    mock_model_manager.rerank_async.side_effect = InsufficientMemoryError(
                        "Not enough GPU memory for reranking"
                    )

                    response = test_client_for_search_api.post(
                        "/search", json={"query": "test query", "k": 1, "use_reranker": True}
                    )

                    assert response.status_code == 507
                    # The detail is a dict with 'error' key
                    detail = response.json()["detail"]
                    assert isinstance(detail, dict)
                    assert detail["error"] == "insufficient_memory"


class TestCollectionResolution:
    """Test collection name resolution with operation_uuid."""

    @pytest.mark.asyncio()
    async def test_explicit_collection_takes_priority(self) -> None:
        """Explicit collection should override operation_uuid."""
        from vecpipe.search.service import resolve_collection_name

        with patch("vecpipe.search.service._lookup_collection_from_operation") as mock_lookup:
            result = await resolve_collection_name(
                request_collection="explicit_collection",
                operation_uuid="some-uuid",
                default_collection="default_collection",
            )
            assert result == "explicit_collection"
            mock_lookup.assert_not_called()

    @pytest.mark.asyncio()
    async def test_operation_uuid_lookup_success(self) -> None:
        """operation_uuid should resolve to collection when found."""
        from vecpipe.search.service import resolve_collection_name

        with patch("vecpipe.search.service._lookup_collection_from_operation") as mock_lookup:
            mock_lookup.return_value = "operation_collection"
            result = await resolve_collection_name(
                request_collection=None,
                operation_uuid="valid-uuid",
                default_collection="default",
            )
            assert result == "operation_collection"
            mock_lookup.assert_called_once_with("valid-uuid")

    @pytest.mark.asyncio()
    async def test_operation_uuid_not_found_raises_404(self) -> None:
        """Missing operation should raise 404 when operation_uuid provided."""
        from vecpipe.search.service import resolve_collection_name

        with patch("vecpipe.search.service._lookup_collection_from_operation") as mock_lookup:
            mock_lookup.return_value = None
            with pytest.raises(HTTPException) as exc:
                await resolve_collection_name(
                    request_collection=None,
                    operation_uuid="nonexistent-uuid",
                    default_collection="default",
                )
            assert exc.value.status_code == 404
            assert "nonexistent-uuid" in str(exc.value.detail)

    @pytest.mark.asyncio()
    async def test_default_fallback_when_no_operation_uuid(self) -> None:
        """Default should be used when no collection or operation_uuid."""
        from vecpipe.search.service import resolve_collection_name

        result = await resolve_collection_name(
            request_collection=None,
            operation_uuid=None,
            default_collection="default_collection",
        )
        assert result == "default_collection"

    @pytest.mark.asyncio()
    async def test_lookup_collection_from_operation_returns_none_when_not_found(self) -> None:
        """Should return None when operation has no collection."""
        from vecpipe.search.service import _lookup_collection_from_operation

        # The function returns None on any exception, which includes when DB is not available
        # This tests the graceful fallback behavior
        result = await _lookup_collection_from_operation("nonexistent-uuid")
        # Should return None (not raise) due to error handling
        assert result is None

    @pytest.mark.asyncio()
    async def test_lookup_collection_returns_none_on_error(self) -> None:
        """Should return None and log warning on database error."""
        from vecpipe.search.service import _lookup_collection_from_operation

        with patch(
            "shared.database.database.ensure_async_sessionmaker",
            side_effect=Exception("Database connection failed"),
        ):
            result = await _lookup_collection_from_operation("test-uuid")
            assert result is None


# NOTE: TestHybridSearchRouting class removed - legacy hybrid search deleted
# Use search_mode="hybrid" with RRF fusion instead


class TestScoreThresholdFiltering:
    """Test score_threshold filtering in perform_search."""

    def test_score_threshold_filters_low_scores(
        self, mock_qdrant_client, mock_model_manager, test_client_for_search_api
    ) -> None:
        """Results below score_threshold should be excluded."""
        # Mock collection info
        mock_get_response = Mock()
        mock_get_response.json.return_value = {"result": {"config": {"params": {"vectors": {"size": 1024}}}}}
        mock_get_response.raise_for_status = Mock()
        mock_qdrant_client.get.return_value = mock_get_response

        with (
            patch("vecpipe.search.service._get_cached_collection_metadata") as mock_meta,
            patch("vecpipe.search.service.search_qdrant") as mock_search,
        ):
            mock_meta.return_value = {"model_name": "test-model", "quantization": "float32"}
            mock_search.return_value = [
                {"score": 0.9, "payload": {"path": "/a", "chunk_id": "1", "doc_id": "d1"}},
                {"score": 0.5, "payload": {"path": "/b", "chunk_id": "2", "doc_id": "d2"}},
                {"score": 0.3, "payload": {"path": "/c", "chunk_id": "3", "doc_id": "d3"}},
            ]

            response = test_client_for_search_api.post(
                "/search",
                json={"query": "test", "k": 10, "score_threshold": 0.6},
            )

            assert response.status_code == 200
            result = response.json()
            assert result["num_results"] == 1
            assert len(result["results"]) == 1
            assert result["results"][0]["score"] == 0.9

    def test_score_threshold_zero_includes_all(
        self, mock_qdrant_client, mock_model_manager, test_client_for_search_api
    ) -> None:
        """score_threshold=0.0 (default) should include all results."""
        # Mock collection info
        mock_get_response = Mock()
        mock_get_response.json.return_value = {"result": {"config": {"params": {"vectors": {"size": 1024}}}}}
        mock_get_response.raise_for_status = Mock()
        mock_qdrant_client.get.return_value = mock_get_response

        with (
            patch("vecpipe.search.service._get_cached_collection_metadata") as mock_meta,
            patch("vecpipe.search.service.search_qdrant") as mock_search,
        ):
            mock_meta.return_value = {"model_name": "test-model", "quantization": "float32"}
            mock_search.return_value = [
                {"score": 0.9, "payload": {"path": "/a", "chunk_id": "1", "doc_id": "d1"}},
                {"score": 0.1, "payload": {"path": "/b", "chunk_id": "2", "doc_id": "d2"}},
            ]

            response = test_client_for_search_api.post(
                "/search",
                json={"query": "test", "k": 10, "score_threshold": 0.0},
            )

            assert response.status_code == 200
            result = response.json()
            assert result["num_results"] == 2


class TestUpsertWaitParameter:
    """Test upsert wait parameter is passed as query param."""

    def test_upsert_wait_false_omits_query_parameter(self, mock_qdrant_client, test_client_for_search_api) -> None:
        """wait=False should not add query parameter."""
        # Mock collection info
        mock_get_response = Mock()
        mock_get_response.json.return_value = {"result": {"config": {"params": {"vectors": {"size": 768}}}}}
        mock_get_response.raise_for_status = Mock()
        mock_qdrant_client.get.return_value = mock_get_response

        # Mock upsert response
        mock_put_response = Mock()
        mock_put_response.raise_for_status = Mock()
        mock_qdrant_client.put.return_value = mock_put_response

        response = test_client_for_search_api.post(
            "/upsert",
            json={
                "collection_name": "test_collection",
                "points": [
                    {
                        "id": "point-1",
                        "vector": [0.1] * 768,
                        "payload": {
                            "doc_id": "doc-1",
                            "chunk_id": "chunk-1",
                            "path": "/test/file.txt",
                        },
                    }
                ],
                "wait": False,
            },
        )

        assert response.status_code == 200

        # Verify URL does NOT contain wait query parameter
        call_args = mock_qdrant_client.put.call_args
        url_called = call_args[0][0]
        assert "?wait=" not in url_called
        # Also verify wait is not in body
        assert "wait" not in call_args[1]["json"]
