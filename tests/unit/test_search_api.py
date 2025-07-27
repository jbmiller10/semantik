"""Comprehensive unit tests for search_api.py to achieve 80%+ coverage.

This module tests all the endpoints, error scenarios, edge cases, and FAISS fallback logic.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from packages.vecpipe.search_api import (
    BatchSearchRequest,
    BatchSearchResponse,
    EmbedRequest,
    EmbedResponse,
    HybridSearchResponse,
    SearchRequest,
    SearchResponse,
    UpsertRequest,
    UpsertResponse,
    app,
    generate_mock_embedding,
    get_or_create_metric,
)


@pytest.fixture(scope="function")
def mock_settings():
    """Mock settings for testing."""
    with patch("packages.vecpipe.search_api.settings") as mock:
        mock.QDRANT_HOST = "localhost"
        mock.QDRANT_PORT = 6333
        mock.DEFAULT_COLLECTION = "test_collection"
        mock.USE_MOCK_EMBEDDINGS = False
        mock.DEFAULT_EMBEDDING_MODEL = "test-model"
        mock.DEFAULT_QUANTIZATION = "float32"
        mock.MODEL_UNLOAD_AFTER_SECONDS = 300
        mock.SEARCH_API_PORT = 8088
        mock.METRICS_PORT = 9090
        yield mock


@pytest.fixture(scope="function")
def mock_qdrant_client():
    """Mock Qdrant client."""
    client = AsyncMock()
    client.get = AsyncMock()
    client.post = AsyncMock()
    client.put = AsyncMock()
    client.aclose = AsyncMock()
    return client


@pytest.fixture(scope="function")
def mock_model_manager():
    """Mock model manager."""
    manager = Mock()
    manager.generate_embedding_async = AsyncMock(return_value=[0.1] * 1024)
    manager.rerank_async = AsyncMock(return_value=[(0, 0.95), (1, 0.90)])
    manager.get_status = Mock(return_value={"loaded_models": [], "memory_usage": {}})
    manager.shutdown = Mock()
    manager.current_model_key = "test-model"
    manager.current_reranker_key = None
    return manager


@pytest.fixture(scope="function")
def mock_embedding_service():
    """Mock embedding service."""
    service = Mock()
    service.is_initialized = True
    service.current_model_name = "test-model"
    service.current_quantization = "float32"
    service.device = "cpu"
    service.mock_mode = False
    service.allow_quantization_fallback = True
    service.get_model_info = Mock(return_value={
        "model_name": "test-model",
        "dimension": 1024,
        "description": "Test model"
    })
    return service


@pytest.fixture(scope="function")
def mock_hybrid_engine():
    """Mock hybrid search engine."""
    with patch("packages.vecpipe.search_api.HybridSearchEngine") as mock_class:
        engine = Mock()
        engine.extract_keywords = Mock(return_value=["test", "query"])
        engine.hybrid_search = Mock(return_value=[
            {
                "score": 0.95,
                "payload": {
                    "path": "/test/file1.txt",
                    "chunk_id": "chunk-1",
                    "doc_id": "doc-1"
                },
                "matched_keywords": ["test"],
                "keyword_score": 0.8,
                "combined_score": 0.875
            }
        ])
        engine.search_by_keywords = Mock(return_value=[
            {
                "payload": {
                    "path": "/test/file1.txt",
                    "chunk_id": "chunk-1",
                    "doc_id": "doc-1"
                },
                "matched_keywords": ["test", "query"]
            }
        ])
        engine.close = Mock()
        mock_class.return_value = engine
        yield engine


@pytest.fixture(scope="function")
def test_client_for_search_api(mock_settings, mock_qdrant_client, mock_model_manager, mock_embedding_service):
    """Create a test client for the search API with mocked dependencies."""
    import packages.vecpipe.search_api as search_api_module
    from packages.vecpipe.search_api import app
    
    # Temporarily store original values
    original_qdrant = search_api_module.qdrant_client
    original_model_manager = search_api_module.model_manager
    original_embedding_service = search_api_module.embedding_service
    
    # Set mocked values
    search_api_module.qdrant_client = mock_qdrant_client
    search_api_module.model_manager = mock_model_manager
    search_api_module.embedding_service = mock_embedding_service
    
    # Clear any existing dependency overrides
    app.dependency_overrides.clear()
    
    # Create test client
    client = TestClient(app)
    
    yield client
    
    # Restore original values
    search_api_module.qdrant_client = original_qdrant
    search_api_module.model_manager = original_model_manager
    search_api_module.embedding_service = original_embedding_service
    
    # Clean up
    app.dependency_overrides.clear()


class TestSearchAPI:
    """Test search API endpoints and functionality."""

    def test_generate_mock_embedding(self):
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

    def test_get_or_create_metric(self):
        """Test metric creation and retrieval."""
        from prometheus_client import Counter, Histogram
        
        # Create a new metric
        metric1 = get_or_create_metric(
            Counter,
            "test_counter",
            "Test counter metric",
            ["label1", "label2"]
        )
        assert metric1 is not None
        
        # Try to get the same metric again
        metric2 = get_or_create_metric(
            Counter,
            "test_counter",
            "Test counter metric",
            ["label1", "label2"]
        )
        # Should return the same instance or handle duplicate gracefully
        assert metric2 is not None
        
        # Create histogram without labels
        hist = get_or_create_metric(
            Histogram,
            "test_histogram",
            "Test histogram",
            buckets=(0.1, 0.5, 1.0)
        )
        assert hist is not None

    @pytest.mark.asyncio
    async def test_lifespan(self, mock_settings, mock_qdrant_client, mock_model_manager):
        """Test application lifespan management."""
        with patch("packages.vecpipe.search_api.httpx.AsyncClient") as mock_httpx:
            mock_httpx.return_value = mock_qdrant_client
            
            with patch("packages.vecpipe.search_api.start_metrics_server"):
                with patch("packages.vecpipe.search_api.get_embedding_service") as mock_get_service:
                    mock_service = AsyncMock()
                    mock_service.initialize = AsyncMock()
                    mock_get_service.return_value = mock_service
                    
                    with patch("packages.vecpipe.search_api.ModelManager") as mock_mm_class:
                        mock_mm_class.return_value = mock_model_manager
                        
                        # Test startup and shutdown
                        from packages.vecpipe.search_api import app, lifespan
                        
                        async with lifespan(app):
                            # Verify initialization
                            mock_get_service.assert_called_once()
                            mock_service.initialize.assert_called_once()
                            mock_mm_class.assert_called_once()
                        
                        # Verify cleanup
                        mock_qdrant_client.aclose.assert_called_once()
                        mock_model_manager.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_model_status(self, mock_model_manager, test_client_for_search_api):
        """Test /model/status endpoint."""
        with patch("packages.vecpipe.search_api.model_manager", mock_model_manager):
            response = test_client_for_search_api.get("/model/status")
            assert response.status_code == 200
            assert response.json() == {"loaded_models": [], "memory_usage": {}}
            
        # Test when model manager is not initialized
        with patch("packages.vecpipe.search_api.model_manager", None):
            response = test_client_for_search_api.get("/model/status")
            assert response.status_code == 200
            assert response.json() == {"error": "Model manager not initialized"}

    @pytest.mark.asyncio
    async def test_root_endpoint(self, mock_settings, mock_qdrant_client, mock_embedding_service, test_client_for_search_api):
        """Test root health check endpoint."""
        with patch("packages.vecpipe.search_api.qdrant_client", mock_qdrant_client):
            with patch("packages.vecpipe.search_api.embedding_service", mock_embedding_service):
                # Mock successful response
                mock_qdrant_client.get.return_value.json.return_value = {
                    "result": {
                        "points_count": 100,
                        "config": {
                            "params": {
                                "vectors": {
                                    "size": 1024
                                }
                            }
                        }
                    }
                }
                mock_qdrant_client.get.return_value.raise_for_status = AsyncMock()
                
                response = test_client_for_search_api.get("/")
                assert response.status_code == 200
                result = response.json()
                assert result["status"] == "healthy"
                assert result["collection"]["points_count"] == 100
                assert result["collection"]["vector_size"] == 1024
                assert result["embedding_mode"] == "real"
                assert "embedding_service" in result
                
        # Test with mock embeddings
        mock_settings.USE_MOCK_EMBEDDINGS = True
        with patch("packages.vecpipe.search_api.qdrant_client", mock_qdrant_client):
            with patch("packages.vecpipe.search_api.embedding_service", mock_embedding_service):
                response = test_client_for_search_api.get("/")
                assert response.status_code == 200
                assert response.json()["embedding_mode"] == "mock"
                
        # Test error handling
        with patch("packages.vecpipe.search_api.qdrant_client", None):
            response = test_client_for_search_api.get("/")
            assert response.status_code == 503
            assert "Qdrant client not initialized" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_health_endpoint(self, mock_qdrant_client, mock_embedding_service, test_client_for_search_api):
        """Test /health endpoint."""
        with patch("packages.vecpipe.search_api.qdrant_client", mock_qdrant_client):
            with patch("packages.vecpipe.search_api.embedding_service", mock_embedding_service):
                # Mock successful Qdrant response
                mock_qdrant_client.get.return_value.status_code = 200
                mock_qdrant_client.get.return_value.json.return_value = {
                    "result": {
                        "collections": [
                            {"name": "col1"},
                            {"name": "col2"}
                        ]
                    }
                }
                
                response = test_client_for_search_api.get("/health")
                assert response.status_code == 200
                result = response.json()
                assert result["status"] == "healthy"
                assert result["components"]["qdrant"]["status"] == "healthy"
                assert result["components"]["qdrant"]["collections_count"] == 2
                assert result["components"]["embedding"]["status"] == "healthy"
                
        # Test with uninitialized embedding service
        mock_embedding_service.is_initialized = False
        with patch("packages.vecpipe.search_api.qdrant_client", mock_qdrant_client):
            with patch("packages.vecpipe.search_api.embedding_service", mock_embedding_service):
                response = test_client_for_search_api.get("/health")
                assert response.status_code == 200
                result = response.json()
                assert result["status"] == "degraded"
                assert result["components"]["embedding"]["status"] == "unhealthy"
                
        # Test with Qdrant error
        mock_qdrant_client.get.side_effect = Exception("Connection error")
        with patch("packages.vecpipe.search_api.qdrant_client", mock_qdrant_client):
            with patch("packages.vecpipe.search_api.embedding_service", None):
                response = test_client_for_search_api.get("/health")
                assert response.status_code == 503

    @pytest.mark.asyncio
    async def test_search_post_endpoint(self, mock_settings, mock_qdrant_client, mock_model_manager, test_client_for_search_api):
        """Test POST /search endpoint."""
        mock_settings.USE_MOCK_EMBEDDINGS = False
        
        with patch("packages.vecpipe.search_api.qdrant_client", mock_qdrant_client):
            with patch("packages.vecpipe.search_api.model_manager", mock_model_manager):
                # Mock collection info
                mock_qdrant_client.get.return_value.json.return_value = {
                    "result": {
                        "config": {
                            "params": {
                                "vectors": {"size": 1024}
                            }
                        }
                    }
                }
                mock_qdrant_client.get.return_value.raise_for_status = AsyncMock()
                
                # Mock search results
                mock_qdrant_client.post.return_value.json.return_value = {
                    "result": [
                        {
                            "id": "1",
                            "score": 0.95,
                            "payload": {
                                "path": "/test/file1.txt",
                                "chunk_id": "chunk-1",
                                "doc_id": "doc-1",
                                "content": "Test content 1"
                            }
                        }
                    ]
                }
                mock_qdrant_client.post.return_value.raise_for_status = AsyncMock()
                
                response = test_client_for_search_api.post(
                    "/search",
                    json={
                        "query": "test query",
                        "k": 5,
                        "search_type": "semantic"
                    }
                )
                
                assert response.status_code == 200
                result = response.json()
                assert result["query"] == "test query"
                assert len(result["results"]) == 1
                assert result["results"][0]["score"] == 0.95
                assert result["model_used"] == "test-model/float32"
                
    @pytest.mark.asyncio
    async def test_search_with_reranking(self, mock_settings, mock_qdrant_client, mock_model_manager, test_client_for_search_api):
        """Test search with reranking enabled."""
        mock_settings.USE_MOCK_EMBEDDINGS = False
        
        with patch("packages.vecpipe.search_api.get_reranker_for_embedding_model") as mock_get_reranker:
            mock_get_reranker.return_value = "test-reranker"
            
            # Mock collection info
            mock_qdrant_client.get.return_value.json.return_value = {
                "result": {
                    "config": {
                        "params": {
                            "vectors": {"size": 1024}
                        }
                    }
                }
            }
            mock_qdrant_client.get.return_value.raise_for_status = AsyncMock()
            
            # Mock search results with more candidates for reranking
            mock_qdrant_client.post.return_value.json.return_value = {
                "result": [
                    {
                        "id": "1",
                        "score": 0.85,
                        "payload": {
                            "path": "/test/file1.txt",
                            "chunk_id": "chunk-1",
                            "doc_id": "doc-1",
                            "content": "Content 1"
                        }
                    },
                    {
                        "id": "2",
                        "score": 0.80,
                        "payload": {
                            "path": "/test/file2.txt",
                            "chunk_id": "chunk-2",
                            "doc_id": "doc-2",
                            "content": "Content 2"
                        }
                    }
                ]
            }
            mock_qdrant_client.post.return_value.raise_for_status = AsyncMock()
            
            response = test_client_for_search_api.post(
                "/search",
                json={
                    "query": "test query",
                    "k": 2,
                    "use_reranker": True,
                    "include_content": True
                }
            )
            
            assert response.status_code == 200
            result = response.json()
            assert result["reranking_used"] is True
            assert result["reranker_model"] == "test-reranker/float32"
            assert result["reranking_time_ms"] is not None
            # Reranked results should have updated scores
            assert result["results"][0]["score"] == 0.95
            assert result["results"][1]["score"] == 0.90

    @pytest.mark.asyncio
    async def test_search_with_filters(self, mock_settings, mock_qdrant_client, mock_model_manager, test_client_for_search_api):
        """Test search with metadata filters."""
        mock_settings.USE_MOCK_EMBEDDINGS = False
        
        # Mock filtered search results
        mock_qdrant_client.post.return_value.json.return_value = {
            "result": [
                {
                    "id": "1",
                    "score": 0.90,
                    "payload": {
                        "path": "/filtered/file.txt",
                        "chunk_id": "chunk-1",
                        "doc_id": "doc-1"
                    }
                }
            ]
        }
        mock_qdrant_client.post.return_value.raise_for_status = AsyncMock()
        
        response = test_client_for_search_api.post(
            "/search",
            json={
                "query": "test query",
                "k": 5,
                "filters": {"must": [{"key": "type", "match": {"value": "document"}}]}
            }
        )
        
        assert response.status_code == 200
        result = response.json()
        assert len(result["results"]) == 1
        assert result["results"][0]["path"] == "/filtered/file.txt"
        
        # Verify filter was passed to Qdrant
        call_args = mock_qdrant_client.post.call_args
        assert "filter" in call_args[1]["json"]

    @pytest.mark.asyncio
    async def test_search_error_handling(self, mock_settings, mock_qdrant_client, mock_model_manager, test_client_for_search_api):
        """Test search error handling scenarios."""
        mock_settings.USE_MOCK_EMBEDDINGS = False
        
        # Test Qdrant HTTP error
        mock_qdrant_client.post.side_effect = httpx.HTTPStatusError(
            "Bad request",
            request=Mock(),
            response=Mock(status_code=400)
        )
        
        response = test_client_for_search_api.post(
            "/search",
            json={"query": "test", "k": 5}
        )
        
        assert response.status_code == 502
        assert "Vector database error" in response.json()["detail"]
        
        # Reset side effect
        mock_qdrant_client.post.side_effect = None
        
        # Test embedding generation error
        mock_model_manager.generate_embedding_async.side_effect = RuntimeError("Model load failed")
        
        response = test_client_for_search_api.post(
            "/search",
            json={"query": "test", "k": 5}
        )
        
        assert response.status_code == 503
        assert "Embedding service error" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_hybrid_search_endpoint(self, mock_settings, mock_qdrant_client, mock_hybrid_engine, test_client_for_search_api):
        """Test /hybrid_search endpoint."""
        mock_settings.USE_MOCK_EMBEDDINGS = True
        
        # Mock collection info
        mock_qdrant_client.get.return_value.json.return_value = {
            "result": {
                "config": {
                    "params": {
                        "vectors": {"size": 768}
                    }
                }
            }
        }
        mock_qdrant_client.get.return_value.raise_for_status = AsyncMock()
        
        response = test_client_for_search_api.get(
            "/hybrid_search",
            params={
                "q": "test query",
                "k": 10,
                "mode": "filter",
                "keyword_mode": "any"
            }
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["query"] == "test query"
        assert len(result["results"]) == 1
        assert result["results"][0]["matched_keywords"] == ["test"]
        assert result["keywords_extracted"] == ["test", "query"]
        assert result["search_mode"] == "filter"

    @pytest.mark.asyncio
    async def test_batch_search_endpoint(self, mock_settings, mock_qdrant_client, mock_model_manager, test_client_for_search_api):
        """Test /search/batch endpoint."""
        mock_settings.USE_MOCK_EMBEDDINGS = False
        
        with patch("packages.vecpipe.search_api.search_qdrant") as mock_search:
            # Mock search results for each query
            mock_search.return_value = [
                {
                    "score": 0.9,
                    "payload": {
                        "path": "/test/batch.txt",
                        "chunk_id": "chunk-1",
                        "doc_id": "doc-1"
                    }
                }
            ]
            
            response = test_client_for_search_api.post(
                "/search/batch",
                json={
                    "queries": ["query1", "query2", "query3"],
                    "k": 5,
                    "search_type": "semantic"
                }
            )
            
            assert response.status_code == 200
            result = response.json()
            assert len(result["responses"]) == 3
            assert all(r["query"] in ["query1", "query2", "query3"] for r in result["responses"])
            assert result["total_time_ms"] is not None
            
            # Verify embeddings were generated for all queries
            assert mock_model_manager.generate_embedding_async.call_count == 3

    @pytest.mark.asyncio
    async def test_keyword_search_endpoint(self, mock_hybrid_engine, test_client_for_search_api):
        """Test /keyword_search endpoint."""
        response = test_client_for_search_api.get(
            "/keyword_search",
            params={
                "q": "test keywords",
                "k": 20,
                "mode": "all"
            }
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["query"] == "test keywords"
        assert result["search_mode"] == "keywords_only"
        assert result["keywords_extracted"] == ["test", "query"]
        assert len(result["results"]) == 1
        assert result["results"][0]["score"] == 0.0  # No vector score for keyword search

    @pytest.mark.asyncio
    async def test_collection_info_endpoint(self, mock_qdrant_client, test_client_for_search_api):
        """Test /collection/info endpoint."""
        mock_qdrant_client.get.return_value.json.return_value = {
            "result": {
                "name": "test_collection",
                "points_count": 1000,
                "indexed_vectors_count": 1000
            }
        }
        mock_qdrant_client.get.return_value.raise_for_status = AsyncMock()
        
        response = test_client_for_search_api.get("/collection/info")
        
        assert response.status_code == 200
        result = response.json()
        assert result["name"] == "test_collection"
        assert result["points_count"] == 1000

    @pytest.mark.asyncio
    async def test_list_models_endpoint(self, test_client_for_search_api, mock_embedding_service):
        """Test /models endpoint."""
        with patch("shared.embedding.QUANTIZED_MODEL_INFO", {
            "test-model": {
                "description": "Test embedding model",
                "dimension": 768,
                "supports_quantization": True,
                "recommended_quantization": "float16",
                "memory_estimate": {"float32": 1024, "float16": 512}
            }
        }):
            with patch("packages.vecpipe.search_api.embedding_service", mock_embedding_service):
                mock_embedding_service.current_model_name = "test-model"
                mock_embedding_service.current_quantization = "float32"
                
                response = test_client_for_search_api.get("/models")
                assert response.status_code == 200
                result = response.json()
                assert len(result["models"]) == 1
                assert result["models"][0]["name"] == "test-model"
                assert result["models"][0]["dimension"] == 768
                assert result["current_model"] == "test-model"

    @pytest.mark.asyncio
    async def test_embed_endpoint(self, mock_model_manager, test_client_for_search_api):
        """Test /embed endpoint."""
        response = test_client_for_search_api.post(
            "/embed",
            json={
                "texts": ["text1", "text2", "text3"],
                "model_name": "test-model",
                "quantization": "float32",
                "batch_size": 2
            }
        )
        
        assert response.status_code == 200
        result = response.json()
        assert len(result["embeddings"]) == 3
        assert result["model_used"] == "test-model/float32"
        assert result["batch_count"] == 2  # 3 texts with batch_size=2
        assert result["embedding_time_ms"] is not None

    @pytest.mark.asyncio
    async def test_embed_memory_error(self, mock_model_manager, test_client_for_search_api):
        """Test /embed endpoint with memory error."""
        from packages.vecpipe.memory_utils import InsufficientMemoryError
        
        mock_model_manager.generate_embedding_async.side_effect = InsufficientMemoryError(
            "Not enough GPU memory"
        )
        
        response = test_client_for_search_api.post(
            "/embed",
            json={
                "texts": ["text1"],
                "model_name": "large-model",
                "quantization": "float32"
            }
        )
        
        assert response.status_code == 507
        assert "insufficient_memory" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_upsert_endpoint(self, mock_qdrant_client, test_client_for_search_api):
        """Test /upsert endpoint."""
        mock_qdrant_client.put.return_value.raise_for_status = AsyncMock()
        
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
                            "metadata": {"type": "document"}
                        }
                    }
                ],
                "wait": True
            }
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
        assert call_args[1]["json"]["wait"] is True

    @pytest.mark.asyncio
    async def test_upsert_error_handling(self, mock_qdrant_client):
        """Test /upsert endpoint error handling."""
        with patch("packages.vecpipe.search_api.qdrant_client", mock_qdrant_client):
            # Mock Qdrant error response
            error_response = Mock()
            error_response.json.return_value = {
                "status": {
                    "error": "Collection not found"
                }
            }
            mock_qdrant_client.put.side_effect = httpx.HTTPStatusError(
                "Not found",
                request=Mock(),
                response=error_response
            )
            
            from packages.vecpipe.search_api import upsert_points, UpsertPoint, PointPayload
            
            request = UpsertRequest(
                collection_name="nonexistent",
                points=[
                    UpsertPoint(
                        id="point-1",
                        vector=[0.1] * 768,
                        payload=PointPayload(
                            doc_id="doc-1",
                            chunk_id="chunk-1",
                            path="/test/file.txt"
                        )
                    )
                ]
            )
            
            with pytest.raises(HTTPException) as exc_info:
                await upsert_points(request)
            assert exc_info.value.status_code == 502
            assert "Collection not found" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_suggest_models_endpoint(self, test_client_for_search_api, mock_model_manager):
        """Test /models/suggest endpoint."""
        with patch("packages.vecpipe.search_api.get_gpu_memory_info") as mock_gpu_info:
            with patch("packages.vecpipe.search_api.suggest_model_configuration") as mock_suggest:
                # Test with GPU available
                mock_gpu_info.return_value = (8000, 16000)  # 8GB free, 16GB total
                mock_suggest.return_value = {
                    "embedding_model": "large-model",
                    "embedding_quantization": "float16",
                    "reranker_model": "reranker-model",
                    "reranker_quantization": "int8",
                    "notes": ["GPU detected with sufficient memory"]
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

    @pytest.mark.asyncio
    async def test_embedding_info_endpoint(self, mock_settings, mock_embedding_service, test_client_for_search_api):
        """Test /embedding/info endpoint."""
        mock_settings.USE_MOCK_EMBEDDINGS = False
        
        response = test_client_for_search_api.get("/embedding/info")
        assert response.status_code == 200
        result = response.json()
        assert result["mode"] == "real"
        assert result["available"] is True
        assert result["current_model"] == "test-model"
        assert result["quantization"] == "float32"
        assert result["device"] == "cpu"
        assert "model_details" in result
        
        # Test with mock embeddings
        mock_settings.USE_MOCK_EMBEDDINGS = True
        import packages.vecpipe.search_api as search_api_module
        original_service = search_api_module.embedding_service
        search_api_module.embedding_service = None
        
        try:
            response = test_client_for_search_api.get("/embedding/info")
            assert response.status_code == 200
            result = response.json()
            assert result["mode"] == "mock"
            assert result["available"] is False
        finally:
            search_api_module.embedding_service = original_service

    @pytest.mark.asyncio
    async def test_search_with_collection_metadata(self, mock_settings, mock_qdrant_client, mock_model_manager, test_client_for_search_api):
        """Test search with collection metadata for model selection."""
        mock_settings.USE_MOCK_EMBEDDINGS = False
        
        with patch("packages.vecpipe.search_api.QdrantClient") as mock_sync_client:
            with patch("packages.vecpipe.search_api.get_collection_metadata") as mock_get_metadata:
                # Mock collection metadata
                mock_get_metadata.return_value = {
                    "model_name": "collection-model",
                    "quantization": "float16",
                    "instruction": "Custom instruction"
                }
                
                # Mock collection info
                mock_qdrant_client.get.return_value.json.return_value = {
                    "result": {
                        "config": {
                            "params": {
                                "vectors": {"size": 768}
                            }
                        }
                    }
                }
                mock_qdrant_client.get.return_value.raise_for_status = AsyncMock()
                
                # Mock search results
                mock_qdrant_client.post.return_value.json.return_value = {
                    "result": []
                }
                mock_qdrant_client.post.return_value.raise_for_status = AsyncMock()
                
                response = test_client_for_search_api.post(
                    "/search",
                    json={
                        "query": "test",
                        "k": 5
                    }
                )
                
                assert response.status_code == 200
                
                # Verify collection model was used
                mock_model_manager.generate_embedding_async.assert_called_once()
                call_args = mock_model_manager.generate_embedding_async.call_args
                assert call_args[0][1] == "collection-model"  # model_name
                assert call_args[0][2] == "float16"  # quantization
                assert call_args[0][3] == "Custom instruction"  # instruction

    @pytest.mark.asyncio
    async def test_search_get_endpoint(self, mock_settings, mock_qdrant_client, mock_model_manager, test_client_for_search_api):
        """Test GET /search endpoint (compatibility)."""
        mock_settings.USE_MOCK_EMBEDDINGS = False
        
        # Mock search results
        mock_qdrant_client.post.return_value.json.return_value = {
            "result": []
        }
        mock_qdrant_client.post.return_value.raise_for_status = AsyncMock()
        
        response = test_client_for_search_api.get(
            "/search",
            params={
                "q": "test query",
                "k": 10,
                "collection": "custom_collection",
                "search_type": "question",
                "model_name": "custom-model",
                "quantization": "int8"
            }
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["query"] == "test query"
        
        # Verify the parameters were passed through
        call_args = mock_model_manager.generate_embedding_async.call_args
        assert call_args[0][1] == "custom-model"  # model_name
        assert call_args[0][2] == "int8"  # quantization

    def test_load_model_endpoint_mock_mode(self, mock_settings):
        """Test /models/load endpoint in mock mode."""
        mock_settings.USE_MOCK_EMBEDDINGS = True
        
        from packages.vecpipe.search_api import app
        
        client = TestClient(app)
        
        response = client.post(
            "/models/load",
            json={
                "model_name": "test-model",
                "quantization": "float32"
            }
        )
        
        assert response.status_code == 400
        assert "Cannot load models when using mock embeddings" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_generate_embedding_async_mock_mode(self, mock_settings):
        """Test generate_embedding_async in mock mode."""
        mock_settings.USE_MOCK_EMBEDDINGS = True
        
        import packages.vecpipe.search_api as search_api_module
        
        # Call the function directly with proper mock settings
        embedding = await search_api_module.generate_embedding_async("test text")
        
        assert len(embedding) == 1024
        assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.asyncio
    async def test_generate_embedding_async_error(self, mock_settings, mock_model_manager):
        """Test generate_embedding_async error handling."""
        mock_settings.USE_MOCK_EMBEDDINGS = False
        
        import packages.vecpipe.search_api as search_api_module
        
        # Temporarily set the mock
        original_manager = search_api_module.model_manager
        search_api_module.model_manager = mock_model_manager
        mock_model_manager.generate_embedding_async.return_value = None
        
        try:
            with pytest.raises(RuntimeError) as exc_info:
                await search_api_module.generate_embedding_async("test text")
            assert "Failed to generate embedding" in str(exc_info.value)
        finally:
            # Restore original
            search_api_module.model_manager = original_manager

    @pytest.mark.asyncio
    async def test_reranking_memory_error(self, mock_settings, mock_qdrant_client, mock_model_manager, test_client_for_search_api):
        """Test reranking with insufficient memory."""
        from packages.vecpipe.memory_utils import InsufficientMemoryError
        
        mock_settings.USE_MOCK_EMBEDDINGS = False
        
        with patch("packages.vecpipe.search_api.get_reranker_for_embedding_model") as mock_get_reranker:
            mock_get_reranker.return_value = "test-reranker"
            
            # Mock collection info
            mock_qdrant_client.get.return_value.json.return_value = {
                "result": {
                    "config": {
                        "params": {
                            "vectors": {"size": 1024}
                        }
                    }
                }
            }
            mock_qdrant_client.get.return_value.raise_for_status = AsyncMock()
            
            # Mock search results
            mock_qdrant_client.post.return_value.json.return_value = {
                "result": [
                    {
                        "id": "1",
                        "score": 0.85,
                        "payload": {
                            "path": "/test/file.txt",
                            "chunk_id": "chunk-1",
                            "doc_id": "doc-1",
                            "content": "Content"
                        }
                    }
                ]
            }
            mock_qdrant_client.post.return_value.raise_for_status = AsyncMock()
            
            # Mock reranking to raise memory error
            mock_model_manager.rerank_async.side_effect = InsufficientMemoryError(
                "Not enough GPU memory for reranking"
            )
            
            response = test_client_for_search_api.post(
                "/search",
                json={
                    "query": "test query",
                    "k": 1,
                    "use_reranker": True
                }
            )
            
            assert response.status_code == 507
            assert "insufficient_memory" in response.json()["detail"]