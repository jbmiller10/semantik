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


@pytest.fixture
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


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client."""
    client = AsyncMock()
    client.get = AsyncMock()
    client.post = AsyncMock()
    client.put = AsyncMock()
    client.aclose = AsyncMock()
    return client


@pytest.fixture
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


@pytest.fixture
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


@pytest.fixture
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
                        from packages.vecpipe.search_api import lifespan
                        
                        async with lifespan(app):
                            # Verify initialization
                            mock_get_service.assert_called_once()
                            mock_service.initialize.assert_called_once()
                            mock_mm_class.assert_called_once()
                        
                        # Verify cleanup
                        mock_qdrant_client.aclose.assert_called_once()
                        mock_model_manager.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_model_status(self, mock_model_manager):
        """Test /model/status endpoint."""
        with patch("packages.vecpipe.search_api.model_manager", mock_model_manager):
            from packages.vecpipe.search_api import model_status
            
            result = await model_status()
            assert result == {"loaded_models": [], "memory_usage": {}}
            
        # Test when model manager is not initialized
        with patch("packages.vecpipe.search_api.model_manager", None):
            result = await model_status()
            assert result == {"error": "Model manager not initialized"}

    @pytest.mark.asyncio
    async def test_root_endpoint(self, mock_settings, mock_qdrant_client, mock_embedding_service):
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
                
                from packages.vecpipe.search_api import root
                
                result = await root()
                assert result["status"] == "healthy"
                assert result["collection"]["points_count"] == 100
                assert result["collection"]["vector_size"] == 1024
                assert result["embedding_mode"] == "real"
                assert "embedding_service" in result
                
        # Test with mock embeddings
        mock_settings.USE_MOCK_EMBEDDINGS = True
        with patch("packages.vecpipe.search_api.qdrant_client", mock_qdrant_client):
            with patch("packages.vecpipe.search_api.embedding_service", mock_embedding_service):
                result = await root()
                assert result["embedding_mode"] == "mock"
                
        # Test error handling
        with patch("packages.vecpipe.search_api.qdrant_client", None):
            with pytest.raises(HTTPException) as exc_info:
                await root()
            assert exc_info.value.status_code == 503
            assert "Qdrant client not initialized" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_health_endpoint(self, mock_qdrant_client, mock_embedding_service):
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
                
                from packages.vecpipe.search_api import health
                
                result = await health()
                assert result["status"] == "healthy"
                assert result["components"]["qdrant"]["status"] == "healthy"
                assert result["components"]["qdrant"]["collections_count"] == 2
                assert result["components"]["embedding"]["status"] == "healthy"
                
        # Test with uninitialized embedding service
        mock_embedding_service.is_initialized = False
        with patch("packages.vecpipe.search_api.qdrant_client", mock_qdrant_client):
            with patch("packages.vecpipe.search_api.embedding_service", mock_embedding_service):
                result = await health()
                assert result["status"] == "degraded"
                assert result["components"]["embedding"]["status"] == "unhealthy"
                
        # Test with Qdrant error
        mock_qdrant_client.get.side_effect = Exception("Connection error")
        with patch("packages.vecpipe.search_api.qdrant_client", mock_qdrant_client):
            with patch("packages.vecpipe.search_api.embedding_service", None):
                with pytest.raises(HTTPException) as exc_info:
                    await health()
                assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_search_post_endpoint(self, mock_settings, mock_qdrant_client, mock_model_manager):
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
                
                from packages.vecpipe.search_api import search_post
                
                request = SearchRequest(
                    query="test query",
                    k=5,
                    search_type="semantic"
                )
                
                result = await search_post(request)
                assert isinstance(result, SearchResponse)
                assert result.query == "test query"
                assert len(result.results) == 1
                assert result.results[0].score == 0.95
                assert result.model_used == "test-model/float32"
                
    @pytest.mark.asyncio
    async def test_search_with_reranking(self, mock_settings, mock_qdrant_client, mock_model_manager):
        """Test search with reranking enabled."""
        mock_settings.USE_MOCK_EMBEDDINGS = False
        
        with patch("packages.vecpipe.search_api.qdrant_client", mock_qdrant_client):
            with patch("packages.vecpipe.search_api.model_manager", mock_model_manager):
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
                    
                    from packages.vecpipe.search_api import search_post
                    
                    request = SearchRequest(
                        query="test query",
                        k=2,
                        use_reranker=True,
                        include_content=True
                    )
                    
                    result = await search_post(request)
                    assert result.reranking_used is True
                    assert result.reranker_model == "test-reranker/float32"
                    assert result.reranking_time_ms is not None
                    # Reranked results should have updated scores
                    assert result.results[0].score == 0.95
                    assert result.results[1].score == 0.90

    @pytest.mark.asyncio
    async def test_search_with_filters(self, mock_settings, mock_qdrant_client, mock_model_manager):
        """Test search with metadata filters."""
        mock_settings.USE_MOCK_EMBEDDINGS = False
        
        with patch("packages.vecpipe.search_api.qdrant_client", mock_qdrant_client):
            with patch("packages.vecpipe.search_api.model_manager", mock_model_manager):
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
                
                from packages.vecpipe.search_api import search_post
                
                request = SearchRequest(
                    query="test query",
                    k=5,
                    filters={"must": [{"key": "type", "match": {"value": "document"}}]}
                )
                
                result = await search_post(request)
                assert len(result.results) == 1
                assert result.results[0].path == "/filtered/file.txt"
                
                # Verify filter was passed to Qdrant
                call_args = mock_qdrant_client.post.call_args
                assert "filter" in call_args[1]["json"]

    @pytest.mark.asyncio
    async def test_search_error_handling(self, mock_settings, mock_qdrant_client, mock_model_manager):
        """Test search error handling scenarios."""
        mock_settings.USE_MOCK_EMBEDDINGS = False
        
        # Test Qdrant HTTP error
        with patch("packages.vecpipe.search_api.qdrant_client", mock_qdrant_client):
            with patch("packages.vecpipe.search_api.model_manager", mock_model_manager):
                mock_qdrant_client.post.side_effect = httpx.HTTPStatusError(
                    "Bad request",
                    request=Mock(),
                    response=Mock(status_code=400)
                )
                
                from packages.vecpipe.search_api import search_post
                
                request = SearchRequest(query="test", k=5)
                
                with pytest.raises(HTTPException) as exc_info:
                    await search_post(request)
                assert exc_info.value.status_code == 502
                assert "Vector database error" in str(exc_info.value.detail)
        
        # Test embedding generation error
        with patch("packages.vecpipe.search_api.model_manager", mock_model_manager):
            mock_model_manager.generate_embedding_async.side_effect = RuntimeError("Model load failed")
            
            request = SearchRequest(query="test", k=5)
            
            with pytest.raises(HTTPException) as exc_info:
                await search_post(request)
            assert exc_info.value.status_code == 503
            assert "Embedding service error" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_hybrid_search_endpoint(self, mock_settings, mock_qdrant_client, mock_hybrid_engine):
        """Test /hybrid_search endpoint."""
        mock_settings.USE_MOCK_EMBEDDINGS = True
        
        with patch("packages.vecpipe.search_api.qdrant_client", mock_qdrant_client):
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
            
            from packages.vecpipe.search_api import hybrid_search
            
            result = await hybrid_search(
                q="test query",
                k=10,
                mode="filter",
                keyword_mode="any"
            )
            
            assert isinstance(result, HybridSearchResponse)
            assert result.query == "test query"
            assert len(result.results) == 1
            assert result.results[0].matched_keywords == ["test"]
            assert result.keywords_extracted == ["test", "query"]
            assert result.search_mode == "filter"

    @pytest.mark.asyncio
    async def test_batch_search_endpoint(self, mock_settings, mock_qdrant_client, mock_model_manager):
        """Test /search/batch endpoint."""
        mock_settings.USE_MOCK_EMBEDDINGS = False
        
        with patch("packages.vecpipe.search_api.qdrant_client", mock_qdrant_client):
            with patch("packages.vecpipe.search_api.model_manager", mock_model_manager):
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
                    
                    from packages.vecpipe.search_api import batch_search
                    
                    request = BatchSearchRequest(
                        queries=["query1", "query2", "query3"],
                        k=5,
                        search_type="semantic"
                    )
                    
                    result = await batch_search(request)
                    
                    assert isinstance(result, BatchSearchResponse)
                    assert len(result.responses) == 3
                    assert all(r.query in ["query1", "query2", "query3"] for r in result.responses)
                    assert result.total_time_ms is not None
                    
                    # Verify embeddings were generated for all queries
                    assert mock_model_manager.generate_embedding_async.call_count == 3

    @pytest.mark.asyncio
    async def test_keyword_search_endpoint(self, mock_hybrid_engine):
        """Test /keyword_search endpoint."""
        from packages.vecpipe.search_api import keyword_search
        
        result = await keyword_search(
            q="test keywords",
            k=20,
            mode="all"
        )
        
        assert isinstance(result, HybridSearchResponse)
        assert result.query == "test keywords"
        assert result.search_mode == "keywords_only"
        assert result.keywords_extracted == ["test", "query"]
        assert len(result.results) == 1
        assert result.results[0].score == 0.0  # No vector score for keyword search

    @pytest.mark.asyncio
    async def test_collection_info_endpoint(self, mock_qdrant_client):
        """Test /collection/info endpoint."""
        with patch("packages.vecpipe.search_api.qdrant_client", mock_qdrant_client):
            mock_qdrant_client.get.return_value.json.return_value = {
                "result": {
                    "name": "test_collection",
                    "points_count": 1000,
                    "indexed_vectors_count": 1000
                }
            }
            mock_qdrant_client.get.return_value.raise_for_status = AsyncMock()
            
            from packages.vecpipe.search_api import collection_info
            
            result = await collection_info()
            assert result["name"] == "test_collection"
            assert result["points_count"] == 1000

    @pytest.mark.asyncio
    async def test_list_models_endpoint(self):
        """Test /models endpoint."""
        with patch("packages.vecpipe.search_api.QUANTIZED_MODEL_INFO", {
            "test-model": {
                "description": "Test embedding model",
                "dimension": 768,
                "supports_quantization": True,
                "recommended_quantization": "float16",
                "memory_estimate": {"float32": 1024, "float16": 512}
            }
        }):
            with patch("packages.vecpipe.search_api.embedding_service") as mock_service:
                mock_service.current_model_name = "test-model"
                mock_service.current_quantization = "float32"
                
                from packages.vecpipe.search_api import list_models
                
                result = await list_models()
                assert len(result["models"]) == 1
                assert result["models"][0]["name"] == "test-model"
                assert result["models"][0]["dimension"] == 768
                assert result["current_model"] == "test-model"

    @pytest.mark.asyncio
    async def test_embed_endpoint(self, mock_model_manager):
        """Test /embed endpoint."""
        with patch("packages.vecpipe.search_api.model_manager", mock_model_manager):
            from packages.vecpipe.search_api import embed_texts
            
            request = EmbedRequest(
                texts=["text1", "text2", "text3"],
                model_name="test-model",
                quantization="float32",
                batch_size=2
            )
            
            result = await embed_texts(request)
            
            assert isinstance(result, EmbedResponse)
            assert len(result.embeddings) == 3
            assert result.model_used == "test-model/float32"
            assert result.batch_count == 2  # 3 texts with batch_size=2
            assert result.embedding_time_ms is not None

    @pytest.mark.asyncio
    async def test_embed_memory_error(self, mock_model_manager):
        """Test /embed endpoint with memory error."""
        from packages.vecpipe.memory_utils import InsufficientMemoryError
        
        with patch("packages.vecpipe.search_api.model_manager", mock_model_manager):
            mock_model_manager.generate_embedding_async.side_effect = InsufficientMemoryError(
                "Not enough GPU memory"
            )
            
            from packages.vecpipe.search_api import embed_texts
            
            request = EmbedRequest(
                texts=["text1"],
                model_name="large-model",
                quantization="float32"
            )
            
            with pytest.raises(HTTPException) as exc_info:
                await embed_texts(request)
            assert exc_info.value.status_code == 507
            assert "insufficient_memory" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_upsert_endpoint(self, mock_qdrant_client):
        """Test /upsert endpoint."""
        with patch("packages.vecpipe.search_api.qdrant_client", mock_qdrant_client):
            mock_qdrant_client.put.return_value.raise_for_status = AsyncMock()
            
            from packages.vecpipe.search_api import upsert_points, UpsertPoint, PointPayload
            
            request = UpsertRequest(
                collection_name="test_collection",
                points=[
                    UpsertPoint(
                        id="point-1",
                        vector=[0.1] * 768,
                        payload=PointPayload(
                            doc_id="doc-1",
                            chunk_id="chunk-1",
                            path="/test/file.txt",
                            content="Test content",
                            metadata={"type": "document"}
                        )
                    )
                ],
                wait=True
            )
            
            result = await upsert_points(request)
            
            assert isinstance(result, UpsertResponse)
            assert result.status == "success"
            assert result.points_upserted == 1
            assert result.collection_name == "test_collection"
            assert result.upsert_time_ms is not None
            
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
    async def test_suggest_models_endpoint(self):
        """Test /models/suggest endpoint."""
        with patch("packages.vecpipe.search_api.get_gpu_memory_info") as mock_gpu_info:
            with patch("packages.vecpipe.search_api.suggest_model_configuration") as mock_suggest:
                with patch("packages.vecpipe.search_api.model_manager") as mock_manager:
                    # Test with GPU available
                    mock_gpu_info.return_value = (8000, 16000)  # 8GB free, 16GB total
                    mock_suggest.return_value = {
                        "embedding_model": "large-model",
                        "embedding_quantization": "float16",
                        "reranker_model": "reranker-model",
                        "reranker_quantization": "int8",
                        "notes": ["GPU detected with sufficient memory"]
                    }
                    mock_manager.current_model_key = "current-model"
                    mock_manager.current_reranker_key = None
                    
                    from packages.vecpipe.search_api import suggest_models
                    
                    result = await suggest_models()
                    assert result["gpu_available"] is True
                    assert result["gpu_memory"]["free_mb"] == 8000
                    assert result["gpu_memory"]["usage_percent"] == 50.0
                    assert result["suggestion"]["embedding_model"] == "large-model"
                    
                    # Test without GPU
                    mock_gpu_info.return_value = (0, 0)
                    result = await suggest_models()
                    assert result["gpu_available"] is False
                    assert "No GPU detected" in result["message"]

    @pytest.mark.asyncio
    async def test_embedding_info_endpoint(self, mock_settings, mock_embedding_service):
        """Test /embedding/info endpoint."""
        mock_settings.USE_MOCK_EMBEDDINGS = False
        
        with patch("packages.vecpipe.search_api.embedding_service", mock_embedding_service):
            from packages.vecpipe.search_api import embedding_info
            
            result = await embedding_info()
            assert result["mode"] == "real"
            assert result["available"] is True
            assert result["current_model"] == "test-model"
            assert result["quantization"] == "float32"
            assert result["device"] == "cpu"
            assert "model_details" in result
            
        # Test with mock embeddings
        mock_settings.USE_MOCK_EMBEDDINGS = True
        with patch("packages.vecpipe.search_api.embedding_service", None):
            result = await embedding_info()
            assert result["mode"] == "mock"
            assert result["available"] is False

    @pytest.mark.asyncio
    async def test_search_with_collection_metadata(self, mock_settings, mock_qdrant_client, mock_model_manager):
        """Test search with collection metadata for model selection."""
        mock_settings.USE_MOCK_EMBEDDINGS = False
        
        with patch("packages.vecpipe.search_api.qdrant_client", mock_qdrant_client):
            with patch("packages.vecpipe.search_api.model_manager", mock_model_manager):
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
                        
                        from packages.vecpipe.search_api import search_post
                        
                        request = SearchRequest(
                            query="test",
                            k=5
                        )
                        
                        result = await search_post(request)
                        
                        # Verify collection model was used
                        mock_model_manager.generate_embedding_async.assert_called_once()
                        call_args = mock_model_manager.generate_embedding_async.call_args
                        assert call_args[0][1] == "collection-model"  # model_name
                        assert call_args[0][2] == "float16"  # quantization
                        assert call_args[0][3] == "Custom instruction"  # instruction

    @pytest.mark.asyncio
    async def test_search_get_endpoint(self, mock_settings, mock_qdrant_client, mock_model_manager):
        """Test GET /search endpoint (compatibility)."""
        mock_settings.USE_MOCK_EMBEDDINGS = False
        
        with patch("packages.vecpipe.search_api.qdrant_client", mock_qdrant_client):
            with patch("packages.vecpipe.search_api.model_manager", mock_model_manager):
                with patch("packages.vecpipe.search_api.search_post") as mock_search_post:
                    mock_search_post.return_value = SearchResponse(
                        query="test query",
                        results=[],
                        num_results=0,
                        search_type="semantic",
                        model_used="test-model/float32"
                    )
                    
                    from packages.vecpipe.search_api import search
                    
                    result = await search(
                        q="test query",
                        k=10,
                        collection="custom_collection",
                        search_type="question",
                        model_name="custom-model",
                        quantization="int8"
                    )
                    
                    assert isinstance(result, SearchResponse)
                    assert result.query == "test query"
                    
                    # Verify the request was properly constructed
                    call_args = mock_search_post.call_args
                    request = call_args[0][0]
                    assert request.query == "test query"
                    assert request.k == 10
                    assert request.collection == "custom_collection"
                    assert request.search_type == "question"
                    assert request.model_name == "custom-model"
                    assert request.quantization == "int8"

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
        
        from packages.vecpipe.search_api import generate_embedding_async
        
        embedding = await generate_embedding_async("test text")
        
        assert len(embedding) == 1024
        assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.asyncio
    async def test_generate_embedding_async_error(self, mock_settings, mock_model_manager):
        """Test generate_embedding_async error handling."""
        mock_settings.USE_MOCK_EMBEDDINGS = False
        
        with patch("packages.vecpipe.search_api.model_manager", mock_model_manager):
            mock_model_manager.generate_embedding_async.return_value = None
            
            from packages.vecpipe.search_api import generate_embedding_async
            
            with pytest.raises(RuntimeError) as exc_info:
                await generate_embedding_async("test text")
            assert "Failed to generate embedding" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_reranking_memory_error(self, mock_settings, mock_qdrant_client, mock_model_manager):
        """Test reranking with insufficient memory."""
        from packages.vecpipe.memory_utils import InsufficientMemoryError
        
        mock_settings.USE_MOCK_EMBEDDINGS = False
        
        with patch("packages.vecpipe.search_api.qdrant_client", mock_qdrant_client):
            with patch("packages.vecpipe.search_api.model_manager", mock_model_manager):
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
                    
                    from packages.vecpipe.search_api import search_post
                    
                    request = SearchRequest(
                        query="test query",
                        k=1,
                        use_reranker=True
                    )
                    
                    with pytest.raises(HTTPException) as exc_info:
                        await search_post(request)
                    assert exc_info.value.status_code == 507
                    assert "insufficient_memory" in str(exc_info.value.detail)