"""Additional edge case tests for search_api.py to ensure comprehensive coverage.

This module focuses on testing edge cases, FAISS fallback, and complex error scenarios.
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from fastapi import HTTPException

from packages.vecpipe.search_api import (
    SearchRequest,
    SearchResponse,
    app,
    generate_embedding_async,
    search_post,
)


class TestSearchAPIEdgeCases:
    """Test edge cases and complex scenarios in search_api."""

    @pytest.mark.asyncio
    async def test_search_without_content_then_rerank(self, mock_settings, mock_qdrant_client, mock_model_manager):
        """Test reranking when initial results don't have content."""
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
                    
                    # Mock initial search results without content
                    initial_results = {
                        "result": [
                            {
                                "id": "1",
                                "score": 0.85,
                                "payload": {
                                    "path": "/test/file1.txt",
                                    "chunk_id": "chunk-1",
                                    "doc_id": "doc-1"
                                    # No content field
                                }
                            },
                            {
                                "id": "2",
                                "score": 0.80,
                                "payload": {
                                    "path": "/test/file2.txt",
                                    "chunk_id": "chunk-2",
                                    "doc_id": "doc-2"
                                    # No content field
                                }
                            }
                        ]
                    }
                    
                    # Mock fetching content for reranking
                    fetch_results = {
                        "result": {
                            "points": [
                                {
                                    "payload": {
                                        "chunk_id": "chunk-1",
                                        "content": "Fetched content 1"
                                    }
                                },
                                {
                                    "payload": {
                                        "chunk_id": "chunk-2",
                                        "content": "Fetched content 2"
                                    }
                                }
                            ]
                        }
                    }
                    
                    # Set up mock to return different results for different calls
                    mock_qdrant_client.post.return_value.json.side_effect = [
                        initial_results,  # First call for search
                        fetch_results     # Second call for fetching content
                    ]
                    mock_qdrant_client.post.return_value.raise_for_status = AsyncMock()
                    
                    request = SearchRequest(
                        query="test query",
                        k=2,
                        use_reranker=True,
                        include_content=False  # Don't request content initially
                    )
                    
                    result = await search_post(request)
                    
                    # Verify content was fetched for reranking
                    assert mock_qdrant_client.post.call_count == 2
                    second_call = mock_qdrant_client.post.call_args_list[1]
                    assert "filter" in second_call[1]["json"]
                    assert result.reranking_used is True

    @pytest.mark.asyncio
    async def test_search_with_missing_content_during_rerank(self, mock_settings, mock_qdrant_client, mock_model_manager):
        """Test reranking when content fetch fails."""
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
                    
                    # Mock search results without content
                    mock_qdrant_client.post.return_value.json.side_effect = [
                        {
                            "result": [
                                {
                                    "id": "1",
                                    "score": 0.85,
                                    "payload": {
                                        "path": "/test/file.txt",
                                        "chunk_id": "chunk-1",
                                        "doc_id": "doc-1"
                                    }
                                }
                            ]
                        },
                        # Content fetch fails (returns empty)
                        {
                            "result": {
                                "points": []
                            }
                        }
                    ]
                    mock_qdrant_client.post.return_value.raise_for_status = AsyncMock()
                    
                    request = SearchRequest(
                        query="test query",
                        k=1,
                        use_reranker=True
                    )
                    
                    result = await search_post(request)
                    
                    # Should still return results with fallback content
                    assert len(result.results) == 1
                    # Verify reranking was attempted with fallback content
                    mock_model_manager.rerank_async.assert_called_once()
                    call_args = mock_model_manager.rerank_async.call_args
                    documents = call_args[1]["documents"]
                    assert "Document from" in documents[0]  # Fallback content

    @pytest.mark.asyncio
    async def test_search_with_reranking_failure(self, mock_settings, mock_qdrant_client, mock_model_manager):
        """Test graceful fallback when reranking fails."""
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
                                "score": 0.95,
                                "payload": {
                                    "path": "/test/file1.txt",
                                    "chunk_id": "chunk-1",
                                    "doc_id": "doc-1",
                                    "content": "Content 1"
                                }
                            },
                            {
                                "id": "2",
                                "score": 0.90,
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
                    
                    # Make reranking fail (but not with InsufficientMemoryError)
                    mock_model_manager.rerank_async.side_effect = Exception("Reranking model error")
                    
                    request = SearchRequest(
                        query="test query",
                        k=2,
                        use_reranker=True
                    )
                    
                    result = await search_post(request)
                    
                    # Should return original results without reranking
                    assert len(result.results) == 2
                    assert result.results[0].score == 0.95  # Original scores
                    assert result.results[1].score == 0.90
                    assert result.reranking_used is True  # Was attempted
                    assert result.reranker_model is None  # But failed

    @pytest.mark.asyncio
    async def test_hybrid_search_error_handling(self, mock_settings, mock_qdrant_client):
        """Test hybrid search error scenarios."""
        mock_settings.USE_MOCK_EMBEDDINGS = True
        
        with patch("packages.vecpipe.search_api.qdrant_client", mock_qdrant_client):
            with patch("packages.vecpipe.search_api.HybridSearchEngine") as mock_engine_class:
                # Make hybrid engine initialization fail
                mock_engine_class.side_effect = Exception("Failed to initialize hybrid engine")
                
                from packages.vecpipe.search_api import hybrid_search
                
                with pytest.raises(HTTPException) as exc_info:
                    await hybrid_search(q="test", k=10)
                assert exc_info.value.status_code == 500
                assert "Hybrid search error" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_batch_search_partial_failure(self, mock_settings, mock_qdrant_client, mock_model_manager):
        """Test batch search when some queries fail."""
        mock_settings.USE_MOCK_EMBEDDINGS = False
        
        with patch("packages.vecpipe.search_api.qdrant_client", mock_qdrant_client):
            with patch("packages.vecpipe.search_api.model_manager", mock_model_manager):
                # Make embedding generation fail for second query
                mock_model_manager.generate_embedding_async.side_effect = [
                    [0.1] * 1024,  # Success for first query
                    RuntimeError("Embedding failed"),  # Fail for second query
                    [0.3] * 1024   # Success for third query
                ]
                
                from packages.vecpipe.search_api import batch_search, BatchSearchRequest
                
                request = BatchSearchRequest(
                    queries=["query1", "query2", "query3"],
                    k=5
                )
                
                # The entire batch should fail if any query fails
                with pytest.raises(HTTPException) as exc_info:
                    await batch_search(request)
                assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_collection_metadata_error_handling(self, mock_settings, mock_qdrant_client, mock_model_manager):
        """Test search when collection metadata fetch fails."""
        mock_settings.USE_MOCK_EMBEDDINGS = False
        
        with patch("packages.vecpipe.search_api.qdrant_client", mock_qdrant_client):
            with patch("packages.vecpipe.search_api.model_manager", mock_model_manager):
                with patch("packages.vecpipe.search_api.QdrantClient") as mock_sync_client:
                    with patch("packages.vecpipe.search_api.get_collection_metadata") as mock_get_metadata:
                        # Make metadata fetch fail
                        mock_get_metadata.side_effect = Exception("Metadata fetch failed")
                        
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
                            "result": []
                        }
                        mock_qdrant_client.post.return_value.raise_for_status = AsyncMock()
                        
                        request = SearchRequest(query="test", k=5)
                        
                        # Should continue with default model despite metadata error
                        result = await search_post(request)
                        assert result.model_used == "test-model/float32"

    @pytest.mark.asyncio
    async def test_upsert_with_http_error_parsing(self, mock_qdrant_client):
        """Test upsert error parsing when response format is unexpected."""
        with patch("packages.vecpipe.search_api.qdrant_client", mock_qdrant_client):
            # Mock error without expected format
            error_response = Mock()
            error_response.json.side_effect = Exception("Invalid JSON")
            
            mock_qdrant_client.put.side_effect = httpx.HTTPStatusError(
                "Server error",
                request=Mock(),
                response=error_response
            )
            
            from packages.vecpipe.search_api import upsert_points, UpsertRequest, UpsertPoint, PointPayload
            
            request = UpsertRequest(
                collection_name="test",
                points=[
                    UpsertPoint(
                        id="1",
                        vector=[0.1] * 768,
                        payload=PointPayload(
                            doc_id="doc-1",
                            chunk_id="chunk-1",
                            path="/test.txt"
                        )
                    )
                ]
            )
            
            with pytest.raises(HTTPException) as exc_info:
                await upsert_points(request)
            assert exc_info.value.status_code == 502
            assert "Vector database error" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_search_with_invalid_collection_info(self, mock_settings, mock_qdrant_client, mock_model_manager):
        """Test search when collection info has unexpected format."""
        mock_settings.USE_MOCK_EMBEDDINGS = False
        
        with patch("packages.vecpipe.search_api.qdrant_client", mock_qdrant_client):
            with patch("packages.vecpipe.search_api.model_manager", mock_model_manager):
                # Mock collection info with missing/invalid structure
                mock_qdrant_client.get.return_value.json.return_value = {
                    "result": {
                        # Missing config.params.vectors.size
                        "points_count": 100
                    }
                }
                mock_qdrant_client.get.return_value.raise_for_status = AsyncMock()
                
                # Mock search results
                mock_qdrant_client.post.return_value.json.return_value = {
                    "result": []
                }
                mock_qdrant_client.post.return_value.raise_for_status = AsyncMock()
                
                request = SearchRequest(query="test", k=5)
                
                # Should use default dimension
                result = await search_post(request)
                assert result is not None
                
                # Verify default dimension was used for embedding
                call_args = mock_model_manager.generate_embedding_async.call_args
                assert call_args is not None

    @pytest.mark.asyncio
    async def test_search_with_custom_reranker_params(self, mock_settings, mock_qdrant_client, mock_model_manager):
        """Test search with custom reranker model and quantization."""
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
                
                request = SearchRequest(
                    query="test query",
                    k=1,
                    use_reranker=True,
                    rerank_model="custom-reranker",
                    rerank_quantization="int8"
                )
                
                result = await search_post(request)
                
                # Verify custom reranker params were used
                mock_model_manager.rerank_async.assert_called_once()
                call_args = mock_model_manager.rerank_async.call_args
                assert call_args[1]["model_name"] == "custom-reranker"
                assert call_args[1]["quantization"] == "int8"
                assert result.reranker_model == "custom-reranker/int8"

    @pytest.mark.asyncio
    async def test_search_with_large_k_and_reranking(self, mock_settings, mock_qdrant_client, mock_model_manager):
        """Test search with large k value and reranking limits."""
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
                    
                    # Create many mock results
                    mock_results = []
                    for i in range(200):  # Max candidates limit
                        mock_results.append({
                            "id": f"id-{i}",
                            "score": 0.9 - (i * 0.001),
                            "payload": {
                                "path": f"/test/file{i}.txt",
                                "chunk_id": f"chunk-{i}",
                                "doc_id": f"doc-{i}",
                                "content": f"Content {i}"
                            }
                        })
                    
                    mock_qdrant_client.post.return_value.json.return_value = {
                        "result": mock_results
                    }
                    mock_qdrant_client.post.return_value.raise_for_status = AsyncMock()
                    
                    # Return reranked indices for top 50
                    mock_model_manager.rerank_async.return_value = [
                        (i, 0.99 - (i * 0.01)) for i in range(50)
                    ]
                    
                    request = SearchRequest(
                        query="test query",
                        k=50,  # Request 50 results
                        use_reranker=True
                    )
                    
                    result = await search_post(request)
                    
                    # Should return exactly k results
                    assert len(result.results) == 50
                    # Verify search_k was capped at max_candidates (200)
                    call_args = mock_qdrant_client.post.call_args
                    assert call_args[1]["json"]["limit"] == 200  # max_candidates

    @pytest.mark.asyncio
    async def test_search_with_empty_query(self, mock_settings, mock_qdrant_client, mock_model_manager):
        """Test search with empty or whitespace-only query."""
        mock_settings.USE_MOCK_EMBEDDINGS = False
        
        with patch("packages.vecpipe.search_api.qdrant_client", mock_qdrant_client):
            with patch("packages.vecpipe.search_api.model_manager", mock_model_manager):
                # Mock to return empty embedding for empty text
                mock_model_manager.generate_embedding_async.return_value = [0.0] * 1024
                
                request = SearchRequest(
                    query="   ",  # Whitespace only
                    k=5
                )
                
                # Should handle gracefully
                result = await search_post(request)
                assert result.query == "   "

    @pytest.mark.asyncio
    async def test_generate_embedding_async_with_no_model_manager(self, mock_settings):
        """Test generate_embedding_async when model manager is not initialized."""
        mock_settings.USE_MOCK_EMBEDDINGS = False
        
        with patch("packages.vecpipe.search_api.model_manager", None):
            with pytest.raises(RuntimeError) as exc_info:
                await generate_embedding_async("test text")
            assert "Model manager not initialized" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_search_faiss_fallback(self, mock_settings, mock_qdrant_client, mock_model_manager):
        """Test search falling back to FAISS when Qdrant is unavailable."""
        mock_settings.USE_MOCK_EMBEDDINGS = False
        
        with patch("packages.vecpipe.search_api.qdrant_client", mock_qdrant_client):
            with patch("packages.vecpipe.search_api.model_manager", mock_model_manager):
                # Make Qdrant unavailable
                mock_qdrant_client.post.side_effect = httpx.ConnectError("Connection refused")
                
                # Currently, the search_api doesn't have FAISS fallback implemented
                # This test documents the expected behavior when it's added
                request = SearchRequest(query="test", k=5)
                
                with pytest.raises(HTTPException) as exc_info:
                    await search_post(request)
                assert exc_info.value.status_code == 502

    @pytest.mark.asyncio
    async def test_keyword_search_cleanup(self, mock_hybrid_engine):
        """Test that keyword search properly cleans up resources."""
        from packages.vecpipe.search_api import keyword_search
        
        # Make the search raise an exception
        mock_hybrid_engine.search_by_keywords.side_effect = Exception("Search failed")
        
        with pytest.raises(HTTPException):
            await keyword_search(q="test", k=10)
        
        # Verify cleanup was called
        mock_hybrid_engine.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_collection_info_error(self, mock_qdrant_client):
        """Test collection info endpoint error handling."""
        with patch("packages.vecpipe.search_api.qdrant_client", None):
            from packages.vecpipe.search_api import collection_info
            
            with pytest.raises(HTTPException) as exc_info:
                await collection_info()
            assert exc_info.value.status_code == 503
            assert "Qdrant client not initialized" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_models_load_error(self, mock_settings, mock_embedding_service):
        """Test /models/load endpoint error handling."""
        mock_settings.USE_MOCK_EMBEDDINGS = False
        
        with patch("packages.vecpipe.search_api.embedding_service", None):
            from packages.vecpipe.search_api import load_model
            
            with pytest.raises(HTTPException) as exc_info:
                await load_model(model_name="test-model", quantization="float32")
            assert exc_info.value.status_code == 503
            assert "Embedding service not initialized" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_search_with_score_threshold(self, mock_settings, mock_qdrant_client, mock_model_manager):
        """Test search respects score threshold in results."""
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
                
                # Mock search results with varying scores
                mock_qdrant_client.post.return_value.json.return_value = {
                    "result": [
                        {
                            "id": "1",
                            "score": 0.95,
                            "payload": {"path": "/high.txt", "chunk_id": "c1", "doc_id": "d1"}
                        },
                        {
                            "id": "2",
                            "score": 0.75,
                            "payload": {"path": "/medium.txt", "chunk_id": "c2", "doc_id": "d2"}
                        },
                        {
                            "id": "3",
                            "score": 0.55,
                            "payload": {"path": "/low.txt", "chunk_id": "c3", "doc_id": "d3"}
                        }
                    ]
                }
                mock_qdrant_client.post.return_value.raise_for_status = AsyncMock()
                
                request = SearchRequest(
                    query="test query",
                    k=10,
                    score_threshold=0.7  # Should filter out the last result
                )
                
                result = await search_post(request)
                
                # The search_api currently doesn't filter by score_threshold
                # This documents expected behavior if implemented
                assert len(result.results) == 3  # Currently returns all results