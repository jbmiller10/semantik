"""Integration test for search_api's embedding generation flow.

This test verifies the flow from search endpoint to embedding generation,
acknowledging current architectural constraints where settings are loaded
at module import time.

TODO: After CORE-003, update this test to properly verify the embedding service integration
when it's moved to a shared package with better dependency injection.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


class TestSearchAPIEmbeddingFlow:
    """Test the embedding generation flow in search_api.
    
    NOTE: Due to settings being loaded at module import time, these tests
    may run with USE_MOCK_EMBEDDINGS=True depending on the environment.
    This is a known limitation that will be addressed in the CORE-003 refactor.
    """

    def test_search_endpoint_embedding_flow(self):
        """Test that /search endpoint follows the expected embedding generation flow.
        
        This test verifies:
        1. Search endpoint accepts a query
        2. Embedding generation is triggered
        3. Results are returned properly
        
        It does NOT verify:
        - The actual embedding service implementation (may be mocked)
        - The specific embedding model used (depends on settings)
        """
        with patch("packages.vecpipe.search_utils.AsyncQdrantClient") as mock_qdrant_client_class:
            with patch("httpx.AsyncClient.get") as mock_get:
                # Mock Qdrant collection info
                mock_get.return_value = AsyncMock(
                    status_code=200,
                    json=lambda: {
                        "result": {
                            "points_count": 100,
                            "config": {"params": {"vectors": {"size": 1024}}},
                        }
                    },
                )
                mock_get.return_value.raise_for_status = AsyncMock()

                # Mock Qdrant search results
                mock_qdrant_instance = mock_qdrant_client_class.return_value
                mock_result = MagicMock()
                mock_result.id = "test-id"
                mock_result.score = 0.95
                mock_result.payload = {
                    "path": "/test/file.txt",
                    "chunk_id": "chunk-1",
                    "doc_id": "doc-1",
                }
                mock_qdrant_instance.search = AsyncMock(return_value=[mock_result])

                # Import and test
                from packages.vecpipe.search_api import app

                with TestClient(app) as client:
                    query_text = "test query"
                    response = client.post(
                        "/search",
                        json={
                            "query": query_text,
                            "k": 5,
                            "search_type": "semantic",
                        },
                    )

                    # Verify response
                    assert response.status_code == 200
                    result = response.json()
                    assert result["query"] == query_text
                    assert len(result["results"]) == 1
                    assert result["results"][0]["path"] == "/test/file.txt"
                    
                    # Verify Qdrant was searched
                    mock_qdrant_instance.search.assert_called_once()
                    search_call = mock_qdrant_instance.search.call_args
                    
                    # Should have been called with a vector (embedding)
                    assert "query_vector" in search_call.kwargs
                    assert isinstance(search_call.kwargs["query_vector"], list)
                    assert len(search_call.kwargs["query_vector"]) > 0

    @patch("packages.vecpipe.search_api.model_manager")
    def test_embedding_service_dependency_structure(self, mock_model_manager):
        """Document and verify the current dependency structure.
        
        This test documents the current problematic dependency where:
        - vecpipe/search_api imports from webui.embedding_service
        - vecpipe/model_manager imports from webui.embedding_service
        
        After CORE-003, these imports should come from a shared package.
        """
        # Document current imports (these would fail if structure changes)
        from packages.vecpipe import search_api
        from packages.vecpipe import model_manager
        
        # Verify the imports exist (will help catch when refactoring happens)
        assert hasattr(search_api, 'EmbeddingService')
        assert hasattr(model_manager, 'EmbeddingService')
        
        # Document that both import from webui
        import inspect
        search_api_source = inspect.getsource(search_api)
        model_manager_source = inspect.getsource(model_manager)
        
        assert "from webui.embedding_service import EmbeddingService" in search_api_source
        assert "from webui.embedding_service import EmbeddingService" in model_manager_source
        
        # This assertion will need to be updated after CORE-003
        # to verify imports come from shared.embedding_service instead

    def test_search_with_custom_parameters_flow(self):
        """Test that custom model parameters are handled in the flow.
        
        This verifies that custom parameters are accepted and processed,
        though the actual model used depends on settings and availability.
        """
        with patch("packages.vecpipe.search_utils.AsyncQdrantClient") as mock_qdrant_client_class:
            with patch("httpx.AsyncClient.get") as mock_get:
                # Mock responses
                mock_get.return_value = AsyncMock(
                    status_code=200,
                    json=lambda: {
                        "result": {
                            "points_count": 50,
                            "config": {"params": {"vectors": {"size": 768}}},
                        }
                    },
                )
                mock_get.return_value.raise_for_status = AsyncMock()

                mock_qdrant_instance = mock_qdrant_client_class.return_value
                mock_qdrant_instance.search = AsyncMock(return_value=[])

                # Need to patch metrics server to avoid port conflicts
                with patch("packages.vecpipe.search_api.start_metrics_server"):
                    from packages.vecpipe.search_api import app

                    with TestClient(app) as client:
                        response = client.post(
                            "/search",
                            json={
                                "query": "test with custom params",
                                "k": 10,
                                "search_type": "question",
                                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                                "quantization": "int8",
                            },
                        )

                        assert response.status_code == 200
                        result = response.json()
                        
                        # The response should indicate the search type
                        assert result["search_type"] == "question"
                        
                        # Note: model_used in response depends on USE_MOCK_EMBEDDINGS
                        # so we don't assert on it here