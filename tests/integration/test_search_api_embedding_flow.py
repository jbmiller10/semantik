"""Integration test for search_api's embedding generation flow.

This test verifies the flow from search endpoint to embedding generation,
acknowledging current architectural constraints where settings are loaded
at module import time.

NOTE: The embedding service has been moved to a shared package with dependency injection
as part of CORE-003. This test now verifies the flow with the updated architecture.
"""

import inspect
from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from shared.config import settings
from vecpipe import model_manager, search_api
from vecpipe.search_api import app


class TestSearchAPIEmbeddingFlow:
    """Test the embedding generation flow in search_api.

    NOTE: Due to settings being loaded at module import time, these tests
        may run with USE_MOCK_EMBEDDINGS=True depending on the environment.
        This is a known limitation that will be addressed in the CORE-003 refactor.
    """

    @pytest.fixture(autouse=True)
    def _clear_cache(self) -> Generator[Any, None, None]:
        """Clear collection info/metadata cache to avoid stale data between tests."""
        from vecpipe.search.cache import clear_cache

        clear_cache()
        yield
        clear_cache()

    def test_search_endpoint_embedding_flow(self) -> None:
        """Test that /search endpoint follows the expected embedding generation flow.

        This test verifies:
        1. Search endpoint accepts a query
        2. Embedding generation is triggered
        3. Results are returned properly

        It does NOT verify:
        - The actual embedding service implementation (may be mocked)
        - The specific embedding model used (depends on settings)
        """
        original_internal_key = settings.INTERNAL_API_KEY
        settings.INTERNAL_API_KEY = "test-internal-key"
        try:
            with (
                patch("vecpipe.search_utils.AsyncQdrantClient") as mock_qdrant_client_class,
                patch("httpx.AsyncClient.get") as mock_get,
            ):
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
                mock_qdrant_instance.close = AsyncMock()

                # Import and test

                with TestClient(app) as client:
                    client.headers.update({"X-Internal-Api-Key": settings.INTERNAL_API_KEY})
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
        finally:
            settings.INTERNAL_API_KEY = original_internal_key

    @patch("vecpipe.search_api.model_manager")
    def test_embedding_service_dependency_structure(self, mock_model_manager) -> None:
        """Document and verify the current dependency structure.

        This test documents the CORE-003 refactored architecture where:
        - vecpipe/search_api imports get_embedding_service from shared.embedding.service
        - vecpipe/model_manager uses the plugin-aware provider system via EmbeddingProviderFactory

        Both now use the shared.embedding package with proper dependency injection.
        """
        # Verify the imports exist (will help catch when refactoring happens)
        # search_api uses get_embedding_service for dependency injection
        assert hasattr(search_api, "get_embedding_service")
        # model_manager uses the provider factory system
        assert hasattr(model_manager, "EmbeddingProviderFactory")

        # Document that both import from shared.embedding

        search_api_source = inspect.getsource(search_api)
        model_manager_source = inspect.getsource(model_manager)

        # search_api imports from shared.embedding.service
        assert "from shared.embedding.service import get_embedding_service" in search_api_source
        # model_manager imports from shared.embedding.factory (provider-based architecture)
        assert "from shared.embedding.factory import EmbeddingProviderFactory" in model_manager_source

    def test_search_with_custom_parameters_flow(self) -> None:
        """Test that custom model parameters are handled in the flow.

        This verifies that custom parameters are accepted and processed,
        though the actual model used depends on settings and availability.
        """
        original_internal_key = settings.INTERNAL_API_KEY
        settings.INTERNAL_API_KEY = "test-internal-key"
        try:
            with (
                patch("vecpipe.search_utils.AsyncQdrantClient") as mock_qdrant_client_class,
                patch("httpx.AsyncClient.get") as mock_get,
            ):
                # Mock responses - use 384 to match all-MiniLM-L6-v2 dimensions
                mock_get.return_value = AsyncMock(
                    status_code=200,
                    json=lambda: {
                        "result": {
                            "points_count": 50,
                            "config": {"params": {"vectors": {"size": 384}}},
                        }
                    },
                )
                mock_get.return_value.raise_for_status = AsyncMock()

                mock_qdrant_instance = mock_qdrant_client_class.return_value
                mock_qdrant_instance.search = AsyncMock(return_value=[])
                mock_qdrant_instance.close = AsyncMock()

                # Need to patch metrics server to avoid port conflicts
                with patch("vecpipe.search_api.start_metrics_server"), TestClient(app) as client:
                    client.headers.update({"X-Internal-Api-Key": settings.INTERNAL_API_KEY})
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
        finally:
            settings.INTERNAL_API_KEY = original_internal_key
