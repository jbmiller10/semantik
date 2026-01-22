"""Additional edge case tests for search_api.py to ensure comprehensive coverage.

This module focuses on testing edge cases, FAISS fallback, and complex error scenarios.
"""

from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import vecpipe.search_api as search_api_module
from shared.contracts.search import BatchSearchRequest, SearchRequest
from vecpipe.search_api import PointPayload, UpsertPoint, UpsertRequest, app, batch_search, search_post, upsert_points


@pytest.fixture()
def mock_settings() -> Generator[Any, None, None]:
    """Mock settings for testing."""
    with patch("vecpipe.search_api.settings") as mock:
        mock.QDRANT_HOST = "localhost"
        mock.QDRANT_PORT = 6333
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
        yield mock


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
    """Mock model manager."""
    manager = Mock()
    manager.generate_embedding_async = AsyncMock(return_value=[0.1] * 1024)
    manager.rerank_async = AsyncMock(return_value=[(0, 0.95), (1, 0.90)])
    manager.get_status = Mock(return_value={"loaded_models": [], "memory_usage": {}})
    manager.shutdown = Mock()
    manager.current_model_key = "test-model"
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
    service.get_model_info = Mock(
        return_value={"model_name": "test-model", "dimension": 1024, "description": "Test model"}
    )
    return service


# NOTE: mock_hybrid_engine fixture removed - legacy hybrid search deleted


@pytest.fixture()
def test_client_for_search_api(
    mock_settings, mock_qdrant_client, mock_model_manager, mock_embedding_service
) -> Generator[Any, None, None]:
    """Create a test client for the search API with mocked dependencies."""

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

    # Patch settings during test
    with (
        patch("vecpipe.search_api.settings", mock_settings),
        patch("vecpipe.search.auth.settings", mock_settings),
    ):
        # Create test client
        client = TestClient(app)
        client.headers.update({"X-Internal-Api-Key": mock_settings.INTERNAL_API_KEY})
        yield client

    # Restore original values
    search_api_module.qdrant_client = original_qdrant
    search_api_module.model_manager = original_model_manager
    search_api_module.embedding_service = original_embedding_service

    # Clean up
    app.dependency_overrides.clear()


class TestSearchAPIEdgeCases:
    """Test edge cases and complex scenarios in search_api."""

    def test_search_without_content_then_rerank(
        self, mock_settings, mock_qdrant_client, mock_model_manager, test_client_for_search_api
    ) -> None:
        """Test reranking when initial results don't have content."""
        mock_settings.USE_MOCK_EMBEDDINGS = False

        with (
            patch("vecpipe.search_api.qdrant_client", mock_qdrant_client),
            patch("vecpipe.search_api.model_manager", mock_model_manager),
            patch("vecpipe.search_api.get_reranker_for_embedding_model") as mock_get_reranker,
            patch("vecpipe.search_api.search_qdrant") as mock_search_qdrant,
        ):
            mock_get_reranker.return_value = "test-reranker"

            # Mock collection info
            mock_qdrant_client.get.return_value.json.return_value = {
                "result": {"config": {"params": {"vectors": {"size": 1024}}}}
            }
            mock_qdrant_client.get.return_value.raise_for_status = AsyncMock()

            # Mock initial search results without content
            mock_search_qdrant.return_value = [
                {
                    "id": "1",
                    "score": 0.85,
                    "payload": {
                        "path": "/test/file1.txt",
                        "chunk_id": "chunk-1",
                        "doc_id": "doc-1",
                        # No content field
                    },
                },
                {
                    "id": "2",
                    "score": 0.80,
                    "payload": {
                        "path": "/test/file2.txt",
                        "chunk_id": "chunk-2",
                        "doc_id": "doc-2",
                        # No content field
                    },
                },
            ]

            # Mock fetching content for reranking
            fetch_results = {
                "result": {
                    "points": [
                        {"payload": {"chunk_id": "chunk-1", "content": "Fetched content 1"}},
                        {"payload": {"chunk_id": "chunk-2", "content": "Fetched content 2"}},
                    ]
                }
            }

            # Set up mock for fetching content
            mock_qdrant_client.post.return_value.json.return_value = fetch_results
            mock_qdrant_client.post.return_value.raise_for_status = AsyncMock()

            response = test_client_for_search_api.post(
                "/search",
                json={
                    "query": "test query",
                    "k": 2,
                    "use_reranker": True,
                    "include_content": False,  # Don't request content initially
                },
            )

            assert response.status_code == 200
            result = response.json()

            # Verify content was fetched for reranking
            assert mock_qdrant_client.post.call_count == 1  # Only for fetching content
            call_args = mock_qdrant_client.post.call_args
            assert "/points/scroll" in call_args[0][0]
            assert "filter" in call_args[1]["json"]
            assert result["reranking_used"] is True

    @pytest.mark.asyncio()
    async def test_search_with_missing_content_during_rerank(
        self, mock_settings, mock_qdrant_client, mock_model_manager
    ) -> None:
        """Test reranking when content fetch fails."""
        mock_settings.USE_MOCK_EMBEDDINGS = False

        with (
            patch("vecpipe.search_api.qdrant_client", mock_qdrant_client),
            patch("vecpipe.search_api.model_manager", mock_model_manager),
            patch("vecpipe.search_api.get_reranker_for_embedding_model") as mock_get_reranker,
            patch("vecpipe.search_api.search_qdrant") as mock_search_qdrant,
        ):
            mock_get_reranker.return_value = "test-reranker"

            # Mock collection info
            mock_qdrant_client.get.return_value.json.return_value = {
                "result": {"config": {"params": {"vectors": {"size": 1024}}}}
            }
            mock_qdrant_client.get.return_value.raise_for_status = AsyncMock()

            # Mock search results without content
            mock_search_qdrant.return_value = [
                {
                    "id": "1",
                    "score": 0.85,
                    "payload": {"path": "/test/file.txt", "chunk_id": "chunk-1", "doc_id": "doc-1"},
                }
            ]

            # Content fetch fails (returns empty)
            mock_qdrant_client.post.return_value.json.return_value = {"result": {"points": []}}
            mock_qdrant_client.post.return_value.raise_for_status = AsyncMock()

            request = SearchRequest(query="test query", k=1, use_reranker=True)

            result = await search_post(request)

            # Should still return results with fallback content
            assert len(result.results) == 1
            # Verify reranking was attempted with fallback content
            mock_model_manager.rerank_async.assert_called_once()
            call_args = mock_model_manager.rerank_async.call_args
            documents = call_args[1]["documents"]
            assert "Document from" in documents[0]  # Fallback content

    @pytest.mark.asyncio()
    async def test_search_with_reranking_failure(self, mock_settings, mock_qdrant_client, mock_model_manager) -> None:
        """Test graceful fallback when reranking fails."""
        mock_settings.USE_MOCK_EMBEDDINGS = False

        with (
            patch("vecpipe.search_api.qdrant_client", mock_qdrant_client),
            patch("vecpipe.search_api.model_manager", mock_model_manager),
            patch("vecpipe.search_api.get_reranker_for_embedding_model") as mock_get_reranker,
            patch("vecpipe.search_api.search_qdrant") as mock_search_qdrant,
        ):
            mock_get_reranker.return_value = "test-reranker"

            # Mock collection info
            mock_qdrant_client.get.return_value.json.return_value = {
                "result": {"config": {"params": {"vectors": {"size": 1024}}}}
            }
            mock_qdrant_client.get.return_value.raise_for_status = AsyncMock()

            # Mock search results
            mock_search_qdrant.return_value = [
                {
                    "id": "1",
                    "score": 0.95,
                    "payload": {
                        "path": "/test/file1.txt",
                        "chunk_id": "chunk-1",
                        "doc_id": "doc-1",
                        "content": "Content 1",
                    },
                },
                {
                    "id": "2",
                    "score": 0.90,
                    "payload": {
                        "path": "/test/file2.txt",
                        "chunk_id": "chunk-2",
                        "doc_id": "doc-2",
                        "content": "Content 2",
                    },
                },
            ]
            mock_qdrant_client.post.return_value.raise_for_status = AsyncMock()

            # Make reranking fail (but not with InsufficientMemoryError)
            mock_model_manager.rerank_async.side_effect = Exception("Reranking model error")

            request = SearchRequest(query="test query", k=2, use_reranker=True)

            result = await search_post(request)

            # Should return original results without reranking
            assert len(result.results) == 2
            assert result.results[0].score == 0.95  # Original scores
            assert result.results[1].score == 0.90
            assert result.reranking_used is True  # Was attempted
            assert result.reranker_model is None  # But failed

    # NOTE: test_hybrid_search_error_handling removed - legacy hybrid search deleted

    @pytest.mark.asyncio()
    async def test_batch_search_partial_failure(self, mock_settings, mock_qdrant_client, mock_model_manager) -> None:
        """Test batch search when some queries fail."""
        mock_settings.USE_MOCK_EMBEDDINGS = False

        with (
            patch("vecpipe.search_api.qdrant_client", mock_qdrant_client),
            patch("vecpipe.search_api.model_manager", mock_model_manager),
        ):
            # Make embedding generation fail for second query
            mock_model_manager.generate_embedding_async.side_effect = [
                [0.1] * 1024,  # Success for first query
                RuntimeError("Embedding failed"),  # Fail for second query
                [0.3] * 1024,  # Success for third query
            ]

            request = BatchSearchRequest(queries=["query1", "query2", "query3"], k=5)

            # The entire batch should fail if any query fails
            with pytest.raises(HTTPException) as exc_info:
                await batch_search(request)
            assert exc_info.value.status_code == 500

    @pytest.mark.asyncio()
    async def test_collection_metadata_error_handling(
        self, mock_settings, mock_qdrant_client, mock_model_manager
    ) -> None:
        """Test search when collection metadata fetch fails."""
        from vecpipe.search.cache import clear_cache

        mock_settings.USE_MOCK_EMBEDDINGS = False

        # Clear cache to ensure metadata fetch is attempted
        clear_cache()

        with (
            patch("vecpipe.search_api.qdrant_client", mock_qdrant_client),
            patch("vecpipe.search_api.model_manager", mock_model_manager),
            patch("qdrant_client.QdrantClient"),
            patch(
                "shared.database.collection_metadata.get_collection_metadata_async",
                new_callable=AsyncMock,
            ) as mock_get_metadata,
            patch("vecpipe.search_api.search_qdrant") as mock_search_qdrant,
        ):
            # Make metadata fetch fail
            mock_get_metadata.side_effect = Exception("Metadata fetch failed")

            # Mock collection info
            mock_qdrant_client.get.return_value.json.return_value = {
                "result": {"config": {"params": {"vectors": {"size": 1024}}}}
            }
            mock_qdrant_client.get.return_value.raise_for_status = AsyncMock()

            # Mock search results
            mock_search_qdrant.return_value = []

            request = SearchRequest(query="test", k=5)

            # Should continue with default model despite metadata error
            result = await search_post(request)
            assert result.model_used == "test-model/float32"

    @pytest.mark.asyncio()
    async def test_upsert_with_http_error_parsing(self, mock_qdrant_client) -> None:
        """Test upsert error parsing when response format is unexpected."""
        with patch("vecpipe.search_api.qdrant_client", mock_qdrant_client):
            # Mock collection info first
            mock_get_response = Mock()
            mock_get_response.json.return_value = {"result": {"config": {"params": {"vectors": {"size": 768}}}}}
            mock_get_response.raise_for_status = Mock()
            mock_qdrant_client.get.return_value = mock_get_response

            # Mock error without expected format
            error_response = Mock()
            error_response.json.side_effect = Exception("Invalid JSON")

            mock_qdrant_client.put.side_effect = httpx.HTTPStatusError(
                "Server error", request=Mock(), response=error_response
            )

            request = UpsertRequest(
                collection_name="test",
                points=[
                    UpsertPoint(
                        id="1",
                        vector=[0.1] * 768,
                        payload=PointPayload(doc_id="doc-1", chunk_id="chunk-1", path="/test.txt"),
                    )
                ],
            )

            with pytest.raises(HTTPException) as exc_info:
                await upsert_points(request)
            assert exc_info.value.status_code == 502
            assert "Vector database error" in str(exc_info.value.detail)

    @pytest.mark.asyncio()
    async def test_search_with_invalid_collection_info(
        self, mock_settings, mock_qdrant_client, mock_model_manager
    ) -> None:
        """Test search when collection info has unexpected format."""
        mock_settings.USE_MOCK_EMBEDDINGS = False

        with (
            patch("vecpipe.search_api.qdrant_client", mock_qdrant_client),
            patch("vecpipe.search_api.model_manager", mock_model_manager),
            patch("vecpipe.search_api.search_qdrant") as mock_search_qdrant,
        ):
            # Mock collection info with missing/invalid structure
            mock_qdrant_client.get.return_value.json.return_value = {
                "result": {
                    # Missing config.params.vectors.size
                    "points_count": 100
                }
            }
            mock_qdrant_client.get.return_value.raise_for_status = AsyncMock()

            # Mock search results
            mock_search_qdrant.return_value = []

            request = SearchRequest(query="test", k=5)

            # Should use default dimension
            result = await search_post(request)
            assert result is not None

            # Verify default dimension was used for embedding
            call_args = mock_model_manager.generate_embedding_async.call_args
            assert call_args is not None

    @pytest.mark.asyncio()
    async def test_search_with_custom_reranker_params(
        self, mock_settings, mock_qdrant_client, mock_model_manager
    ) -> None:
        """Test search with custom reranker model and quantization."""
        mock_settings.USE_MOCK_EMBEDDINGS = False

        with (
            patch("vecpipe.search_api.qdrant_client", mock_qdrant_client),
            patch("vecpipe.search_api.model_manager", mock_model_manager),
            patch("vecpipe.search_api.search_qdrant") as mock_search_qdrant,
        ):
            # Mock collection info
            mock_qdrant_client.get.return_value.json.return_value = {
                "result": {"config": {"params": {"vectors": {"size": 1024}}}}
            }
            mock_qdrant_client.get.return_value.raise_for_status = AsyncMock()

            # Mock search results
            mock_search_qdrant.return_value = [
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
            mock_qdrant_client.post.return_value.raise_for_status = AsyncMock()

            # Make reranking return valid results for 1 document
            mock_model_manager.rerank_async.return_value = [(0, 0.95)]

            request = SearchRequest(
                query="test query",
                k=1,
                use_reranker=True,
                rerank_model="custom-reranker",
                rerank_quantization="int8",
            )

            result = await search_post(request)

            # Verify custom reranker params were used
            mock_model_manager.rerank_async.assert_called_once()
            call_args = mock_model_manager.rerank_async.call_args
            assert call_args[1]["model_name"] == "custom-reranker"
            assert call_args[1]["quantization"] == "int8"
            assert result.reranker_model == "custom-reranker/int8"

    @pytest.mark.asyncio()
    async def test_search_with_large_k_and_reranking(
        self, mock_settings, mock_qdrant_client, mock_model_manager
    ) -> None:
        """Test search with large k value and reranking limits."""
        mock_settings.USE_MOCK_EMBEDDINGS = False

        with (
            patch("vecpipe.search_api.qdrant_client", mock_qdrant_client),
            patch("vecpipe.search_api.model_manager", mock_model_manager),
            patch("vecpipe.search_api.get_reranker_for_embedding_model") as mock_get_reranker,
            patch("vecpipe.search_api.search_qdrant") as mock_search_qdrant,
        ):
            mock_get_reranker.return_value = "test-reranker"

            # Mock collection info
            mock_qdrant_client.get.return_value.json.return_value = {
                "result": {"config": {"params": {"vectors": {"size": 1024}}}}
            }
            mock_qdrant_client.get.return_value.raise_for_status = AsyncMock()

            # Create many mock results
            mock_results = []
            for i in range(200):  # Max candidates limit
                mock_results.append(
                    {
                        "id": f"id-{i}",
                        "score": 0.9 - (i * 0.001),
                        "payload": {
                            "path": f"/test/file{i}.txt",
                            "chunk_id": f"chunk-{i}",
                            "doc_id": f"doc-{i}",
                            "content": f"Content {i}",
                        },
                    }
                )

            mock_search_qdrant.return_value = mock_results

            # Return reranked indices for top 50
            mock_model_manager.rerank_async.return_value = [(i, 0.99 - (i * 0.01)) for i in range(50)]

            request = SearchRequest(query="test query", k=50, use_reranker=True)  # Request 50 results

            result = await search_post(request)

            # Should return exactly k results
            assert len(result.results) == 50
            # Verify search_k was capped at max_candidates (200)
            call_args = mock_search_qdrant.call_args
            assert call_args[0][4] == 200  # max_candidates (5th arg to search_qdrant)

    @pytest.mark.asyncio()
    async def test_search_with_minimal_query(self, mock_settings, mock_qdrant_client, mock_model_manager) -> None:
        """Test search with minimal query (single character)."""
        mock_settings.USE_MOCK_EMBEDDINGS = False

        with (
            patch("vecpipe.search_api.qdrant_client", mock_qdrant_client),
            patch("vecpipe.search_api.model_manager", mock_model_manager),
            patch("vecpipe.search_api.search_qdrant") as mock_search_qdrant,
        ):
            # Mock collection info
            mock_qdrant_client.get.return_value.json.return_value = {
                "result": {"config": {"params": {"vectors": {"size": 1024}}}}
            }
            mock_qdrant_client.get.return_value.raise_for_status = AsyncMock()

            # Mock to return embedding for minimal text
            mock_model_manager.generate_embedding_async.return_value = [0.1] * 1024

            # Mock search results
            mock_search_qdrant.return_value = []

            request = SearchRequest(query="a", k=5)  # Single character - minimum valid query

            # Should handle gracefully
            result = await search_post(request)
            assert result.query == "a"
            assert result.num_results == 0  # No results found

    @pytest.mark.asyncio()
    async def test_generate_embedding_async_with_no_model_manager(self, mock_settings) -> None:
        """Test generate_embedding_async when model manager is not initialized."""
        mock_settings.USE_MOCK_EMBEDDINGS = False

        # Temporarily set model_manager to None
        original_manager = search_api_module.model_manager
        search_api_module.model_manager = None

        try:
            with pytest.raises(RuntimeError) as exc_info:
                await search_api_module.generate_embedding_async("test text")
            assert "Model manager not initialized" in str(exc_info.value)
        finally:
            # Restore original
            search_api_module.model_manager = original_manager

    @pytest.mark.asyncio()
    async def test_search_faiss_fallback(self, mock_settings, mock_qdrant_client, mock_model_manager) -> None:
        """Test search falling back to FAISS when Qdrant is unavailable."""
        mock_settings.USE_MOCK_EMBEDDINGS = False

        with (
            patch("vecpipe.search_api.qdrant_client", mock_qdrant_client),
            patch("vecpipe.search_api.model_manager", mock_model_manager),
            patch("vecpipe.search_api.search_qdrant") as mock_search_qdrant,
        ):
            # Mock collection info
            mock_qdrant_client.get.return_value.json.return_value = {
                "result": {"config": {"params": {"vectors": {"size": 1024}}}}
            }
            mock_qdrant_client.get.return_value.raise_for_status = AsyncMock()

            # Make Qdrant unavailable
            mock_search_qdrant.side_effect = httpx.ConnectError("Connection refused")

            # Currently, the search_api doesn't have FAISS fallback implemented
            # This test documents the expected behavior when it's added
            request = SearchRequest(query="test", k=5)

            with pytest.raises(HTTPException) as exc_info:
                await search_post(request)
            # The general exception handler catches all errors and returns 500
            assert exc_info.value.status_code == 500

    # NOTE: test_keyword_search_cleanup removed - use search_mode="sparse" instead

    def test_collection_info_error(self, mock_qdrant_client, test_client_for_search_api) -> None:
        """Test collection info endpoint error handling."""

        # Temporarily set qdrant_client to None
        original_client = search_api_module.qdrant_client
        search_api_module.qdrant_client = None

        try:
            response = test_client_for_search_api.get("/collection/info")
            # The endpoint catches all exceptions including HTTPException and re-raises as 502
            assert response.status_code == 502
            assert "Failed to get collection info" in response.json()["detail"]
        finally:
            # Restore original
            search_api_module.qdrant_client = original_client

    def test_models_load_error(self, mock_settings, mock_embedding_service, test_client_for_search_api) -> None:
        """Test /models/load endpoint error handling."""
        mock_settings.USE_MOCK_EMBEDDINGS = False

        # First test with mock embeddings enabled
        mock_settings.USE_MOCK_EMBEDDINGS = True
        response = test_client_for_search_api.post(
            "/models/load", json={"model_name": "test-model", "quantization": "float32"}
        )
        assert response.status_code == 400
        assert "Cannot load models when using mock embeddings" in response.json()["detail"]

        # Now test with model manager not initialized
        mock_settings.USE_MOCK_EMBEDDINGS = False

        # Temporarily set model_manager to None
        original_manager = search_api_module.model_manager
        search_api_module.model_manager = None

        try:
            response = test_client_for_search_api.post(
                "/models/load", json={"model_name": "test-model", "quantization": "float32"}
            )
            # The endpoint catches HTTPException and re-raises as 500
            assert response.status_code == 500
            assert "Model load failed" in response.json()["detail"]
        finally:
            # Restore original
            search_api_module.model_manager = original_manager

    @pytest.mark.asyncio()
    async def test_search_with_score_threshold(self, mock_settings, mock_qdrant_client, mock_model_manager) -> None:
        """Test search respects score threshold in results."""
        mock_settings.USE_MOCK_EMBEDDINGS = False

        with (
            patch("vecpipe.search_api.qdrant_client", mock_qdrant_client),
            patch("vecpipe.search_api.model_manager", mock_model_manager),
            patch("vecpipe.search_api.search_qdrant") as mock_search_qdrant,
        ):
            # Mock collection info
            mock_qdrant_client.get.return_value.json.return_value = {
                "result": {"config": {"params": {"vectors": {"size": 1024}}}}
            }
            mock_qdrant_client.get.return_value.raise_for_status = AsyncMock()

            # Mock search results with varying scores
            mock_search_qdrant.return_value = [
                {"id": "1", "score": 0.95, "payload": {"path": "/high.txt", "chunk_id": "c1", "doc_id": "d1"}},
                {
                    "id": "2",
                    "score": 0.75,
                    "payload": {"path": "/medium.txt", "chunk_id": "c2", "doc_id": "d2"},
                },
                {"id": "3", "score": 0.55, "payload": {"path": "/low.txt", "chunk_id": "c3", "doc_id": "d3"}},
            ]
            mock_qdrant_client.post.return_value.raise_for_status = AsyncMock()

            request = SearchRequest(query="test query", k=10, score_threshold=0.7)  # Should filter out the last result

            result = await search_post(request)

            # score_threshold filtering is now implemented - results below 0.7 are excluded
            assert len(result.results) == 2  # Only results with score >= 0.7 are returned
            assert result.results[0].score == 0.95  # /high.txt
            assert result.results[1].score == 0.75  # /medium.txt
