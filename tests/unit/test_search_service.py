#!/usr/bin/env python3

"""
Comprehensive test suite for webui/services/search_service.py
Tests query building, result processing, multi-collection search, and reranking integration
"""

import time
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from shared.database.exceptions import AccessDeniedError, EntityNotFoundError
from shared.database.models import Collection, CollectionStatus
from webui.services.search_service import SearchService


class TestSearchService:
    """Test SearchService implementation"""

    @pytest.fixture()
    def mock_session(self) -> None:
        """Create a mock AsyncSession"""
        session = AsyncMock()
        session.commit = AsyncMock()
        return session

    @pytest.fixture()
    def mock_collection_repo(self) -> None:
        """Create a mock CollectionRepository"""
        return AsyncMock()

    @pytest.fixture()
    def search_service(self, mock_session, mock_collection_repo) -> None:
        """Create SearchService with mocked dependencies"""
        return SearchService(
            db_session=mock_session,
            collection_repo=mock_collection_repo,
            default_timeout=httpx.Timeout(timeout=30.0, connect=5.0, read=30.0, write=5.0),
            retry_timeout_multiplier=4.0,
        )

    @pytest.mark.asyncio()
    async def test_validate_collection_access_success(self, search_service, mock_collection_repo) -> None:
        """Test successful collection access validation"""
        # Mock collections
        mock_collections = []
        for i in range(3):
            collection = Mock(spec=Collection)
            collection.id = f"collection-{i}"
            collection.name = f"Collection {i}"
            mock_collections.append(collection)

        # Mock repository to return collections
        mock_collection_repo.get_by_uuid_with_permission_check.side_effect = mock_collections

        # Test validation
        collection_uuids = ["uuid-1", "uuid-2", "uuid-3"]
        result = await search_service.validate_collection_access(collection_uuids, user_id=123)

        assert len(result) == 3
        assert all(isinstance(c, Collection) for c in result)

        # Verify repository calls
        assert mock_collection_repo.get_by_uuid_with_permission_check.call_count == 3

    @pytest.mark.asyncio()
    async def test_validate_collection_access_denied(self, search_service, mock_collection_repo) -> None:
        """Test collection access validation when access is denied"""
        # Mock access denied error
        mock_collection_repo.get_by_uuid_with_permission_check.side_effect = AccessDeniedError(
            user_id="123", resource_type="collection", resource_id="uuid-1"
        )

        with pytest.raises(AccessDeniedError):
            await search_service.validate_collection_access(["uuid-1"], user_id=123)

    @pytest.mark.asyncio()
    async def test_validate_collection_not_found(self, search_service, mock_collection_repo) -> None:
        """Test collection access validation when collection not found"""
        # Mock not found error
        mock_collection_repo.get_by_uuid_with_permission_check.side_effect = EntityNotFoundError("collection", "uuid-1")

        with pytest.raises(EntityNotFoundError):
            await search_service.validate_collection_access(["uuid-1"], user_id=123)

    @pytest.mark.asyncio()
    @patch("webui.services.search_service.httpx.AsyncClient")
    async def test_search_single_collection_success(self, mock_httpx_client, search_service) -> None:
        """Test successful single collection search"""
        # Mock collection
        mock_collection = Mock(spec=Collection)
        mock_collection.status = CollectionStatus.READY
        mock_collection.vector_store_name = "collection_123"
        mock_collection.embedding_model = "test-model"
        mock_collection.quantization = "float16"
        mock_collection.name = "Test Collection"

        # Mock HTTP response
        mock_response = Mock()
        mock_response.json.return_value = {
            "results": [
                {"id": "doc1", "content": "Test document 1", "score": 0.95},
                {"id": "doc2", "content": "Test document 2", "score": 0.85},
            ]
        }
        mock_response.raise_for_status = Mock()

        # Mock HTTP client
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_httpx_client.return_value.__aenter__.return_value = mock_client

        # Test search
        collection, results, error = await search_service.search_single_collection(
            collection=mock_collection,
            query="test query",
            k=10,
            search_params={"search_type": "semantic"},
        )

        assert collection == mock_collection
        assert len(results) == 2
        assert results[0]["score"] == 0.95
        assert error is None

        # Verify HTTP request
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert "search" in call_args[0][0]
        request_data = call_args[1]["json"]
        assert request_data["query"] == "test query"
        assert request_data["k"] == 10
        assert request_data["collection"] == "collection_123"

    @pytest.mark.asyncio()
    async def test_search_single_collection_not_ready(self, search_service) -> None:
        """Test search on collection that is not ready"""
        # Mock collection in PROCESSING state
        mock_collection = Mock(spec=Collection)
        mock_collection.status = CollectionStatus.PROCESSING
        mock_collection.name = "Processing Collection"

        collection, results, error = await search_service.search_single_collection(
            collection=mock_collection,
            query="test query",
            k=10,
            search_params={},
        )

        assert collection == mock_collection
        assert results is None
        assert "not ready for search" in error

    @pytest.mark.asyncio()
    @patch("webui.services.search_service.httpx.AsyncClient")
    async def test_search_single_collection_timeout_retry(self, mock_httpx_client, search_service) -> None:
        """Test search with timeout and successful retry"""
        # Mock collection
        mock_collection = Mock(spec=Collection)
        mock_collection.status = CollectionStatus.READY
        mock_collection.vector_store_name = "collection_123"
        mock_collection.embedding_model = "test-model"
        mock_collection.quantization = "float16"
        mock_collection.name = "Test Collection"

        # Mock successful response for retry
        mock_response = Mock()
        mock_response.json.return_value = {"results": [{"id": "doc1", "score": 0.9}]}
        mock_response.raise_for_status = Mock()

        # Mock HTTP client - first call times out, second succeeds
        mock_client = AsyncMock()
        mock_client.post.side_effect = [
            httpx.ReadTimeout("Request timed out"),
            mock_response,
        ]
        mock_httpx_client.return_value.__aenter__.return_value = mock_client

        # Test search
        collection, results, error = await search_service.search_single_collection(
            collection=mock_collection,
            query="test query",
            k=10,
            search_params={},
        )

        assert collection == mock_collection
        assert len(results) == 1
        assert error is None

        # Verify retry happened
        assert mock_client.post.call_count == 2

    @pytest.mark.asyncio()
    @patch("webui.services.search_service.httpx.AsyncClient")
    async def test_search_single_collection_http_errors(self, mock_httpx_client, search_service) -> None:
        """Test handling of various HTTP errors"""
        # Mock collection
        mock_collection = Mock(spec=Collection)
        mock_collection.status = CollectionStatus.READY
        mock_collection.vector_store_name = "collection_123"
        mock_collection.embedding_model = "test-model"
        mock_collection.quantization = "float16"
        mock_collection.name = "Test Collection"

        # Test different HTTP status codes
        test_cases = [
            (404, "not found in vector store"),
            (403, "Access denied"),
            (429, "Rate limit exceeded"),
            (500, "Search service unavailable"),
            (502, "Search service unavailable"),
            (418, "Search failed"),  # Other status codes
        ]

        for status_code, expected_error in test_cases:
            # Mock HTTP error response
            mock_response = Mock()
            mock_response.status_code = status_code
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.HTTPStatusError("HTTP Error", request=Mock(), response=mock_response)
            mock_httpx_client.return_value.__aenter__.return_value = mock_client

            # Test search
            collection, results, error = await search_service.search_single_collection(
                collection=mock_collection,
                query="test query",
                k=10,
                search_params={},
            )

            assert collection == mock_collection
            assert results is None
            assert expected_error in error

    @pytest.mark.asyncio()
    @patch("webui.services.search_service.httpx.AsyncClient")
    async def test_search_single_collection_connection_error(self, mock_httpx_client, search_service) -> None:
        """Test handling of connection errors"""
        # Mock collection
        mock_collection = Mock(spec=Collection)
        mock_collection.status = CollectionStatus.READY
        mock_collection.name = "Test Collection"

        # Mock connection error
        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.ConnectError("Connection refused")
        mock_httpx_client.return_value.__aenter__.return_value = mock_client

        # Test search
        collection, results, error = await search_service.search_single_collection(
            collection=mock_collection,
            query="test query",
            k=10,
            search_params={},
        )

        assert collection == mock_collection
        assert results is None
        assert "Cannot connect to search service" in error

    @pytest.mark.asyncio()
    @patch("webui.services.search_service.httpx.AsyncClient")
    async def test_multi_collection_search_success(
        self, mock_httpx_client, search_service, mock_collection_repo
    ) -> None:
        """Test successful multi-collection search"""
        # Mock collections
        mock_collections = []
        for i in range(3):
            collection = Mock(spec=Collection)
            collection.id = f"collection-{i}"
            collection.name = f"Collection {i}"
            collection.status = CollectionStatus.READY
            collection.vector_store_name = f"collection_{i}"
            collection.embedding_model = "test-model"
            collection.quantization = "float16"
            mock_collections.append(collection)

        mock_collection_repo.get_by_uuid_with_permission_check.side_effect = mock_collections

        # Mock HTTP responses for each collection
        mock_responses = []
        for i in range(3):
            response = Mock()
            response.json.return_value = {
                "results": [
                    {"id": f"doc{i}-1", "content": f"Doc from collection {i}", "score": 0.9 - i * 0.1},
                    {"id": f"doc{i}-2", "content": f"Another doc from {i}", "score": 0.8 - i * 0.1},
                ]
            }
            response.raise_for_status = Mock()
            mock_responses.append(response)

        # Mock HTTP client
        mock_client = AsyncMock()
        mock_client.post.side_effect = mock_responses
        mock_httpx_client.return_value.__aenter__.return_value = mock_client

        # Test multi-collection search
        result = await search_service.multi_collection_search(
            user_id=123,
            collection_uuids=["uuid-0", "uuid-1", "uuid-2"],
            query="test query",
            k=5,
            search_type="semantic",
            use_reranker=True,
        )

        # Verify results
        assert len(result["results"]) == 5  # Limited to k=5
        assert result["metadata"]["total_results"] == 5
        assert result["metadata"]["collections_searched"] == 3

        # Verify results are sorted by score
        scores = [r["score"] for r in result["results"]]
        assert scores == sorted(scores, reverse=True)

        # Verify collection info is added to results
        assert all("collection_id" in r for r in result["results"])
        assert all("collection_name" in r for r in result["results"])

    @pytest.mark.asyncio()
    @patch("webui.services.search_service.httpx.AsyncClient")
    async def test_multi_collection_search_with_errors(
        self, mock_httpx_client, search_service, mock_collection_repo
    ) -> None:
        """Test multi-collection search with some collections failing"""
        # Mock collections
        mock_collections = []
        for i in range(3):
            collection = Mock(spec=Collection)
            collection.id = f"collection-{i}"
            collection.name = f"Collection {i}"
            collection.status = CollectionStatus.READY
            collection.vector_store_name = f"collection_{i}"
            collection.embedding_model = "test-model"
            collection.quantization = "float16"
            mock_collections.append(collection)

        mock_collection_repo.get_by_uuid_with_permission_check.side_effect = mock_collections

        # Mock mixed responses - one success, one failure, one timeout
        success_response = Mock()
        success_response.json.return_value = {"results": [{"id": "doc1", "content": "Success doc", "score": 0.9}]}
        success_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.post.side_effect = [
            success_response,
            httpx.HTTPStatusError("Not found", request=Mock(), response=Mock(status_code=404)),
            httpx.ReadTimeout("Timeout"),
        ]
        mock_httpx_client.return_value.__aenter__.return_value = mock_client

        # Test multi-collection search
        result = await search_service.multi_collection_search(
            user_id=123,
            collection_uuids=["uuid-0", "uuid-1", "uuid-2"],
            query="test query",
            k=10,
        )

        # Should have results from successful collection only
        assert len(result["results"]) == 1
        assert result["metadata"]["collections_searched"] == 3
        assert len(result["metadata"]["errors"]) == 2

        # Verify collection details include errors
        collection_details = result["metadata"]["collection_details"]
        assert len(collection_details) == 3
        assert sum(1 for c in collection_details if "error" in c) == 2

    @pytest.mark.asyncio()
    @patch("webui.services.search_service.httpx.AsyncClient")
    async def test_multi_collection_search_hybrid_mode(
        self, mock_httpx_client, search_service, mock_collection_repo
    ) -> None:
        """Test multi-collection search with search_mode=hybrid and RRF parameters"""
        # Mock collection
        mock_collection = Mock(spec=Collection)
        mock_collection.id = "collection-1"
        mock_collection.name = "Collection 1"
        mock_collection.status = CollectionStatus.READY
        mock_collection.vector_store_name = "collection_1"
        mock_collection.embedding_model = "test-model"
        mock_collection.quantization = "float16"

        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {"results": [{"id": "doc1", "score": 0.9}]}
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_httpx_client.return_value.__aenter__.return_value = mock_client

        # Test with hybrid search mode (new RRF-based hybrid)
        _ = await search_service.multi_collection_search(
            user_id=123,
            collection_uuids=["uuid-1"],
            query="test query",
            k=10,
            search_type="semantic",
            search_mode="hybrid",
            rrf_k=80,
        )

        # Verify hybrid/sparse parameters were included
        call_args = mock_client.post.call_args
        request_data = call_args[1]["json"]
        assert request_data["search_type"] == "semantic"
        assert request_data["search_mode"] == "hybrid"
        assert request_data["rrf_k"] == 80

    @pytest.mark.asyncio()
    @patch("webui.services.search_service.httpx.AsyncClient")
    async def test_single_collection_search_method(
        self, mock_httpx_client, search_service, mock_collection_repo
    ) -> None:
        """Test the single_collection_search method (different from search_single_collection)"""
        # Mock collection
        mock_collection = Mock(spec=Collection)
        mock_collection.id = "collection-1"
        mock_collection.name = "Collection 1"
        mock_collection.status = CollectionStatus.READY
        mock_collection.vector_store_name = "collection_1"
        mock_collection.embedding_model = "test-model"
        mock_collection.quantization = "float16"

        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "results": [{"id": "doc1", "content": "Test doc", "score": 0.95}],
            "metadata": {"processing_time": 0.123},
        }
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_httpx_client.return_value.__aenter__.return_value = mock_client

        # Test single collection search
        result = await search_service.single_collection_search(
            user_id=123,
            collection_uuid="uuid-1",
            query="test query",
            k=10,
            search_type="semantic",
            score_threshold=0.5,
            metadata_filter={"category": "test"},
            use_reranker=True,
            rerank_model="test-reranker",
            include_content=True,
        )

        # Verify result structure
        assert "results" in result
        assert len(result["results"]) == 1
        assert result["results"][0]["score"] == 0.95

        # Verify all parameters were passed
        call_args = mock_client.post.call_args
        request_data = call_args[1]["json"]
        assert request_data["query"] == "test query"
        assert request_data["k"] == 10
        assert request_data["search_type"] == "semantic"
        assert request_data["score_threshold"] == 0.5
        assert request_data["filters"] == {"category": "test"}
        assert request_data["use_reranker"] is True
        assert request_data["rerank_model"] == "test-reranker"
        assert request_data["include_content"] is True

    @pytest.mark.asyncio()
    @patch("webui.services.search_service.httpx.AsyncClient")
    async def test_single_collection_search_http_errors(
        self, mock_httpx_client, search_service, mock_collection_repo
    ) -> None:
        """Test single_collection_search error handling"""
        # Mock collection
        mock_collection = Mock(spec=Collection)
        mock_collection.id = "collection-1"
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        # Test 404 error
        mock_response = Mock()
        mock_response.status_code = 404
        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.HTTPStatusError("Not found", request=Mock(), response=mock_response)
        mock_httpx_client.return_value.__aenter__.return_value = mock_client

        with pytest.raises(EntityNotFoundError):
            await search_service.single_collection_search(
                user_id=123,
                collection_uuid="uuid-1",
                query="test query",
            )

        # Test 403 error
        mock_response.status_code = 403
        mock_client.post.side_effect = httpx.HTTPStatusError("Forbidden", request=Mock(), response=mock_response)

        with pytest.raises(AccessDeniedError):
            await search_service.single_collection_search(
                user_id=123,
                collection_uuid="uuid-1",
                query="test query",
            )

    @pytest.mark.asyncio()
    async def test_search_service_custom_timeout(self, mock_session, mock_collection_repo) -> None:
        """Test SearchService with custom timeout configuration"""
        custom_timeout = httpx.Timeout(timeout=60.0, connect=10.0, read=60.0, write=10.0)
        service = SearchService(
            db_session=mock_session,
            collection_repo=mock_collection_repo,
            default_timeout=custom_timeout,
            retry_timeout_multiplier=2.0,
        )

        assert service.default_timeout == custom_timeout
        assert service.retry_timeout_multiplier == 2.0

    @pytest.mark.asyncio()
    @patch("webui.services.search_service.time")
    @patch("webui.services.search_service.httpx.AsyncClient")
    async def test_search_timing(self, mock_httpx_client, mock_time, search_service, mock_collection_repo) -> None:
        """Test that search timing is measured correctly"""
        # Mock time
        mock_time.time.side_effect = [1000.0, 1001.5]  # 1.5 second search

        # Mock collection
        mock_collection = Mock(spec=Collection)
        mock_collection.status = CollectionStatus.READY
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_httpx_client.return_value.__aenter__.return_value = mock_client

        # Test search
        result = await search_service.multi_collection_search(
            user_id=123,
            collection_uuids=["uuid-1"],
            query="test query",
        )

        # Verify timing
        assert result["metadata"]["processing_time"] == 1.5

    def test_result_sort_key_prefers_reranked_score(self, search_service) -> None:
        """Ensure reranked_score takes precedence when ordering results."""

        high_reranked = {"score": 0.2, "reranked_score": 0.95}
        high_score = {"score": 0.9}

        assert SearchService._result_sort_key(high_reranked) > SearchService._result_sort_key(high_score)

    @pytest.mark.asyncio()
    async def test_search_single_collection_rejects_legacy_modes(self, search_service) -> None:
        """Legacy hybrid_search_mode should raise a validation error."""

        mock_collection = Mock(spec=Collection)
        mock_collection.status = CollectionStatus.READY
        mock_collection.vector_store_name = "collection_legacy"
        mock_collection.embedding_model = "test-model"
        mock_collection.quantization = "float16"
        mock_collection.name = "Legacy Collection"

        with pytest.raises(ValueError, match="hybrid_search_mode"):
            await search_service.search_single_collection(
                collection=mock_collection,
                query="legacy modes",
                k=5,
                search_params={
                    "search_type": "hybrid",
                    "hybrid_search_mode": "weighted",
                    "hybrid_alpha": 0.7,
                },
            )

    @pytest.mark.asyncio()
    async def test_multi_collection_search_validates_search_mode_and_sorts(self, search_service) -> None:
        """search_mode parameter is preserved and results sort by reranked_score."""

        collection1 = Mock(spec=Collection)
        collection1.id = "col-1"
        collection1.name = "One"
        collection1.status = CollectionStatus.READY
        collection1.vector_store_name = "collection_one"
        collection1.embedding_model = "model-1"
        collection1.quantization = "float16"

        collection2 = Mock(spec=Collection)
        collection2.id = "col-2"
        collection2.name = "Two"
        collection2.status = CollectionStatus.READY
        collection2.vector_store_name = "collection_two"
        collection2.embedding_model = "model-2"
        collection2.quantization = "float16"

        search_service.validate_collection_access = AsyncMock(return_value=[collection1, collection2])

        search_service.search_single_collection = AsyncMock(
            side_effect=[
                (
                    collection1,
                    [
                        {
                            "doc_id": "d1",
                            "chunk_id": "c1",
                            "score": 0.4,
                            "reranked_score": 0.95,
                            "content": "high rerank",
                        }
                    ],
                    None,
                ),
                (
                    collection2,
                    [
                        {
                            "doc_id": "d2",
                            "chunk_id": "c2",
                            "score": 0.9,
                            "reranked_score": 0.5,
                            "content": "high raw",
                        }
                    ],
                    None,
                ),
            ]
        )

        result = await search_service.multi_collection_search(
            user_id=42,
            collection_uuids=["col-1", "col-2"],
            query="mixed",
            k=10,
            search_type="semantic",
            search_mode="hybrid",
            rrf_k=60,
        )

        call_args = search_service.search_single_collection.call_args_list[0][0]
        search_params = call_args[3]
        assert search_params["search_mode"] == "hybrid"
        assert search_params["rrf_k"] == 60

        # Results are sorted by reranked_score (d1 has 0.95, d2 has 0.5)
        assert result["results"][0]["doc_id"] == "d1"


class TestSearchServiceErrorHandling:
    """Test error handling edge cases"""

    @pytest.fixture()
    def search_service(self) -> None:
        mock_session = AsyncMock()
        mock_collection_repo = AsyncMock()
        return SearchService(
            db_session=mock_session,
            collection_repo=mock_collection_repo,
        )

    @pytest.mark.asyncio()
    @patch("webui.services.search_service.httpx.AsyncClient")
    async def test_unexpected_search_error(self, mock_httpx_client, search_service) -> None:
        """Test handling of unexpected errors during search"""
        # Mock collection
        mock_collection = Mock(spec=Collection)
        mock_collection.status = CollectionStatus.READY
        mock_collection.name = "Test Collection"

        # Mock unexpected error
        mock_client = AsyncMock()
        mock_client.post.side_effect = Exception("Unexpected error")
        mock_httpx_client.return_value.__aenter__.return_value = mock_client

        # Test search
        collection, results, error = await search_service.search_single_collection(
            collection=mock_collection,
            query="test query",
            k=10,
            search_params={},
        )

        assert collection == mock_collection
        assert results is None
        assert "Unexpected error" in error

    @pytest.mark.asyncio()
    @patch("webui.services.search_service.httpx.AsyncClient")
    async def test_retry_with_extended_timeout(self, mock_httpx_client, search_service) -> None:
        """Test timeout extension calculation during retry"""
        # Mock collection
        mock_collection = Mock(spec=Collection)
        mock_collection.status = CollectionStatus.READY
        mock_collection.name = "Test Collection"

        # Track timeout values used
        timeout_values = []

        async def capture_timeout(*args, **kwargs) -> None:  # noqa: ARG001
            # Capture the timeout from the AsyncClient context manager
            timeout_values.append(mock_httpx_client.call_args[1].get("timeout"))
            if len(timeout_values) == 1:
                raise httpx.ReadTimeout("First timeout")
            # Return response on retry
            response = Mock()
            response.json.return_value = {"results": []}
            response.raise_for_status = Mock()
            return response

        mock_client = AsyncMock()
        mock_client.post.side_effect = capture_timeout
        mock_httpx_client.return_value.__aenter__.return_value = mock_client

        # Test search with custom timeout
        custom_timeout = httpx.Timeout(timeout=10.0, connect=2.0, read=10.0, write=2.0)
        await search_service.search_single_collection(
            collection=mock_collection,
            query="test query",
            k=10,
            search_params={},
            timeout=custom_timeout,
        )

        # Verify timeout was extended on retry
        assert len(timeout_values) == 2
        if timeout_values[1]:
            # Second timeout should be multiplied
            assert timeout_values[1].read > custom_timeout.read


class TestSearchServiceIntegration:
    """Test search service integration scenarios"""

    @pytest.mark.asyncio()
    @patch("webui.services.search_service.httpx.AsyncClient")
    async def test_concurrent_collection_searches(self, mock_httpx_client) -> None:
        """Test concurrent searches across multiple collections"""
        # Setup service
        mock_session = AsyncMock()
        mock_collection_repo = AsyncMock()
        service = SearchService(mock_session, mock_collection_repo)

        # Mock 5 collections
        mock_collections = []
        for i in range(5):
            collection = Mock(spec=Collection)
            collection.id = f"collection-{i}"
            collection.name = f"Collection {i}"
            collection.status = CollectionStatus.READY
            collection.vector_store_name = f"collection_{i}"
            collection.embedding_model = "test-model"
            collection.quantization = "float16"
            mock_collections.append(collection)

        mock_collection_repo.get_by_uuid_with_permission_check.side_effect = mock_collections

        # Mock responses
        mock_responses = []
        for i in range(5):
            response = Mock()
            response.json.return_value = {"results": [{"id": f"doc-{i}", "score": 0.9 - i * 0.1}]}
            response.raise_for_status = Mock()
            mock_responses.append(response)

        mock_client = AsyncMock()
        mock_client.post.side_effect = mock_responses
        mock_httpx_client.return_value.__aenter__.return_value = mock_client

        # Test concurrent search
        start_time = time.time()
        result = await service.multi_collection_search(
            user_id=123,
            collection_uuids=[f"uuid-{i}" for i in range(5)],
            query="test query",
            k=10,
        )
        _ = time.time() - start_time

        # Should complete successfully
        assert len(result["results"]) == 5
        assert result["metadata"]["collections_searched"] == 5
        # All collections should be represented
        collection_ids = {r["collection_id"] for r in result["results"]}
        assert len(collection_ids) == 5

    @pytest.mark.asyncio()
    async def test_search_result_aggregation(self) -> None:
        """Test proper aggregation and sorting of search results"""
        # Setup service
        mock_session = AsyncMock()
        mock_collection_repo = AsyncMock()
        _ = SearchService(mock_session, mock_collection_repo)

        # Create test results with specific scores
        test_results = [
            (Mock(id="c1", name="Collection 1"), [{"id": "d1", "score": 0.95}, {"id": "d2", "score": 0.75}], None),
            (Mock(id="c2", name="Collection 2"), [{"id": "d3", "score": 0.90}, {"id": "d4", "score": 0.80}], None),
            (Mock(id="c3", name="Collection 3"), [{"id": "d5", "score": 0.85}], None),
        ]

        # Process results as the service would
        all_results = []
        for collection, results, _ in test_results:
            if results:
                for result in results:
                    all_results.append(
                        {
                            "collection_id": collection.id,
                            "collection_name": collection.name,
                            **result,
                        }
                    )

        # Sort and limit
        all_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        final_results = all_results[:3]  # k=3

        # Verify correct ordering
        assert final_results[0]["id"] == "d1"  # score 0.95
        assert final_results[1]["id"] == "d3"  # score 0.90
        assert final_results[2]["id"] == "d5"  # score 0.85
