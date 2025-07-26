"""
Tests for v2 search API endpoints.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.requests import Request

from packages.shared.database.models import Collection, CollectionStatus
from packages.webui.api.v2.schemas import (
    CollectionSearchRequest,
    CollectionSearchResponse,
    SingleCollectionSearchRequest,
)
from packages.webui.api.v2.search import (
    multi_collection_search,
    single_collection_search,
)


@pytest.fixture()
def mock_user() -> dict[str, Any]:
    """Mock authenticated user."""
    return {"id": 1, "username": "testuser"}


@pytest.fixture()
def mock_collections() -> list[MagicMock]:
    """Mock collection objects."""
    collection1 = MagicMock(spec=Collection)
    collection1.id = "123e4567-e89b-12d3-a456-426614174000"
    collection1.name = "Documentation"
    collection1.vector_store_name = "qdrant_docs_collection"
    collection1.embedding_model = "Qwen/Qwen3-Embedding-0.6B"
    collection1.status = CollectionStatus.READY
    collection1.document_count = 100

    collection2 = MagicMock(spec=Collection)
    collection2.id = "456e7890-e89b-12d3-a456-426614174001"
    collection2.name = "Research Papers"
    collection2.vector_store_name = "qdrant_research_collection"
    collection2.embedding_model = "BAAI/bge-small-en-v1.5"
    collection2.status = CollectionStatus.READY
    collection2.document_count = 200

    return [collection1, collection2]


@pytest.fixture()
def mock_search_results() -> dict[str, list[dict[str, Any]]]:
    """Mock search results from Qdrant."""
    return {
        "results": [
            {
                "doc_id": "doc_123",
                "chunk_id": "chunk_456",
                "score": 0.95,
                "path": "/docs/auth_guide.md",
                "content": "To implement authentication, you can use JWT tokens...",
                "metadata": {"page": 1, "section": "Authentication"},
            },
            {
                "doc_id": "doc_234",
                "chunk_id": "chunk_567",
                "score": 0.85,
                "path": "/docs/api_reference.md",
                "content": "The authentication endpoint accepts POST requests...",
                "metadata": {"page": 5, "section": "API Reference"},
            },
        ]
    }


# Note: Internal functions (validate_collection_access, search_single_collection,
# rerank_merged_results) have been moved to SearchService and are tested separately.
# This file now tests only the API endpoints.


class TestMultiCollectionSearch:
    """Test multi-collection search endpoint."""

    @pytest.mark.asyncio()
    async def test_multi_collection_search_success(
        self, mock_user: dict[str, Any], mock_collections: list[MagicMock], mock_search_results: list[dict[str, Any]]
    ) -> None:
        """Test successful multi-collection search."""
        # Create a proper Request object with minimal required attributes
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/api/v2/search",
            "headers": [],
        }
        mock_request = Request(scope)
        mock_search_service = AsyncMock()

        search_request = CollectionSearchRequest(
            collection_uuids=[c.id for c in mock_collections],
            query="authentication",
            k=10,
            use_reranker=True,
            rerank_model=None,
            metadata_filter=None,
        )

        # Mock the service response
        mock_search_service.multi_collection_search.return_value = {
            "results": [
                {
                    "doc_id": "doc_123",
                    "chunk_id": "chunk_456",
                    "score": 0.98,
                    "reranked_score": 0.98,
                    "content": mock_search_results["results"][0]["content"],
                    "path": mock_search_results["results"][0]["path"],
                    "metadata": mock_search_results["results"][0]["metadata"],
                    "collection_id": mock_collections[0].id,
                    "collection_name": mock_collections[0].name,
                    "embedding_model": mock_collections[0].embedding_model,
                },
                {
                    "doc_id": "doc_234",
                    "chunk_id": "chunk_567",
                    "score": 0.92,
                    "reranked_score": 0.92,
                    "content": mock_search_results["results"][1]["content"],
                    "path": mock_search_results["results"][1]["path"],
                    "metadata": mock_search_results["results"][1]["metadata"],
                    "collection_id": mock_collections[1].id,
                    "collection_name": mock_collections[1].name,
                    "embedding_model": mock_collections[1].embedding_model,
                },
            ],
            "metadata": {
                "total_results": 2,
                "processing_time": 0.1,
                "collection_details": [
                    {
                        "collection_id": mock_collections[0].id,
                        "collection_name": mock_collections[0].name,
                        "result_count": 1,
                    },
                    {
                        "collection_id": mock_collections[1].id,
                        "collection_name": mock_collections[1].name,
                        "result_count": 1,
                    },
                ],
            },
        }

        with patch("packages.webui.api.v2.search.get_search_service", return_value=mock_search_service):
            response = await multi_collection_search(mock_request, search_request, mock_user, mock_search_service)

        assert isinstance(response, CollectionSearchResponse)
        assert response.query == "authentication"
        assert len(response.results) == 2
        assert response.total_results == 2
        assert response.reranking_used is True
        assert len(response.collections_searched) == 2
        assert response.partial_failure is False

    @pytest.mark.asyncio()
    async def test_multi_collection_search_partial_failure(self, mock_user, mock_collections):
        """Test multi-collection search with partial failures."""
        # Create a proper Request object with minimal required attributes
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/api/v2/search",
            "headers": [],
        }
        mock_request = Request(scope)
        mock_search_service = AsyncMock()

        search_request = CollectionSearchRequest(
            collection_uuids=[c.id for c in mock_collections],
            query="test",
            k=10,
            rerank_model=None,
            metadata_filter=None,
        )

        # Mock the service response with partial failure
        mock_search_service.multi_collection_search.return_value = {
            "results": [
                {
                    "doc_id": "1",
                    "chunk_id": "chunk_1",
                    "score": 0.9,
                    "content": "Test",
                    "path": "/test.txt",
                    "metadata": {},
                    "collection_id": mock_collections[0].id,
                    "collection_name": mock_collections[0].name,
                    "embedding_model": mock_collections[0].embedding_model,
                },
            ],
            "metadata": {
                "total_results": 1,
                "processing_time": 0.1,
                "errors": True,
                "collection_details": [
                    {
                        "collection_id": mock_collections[0].id,
                        "collection_name": mock_collections[0].name,
                        "result_count": 1,
                    },
                    {
                        "collection_id": mock_collections[1].id,
                        "collection_name": mock_collections[1].name,
                        "error": "Search failed: Connection error",
                    },
                ],
            },
        }

        with patch("packages.webui.api.v2.search.get_search_service", return_value=mock_search_service):
            response = await multi_collection_search(mock_request, search_request, mock_user, mock_search_service)

        assert response.partial_failure is True
        assert len(response.failed_collections) == 1
        assert response.failed_collections[0]["collection_id"] == str(mock_collections[1].id)
        assert "Connection error" in response.failed_collections[0]["error"]
        assert len(response.results) == 1  # Only results from successful collection

    @pytest.mark.asyncio()
    async def test_multi_collection_search_no_reranking_same_model(self, mock_user, mock_collections):
        """Test no re-ranking when all collections use the same model."""
        # Create a proper Request object with minimal required attributes
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/api/v2/search",
            "headers": [],
        }
        mock_request = Request(scope)
        mock_search_service = AsyncMock()

        # Set same model for both collections
        mock_collections[1].embedding_model = mock_collections[0].embedding_model

        search_request = CollectionSearchRequest(
            collection_uuids=[c.id for c in mock_collections],
            query="test",
            k=10,
            use_reranker=False,  # Explicitly disable
            rerank_model=None,
            metadata_filter=None,
        )

        # Mock the service response
        mock_search_service.multi_collection_search.return_value = {
            "results": [
                {
                    "doc_id": "1",
                    "chunk_id": "chunk_1",
                    "score": 0.9,
                    "content": "Test",
                    "path": "/test1.txt",
                    "metadata": {},
                    "collection_id": mock_collections[0].id,
                    "collection_name": mock_collections[0].name,
                    "embedding_model": mock_collections[0].embedding_model,
                },
                {
                    "doc_id": "2",
                    "chunk_id": "chunk_2",
                    "score": 0.8,
                    "content": "Test2",
                    "path": "/test2.txt",
                    "metadata": {},
                    "collection_id": mock_collections[1].id,
                    "collection_name": mock_collections[1].name,
                    "embedding_model": mock_collections[1].embedding_model,
                },
            ],
            "metadata": {
                "total_results": 2,
                "processing_time": 0.1,
                "collection_details": [
                    {
                        "collection_id": mock_collections[0].id,
                        "collection_name": mock_collections[0].name,
                        "result_count": 1,
                    },
                    {
                        "collection_id": mock_collections[1].id,
                        "collection_name": mock_collections[1].name,
                        "result_count": 1,
                    },
                ],
            },
        }

        with patch("packages.webui.api.v2.search.get_search_service", return_value=mock_search_service):
            response = await multi_collection_search(mock_request, search_request, mock_user, mock_search_service)

        assert response.reranking_used is False
        assert response.reranker_model is None
        assert len(response.results) == 2
        # Results should be sorted by score
        assert response.results[0].score == 0.9
        assert response.results[1].score == 0.8


class TestSearchReranking:
    """Test search reranking functionality."""

    @pytest.mark.asyncio()
    async def test_search_with_reranking_disabled(self, mock_user, mock_collections):
        """Test that search works correctly with reranking disabled."""
        # Create a proper Request object
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/api/v2/search",
            "headers": [],
        }
        mock_request = Request(scope)
        mock_search_service = AsyncMock()

        search_request = CollectionSearchRequest(
            collection_uuids=[mock_collections[0].id],
            query="test query",
            k=10,
            use_reranker=False,  # Explicitly disable reranking
            rerank_model=None,
            metadata_filter=None,
        )

        # Mock the service response without reranking
        mock_search_service.multi_collection_search.return_value = {
            "results": [
                {
                    "doc_id": "doc_1",
                    "chunk_id": "chunk_1",
                    "score": 0.85,
                    "content": "Test content without reranking",
                    "path": "/test.txt",
                    "metadata": {},
                    "collection_id": mock_collections[0].id,
                    "collection_name": mock_collections[0].name,
                    "embedding_model": mock_collections[0].embedding_model,
                },
            ],
            "metadata": {
                "total_results": 1,
                "processing_time": 0.05,
                "collection_details": [
                    {
                        "collection_id": mock_collections[0].id,
                        "collection_name": mock_collections[0].name,
                        "result_count": 1,
                    },
                ],
            },
        }

        with patch("packages.webui.api.v2.search.get_search_service", return_value=mock_search_service):
            response = await multi_collection_search(mock_request, search_request, mock_user, mock_search_service)

        # Verify the service was called with correct reranking parameters
        mock_search_service.multi_collection_search.assert_called_once_with(
            user_id=mock_user["id"],
            collection_uuids=[mock_collections[0].id],
            query="test query",
            k=10,
            search_type="semantic",
            score_threshold=0.0,
            metadata_filter=None,
            use_reranker=False,
            rerank_model=None,
            hybrid_alpha=0.7,
            hybrid_search_mode="rerank",
        )

        assert response.reranking_used is False
        assert response.reranker_model is None
        assert len(response.results) == 1
        assert response.results[0].score == 0.85
        assert response.results[0].reranked_score is None

    @pytest.mark.asyncio()
    async def test_search_with_reranking_enabled(self, mock_user, mock_collections):
        """Test that search works correctly with reranking enabled."""
        # Create a proper Request object
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/api/v2/search",
            "headers": [],
        }
        mock_request = Request(scope)
        mock_search_service = AsyncMock()

        search_request = CollectionSearchRequest(
            collection_uuids=[mock_collections[0].id],
            query="test query",
            k=10,
            use_reranker=True,  # Enable reranking
            rerank_model="Qwen/Qwen3-Reranker-0.6B",
            metadata_filter=None,
        )

        # Mock the service response with reranking
        mock_search_service.multi_collection_search.return_value = {
            "results": [
                {
                    "doc_id": "doc_1",
                    "chunk_id": "chunk_1",
                    "score": 0.95,  # Higher score after reranking
                    "reranked_score": 0.95,
                    "content": "Test content with reranking",
                    "path": "/test.txt",
                    "metadata": {},
                    "collection_id": mock_collections[0].id,
                    "collection_name": mock_collections[0].name,
                    "embedding_model": mock_collections[0].embedding_model,
                },
                {
                    "doc_id": "doc_2",
                    "chunk_id": "chunk_2",
                    "score": 0.92,
                    "reranked_score": 0.92,
                    "content": "Another test content",
                    "path": "/test2.txt",
                    "metadata": {},
                    "collection_id": mock_collections[0].id,
                    "collection_name": mock_collections[0].name,
                    "embedding_model": mock_collections[0].embedding_model,
                },
            ],
            "metadata": {
                "total_results": 2,
                "processing_time": 0.15,  # Longer processing time with reranking
                "collection_details": [
                    {
                        "collection_id": mock_collections[0].id,
                        "collection_name": mock_collections[0].name,
                        "result_count": 2,
                    },
                ],
            },
        }

        with patch("packages.webui.api.v2.search.get_search_service", return_value=mock_search_service):
            response = await multi_collection_search(mock_request, search_request, mock_user, mock_search_service)

        # Verify the service was called with correct reranking parameters
        mock_search_service.multi_collection_search.assert_called_once_with(
            user_id=mock_user["id"],
            collection_uuids=[mock_collections[0].id],
            query="test query",
            k=10,
            search_type="semantic",
            score_threshold=0.0,
            metadata_filter=None,
            use_reranker=True,
            rerank_model="Qwen/Qwen3-Reranker-0.6B",
            hybrid_alpha=0.7,
            hybrid_search_mode="rerank",
        )

        assert response.reranking_used is True
        assert response.reranker_model == "Qwen/Qwen3-Reranker-0.6B"
        assert len(response.results) == 2
        assert response.results[0].score == 0.95
        assert response.results[0].reranked_score == 0.95
        assert response.results[1].score == 0.92
        assert response.results[1].reranked_score == 0.92

    @pytest.mark.asyncio()
    async def test_search_reranking_different_scores(self, mock_user, mock_collections):
        """Test that reranking produces different scores from original scores."""
        # Create a proper Request object
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/api/v2/search",
            "headers": [],
        }
        mock_request = Request(scope)
        mock_search_service = AsyncMock()

        search_request = CollectionSearchRequest(
            collection_uuids=[mock_collections[0].id, mock_collections[1].id],
            query="complex query",
            k=5,
            use_reranker=True,
            rerank_model=None,  # Use default model
            metadata_filter=None,
        )

        # Mock the service response with different original and reranked scores
        mock_search_service.multi_collection_search.return_value = {
            "results": [
                {
                    "doc_id": "doc_1",
                    "chunk_id": "chunk_1",
                    "score": 0.98,  # Reranked score (was originally lower)
                    "reranked_score": 0.98,
                    "content": "Highly relevant after reranking",
                    "path": "/doc1.txt",
                    "metadata": {},
                    "collection_id": mock_collections[0].id,
                    "collection_name": mock_collections[0].name,
                    "embedding_model": mock_collections[0].embedding_model,
                },
                {
                    "doc_id": "doc_2",
                    "chunk_id": "chunk_2",
                    "score": 0.85,  # Reranked score (was originally higher)
                    "reranked_score": 0.85,
                    "content": "Less relevant after reranking",
                    "path": "/doc2.txt",
                    "metadata": {},
                    "collection_id": mock_collections[1].id,
                    "collection_name": mock_collections[1].name,
                    "embedding_model": mock_collections[1].embedding_model,
                },
            ],
            "metadata": {
                "total_results": 2,
                "processing_time": 0.2,
                "collection_details": [
                    {
                        "collection_id": mock_collections[0].id,
                        "collection_name": mock_collections[0].name,
                        "result_count": 1,
                    },
                    {
                        "collection_id": mock_collections[1].id,
                        "collection_name": mock_collections[1].name,
                        "result_count": 1,
                    },
                ],
            },
        }

        with patch("packages.webui.api.v2.search.get_search_service", return_value=mock_search_service):
            response = await multi_collection_search(mock_request, search_request, mock_user, mock_search_service)

        assert response.reranking_used is True
        assert len(response.results) == 2
        # Results should be ordered by reranked score
        assert response.results[0].score == 0.98
        assert response.results[1].score == 0.85

    @pytest.mark.asyncio()
    async def test_single_collection_search_with_reranking(self, mock_user):
        """Test single collection search with reranking enabled."""
        # Create a proper Request object
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/api/v2/search/single",
            "headers": [],
        }
        mock_request = Request(scope)
        mock_search_service = AsyncMock()

        search_request = SingleCollectionSearchRequest(
            collection_id="123e4567-e89b-12d3-a456-426614174000",
            query="test query",
            k=5,
            use_reranker=True,
            metadata_filter=None,
        )

        # Mock the service response with reranking
        mock_search_service.single_collection_search.return_value = {
            "results": [
                {
                    "doc_id": "doc_1",
                    "chunk_id": "chunk_1",
                    "score": 0.95,
                    "reranked_score": 0.95,
                    "content": "Reranked result",
                    "path": "/test/doc.txt",
                    "metadata": {"section": "intro"},
                },
            ],
            "processing_time_ms": 150,
        }

        with patch("packages.webui.api.v2.search.get_search_service", return_value=mock_search_service):
            response = await single_collection_search(mock_request, search_request, mock_user, mock_search_service)

        # Verify service was called with reranking enabled
        mock_search_service.single_collection_search.assert_called_once_with(
            user_id=mock_user["id"],
            collection_uuid=search_request.collection_id,
            query=search_request.query,
            k=search_request.k,
            search_type=search_request.search_type,
            score_threshold=search_request.score_threshold,
            metadata_filter=search_request.metadata_filter,
            use_reranker=True,
            include_content=search_request.include_content,
        )

        assert response.reranking_used is True
        assert len(response.results) == 1
        assert response.results[0].reranked_score == 0.95


class TestSingleCollectionSearch:
    """Test single collection search endpoint."""

    @pytest.mark.asyncio()
    async def test_single_collection_search_success(self, mock_user):
        """Test successful single collection search."""
        # Create a proper Request object with minimal required attributes
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/api/v2/search/single",
            "headers": [],
        }
        mock_request = Request(scope)
        mock_search_service = AsyncMock()

        search_request = SingleCollectionSearchRequest(
            collection_id="123e4567-e89b-12d3-a456-426614174000",
            query="test",
            k=5,
            metadata_filter=None,
        )

        # Mock the service response
        mock_search_service.single_collection_search.return_value = {
            "results": [
                {
                    "doc_id": "doc_1",
                    "chunk_id": "chunk_1",
                    "score": 0.95,
                    "content": "Test result content",
                    "path": "/test/doc.txt",
                    "metadata": {"section": "intro"},
                },
            ],
            "processing_time_ms": 100,
        }

        with patch("packages.webui.api.v2.search.get_search_service", return_value=mock_search_service):
            response = await single_collection_search(mock_request, search_request, mock_user, mock_search_service)

        # Verify service was called with correct params
        mock_search_service.single_collection_search.assert_called_once_with(
            user_id=mock_user["id"],
            collection_uuid=search_request.collection_id,
            query=search_request.query,
            k=search_request.k,
            search_type=search_request.search_type,
            score_threshold=search_request.score_threshold,
            metadata_filter=search_request.metadata_filter,
            use_reranker=search_request.use_reranker,
            include_content=search_request.include_content,
        )

        assert isinstance(response, CollectionSearchResponse)
        assert response.query == "test"
        assert len(response.results) == 1
        assert response.results[0].document_id == "doc_1"
        assert response.results[0].score == 0.95
        assert response.total_results == 1
        assert response.search_time_ms == 100
