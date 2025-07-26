"""
Tests for v2 search API endpoints.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from packages.shared.database.exceptions import AccessDeniedError, EntityNotFoundError
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

    @pytest.mark.asyncio()
    async def test_single_collection_search_not_found(self, mock_user):
        """Test single collection search with non-existent collection."""
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
        )

        # Mock service raising EntityNotFoundError
        mock_search_service.single_collection_search.side_effect = EntityNotFoundError("Collection", search_request.collection_id)

        with patch("packages.webui.api.v2.search.get_search_service", return_value=mock_search_service):
            with pytest.raises(HTTPException) as exc_info:
                await single_collection_search(mock_request, search_request, mock_user, mock_search_service)

        assert exc_info.value.status_code == 404
        assert "Collection" in str(exc_info.value.detail)
        assert search_request.collection_id in str(exc_info.value.detail)

    @pytest.mark.asyncio()
    async def test_single_collection_search_access_denied(self, mock_user):
        """Test single collection search with access denied."""
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
        )

        # Mock service raising AccessDeniedError
        mock_search_service.single_collection_search.side_effect = AccessDeniedError(str(mock_user["id"]), "Collection", search_request.collection_id)

        with patch("packages.webui.api.v2.search.get_search_service", return_value=mock_search_service):
            with pytest.raises(HTTPException) as exc_info:
                await single_collection_search(mock_request, search_request, mock_user, mock_search_service)

        assert exc_info.value.status_code == 403
        assert "does not have access" in str(exc_info.value.detail)

    @pytest.mark.asyncio()
    async def test_single_collection_search_general_error(self, mock_user):
        """Test single collection search with general error."""
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
        )

        # Mock service raising generic exception
        mock_search_service.single_collection_search.side_effect = Exception("Database connection failed")

        with patch("packages.webui.api.v2.search.get_search_service", return_value=mock_search_service):
            with pytest.raises(HTTPException) as exc_info:
                await single_collection_search(mock_request, search_request, mock_user, mock_search_service)

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Search failed"


class TestMultiCollectionSearchEdgeCases:
    """Test edge cases for multi-collection search."""

    @pytest.mark.asyncio()
    async def test_multi_collection_search_with_special_characters(self, mock_user, mock_collections):
        """Test search with special characters in query."""
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/api/v2/search",
            "headers": [],
        }
        mock_request = Request(scope)
        mock_search_service = AsyncMock()

        # Query with special characters
        search_request = CollectionSearchRequest(
            collection_uuids=[mock_collections[0].id],
            query="test @#$%^&*() <script>alert('xss')</script>",
            k=10,
        )

        mock_search_service.multi_collection_search.return_value = {
            "results": [],
            "metadata": {
                "total_results": 0,
                "processing_time": 0.05,
                "collection_details": [
                    {
                        "collection_id": mock_collections[0].id,
                        "collection_name": mock_collections[0].name,
                        "result_count": 0,
                    },
                ],
            },
        }

        with patch("packages.webui.api.v2.search.get_search_service", return_value=mock_search_service):
            response = await multi_collection_search(mock_request, search_request, mock_user, mock_search_service)

        assert response.query == "test @#$%^&*() <script>alert('xss')</script>"
        assert response.total_results == 0
        assert len(response.results) == 0

    @pytest.mark.asyncio()
    async def test_multi_collection_search_with_metadata_filter(self, mock_user, mock_collections):
        """Test search with metadata filters."""
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/api/v2/search",
            "headers": [],
        }
        mock_request = Request(scope)
        mock_search_service = AsyncMock()

        metadata_filter = {"file_type": "pdf", "language": "en"}
        search_request = CollectionSearchRequest(
            collection_uuids=[mock_collections[0].id],
            query="test",
            k=10,
            metadata_filter=metadata_filter,
        )

        mock_search_service.multi_collection_search.return_value = {
            "results": [
                {
                    "doc_id": "doc_1",
                    "chunk_id": "chunk_1",
                    "score": 0.9,
                    "content": "Filtered result",
                    "path": "/test.pdf",
                    "metadata": {"file_type": "pdf", "language": "en"},
                    "collection_id": mock_collections[0].id,
                    "collection_name": mock_collections[0].name,
                    "embedding_model": mock_collections[0].embedding_model,
                },
            ],
            "metadata": {
                "total_results": 1,
                "processing_time": 0.08,
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

        # Verify metadata filter was passed to service
        mock_search_service.multi_collection_search.assert_called_once()
        call_args = mock_search_service.multi_collection_search.call_args
        assert call_args.kwargs["metadata_filter"] == metadata_filter

        assert len(response.results) == 1
        assert response.results[0].metadata["file_type"] == "pdf"

    @pytest.mark.asyncio()
    async def test_multi_collection_search_different_search_types(self, mock_user, mock_collections):
        """Test search with different search types."""
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/api/v2/search",
            "headers": [],
        }
        mock_request = Request(scope)
        mock_search_service = AsyncMock()

        # Test with 'question' search type
        search_request = CollectionSearchRequest(
            collection_uuids=[mock_collections[0].id],
            query="What is authentication?",
            k=5,
            search_type="question",
        )

        mock_search_service.multi_collection_search.return_value = {
            "results": [
                {
                    "doc_id": "doc_1",
                    "chunk_id": "chunk_1",
                    "score": 0.95,
                    "content": "Authentication is the process...",
                    "path": "/auth.md",
                    "metadata": {},
                    "collection_id": mock_collections[0].id,
                    "collection_name": mock_collections[0].name,
                    "embedding_model": mock_collections[0].embedding_model,
                },
            ],
            "metadata": {
                "total_results": 1,
                "processing_time": 0.1,
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

        # Verify search type was passed correctly
        call_args = mock_search_service.multi_collection_search.call_args
        assert call_args.kwargs["search_type"] == "question"
        assert response.search_type == "question"

    @pytest.mark.asyncio()
    async def test_multi_collection_search_with_score_threshold(self, mock_user, mock_collections):
        """Test search with score threshold filtering."""
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
            query="test",
            k=10,
            score_threshold=0.8,
        )

        # Service returns only high-scoring results
        mock_search_service.multi_collection_search.return_value = {
            "results": [
                {
                    "doc_id": "doc_1",
                    "chunk_id": "chunk_1",
                    "score": 0.92,
                    "content": "High score result",
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

        # Verify score threshold was passed
        call_args = mock_search_service.multi_collection_search.call_args
        assert call_args.kwargs["score_threshold"] == 0.8

        assert len(response.results) == 1
        assert response.results[0].score >= 0.8

    @pytest.mark.asyncio()
    async def test_multi_collection_search_access_denied(self, mock_user, mock_collections):
        """Test multi-collection search with access denied error."""
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
            query="test",
            k=10,
        )

        # Mock service raising AccessDeniedError
        mock_search_service.multi_collection_search.side_effect = AccessDeniedError(str(mock_user["id"]), "Collections", ",".join(search_request.collection_uuids))

        with patch("packages.webui.api.v2.search.get_search_service", return_value=mock_search_service):
            with pytest.raises(HTTPException) as exc_info:
                await multi_collection_search(mock_request, search_request, mock_user, mock_search_service)

        assert exc_info.value.status_code == 403
        assert "does not have access" in str(exc_info.value.detail)

    @pytest.mark.asyncio()
    async def test_multi_collection_search_general_error(self, mock_user, mock_collections):
        """Test multi-collection search with general error."""
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
            query="test",
            k=10,
        )

        # Mock service raising generic exception
        mock_search_service.multi_collection_search.side_effect = Exception("Unexpected error")

        with patch("packages.webui.api.v2.search.get_search_service", return_value=mock_search_service):
            with pytest.raises(HTTPException) as exc_info:
                await multi_collection_search(mock_request, search_request, mock_user, mock_search_service)

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Search failed"


class TestHybridSearchParameters:
    """Test hybrid search functionality."""

    @pytest.mark.asyncio()
    async def test_hybrid_search_with_custom_alpha(self, mock_user, mock_collections):
        """Test hybrid search with custom alpha parameter."""
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
            query="test hybrid search",
            k=10,
            search_type="hybrid",
            hybrid_alpha=0.3,  # More weight on keyword search
            hybrid_mode="filter",
            keyword_mode="all",
        )

        mock_search_service.multi_collection_search.return_value = {
            "results": [
                {
                    "doc_id": "doc_1",
                    "chunk_id": "chunk_1",
                    "score": 0.88,
                    "content": "Hybrid search result",
                    "path": "/hybrid.txt",
                    "metadata": {},
                    "collection_id": mock_collections[0].id,
                    "collection_name": mock_collections[0].name,
                    "embedding_model": mock_collections[0].embedding_model,
                },
            ],
            "metadata": {
                "total_results": 1,
                "processing_time": 0.12,
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

        # Verify hybrid parameters were passed
        call_args = mock_search_service.multi_collection_search.call_args
        assert call_args.kwargs["search_type"] == "hybrid"
        assert call_args.kwargs["hybrid_alpha"] == 0.3
        assert call_args.kwargs["hybrid_search_mode"] == "filter"

        assert response.search_type == "hybrid"
        assert len(response.results) == 1

    @pytest.mark.asyncio()
    async def test_code_search_type(self, mock_user, mock_collections):
        """Test search with code search type."""
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
            query="def authenticate(user):",
            k=5,
            search_type="code",
        )

        mock_search_service.multi_collection_search.return_value = {
            "results": [
                {
                    "doc_id": "doc_1",
                    "chunk_id": "chunk_1",
                    "score": 0.97,
                    "content": "def authenticate(user):\n    # Authentication logic\n    return True",
                    "path": "/auth.py",
                    "metadata": {"language": "python"},
                    "collection_id": mock_collections[0].id,
                    "collection_name": mock_collections[0].name,
                    "embedding_model": mock_collections[0].embedding_model,
                },
            ],
            "metadata": {
                "total_results": 1,
                "processing_time": 0.08,
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

        assert response.search_type == "code"
        assert len(response.results) == 1
        assert "def authenticate" in response.results[0].text


class TestSearchValidation:
    """Test request validation for search endpoints."""

    def test_invalid_uuid_format_validation(self):
        """Test that invalid UUID format is rejected."""
        with pytest.raises(ValueError, match="Invalid UUID format"):
            CollectionSearchRequest(
                collection_uuids=["not-a-valid-uuid"],
                query="test",
                k=10,
            )

    def test_empty_collection_list_validation(self):
        """Test that empty collection list is rejected."""
        with pytest.raises(ValueError):
            CollectionSearchRequest(
                collection_uuids=[],
                query="test",
                k=10,
            )

    def test_too_many_collections_validation(self):
        """Test that too many collections are rejected."""
        import uuid
        
        # Generate 11 valid UUIDs (exceeds max of 10)
        uuids = [str(uuid.uuid4()) for _ in range(11)]
        
        with pytest.raises(ValueError):
            CollectionSearchRequest(
                collection_uuids=uuids,
                query="test",
                k=10,
            )

    def test_query_length_validation(self):
        """Test query length limits."""
        import uuid
        
        valid_uuid = str(uuid.uuid4())
        
        # Test empty query
        with pytest.raises(ValueError):
            CollectionSearchRequest(
                collection_uuids=[valid_uuid],
                query="",
                k=10,
            )
        
        # Test query that's too long
        with pytest.raises(ValueError):
            CollectionSearchRequest(
                collection_uuids=[valid_uuid],
                query="x" * 1001,  # Exceeds max length of 1000
                k=10,
            )

    def test_k_parameter_validation(self):
        """Test k parameter limits."""
        import uuid
        
        valid_uuid = str(uuid.uuid4())
        
        # Test k = 0
        with pytest.raises(ValueError):
            CollectionSearchRequest(
                collection_uuids=[valid_uuid],
                query="test",
                k=0,
            )
        
        # Test k > 100
        with pytest.raises(ValueError):
            CollectionSearchRequest(
                collection_uuids=[valid_uuid],
                query="test",
                k=101,
            )

    def test_score_threshold_validation(self):
        """Test score threshold limits."""
        import uuid
        
        valid_uuid = str(uuid.uuid4())
        
        # Test negative score threshold
        with pytest.raises(ValueError):
            CollectionSearchRequest(
                collection_uuids=[valid_uuid],
                query="test",
                k=10,
                score_threshold=-0.1,
            )
        
        # Test score threshold > 1.0
        with pytest.raises(ValueError):
            CollectionSearchRequest(
                collection_uuids=[valid_uuid],
                query="test",
                k=10,
                score_threshold=1.1,
            )

    def test_hybrid_alpha_validation(self):
        """Test hybrid alpha parameter limits."""
        import uuid
        
        valid_uuid = str(uuid.uuid4())
        
        # Test negative alpha
        with pytest.raises(ValueError):
            CollectionSearchRequest(
                collection_uuids=[valid_uuid],
                query="test",
                k=10,
                hybrid_alpha=-0.1,
            )
        
        # Test alpha > 1.0
        with pytest.raises(ValueError):
            CollectionSearchRequest(
                collection_uuids=[valid_uuid],
                query="test",
                k=10,
                hybrid_alpha=1.1,
            )


class TestSearchResultFormatting:
    """Test response formatting for search results."""

    @pytest.mark.asyncio()
    async def test_multi_collection_search_empty_results(self, mock_user, mock_collections):
        """Test search with no results."""
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
            query="non-existent query xyz123",
            k=10,
        )

        mock_search_service.multi_collection_search.return_value = {
            "results": [],
            "metadata": {
                "total_results": 0,
                "processing_time": 0.03,
                "collection_details": [
                    {
                        "collection_id": mock_collections[0].id,
                        "collection_name": mock_collections[0].name,
                        "result_count": 0,
                    },
                ],
            },
        }

        with patch("packages.webui.api.v2.search.get_search_service", return_value=mock_search_service):
            response = await multi_collection_search(mock_request, search_request, mock_user, mock_search_service)

        assert response.total_results == 0
        assert len(response.results) == 0
        assert response.partial_failure is False
        assert response.failed_collections is None

    @pytest.mark.asyncio()
    async def test_result_file_path_handling(self, mock_user, mock_collections):
        """Test file path and file name extraction."""
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
            query="test",
            k=5,
        )

        mock_search_service.multi_collection_search.return_value = {
            "results": [
                {
                    "doc_id": "doc_1",
                    "chunk_id": "chunk_1",
                    "score": 0.9,
                    "content": "Test content",
                    "path": "/path/to/document/file.txt",
                    "metadata": {},
                    "collection_id": mock_collections[0].id,
                    "collection_name": mock_collections[0].name,
                    "embedding_model": mock_collections[0].embedding_model,
                },
                {
                    "doc_id": "doc_2",
                    "chunk_id": "chunk_2",
                    "score": 0.85,
                    "content": "Another test",
                    "path": "",  # Empty path
                    "metadata": {},
                    "collection_id": mock_collections[0].id,
                    "collection_name": mock_collections[0].name,
                    "embedding_model": mock_collections[0].embedding_model,
                },
                {
                    "doc_id": "doc_3",
                    "chunk_id": "chunk_3",
                    "score": 0.8,
                    "content": "No path test",
                    # No path key at all
                    "metadata": {},
                    "collection_id": mock_collections[0].id,
                    "collection_name": mock_collections[0].name,
                    "embedding_model": mock_collections[0].embedding_model,
                },
            ],
            "metadata": {
                "total_results": 3,
                "processing_time": 0.05,
                "collection_details": [
                    {
                        "collection_id": mock_collections[0].id,
                        "collection_name": mock_collections[0].name,
                        "result_count": 3,
                    },
                ],
            },
        }

        with patch("packages.webui.api.v2.search.get_search_service", return_value=mock_search_service):
            response = await multi_collection_search(mock_request, search_request, mock_user, mock_search_service)

        # Check file name extraction
        assert response.results[0].file_name == "file.txt"
        assert response.results[0].file_path == "/path/to/document/file.txt"
        
        # Check empty path handling
        assert response.results[1].file_name == "Unknown"
        assert response.results[1].file_path == ""
        
        # Check missing path handling
        assert response.results[2].file_name == "Unknown"
        assert response.results[2].file_path == ""

    @pytest.mark.asyncio()
    async def test_result_missing_optional_fields(self, mock_user, mock_collections):
        """Test handling of results with missing optional fields."""
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
            query="test",
            k=5,
        )

        # Result with minimal fields
        mock_search_service.multi_collection_search.return_value = {
            "results": [
                {
                    # Minimal required fields only
                    "score": 0.9,
                    "collection_id": mock_collections[0].id,
                    "collection_name": mock_collections[0].name,
                    "embedding_model": mock_collections[0].embedding_model,
                },
            ],
            "metadata": {
                "total_results": 1,
                "processing_time": 0.04,
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

        # Check defaults for missing fields
        result = response.results[0]
        assert result.document_id == ""
        assert result.chunk_id == ""
        assert result.text == ""
        assert result.metadata == {}
        assert result.file_name == "Unknown"
        assert result.file_path == ""
        assert result.score == 0.9
        assert result.original_score == 0.9
        assert result.reranked_score is None

    @pytest.mark.asyncio()
    async def test_single_collection_search_with_metadata_filter(self, mock_user):
        """Test single collection search with metadata filtering."""
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/api/v2/search/single",
            "headers": [],
        }
        mock_request = Request(scope)
        mock_search_service = AsyncMock()

        metadata_filter = {"file_type": "markdown", "author": "admin"}
        search_request = SingleCollectionSearchRequest(
            collection_id="123e4567-e89b-12d3-a456-426614174000",
            query="authentication guide",
            k=5,
            metadata_filter=metadata_filter,
            include_content=False,  # Test without content
        )

        mock_search_service.single_collection_search.return_value = {
            "results": [
                {
                    "doc_id": "doc_1",
                    "chunk_id": "chunk_1",
                    "score": 0.92,
                    "content": "",  # Empty when include_content=False
                    "path": "/docs/auth.md",
                    "metadata": {"file_type": "markdown", "author": "admin"},
                },
            ],
            "processing_time_ms": 75,
        }

        with patch("packages.webui.api.v2.search.get_search_service", return_value=mock_search_service):
            response = await single_collection_search(mock_request, search_request, mock_user, mock_search_service)

        # Verify metadata filter was passed
        call_args = mock_search_service.single_collection_search.call_args
        assert call_args.kwargs["metadata_filter"] == metadata_filter
        assert call_args.kwargs["include_content"] is False

        assert len(response.results) == 1
        assert response.results[0].text == ""
        assert response.results[0].metadata["file_type"] == "markdown"

    @pytest.mark.asyncio()
    async def test_multi_collection_timing_metrics(self, mock_user, mock_collections):
        """Test timing metrics in response."""
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
            query="test",
            k=5,
        )

        mock_search_service.multi_collection_search.return_value = {
            "results": [],
            "metadata": {
                "total_results": 0,
                "processing_time": 0.12345,  # In seconds
                "collection_details": [
                    {
                        "collection_id": mock_collections[0].id,
                        "collection_name": mock_collections[0].name,
                        "result_count": 0,
                    },
                ],
            },
        }

        with patch("packages.webui.api.v2.search.get_search_service", return_value=mock_search_service):
            response = await multi_collection_search(mock_request, search_request, mock_user, mock_search_service)

        # Check timing conversion (seconds to milliseconds)
        assert response.search_time_ms == 123.45
        assert response.total_time_ms == 123.45
        assert response.reranking_time_ms is None
        assert response.embedding_time_ms is None
