"""
Tests for v2 search API endpoints.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.requests import Request
from starlette.testclient import TestClient

from packages.shared.database.models import Collection, CollectionStatus
from packages.webui.api.v2.schemas import (
    CollectionSearchRequest,
    CollectionSearchResponse,
    SingleCollectionSearchRequest,
)
from packages.webui.api.v2.search import (
    multi_collection_search,
    rerank_merged_results,
    search_single_collection,
    single_collection_search,
    validate_collection_access,
)


@pytest.fixture()
def mock_user():
    """Mock authenticated user."""
    return {"id": 1, "username": "testuser"}


@pytest.fixture()
def mock_collections():
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
def mock_search_results():
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


class TestValidateCollectionAccess:
    """Test collection access validation."""

    @pytest.mark.asyncio()
    async def test_validate_collection_access_success(self, mock_collections):
        """Test successful validation of collection access."""
        mock_db = AsyncMock(spec=AsyncSession)
        mock_repo = AsyncMock()

        async def get_by_uuid_side_effect(collection_uuid, user_id):  # noqa: ARG001
            if collection_uuid == mock_collections[0].id:
                return mock_collections[0]
            if collection_uuid == mock_collections[1].id:
                return mock_collections[1]
            raise Exception("Collection not found")

        mock_repo.get_by_uuid_with_permission_check.side_effect = get_by_uuid_side_effect

        with patch("packages.webui.api.v2.search.CollectionRepository", return_value=mock_repo):
            result = await validate_collection_access([c.id for c in mock_collections], 1, mock_db)

        assert len(result) == 2
        assert result[0].id == mock_collections[0].id
        assert result[1].id == mock_collections[1].id

    @pytest.mark.asyncio()
    async def test_validate_collection_access_denied(self):
        """Test validation fails when access is denied."""
        mock_db = AsyncMock(spec=AsyncSession)
        mock_repo = AsyncMock()
        mock_repo.get_by_uuid_with_permission_check.side_effect = Exception("Access denied")

        with patch("packages.webui.api.v2.search.CollectionRepository", return_value=mock_repo), pytest.raises(
            HTTPException
        ) as exc_info:
            await validate_collection_access(["invalid-uuid"], 1, mock_db)

        assert exc_info.value.status_code == 403


class TestSearchSingleCollection:
    """Test single collection search."""

    @pytest.mark.asyncio()
    async def test_search_single_collection_success(self, mock_collections, mock_search_results):
        """Test successful search in a single collection."""
        collection = mock_collections[0]

        async def mock_post(*args, **kwargs):
            mock_response = AsyncMock()
            mock_response.json = lambda: mock_search_results
            mock_response.raise_for_status = AsyncMock()
            return mock_response

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = mock_post
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await search_single_collection(
                collection,
                "authentication",
                10,
                {"search_type": "semantic"},
                httpx.Timeout(timeout=60.0),
            )

        assert result[0] == collection
        assert result[1] == mock_search_results["results"]
        assert result[2] is None  # No error

    @pytest.mark.asyncio()
    async def test_search_single_collection_not_ready(self, mock_collections):
        """Test search skipped when collection is not ready."""
        collection = mock_collections[0]
        collection.status = CollectionStatus.PROCESSING

        result = await search_single_collection(
            collection,
            "test query",
            10,
            {},
            httpx.Timeout(timeout=60.0),
        )

        assert result[0] == collection
        assert result[1] is None
        assert "not ready" in result[2]

    @pytest.mark.asyncio()
    async def test_search_single_collection_timeout_retry(self, mock_collections, mock_search_results):
        """Test search retry on timeout."""
        collection = mock_collections[0]

        with patch("httpx.AsyncClient") as mock_client_class:
            # First attempt times out
            mock_client1 = AsyncMock()
            mock_client1.post.side_effect = httpx.ReadTimeout("Timeout")

            # Second attempt succeeds
            mock_client2 = AsyncMock()
            mock_response = AsyncMock()
            mock_response.json = lambda: mock_search_results
            mock_response.raise_for_status = AsyncMock()
            mock_client2.post.return_value = mock_response

            mock_client_class.return_value.__aenter__.side_effect = [mock_client1, mock_client2]

            result = await search_single_collection(
                collection,
                "test query",
                10,
                {},
                httpx.Timeout(timeout=60.0),
            )

        assert result[0] == collection
        assert result[1] == mock_search_results["results"]
        assert result[2] is None


class TestRerankMergedResults:
    """Test result re-ranking."""

    @pytest.mark.asyncio()
    async def test_rerank_merged_results_success(self, mock_collections):
        """Test successful re-ranking of merged results."""
        results = [
            (mock_collections[0], {"content": "Document 1", "score": 0.8}),
            (mock_collections[1], {"content": "Document 2", "score": 0.9}),
            (mock_collections[0], {"content": "Document 3", "score": 0.7}),
        ]

        rerank_response = {"results": [(1, 0.95), (0, 0.85), (2, 0.75)]}  # Reordered with new scores

        async def mock_post(*args, **kwargs):
            mock_response = AsyncMock()
            mock_response.json = lambda: rerank_response
            mock_response.raise_for_status = AsyncMock()
            return mock_response

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = mock_post
            mock_client_class.return_value.__aenter__.return_value = mock_client

            reranked = await rerank_merged_results("test query", results, k=3)

        assert len(reranked) == 3
        assert reranked[0][2] == 0.95  # Highest score
        assert reranked[0][1]["content"] == "Document 2"  # Was index 1
        assert reranked[1][2] == 0.85
        assert reranked[1][1]["content"] == "Document 1"  # Was index 0

    @pytest.mark.asyncio()
    async def test_rerank_merged_results_fallback(self, mock_collections):
        """Test fallback to original scores on re-ranking failure."""
        results = [
            (mock_collections[0], {"content": "Document 1", "score": 0.8}),
            (mock_collections[1], {"content": "Document 2", "score": 0.9}),
        ]

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.side_effect = Exception("Reranking failed")
            mock_client_class.return_value.__aenter__.return_value = mock_client

            reranked = await rerank_merged_results("test query", results, k=2)

        assert len(reranked) == 2
        assert reranked[0][2] == 0.9  # Original highest score
        assert reranked[0][1]["content"] == "Document 2"
        assert reranked[1][2] == 0.8
        assert reranked[1][1]["content"] == "Document 1"


class TestMultiCollectionSearch:
    """Test multi-collection search endpoint."""

    @pytest.mark.asyncio()
    async def test_multi_collection_search_success(self, mock_user, mock_collections, mock_search_results):
        """Test successful multi-collection search."""
        # Create a proper Request object with minimal required attributes
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/api/v2/search",
            "headers": [],
        }
        mock_request = Request(scope)
        mock_db = AsyncMock(spec=AsyncSession)

        search_request = CollectionSearchRequest(
            collection_uuids=[c.id for c in mock_collections],
            query="authentication",
            k=10,
            use_reranker=True,
        )

        with patch("packages.webui.api.v2.search.validate_collection_access") as mock_validate:
            mock_validate.return_value = mock_collections

            with patch("packages.webui.api.v2.search.search_single_collection") as mock_search:
                # Different results for each collection
                mock_search.side_effect = [
                    (mock_collections[0], mock_search_results["results"], None),
                    (mock_collections[1], mock_search_results["results"][:1], None),
                ]

                with patch("packages.webui.api.v2.search.rerank_merged_results") as mock_rerank:
                    mock_rerank.return_value = [
                        (mock_collections[0], mock_search_results["results"][0], 0.98),
                        (mock_collections[1], mock_search_results["results"][0], 0.92),
                    ]

                    response = await multi_collection_search(mock_request, search_request, mock_user, mock_db)

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
        mock_db = AsyncMock(spec=AsyncSession)

        search_request = CollectionSearchRequest(
            collection_uuids=[c.id for c in mock_collections],
            query="test",
            k=10,
        )

        with patch("packages.webui.api.v2.search.validate_collection_access") as mock_validate:
            mock_validate.return_value = mock_collections

            with patch("packages.webui.api.v2.search.search_single_collection") as mock_search:
                # First collection succeeds, second fails
                mock_search.side_effect = [
                    (mock_collections[0], [{"doc_id": "1", "score": 0.9, "content": "Test"}], None),
                    (mock_collections[1], None, "Search failed: Connection error"),
                ]

                response = await multi_collection_search(mock_request, search_request, mock_user, mock_db)

        assert response.partial_failure is True
        assert len(response.failed_collections) == 1
        assert response.failed_collections[0]["collection_id"] == mock_collections[1].id
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
        mock_db = AsyncMock(spec=AsyncSession)

        # Set same model for both collections
        mock_collections[1].embedding_model = mock_collections[0].embedding_model

        search_request = CollectionSearchRequest(
            collection_uuids=[c.id for c in mock_collections],
            query="test",
            k=10,
            use_reranker=False,  # Explicitly disable
        )

        with patch("packages.webui.api.v2.search.validate_collection_access") as mock_validate:
            mock_validate.return_value = mock_collections

            with patch("packages.webui.api.v2.search.search_single_collection") as mock_search:
                mock_search.side_effect = [
                    (mock_collections[0], [{"doc_id": "1", "score": 0.9, "content": "Test"}], None),
                    (mock_collections[1], [{"doc_id": "2", "score": 0.8, "content": "Test2"}], None),
                ]

                with patch("packages.webui.api.v2.search.rerank_merged_results") as mock_rerank:
                    response = await multi_collection_search(mock_request, search_request, mock_user, mock_db)

                    # Should not call reranking
                    mock_rerank.assert_not_called()

        assert response.reranking_used is False
        assert response.reranker_model is None
        assert len(response.results) == 2
        # Results should be sorted by score
        assert response.results[0].score == 0.9
        assert response.results[1].score == 0.8


class TestSingleCollectionSearch:
    """Test single collection search endpoint."""

    @pytest.mark.asyncio()
    async def test_single_collection_search_delegates_to_multi(self, mock_user):
        """Test single collection search delegates to multi-collection search."""
        # Create a proper Request object with minimal required attributes
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/api/v2/search",
            "headers": [],
        }
        mock_request = Request(scope)
        mock_db = AsyncMock(spec=AsyncSession)

        search_request = SingleCollectionSearchRequest(
            collection_id="123e4567-e89b-12d3-a456-426614174000",
            query="test",
            k=5,
        )

        with patch("packages.webui.api.v2.search.multi_collection_search") as mock_multi:
            mock_response = CollectionSearchResponse(
                query="test",
                results=[],
                total_results=0,
                collections_searched=[],
                search_type="semantic",
                reranking_used=False,
                search_time_ms=100.0,
                total_time_ms=100.0,
            )
            mock_multi.return_value = mock_response

            response = await single_collection_search(mock_request, search_request, mock_user, mock_db)

            # Verify it called multi_collection_search with correct params
            mock_multi.assert_called_once()
            call_args = mock_multi.call_args
            multi_request = call_args[0][1]

            assert isinstance(multi_request, CollectionSearchRequest)
            assert multi_request.collection_uuids == [search_request.collection_id]
            assert multi_request.query == search_request.query
            assert multi_request.k == search_request.k

        assert response == mock_response
