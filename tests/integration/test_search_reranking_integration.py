"""
Integration tests for search reranking functionality.

These tests verify the complete flow from API endpoint through service layer
with proper mocking to avoid external dependencies.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from shared.database.models import Collection, CollectionStatus
from webui.api.v2 import search as search_api
from webui.auth import get_current_user
from webui.services.factory import get_search_service
from webui.services.search_service import SearchService


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
    collection1.quantization = "float16"
    collection1.status = CollectionStatus.READY

    collection2 = MagicMock(spec=Collection)
    collection2.id = "456e7890-e89b-12d3-a456-426614174001"
    collection2.name = "Research Papers"
    collection2.vector_store_name = "qdrant_research_collection"
    collection2.embedding_model = "BAAI/bge-small-en-v1.5"
    collection2.quantization = "int8"
    collection2.status = CollectionStatus.READY

    return [collection1, collection2]


@pytest.fixture()
def app() -> FastAPI:
    """Create FastAPI app with search endpoints."""
    app = FastAPI()
    app.include_router(search_api.router)
    return app


@pytest.fixture()
def client(app: FastAPI, mock_user: dict[str, Any]) -> TestClient:
    """Create test client with mocked dependencies."""
    # Override authentication dependency
    app.dependency_overrides[get_current_user] = lambda: mock_user

    return TestClient(app)


class TestSearchRerankingIntegration:
    """Integration tests for search reranking."""

    def test_search_api_with_reranking_disabled(self, client: TestClient, mock_collections: list[MagicMock]) -> None:
        """Test search API with reranking disabled."""
        # Mock the search service
        mock_search_service = MagicMock(spec=SearchService)

        # Create async mock for the method
        async def mock_multi_search(*args: Any, **kwargs: Any) -> dict[str, Any]:  # noqa: ARG001
            return {
                "results": [
                    {
                        "doc_id": "doc_1",
                        "chunk_id": "chunk_1",
                        "score": 0.85,
                        "content": "Test content",
                        "path": "/test.txt",
                        "metadata": {},
                        "collection_id": mock_collections[0].id,
                        "collection_name": mock_collections[0].name,
                        "embedding_model": mock_collections[0].embedding_model,
                    }
                ],
                "metadata": {
                    "total_results": 1,
                    "processing_time": 0.05,
                    "collection_details": [
                        {
                            "collection_id": mock_collections[0].id,
                            "collection_name": mock_collections[0].name,
                            "result_count": 1,
                        }
                    ],
                },
            }

        mock_search_service.multi_collection_search = mock_multi_search

        # Override the service factory
        client.app.dependency_overrides[get_search_service] = lambda: mock_search_service

        # Make request
        response = client.post(
            "/api/v2/search",
            json={
                "collection_uuids": [mock_collections[0].id],
                "query": "test query",
                "k": 10,
                "use_reranker": False,
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert data["query"] == "test query"
        assert data["reranking_used"] is False
        assert data["reranker_model"] is None
        assert len(data["results"]) == 1
        assert data["results"][0]["score"] == 0.85
        assert data["results"][0]["reranked_score"] is None

    def test_search_api_with_reranking_enabled(self, client: TestClient, mock_collections: list[MagicMock]) -> None:
        """Test search API with reranking enabled."""
        # Mock the search service
        mock_search_service = MagicMock(spec=SearchService)

        # Create async mock for the method
        async def mock_multi_search(*args: Any, **kwargs: Any) -> dict[str, Any]:  # noqa: ARG001
            # Verify reranking parameters were passed
            assert kwargs["use_reranker"] is True
            assert kwargs["rerank_model"] == "Qwen/Qwen3-Reranker-0.6B"

            return {
                "results": [
                    {
                        "doc_id": "doc_1",
                        "chunk_id": "chunk_1",
                        "score": 0.95,  # Higher score after reranking
                        "reranked_score": 0.95,
                        "content": "Test content reranked",
                        "path": "/test.txt",
                        "metadata": {},
                        "collection_id": mock_collections[0].id,
                        "collection_name": mock_collections[0].name,
                        "embedding_model": mock_collections[0].embedding_model,
                    }
                ],
                "metadata": {
                    "total_results": 1,
                    "processing_time": 0.15,  # Longer with reranking
                    "collection_details": [
                        {
                            "collection_id": mock_collections[0].id,
                            "collection_name": mock_collections[0].name,
                            "result_count": 1,
                        }
                    ],
                },
            }

        mock_search_service.multi_collection_search = mock_multi_search

        # Override the service factory
        client.app.dependency_overrides[get_search_service] = lambda: mock_search_service

        # Make request
        response = client.post(
            "/api/v2/search",
            json={
                "collection_uuids": [mock_collections[0].id],
                "query": "test query",
                "k": 10,
                "use_reranker": True,
                "rerank_model": "Qwen/Qwen3-Reranker-0.6B",
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert data["query"] == "test query"
        assert data["reranking_used"] is True
        assert data["reranker_model"] == "Qwen/Qwen3-Reranker-0.6B"
        assert len(data["results"]) == 1
        assert data["results"][0]["score"] == 0.95
        assert data["results"][0]["reranked_score"] == 0.95

    def test_search_api_reranking_with_multiple_collections(
        self, client: TestClient, mock_collections: list[MagicMock]
    ) -> None:
        """Test search API with reranking across multiple collections."""
        # Mock the search service
        mock_search_service = MagicMock(spec=SearchService)

        # Create async mock for the method
        async def mock_multi_search(*args: Any, **kwargs: Any) -> dict[str, Any]:  # noqa: ARG001
            return {
                "results": [
                    {
                        "doc_id": "doc_1",
                        "chunk_id": "chunk_1",
                        "score": 0.98,
                        "reranked_score": 0.98,
                        "content": "Highly relevant from collection 1",
                        "path": "/doc1.txt",
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
                        "content": "Relevant from collection 2",
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

        mock_search_service.multi_collection_search = mock_multi_search

        # Override the service factory
        client.app.dependency_overrides[get_search_service] = lambda: mock_search_service

        # Make request
        response = client.post(
            "/api/v2/search",
            json={
                "collection_uuids": [c.id for c in mock_collections],
                "query": "test query",
                "k": 10,
                "use_reranker": True,
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Verify results from multiple collections
        assert len(data["results"]) == 2
        assert len(data["collections_searched"]) == 2

        # Results should be sorted by score
        assert data["results"][0]["score"] > data["results"][1]["score"]

        # Each result should have collection info
        assert data["results"][0]["collection_id"] == str(mock_collections[0].id)
        assert data["results"][1]["collection_id"] == str(mock_collections[1].id)

    def test_search_api_reranking_with_hybrid_search(
        self, client: TestClient, mock_collections: list[MagicMock]
    ) -> None:
        """Test search API with reranking and hybrid search."""
        # Mock the search service
        mock_search_service = MagicMock(spec=SearchService)

        # Create async mock for the method
        async def mock_multi_search(*args: Any, **kwargs: Any) -> dict[str, Any]:  # noqa: ARG001
            # Verify hybrid search parameters
            assert kwargs["search_type"] == "hybrid"
            assert kwargs["hybrid_alpha"] == 0.5
            assert kwargs["hybrid_mode"] == "weighted"
            assert kwargs["use_reranker"] is True
            assert "hybrid_search_mode" not in kwargs

            return {
                "results": [
                    {
                        "doc_id": "doc_1",
                        "chunk_id": "chunk_1",
                        "score": 0.96,
                        "reranked_score": 0.96,
                        "content": "Hybrid search result",
                        "path": "/test.txt",
                        "metadata": {},
                        "collection_id": mock_collections[0].id,
                        "collection_name": mock_collections[0].name,
                        "embedding_model": mock_collections[0].embedding_model,
                    }
                ],
                "metadata": {
                    "total_results": 1,
                    "processing_time": 0.18,
                    "collection_details": [
                        {
                            "collection_id": mock_collections[0].id,
                            "collection_name": mock_collections[0].name,
                            "result_count": 1,
                        }
                    ],
                },
            }

        mock_search_service.multi_collection_search = mock_multi_search

        # Override the service factory
        client.app.dependency_overrides[get_search_service] = lambda: mock_search_service

        # Make request
        response = client.post(
            "/api/v2/search",
            json={
                "collection_uuids": [mock_collections[0].id],
                "query": "test query",
                "k": 10,
                "search_type": "hybrid",
                "use_reranker": True,
                "hybrid_alpha": 0.5,
                "hybrid_mode": "weighted",
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert data["search_type"] == "hybrid"
        assert data["reranking_used"] is True
        assert len(data["results"]) == 1

    def test_single_collection_search_with_reranking(
        self, client: TestClient, mock_collections: list[MagicMock]
    ) -> None:
        """Test single collection search endpoint with reranking."""
        # Mock the search service
        mock_search_service = MagicMock(spec=SearchService)

        # Create async mock for the method
        async def mock_single_search(*args: Any, **kwargs: Any) -> dict[str, Any]:  # noqa: ARG001
            assert kwargs["use_reranker"] is True

            return {
                "results": [
                    {
                        "doc_id": "doc_1",
                        "chunk_id": "chunk_1",
                        "score": 0.95,
                        "reranked_score": 0.95,
                        "content": "Single collection result",
                        "path": "/test.txt",
                        "metadata": {},
                    }
                ],
                "processing_time_ms": 120,
            }

        mock_search_service.single_collection_search = mock_single_search

        # Override the service factory
        client.app.dependency_overrides[get_search_service] = lambda: mock_search_service

        # Make request
        response = client.post(
            "/api/v2/search/single",
            json={
                "collection_id": mock_collections[0].id,
                "query": "test query",
                "k": 5,
                "use_reranker": True,
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert data["query"] == "test query"
        assert data["reranking_used"] is True
        assert len(data["results"]) == 1
        assert data["results"][0]["reranked_score"] == 0.95
