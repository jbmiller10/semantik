"""
Tests for SearchService reranking functionality.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from shared.database.models import Collection, CollectionStatus
from webui.services.search_service import SearchService

# Database and repository fixtures are now imported from conftest.py


@pytest.fixture()
def search_service(mock_db_session: AsyncMock, mock_collection_repo: AsyncMock) -> SearchService:
    """Create a SearchService instance with mocked dependencies."""
    return SearchService(db_session=mock_db_session, collection_repo=mock_collection_repo)


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


class TestSearchServiceReranking:
    """Test SearchService reranking functionality."""

    @pytest.mark.asyncio()
    async def test_single_collection_search_with_reranking(
        self, search_service: SearchService, mock_collection_repo: AsyncMock, mock_collections: list[MagicMock]
    ) -> None:
        """Test single collection search passes reranking parameters correctly."""
        # Setup
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collections[0]

        # Mock vecpipe response with reranking
        mock_response = {
            "results": [
                {
                    "doc_id": "doc_1",
                    "chunk_id": "chunk_1",
                    "score": 0.95,
                    "reranked_score": 0.95,
                    "content": "Test content",
                    "path": "/test.txt",
                    "metadata": {},
                }
            ],
            "processing_time_ms": 100,
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response_obj

            result = await search_service.single_collection_search(
                user_id=1,
                collection_uuid=mock_collections[0].id,
                query="test query",
                k=10,
                search_type="semantic",
                use_reranker=True,
                rerank_model="Qwen/Qwen3-Reranker-0.6B",
            )

            # Verify the request sent to vecpipe
            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args
            assert call_args[0][0].endswith("/search")

            request_data = call_args[1]["json"]
            assert request_data["query"] == "test query"
            assert request_data["k"] == 10
            assert request_data["use_reranker"] is True
            assert request_data["rerank_model"] == "Qwen/Qwen3-Reranker-0.6B"
            assert request_data["collection"] == mock_collections[0].vector_store_name
            assert request_data["model_name"] == mock_collections[0].embedding_model
            assert request_data["quantization"] == mock_collections[0].quantization

            # Verify the response
            assert result["results"][0]["reranked_score"] == 0.95

    @pytest.mark.asyncio()
    async def test_multi_collection_search_with_reranking(
        self, search_service: SearchService, mock_collection_repo: AsyncMock, mock_collections: list[MagicMock]
    ) -> None:
        """Test multi-collection search with reranking enabled."""
        # Setup
        mock_collection_repo.get_by_uuid_with_permission_check.side_effect = mock_collections

        # Mock vecpipe responses - results already reranked by vecpipe
        mock_responses = [
            {
                "results": [
                    {
                        "doc_id": "doc_1",
                        "chunk_id": "chunk_1",
                        "score": 0.98,  # Already reranked score
                        "content": "Highly relevant content",
                        "path": "/doc1.txt",
                        "metadata": {},
                    }
                ],
            },
            {
                "results": [
                    {
                        "doc_id": "doc_2",
                        "chunk_id": "chunk_2",
                        "score": 0.85,  # Already reranked score
                        "content": "Less relevant content",
                        "path": "/doc2.txt",
                        "metadata": {},
                    }
                ],
            },
        ]

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            # Setup different responses for each call
            response_objs = []
            for mock_resp in mock_responses:
                resp_obj = MagicMock()
                resp_obj.json.return_value = mock_resp
                resp_obj.raise_for_status = MagicMock()
                response_objs.append(resp_obj)

            mock_client.post.side_effect = response_objs

            result = await search_service.multi_collection_search(
                user_id=1,
                collection_uuids=[c.id for c in mock_collections],
                query="test query",
                k=10,
                use_reranker=True,
                rerank_model="Qwen/Qwen3-Reranker-0.6B",
            )

            # Verify both collections were searched with reranking
            assert mock_client.post.call_count == 2

            # Check first call
            first_call = mock_client.post.call_args_list[0]
            first_request = first_call[1]["json"]
            assert first_request["use_reranker"] is True
            assert first_request["rerank_model"] == "Qwen/Qwen3-Reranker-0.6B"

            # Check second call
            second_call = mock_client.post.call_args_list[1]
            second_request = second_call[1]["json"]
            assert second_request["use_reranker"] is True
            assert second_request["rerank_model"] == "Qwen/Qwen3-Reranker-0.6B"

            # Verify results are sorted by score (already reranked by vecpipe)
            assert len(result["results"]) == 2
            assert result["results"][0]["score"] == 0.98
            assert result["results"][1]["score"] == 0.85

    @pytest.mark.asyncio()
    async def test_search_without_reranking(
        self, search_service: SearchService, mock_collection_repo: AsyncMock, mock_collections: list[MagicMock]
    ) -> None:
        """Test that search works correctly when reranking is disabled."""
        # Setup
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collections[0]

        # Mock vecpipe response without reranking
        mock_response = {
            "results": [
                {
                    "doc_id": "doc_1",
                    "chunk_id": "chunk_1",
                    "score": 0.75,  # Lower score without reranking
                    "content": "Test content",
                    "path": "/test.txt",
                    "metadata": {},
                }
            ],
            "processing_time_ms": 50,  # Faster without reranking
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response_obj

            result = await search_service.single_collection_search(
                user_id=1,
                collection_uuid=mock_collections[0].id,
                query="test query",
                k=10,
                use_reranker=False,  # Explicitly disable reranking
            )

            # Verify the request sent to vecpipe
            request_data = mock_client.post.call_args[1]["json"]
            assert request_data["use_reranker"] is False
            assert "rerank_model" not in request_data or request_data["rerank_model"] is None

            # Verify the response doesn't have reranked scores
            assert "reranked_score" not in result["results"][0]

    @pytest.mark.asyncio()
    async def test_search_reranking_with_hybrid_mode(
        self, search_service: SearchService, mock_collection_repo: AsyncMock, mock_collections: list[MagicMock]
    ) -> None:
        """Test that reranking works with search_mode=hybrid."""
        # Setup
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collections[0]

        # Mock vecpipe response with hybrid search and reranking
        mock_response = {
            "results": [
                {
                    "doc_id": "doc_1",
                    "chunk_id": "chunk_1",
                    "score": 0.92,
                    "reranked_score": 0.92,
                    "content": "Hybrid search result",
                    "path": "/test.txt",
                    "metadata": {},
                }
            ],
            "processing_time_ms": 120,
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response_obj

            await search_service.single_collection_search(
                user_id=1,
                collection_uuid=mock_collections[0].id,
                query="test query",
                k=10,
                search_type="semantic",
                search_mode="hybrid",
                rrf_k=60,
                use_reranker=True,
            )

            # Verify the request includes hybrid search params
            request_data = mock_client.post.call_args[1]["json"]
            assert request_data["search_type"] == "semantic"
            assert request_data["search_mode"] == "hybrid"
            assert request_data["rrf_k"] == 60
            assert request_data["use_reranker"] is True

    @pytest.mark.asyncio()
    async def test_search_reranking_error_handling(
        self, search_service: SearchService, mock_collection_repo: AsyncMock, mock_collections: list[MagicMock]
    ) -> None:
        """Test error handling when reranking fails (e.g., insufficient GPU memory)."""
        # Setup
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collections[0]

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            # Simulate insufficient memory error
            mock_error_response = MagicMock()
            mock_error_response.status_code = 507
            mock_error_response.json.return_value = {
                "detail": {
                    "error": "insufficient_memory",
                    "message": "Insufficient GPU memory for reranking",
                    "suggestion": "Try using a smaller model or different quantization",
                }
            }
            mock_client.post.side_effect = httpx.HTTPStatusError(
                message="507 Insufficient Storage", request=MagicMock(), response=mock_error_response
            )

            # The service should propagate the error
            with pytest.raises(httpx.HTTPStatusError):
                await search_service.single_collection_search(
                    user_id=1,
                    collection_uuid=mock_collections[0].id,
                    query="test query",
                    k=10,
                    use_reranker=True,
                    rerank_model="Qwen/Qwen3-Reranker-8B",  # Large model
                )

    @pytest.mark.asyncio()
    async def test_multi_collection_search_reranking_partial_failure(
        self, search_service: SearchService, mock_collection_repo: AsyncMock, mock_collections: list[MagicMock]
    ) -> None:
        """Test multi-collection search when reranking partially fails."""
        # Setup
        mock_collection_repo.get_by_uuid_with_permission_check.side_effect = mock_collections

        # Mock responses - first succeeds, second fails
        success_response = {
            "results": [
                {
                    "doc_id": "doc_1",
                    "chunk_id": "chunk_1",
                    "score": 0.9,
                    "content": "Success",
                    "path": "/doc1.txt",
                    "metadata": {},
                }
            ],
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            # First call succeeds
            success_resp_obj = MagicMock()
            success_resp_obj.json.return_value = success_response
            success_resp_obj.raise_for_status = MagicMock()

            # Second call fails
            error_resp_obj = MagicMock()
            error_resp_obj.status_code = 507

            mock_client.post.side_effect = [
                success_resp_obj,
                httpx.HTTPStatusError(message="507 Insufficient Storage", request=MagicMock(), response=error_resp_obj),
            ]

            result = await search_service.multi_collection_search(
                user_id=1,
                collection_uuids=[c.id for c in mock_collections],
                query="test query",
                k=10,
                use_reranker=True,
            )

            # Should still return results from successful collection
            assert len(result["results"]) == 1
            assert result["results"][0]["doc_id"] == "doc_1"

            # Should indicate partial failure
            assert result["metadata"]["errors"] is not None
            assert len(result["metadata"]["errors"]) == 1

            # Should have details for both collections
            collection_details = result["metadata"]["collection_details"]
            assert len(collection_details) == 2
            assert collection_details[0]["result_count"] == 1
            assert "error" in collection_details[1]

    @pytest.mark.asyncio()
    async def test_multi_collection_search_orders_by_reranked_score(
        self,
        search_service: SearchService,
        mock_collection_repo: AsyncMock,
        mock_collections: list[MagicMock],
    ) -> None:
        """Results with reranked_score should appear ahead of score-only results."""

        mock_collection_repo.get_by_uuid_with_permission_check.side_effect = mock_collections

        vecpipe_payloads = [
            {
                "results": [
                    {
                        "doc_id": "doc_high_rerank",
                        "chunk_id": "chunk_high_rerank",
                        "score": 0.72,
                        "reranked_score": 0.94,
                        "content": "High reranked content",
                        "path": "/high_rerank.md",
                        "metadata": {},
                    },
                    {
                        "doc_id": "doc_low_rerank",
                        "chunk_id": "chunk_low_rerank",
                        "score": 0.88,
                        "reranked_score": 0.55,
                        "content": "Lower reranked content",
                        "path": "/low_rerank.md",
                        "metadata": {},
                    },
                ],
            },
            {
                "results": [
                    {
                        "doc_id": "doc_no_rerank",
                        "chunk_id": "chunk_no_rerank",
                        "score": 0.81,
                        "content": "No rerank score",
                        "path": "/no_rerank.md",
                        "metadata": {},
                    }
                ],
            },
        ]

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post.side_effect = [
                MagicMock(json=MagicMock(return_value=payload), raise_for_status=MagicMock())
                for payload in vecpipe_payloads
            ]

            result = await search_service.multi_collection_search(
                user_id=1,
                collection_uuids=[c.id for c in mock_collections],
                query="rerank ordering",
                k=5,
                use_reranker=True,
            )

        doc_ids = [res["doc_id"] for res in result["results"]]
        assert doc_ids == [
            "doc_high_rerank",
            "doc_no_rerank",
            "doc_low_rerank",
        ]
