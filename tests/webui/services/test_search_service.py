"""
Comprehensive tests for SearchService covering all methods and edge cases.
"""

import asyncio
from typing import Any
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from packages.shared.database.exceptions import AccessDeniedError, EntityNotFoundError
from packages.shared.database.models import Collection, CollectionStatus
from packages.shared.database.repositories.collection_repository import CollectionRepository
from packages.webui.services.search_service import SearchService

# Database and repository fixtures are now imported from conftest.py


@pytest.fixture()
def search_service(mock_db_session: AsyncMock, mock_collection_repo: AsyncMock) -> SearchService:
    """Create a SearchService instance with mocked dependencies."""
    return SearchService(db_session=mock_db_session, collection_repo=mock_collection_repo)


@pytest.fixture()
def mock_collection() -> MagicMock:
    """Mock a single collection object."""
    collection = MagicMock(spec=Collection)
    collection.id = "123e4567-e89b-12d3-a456-426614174000"
    collection.name = "Test Collection"
    collection.vector_store_name = "qdrant_test_collection"
    collection.embedding_model = "test-embedding-model"
    collection.quantization = "float16"
    collection.status = CollectionStatus.READY
    return collection


@pytest.fixture()
def mock_collections() -> list[MagicMock]:
    """Mock multiple collection objects."""
    collection1 = MagicMock(spec=Collection)
    collection1.id = "123e4567-e89b-12d3-a456-426614174000"
    collection1.name = "Collection 1"
    collection1.vector_store_name = "qdrant_collection_1"
    collection1.embedding_model = "model-1"
    collection1.quantization = "float16"
    collection1.status = CollectionStatus.READY

    collection2 = MagicMock(spec=Collection)
    collection2.id = "456e7890-e89b-12d3-a456-426614174001"
    collection2.name = "Collection 2"
    collection2.vector_store_name = "qdrant_collection_2"
    collection2.embedding_model = "model-2"
    collection2.quantization = "int8"
    collection2.status = CollectionStatus.READY

    collection3 = MagicMock(spec=Collection)
    collection3.id = "789e0123-e89b-12d3-a456-426614174002"
    collection3.name = "Collection 3"
    collection3.vector_store_name = "qdrant_collection_3"
    collection3.embedding_model = "model-3"
    collection3.quantization = "binary"
    collection3.status = CollectionStatus.PROCESSING  # Not ready

    return [collection1, collection2, collection3]


class TestSearchServiceInit:
    """Test SearchService initialization."""

    def test_init_with_default_timeout(self, mock_db_session: AsyncMock, mock_collection_repo: AsyncMock) -> None:
        """Test service initialization with default timeout."""
        service = SearchService(db_session=mock_db_session, collection_repo=mock_collection_repo)
        
        assert service.db_session == mock_db_session
        assert service.collection_repo == mock_collection_repo
        # httpx.Timeout object doesn't have direct attribute access
        # Check that it's an httpx.Timeout instance with correct values
        import httpx
        assert isinstance(service.default_timeout, httpx.Timeout)
        assert service.default_timeout.connect == 5.0
        assert service.default_timeout.read == 30.0
        assert service.default_timeout.write == 5.0
        assert service.retry_timeout_multiplier == 4.0

    def test_init_with_custom_timeout(self, mock_db_session: AsyncMock, mock_collection_repo: AsyncMock) -> None:
        """Test service initialization with custom timeout."""
        custom_timeout = httpx.Timeout(timeout=60.0, connect=10.0, read=60.0, write=10.0)
        service = SearchService(
            db_session=mock_db_session,
            collection_repo=mock_collection_repo,
            default_timeout=custom_timeout,
            retry_timeout_multiplier=2.0
        )
        
        assert service.default_timeout == custom_timeout
        assert service.retry_timeout_multiplier == 2.0


class TestValidateCollectionAccess:
    """Test validate_collection_access method."""

    @pytest.mark.asyncio()
    async def test_validate_collection_access_success(
        self, search_service: SearchService, mock_collection_repo: AsyncMock, mock_collections: list[MagicMock]
    ) -> None:
        """Test successful validation of collection access."""
        mock_collection_repo.get_by_uuid_with_permission_check.side_effect = mock_collections[:2]
        
        result = await search_service.validate_collection_access(
            [mock_collections[0].id, mock_collections[1].id], user_id=1
        )
        
        assert len(result) == 2
        assert result[0] == mock_collections[0]
        assert result[1] == mock_collections[1]
        assert mock_collection_repo.get_by_uuid_with_permission_check.call_count == 2

    @pytest.mark.asyncio()
    async def test_validate_collection_access_not_found(
        self, search_service: SearchService, mock_collection_repo: AsyncMock
    ) -> None:
        """Test validation when collection is not found."""
        mock_collection_repo.get_by_uuid_with_permission_check.side_effect = EntityNotFoundError("Collection", "invalid-uuid")
        
        with pytest.raises(AccessDeniedError) as exc_info:
            await search_service.validate_collection_access(["invalid-uuid"], user_id=1)
        
        assert "Access denied or collection not found: invalid-uuid" in str(exc_info.value)

    @pytest.mark.asyncio()
    async def test_validate_collection_access_denied(
        self, search_service: SearchService, mock_collection_repo: AsyncMock
    ) -> None:
        """Test validation when access is denied."""
        mock_collection_repo.get_by_uuid_with_permission_check.side_effect = AccessDeniedError("1", "Collection", "some-uuid")
        
        with pytest.raises(AccessDeniedError) as exc_info:
            await search_service.validate_collection_access(["some-uuid"], user_id=1)
        
        assert "Access denied or collection not found: some-uuid" in str(exc_info.value)

    @pytest.mark.asyncio()
    async def test_validate_collection_access_mixed_permissions(
        self, search_service: SearchService, mock_collection_repo: AsyncMock, mock_collections: list[MagicMock]
    ) -> None:
        """Test validation with mixed permissions (some allowed, some denied)."""
        mock_collection_repo.get_by_uuid_with_permission_check.side_effect = [
            mock_collections[0],
            AccessDeniedError("1", "Collection", "denied-uuid")
        ]
        
        with pytest.raises(AccessDeniedError) as exc_info:
            await search_service.validate_collection_access(
                [mock_collections[0].id, "denied-uuid"], user_id=1
            )
        
        assert "Access denied or collection not found: denied-uuid" in str(exc_info.value)


class TestSearchSingleCollection:
    """Test search_single_collection method."""

    @pytest.mark.asyncio()
    async def test_search_single_collection_success(
        self, search_service: SearchService, mock_collection: MagicMock
    ) -> None:
        """Test successful single collection search."""
        mock_response = {
            "results": [
                {
                    "doc_id": "doc_1",
                    "chunk_id": "chunk_1",
                    "score": 0.95,
                    "content": "Test content",
                    "path": "/test.txt",
                    "metadata": {"key": "value"},
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

            result = await search_service.search_single_collection(
                collection=mock_collection,
                query="test query",
                k=10,
                search_params={"search_type": "semantic"},
            )

            assert result[0] == mock_collection
            assert result[1] == mock_response["results"]
            assert result[2] is None  # No error

            # Verify request parameters
            call_args = mock_client.post.call_args
            request_data = call_args[1]["json"]
            assert request_data["query"] == "test query"
            assert request_data["k"] == 10
            assert request_data["collection"] == mock_collection.vector_store_name
            assert request_data["model_name"] == mock_collection.embedding_model
            assert request_data["quantization"] == mock_collection.quantization
            assert request_data["include_content"] is True
            assert request_data["search_type"] == "semantic"

    @pytest.mark.asyncio()
    async def test_search_single_collection_not_ready(
        self, search_service: SearchService, mock_collection: MagicMock
    ) -> None:
        """Test search when collection is not ready."""
        mock_collection.status = CollectionStatus.PROCESSING
        
        result = await search_service.search_single_collection(
            collection=mock_collection,
            query="test query",
            k=10,
            search_params={},
        )
        
        assert result[0] == mock_collection
        assert result[1] is None
        assert f"Collection {mock_collection.name} is not ready for search" in result[2]

    @pytest.mark.asyncio()
    async def test_search_single_collection_timeout_with_retry(
        self, search_service: SearchService, mock_collection: MagicMock
    ) -> None:
        """Test search timeout with successful retry."""
        mock_response = {
            "results": [{"doc_id": "doc_1", "score": 0.9}],
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            # First call times out
            mock_client.post.side_effect = [
                httpx.ReadTimeout("Request timed out"),
                # Second call (retry) succeeds
                MagicMock(json=lambda: mock_response, raise_for_status=lambda: None)
            ]

            result = await search_service.search_single_collection(
                collection=mock_collection,
                query="test query",
                k=10,
                search_params={},
            )

            assert result[0] == mock_collection
            assert result[1] == mock_response["results"]
            assert result[2] is None
            assert mock_client.post.call_count == 2

            # Verify retry used extended timeout
            retry_call = mock_client_class.call_args_list[1]
            retry_timeout = retry_call[1]["timeout"]
            # httpx.Timeout doesn't have .timeout attribute
            # The overall timeout is increased but individual components are also multiplied
            assert retry_timeout.connect == 20.0   # 5.0 * 4.0
            assert retry_timeout.read == 120.0     # 30.0 * 4.0
            assert retry_timeout.write == 20.0     # 5.0 * 4.0

    @pytest.mark.asyncio()
    async def test_search_single_collection_timeout_retry_fails(
        self, search_service: SearchService, mock_collection: MagicMock
    ) -> None:
        """Test search timeout with failed retry."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            # Both calls timeout
            mock_client.post.side_effect = [
                httpx.ReadTimeout("Request timed out"),
                httpx.ReadTimeout("Request timed out again")
            ]

            result = await search_service.search_single_collection(
                collection=mock_collection,
                query="test query",
                k=10,
                search_params={},
            )

            assert result[0] == mock_collection
            assert result[1] is None
            assert "Search failed after retry" in result[2]

    @pytest.mark.asyncio()
    async def test_search_single_collection_http_errors(
        self, search_service: SearchService, mock_collection: MagicMock
    ) -> None:
        """Test handling of various HTTP errors."""
        error_cases = [
            (404, f"Collection '{mock_collection.name}' not found in vector store"),
            (403, f"Access denied to collection '{mock_collection.name}'"),
            (429, f"Rate limit exceeded for collection '{mock_collection.name}'"),
            (500, f"Search service unavailable for collection '{mock_collection.name}' (status: 500)"),
            (502, f"Search service unavailable for collection '{mock_collection.name}' (status: 502)"),
            (400, f"Search failed for collection '{mock_collection.name}' (status: 400)"),
        ]

        for status_code, expected_error in error_cases:
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client_class.return_value.__aenter__.return_value = mock_client
                
                mock_response = MagicMock()
                mock_response.status_code = status_code
                mock_client.post.side_effect = httpx.HTTPStatusError(
                    message=f"HTTP {status_code}",
                    request=MagicMock(),
                    response=mock_response
                )

                result = await search_service.search_single_collection(
                    collection=mock_collection,
                    query="test query",
                    k=10,
                    search_params={},
                )

                assert result[0] == mock_collection
                assert result[1] is None
                assert result[2] == expected_error

    @pytest.mark.asyncio()
    async def test_search_single_collection_connect_error(
        self, search_service: SearchService, mock_collection: MagicMock
    ) -> None:
        """Test handling of connection errors."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post.side_effect = httpx.ConnectError("Cannot connect")

            result = await search_service.search_single_collection(
                collection=mock_collection,
                query="test query",
                k=10,
                search_params={},
            )

            assert result[0] == mock_collection
            assert result[1] is None
            assert f"Cannot connect to search service for collection '{mock_collection.name}'" in result[2]

    @pytest.mark.asyncio()
    async def test_search_single_collection_request_error(
        self, search_service: SearchService, mock_collection: MagicMock
    ) -> None:
        """Test handling of request errors."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post.side_effect = httpx.RequestError("Network error")

            result = await search_service.search_single_collection(
                collection=mock_collection,
                query="test query",
                k=10,
                search_params={},
            )

            assert result[0] == mock_collection
            assert result[1] is None
            assert f"Network error searching collection '{mock_collection.name}'" in result[2]

    @pytest.mark.asyncio()
    async def test_search_single_collection_unexpected_error(
        self, search_service: SearchService, mock_collection: MagicMock
    ) -> None:
        """Test handling of unexpected errors."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post.side_effect = ValueError("Unexpected error")

            result = await search_service.search_single_collection(
                collection=mock_collection,
                query="test query",
                k=10,
                search_params={},
            )

            assert result[0] == mock_collection
            assert result[1] is None
            assert f"Unexpected error searching collection '{mock_collection.name}'" in result[2]

    @pytest.mark.asyncio()
    async def test_search_single_collection_with_custom_timeout(
        self, search_service: SearchService, mock_collection: MagicMock
    ) -> None:
        """Test search with custom timeout parameter."""
        custom_timeout = httpx.Timeout(timeout=60.0, connect=10.0, read=60.0, write=10.0)
        
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = {"results": []}
            mock_response_obj.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response_obj

            await search_service.search_single_collection(
                collection=mock_collection,
                query="test query",
                k=10,
                search_params={},
                timeout=custom_timeout,
            )

            # Verify custom timeout was used
            call_args = mock_client_class.call_args
            assert call_args[1]["timeout"] == custom_timeout


class TestHandleHttpError:
    """Test _handle_http_error method."""

    def test_handle_http_error_various_status_codes(
        self, search_service: SearchService, mock_collection: MagicMock
    ) -> None:
        """Test error message generation for various HTTP status codes."""
        test_cases = [
            (404, False, f"Collection '{mock_collection.name}' not found in vector store"),
            (404, True, f"Collection '{mock_collection.name}' not found in vector store after retry"),
            (403, False, f"Access denied to collection '{mock_collection.name}'"),
            (429, False, f"Rate limit exceeded for collection '{mock_collection.name}'"),
            (500, False, f"Search service unavailable for collection '{mock_collection.name}' (status: 500)"),
            (503, False, f"Search service unavailable for collection '{mock_collection.name}' (status: 503)"),
            (400, False, f"Search failed for collection '{mock_collection.name}' (status: 400)"),
            (422, True, f"Search failed for collection '{mock_collection.name}' after retry (status: 422)"),
        ]

        for status_code, retry, expected_message in test_cases:
            mock_response = MagicMock()
            mock_response.status_code = status_code
            error = httpx.HTTPStatusError(
                message=f"HTTP {status_code}",
                request=MagicMock(),
                response=mock_response
            )

            result = search_service._handle_http_error(error, mock_collection, retry)

            assert result[0] == mock_collection
            assert result[1] is None
            assert result[2] == expected_message


class TestMultiCollectionSearch:
    """Test multi_collection_search method."""

    @pytest.mark.asyncio()
    async def test_multi_collection_search_success(
        self, search_service: SearchService, mock_collection_repo: AsyncMock, mock_collections: list[MagicMock]
    ) -> None:
        """Test successful multi-collection search."""
        # Setup - only use ready collections
        ready_collections = mock_collections[:2]
        mock_collection_repo.get_by_uuid_with_permission_check.side_effect = ready_collections

        # Mock search responses
        search_responses = [
            {
                "results": [
                    {"doc_id": "doc_1", "score": 0.95, "content": "Result from collection 1"},
                    {"doc_id": "doc_2", "score": 0.85, "content": "Another result from collection 1"},
                ]
            },
            {
                "results": [
                    {"doc_id": "doc_3", "score": 0.90, "content": "Result from collection 2"},
                ]
            },
        ]

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            response_objs = []
            for resp in search_responses:
                resp_obj = MagicMock()
                resp_obj.json.return_value = resp
                resp_obj.raise_for_status = MagicMock()
                response_objs.append(resp_obj)
            
            mock_client.post.side_effect = response_objs

            result = await search_service.multi_collection_search(
                user_id=1,
                collection_uuids=[c.id for c in ready_collections],
                query="test query",
                k=10,
                search_type="semantic",
                score_threshold=0.5,
                use_reranker=False,
            )

            # Verify results are sorted by score
            assert len(result["results"]) == 3
            assert result["results"][0]["score"] == 0.95
            assert result["results"][0]["collection_id"] == ready_collections[0].id
            assert result["results"][1]["score"] == 0.90
            assert result["results"][1]["collection_id"] == ready_collections[1].id
            assert result["results"][2]["score"] == 0.85

            # Verify metadata
            metadata = result["metadata"]
            assert metadata["total_results"] == 3
            assert metadata["collections_searched"] == 2
            assert metadata["errors"] is None
            assert metadata["processing_time"] > 0

            # Verify collection details
            collection_details = metadata["collection_details"]
            assert len(collection_details) == 2
            assert collection_details[0]["collection_id"] == ready_collections[0].id
            assert collection_details[0]["result_count"] == 2
            assert collection_details[1]["collection_id"] == ready_collections[1].id
            assert collection_details[1]["result_count"] == 1

    @pytest.mark.asyncio()
    async def test_multi_collection_search_with_hybrid_params(
        self, search_service: SearchService, mock_collection_repo: AsyncMock, mock_collection: MagicMock
    ) -> None:
        """Test multi-collection search with hybrid search parameters."""
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = {"results": []}
            mock_response_obj.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response_obj

            await search_service.multi_collection_search(
                user_id=1,
                collection_uuids=[mock_collection.id],
                query="test query",
                k=10,
                search_type="hybrid",
                hybrid_alpha=0.7,
                hybrid_search_mode="weighted",
            )

            # Verify hybrid parameters were included
            call_args = mock_client.post.call_args
            request_data = call_args[1]["json"]
            assert request_data["search_type"] == "hybrid"
            assert request_data["hybrid_alpha"] == 0.7
            assert request_data["hybrid_search_mode"] == "weighted"

    @pytest.mark.asyncio()
    async def test_multi_collection_search_partial_failures(
        self, search_service: SearchService, mock_collection_repo: AsyncMock, mock_collections: list[MagicMock]
    ) -> None:
        """Test multi-collection search with some collections failing."""
        ready_collections = mock_collections[:3]
        ready_collections[2].status = CollectionStatus.READY  # Make all ready for this test
        mock_collection_repo.get_by_uuid_with_permission_check.side_effect = ready_collections

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            # First collection succeeds
            success_resp = MagicMock()
            success_resp.json.return_value = {
                "results": [{"doc_id": "doc_1", "score": 0.9}]
            }
            success_resp.raise_for_status = MagicMock()
            
            # Second collection fails with 404
            error_resp = MagicMock()
            error_resp.status_code = 404
            
            # Third collection times out
            mock_client.post.side_effect = [
                success_resp,
                httpx.HTTPStatusError("404", request=MagicMock(), response=error_resp),
                httpx.ConnectError("Connection failed"),
            ]

            result = await search_service.multi_collection_search(
                user_id=1,
                collection_uuids=[c.id for c in ready_collections],
                query="test query",
                k=10,
            )

            # Should have results from successful collection
            assert len(result["results"]) == 1
            assert result["results"][0]["doc_id"] == "doc_1"

            # Should have errors
            errors = result["metadata"]["errors"]
            assert errors is not None
            assert len(errors) == 2
            assert "not found in vector store" in errors[0]
            assert "Cannot connect" in errors[1]

            # Collection details should reflect successes and failures
            collection_details = result["metadata"]["collection_details"]
            assert len(collection_details) == 3
            assert collection_details[0]["result_count"] == 1
            assert "error" not in collection_details[0]
            assert collection_details[1]["result_count"] == 0
            assert "error" in collection_details[1]
            assert collection_details[2]["result_count"] == 0
            assert "error" in collection_details[2]

    @pytest.mark.asyncio()
    async def test_multi_collection_search_result_limit(
        self, search_service: SearchService, mock_collection_repo: AsyncMock, mock_collections: list[MagicMock]
    ) -> None:
        """Test that results are limited to k parameter."""
        ready_collections = mock_collections[:2]
        mock_collection_repo.get_by_uuid_with_permission_check.side_effect = ready_collections

        # Create many results
        many_results = [{"doc_id": f"doc_{i}", "score": 0.9 - i * 0.01} for i in range(10)]
        
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            response_objs = []
            for _ in range(2):
                resp_obj = MagicMock()
                resp_obj.json.return_value = {"results": many_results[:5]}  # 5 results each
                resp_obj.raise_for_status = MagicMock()
                response_objs.append(resp_obj)
            
            mock_client.post.side_effect = response_objs

            result = await search_service.multi_collection_search(
                user_id=1,
                collection_uuids=[c.id for c in ready_collections],
                query="test query",
                k=3,  # Limit to 3 results
            )

            # Should only return k results
            assert len(result["results"]) == 3
            # Should be the highest scoring results
            assert all(result["results"][i]["score"] >= result["results"][i+1]["score"] for i in range(2))

    @pytest.mark.asyncio()
    async def test_multi_collection_search_empty_results(
        self, search_service: SearchService, mock_collection_repo: AsyncMock, mock_collection: MagicMock
    ) -> None:
        """Test multi-collection search with no results."""
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = {"results": []}
            mock_response_obj.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response_obj

            result = await search_service.multi_collection_search(
                user_id=1,
                collection_uuids=[mock_collection.id],
                query="test query",
                k=10,
            )

            assert len(result["results"]) == 0
            assert result["metadata"]["total_results"] == 0
            assert result["metadata"]["collection_details"][0]["result_count"] == 0

    @pytest.mark.asyncio()
    async def test_multi_collection_search_access_denied(
        self, search_service: SearchService, mock_collection_repo: AsyncMock
    ) -> None:
        """Test multi-collection search when access is denied."""
        mock_collection_repo.get_by_uuid_with_permission_check.side_effect = AccessDeniedError("1", "Collection", "some-uuid")

        with pytest.raises(AccessDeniedError):
            await search_service.multi_collection_search(
                user_id=1,
                collection_uuids=["some-uuid"],
                query="test query",
            )


class TestSingleCollectionSearch:
    """Test single_collection_search method."""

    @pytest.mark.asyncio()
    async def test_single_collection_search_success(
        self, search_service: SearchService, mock_collection_repo: AsyncMock, mock_collection: MagicMock
    ) -> None:
        """Test successful single collection search."""
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        mock_response = {
            "results": [
                {"doc_id": "doc_1", "score": 0.95, "content": "Test content"}
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
                collection_uuid=mock_collection.id,
                query="test query",
                k=10,
                search_type="semantic",
                score_threshold=0.5,
                metadata_filter={"key": "value"},
                use_reranker=True,
                rerank_model="test-reranker",
                include_content=True,
            )

            assert result == mock_response

            # Verify request parameters
            call_args = mock_client.post.call_args
            request_data = call_args[1]["json"]
            assert request_data["query"] == "test query"
            assert request_data["k"] == 10
            assert request_data["collection"] == mock_collection.vector_store_name
            assert request_data["model_name"] == mock_collection.embedding_model
            assert request_data["quantization"] == mock_collection.quantization
            assert request_data["search_type"] == "semantic"
            assert request_data["score_threshold"] == 0.5
            assert request_data["filters"] == {"key": "value"}
            assert request_data["use_reranker"] is True
            assert request_data["rerank_model"] == "test-reranker"
            assert request_data["include_content"] is True

            # Verify timeout was doubled
            timeout_used = mock_client_class.call_args[1]["timeout"]
            # httpx.Timeout doesn't have .timeout attribute, only .connect, .read, .write
            assert timeout_used.read == 60.0     # 30.0 * 2

    @pytest.mark.asyncio()
    async def test_single_collection_search_with_hybrid(
        self, search_service: SearchService, mock_collection_repo: AsyncMock, mock_collection: MagicMock
    ) -> None:
        """Test single collection search with hybrid parameters."""
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = {"results": []}
            mock_response_obj.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response_obj

            await search_service.single_collection_search(
                user_id=1,
                collection_uuid=mock_collection.id,
                query="test query",
                search_type="hybrid",
                hybrid_alpha=0.8,
                hybrid_search_mode="reciprocal",
            )

            # Verify hybrid parameters
            request_data = mock_client.post.call_args[1]["json"]
            assert request_data["search_type"] == "hybrid"
            assert request_data["hybrid_alpha"] == 0.8
            assert request_data["hybrid_search_mode"] == "reciprocal"

    @pytest.mark.asyncio()
    async def test_single_collection_search_http_errors(
        self, search_service: SearchService, mock_collection_repo: AsyncMock, mock_collection: MagicMock
    ) -> None:
        """Test single collection search HTTP error handling."""
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        # Test 404 error
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_client.post.side_effect = httpx.HTTPStatusError(
                message="404 Not Found",
                request=MagicMock(),
                response=mock_response
            )

            with pytest.raises(EntityNotFoundError) as exc_info:
                await search_service.single_collection_search(
                    user_id=1,
                    collection_uuid=mock_collection.id,
                    query="test query",
                )
            
            assert f"Collection '{mock_collection.name}' not found in vector store" in str(exc_info.value)

        # Test 403 error
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_response = MagicMock()
            mock_response.status_code = 403
            mock_client.post.side_effect = httpx.HTTPStatusError(
                message="403 Forbidden",
                request=MagicMock(),
                response=mock_response
            )

            with pytest.raises(AccessDeniedError) as exc_info:
                await search_service.single_collection_search(
                    user_id=1,
                    collection_uuid=mock_collection.id,
                    query="test query",
                )
            
            assert f"Access denied to collection '{mock_collection.name}'" in str(exc_info.value)

        # Test other HTTP errors
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_client.post.side_effect = httpx.HTTPStatusError(
                message="500 Server Error",
                request=MagicMock(),
                response=mock_response
            )

            with pytest.raises(httpx.HTTPStatusError):
                await search_service.single_collection_search(
                    user_id=1,
                    collection_uuid=mock_collection.id,
                    query="test query",
                )

    @pytest.mark.asyncio()
    async def test_single_collection_search_general_error(
        self, search_service: SearchService, mock_collection_repo: AsyncMock, mock_collection: MagicMock
    ) -> None:
        """Test single collection search general error handling."""
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post.side_effect = ValueError("Some error")

            with pytest.raises(ValueError):
                await search_service.single_collection_search(
                    user_id=1,
                    collection_uuid=mock_collection.id,
                    query="test query",
                )

    @pytest.mark.asyncio()
    async def test_single_collection_search_access_validation(
        self, search_service: SearchService, mock_collection_repo: AsyncMock
    ) -> None:
        """Test that single collection search validates access."""
        mock_collection_repo.get_by_uuid_with_permission_check.side_effect = AccessDeniedError("1", "Collection", "some-uuid")

        with pytest.raises(AccessDeniedError):
            await search_service.single_collection_search(
                user_id=1,
                collection_uuid="some-uuid",
                query="test query",
            )


class TestSearchServiceEdgeCases:
    """Test edge cases and special scenarios."""

    @pytest.mark.asyncio()
    async def test_search_with_none_timeout_values(
        self, search_service: SearchService, mock_collection: MagicMock
    ) -> None:
        """Test retry logic when timeout has None values."""
        # Create a timeout with None values
        partial_timeout = httpx.Timeout(timeout=None, connect=None, read=None, write=None)
        
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            # First call times out
            mock_client.post.side_effect = [
                httpx.ReadTimeout("Request timed out"),
                MagicMock(json=lambda: {"results": []}, raise_for_status=lambda: None)
            ]

            result = await search_service.search_single_collection(
                collection=mock_collection,
                query="test query",
                k=10,
                search_params={},
                timeout=partial_timeout,
            )

            # Verify retry used default values when original was None
            retry_call = mock_client_class.call_args_list[1]
            retry_timeout = retry_call[1]["timeout"]
            # httpx.Timeout doesn't have .timeout attribute
            # Check that timeout values are multiplied correctly
            assert retry_timeout.connect == 20.0   # Default when None
            assert retry_timeout.read == 120.0     # Default when None
            assert retry_timeout.write == 20.0     # Default when None

    @pytest.mark.asyncio()
    async def test_multi_collection_search_mixed_statuses(
        self, search_service: SearchService, mock_collection_repo: AsyncMock, mock_collections: list[MagicMock]
    ) -> None:
        """Test multi-collection search with collections in different statuses."""
        # Collection 3 is not ready (PROCESSING status)
        all_collections = mock_collections[:3]
        mock_collection_repo.get_by_uuid_with_permission_check.side_effect = all_collections

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            # Only ready collections should be searched
            response_objs = []
            for i in range(2):  # Only 2 ready collections
                resp_obj = MagicMock()
                resp_obj.json.return_value = {
                    "results": [{"doc_id": f"doc_{i}", "score": 0.9}]
                }
                resp_obj.raise_for_status = MagicMock()
                response_objs.append(resp_obj)
            
            mock_client.post.side_effect = response_objs

            result = await search_service.multi_collection_search(
                user_id=1,
                collection_uuids=[c.id for c in all_collections],
                query="test query",
            )

            # Should only search 2 ready collections
            assert mock_client.post.call_count == 2
            
            # Should have results from ready collections
            assert len(result["results"]) == 2
            
            # Should have error for non-ready collection
            errors = result["metadata"]["errors"]
            assert errors is not None
            assert len(errors) == 1
            assert "not ready for search" in errors[0]
            
            # Collection details should show all 3
            collection_details = result["metadata"]["collection_details"]
            assert len(collection_details) == 3
            assert collection_details[2]["result_count"] == 0
            assert "error" in collection_details[2]

    @pytest.mark.asyncio()
    async def test_search_retry_with_http_error_after_timeout(
        self, search_service: SearchService, mock_collection: MagicMock
    ) -> None:
        """Test retry logic when HTTP error occurs after timeout retry."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            # First call times out
            # Second call (retry) gets HTTP error
            mock_response = MagicMock()
            mock_response.status_code = 429
            mock_client.post.side_effect = [
                httpx.ReadTimeout("Request timed out"),
                httpx.HTTPStatusError("429 Too Many Requests", request=MagicMock(), response=mock_response)
            ]

            result = await search_service.search_single_collection(
                collection=mock_collection,
                query="test query",
                k=10,
                search_params={},
            )

            assert result[0] == mock_collection
            assert result[1] is None
            assert "Rate limit exceeded" in result[2]
            assert "after retry" in result[2]

    @pytest.mark.asyncio()
    async def test_parallel_search_execution(
        self, search_service: SearchService, mock_collection_repo: AsyncMock, mock_collections: list[MagicMock]
    ) -> None:
        """Test that searches are executed in parallel for multiple collections."""
        ready_collections = mock_collections[:2]
        mock_collection_repo.get_by_uuid_with_permission_check.side_effect = ready_collections

        search_delays = [0.1, 0.1]  # Both searches take 100ms
        call_order = []

        async def mock_post(*args: Any, **kwargs: Any) -> MagicMock:
            collection_name = kwargs["json"]["collection"]
            call_order.append(f"start_{collection_name}")
            await asyncio.sleep(search_delays.pop(0))
            call_order.append(f"end_{collection_name}")
            
            resp = MagicMock()
            resp.json.return_value = {"results": []}
            resp.raise_for_status = MagicMock()
            return resp

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post.side_effect = mock_post

            import time
            start_time = time.time()
            
            await search_service.multi_collection_search(
                user_id=1,
                collection_uuids=[c.id for c in ready_collections],
                query="test query",
            )
            
            elapsed_time = time.time() - start_time

            # If executed in parallel, should take ~100ms, not 200ms
            assert elapsed_time < 0.15  # Allow some overhead
            
            # Verify interleaved execution
            assert call_order[0].startswith("start_")
            assert call_order[1].startswith("start_")
            assert call_order[2].startswith("end_")
            assert call_order[3].startswith("end_")