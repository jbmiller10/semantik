"""Unit tests for shared API contracts."""

import pytest
from pydantic import ValidationError
from shared.contracts.errors import (
    ErrorResponse,
    create_insufficient_memory_error,
    create_not_found_error,
    create_validation_error,
)
from shared.contracts.search import SearchRequest, SearchResult


class TestSearchContracts:
    """Test search-related contracts."""

    def test_search_request_with_k(self) -> None:
        """Test SearchRequest with canonical field 'k'."""
        req = SearchRequest(query="test query", k=5)
        assert req.query == "test query"
        assert req.k == 5
        assert req.search_type == "semantic"  # default

    def test_search_request_with_top_k_alias(self) -> None:
        """Test SearchRequest with alias field 'top_k'."""
        req_data = {"query": "test query", "top_k": 10}
        req = SearchRequest(**req_data)
        assert req.query == "test query"
        assert req.k == 10  # alias mapped to canonical field

    def test_search_request_query_validation(self) -> None:
        """Test query field validation."""
        # Empty query should fail
        with pytest.raises(ValidationError) as exc_info:
            SearchRequest(query="")
        assert "at least 1 character" in str(exc_info.value)

        # Query exceeding max length should fail
        with pytest.raises(ValidationError) as exc_info:
            SearchRequest(query="x" * 1001)
        assert "at most 1000 characters" in str(exc_info.value)

    def test_search_request_search_type_mapping(self) -> None:
        """Test search_type validation and mapping."""
        # 'vector' should be mapped to 'semantic'
        req = SearchRequest(query="test", search_type="vector")
        assert req.search_type == "semantic"

        # Invalid search type should fail
        with pytest.raises(ValidationError) as exc_info:
            SearchRequest(query="test", search_type="invalid")
        assert "Invalid search_type" in str(exc_info.value)

    def test_search_result_optional_fields(self) -> None:
        """Test SearchResult with optional fields."""
        result = SearchResult(doc_id="doc123", chunk_id="chunk456", score=0.95, path="/data/test.txt")
        assert result.content is None
        assert result.metadata == {}  # default factory
        assert result.file_name is None


class TestErrorContracts:
    """Test error-related contracts."""

    def test_basic_error_response(self) -> None:
        """Test basic ErrorResponse."""
        error = ErrorResponse(error="TestError", message="Something went wrong", status_code=500)
        assert error.error == "TestError"
        assert error.message == "Something went wrong"
        assert error.status_code == 500
        assert error.details is None

    def test_create_validation_error(self) -> None:
        """Test create_validation_error helper."""
        errors = [("field1", "Field 1 is required"), ("field2", "Field 2 must be positive")]
        error = create_validation_error(errors)

        assert error.error == "ValidationError"
        assert error.message == "Validation failed"
        assert error.status_code == 400
        assert len(error.details) == 2
        assert error.details[0].field == "field1"
        assert error.details[0].message == "Field 1 is required"

    def test_create_not_found_error(self) -> None:
        """Test create_not_found_error helper."""
        error = create_not_found_error("Collection", "collection123")

        assert error.error == "NotFoundError"
        assert error.message == "Collection not found"
        assert error.resource_type == "Collection"
        assert error.resource_id == "collection123"
        assert error.status_code == 404

    def test_create_insufficient_memory_error(self) -> None:
        """Test create_insufficient_memory_error helper."""
        error = create_insufficient_memory_error(
            required="4GB", available="2GB", suggestion="Try using a smaller model"
        )

        assert error.error == "InsufficientResourcesError"
        assert error.message == "Insufficient GPU memory for operation"
        assert error.resource_type == "gpu_memory"
        assert error.required == "4GB"
        assert error.available == "2GB"
        assert error.suggestion == "Try using a smaller model"
        assert error.status_code == 507


class TestSearchContractsExtended:
    """Extended tests for search contracts including edge cases."""

    def test_search_result_required_doc_id(self) -> None:
        """Test that doc_id is required in SearchResult."""
        # Should fail without doc_id
        with pytest.raises(ValidationError) as exc_info:
            SearchResult(chunk_id="chunk1", score=0.95, path="/test.txt")
        assert "doc_id" in str(exc_info.value)

    def test_hybrid_search_result_required_doc_id(self) -> None:
        """Test that doc_id is required in HybridSearchResult."""
        from shared.contracts.search import HybridSearchResult

        # Should fail without doc_id
        with pytest.raises(ValidationError) as exc_info:
            HybridSearchResult(path="/test.txt", chunk_id="chunk1", score=0.95)
        assert "doc_id" in str(exc_info.value)

        # Should succeed with doc_id
        result = HybridSearchResult(
            path="/test.txt",
            chunk_id="chunk1",
            score=0.95,
            doc_id="doc123",
            matched_keywords=["test", "keyword"],
            keyword_score=0.8,
            combined_score=0.875,
        )
        assert result.doc_id == "doc123"
        assert result.matched_keywords == ["test", "keyword"]

    def test_batch_search_request(self) -> None:
        """Test BatchSearchRequest validation."""
        from shared.contracts.search import BatchSearchRequest

        # Valid batch request
        batch = BatchSearchRequest(queries=["query1", "query2", "query3"], k=5, search_type="semantic")
        assert len(batch.queries) == 3
        assert batch.k == 5

        # Empty queries should fail
        with pytest.raises(ValidationError) as exc_info:
            BatchSearchRequest(queries=[])
        assert "at least 1 item" in str(exc_info.value)

        # Too many queries should fail
        with pytest.raises(ValidationError) as exc_info:
            BatchSearchRequest(queries=["q"] * 101)
        assert "at most 100 items" in str(exc_info.value)

    def test_hybrid_search_request(self) -> None:
        """Test HybridSearchRequest validation."""
        from shared.contracts.search import HybridSearchRequest

        req = HybridSearchRequest(query="test query", k=15, mode="rerank", keyword_mode="all", score_threshold=0.7)
        assert req.query == "test query"
        assert req.k == 15
        assert req.mode == "rerank"
        assert req.keyword_mode == "all"

    def test_preload_model_request_response(self) -> None:
        """Test PreloadModelRequest and Response."""
        from shared.contracts.search import PreloadModelRequest, PreloadModelResponse

        # Request
        req = PreloadModelRequest(model_name="sentence-transformers/all-MiniLM-L6-v2", quantization="float16")
        assert req.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert req.quantization == "float16"

        # Response
        resp = PreloadModelResponse(status="success", message="Model preloaded successfully")
        assert resp.status == "success"

    def test_search_response(self) -> None:
        """Test SearchResponse model."""
        from shared.contracts.search import SearchResponse

        response = SearchResponse(
            query="test query",
            results=[
                SearchResult(doc_id="doc1", chunk_id="chunk1", score=0.95, path="/test1.txt"),
                SearchResult(doc_id="doc2", chunk_id="chunk2", score=0.90, path="/test2.txt"),
            ],
            num_results=2,
            search_type="semantic",
            model_used="test-model",
            embedding_time_ms=10.5,
            search_time_ms=5.3,
            reranking_used=True,
            reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            reranking_time_ms=15.2,
        )
        assert len(response.results) == 2
        assert response.embedding_time_ms == 10.5
        assert response.reranking_used is True

    def test_populate_by_name_behavior(self) -> None:
        """Test that populate_by_name allows both field names."""
        from shared.contracts.search import SearchRequest

        # Should accept both 'k' and 'top_k'
        req1 = SearchRequest.model_validate({"query": "test", "k": 5})
        req2 = SearchRequest.model_validate({"query": "test", "top_k": 5})
        assert req1.k == 5
        assert req2.k == 5

        # Should not accept random field names
        with pytest.raises(ValidationError):
            SearchRequest.model_validate({"query": "test", "random_field": "value"})


class TestErrorContractsExtended:
    """Extended tests for error contracts."""

    def test_validation_error_response(self) -> None:
        """Test ValidationErrorResponse model."""
        from shared.contracts.errors import ErrorDetail, ValidationErrorResponse

        error = ValidationErrorResponse(
            error="ValidationError",
            message="Multiple validation errors",
            status_code=400,
            details=[
                ErrorDetail(field="name", message="Name is required"),
                ErrorDetail(field="age", message="Age must be positive"),
            ],
        )
        assert len(error.details) == 2
        assert error.details[0].field == "name"

    def test_not_found_error_response(self) -> None:
        """Test NotFoundErrorResponse model."""
        from shared.contracts.errors import NotFoundErrorResponse

        error = NotFoundErrorResponse(
            error="NotFoundError",
            message="Resource not found",
            status_code=404,
            resource_type="Collection",
            resource_id="collection123",
        )
        assert error.resource_type == "Collection"
        assert error.resource_id == "collection123"

    def test_insufficient_resources_error(self) -> None:
        """Test InsufficientResourcesErrorResponse model."""
        from shared.contracts.errors import InsufficientResourcesErrorResponse

        error = InsufficientResourcesErrorResponse(
            error="InsufficientResourcesError",
            message="Not enough GPU memory",
            status_code=507,
            resource_type="gpu_memory",
            required="8GB",
            available="4GB",
            suggestion="Try reducing batch size",
        )
        assert error.required == "8GB"
        assert error.available == "4GB"


class TestStringLengthValidation:
    """Test max_length validation for string fields to prevent DoS attacks."""

    def test_search_result_max_length_validation(self) -> None:
        """Test that string fields in SearchResult respect max_length."""
        # Valid lengths
        result = SearchResult(
            doc_id="d" * 200,  # Max length 200
            chunk_id="c" * 200,  # Max length 200
            score=0.95,
            path="/" + "p" * 4095,  # Max length 4096
            content="x" * 10000,  # Max length 10000
            file_name="f" * 255,  # Max length 255
            operation_id="o" * 200,  # Max length 200
        )
        assert len(result.doc_id) == 200
        assert len(result.path) == 4096

        # Exceeding max length should fail
        with pytest.raises(ValidationError) as exc_info:
            SearchResult(doc_id="d" * 201, chunk_id="chunk1", score=0.95, path="/test.txt")  # Exceeds max length
        assert "at most 200 characters" in str(exc_info.value)

        # Path exceeding max length
        with pytest.raises(ValidationError) as exc_info:
            SearchResult(doc_id="doc1", chunk_id="chunk1", score=0.95, path="/" + "p" * 4096)  # 4097 chars, exceeds max
        assert "at most 4096 characters" in str(exc_info.value)

    def test_batch_search_request_query_validation(self) -> None:
        """Test that each query in BatchSearchRequest respects max length."""
        from shared.contracts.search import BatchSearchRequest

        # Valid queries
        batch = BatchSearchRequest(queries=["q" * 1000, "short query", "x" * 500])
        assert len(batch.queries[0]) == 1000

        # Query exceeding max length
        with pytest.raises(ValidationError) as exc_info:
            BatchSearchRequest(queries=["valid query", "x" * 1001])  # Second query exceeds max
        assert "Each query must not exceed 1000 characters" in str(exc_info.value)

    def test_error_response_max_length_validation(self) -> None:
        """Test max_length validation for error response models."""
        # Valid lengths
        error = ErrorResponse(
            error="E" * 100,  # Max length 100
            message="M" * 1000,  # Max length 1000
            request_id="R" * 100,  # Max length 100
            timestamp="2024-01-01T00:00:00Z",
        )
        assert len(error.error) == 100
        assert len(error.message) == 1000

        # Message exceeding max length
        with pytest.raises(ValidationError) as exc_info:
            ErrorResponse(error="TestError", message="M" * 1001)  # Exceeds max length
        assert "at most 1000 characters" in str(exc_info.value)
