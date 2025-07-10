"""Unit tests for shared API contracts."""

import pytest
from pydantic import ValidationError

from shared.contracts.errors import (
    ErrorResponse,
    create_insufficient_memory_error,
    create_not_found_error,
    create_validation_error,
)
from shared.contracts.jobs import CreateJobRequest, JobResponse
from shared.contracts.search import SearchRequest, SearchResponse, SearchResult


class TestSearchContracts:
    """Test search-related contracts."""

    def test_search_request_with_k(self):
        """Test SearchRequest with canonical field 'k'."""
        req = SearchRequest(query="test query", k=5)
        assert req.query == "test query"
        assert req.k == 5
        assert req.search_type == "semantic"  # default

    def test_search_request_with_top_k_alias(self):
        """Test SearchRequest with alias field 'top_k'."""
        req_data = {"query": "test query", "top_k": 10}
        req = SearchRequest(**req_data)
        assert req.query == "test query"
        assert req.k == 10  # alias mapped to canonical field

    def test_search_request_query_validation(self):
        """Test query field validation."""
        # Empty query should fail
        with pytest.raises(ValidationError) as exc_info:
            SearchRequest(query="")
        assert "at least 1 character" in str(exc_info.value)

        # Query exceeding max length should fail
        with pytest.raises(ValidationError) as exc_info:
            SearchRequest(query="x" * 1001)
        assert "at most 1000 characters" in str(exc_info.value)

    def test_search_request_search_type_mapping(self):
        """Test search_type validation and mapping."""
        # 'vector' should be mapped to 'semantic'
        req = SearchRequest(query="test", search_type="vector")
        assert req.search_type == "semantic"

        # Invalid search type should fail
        with pytest.raises(ValidationError) as exc_info:
            SearchRequest(query="test", search_type="invalid")
        assert "Invalid search_type" in str(exc_info.value)

    def test_search_result_optional_fields(self):
        """Test SearchResult with optional fields."""
        result = SearchResult(
            doc_id="doc123",
            chunk_id="chunk456",
            score=0.95,
            path="/data/test.txt"
        )
        assert result.content is None
        assert result.metadata == {}  # default factory
        assert result.file_name is None


class TestJobContracts:
    """Test job-related contracts."""

    def test_create_job_request_defaults(self):
        """Test CreateJobRequest with default values."""
        req = CreateJobRequest(
            name="Test Job",
            directory_path="/data/test"
        )
        assert req.name == "Test Job"
        assert req.directory_path == "/data/test"
        assert req.chunk_size == 600
        assert req.chunk_overlap == 200
        assert req.model_name == "Qwen/Qwen3-Embedding-0.6B"

    def test_chunk_size_validation(self):
        """Test chunk_size validation."""
        # Too small (Field validation happens before custom validator)
        with pytest.raises(ValidationError) as exc_info:
            CreateJobRequest(name="Test", directory_path="/test", chunk_size=50)
        assert "greater than or equal to 100" in str(exc_info.value)

        # Too large (Field validation happens before custom validator)
        with pytest.raises(ValidationError) as exc_info:
            CreateJobRequest(name="Test", directory_path="/test", chunk_size=60000)
        assert "less than or equal to 50000" in str(exc_info.value)

    def test_chunk_overlap_validation(self):
        """Test chunk_overlap validation."""
        # Negative overlap (Field validation happens before custom validator)
        with pytest.raises(ValidationError) as exc_info:
            CreateJobRequest(name="Test", directory_path="/test", chunk_overlap=-10)
        assert "greater than or equal to 0" in str(exc_info.value)

        # Overlap >= chunk_size (custom validator)
        with pytest.raises(ValidationError) as exc_info:
            CreateJobRequest(name="Test", directory_path="/test", chunk_size=500, chunk_overlap=500)
        assert "must be less than chunk_size" in str(exc_info.value)

    def test_directory_path_security_validation(self):
        """Test directory path security validation."""
        # Path traversal attempt with ..
        with pytest.raises(ValidationError) as exc_info:
            CreateJobRequest(name="Test", directory_path="/data/../etc/passwd")
        assert "Path traversal not allowed" in str(exc_info.value)

        # Path starting with ~
        with pytest.raises(ValidationError) as exc_info:
            CreateJobRequest(name="Test", directory_path="~/sensitive")
        assert "Path traversal not allowed" in str(exc_info.value)

        # Empty path
        with pytest.raises(ValidationError) as exc_info:
            CreateJobRequest(name="Test", directory_path="   ")
        assert "Directory path cannot be empty" in str(exc_info.value)

        # Relative path (should fail)
        with pytest.raises(ValidationError) as exc_info:
            CreateJobRequest(name="Test", directory_path="relative/path")
        assert "Only absolute paths are allowed" in str(exc_info.value)

    def test_quantization_normalization(self):
        """Test quantization field normalization."""
        # fp32 should be normalized to float32
        req = CreateJobRequest(name="Test", directory_path="/test", quantization="fp32")
        assert req.quantization == "float32"

        # fp16 should be normalized to float16
        req = CreateJobRequest(name="Test", directory_path="/test", quantization="fp16")
        assert req.quantization == "float16"

        # Invalid quantization
        with pytest.raises(ValidationError) as exc_info:
            CreateJobRequest(name="Test", directory_path="/test", quantization="invalid")
        assert "Invalid quantization" in str(exc_info.value)

    def test_job_response_aliasing(self):
        """Test JobResponse field aliasing."""
        from datetime import datetime

        # Test with 'id' field
        resp = JobResponse(
            id="job123",
            name="Test Job",
            status="running",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            directory_path="/test",
            model_name="test-model"
        )
        assert resp.id == "job123"
        
        # Test to_dict includes both id and job_id
        resp_dict = resp.to_dict()
        assert resp_dict["id"] == "job123"
        assert resp_dict["job_id"] == "job123"

        # Test with 'job_id' alias
        resp2 = JobResponse(
            job_id="job456",
            name="Test Job 2",
            status="completed",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            directory_path="/test2",
            model_name="test-model2"
        )
        assert resp2.id == "job456"


class TestErrorContracts:
    """Test error-related contracts."""

    def test_basic_error_response(self):
        """Test basic ErrorResponse."""
        error = ErrorResponse(
            error="TestError",
            message="Something went wrong",
            status_code=500
        )
        assert error.error == "TestError"
        assert error.message == "Something went wrong"
        assert error.status_code == 500
        assert error.details is None

    def test_create_validation_error(self):
        """Test create_validation_error helper."""
        errors = [
            ("field1", "Field 1 is required"),
            ("field2", "Field 2 must be positive")
        ]
        error = create_validation_error(errors)
        
        assert error.error == "ValidationError"
        assert error.message == "Validation failed"
        assert error.status_code == 400
        assert len(error.details) == 2
        assert error.details[0].field == "field1"
        assert error.details[0].message == "Field 1 is required"

    def test_create_not_found_error(self):
        """Test create_not_found_error helper."""
        error = create_not_found_error("Job", "job123")
        
        assert error.error == "NotFoundError"
        assert error.message == "Job not found"
        assert error.resource_type == "Job"
        assert error.resource_id == "job123"
        assert error.status_code == 404

    def test_create_insufficient_memory_error(self):
        """Test create_insufficient_memory_error helper."""
        error = create_insufficient_memory_error(
            required="4GB",
            available="2GB",
            suggestion="Try using a smaller model"
        )
        
        assert error.error == "InsufficientResourcesError"
        assert error.message == "Insufficient GPU memory for operation"
        assert error.resource_type == "gpu_memory"
        assert error.required == "4GB"
        assert error.available == "2GB"
        assert error.suggestion == "Try using a smaller model"
        assert error.status_code == 507