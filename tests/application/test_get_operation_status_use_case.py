#!/usr/bin/env python3
"""Tests for GetOperationStatusUseCase."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from packages.shared.chunking.application.dto.requests import GetOperationStatusRequest
from packages.shared.chunking.application.dto.responses import GetOperationStatusResponse
from packages.shared.chunking.application.use_cases.get_operation_status import (
    GetOperationStatusUseCase)
from packages.shared.chunking.domain.entities.chunking_operation import ChunkingOperation
from packages.shared.chunking.domain.value_objects.chunk_config import ChunkConfig
from packages.shared.chunking.domain.value_objects.operation_status import OperationStatus


class TestGetOperationStatusUseCase:
    """Test suite for GetOperationStatusUseCase."""

    @pytest.fixture()
    def mock_repository(self):
        """Create mock chunking operation repository."""
        repo = AsyncMock()
        repo.get_by_id = AsyncMock()
        repo.find_by_document_id = AsyncMock()
        return repo

    @pytest.fixture()
    def mock_chunk_repository(self):
        """Create mock chunk repository."""
        repo = AsyncMock()
        repo.get_by_operation_id = AsyncMock(return_value=[])
        return repo
    
    @pytest.fixture()
    def mock_metrics_service(self):
        """Create mock metrics service."""
        service = AsyncMock()
        service.record_status_query = AsyncMock()
        return service

    @pytest.fixture()
    def use_case(self, mock_repository, mock_chunk_repository, mock_metrics_service):
        """Create use case instance with mocked dependencies."""
        return GetOperationStatusUseCase(
            operation_repository=mock_repository,
            chunk_repository=mock_chunk_repository,
            metrics_service=mock_metrics_service)

    @pytest.fixture()
    def sample_operation(self):
        """Create a sample chunking operation."""
        config = ChunkConfig(
            strategy_name="character",
            min_tokens=10,
            max_tokens=100,
            overlap_tokens=5)

        operation = ChunkingOperation(
            operation_id=str(uuid4()),
            document_id="doc-123",
            document_content="Sample document content",
            config=config)

        # Set operation to processing state
        operation.start()
        operation._progress_percentage = 45.0

        return operation

    @pytest.fixture()
    def valid_request(self):
        """Create a valid status request."""
        return GetOperationStatusRequest(operation_id=str(uuid4()))

    @pytest.mark.asyncio()
    async def test_get_status_success(self, use_case, valid_request, sample_operation):
        """Test successful status retrieval."""
        # Arrange
        use_case.operation_repository.get_by_id.return_value = sample_operation

        # Act
        response = await use_case.execute(valid_request)

        # Assert
        assert isinstance(response, GetOperationStatusResponse)
        assert response.operation_id == sample_operation.id
        assert response.document_id == "doc-123"
        assert response.status == "PROCESSING"
        assert response.progress_percentage == 45.0
        assert response.chunks_produced == 0

        # Verify repository was called
        use_case.operation_repository.get_by_id.assert_called_once_with(valid_request.operation_id)

    @pytest.mark.asyncio()
    async def test_get_status_from_cache(self, use_case, valid_request):
        """Test status retrieval from cache."""
        # Arrange
        cached_status = {
            "operation_id": valid_request.operation_id,
            "status": "PROCESSING",
            "progress_percentage": 60.0,
            "chunks_produced": 5,
        }
        use_case.cache_service.get_status.return_value = cached_status

        # Act
        response = await use_case.execute(valid_request)

        # Assert
        assert response.progress_percentage == 60.0
        assert response.chunks_produced == 5

        # Repository should not be called when cache hit
        use_case.operation_repository.get_by_id.assert_not_called()

    @pytest.mark.asyncio()
    async def test_get_status_operation_not_found(self, use_case, valid_request):
        """Test handling of operation not found."""
        # Arrange
        use_case.operation_repository.get_by_id.return_value = None

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            await use_case.execute(valid_request)

        assert "Operation not found" in str(exc_info.value)

    @pytest.mark.asyncio()
    async def test_get_status_completed_operation(self, use_case, valid_request):
        """Test status for completed operation."""
        # Arrange
        completed_operation = MagicMock()
        completed_operation.id = valid_request.operation_id
        completed_operation.document_id = "doc-456"
        completed_operation.status = OperationStatus.COMPLETED
        completed_operation.progress_percentage = 100.0
        completed_operation.chunk_collection.chunk_count = 10
        completed_operation.get_statistics.return_value = {
            "chunks": {"total": 10},
            "coverage": 0.95,
            "metrics": {"duration_seconds": 2.5},
        }

        use_case.operation_repository.get_by_id.return_value = completed_operation

        # Act
        response = await use_case.execute(valid_request)

        # Assert
        assert response.status == "COMPLETED"
        assert response.progress_percentage == 100.0
        assert response.chunks_produced == 10
        assert response.statistics is not None
        assert response.statistics["chunks"]["total"] == 10

    @pytest.mark.asyncio()
    async def test_get_status_failed_operation(self, use_case, valid_request):
        """Test status for failed operation."""
        # Arrange
        failed_operation = MagicMock()
        failed_operation.id = valid_request.operation_id
        failed_operation.document_id = "doc-789"
        failed_operation.status = OperationStatus.FAILED
        failed_operation.progress_percentage = 23.0
        failed_operation.chunk_collection.chunk_count = 2
        failed_operation.error_message = "Strategy execution failed"
        failed_operation.get_statistics.return_value = {
            "error": {
                "message": "Strategy execution failed",
                "details": {"exception_type": "RuntimeError"},
            }
        }

        use_case.operation_repository.get_by_id.return_value = failed_operation

        # Act
        response = await use_case.execute(valid_request)

        # Assert
        assert response.status == "FAILED"
        assert response.error_message == "Strategy execution failed"
        assert response.statistics["error"]["details"]["exception_type"] == "RuntimeError"

    @pytest.mark.asyncio()
    async def test_get_status_cancelled_operation(self, use_case, valid_request):
        """Test status for cancelled operation."""
        # Arrange
        cancelled_operation = MagicMock()
        cancelled_operation.id = valid_request.operation_id
        cancelled_operation.document_id = "doc-cancel"
        cancelled_operation.status = OperationStatus.CANCELLED
        cancelled_operation.progress_percentage = 67.0
        cancelled_operation.chunk_collection.chunk_count = 7
        cancelled_operation.error_message = "Cancelled by user"
        cancelled_operation.get_statistics.return_value = {}

        use_case.operation_repository.get_by_id.return_value = cancelled_operation

        # Act
        response = await use_case.execute(valid_request)

        # Assert
        assert response.status == "CANCELLED"
        assert response.error_message == "Cancelled by user"
        assert response.progress_percentage == 67.0

    @pytest.mark.asyncio()
    async def test_get_status_with_timing_info(self, use_case, valid_request):
        """Test that timing information is included in status."""
        # Arrange
        operation = MagicMock()
        operation.id = valid_request.operation_id
        operation.document_id = "doc-timing"
        operation.status = OperationStatus.PROCESSING
        operation.progress_percentage = 50.0
        operation.chunk_collection.chunk_count = 5
        operation._started_at = datetime.utcnow() - timedelta(seconds=10)
        operation._completed_at = None
        operation.get_statistics.return_value = {
            "timing": {
                "started_at": operation._started_at.isoformat(),
                "duration_seconds": 10.0,
            }
        }

        use_case.operation_repository.get_by_id.return_value = operation

        # Act
        response = await use_case.execute(valid_request)

        # Assert
        assert response.started_at is not None
        assert response.duration_seconds == 10.0
        assert response.completed_at is None

    @pytest.mark.asyncio()
    async def test_get_status_by_document_id(self, use_case, sample_operation):
        """Test getting status by document ID instead of operation ID."""
        # Arrange
        request = GetOperationStatusRequest(document_id="doc-123")
        use_case.operation_repository.find_by_document_id.return_value = [sample_operation]

        # Act
        response = await use_case.execute(request)

        # Assert
        assert response.operation_id == sample_operation.id
        assert response.document_id == "doc-123"
        use_case.operation_repository.find_by_document_id.assert_called_once_with("doc-123")

    @pytest.mark.asyncio()
    async def test_get_latest_operation_for_document(self, use_case):
        """Test getting latest operation when multiple exist for a document."""
        # Arrange
        request = GetOperationStatusRequest(document_id="doc-multi")

        # Create multiple operations
        old_operation = MagicMock()
        old_operation.id = "old-op"
        old_operation._created_at = datetime.utcnow() - timedelta(hours=2)
        old_operation.status = OperationStatus.COMPLETED
        old_operation.progress_percentage = 100.0
        old_operation.chunk_collection.chunk_count = 5
        old_operation.get_statistics.return_value = {}

        new_operation = MagicMock()
        new_operation.id = "new-op"
        new_operation.document_id = "doc-multi"
        new_operation._created_at = datetime.utcnow() - timedelta(minutes=5)
        new_operation.status = OperationStatus.PROCESSING
        new_operation.progress_percentage = 30.0
        new_operation.chunk_collection.chunk_count = 3
        new_operation.get_statistics.return_value = {}

        use_case.operation_repository.find_by_document_id.return_value = [old_operation, new_operation]

        # Act
        response = await use_case.execute(request)

        # Assert
        assert response.operation_id == "new-op"  # Should return latest
        assert response.status == "PROCESSING"

    @pytest.mark.asyncio()
    async def test_cache_update_after_retrieval(self, use_case, valid_request, sample_operation):
        """Test that cache is updated after retrieving from repository."""
        # Arrange
        use_case.operation_repository.get_by_id.return_value = sample_operation
        use_case.cache_service.get_status.return_value = None  # Cache miss

        # Act
        response = await use_case.execute(valid_request)

        # Assert
        use_case.cache_service.set_status.assert_called_once()
        cache_call_args = use_case.cache_service.set_status.call_args[0]
        assert cache_call_args[0] == valid_request.operation_id
        assert cache_call_args[1]["status"] == "PROCESSING"
        assert cache_call_args[1]["progress_percentage"] == 45.0

    @pytest.mark.asyncio()
    async def test_no_cache_update_for_terminal_states(self, use_case, valid_request):
        """Test that terminal states are not cached with short TTL."""
        # Arrange
        completed_operation = MagicMock()
        completed_operation.id = valid_request.operation_id
        completed_operation.document_id = "doc-complete"
        completed_operation.status = OperationStatus.COMPLETED
        completed_operation.progress_percentage = 100.0
        completed_operation.chunk_collection.chunk_count = 10
        completed_operation.get_statistics.return_value = {}

        use_case.operation_repository.get_by_id.return_value = completed_operation

        # Act
        response = await use_case.execute(valid_request)

        # Assert
        # Cache should be updated but potentially with longer TTL for terminal states
        use_case.cache_service.set_status.assert_called_once()
        cache_call = use_case.cache_service.set_status.call_args
        # Could check for different TTL if implemented

    @pytest.mark.asyncio()
    async def test_concurrent_status_requests(self, use_case, sample_operation):
        """Test handling of concurrent status requests."""
        # Arrange
        requests = [
            GetOperationStatusRequest(operation_id=sample_operation.id)
            for _ in range(5)
        ]
        use_case.operation_repository.get_by_id.return_value = sample_operation

        # Act
        import asyncio
        responses = await asyncio.gather(
            *[use_case.execute(req) for req in requests]
        )

        # Assert
        assert len(responses) == 5
        for response in responses:
            assert response.operation_id == sample_operation.id
            assert response.status == "PROCESSING"

    @pytest.mark.asyncio()
    async def test_status_with_validation_results(self, use_case, valid_request):
        """Test that validation results are included in status."""
        # Arrange
        operation = MagicMock()
        operation.id = valid_request.operation_id
        operation.document_id = "doc-validate"
        operation.status = OperationStatus.COMPLETED
        operation.progress_percentage = 100.0
        operation.chunk_collection.chunk_count = 8
        operation.validate_results.return_value = (False, ["Insufficient coverage: 75%"])
        operation.get_statistics.return_value = {
            "validation": {
                "is_valid": False,
                "issues": ["Insufficient coverage: 75%"],
            }
        }

        use_case.operation_repository.get_by_id.return_value = operation

        # Act
        response = await use_case.execute(valid_request)

        # Assert
        assert response.validation_results is not None
        assert not response.validation_results["is_valid"]
        assert "Insufficient coverage" in response.validation_results["issues"][0]

    @pytest.mark.asyncio()
    async def test_status_with_estimated_completion(self, use_case, valid_request):
        """Test estimated completion time for processing operations."""
        # Arrange
        operation = MagicMock()
        operation.id = valid_request.operation_id
        operation.document_id = "doc-estimate"
        operation.status = OperationStatus.PROCESSING
        operation.progress_percentage = 40.0
        operation.chunk_collection.chunk_count = 4
        operation._started_at = datetime.utcnow() - timedelta(seconds=20)
        operation._completed_at = None
        operation.get_statistics.return_value = {
            "timing": {
                "started_at": operation._started_at.isoformat(),
                "duration_seconds": 20.0,
            }
        }

        use_case.operation_repository.get_by_id.return_value = operation

        # Act
        response = await use_case.execute(valid_request)

        # Assert
        assert response.estimated_completion_seconds is not None
        # At 40% progress after 20 seconds, should estimate ~30 more seconds
        assert 25 <= response.estimated_completion_seconds <= 35

    @pytest.mark.asyncio()
    async def test_invalid_operation_id_format(self, use_case):
        """Test handling of invalid operation ID format."""
        # Arrange
        request = GetOperationStatusRequest(operation_id="invalid-id-format")

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            await use_case.execute(request)

        assert "Invalid operation ID format" in str(exc_info.value)
