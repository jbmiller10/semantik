#!/usr/bin/env python3

"""Tests for GetOperationStatusUseCase."""

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from packages.shared.chunking.application.dto.requests import GetOperationStatusRequest
from packages.shared.chunking.application.dto.responses import (
    GetOperationStatusResponse,
)
from packages.shared.chunking.application.dto.responses import (
    OperationStatus as DTOOperationStatus,
)
from packages.shared.chunking.application.use_cases.get_operation_status import GetOperationStatusUseCase
from packages.shared.chunking.domain.entities.chunking_operation import ChunkingOperation
from packages.shared.chunking.domain.value_objects.chunk_config import ChunkConfig
from packages.shared.chunking.domain.value_objects.operation_status import OperationStatus


class TestGetOperationStatusUseCase:
    """Test suite for GetOperationStatusUseCase."""

    @pytest.fixture()
    def mock_repository(self) -> None:
        """Create mock chunking operation repository."""
        repo = AsyncMock()
        repo.find_by_id = AsyncMock()  # Changed from get_by_id to match implementation
        repo.find_by_document = AsyncMock()
        return repo

    @pytest.fixture()
    def mock_chunk_repository(self) -> None:
        """Create mock chunk repository."""
        repo = AsyncMock()
        repo.find_by_operation = AsyncMock(return_value=[])  # Changed to match implementation
        return repo

    @pytest.fixture()
    def mock_metrics_service(self) -> None:
        """Create mock metrics service."""
        service = AsyncMock()
        service.get_operation_metrics = AsyncMock(return_value=None)
        return service

    @pytest.fixture()
    def use_case(self, mock_repository, mock_chunk_repository, mock_metrics_service) -> None:
        """Create use case instance with mocked dependencies."""
        return GetOperationStatusUseCase(
            operation_repository=mock_repository,
            chunk_repository=mock_chunk_repository,
            metrics_service=mock_metrics_service,
        )

    @pytest.fixture()
    def sample_operation(self) -> None:
        """Create a sample chunking operation."""
        config = ChunkConfig(strategy_name="character", min_tokens=10, max_tokens=100, overlap_tokens=5)

        operation = ChunkingOperation(
            operation_id=str(uuid4()), document_id="doc-123", document_content="Sample document content", config=config
        )

        # Set operation to processing state
        operation.start()
        operation._progress_percentage = 45.0

        return operation

    @pytest.fixture()
    def valid_request(self) -> None:
        """Create a valid status request."""
        return GetOperationStatusRequest(operation_id=str(uuid4()))

    @pytest.mark.asyncio()
    async def test_get_status_success(self, use_case, valid_request, sample_operation) -> None:
        """Test successful status retrieval."""
        # Arrange
        # Set attributes directly on the domain entity
        sample_operation.total_chunks = 10
        sample_operation.chunks_processed = 5  # Should give 50% progress
        use_case.operation_repository.find_by_id.return_value = sample_operation

        # Act
        response = await use_case.execute(valid_request)

        # Assert
        assert isinstance(response, GetOperationStatusResponse)
        assert response.operation_id == sample_operation.id
        assert response.status == DTOOperationStatus.IN_PROGRESS  # Domain PROCESSING maps to DTO IN_PROGRESS
        assert response.progress_percentage == 50.0  # 5/10 * 100

        # Verify repository was called
        use_case.operation_repository.find_by_id.assert_called_once_with(valid_request.operation_id)

    @pytest.mark.asyncio()
    async def test_get_status_from_repository(self, use_case, valid_request) -> None:
        """Test status retrieval directly from repository."""
        # Arrange
        # Create a mock operation to return from repository
        operation = MagicMock()
        operation.id = valid_request.operation_id
        operation.status = "in_progress"
        operation.total_chunks = 10
        operation.chunks_processed = 6
        operation.created_at = datetime.now(tz=UTC)
        operation.updated_at = datetime.now(tz=UTC)
        operation.error_message = None

        use_case.operation_repository.find_by_id.return_value = operation

        # Act
        response = await use_case.execute(valid_request)

        # Assert
        assert response.progress_percentage == 60.0  # 6/10 * 100
        assert response.chunks_processed == 6

        # Repository should be called
        use_case.operation_repository.find_by_id.assert_called_once()

    @pytest.mark.asyncio()
    async def test_get_status_operation_not_found(self, use_case, valid_request) -> None:
        """Test handling of operation not found."""
        # Arrange
        use_case.operation_repository.find_by_id.return_value = None

        # Act & Assert
        with pytest.raises(ValueError, match="Operation not found"):
            await use_case.execute(valid_request)

    @pytest.mark.asyncio()
    async def test_get_status_completed_operation(self, use_case, valid_request) -> None:
        """Test status for completed operation."""
        # Arrange
        completed_operation = MagicMock()
        completed_operation.id = valid_request.operation_id
        completed_operation.status = "completed"  # Use lowercase
        completed_operation.total_chunks = 10
        completed_operation.chunks_processed = 10
        completed_operation.created_at = datetime.now(tz=UTC)
        completed_operation.updated_at = datetime.now(tz=UTC)
        completed_operation.completed_at = datetime.now(tz=UTC)
        completed_operation.error_message = None

        use_case.operation_repository.find_by_id.return_value = completed_operation

        # Act
        response = await use_case.execute(valid_request)

        # Assert
        assert response.operation_id == valid_request.operation_id
        assert response.status == DTOOperationStatus.COMPLETED
        assert response.progress_percentage == 100.0
        assert response.chunks_processed == 10
        assert response.total_chunks == 10

    @pytest.mark.asyncio()
    async def test_get_status_failed_operation(self, use_case, valid_request) -> None:
        """Test status for failed operation."""
        # Arrange
        failed_operation = MagicMock()
        failed_operation.id = valid_request.operation_id
        failed_operation.status = "failed"  # Use lowercase
        failed_operation.total_chunks = 10
        failed_operation.chunks_processed = 2
        failed_operation.error_message = "Strategy execution failed"
        failed_operation.error_type = "RuntimeError"
        failed_operation.failed_at = datetime.now(tz=UTC)
        failed_operation.last_checkpoint = None
        failed_operation.created_at = datetime.now(tz=UTC)
        failed_operation.updated_at = datetime.now(tz=UTC)

        use_case.operation_repository.find_by_id.return_value = failed_operation

        # Act
        response = await use_case.execute(valid_request)

        # Assert
        assert response.operation_id == valid_request.operation_id
        assert response.status == DTOOperationStatus.FAILED
        assert response.error_message == "Strategy execution failed"
        assert response.error_details is not None
        assert response.error_details["error_type"] == "RuntimeError"

    @pytest.mark.asyncio()
    async def test_get_status_cancelled_operation(self, use_case, valid_request) -> None:
        """Test status for cancelled operation."""
        # Arrange
        cancelled_operation = MagicMock()
        cancelled_operation.id = valid_request.operation_id
        cancelled_operation.status = "cancelled"  # Use lowercase
        cancelled_operation.total_chunks = 10
        cancelled_operation.chunks_processed = 7
        cancelled_operation.error_message = "Cancelled by user"
        cancelled_operation.created_at = datetime.now(tz=UTC)
        cancelled_operation.updated_at = datetime.now(tz=UTC)

        use_case.operation_repository.find_by_id.return_value = cancelled_operation

        # Act
        response = await use_case.execute(valid_request)

        # Assert
        assert response.operation_id == valid_request.operation_id
        assert response.status == DTOOperationStatus.CANCELLED
        assert response.error_message == "Cancelled by user"
        assert response.progress_percentage == 70.0  # 7/10 * 100

    @pytest.mark.asyncio()
    async def test_get_status_with_timing_info(self, use_case, valid_request) -> None:
        """Test that timing information is included in status."""
        # Arrange
        operation = MagicMock()
        operation.id = valid_request.operation_id
        operation.status = "in_progress"  # Use lowercase
        operation.total_chunks = 10
        operation.chunks_processed = 5
        operation.created_at = datetime.now(tz=UTC) - timedelta(seconds=10)
        operation.updated_at = datetime.now(tz=UTC)
        operation.completed_at = None
        operation.error_message = None

        use_case.operation_repository.find_by_id.return_value = operation

        # Act
        response = await use_case.execute(valid_request)

        # Assert
        assert response.started_at is not None
        assert response.completed_at is None

    @pytest.mark.asyncio()
    async def test_get_status_by_document_id(self, use_case) -> None:
        """Test getting status by document ID instead of operation ID."""
        # Arrange
        request = GetOperationStatusRequest(document_id="doc-123")

        # Create a mock operation since we can't modify ChunkingOperation.status
        operation = MagicMock()
        operation.id = "op-123"
        operation.status = OperationStatus.PROCESSING  # Use domain enum
        operation._created_at = datetime.now(tz=UTC)
        operation.created_at = operation._created_at
        operation.updated_at = datetime.now(tz=UTC)
        operation.total_chunks = 10
        operation.chunks_processed = 5
        operation.error_message = None

        use_case.operation_repository.find_by_document.return_value = [operation]

        # Act
        response = await use_case.execute(request)

        # Assert
        assert response.operation_id == operation.id
        assert response.status == DTOOperationStatus.IN_PROGRESS  # PROCESSING maps to IN_PROGRESS
        use_case.operation_repository.find_by_document.assert_called_once_with("doc-123")

    @pytest.mark.asyncio()
    async def test_get_latest_operation_for_document(self, use_case) -> None:
        """Test getting latest operation when multiple exist for a document."""
        # Arrange
        request = GetOperationStatusRequest(document_id="doc-multi")

        # Create multiple operations
        old_operation = MagicMock()
        old_operation.id = "old-op"
        old_operation._created_at = datetime.now(tz=UTC) - timedelta(hours=2)
        old_operation.status = "completed"
        old_operation.total_chunks = 5
        old_operation.chunks_processed = 5
        old_operation.created_at = old_operation._created_at
        old_operation.updated_at = datetime.now(tz=UTC) - timedelta(hours=2)
        old_operation.error_message = None

        new_operation = MagicMock()
        new_operation.id = "new-op"
        new_operation._created_at = datetime.now(tz=UTC) - timedelta(minutes=5)
        new_operation.status = "in_progress"
        new_operation.total_chunks = 10
        new_operation.chunks_processed = 3
        new_operation.created_at = new_operation._created_at
        new_operation.updated_at = datetime.now(tz=UTC)
        new_operation.error_message = None

        use_case.operation_repository.find_by_document.return_value = [old_operation, new_operation]

        # Act
        response = await use_case.execute(request)

        # Assert
        assert response.operation_id == "new-op"  # Should return latest
        assert response.status == DTOOperationStatus.IN_PROGRESS

    @pytest.mark.asyncio()
    async def test_status_retrieval_from_repository(self, use_case, valid_request, sample_operation) -> None:
        """Test that status is correctly retrieved from repository."""
        # Arrange
        # Set attributes for progress calculation
        sample_operation.total_chunks = 10
        sample_operation.chunks_processed = 5  # 50% progress
        use_case.operation_repository.find_by_id.return_value = sample_operation

        # Act
        response = await use_case.execute(valid_request)

        # Assert
        use_case.operation_repository.find_by_id.assert_called_once_with(valid_request.operation_id)
        assert response.operation_id == sample_operation.id
        assert response.status == DTOOperationStatus.IN_PROGRESS  # PROCESSING maps to IN_PROGRESS
        assert response.progress_percentage == 50.0  # 5/10 * 100

    @pytest.mark.asyncio()
    async def test_terminal_state_retrieval(self, use_case, valid_request) -> None:
        """Test that terminal states are correctly retrieved."""
        # Arrange
        completed_operation = MagicMock()
        completed_operation.id = valid_request.operation_id
        completed_operation.document_id = "doc-complete"
        completed_operation.status = OperationStatus.COMPLETED
        completed_operation.total_chunks = 10
        completed_operation.chunks_processed = 10
        completed_operation.created_at = datetime.now(tz=UTC)
        completed_operation.updated_at = datetime.now(tz=UTC)
        completed_operation.error_message = None

        use_case.operation_repository.find_by_id.return_value = completed_operation

        # Act
        response = await use_case.execute(valid_request)

        # Assert
        assert response.status == DTOOperationStatus.COMPLETED
        assert response.progress_percentage == 100.0
        use_case.operation_repository.find_by_id.assert_called_once_with(valid_request.operation_id)

    @pytest.mark.asyncio()
    async def test_concurrent_status_requests(self, use_case, sample_operation) -> None:
        """Test handling of concurrent status requests."""
        # Arrange
        requests = [GetOperationStatusRequest(operation_id=sample_operation.id) for _ in range(5)]
        use_case.operation_repository.find_by_id.return_value = sample_operation

        # Act

        responses = await asyncio.gather(*[use_case.execute(req) for req in requests])

        # Assert
        assert len(responses) == 5
        for response in responses:
            assert response.operation_id == sample_operation.id
            assert response.status == DTOOperationStatus.IN_PROGRESS  # PROCESSING maps to IN_PROGRESS

    @pytest.mark.asyncio()
    async def test_status_with_completed_operation_details(self, use_case, valid_request) -> None:
        """Test that completed operation returns all expected details."""
        # Arrange
        operation = MagicMock()
        operation.id = valid_request.operation_id
        operation.document_id = "doc-validate"
        operation.status = OperationStatus.COMPLETED
        operation.total_chunks = 8
        operation.chunks_processed = 8
        operation.created_at = datetime.now(tz=UTC)
        operation.updated_at = datetime.now(tz=UTC)
        operation.completed_at = datetime.now(tz=UTC)
        operation.error_message = None

        use_case.operation_repository.find_by_id.return_value = operation

        # Act
        response = await use_case.execute(valid_request)

        # Assert
        assert response.status == DTOOperationStatus.COMPLETED
        assert response.progress_percentage == 100.0
        assert response.chunks_processed == 8
        assert response.total_chunks == 8
        assert response.error_message is None

    @pytest.mark.asyncio()
    async def test_status_with_processing_operation_timing(self, use_case, valid_request) -> None:
        """Test timing information for processing operations."""
        # Arrange
        operation = MagicMock()
        operation.id = valid_request.operation_id
        operation.document_id = "doc-estimate"
        operation.status = OperationStatus.PROCESSING
        operation.total_chunks = 10
        operation.chunks_processed = 4  # 40% progress
        operation.created_at = datetime.now(tz=UTC) - timedelta(seconds=20)
        operation.updated_at = datetime.now(tz=UTC)
        operation.completed_at = None
        operation.error_message = None

        use_case.operation_repository.find_by_id.return_value = operation

        # Act
        response = await use_case.execute(valid_request)

        # Assert
        assert response.status == DTOOperationStatus.IN_PROGRESS
        assert response.progress_percentage == 40.0  # 4/10 * 100
        assert response.started_at is not None
        assert response.updated_at is not None
        assert response.completed_at is None

    @pytest.mark.asyncio()
    async def test_invalid_operation_id_format(self, use_case) -> None:
        """Test handling of operation not found."""
        # Arrange
        request = GetOperationStatusRequest(operation_id="invalid-id-format")
        use_case.operation_repository.find_by_id.return_value = None

        # Act & Assert
        with pytest.raises(ValueError, match="Operation not found"):
            await use_case.execute(request)
