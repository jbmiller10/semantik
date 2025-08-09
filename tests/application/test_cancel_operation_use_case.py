#!/usr/bin/env python3
"""Tests for CancelOperationUseCase."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from packages.shared.chunking.application.dto.requests import CancelRequest
from packages.shared.chunking.application.dto.responses import CancelResponse
from packages.shared.chunking.application.use_cases.cancel_operation import (
    CancelOperationUseCase,
)
from packages.shared.chunking.domain.entities.chunking_operation import ChunkingOperation
from packages.shared.chunking.domain.exceptions import InvalidStateError
from packages.shared.chunking.domain.value_objects.chunk_config import ChunkConfig
from packages.shared.chunking.domain.value_objects.operation_status import OperationStatus


class TestCancelOperationUseCase:
    """Test suite for CancelOperationUseCase."""

    @pytest.fixture
    def mock_repository(self):
        """Create mock chunking operation repository."""
        repo = AsyncMock()
        repo.get_by_id = AsyncMock()
        repo.update = AsyncMock()
        return repo

    @pytest.fixture
    def mock_unit_of_work(self, mock_repository):
        """Create mock unit of work."""
        uow = AsyncMock()
        uow.__aenter__ = AsyncMock(return_value=uow)
        uow.__aexit__ = AsyncMock(return_value=None)
        uow.chunking_operations = mock_repository
        uow.commit = AsyncMock()
        uow.rollback = AsyncMock()
        return uow

    @pytest.fixture
    def mock_notification_service(self):
        """Create mock notification service."""
        service = AsyncMock()
        service.notify_operation_cancelled = AsyncMock()
        return service

    @pytest.fixture
    def mock_event_publisher(self):
        """Create mock event publisher."""
        publisher = AsyncMock()
        publisher.publish_operation_cancelled = AsyncMock()
        return publisher

    @pytest.fixture
    def mock_task_manager(self):
        """Create mock task manager."""
        manager = AsyncMock()
        manager.cancel_task = AsyncMock(return_value=True)
        manager.get_task_status = AsyncMock(return_value="RUNNING")
        return manager

    @pytest.fixture
    def use_case(
        self,
        mock_unit_of_work,
        mock_notification_service,
        mock_event_publisher,
        mock_task_manager,
    ):
        """Create use case instance with mocked dependencies."""
        return CancelOperationUseCase(
            unit_of_work=mock_unit_of_work,
            notification_service=mock_notification_service,
            event_publisher=mock_event_publisher,
            task_manager=mock_task_manager,
        )

    @pytest.fixture
    def processing_operation(self):
        """Create an operation in PROCESSING state."""
        config = ChunkConfig(
            strategy_name="character",
            min_tokens=10,
            max_tokens=100,
            overlap_tokens=5,
        )
        
        operation = ChunkingOperation(
            operation_id=str(uuid4()),
            document_id="doc-123",
            document_content="Sample document content for cancellation",
            config=config,
        )
        
        # Set to processing state
        operation.start()
        operation._progress_percentage = 35.0
        
        return operation

    @pytest.fixture
    def pending_operation(self):
        """Create an operation in PENDING state."""
        config = ChunkConfig(
            strategy_name="semantic",
            min_tokens=20,
            max_tokens=200,
            overlap_tokens=10,
        )
        
        operation = ChunkingOperation(
            operation_id=str(uuid4()),
            document_id="doc-pending",
            document_content="Document waiting to be processed",
            config=config,
        )
        
        return operation

    @pytest.fixture
    def valid_request(self, processing_operation):
        """Create a valid cancel request."""
        return CancelRequest(
            operation_id=processing_operation.id,
            reason="User requested cancellation",
        )

    @pytest.mark.asyncio
    async def test_successful_cancellation_processing_operation(
        self, use_case, valid_request, processing_operation
    ):
        """Test successful cancellation of a processing operation."""
        # Arrange
        use_case.unit_of_work.chunking_operations.get_by_id.return_value = (
            processing_operation
        )

        # Act
        response = await use_case.execute(valid_request)

        # Assert
        assert isinstance(response, CancelResponse)
        assert response.operation_id == processing_operation.id
        assert response.success is True
        assert response.status == "CANCELLED"
        assert "successfully cancelled" in response.message.lower()

        # Verify workflow
        use_case.unit_of_work.chunking_operations.get_by_id.assert_called_once_with(
            valid_request.operation_id
        )
        use_case.unit_of_work.chunking_operations.update.assert_called_once()
        use_case.unit_of_work.commit.assert_called_once()
        use_case.task_manager.cancel_task.assert_called_once_with(
            valid_request.operation_id
        )
        use_case.notification_service.notify_operation_cancelled.assert_called_once()
        use_case.event_publisher.publish_operation_cancelled.assert_called_once()

    @pytest.mark.asyncio
    async def test_successful_cancellation_pending_operation(
        self, use_case, pending_operation
    ):
        """Test successful cancellation of a pending operation."""
        # Arrange
        request = CancelRequest(
            operation_id=pending_operation.id,
            reason="Cancelled before processing started",
        )
        use_case.unit_of_work.chunking_operations.get_by_id.return_value = (
            pending_operation
        )

        # Act
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.status == "CANCELLED"
        assert pending_operation.status == OperationStatus.CANCELLED
        assert pending_operation.error_message == "Cancelled before processing started"

    @pytest.mark.asyncio
    async def test_cancellation_without_reason(
        self, use_case, processing_operation
    ):
        """Test cancellation without providing a reason."""
        # Arrange
        request = CancelRequest(operation_id=processing_operation.id)
        use_case.unit_of_work.chunking_operations.get_by_id.return_value = (
            processing_operation
        )

        # Act
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.status == "CANCELLED"
        # Default reason should be applied
        assert processing_operation.error_message == "Operation cancelled by user"

    @pytest.mark.asyncio
    async def test_cancel_operation_not_found(self, use_case, valid_request):
        """Test cancellation of non-existent operation."""
        # Arrange
        use_case.unit_of_work.chunking_operations.get_by_id.return_value = None

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            await use_case.execute(valid_request)

        assert "Operation not found" in str(exc_info.value)
        use_case.unit_of_work.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_completed_operation_fails(self, use_case):
        """Test that cancelling a completed operation fails."""
        # Arrange
        completed_operation = MagicMock()
        completed_operation.id = str(uuid4())
        completed_operation.status = OperationStatus.COMPLETED
        completed_operation.cancel.side_effect = InvalidStateError(
            "Cannot cancel operation in COMPLETED state"
        )

        request = CancelRequest(
            operation_id=completed_operation.id,
            reason="Trying to cancel completed",
        )
        use_case.unit_of_work.chunking_operations.get_by_id.return_value = (
            completed_operation
        )

        # Act
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert response.status == "COMPLETED"
        assert "cannot be cancelled" in response.message.lower()
        use_case.unit_of_work.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_failed_operation_fails(self, use_case):
        """Test that cancelling a failed operation fails."""
        # Arrange
        failed_operation = MagicMock()
        failed_operation.id = str(uuid4())
        failed_operation.status = OperationStatus.FAILED
        failed_operation.cancel.side_effect = InvalidStateError(
            "Cannot cancel operation in FAILED state"
        )

        request = CancelRequest(
            operation_id=failed_operation.id,
            reason="Trying to cancel failed",
        )
        use_case.unit_of_work.chunking_operations.get_by_id.return_value = (
            failed_operation
        )

        # Act
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert response.status == "FAILED"
        assert "cannot be cancelled" in response.message.lower()

    @pytest.mark.asyncio
    async def test_cancel_already_cancelled_operation(self, use_case):
        """Test cancelling an already cancelled operation."""
        # Arrange
        cancelled_operation = MagicMock()
        cancelled_operation.id = str(uuid4())
        cancelled_operation.status = OperationStatus.CANCELLED
        cancelled_operation.cancel.side_effect = InvalidStateError(
            "Cannot cancel operation in CANCELLED state"
        )

        request = CancelRequest(
            operation_id=cancelled_operation.id,
            reason="Trying to cancel again",
        )
        use_case.unit_of_work.chunking_operations.get_by_id.return_value = (
            cancelled_operation
        )

        # Act
        response = await use_case.execute(request)

        # Assert
        assert response.success is False
        assert response.status == "CANCELLED"
        assert "already cancelled" in response.message.lower()

    @pytest.mark.asyncio
    async def test_task_manager_cancellation_failure(
        self, use_case, valid_request, processing_operation
    ):
        """Test handling when task manager fails to cancel task."""
        # Arrange
        use_case.unit_of_work.chunking_operations.get_by_id.return_value = (
            processing_operation
        )
        use_case.task_manager.cancel_task.return_value = False  # Failed to cancel

        # Act
        response = await use_case.execute(valid_request)

        # Assert
        assert response.success is True  # Domain cancellation still succeeds
        assert response.status == "CANCELLED"
        # Warning should be included about task cancellation failure
        assert "warning" in response.message.lower() or response.warnings

    @pytest.mark.asyncio
    async def test_notification_failure_does_not_affect_cancellation(
        self, use_case, valid_request, processing_operation
    ):
        """Test that notification failure doesn't prevent cancellation."""
        # Arrange
        use_case.unit_of_work.chunking_operations.get_by_id.return_value = (
            processing_operation
        )
        use_case.notification_service.notify_operation_cancelled.side_effect = Exception(
            "Notification failed"
        )

        # Act
        response = await use_case.execute(valid_request)

        # Assert
        assert response.success is True
        assert response.status == "CANCELLED"
        # Cancellation should succeed despite notification failure
        use_case.unit_of_work.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_event_publishing_failure_does_not_affect_cancellation(
        self, use_case, valid_request, processing_operation
    ):
        """Test that event publishing failure doesn't prevent cancellation."""
        # Arrange
        use_case.unit_of_work.chunking_operations.get_by_id.return_value = (
            processing_operation
        )
        use_case.event_publisher.publish_operation_cancelled.side_effect = Exception(
            "Event publishing failed"
        )

        # Act
        response = await use_case.execute(valid_request)

        # Assert
        assert response.success is True
        assert response.status == "CANCELLED"
        use_case.unit_of_work.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_transaction_rollback_on_error(
        self, use_case, valid_request, processing_operation
    ):
        """Test that transaction is rolled back on error."""
        # Arrange
        use_case.unit_of_work.chunking_operations.get_by_id.return_value = (
            processing_operation
        )
        use_case.unit_of_work.commit.side_effect = Exception("Database error")

        # Act & Assert
        with pytest.raises(Exception) as exc_info:
            await use_case.execute(valid_request)

        assert "Database error" in str(exc_info.value)
        use_case.unit_of_work.rollback.assert_called()

    @pytest.mark.asyncio
    async def test_force_cancellation(self, use_case):
        """Test force cancellation bypasses state checks."""
        # Arrange
        completed_operation = MagicMock()
        completed_operation.id = str(uuid4())
        completed_operation.status = OperationStatus.COMPLETED

        request = CancelRequest(
            operation_id=completed_operation.id,
            reason="Force cancel",
            force=True,  # Force cancellation
        )
        use_case.unit_of_work.chunking_operations.get_by_id.return_value = (
            completed_operation
        )

        # Act
        response = await use_case.execute(request)

        # Assert
        assert response.success is True
        assert response.status == "CANCELLED"
        assert "forced" in response.message.lower()

    @pytest.mark.asyncio
    async def test_concurrent_cancellation_requests(
        self, use_case, processing_operation
    ):
        """Test handling of concurrent cancellation requests."""
        # Arrange
        requests = [
            CancelRequest(
                operation_id=processing_operation.id,
                reason=f"Concurrent cancel {i}",
            )
            for i in range(3)
        ]
        use_case.unit_of_work.chunking_operations.get_by_id.return_value = (
            processing_operation
        )

        # Simulate first request succeeds, others see already cancelled
        call_count = 0

        def cancel_side_effect(reason=None):
            nonlocal call_count
            if call_count == 0:
                processing_operation._status = OperationStatus.CANCELLED
                processing_operation._error_message = reason or "Operation cancelled by user"
                call_count += 1
            else:
                raise InvalidStateError("Cannot cancel operation in CANCELLED state")

        processing_operation.cancel = MagicMock(side_effect=cancel_side_effect)

        # Act
        import asyncio
        responses = await asyncio.gather(
            *[use_case.execute(req) for req in requests],
            return_exceptions=True
        )

        # Assert
        successful_responses = [r for r in responses if isinstance(r, CancelResponse) and r.success]
        assert len(successful_responses) >= 1  # At least one should succeed

    @pytest.mark.asyncio
    async def test_cancel_with_cleanup_operations(
        self, use_case, valid_request, processing_operation
    ):
        """Test that cancellation performs cleanup operations."""
        # Arrange
        use_case.unit_of_work.chunking_operations.get_by_id.return_value = (
            processing_operation
        )
        
        # Add cleanup service
        cleanup_service = AsyncMock()
        cleanup_service.cleanup_operation = AsyncMock()
        use_case.cleanup_service = cleanup_service

        # Act
        response = await use_case.execute(valid_request)

        # Assert
        assert response.success is True
        # Cleanup should be called
        if hasattr(use_case, 'cleanup_service'):
            use_case.cleanup_service.cleanup_operation.assert_called_once_with(
                valid_request.operation_id
            )

    @pytest.mark.asyncio
    async def test_cancel_returns_partial_results(
        self, use_case, valid_request, processing_operation
    ):
        """Test that cancellation returns partial results if available."""
        # Arrange
        from packages.shared.chunking.domain.entities.chunk import Chunk
        from packages.shared.chunking.domain.value_objects.chunk_metadata import ChunkMetadata
        
        # Add some chunks to the operation
        for i in range(3):
            chunk = Chunk(
                content=f"Partial chunk {i}",
                start_position=i * 10,
                end_position=(i + 1) * 10,
                metadata=ChunkMetadata(token_count=3),
            )
            processing_operation._chunk_collection.add_chunk(chunk)
        
        use_case.unit_of_work.chunking_operations.get_by_id.return_value = (
            processing_operation
        )

        # Act
        response = await use_case.execute(valid_request)

        # Assert
        assert response.success is True
        assert response.partial_results is not None
        assert response.partial_results["chunks_produced"] == 3
        assert response.partial_results["progress_percentage"] == 35.0

    @pytest.mark.asyncio
    async def test_invalid_operation_id_format(self, use_case):
        """Test handling of invalid operation ID format."""
        # Arrange
        request = CancelRequest(
            operation_id="invalid-uuid-format",
            reason="Test",
        )

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            await use_case.execute(request)

        assert "Invalid operation ID" in str(exc_info.value)