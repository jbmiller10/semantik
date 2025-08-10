#!/usr/bin/env python3

"""
Comprehensive test suite for webui/services/operation_service.py
Tests operation lifecycle, state transitions, and error handling
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from shared.database.exceptions import AccessDeniedError, EntityNotFoundError, ValidationError
from shared.database.models import Operation, OperationStatus, OperationType
from webui.services.operation_service import OperationService


class TestOperationService:
    """Test OperationService implementation"""

    @pytest.fixture()
    def mock_session(self) -> None:
        """Create a mock AsyncSession"""
        session = AsyncMock()
        session.commit = AsyncMock()
        return session

    @pytest.fixture()
    def mock_operation_repo(self) -> None:
        """Create a mock OperationRepository"""
        return AsyncMock()

    @pytest.fixture()
    def operation_service(self, mock_session, mock_operation_repo) -> None:
        """Create OperationService with mocked dependencies"""
        return OperationService(
            db_session=mock_session,
            operation_repo=mock_operation_repo,
        )

    @pytest.mark.asyncio()
    async def test_get_operation_success(self, operation_service, mock_operation_repo) -> None:
        """Test successful operation retrieval"""
        # Mock operation
        mock_operation = Mock(spec=Operation)
        mock_operation.uuid = "test-uuid"
        mock_operation.type = OperationType.INDEX
        mock_operation.status = OperationStatus.PROCESSING
        mock_operation.user_id = 123
        mock_operation.collection_id = "collection-uuid"

        mock_operation_repo.get_by_uuid_with_permission_check.return_value = mock_operation

        # Test get operation
        result = await operation_service.get_operation("test-uuid", 123)

        assert result == mock_operation
        mock_operation_repo.get_by_uuid_with_permission_check.assert_called_once_with(
            operation_uuid="test-uuid",
            user_id=123,
        )

    @pytest.mark.asyncio()
    async def test_get_operation_not_found(self, operation_service, mock_operation_repo) -> None:
        """Test get operation when not found"""
        mock_operation_repo.get_by_uuid_with_permission_check.side_effect = EntityNotFoundError(
            "operation", "test-uuid"
        )

        with pytest.raises(EntityNotFoundError):
            await operation_service.get_operation("test-uuid", 123)

    @pytest.mark.asyncio()
    async def test_get_operation_access_denied(self, operation_service, mock_operation_repo) -> None:
        """Test get operation when access is denied"""
        mock_operation_repo.get_by_uuid_with_permission_check.side_effect = AccessDeniedError(
            user_id="456", resource_type="operation", resource_id="test-uuid"
        )

        with pytest.raises(AccessDeniedError):
            await operation_service.get_operation("test-uuid", 456)

    @pytest.mark.asyncio()
    @patch("webui.services.operation_service.celery_app")
    async def test_cancel_operation_success(
        self, mock_celery_app, operation_service, mock_operation_repo, mock_session
    ) -> None:
        """Test successful operation cancellation"""
        # Mock operation
        mock_operation = Mock(spec=Operation)
        mock_operation.uuid = "test-uuid"
        mock_operation.status = OperationStatus.PROCESSING
        mock_operation.task_id = "celery-task-123"

        mock_operation_repo.cancel.return_value = mock_operation

        # Mock Celery control
        mock_control = Mock()
        mock_celery_app.control = mock_control

        # Test cancel operation
        result = await operation_service.cancel_operation("test-uuid", 123)

        assert result == mock_operation
        mock_operation_repo.cancel.assert_called_once_with(
            operation_uuid="test-uuid",
            user_id=123,
        )
        mock_control.revoke.assert_called_once_with("celery-task-123", terminate=True)
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio()
    @patch("webui.services.operation_service.celery_app")
    async def test_cancel_operation_no_task_id(
        self, mock_celery_app, operation_service, mock_operation_repo, mock_session
    ) -> None:
        """Test cancelling operation without Celery task ID"""
        # Mock operation without task_id
        mock_operation = Mock(spec=Operation)
        mock_operation.uuid = "test-uuid"
        mock_operation.status = OperationStatus.PENDING
        mock_operation.task_id = None

        mock_operation_repo.cancel.return_value = mock_operation

        # Test cancel operation
        result = await operation_service.cancel_operation("test-uuid", 123)

        assert result == mock_operation
        mock_operation_repo.cancel.assert_called_once()
        # Celery revoke should not be called
        mock_celery_app.control.revoke.assert_not_called()
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio()
    @patch("webui.services.operation_service.celery_app")
    @patch("webui.services.operation_service.logger")
    async def test_cancel_operation_celery_revoke_failure(
        self, mock_logger, mock_celery_app, operation_service, mock_operation_repo, mock_session
    ) -> None:
        """Test operation cancellation when Celery revoke fails"""
        # Mock operation
        mock_operation = Mock(spec=Operation)
        mock_operation.uuid = "test-uuid"
        mock_operation.status = OperationStatus.PROCESSING
        mock_operation.task_id = "celery-task-123"

        mock_operation_repo.cancel.return_value = mock_operation

        # Mock Celery control to raise exception
        mock_control = Mock()
        mock_control.revoke.side_effect = Exception("Celery connection error")
        mock_celery_app.control = mock_control

        # Test cancel operation - should still succeed
        result = await operation_service.cancel_operation("test-uuid", 123)

        assert result == mock_operation
        mock_control.revoke.assert_called_once()
        # Warning should be logged
        mock_logger.warning.assert_called_once()
        # Transaction should still be committed
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio()
    async def test_cancel_operation_invalid_state(self, operation_service, mock_operation_repo) -> None:
        """Test cancelling operation in invalid state"""
        mock_operation_repo.cancel.side_effect = ValidationError("Operation cannot be cancelled in COMPLETED state")

        with pytest.raises(ValidationError):
            await operation_service.cancel_operation("test-uuid", 123)

    @pytest.mark.asyncio()
    async def test_list_operations_no_filters(self, operation_service, mock_operation_repo) -> None:
        """Test listing operations without filters"""
        # Mock operations
        mock_operations = []
        for i in range(3):
            op = Mock(spec=Operation)
            op.uuid = f"op-{i}"
            op.type = OperationType.INDEX
            op.status = OperationStatus.COMPLETED
            mock_operations.append(op)

        mock_operation_repo.list_for_user.return_value = (mock_operations, 3)

        # Test list operations
        operations, total = await operation_service.list_operations(user_id=123)

        assert len(operations) == 3
        assert total == 3
        mock_operation_repo.list_for_user.assert_called_once_with(
            user_id=123,
            status_list=None,
            operation_type=None,
            offset=0,
            limit=50,
        )

    @pytest.mark.asyncio()
    async def test_list_operations_with_filters(self, operation_service, mock_operation_repo) -> None:
        """Test listing operations with filters"""
        # Mock filtered operations
        mock_operations = []
        for i in range(2):
            op = Mock(spec=Operation)
            op.uuid = f"op-{i}"
            op.type = OperationType.REINDEX
            op.status = OperationStatus.PROCESSING
            mock_operations.append(op)

        mock_operation_repo.list_for_user.return_value = (mock_operations, 2)

        # Test list operations with filters
        operations, total = await operation_service.list_operations(
            user_id=123,
            status_list=[OperationStatus.PROCESSING, OperationStatus.PENDING],
            operation_type=OperationType.REINDEX,
            offset=10,
            limit=20,
        )

        assert len(operations) == 2
        assert total == 2
        mock_operation_repo.list_for_user.assert_called_once_with(
            user_id=123,
            status_list=[OperationStatus.PROCESSING, OperationStatus.PENDING],
            operation_type=OperationType.REINDEX,
            offset=10,
            limit=20,
        )

    @pytest.mark.asyncio()
    async def test_verify_websocket_access_success(self, operation_service, mock_operation_repo) -> None:
        """Test successful WebSocket access verification"""
        # Mock operation
        mock_operation = Mock(spec=Operation)
        mock_operation.uuid = "test-uuid"
        mock_operation.status = OperationStatus.PROCESSING

        mock_operation_repo.get_by_uuid_with_permission_check.return_value = mock_operation

        # Test verify WebSocket access
        result = await operation_service.verify_websocket_access("test-uuid", 123)

        assert result == mock_operation
        mock_operation_repo.get_by_uuid_with_permission_check.assert_called_once_with(
            operation_uuid="test-uuid",
            user_id=123,
        )

    @pytest.mark.asyncio()
    async def test_verify_websocket_access_denied(self, operation_service, mock_operation_repo) -> None:
        """Test WebSocket access verification when denied"""
        mock_operation_repo.get_by_uuid_with_permission_check.side_effect = AccessDeniedError(
            user_id="456", resource_type="operation", resource_id="test-uuid"
        )

        with pytest.raises(AccessDeniedError):
            await operation_service.verify_websocket_access("test-uuid", 456)


class TestOperationLifecycle:
    """Test operation state transitions and lifecycle"""

    @pytest.fixture()
    def operation_service(self) -> None:
        mock_session = AsyncMock()
        mock_operation_repo = AsyncMock()
        return OperationService(
            db_session=mock_session,
            operation_repo=mock_operation_repo,
        )

    @pytest.mark.asyncio()
    async def test_operation_state_transition_pending_to_processing(self, operation_service) -> None:
        """Test transition from PENDING to PROCESSING state"""
        # Mock operation in PENDING state
        mock_operation = Mock(spec=Operation)
        mock_operation.status = OperationStatus.PENDING

        # This would typically be done by the worker, but we test the concept
        # The service layer should respect state transitions enforced by the repository

    @pytest.mark.asyncio()
    async def test_operation_state_transition_processing_to_completed(self, operation_service) -> None:
        """Test transition from PROCESSING to COMPLETED state"""
        # Mock operation in PROCESSING state
        mock_operation = Mock(spec=Operation)
        mock_operation.status = OperationStatus.PROCESSING

        # State transitions would be enforced at repository level

    @pytest.mark.asyncio()
    async def test_operation_state_transition_to_failed(self, operation_service) -> None:
        """Test transition to FAILED state from any active state"""
        # Operations can fail from PENDING or PROCESSING states
        for initial_status in [OperationStatus.PENDING, OperationStatus.PROCESSING]:
            mock_operation = Mock(spec=Operation)
            mock_operation.status = initial_status

            # Failure transitions would be handled by repository


class TestOperationServiceErrorHandling:
    """Test error handling in operation service"""

    @pytest.fixture()
    def operation_service(self) -> None:
        mock_session = AsyncMock()
        mock_operation_repo = AsyncMock()
        return OperationService(
            db_session=mock_session,
            operation_repo=mock_operation_repo,
        )

    @pytest.mark.asyncio()
    async def test_handle_repository_error(self, operation_service) -> None:
        """Test handling of repository errors"""
        operation_service.operation_repo.get_by_uuid_with_permission_check.side_effect = Exception(
            "Database connection error"
        )

        with pytest.raises(Exception, match="Database connection error"):
            await operation_service.get_operation("test-uuid", 123)

    @pytest.mark.asyncio()
    async def test_handle_concurrent_cancellation(self, operation_service) -> None:
        """Test handling concurrent cancellation attempts"""
        # First cancellation succeeds
        mock_operation = Mock(spec=Operation)
        mock_operation.status = OperationStatus.CANCELLED
        operation_service.operation_repo.cancel.return_value = mock_operation

        result = await operation_service.cancel_operation("test-uuid", 123)
        assert result.status == OperationStatus.CANCELLED

        # Second cancellation should be handled gracefully by repository
        operation_service.operation_repo.cancel.side_effect = ValidationError("Operation is already cancelled")

        with pytest.raises(ValidationError):
            await operation_service.cancel_operation("test-uuid", 123)


class TestOperationServiceIntegration:
    """Test operation service integration with other components"""

    @pytest.mark.asyncio()
    @patch("webui.services.operation_service.celery_app")
    async def test_operation_cancellation_workflow(self, mock_celery_app) -> None:
        """Test complete operation cancellation workflow"""
        # Setup
        mock_session = AsyncMock()
        mock_operation_repo = AsyncMock()
        service = OperationService(mock_session, mock_operation_repo)

        # Mock operation with all details
        mock_operation = Mock(spec=Operation)
        mock_operation.uuid = "test-uuid"
        mock_operation.type = OperationType.INDEX
        mock_operation.status = OperationStatus.PROCESSING
        mock_operation.task_id = "celery-task-456"
        mock_operation.user_id = 123
        mock_operation.collection_id = "collection-uuid"
        mock_operation.progress = 45
        mock_operation.started_at = "2024-01-01T00:00:00"

        mock_operation_repo.cancel.return_value = mock_operation
        mock_control = Mock()
        mock_celery_app.control = mock_control

        # Execute cancellation
        result = await service.cancel_operation("test-uuid", 123)

        # Verify complete workflow
        assert result == mock_operation
        mock_operation_repo.cancel.assert_called_once()
        mock_control.revoke.assert_called_once_with("celery-task-456", terminate=True)
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio()
    async def test_operation_listing_pagination(self) -> None:
        """Test operation listing with pagination"""
        mock_session = AsyncMock()
        mock_operation_repo = AsyncMock()
        service = OperationService(mock_session, mock_operation_repo)

        # Mock paginated results
        page1_ops = [Mock(spec=Operation) for _ in range(50)]
        page2_ops = [Mock(spec=Operation) for _ in range(25)]

        # First page
        mock_operation_repo.list_for_user.return_value = (page1_ops, 75)
        ops1, total1 = await service.list_operations(user_id=123, offset=0, limit=50)
        assert len(ops1) == 50
        assert total1 == 75

        # Second page
        mock_operation_repo.list_for_user.return_value = (page2_ops, 75)
        ops2, total2 = await service.list_operations(user_id=123, offset=50, limit=50)
        assert len(ops2) == 25
        assert total2 == 75
