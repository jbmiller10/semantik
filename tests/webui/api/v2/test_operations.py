"""
Tests for v2 operations API endpoints.

Comprehensive test coverage for all operation management endpoints including
get, list, cancel operations and WebSocket connections.
"""

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, WebSocket
from starlette.websockets import WebSocketDisconnect

from packages.shared.database.exceptions import AccessDeniedError, EntityNotFoundError, ValidationError
from packages.shared.database.models import Operation, OperationStatus, OperationType
from packages.webui.api.schemas import OperationResponse
from packages.webui.api.v2.operations import cancel_operation, get_operation, list_operations, operation_websocket
from packages.webui.services.operation_service import OperationService


@pytest.fixture()
def mock_user() -> dict[str, Any]:
    """Mock authenticated user."""
    return {"id": 1, "username": "testuser"}


@pytest.fixture()
def mock_operation() -> MagicMock:
    """Mock operation object."""
    operation = MagicMock(spec=Operation)
    operation.uuid = "550e8400-e29b-41d4-a716-446655440000"
    operation.collection_id = "123e4567-e89b-12d3-a456-426614174000"
    operation.type = OperationType.INDEX
    operation.status = OperationStatus.PROCESSING
    operation.config = {"path": "/data/documents", "recursive": True}
    operation.error_message = None
    operation.created_at = datetime.now(UTC)
    operation.started_at = datetime.now(UTC)
    operation.completed_at = None
    operation.task_id = "celery-task-123"
    operation.owner_id = 1
    return operation


@pytest.fixture()
def mock_operation_service() -> AsyncMock:
    """Mock OperationService."""
    return AsyncMock(spec=OperationService)


class TestGetOperation:
    """Test get_operation endpoint."""

    @pytest.mark.asyncio()
    async def test_get_operation_success(
        self,
        mock_user: dict[str, Any],
        mock_operation: MagicMock,
        mock_operation_service: AsyncMock) -> None:
        """Test successful operation retrieval."""
        # Setup
        mock_operation_service.get_operation.return_value = mock_operation

        # Execute
        result = await get_operation(
            operation_uuid=mock_operation.uuid,
            current_user=mock_user,
            service=mock_operation_service)

        # Verify
        assert isinstance(result, OperationResponse)
        assert result.id == mock_operation.uuid
        assert result.collection_id == mock_operation.collection_id
        assert result.type == mock_operation.type.value
        assert result.status == mock_operation.status.value
        assert result.config == mock_operation.config
        assert result.error_message is None
        assert result.created_at == mock_operation.created_at
        assert result.started_at == mock_operation.started_at
        assert result.completed_at is None

        mock_operation_service.get_operation.assert_awaited_once_with(
            operation_uuid=mock_operation.uuid,
            user_id=mock_user["id"])

    @pytest.mark.asyncio()
    async def test_get_operation_not_found(
        self,
        mock_user: dict[str, Any],
        mock_operation_service: AsyncMock) -> None:
        """Test operation not found error."""
        # Setup
        operation_uuid = "non-existent-uuid"
        mock_operation_service.get_operation.side_effect = EntityNotFoundError("Operation", operation_uuid)

        # Execute & Verify
        with pytest.raises(HTTPException) as exc_info:
            await get_operation(
                operation_uuid=operation_uuid,
                current_user=mock_user,
                service=mock_operation_service)

        assert exc_info.value.status_code == 404
        assert f"Operation '{operation_uuid}' not found" in str(exc_info.value.detail)

    @pytest.mark.asyncio()
    async def test_get_operation_access_denied(
        self,
        mock_user: dict[str, Any],
        mock_operation_service: AsyncMock) -> None:
        """Test access denied error."""
        # Setup
        operation_uuid = "550e8400-e29b-41d4-a716-446655440000"
        mock_operation_service.get_operation.side_effect = AccessDeniedError(
            str(mock_user["id"]), "operation", operation_uuid
        )

        # Execute & Verify
        with pytest.raises(HTTPException) as exc_info:
            await get_operation(
                operation_uuid=operation_uuid,
                current_user=mock_user,
                service=mock_operation_service)

        assert exc_info.value.status_code == 403
        assert "You don't have access to this operation" in str(exc_info.value.detail)

    @pytest.mark.asyncio()
    async def test_get_operation_with_error_message(
        self,
        mock_user: dict[str, Any],
        mock_operation: MagicMock,
        mock_operation_service: AsyncMock) -> None:
        """Test getting operation with error message."""
        # Setup
        mock_operation.status = OperationStatus.FAILED
        mock_operation.error_message = "Failed to process documents"
        mock_operation.completed_at = datetime.now(UTC)
        mock_operation_service.get_operation.return_value = mock_operation

        # Execute
        result = await get_operation(
            operation_uuid=mock_operation.uuid,
            current_user=mock_user,
            service=mock_operation_service)

        # Verify
        assert result.status == OperationStatus.FAILED.value
        assert result.error_message == "Failed to process documents"
        assert result.completed_at is not None

    @pytest.mark.asyncio()
    async def test_get_operation_generic_error(
        self,
        mock_user: dict[str, Any],
        mock_operation_service: AsyncMock) -> None:
        """Test generic error handling."""
        # Setup
        mock_operation_service.get_operation.side_effect = Exception("Database connection error")

        # Execute & Verify
        with pytest.raises(HTTPException) as exc_info:
            await get_operation(
                operation_uuid="test-uuid",
                current_user=mock_user,
                service=mock_operation_service)

        assert exc_info.value.status_code == 500
        assert "Failed to get operation" in str(exc_info.value.detail)


class TestCancelOperation:
    """Test cancel_operation endpoint."""

    @pytest.mark.asyncio()
    async def test_cancel_operation_success(
        self,
        mock_user: dict[str, Any],
        mock_operation: MagicMock,
        mock_operation_service: AsyncMock) -> None:
        """Test successful operation cancellation."""
        # Setup
        mock_operation.status = OperationStatus.CANCELLED
        mock_operation.completed_at = datetime.now(UTC)
        mock_operation_service.cancel_operation.return_value = mock_operation

        # Execute
        result = await cancel_operation(
            operation_uuid=mock_operation.uuid,
            current_user=mock_user,
            service=mock_operation_service)

        # Verify
        assert isinstance(result, OperationResponse)
        assert result.status == OperationStatus.CANCELLED.value
        assert result.completed_at is not None

        mock_operation_service.cancel_operation.assert_awaited_once_with(
            operation_uuid=mock_operation.uuid,
            user_id=mock_user["id"])

    @pytest.mark.asyncio()
    async def test_cancel_operation_not_found(
        self,
        mock_user: dict[str, Any],
        mock_operation_service: AsyncMock) -> None:
        """Test cancelling non-existent operation."""
        # Setup
        operation_uuid = "non-existent-uuid"
        mock_operation_service.cancel_operation.side_effect = EntityNotFoundError("Operation", operation_uuid)

        # Execute & Verify
        with pytest.raises(HTTPException) as exc_info:
            await cancel_operation(
                operation_uuid=operation_uuid,
                current_user=mock_user,
                service=mock_operation_service)

        assert exc_info.value.status_code == 404
        assert f"Operation '{operation_uuid}' not found" in str(exc_info.value.detail)

    @pytest.mark.asyncio()
    async def test_cancel_operation_access_denied(
        self,
        mock_user: dict[str, Any],
        mock_operation_service: AsyncMock) -> None:
        """Test access denied when cancelling operation."""
        # Setup
        operation_uuid = "550e8400-e29b-41d4-a716-446655440000"
        mock_operation_service.cancel_operation.side_effect = AccessDeniedError(
            str(mock_user["id"]), "operation", operation_uuid
        )

        # Execute & Verify
        with pytest.raises(HTTPException) as exc_info:
            await cancel_operation(
                operation_uuid=operation_uuid,
                current_user=mock_user,
                service=mock_operation_service)

        assert exc_info.value.status_code == 403
        assert "You don't have permission to cancel this operation" in str(exc_info.value.detail)

    @pytest.mark.asyncio()
    async def test_cancel_operation_validation_error(
        self,
        mock_user: dict[str, Any],
        mock_operation_service: AsyncMock) -> None:
        """Test validation error when operation cannot be cancelled."""
        # Setup
        mock_operation_service.cancel_operation.side_effect = ValidationError(
            "Operation is already completed and cannot be cancelled"
        )

        # Execute & Verify
        with pytest.raises(HTTPException) as exc_info:
            await cancel_operation(
                operation_uuid="test-uuid",
                current_user=mock_user,
                service=mock_operation_service)

        assert exc_info.value.status_code == 400
        assert "Operation is already completed and cannot be cancelled" in str(exc_info.value.detail)

    @pytest.mark.asyncio()
    async def test_cancel_operation_generic_error(
        self,
        mock_user: dict[str, Any],
        mock_operation_service: AsyncMock) -> None:
        """Test generic error handling during cancellation."""
        # Setup
        mock_operation_service.cancel_operation.side_effect = Exception("Celery connection error")

        # Execute & Verify
        with pytest.raises(HTTPException) as exc_info:
            await cancel_operation(
                operation_uuid="test-uuid",
                current_user=mock_user,
                service=mock_operation_service)

        assert exc_info.value.status_code == 500
        assert "Failed to cancel operation" in str(exc_info.value.detail)


class TestListOperations:
    """Test list_operations endpoint."""

    @pytest.mark.asyncio()
    async def test_list_operations_success_no_filters(
        self,
        mock_user: dict[str, Any],
        mock_operation_service: AsyncMock) -> None:
        """Test listing operations without filters."""
        # Setup
        operations = []
        for i in range(3):
            op = MagicMock(spec=Operation)
            op.uuid = f"op-{i}"
            op.collection_id = f"col-{i}"
            op.type = OperationType.INDEX
            op.status = OperationStatus.COMPLETED
            op.config = {}
            op.error_message = None
            op.created_at = datetime.now(UTC)
            op.started_at = datetime.now(UTC)
            op.completed_at = datetime.now(UTC)
            operations.append(op)

        mock_operation_service.list_operations.return_value = (operations, 3)

        # Execute
        result = await list_operations(
            status=None,
            operation_type=None,
            page=1,
            per_page=50,
            current_user=mock_user,
            service=mock_operation_service)

        # Verify
        assert len(result) == 3
        assert all(isinstance(r, OperationResponse) for r in result)
        assert result[0].id == "op-0"
        assert result[1].id == "op-1"
        assert result[2].id == "op-2"

        mock_operation_service.list_operations.assert_awaited_once_with(
            user_id=mock_user["id"],
            status_list=None,
            operation_type=None,
            offset=0,
            limit=50)

    @pytest.mark.asyncio()
    async def test_list_operations_with_status_filter(
        self,
        mock_user: dict[str, Any],
        mock_operation_service: AsyncMock) -> None:
        """Test listing operations with status filter."""
        # Setup
        operations = []
        op = MagicMock(spec=Operation)
        op.uuid = "op-1"
        op.collection_id = "col-1"
        op.type = OperationType.INDEX
        op.status = OperationStatus.PROCESSING
        op.config = {}
        op.error_message = None
        op.created_at = datetime.now(UTC)
        op.started_at = datetime.now(UTC)
        op.completed_at = None
        operations.append(op)

        mock_operation_service.list_operations.return_value = (operations, 1)

        # Execute
        result = await list_operations(
            status="processing,pending",
            operation_type=None,
            page=1,
            per_page=50,
            current_user=mock_user,
            service=mock_operation_service)

        # Verify
        assert len(result) == 1
        assert result[0].status == OperationStatus.PROCESSING.value

        mock_operation_service.list_operations.assert_awaited_once_with(
            user_id=mock_user["id"],
            status_list=[OperationStatus.PROCESSING, OperationStatus.PENDING],
            operation_type=None,
            offset=0,
            limit=50)

    @pytest.mark.asyncio()
    async def test_list_operations_with_type_filter(
        self,
        mock_user: dict[str, Any],
        mock_operation_service: AsyncMock) -> None:
        """Test listing operations with type filter."""
        # Setup
        operations = []
        op = MagicMock(spec=Operation)
        op.uuid = "op-1"
        op.collection_id = "col-1"
        op.type = OperationType.REINDEX
        op.status = OperationStatus.COMPLETED
        op.config = {}
        op.error_message = None
        op.created_at = datetime.now(UTC)
        op.started_at = datetime.now(UTC)
        op.completed_at = datetime.now(UTC)
        operations.append(op)

        mock_operation_service.list_operations.return_value = (operations, 1)

        # Execute
        result = await list_operations(
            status=None,
            operation_type="reindex",
            page=1,
            per_page=50,
            current_user=mock_user,
            service=mock_operation_service)

        # Verify
        assert len(result) == 1
        assert result[0].type == OperationType.REINDEX.value

        mock_operation_service.list_operations.assert_awaited_once_with(
            user_id=mock_user["id"],
            status_list=None,
            operation_type=OperationType.REINDEX,
            offset=0,
            limit=50)

    @pytest.mark.asyncio()
    async def test_list_operations_with_pagination(
        self,
        mock_user: dict[str, Any],
        mock_operation_service: AsyncMock) -> None:
        """Test listing operations with pagination."""
        # Setup
        operations = []
        for i in range(20):
            op = MagicMock(spec=Operation)
            op.uuid = f"op-{i}"
            op.collection_id = f"col-{i}"
            op.type = OperationType.INDEX
            op.status = OperationStatus.COMPLETED
            op.config = {}
            op.error_message = None
            op.created_at = datetime.now(UTC)
            op.started_at = datetime.now(UTC)
            op.completed_at = datetime.now(UTC)
            operations.append(op)

        # Return only page 2 (items 10-19)
        mock_operation_service.list_operations.return_value = (operations[10:20], 20)

        # Execute
        result = await list_operations(
            status=None,
            operation_type=None,
            page=2,
            per_page=10,
            current_user=mock_user,
            service=mock_operation_service)

        # Verify
        assert len(result) == 10
        assert result[0].id == "op-10"
        assert result[9].id == "op-19"

        mock_operation_service.list_operations.assert_awaited_once_with(
            user_id=mock_user["id"],
            status_list=None,
            operation_type=None,
            offset=10,  # (page-1) * per_page
            limit=10)

    @pytest.mark.asyncio()
    async def test_list_operations_invalid_status(
        self,
        mock_user: dict[str, Any],
        mock_operation_service: AsyncMock) -> None:
        """Test listing operations with invalid status."""
        # Execute & Verify
        with pytest.raises(HTTPException) as exc_info:
            await list_operations(
                status="invalid_status",
                operation_type=None,
                page=1,
                per_page=50,
                current_user=mock_user,
                service=mock_operation_service)

        assert exc_info.value.status_code == 400
        assert "Invalid status: invalid_status" in str(exc_info.value.detail)
        assert "Valid values are:" in str(exc_info.value.detail)

    @pytest.mark.asyncio()
    async def test_list_operations_invalid_type(
        self,
        mock_user: dict[str, Any],
        mock_operation_service: AsyncMock) -> None:
        """Test listing operations with invalid type."""
        # Execute & Verify
        with pytest.raises(HTTPException) as exc_info:
            await list_operations(
                status=None,
                operation_type="invalid_type",
                page=1,
                per_page=50,
                current_user=mock_user,
                service=mock_operation_service)

        assert exc_info.value.status_code == 400
        assert "Invalid operation type: invalid_type" in str(exc_info.value.detail)
        assert "Valid values are:" in str(exc_info.value.detail)

    @pytest.mark.asyncio()
    async def test_list_operations_with_failed_operations(
        self,
        mock_user: dict[str, Any],
        mock_operation_service: AsyncMock) -> None:
        """Test listing operations including failed ones with error messages."""
        # Setup
        operations = []

        # Successful operation
        op1 = MagicMock(spec=Operation)
        op1.uuid = "op-1"
        op1.collection_id = "col-1"
        op1.type = OperationType.INDEX
        op1.status = OperationStatus.COMPLETED
        op1.config = {}
        op1.error_message = None
        op1.created_at = datetime.now(UTC)
        op1.started_at = datetime.now(UTC)
        op1.completed_at = datetime.now(UTC)
        operations.append(op1)

        # Failed operation
        op2 = MagicMock(spec=Operation)
        op2.uuid = "op-2"
        op2.collection_id = "col-2"
        op2.type = OperationType.REINDEX
        op2.status = OperationStatus.FAILED
        op2.config = {}
        op2.error_message = "Failed to connect to vector database"
        op2.created_at = datetime.now(UTC)
        op2.started_at = datetime.now(UTC)
        op2.completed_at = datetime.now(UTC)
        operations.append(op2)

        mock_operation_service.list_operations.return_value = (operations, 2)

        # Execute
        result = await list_operations(
            status=None,
            operation_type=None,
            page=1,
            per_page=50,
            current_user=mock_user,
            service=mock_operation_service)

        # Verify
        assert len(result) == 2
        assert result[0].status == OperationStatus.COMPLETED.value
        assert result[0].error_message is None
        assert result[1].status == OperationStatus.FAILED.value
        assert result[1].error_message == "Failed to connect to vector database"

    @pytest.mark.asyncio()
    async def test_list_operations_generic_error(
        self,
        mock_user: dict[str, Any],
        mock_operation_service: AsyncMock) -> None:
        """Test generic error handling when listing operations."""
        # Setup
        mock_operation_service.list_operations.side_effect = Exception("Database connection error")

        # Execute & Verify
        with pytest.raises(HTTPException) as exc_info:
            await list_operations(
                status=None,
                operation_type=None,
                page=1,
                per_page=50,
                current_user=mock_user,
                service=mock_operation_service)

        assert exc_info.value.status_code == 500
        assert "Failed to list operations" in str(exc_info.value.detail)


class TestOperationWebSocket:
    """Test operation_websocket endpoint."""

    @pytest.mark.asyncio()
    async def test_websocket_success(self) -> None:
        """Test successful WebSocket connection and message handling."""
        # Setup
        mock_websocket = AsyncMock(spec=WebSocket)
        mock_websocket.query_params = {"token": "valid-jwt-token"}
        mock_user = {"id": 1, "username": "testuser"}

        with (
            patch("packages.webui.api.v2.operations.get_current_user_websocket") as mock_get_user,
            patch("packages.webui.api.v2.operations.get_db") as mock_get_db,
            patch("packages.webui.api.v2.operations.OperationRepository") as mock_repo_class,
            patch("packages.webui.api.v2.operations.OperationService") as mock_service_class,
            patch("packages.webui.api.v2.operations.ws_manager") as mock_ws_manager):

            # Mock authentication
            mock_get_user.return_value = mock_user

            # Mock database session
            mock_db = AsyncMock()
            mock_get_db.return_value.__aiter__.return_value = [mock_db]

            # Mock repository and service
            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo
            mock_service = AsyncMock()
            mock_service_class.return_value = mock_service
            mock_service.verify_websocket_access.return_value = None

            # Mock WebSocket manager
            mock_ws_manager.connect = AsyncMock()
            mock_ws_manager.disconnect = AsyncMock()

            # Mock WebSocket receive to simulate ping/pong and then disconnect
            mock_websocket.receive_json = AsyncMock()
            mock_websocket.receive_json.side_effect = [
                {"type": "ping"},
                WebSocketDisconnect(),
            ]
            mock_websocket.send_json = AsyncMock()

            # Execute
            await operation_websocket(mock_websocket, "op-123")

            # Verify
            mock_get_user.assert_awaited_once_with("valid-jwt-token")
            mock_service.verify_websocket_access.assert_awaited_once_with(
                operation_uuid="op-123",
                user_id=1)
            mock_ws_manager.connect.assert_awaited_once_with(mock_websocket, "op-123", "1")
            mock_websocket.send_json.assert_awaited_once_with({"type": "pong"})
            mock_ws_manager.disconnect.assert_awaited_once_with(mock_websocket, "op-123", "1")

    @pytest.mark.asyncio()
    async def test_websocket_authentication_failure(self) -> None:
        """Test WebSocket authentication failure."""
        # Setup
        mock_websocket = AsyncMock(spec=WebSocket)
        mock_websocket.query_params = {"token": "invalid-jwt-token"}

        with patch("packages.webui.api.v2.operations.get_current_user_websocket") as mock_get_user:
            # Mock authentication failure
            mock_get_user.side_effect = ValueError("Invalid token")

            # Execute
            await operation_websocket(mock_websocket, "op-123")

            # Verify
            mock_websocket.close.assert_awaited_once_with(code=1008, reason="Invalid token")

    @pytest.mark.asyncio()
    async def test_websocket_no_token(self) -> None:
        """Test WebSocket connection without token."""
        # Setup
        mock_websocket = AsyncMock(spec=WebSocket)
        mock_websocket.query_params = {}

        with patch("packages.webui.api.v2.operations.get_current_user_websocket") as mock_get_user:
            # Mock authentication with None token
            mock_get_user.side_effect = ValueError("No token provided")

            # Execute
            await operation_websocket(mock_websocket, "op-123")

            # Verify
            mock_websocket.close.assert_awaited_once_with(code=1008, reason="No token provided")

    @pytest.mark.asyncio()
    async def test_websocket_operation_not_found(self) -> None:
        """Test WebSocket connection for non-existent operation."""
        # Setup
        mock_websocket = AsyncMock(spec=WebSocket)
        mock_websocket.query_params = {"token": "valid-jwt-token"}
        mock_user = {"id": 1, "username": "testuser"}

        with (
            patch("packages.webui.api.v2.operations.get_current_user_websocket") as mock_get_user,
            patch("packages.webui.api.v2.operations.get_db") as mock_get_db,
            patch("packages.webui.api.v2.operations.OperationRepository") as mock_repo_class,
            patch("packages.webui.api.v2.operations.OperationService") as mock_service_class):

            # Mock authentication
            mock_get_user.return_value = mock_user

            # Mock database session
            mock_db = AsyncMock()
            mock_get_db.return_value.__aiter__.return_value = [mock_db]

            # Mock repository and service
            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo
            mock_service = AsyncMock()
            mock_service_class.return_value = mock_service
            mock_service.verify_websocket_access.side_effect = EntityNotFoundError("Operation", "non-existent-op")

            # Execute
            await operation_websocket(mock_websocket, "non-existent-op")

            # Verify
            mock_websocket.close.assert_awaited_once_with(code=1008, reason="Operation 'non-existent-op' not found")

    @pytest.mark.asyncio()
    async def test_websocket_access_denied(self) -> None:
        """Test WebSocket connection with access denied."""
        # Setup
        mock_websocket = AsyncMock(spec=WebSocket)
        mock_websocket.query_params = {"token": "valid-jwt-token"}
        mock_user = {"id": 2, "username": "otheruser"}

        with (
            patch("packages.webui.api.v2.operations.get_current_user_websocket") as mock_get_user,
            patch("packages.webui.api.v2.operations.get_db") as mock_get_db,
            patch("packages.webui.api.v2.operations.OperationRepository") as mock_repo_class,
            patch("packages.webui.api.v2.operations.OperationService") as mock_service_class):

            # Mock authentication
            mock_get_user.return_value = mock_user

            # Mock database session
            mock_db = AsyncMock()
            mock_get_db.return_value.__aiter__.return_value = [mock_db]

            # Mock repository and service
            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo
            mock_service = AsyncMock()
            mock_service_class.return_value = mock_service
            mock_service.verify_websocket_access.side_effect = AccessDeniedError(
                str(mock_user["id"]), "operation", "op-123"
            )

            # Execute
            await operation_websocket(mock_websocket, "op-123")

            # Verify
            mock_websocket.close.assert_awaited_once_with(code=1008, reason="You don't have access to this operation")

    @pytest.mark.asyncio()
    async def test_websocket_unexpected_error(self) -> None:
        """Test WebSocket connection with unexpected error."""
        # Setup
        mock_websocket = AsyncMock(spec=WebSocket)
        mock_websocket.query_params = {"token": "valid-jwt-token"}

        with patch("packages.webui.api.v2.operations.get_current_user_websocket") as mock_get_user:
            # Mock unexpected error during authentication
            mock_get_user.side_effect = Exception("Unexpected error")

            # Execute
            await operation_websocket(mock_websocket, "op-123")

            # Verify
            mock_websocket.close.assert_awaited_once_with(code=1011, reason="Internal server error")

    @pytest.mark.asyncio()
    async def test_websocket_client_disconnect(self) -> None:
        """Test WebSocket client disconnect handling."""
        # Setup
        mock_websocket = AsyncMock(spec=WebSocket)
        mock_websocket.query_params = {"token": "valid-jwt-token"}
        mock_user = {"id": 1, "username": "testuser"}

        with (
            patch("packages.webui.api.v2.operations.get_current_user_websocket") as mock_get_user,
            patch("packages.webui.api.v2.operations.get_db") as mock_get_db,
            patch("packages.webui.api.v2.operations.OperationRepository") as mock_repo_class,
            patch("packages.webui.api.v2.operations.OperationService") as mock_service_class,
            patch("packages.webui.api.v2.operations.ws_manager") as mock_ws_manager):

            # Mock authentication
            mock_get_user.return_value = mock_user

            # Mock database session
            mock_db = AsyncMock()
            mock_get_db.return_value.__aiter__.return_value = [mock_db]

            # Mock repository and service
            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo
            mock_service = AsyncMock()
            mock_service_class.return_value = mock_service
            mock_service.verify_websocket_access.return_value = None

            # Mock WebSocket manager
            mock_ws_manager.connect = AsyncMock()
            mock_ws_manager.disconnect = AsyncMock()

            # Mock WebSocket receive to simulate immediate disconnect
            mock_websocket.receive_json = AsyncMock()
            mock_websocket.receive_json.side_effect = WebSocketDisconnect()

            # Execute
            await operation_websocket(mock_websocket, "op-123")

            # Verify
            mock_ws_manager.connect.assert_awaited_once_with(mock_websocket, "op-123", "1")
            mock_ws_manager.disconnect.assert_awaited_once_with(mock_websocket, "op-123", "1")

    @pytest.mark.asyncio()
    async def test_websocket_multiple_messages(self) -> None:
        """Test WebSocket handling multiple messages before disconnect."""
        # Setup
        mock_websocket = AsyncMock(spec=WebSocket)
        mock_websocket.query_params = {"token": "valid-jwt-token"}
        mock_user = {"id": 1, "username": "testuser"}

        with (
            patch("packages.webui.api.v2.operations.get_current_user_websocket") as mock_get_user,
            patch("packages.webui.api.v2.operations.get_db") as mock_get_db,
            patch("packages.webui.api.v2.operations.OperationRepository") as mock_repo_class,
            patch("packages.webui.api.v2.operations.OperationService") as mock_service_class,
            patch("packages.webui.api.v2.operations.ws_manager") as mock_ws_manager):

            # Mock authentication
            mock_get_user.return_value = mock_user

            # Mock database session
            mock_db = AsyncMock()
            mock_get_db.return_value.__aiter__.return_value = [mock_db]

            # Mock repository and service
            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo
            mock_service = AsyncMock()
            mock_service_class.return_value = mock_service
            mock_service.verify_websocket_access.return_value = None

            # Mock WebSocket manager
            mock_ws_manager.connect = AsyncMock()
            mock_ws_manager.disconnect = AsyncMock()

            # Mock WebSocket receive to simulate multiple ping/pong exchanges
            mock_websocket.receive_json = AsyncMock()
            mock_websocket.receive_json.side_effect = [
                {"type": "ping"},
                {"type": "ping"},
                {"type": "other"},  # Non-ping message
                WebSocketDisconnect(),
            ]
            mock_websocket.send_json = AsyncMock()

            # Execute
            await operation_websocket(mock_websocket, "op-123")

            # Verify
            assert mock_websocket.send_json.await_count == 2  # Only ping messages get pong responses
            mock_websocket.send_json.assert_any_await({"type": "pong"})
            mock_ws_manager.disconnect.assert_awaited_once_with(mock_websocket, "op-123", "1")
