"""Tests for operation WebSocket endpoints."""

import asyncio
import contextlib
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import WebSocket
from shared.database.models import Operation, OperationStatus, OperationType

from packages.webui.websocket_manager import RedisStreamWebSocketManager


class TestOperationWebSocket:
    """Test operation WebSocket functionality."""

    @pytest.fixture()
    def mock_websocket(self):
        """Create a mock WebSocket instance."""
        ws = MagicMock(spec=WebSocket)
        ws.accept = AsyncMock()
        ws.send_json = AsyncMock()
        ws.receive_text = AsyncMock()
        ws.close = AsyncMock()
        ws.query_params = {"token": "valid_token"}
        return ws

    @pytest.fixture()
    def mock_operation(self):
        """Create a mock operation."""
        operation = MagicMock(spec=Operation)
        operation.uuid = "test-operation-123"
        operation.status = OperationStatus.PROCESSING
        operation.type = OperationType.INDEX
        operation.created_at = MagicMock(isoformat=MagicMock(return_value="2024-01-01T00:00:00"))
        operation.started_at = MagicMock(isoformat=MagicMock(return_value="2024-01-01T00:01:00"))
        operation.completed_at = None
        operation.error_message = None
        operation.collection_id = "test-collection-456"
        operation.user_id = 1
        return operation

    @pytest.fixture()
    def ws_manager(self):
        """Create WebSocket manager instance."""
        manager = RedisStreamWebSocketManager()
        manager.redis = AsyncMock()
        return manager

    @pytest.mark.asyncio()
    async def test_operation_websocket_authentication_success(self, mock_websocket, mock_operation):
        """Test successful authentication for operation WebSocket."""
        from packages.webui.api.jobs import operation_websocket_endpoint

        with (
            patch("packages.webui.api.jobs.get_current_user_websocket") as mock_auth,
            patch("shared.database.database.AsyncSessionLocal") as mock_session,
            patch("shared.database.repositories.operation_repository.OperationRepository") as mock_repo_class,
            patch("packages.webui.api.jobs.ws_manager") as mock_ws_manager,
        ):
            # Setup mocks
            mock_auth.return_value = {"id": 1, "username": "testuser"}
            mock_repo = AsyncMock()
            mock_repo.get_by_uuid_with_permission_check = AsyncMock(return_value=mock_operation)
            mock_repo_class.return_value = mock_repo

            # Mock the async context manager
            mock_db = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_db
            mock_session.return_value.__aexit__.return_value = None

            # Mock WebSocket manager
            mock_ws_manager.connect_operation = AsyncMock()
            mock_ws_manager.disconnect_operation = AsyncMock()

            # Mock WebSocket disconnect
            mock_websocket.receive_text.side_effect = asyncio.CancelledError()

            # Run the endpoint
            with contextlib.suppress(asyncio.CancelledError):
                await operation_websocket_endpoint(mock_websocket, "test-operation-123")

            # Verify authentication was called
            mock_auth.assert_called_once_with("valid_token")

            # Verify permission check was called
            mock_repo.get_by_uuid_with_permission_check.assert_called_once_with("test-operation-123", 1)

            # Verify WebSocket was connected
            mock_ws_manager.connect_operation.assert_called_once_with(mock_websocket, "test-operation-123", "1")

    @pytest.mark.asyncio()
    async def test_operation_websocket_authentication_failure(self, mock_websocket):
        """Test failed authentication for operation WebSocket."""
        from packages.webui.api.jobs import operation_websocket_endpoint

        with patch("packages.webui.api.jobs.get_current_user_websocket") as mock_auth:
            # Setup mock to raise authentication error
            mock_auth.side_effect = ValueError("Invalid token")

            # Run the endpoint
            await operation_websocket_endpoint(mock_websocket, "test-operation-123")

            # Verify WebSocket was closed with error
            mock_websocket.close.assert_called_once_with(code=1008, reason="Invalid token")

    @pytest.mark.asyncio()
    async def test_operation_websocket_permission_denied(self, mock_websocket):
        """Test permission denied for operation WebSocket."""
        from shared.database.exceptions import AccessDeniedError

        from packages.webui.api.jobs import operation_websocket_endpoint

        with (
            patch("packages.webui.api.jobs.get_current_user_websocket") as mock_auth,
            patch("shared.database.database.AsyncSessionLocal") as mock_session,
            patch("shared.database.repositories.operation_repository.OperationRepository") as mock_repo_class,
        ):
            # Setup mocks
            mock_auth.return_value = {"id": 2, "username": "otheruser"}
            mock_repo = AsyncMock()
            mock_repo.get_by_uuid_with_permission_check = AsyncMock(
                side_effect=AccessDeniedError("2", "operation", "test-operation-123")
            )
            mock_repo_class.return_value = mock_repo

            # Mock the async context manager
            mock_db = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_db
            mock_session.return_value.__aexit__.return_value = None

            # Run the endpoint
            await operation_websocket_endpoint(mock_websocket, "test-operation-123")

            # Verify WebSocket was closed with error
            mock_websocket.close.assert_called_once()
            assert mock_websocket.close.call_args[1]["code"] == 1011

    @pytest.mark.asyncio()
    async def test_send_operation_update_to_redis(self, ws_manager):
        """Test sending operation updates to Redis stream."""
        operation_id = "test-op-123"
        update_type = "progress"
        data = {"progress": 50, "message": "Processing files..."}

        # Setup Redis mock
        ws_manager.redis.xadd = AsyncMock()
        ws_manager.redis.expire = AsyncMock()

        # Send update
        await ws_manager.send_operation_update(operation_id, update_type, data)

        # Verify Redis operations
        ws_manager.redis.xadd.assert_called_once()
        call_args = ws_manager.redis.xadd.call_args
        assert call_args[0][0] == f"operation-progress:{operation_id}"

        # Verify message structure
        message_data = json.loads(call_args[0][1]["message"])
        assert message_data["type"] == update_type
        assert message_data["data"] == data
        assert "timestamp" in message_data

    @pytest.mark.asyncio()
    async def test_operation_progress_streaming(self, ws_manager, mock_websocket):
        """Test streaming operation progress updates to connected clients."""
        operation_id = "test-op-123"
        user_id = "1"

        # Connect WebSocket
        ws_manager.connections[f"{user_id}:operation:{operation_id}"] = {mock_websocket}

        # Send progress update
        message = {
            "timestamp": "2024-01-01T00:00:00",
            "type": "progress",
            "data": {"progress": 75, "message": "Almost done..."}
        }

        await ws_manager._broadcast_to_operation(operation_id, message)

        # Verify WebSocket received the message
        mock_websocket.send_json.assert_called_once_with(message)

    @pytest.mark.asyncio()
    async def test_operation_completion_closes_connections(self, ws_manager, mock_websocket):
        """Test that operation completion closes WebSocket connections."""
        operation_id = "test-op-123"
        user_id = "1"

        # Connect WebSocket
        key = f"{user_id}:operation:{operation_id}"
        ws_manager.connections[key] = {mock_websocket}

        # Close connections for completed operation
        await ws_manager._close_operation_connections(operation_id)

        # Verify WebSocket was closed
        mock_websocket.close.assert_called_once_with(code=1000, reason="Operation completed")

        # Verify connection was removed
        assert key not in ws_manager.connections

    @pytest.mark.asyncio()
    async def test_cleanup_operation_stream(self, ws_manager):
        """Test cleanup of Redis stream for completed operation."""
        operation_id = "test-op-123"
        stream_key = f"operation-progress:{operation_id}"

        # Setup Redis mock
        ws_manager.redis.delete = AsyncMock(return_value=1)
        ws_manager.redis.xinfo_groups = AsyncMock(return_value=[{"name": "test-group"}])
        ws_manager.redis.xgroup_destroy = AsyncMock()

        # Cleanup stream
        await ws_manager.cleanup_operation_stream(operation_id)

        # Verify Redis operations
        ws_manager.redis.delete.assert_called_once_with(stream_key)
        ws_manager.redis.xinfo_groups.assert_called_once_with(stream_key)
        ws_manager.redis.xgroup_destroy.assert_called_once_with(stream_key, "test-group")
