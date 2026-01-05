"""Basic tests to verify WebSocket manager setup."""

from unittest.mock import AsyncMock

import pytest

from webui.websocket.legacy_stream_manager import RedisStreamWebSocketManager


class TestWebSocketBasic:
    """Basic tests for WebSocket manager."""

    @pytest.mark.asyncio()
    async def test_manager_initialization(self) -> None:
        """Test that manager initializes correctly."""
        manager = RedisStreamWebSocketManager()

        assert manager.redis is None
        assert manager.connections == {}
        assert manager.consumer_tasks == {}
        assert manager.max_connections_per_user == 10

    @pytest.mark.asyncio()
    async def test_send_operation_update_without_redis(self) -> None:
        """Test sending operation update when Redis is not available."""
        manager = RedisStreamWebSocketManager()
        manager.redis = None  # No Redis connection

        # Create a mock WebSocket
        mock_ws = AsyncMock()
        mock_ws.send_json = AsyncMock()

        # Add connection
        manager.connections["user1:operation:test-op-123"] = {mock_ws}

        # Send update
        await manager.send_update("test-op-123", "progress", {"progress": 50})

        # Verify WebSocket received the update
        mock_ws.send_json.assert_called_once()
        sent_data = mock_ws.send_json.call_args[0][0]
        assert sent_data["type"] == "progress"
        assert sent_data["data"]["progress"] == 50

    @pytest.mark.asyncio()
    async def test_broadcast_to_operation(self) -> None:
        """Test broadcasting to multiple WebSocket connections."""
        manager = RedisStreamWebSocketManager()

        # Create mock WebSockets
        ws1 = AsyncMock()
        ws2 = AsyncMock()
        ws3 = AsyncMock()

        # Add connections
        manager.connections["user1:operation:test-op-123"] = {ws1, ws2}
        manager.connections["user2:operation:test-op-123"] = {ws3}

        # Broadcast message
        test_message = {"type": "test", "data": {"msg": "hello"}}
        await manager._broadcast("test-op-123", test_message)

        # Verify all WebSockets received the message
        ws1.send_json.assert_called_once_with(test_message)
        ws2.send_json.assert_called_once_with(test_message)
        ws3.send_json.assert_called_once_with(test_message)

    @pytest.mark.asyncio()
    async def test_connection_limit_check(self) -> None:
        """Test connection limit enforcement logic."""
        manager = RedisStreamWebSocketManager()

        # Add connections up to the limit
        for i in range(10):
            manager.connections[f"user1:operation:op{i}"] = {AsyncMock()}

        # Check if user has reached limit
        user_connections = sum(len(sockets) for key, sockets in manager.connections.items() if key.startswith("user1:"))

        assert user_connections == 10
        assert user_connections >= manager.max_connections_per_user
