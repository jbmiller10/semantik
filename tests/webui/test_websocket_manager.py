"""Test suite for RedisStreamWebSocketManager."""

import asyncio
import contextlib
import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import redis.asyncio as redis
from fastapi import WebSocket

from packages.webui.websocket_manager import RedisStreamWebSocketManager


class TestRedisStreamWebSocketManager:
    """Test suite for RedisStreamWebSocketManager."""

    @pytest.fixture()
    def mock_redis(self):
        """Create a mock Redis client."""
        # Create a proper async mock that can be awaited
        mock = MagicMock(spec=redis.Redis)

        # Create async mock methods
        async def async_return(value=None):
            return value

        mock.ping = AsyncMock(return_value=True)
        mock.xadd = AsyncMock()
        mock.expire = AsyncMock()
        mock.xrange = AsyncMock(return_value=[])
        mock.xreadgroup = AsyncMock(return_value=[])
        mock.xgroup_create = AsyncMock()
        mock.xack = AsyncMock()
        mock.xgroup_delconsumer = AsyncMock()
        mock.delete = AsyncMock(return_value=1)
        mock.xinfo_groups = AsyncMock(return_value=[])
        mock.xgroup_destroy = AsyncMock()
        mock.close = AsyncMock()

        # Make the mock itself awaitable (for the 'await' in from_url)
        mock.__aenter__ = AsyncMock(return_value=mock)
        mock.__aexit__ = AsyncMock(return_value=None)

        return mock

    @pytest.fixture()
    def mock_websocket(self):
        """Create a mock WebSocket connection."""
        mock = AsyncMock(spec=WebSocket)
        mock.accept = AsyncMock()
        mock.send_json = AsyncMock()
        mock.close = AsyncMock()
        return mock

    @pytest.fixture()
    def manager(self):
        """Create a WebSocket manager instance."""
        return RedisStreamWebSocketManager()

    @pytest.mark.asyncio()
    async def test_startup_success(self, manager, mock_redis):
        """Test successful startup with Redis connection."""

        # Create an async function that returns the mock
        async def async_from_url(*_, **__):
            return mock_redis

        with patch("packages.webui.websocket_manager.redis.from_url", side_effect=async_from_url):
            await manager.startup()

            assert manager.redis is not None
            mock_redis.ping.assert_called_once()

    @pytest.mark.asyncio()
    async def test_startup_retry_logic(self, manager, mock_redis):
        """Test startup retry logic when Redis is initially unavailable."""
        call_count = 0

        async def mock_from_url(*args, **kwargs):  # noqa: ARG001
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Connection failed")
            # Return a mock redis client on the 3rd attempt
            mock_redis.ping = AsyncMock(return_value=True)
            return mock_redis

        with (
            patch("packages.webui.websocket_manager.redis.from_url", side_effect=mock_from_url),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            await manager.startup()

            assert call_count == 3
            assert manager.redis is not None

    @pytest.mark.asyncio()
    async def test_startup_graceful_degradation(self, manager):
        """Test graceful degradation when Redis is completely unavailable."""
        with (
            patch("packages.webui.websocket_manager.redis.from_url", side_effect=Exception("Connection failed")),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            await manager.startup()

            assert manager.redis is None  # Should degrade gracefully

    @pytest.mark.asyncio()
    async def test_shutdown(self, manager, mock_redis, mock_websocket):
        """Test proper shutdown and cleanup."""
        manager.redis = mock_redis

        # Add a connection and consumer task
        manager.connections["user1:operation:operation1"] = {mock_websocket}

        # Create a real asyncio task that can be cancelled and awaited
        async def dummy_coro():
            await asyncio.sleep(10)  # Long sleep that will be cancelled

        mock_task = asyncio.create_task(dummy_coro())

        manager.consumer_tasks["operation1"] = mock_task

        await manager.shutdown()

        # Verify cleanup
        assert mock_task.cancelled()  # Task should be cancelled
        mock_websocket.close.assert_called_once()
        mock_redis.close.assert_called_once()

    @pytest.mark.asyncio()
    async def test_connect_success(self, manager, mock_websocket, mock_redis):
        """Test successful WebSocket connection."""
        manager.redis = mock_redis

        # Mock operation object with proper attributes
        from datetime import UTC, datetime
        from enum import Enum
        from unittest.mock import MagicMock

        # Create mock enums
        class MockStatus(Enum):
            PROCESSING = "processing"

        class MockType(Enum):
            INDEX = "index"

        mock_operation = MagicMock()
        mock_operation.uuid = "operation1"
        mock_operation.status = MockStatus.PROCESSING
        mock_operation.type = MockType.INDEX
        mock_operation.collection_id = "collection1"
        mock_operation.progress = 50
        mock_operation.documents_processed = 5
        mock_operation.total_documents = 10
        mock_operation.created_at = datetime.now(UTC)
        mock_operation.started_at = datetime.now(UTC)
        mock_operation.completed_at = None
        mock_operation.error_message = None

        # Set up the operation getter function
        async def mock_get_operation(operation_id):
            if operation_id == "operation1":
                return mock_operation
            return None

        manager.set_operation_getter(mock_get_operation)

        await manager.connect(mock_websocket, "operation1", "user1")

        # Verify connection accepted
        mock_websocket.accept.assert_called_once()

        # Verify connection stored
        assert mock_websocket in manager.connections["user1:operation:operation1"]

        # Verify current state sent
        mock_websocket.send_json.assert_called()
        sent_data = mock_websocket.send_json.call_args[0][0]
        assert sent_data["type"] == "current_state"
        assert sent_data["data"]["status"] == "processing"

    @pytest.mark.asyncio()
    async def test_connect_connection_limit(self, manager, mock_websocket):
        """Test connection limit enforcement."""
        # Add max connections for user
        for i in range(manager.max_connections_per_user):
            manager.connections[f"user1:operation:operation{i}"] = {AsyncMock()}

        await manager.connect(mock_websocket, "operation_new", "user1")

        # Verify connection rejected
        mock_websocket.close.assert_called_once_with(code=1008, reason="Connection limit exceeded")

    @pytest.mark.asyncio()
    async def test_disconnect(self, manager, mock_websocket):
        """Test WebSocket disconnection handling."""
        # Add connection
        manager.connections["user1:operation:operation1"] = {mock_websocket}

        # Add consumer task
        async def dummy_coro():
            await asyncio.sleep(10)  # Long sleep that will be cancelled

        mock_task = asyncio.create_task(dummy_coro())

        manager.consumer_tasks["operation1"] = mock_task

        await manager.disconnect(mock_websocket, "operation1", "user1")

        # Verify connection removed
        assert "user1:operation:operation1" not in manager.connections

        # Verify consumer task cancelled
        assert mock_task.cancelled()

    @pytest.mark.asyncio()
    async def test_send_operation_update_with_redis(self, manager, mock_redis):
        """Test sending operation update via Redis stream."""
        manager.redis = mock_redis

        update_data = {"progress": 50, "current_file": "test.pdf"}
        await manager.send_update("operation1", "progress", update_data)

        # Verify Redis operations
        mock_redis.xadd.assert_called_once()
        call_args = mock_redis.xadd.call_args
        assert call_args[0][0] == "operation-progress:operation1"  # stream key
        assert "maxlen" in call_args[1]  # keyword argument
        assert call_args[1]["maxlen"] == 1000
        message_data = call_args[0][1]

        # Verify message format
        message = json.loads(message_data["message"])
        assert message["type"] == "progress"
        assert message["data"] == update_data
        assert "timestamp" in message

    @pytest.mark.asyncio()
    async def test_send_operation_update_without_redis(self, manager, mock_websocket):
        """Test fallback to direct broadcast when Redis is unavailable."""
        manager.redis = None
        manager.connections["user1:operation:operation1"] = {mock_websocket}

        update_data = {"status": "completed"}
        await manager.send_update("operation1", "status", update_data)

        # Verify direct broadcast
        mock_websocket.send_json.assert_called_once()
        sent_data = mock_websocket.send_json.call_args[0][0]
        assert sent_data["type"] == "status"
        assert sent_data["data"] == update_data

    @pytest.mark.asyncio()
    async def test_consume_updates(self, manager, mock_redis, mock_websocket):
        """Test consuming updates from Redis stream."""
        manager.redis = mock_redis
        manager.connections["user1:operation:operation1"] = {mock_websocket}

        # Mock stream messages
        test_message = {"timestamp": datetime.now(UTC).isoformat(), "type": "progress", "data": {"progress": 75}}

        mock_redis.xreadgroup.return_value = [
            ("operation-progress:operation1", [("msg-id-1", {"message": json.dumps(test_message)})])
        ]

        # Run consumer for one iteration
        consumer_task = asyncio.create_task(manager._consume_updates("operation1"))
        await asyncio.sleep(0.1)
        consumer_task.cancel()

        with contextlib.suppress(asyncio.CancelledError):
            await consumer_task

        # Verify message sent to WebSocket
        mock_websocket.send_json.assert_called()
        sent_data = mock_websocket.send_json.call_args[0][0]
        assert sent_data["type"] == "progress"
        assert sent_data["data"]["progress"] == 75

        # Verify message acknowledged
        mock_redis.xack.assert_called_with("operation-progress:operation1", manager.consumer_group, "msg-id-1")

    @pytest.mark.asyncio()
    async def test_send_history(self, manager, mock_redis, mock_websocket):
        """Test sending message history to newly connected client."""
        manager.redis = mock_redis

        # Mock historical messages
        historical_messages = [
            (
                "msg-1",
                {
                    "message": json.dumps(
                        {"timestamp": "2024-01-01T00:00:00", "type": "start", "data": {"status": "started"}}
                    )
                },
            ),
            (
                "msg-2",
                {
                    "message": json.dumps(
                        {"timestamp": "2024-01-01T00:01:00", "type": "progress", "data": {"progress": 25}}
                    )
                },
            ),
        ]

        mock_redis.xrange.return_value = historical_messages

        await manager._send_history(mock_websocket, "operation1")

        # Verify all historical messages sent
        assert mock_websocket.send_json.call_count == 2

        # Verify Redis query
        mock_redis.xrange.assert_called_once_with("operation-progress:operation1", min="-", max="+", count=100)

    @pytest.mark.asyncio()
    async def test_broadcast(self, manager):
        """Test broadcasting message to all connections for an operation."""
        # Create multiple WebSocket connections for same operation
        ws1 = AsyncMock(spec=WebSocket)
        ws2 = AsyncMock(spec=WebSocket)
        ws3 = AsyncMock(spec=WebSocket)

        manager.connections["user1:operation:operation1"] = {ws1, ws2}
        manager.connections["user2:operation:operation1"] = {ws3}

        test_message = {"timestamp": datetime.now(UTC).isoformat(), "type": "progress", "data": {"progress": 50}}

        await manager._broadcast("operation1", test_message)

        # Verify all connections received the message
        ws1.send_json.assert_called_once_with(test_message)
        ws2.send_json.assert_called_once_with(test_message)
        ws3.send_json.assert_called_once_with(test_message)

    @pytest.mark.asyncio()
    async def test_broadcast_handles_disconnected_clients(self, manager):
        """Test that broadcast handles and cleans up disconnected clients."""
        # Create WebSocket that will fail
        ws_good = AsyncMock(spec=WebSocket)
        ws_bad = AsyncMock(spec=WebSocket)
        ws_bad.send_json.side_effect = Exception("Connection closed")

        manager.connections["user1:operation:operation1"] = {ws_good, ws_bad}

        test_message = {"type": "test", "data": {}}

        await manager._broadcast("operation1", test_message)

        # Verify good connection still works
        ws_good.send_json.assert_called_once()

        # Verify bad connection was removed
        assert ws_bad not in manager.connections["user1:operation:operation1"]

    @pytest.mark.asyncio()
    async def test_cleanup_operation_stream(self, manager, mock_redis):
        """Test cleaning up Redis stream for completed operation."""
        manager.redis = mock_redis

        # Mock consumer groups
        mock_redis.xinfo_groups.return_value = [{"name": "group1"}, {"name": "group2"}]

        await manager.cleanup_stream("operation1")

        # Verify stream deleted
        mock_redis.delete.assert_called_once_with("operation-progress:operation1")

        # Verify consumer groups destroyed
        assert mock_redis.xgroup_destroy.call_count == 2
        mock_redis.xgroup_destroy.assert_any_call("operation-progress:operation1", "group1")
        mock_redis.xgroup_destroy.assert_any_call("operation-progress:operation1", "group2")

    @pytest.mark.asyncio()
    async def test_cleanup_operation_stream_without_redis(self, manager):
        """Test cleanup gracefully handles missing Redis."""
        manager.redis = None

        # Should not raise exception
        await manager.cleanup_stream("operation1")

    @pytest.mark.asyncio()
    async def test_concurrent_connections(self, manager, mock_redis):
        """Test handling multiple concurrent connections for same operation."""
        manager.redis = mock_redis

        # Create multiple WebSocket connections
        websockets = [AsyncMock(spec=WebSocket) for _ in range(5)]

        # Connect all websockets concurrently
        with patch("packages.shared.database.factory.create_operation_repository") as mock_create_repo:
            mock_repo = AsyncMock()
            mock_operation = MagicMock()
            mock_operation.status = "processing"
            mock_repo.get_by_uuid = AsyncMock(return_value=mock_operation)
            mock_create_repo.return_value = mock_repo

            connect_tasks = [manager.connect(ws, "operation1", f"user{i}") for i, ws in enumerate(websockets)]
            await asyncio.gather(*connect_tasks)

        # Verify all connections established
        total_connections = sum(len(sockets) for sockets in manager.connections.values())
        assert total_connections == 5

        # Verify only one consumer task created
        assert len(manager.consumer_tasks) == 1
        assert "operation1" in manager.consumer_tasks
