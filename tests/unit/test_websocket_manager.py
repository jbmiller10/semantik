#!/usr/bin/env python3
"""
Comprehensive test suite for webui/websocket_manager.py
Tests connection management, message routing, error handling, and reconnection logic
"""

import asyncio
import json
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock, call, patch

import pytest
import redis.asyncio as redis
from fastapi import WebSocket
from webui.websocket_manager import RedisStreamWebSocketManager


class TestWebSocketManager:
    """Test RedisStreamWebSocketManager implementation"""

    @pytest.fixture()
    def ws_manager(self):
        """Create WebSocketManager instance"""
        manager = RedisStreamWebSocketManager()
        # Reset state for testing
        manager.redis = None
        manager.connections = {}
        manager.consumer_tasks = {}
        manager._startup_attempted = False
        manager._get_operation_func = None
        return manager

    @pytest.fixture()
    def mock_redis(self):
        """Create mock Redis client"""
        mock = AsyncMock(spec=redis.Redis)
        mock.ping = AsyncMock()
        mock.xadd = AsyncMock()
        mock.expire = AsyncMock()
        mock.xinfo_stream = AsyncMock()
        mock.xgroup_create = AsyncMock()
        mock.xreadgroup = AsyncMock()
        mock.xack = AsyncMock()
        mock.xgroup_delconsumer = AsyncMock()
        mock.xrange = AsyncMock()
        mock.delete = AsyncMock()
        mock.xinfo_groups = AsyncMock()
        mock.xgroup_destroy = AsyncMock()
        mock.close = AsyncMock()
        return mock

    @pytest.fixture()
    def mock_websocket(self):
        """Create mock WebSocket"""
        ws = AsyncMock(spec=WebSocket)
        ws.accept = AsyncMock()
        ws.send_json = AsyncMock()
        ws.close = AsyncMock()
        return ws

    @pytest.fixture()
    def mock_operation(self):
        """Create mock operation"""
        operation = Mock()
        operation.id = "op-123"
        operation.uuid = "op-uuid-123"
        operation.status = Mock(value="processing")
        operation.type = Mock(value="index")
        operation.created_at = datetime.now(UTC)
        operation.started_at = datetime.now(UTC)
        operation.completed_at = None
        operation.error_message = None
        return operation

    @pytest.mark.asyncio()
    @patch("webui.websocket_manager.redis.from_url")
    async def test_startup_success(self, mock_redis_from_url, ws_manager, mock_redis):
        """Test successful startup and Redis connection"""
        mock_redis_from_url.return_value = mock_redis

        await ws_manager.startup()

        assert ws_manager.redis == mock_redis
        assert ws_manager._startup_attempted is True
        mock_redis.ping.assert_called_once()
        mock_redis_from_url.assert_called_once()

    @pytest.mark.asyncio()
    @patch("webui.websocket_manager.redis.from_url")
    @patch("webui.websocket_manager.asyncio.sleep")
    async def test_startup_retry_on_failure(self, mock_sleep, mock_redis_from_url, ws_manager):
        """Test startup retry logic on connection failure"""
        # First two attempts fail, third succeeds
        mock_redis_success = AsyncMock(spec=redis.Redis)
        mock_redis_success.ping = AsyncMock()

        mock_redis_from_url.side_effect = [
            Exception("Connection failed"),
            Exception("Connection failed"),
            mock_redis_success,
        ]

        await ws_manager.startup()

        assert ws_manager.redis == mock_redis_success
        assert mock_redis_from_url.call_count == 3
        assert mock_sleep.call_count == 2  # Two retries

    @pytest.mark.asyncio()
    @patch("webui.websocket_manager.redis.from_url")
    @patch("webui.websocket_manager.logger")
    async def test_startup_failure_after_max_retries(self, mock_logger, mock_redis_from_url, ws_manager):
        """Test graceful degradation when Redis connection fails"""
        mock_redis_from_url.side_effect = Exception("Connection failed")

        await ws_manager.startup()

        assert ws_manager.redis is None
        assert ws_manager._startup_attempted is True
        mock_logger.error.assert_called()

    @pytest.mark.asyncio()
    async def test_shutdown(self, ws_manager, mock_redis, mock_websocket):
        """Test clean shutdown of manager"""
        # Setup manager state
        ws_manager.redis = mock_redis
        ws_manager.connections = {"user:123:operation:op-1": {mock_websocket}}

        # Create and add a consumer task
        async def dummy_consumer():
            await asyncio.sleep(10)

        task = asyncio.create_task(dummy_consumer())
        ws_manager.consumer_tasks = {"op-1": task}

        # Test shutdown
        await ws_manager.shutdown()

        # Verify cleanup
        assert task.cancelled()
        mock_websocket.close.assert_called_once()
        mock_redis.close.assert_called_once()

    @pytest.mark.asyncio()
    async def test_connect_success(self, ws_manager, mock_redis, mock_websocket, mock_operation):
        """Test successful WebSocket connection"""
        ws_manager.redis = mock_redis
        ws_manager._get_operation_func = AsyncMock(return_value=mock_operation)

        # Mock Redis stream responses
        mock_redis.xrange.return_value = []

        await ws_manager.connect(mock_websocket, "op-123", "user-123")

        # Verify WebSocket accepted
        mock_websocket.accept.assert_called_once()

        # Verify connection stored
        key = "user-123:operation:op-123"
        assert key in ws_manager.connections
        assert mock_websocket in ws_manager.connections[key]

        # Verify current state sent
        mock_websocket.send_json.assert_called()
        sent_data = mock_websocket.send_json.call_args[0][0]
        assert sent_data["type"] == "current_state"
        assert sent_data["data"]["status"] == "processing"

        # Verify consumer task started
        assert "op-123" in ws_manager.consumer_tasks

    @pytest.mark.asyncio()
    async def test_connect_without_redis(self, ws_manager, mock_websocket, mock_operation):
        """Test connection when Redis is not available"""
        ws_manager.redis = None
        ws_manager._get_operation_func = AsyncMock(return_value=mock_operation)

        await ws_manager.connect(mock_websocket, "op-123", "user-123")

        # Should still accept connection
        mock_websocket.accept.assert_called_once()

        # Should send current state
        mock_websocket.send_json.assert_called()

        # Should not start consumer task
        assert len(ws_manager.consumer_tasks) == 0

    @pytest.mark.asyncio()
    async def test_connect_connection_limit(self, ws_manager, mock_websocket):
        """Test connection limit enforcement"""
        # Create max connections for user
        for i in range(ws_manager.max_connections_per_user):
            ws_manager.connections[f"user-123:operation:op-{i}"] = {Mock()}

        await ws_manager.connect(mock_websocket, "op-new", "user-123")

        # Should reject connection
        mock_websocket.close.assert_called_once_with(code=1008, reason="Connection limit exceeded")
        mock_websocket.accept.assert_not_called()

    @pytest.mark.asyncio()
    async def test_disconnect(self, ws_manager, mock_websocket):
        """Test WebSocket disconnection"""
        # Setup connection
        key = "user-123:operation:op-123"
        ws_manager.connections[key] = {mock_websocket, Mock()}

        # Create consumer task
        task = asyncio.create_task(asyncio.sleep(10))
        ws_manager.consumer_tasks["op-123"] = task

        # Test disconnect
        await ws_manager.disconnect(mock_websocket, "op-123", "user-123")

        # Verify connection removed
        assert mock_websocket not in ws_manager.connections[key]

        # Task should remain as another connection exists
        assert "op-123" in ws_manager.consumer_tasks
        assert not task.cancelled()

    @pytest.mark.asyncio()
    async def test_disconnect_last_connection(self, ws_manager, mock_websocket):
        """Test disconnection of last connection for operation"""
        # Setup single connection
        key = "user-123:operation:op-123"
        ws_manager.connections[key] = {mock_websocket}

        # Create consumer task
        task = asyncio.create_task(asyncio.sleep(10))
        ws_manager.consumer_tasks["op-123"] = task

        # Test disconnect
        await ws_manager.disconnect(mock_websocket, "op-123", "user-123")

        # Verify connection removed
        assert key not in ws_manager.connections

        # Task should be cancelled
        assert task.cancelled()
        assert "op-123" not in ws_manager.consumer_tasks

    @pytest.mark.asyncio()
    async def test_send_update_with_redis(self, ws_manager, mock_redis):
        """Test sending update through Redis stream"""
        ws_manager.redis = mock_redis

        await ws_manager.send_update(
            operation_id="op-123",
            update_type="progress",
            data={"percentage": 50, "message": "Processing..."},
        )

        # Verify Redis stream operations
        mock_redis.xadd.assert_called_once()
        call_args = mock_redis.xadd.call_args
        assert call_args[0][0] == "operation-progress:op-123"
        assert "message" in call_args[0][1]
        assert call_args[1]["maxlen"] == 1000

        mock_redis.expire.assert_called_once_with("operation-progress:op-123", 86400)

    @pytest.mark.asyncio()
    async def test_send_update_without_redis(self, ws_manager, mock_websocket):
        """Test direct broadcast when Redis is not available"""
        ws_manager.redis = None
        ws_manager.connections = {"user-123:operation:op-123": {mock_websocket}}

        await ws_manager.send_update(
            operation_id="op-123",
            update_type="progress",
            data={"percentage": 50},
        )

        # Should broadcast directly
        mock_websocket.send_json.assert_called_once()
        sent_data = mock_websocket.send_json.call_args[0][0]
        assert sent_data["type"] == "progress"
        assert sent_data["data"]["percentage"] == 50

    @pytest.mark.asyncio()
    @patch("webui.websocket_manager.logger")
    async def test_send_update_redis_failure(self, mock_logger, ws_manager, mock_redis, mock_websocket):
        """Test fallback to direct broadcast on Redis failure"""
        ws_manager.redis = mock_redis
        ws_manager.connections = {"user-123:operation:op-123": {mock_websocket}}

        # Make Redis fail
        mock_redis.xadd.side_effect = Exception("Redis error")

        await ws_manager.send_update(
            operation_id="op-123",
            update_type="error",
            data={"message": "Something went wrong"},
        )

        # Should log error and broadcast directly
        mock_logger.error.assert_called()
        mock_websocket.send_json.assert_called_once()

    @pytest.mark.asyncio()
    async def test_consume_updates_lifecycle(self, ws_manager, mock_redis):
        """Test consumer lifecycle for processing updates"""
        ws_manager.redis = mock_redis

        # Mock stream exists
        mock_redis.xinfo_stream.return_value = {"length": 5}

        # Mock consumer group creation success
        mock_redis.xgroup_create.return_value = None

        # Mock messages to consume
        test_messages = [
            [
                (
                    "operation-progress:op-123",
                    [
                        (
                            "msg-1",
                            {
                                "message": json.dumps(
                                    {
                                        "type": "progress",
                                        "data": {"percentage": 25},
                                    }
                                )
                            },
                        ),
                        (
                            "msg-2",
                            {
                                "message": json.dumps(
                                    {
                                        "type": "status_update",
                                        "data": {"status": "completed"},
                                    }
                                )
                            },
                        ),
                    ],
                )
            ]
        ]

        # Return messages once, then empty
        mock_redis.xreadgroup.side_effect = [test_messages, [], asyncio.CancelledError()]

        # Setup WebSocket connection
        mock_websocket = Mock()
        mock_websocket.send_json = AsyncMock()
        ws_manager.connections = {"user-123:operation:op-123": {mock_websocket}}

        # Run consumer
        task = asyncio.create_task(ws_manager._consume_updates("op-123"))

        # Let it process
        await asyncio.sleep(0.1)

        # Cancel task
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # Verify messages were broadcast
        assert mock_websocket.send_json.call_count >= 2

        # Verify messages were acknowledged
        assert mock_redis.xack.call_count == 2

        # Verify consumer cleanup attempted
        mock_redis.xgroup_delconsumer.assert_called()

    @pytest.mark.asyncio()
    async def test_consume_updates_stream_not_exists(self, ws_manager, mock_redis):
        """Test consumer behavior when stream doesn't exist yet"""
        ws_manager.redis = mock_redis

        # Stream doesn't exist
        mock_redis.xinfo_stream.side_effect = Exception("Stream does not exist")

        # Should retry after delay
        call_count = 0

        async def track_calls(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count > 2:
                raise asyncio.CancelledError()
            await asyncio.sleep(0.01)

        mock_redis.xreadgroup.side_effect = track_calls

        task = asyncio.create_task(ws_manager._consume_updates("op-123"))

        with pytest.raises(asyncio.CancelledError):
            await task

        # Should have attempted multiple times
        assert call_count > 1

    @pytest.mark.asyncio()
    async def test_send_history(self, ws_manager, mock_redis, mock_websocket):
        """Test sending message history to new connection"""
        ws_manager.redis = mock_redis

        # Mock historical messages
        history_messages = [
            (
                "msg-1",
                {
                    "message": json.dumps(
                        {"type": "progress", "data": {"percentage": 10}}
                    )
                },
            ),
            (
                "msg-2",
                {
                    "message": json.dumps(
                        {"type": "progress", "data": {"percentage": 20}}
                    )
                },
            ),
        ]
        mock_redis.xrange.return_value = history_messages

        await ws_manager._send_history(mock_websocket, "op-123")

        # Verify all historical messages sent
        assert mock_websocket.send_json.call_count == 2

        # Verify correct stream key used
        mock_redis.xrange.assert_called_once_with(
            "operation-progress:op-123", min="-", max="+", count=100
        )

    @pytest.mark.asyncio()
    async def test_broadcast_to_multiple_connections(self, ws_manager):
        """Test broadcasting to multiple WebSocket connections"""
        # Create multiple WebSockets
        websockets = []
        for i in range(3):
            ws = Mock()
            ws.send_json = AsyncMock()
            websockets.append(ws)

        # Setup connections
        ws_manager.connections = {
            "user-123:operation:op-123": {websockets[0], websockets[1]},
            "user-456:operation:op-123": {websockets[2]},
        }

        # Test broadcast
        message = {"type": "progress", "data": {"percentage": 75}}
        await ws_manager._broadcast("op-123", message)

        # All WebSockets should receive message
        for ws in websockets:
            ws.send_json.assert_called_once_with(message)

    @pytest.mark.asyncio()
    async def test_broadcast_with_failed_connections(self, ws_manager):
        """Test broadcast handling when some connections fail"""
        # Create WebSockets with one that fails
        ws_success = Mock()
        ws_success.send_json = AsyncMock()

        ws_fail = Mock()
        ws_fail.send_json = AsyncMock(side_effect=Exception("Connection lost"))

        ws_manager.connections = {
            "user-123:operation:op-123": {ws_success, ws_fail}
        }

        # Test broadcast
        message = {"type": "error", "data": {"message": "Test error"}}
        await ws_manager._broadcast("op-123", message)

        # Successful WebSocket should receive message
        ws_success.send_json.assert_called_once()

        # Failed connection should be removed
        assert ws_fail not in ws_manager.connections["user-123:operation:op-123"]

    @pytest.mark.asyncio()
    async def test_close_connections_on_completion(self, ws_manager):
        """Test closing connections when operation completes"""
        # Create WebSockets
        websockets = []
        for i in range(2):
            ws = Mock()
            ws.close = AsyncMock()
            websockets.append(ws)

        ws_manager.connections = {
            "user-123:operation:op-123": {websockets[0]},
            "user-456:operation:op-123": {websockets[1]},
        }

        await ws_manager._close_connections("op-123")

        # All connections should be closed
        for ws in websockets:
            ws.close.assert_called_once_with(code=1000, reason="Operation completed")

        # Connections should be removed
        assert len(ws_manager.connections) == 0

    @pytest.mark.asyncio()
    async def test_cleanup_stream(self, ws_manager, mock_redis):
        """Test Redis stream cleanup"""
        ws_manager.redis = mock_redis

        # Mock stream groups
        mock_redis.xinfo_groups.return_value = [
            {"name": "group1"},
            {"name": "group2"},
        ]

        await ws_manager.cleanup_stream("op-123")

        # Verify stream deleted
        mock_redis.delete.assert_called_once_with("operation-progress:op-123")

        # Verify groups deleted
        assert mock_redis.xgroup_destroy.call_count == 2

    @pytest.mark.asyncio()
    async def test_cleanup_stream_without_redis(self, ws_manager):
        """Test cleanup when Redis is not available"""
        ws_manager.redis = None

        # Should not raise error
        await ws_manager.cleanup_stream("op-123")

    @pytest.mark.asyncio()
    async def test_set_operation_getter(self, ws_manager):
        """Test dependency injection for operation getter"""
        mock_getter = AsyncMock()
        ws_manager.set_operation_getter(mock_getter)

        assert ws_manager._get_operation_func == mock_getter

    @pytest.mark.asyncio()
    @patch("webui.websocket_manager.AsyncSessionLocal")
    async def test_default_operation_getter(self, mock_session_local, ws_manager, mock_websocket, mock_operation):
        """Test default operation getter when not injected"""
        ws_manager.redis = Mock()

        # Mock database session and repository
        mock_session = AsyncMock()
        mock_session_local.return_value.__aenter__.return_value = mock_session

        with patch("webui.websocket_manager.OperationRepository") as mock_repo_class:
            mock_repo = AsyncMock()
            mock_repo.get_by_uuid.return_value = mock_operation
            mock_repo_class.return_value = mock_repo

            await ws_manager.connect(mock_websocket, "op-123", "user-123")

            # Verify default implementation was used
            mock_repo.get_by_uuid.assert_called_once_with("op-123")


class TestWebSocketManagerErrorHandling:
    """Test error handling scenarios"""

    @pytest.fixture()
    def ws_manager(self):
        manager = RedisStreamWebSocketManager()
        manager.redis = None
        manager.connections = {}
        manager.consumer_tasks = {}
        return manager

    @pytest.mark.asyncio()
    @patch("webui.websocket_manager.logger")
    async def test_consumer_error_recovery(self, mock_logger, ws_manager):
        """Test consumer error recovery"""
        mock_redis = AsyncMock()
        ws_manager.redis = mock_redis

        # Simulate various errors
        error_sequence = [
            Exception("NOGROUP No such consumer group"),
            Exception("Random error"),
            asyncio.CancelledError(),
        ]

        mock_redis.xreadgroup.side_effect = error_sequence

        task = asyncio.create_task(ws_manager._consume_updates("op-123"))

        # Let it process errors
        await asyncio.sleep(0.1)

        # Cancel task
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # Should have logged errors
        assert mock_logger.error.call_count >= 1

    @pytest.mark.asyncio()
    async def test_malformed_message_handling(self, ws_manager):
        """Test handling of malformed messages in stream"""
        mock_redis = AsyncMock()
        ws_manager.redis = mock_redis

        # Mock malformed message
        bad_messages = [
            [
                (
                    "operation-progress:op-123",
                    [
                        ("msg-bad", {"message": "not valid json"}),
                    ],
                )
            ]
        ]

        mock_redis.xreadgroup.side_effect = [bad_messages, asyncio.CancelledError()]
        mock_redis.xinfo_stream.return_value = {"length": 1}

        # Setup connection
        mock_ws = Mock()
        mock_ws.send_json = AsyncMock()
        ws_manager.connections = {"user:123:operation:op-123": {mock_ws}}

        task = asyncio.create_task(ws_manager._consume_updates("op-123"))

        with pytest.raises(asyncio.CancelledError):
            await task

        # Message should still be acknowledged despite error
        mock_redis.xack.assert_called()


class TestWebSocketManagerIntegration:
    """Test integration scenarios"""

    @pytest.mark.asyncio()
    async def test_complete_operation_flow(self):
        """Test complete flow from connection to operation completion"""
        ws_manager = RedisStreamWebSocketManager()
        mock_redis = AsyncMock()
        ws_manager.redis = mock_redis

        # Mock operation
        mock_operation = Mock()
        mock_operation.status = Mock(value="pending")
        mock_operation.type = Mock(value="index")
        mock_operation.created_at = datetime.now(UTC)
        mock_operation.started_at = None
        mock_operation.completed_at = None
        mock_operation.error_message = None

        ws_manager._get_operation_func = AsyncMock(return_value=mock_operation)

        # Mock WebSocket
        mock_ws = AsyncMock()

        # Connect
        await ws_manager.connect(mock_ws, "op-123", "user-123")

        # Send progress updates
        await ws_manager.send_update(
            "op-123",
            "progress",
            {"percentage": 50, "message": "Processing documents"},
        )

        # Complete operation
        await ws_manager.send_update(
            "op-123",
            "status_update",
            {"status": "completed", "message": "Operation completed successfully"},
        )

        # Verify WebSocket accepted and received updates
        mock_ws.accept.assert_called_once()
        assert mock_ws.send_json.call_count >= 1  # At least current state

        # Cleanup
        await ws_manager.cleanup_stream("op-123")
        await ws_manager.shutdown()

    @pytest.mark.asyncio()
    async def test_concurrent_operations(self):
        """Test managing multiple concurrent operations"""
        ws_manager = RedisStreamWebSocketManager()
        mock_redis = AsyncMock()
        ws_manager.redis = mock_redis

        # Create multiple operations
        operations = []
        websockets = []

        for i in range(3):
            mock_op = Mock()
            mock_op.uuid = f"op-{i}"
            mock_op.status = Mock(value="processing")
            mock_op.type = Mock(value="index")
            mock_op.created_at = datetime.now(UTC)
            mock_op.started_at = datetime.now(UTC)
            mock_op.completed_at = None
            mock_op.error_message = None
            operations.append(mock_op)

            mock_ws = AsyncMock()
            websockets.append(mock_ws)

        ws_manager._get_operation_func = AsyncMock(side_effect=operations)

        # Connect all operations
        for i, (op, ws) in enumerate(zip(operations, websockets)):
            await ws_manager.connect(ws, op.uuid, f"user-{i}")

        # Verify all connections established
        assert len(ws_manager.connections) == 3
        assert len(ws_manager.consumer_tasks) == 3

        # Send updates to each operation
        for i in range(3):
            await ws_manager.send_update(
                f"op-{i}",
                "progress",
                {"percentage": (i + 1) * 25},
            )

        # Disconnect all
        for i, (op, ws) in enumerate(zip(operations, websockets)):
            await ws_manager.disconnect(ws, op.uuid, f"user-{i}")

        # Verify cleanup
        assert len(ws_manager.connections) == 0
        assert len(ws_manager.consumer_tasks) == 0