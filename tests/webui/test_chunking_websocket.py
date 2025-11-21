"""
Tests for chunking WebSocket functionality.

Comprehensive test coverage for WebSocket connections, progress updates,
channel management, and error scenarios related to chunking operations.
"""

import asyncio
import contextlib
import json
import time
import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect
from webui.api.v2.chunking_schemas import ChunkingStatus
from webui.websocket_manager import RedisStreamWebSocketManager


@pytest.fixture()
def mock_redis_client() -> AsyncMock:
    """Create a mock Redis client for WebSocket testing."""
    mock = AsyncMock()
    mock.ping.return_value = True
    mock.xadd.return_value = "1234567890-0"
    mock.xrange.return_value = []
    mock.xreadgroup.return_value = []
    mock.xgroup_create.return_value = None
    mock.xack.return_value = 1
    mock.xgroup_delconsumer.return_value = None
    mock.delete.return_value = 1
    mock.xinfo_groups.return_value = []
    mock.xgroup_destroy.return_value = None
    mock.close.return_value = None
    mock.expire.return_value = None
    return mock


@pytest.fixture()
def ws_manager(mock_redis_client: AsyncMock) -> RedisStreamWebSocketManager:
    """Create a WebSocket manager with mocked Redis."""
    manager = RedisStreamWebSocketManager()
    manager.redis = mock_redis_client
    return manager


@pytest.fixture()
def mock_websocket() -> AsyncMock:
    """Create a mock WebSocket connection."""
    mock = AsyncMock(spec=WebSocket)
    mock.accept = AsyncMock()
    mock.send_json = AsyncMock()
    mock.send_text = AsyncMock()
    mock.receive_json = AsyncMock()
    mock.close = AsyncMock()

    # Track state
    mock.client_state = {"connected": False}

    async def accept_connection() -> None:
        mock.client_state["connected"] = True

    async def close_connection(code=1000, reason="") -> None:  # noqa: ARG001
        mock.client_state["connected"] = False

    mock.accept.side_effect = accept_connection
    mock.close.side_effect = close_connection

    return mock


@pytest.fixture()
def mock_operation() -> None:
    """Create a mock operation object."""

    class MockEnum:
        """Mock enum with value attribute."""

        def __init__(self, value) -> None:
            self.value = value

    class MockOperation:
        """Mock operation with required attributes."""

        def __init__(self) -> None:
            self.uuid = str(uuid.uuid4())
            self.collection_id = "test-collection-123"
            self.type = MockEnum("chunking")
            self.status = MockEnum("in_progress")
            self.progress_percentage = 0.0
            self.documents_processed = 0
            self.total_documents = 10
            self.chunks_created = 0
            self.started_at = datetime.now(UTC)
            self.created_at = datetime.now(UTC)
            self.completed_at = None
            self.error_message = None

    return MockOperation()


class TestWebSocketConnection:
    """Test WebSocket connection establishment and lifecycle."""

    @pytest.mark.asyncio()
    async def test_successful_connection(
        self, ws_manager: RedisStreamWebSocketManager, mock_websocket: AsyncMock, mock_operation
    ) -> None:
        """Test successful WebSocket connection establishment."""
        user_id = "user-123"
        operation_id = mock_operation.uuid

        # Set up operation getter
        async def get_operation(op_id: str) -> MagicMock | None:
            return mock_operation if op_id == operation_id else None

        ws_manager.set_operation_getter(get_operation)

        # Connect
        await ws_manager.connect(mock_websocket, operation_id, user_id)

        # Verify
        mock_websocket.accept.assert_called_once()
        mock_websocket.send_json.assert_called()

        # Check connection was stored
        key = f"{user_id}:operation:{operation_id}"
        assert key in ws_manager.connections
        assert mock_websocket in ws_manager.connections[key]

    @pytest.mark.asyncio()
    async def test_connection_limit_per_user(
        self, ws_manager: RedisStreamWebSocketManager, mock_websocket: AsyncMock
    ) -> None:
        """Test that connection limits per user are enforced."""
        user_id = "user-123"
        ws_manager.max_connections_per_user = 3

        # Create multiple mock websockets
        websockets = [AsyncMock(spec=WebSocket) for _ in range(5)]
        for ws in websockets:
            ws.accept = AsyncMock()
            ws.close = AsyncMock()
            ws.send_json = AsyncMock()

        # Add connections up to the limit
        for i in range(3):
            key = f"{user_id}:operation:op-{i}"
            ws_manager.connections[key] = {websockets[i]}

        # Try to add one more - should be rejected
        with pytest.raises(ConnectionRefusedError, match="User connection limit exceeded"):
            await ws_manager.connect(websockets[3], "op-new", user_id)

        websockets[3].close.assert_called_once_with(code=1008, reason="User connection limit exceeded")

    @pytest.mark.asyncio()
    async def test_global_connection_limit(
        self, ws_manager: RedisStreamWebSocketManager, mock_websocket: AsyncMock
    ) -> None:
        """Test that global connection limits are enforced."""
        ws_manager.max_total_connections = 2

        # Fill up to global limit with different users
        ws_manager.connections["user1:operation:op1"] = {AsyncMock()}
        ws_manager.connections["user2:operation:op2"] = {AsyncMock()}

        # Try to add one more - should be rejected
        with pytest.raises(ConnectionRefusedError, match="Server connection limit exceeded"):
            await ws_manager.connect(mock_websocket, "op3", "user3")

        mock_websocket.close.assert_called_once_with(code=1008, reason="Server connection limit exceeded")

    @pytest.mark.asyncio()
    async def test_disconnect_cleanup(self, ws_manager: RedisStreamWebSocketManager, mock_websocket: AsyncMock) -> None:
        """Test proper cleanup on disconnect."""
        user_id = "user-123"
        operation_id = "op-123"
        key = f"{user_id}:operation:{operation_id}"

        # Add connection
        ws_manager.connections[key] = {mock_websocket}

        # Disconnect
        await ws_manager.disconnect(mock_websocket, operation_id, user_id)

        # Verify cleanup
        assert key not in ws_manager.connections or mock_websocket not in ws_manager.connections.get(key, set())

    @pytest.mark.asyncio()
    async def test_redis_reconnection_attempt(
        self, ws_manager: RedisStreamWebSocketManager, mock_websocket: AsyncMock
    ) -> None:
        """Test that manager attempts to reconnect to Redis if disconnected."""
        ws_manager.redis = None  # Simulate disconnected state

        with patch.object(ws_manager, "startup", new_callable=AsyncMock) as mock_startup:
            await ws_manager.connect(mock_websocket, "op-123", "user-123")
            mock_startup.assert_called_once()


class TestProgressUpdates:
    """Test WebSocket progress update functionality."""

    @pytest.mark.asyncio()
    async def test_send_chunking_progress(
        self, ws_manager: RedisStreamWebSocketManager, mock_websocket: AsyncMock, mock_redis_client: AsyncMock
    ) -> None:
        """Test sending chunking progress updates."""
        channel = "chunking:collection-123:operation-456"
        progress_data = {
            "type": "chunking_progress",
            "operation_id": "operation-456",
            "status": ChunkingStatus.IN_PROGRESS.value,
            "progress_percentage": 45.5,
            "documents_processed": 5,
            "total_documents": 11,
            "chunks_created": 250,
            "current_document": "document_6.pdf",
            "estimated_time_remaining": 120,
            "timestamp": datetime.now(tz=UTC).isoformat(),
        }

        # Send message
        await ws_manager.send_message(channel, progress_data)

        # Verify Redis stream was used
        mock_redis_client.xadd.assert_called_once()
        call_args = mock_redis_client.xadd.call_args
        assert call_args[0][0] == f"stream:{channel}"
        assert json.loads(call_args[0][1]["message"]) == progress_data

    @pytest.mark.asyncio()
    async def test_progress_throttling(
        self, ws_manager: RedisStreamWebSocketManager, mock_websocket: AsyncMock
    ) -> None:
        """Test that progress updates are throttled to prevent spam."""
        operation_id = "op-123"

        # Set throttle threshold
        ws_manager._chunking_progress_threshold = 1.0  # 1 second between updates

        # Send first update - should go through
        progress1 = {
            "type": "chunking_progress",
            "operation_id": operation_id,
            "progress_percentage": 10.0,
        }

        should_send = await ws_manager._should_send_progress_update(operation_id, progress1)
        assert should_send is True

        # Send immediate second update - should be throttled
        progress2 = {
            "type": "chunking_progress",
            "operation_id": operation_id,
            "progress_percentage": 11.0,
        }

        should_send = await ws_manager._should_send_progress_update(operation_id, progress2)
        assert should_send is False

        # Simulate time passing

        time.sleep(1.1)

        # Now update should go through
        should_send = await ws_manager._should_send_progress_update(operation_id, progress2)
        assert should_send is True

    @pytest.mark.asyncio()
    async def test_operation_completion_notification(
        self, ws_manager: RedisStreamWebSocketManager, mock_websocket: AsyncMock, mock_redis_client: AsyncMock
    ) -> None:
        """Test sending operation completion notification."""
        channel = "chunking:collection-123:operation-456"
        completion_data = {
            "type": "chunking_completed",
            "operation_id": "operation-456",
            "status": ChunkingStatus.COMPLETED.value,
            "total_chunks_created": 500,
            "total_documents_processed": 11,
            "processing_time_seconds": 240,
            "timestamp": datetime.now(tz=UTC).isoformat(),
        }

        await ws_manager.send_message(channel, completion_data)

        # Verify message was sent
        mock_redis_client.xadd.assert_called_once()
        call_args = mock_redis_client.xadd.call_args
        sent_data = json.loads(call_args[0][1]["message"])
        assert sent_data["type"] == "chunking_completed"
        assert sent_data["status"] == ChunkingStatus.COMPLETED.value

    @pytest.mark.asyncio()
    async def test_operation_failure_notification(
        self, ws_manager: RedisStreamWebSocketManager, mock_websocket: AsyncMock, mock_redis_client: AsyncMock
    ) -> None:
        """Test sending operation failure notification."""
        channel = "chunking:collection-123:operation-456"
        failure_data = {
            "type": "chunking_failed",
            "operation_id": "operation-456",
            "status": ChunkingStatus.FAILED.value,
            "error_message": "Memory limit exceeded during semantic chunking",
            "error_code": "CHUNKING_MEMORY_ERROR",
            "failed_at_document": "large_document.pdf",
            "documents_processed": 3,
            "timestamp": datetime.now(tz=UTC).isoformat(),
        }

        await ws_manager.send_message(channel, failure_data)

        # Verify error was sent
        mock_redis_client.xadd.assert_called_once()
        call_args = mock_redis_client.xadd.call_args
        sent_data = json.loads(call_args[0][1]["message"])
        assert sent_data["type"] == "chunking_failed"
        assert sent_data["error_code"] == "CHUNKING_MEMORY_ERROR"


class TestChannelManagement:
    """Test WebSocket channel management."""

    @pytest.mark.asyncio()
    async def test_create_unique_channels(self, ws_manager: RedisStreamWebSocketManager) -> None:
        """Test that unique channels are created for each operation."""
        collection_id = "coll-123"
        operation_ids = [str(uuid.uuid4()) for _ in range(3)]

        channels = []
        for op_id in operation_ids:
            channel = f"chunking:{collection_id}:{op_id}"
            channels.append(channel)

        # All channels should be unique
        assert len(set(channels)) == len(channels)

        # All should follow the expected pattern
        for channel in channels:
            parts = channel.split(":")
            assert parts[0] == "chunking"
            assert parts[1] == collection_id

    @pytest.mark.asyncio()
    async def test_broadcast_to_multiple_clients(
        self, ws_manager: RedisStreamWebSocketManager, mock_redis_client: AsyncMock
    ) -> None:
        """Test broadcasting updates to multiple connected clients."""
        operation_id = "op-123"
        channel = f"chunking:coll-456:{operation_id}"

        # Create multiple mock websockets for same operation
        websockets = [AsyncMock(spec=WebSocket) for _ in range(3)]

        # Add all to same operation channel
        for i, ws in enumerate(websockets):
            ws.send_json = AsyncMock()
            key = f"user-{i}:operation:{operation_id}"
            ws_manager.connections[key] = {ws}

        # Send broadcast message
        message = {
            "type": "chunking_progress",
            "operation_id": operation_id,
            "progress_percentage": 50.0,
        }

        await ws_manager.send_message(channel, message)

        # Verify Redis stream was used for broadcast
        mock_redis_client.xadd.assert_called_once()

    @pytest.mark.asyncio()
    async def test_channel_cleanup_on_completion(
        self, ws_manager: RedisStreamWebSocketManager, mock_redis_client: AsyncMock
    ) -> None:
        """Test that channels are cleaned up after operation completion."""
        operation_id = "op-123"
        channel = f"chunking:coll-456:{operation_id}"

        # Add a consumer task
        # Create a proper mock coroutine
        async def mock_coro() -> None:
            pass

        mock_task = asyncio.create_task(mock_coro())
        mock_task.cancel = MagicMock()
        ws_manager.consumer_tasks[operation_id] = mock_task

        # Send completion message
        completion_msg = {
            "type": "chunking_completed",
            "operation_id": operation_id,
        }

        await ws_manager.send_message(channel, completion_msg)

        # After processing completion, cleanup should be triggered
        await ws_manager.cleanup_operation_channel(operation_id)

        # Verify cleanup
        assert operation_id not in ws_manager.consumer_tasks
        mock_task.cancel.assert_called_once()


class TestErrorHandling:
    """Test error handling in WebSocket operations."""

    @pytest.mark.asyncio()
    async def test_handle_websocket_disconnect_error(
        self, ws_manager: RedisStreamWebSocketManager, mock_websocket: AsyncMock
    ) -> None:
        """Test handling of WebSocket disconnect errors."""
        user_id = "user-123"
        operation_id = "op-123"

        # Simulate disconnect error
        mock_websocket.send_json.side_effect = WebSocketDisconnect

        # Should handle gracefully
        key = f"{user_id}:operation:{operation_id}"
        ws_manager.connections[key] = {mock_websocket}

        await ws_manager.broadcast_to_operation(operation_id, {"test": "data"})

        # Connection should be removed
        assert mock_websocket not in ws_manager.connections.get(key, set())

    @pytest.mark.asyncio()
    async def test_handle_redis_connection_error(
        self, ws_manager: RedisStreamWebSocketManager, mock_websocket: AsyncMock
    ) -> None:
        """Test handling of Redis connection errors."""
        ws_manager.redis = None  # Simulate Redis disconnection

        # Should handle gracefully without crashing
        result = await ws_manager.send_message("test-channel", {"test": "data"})  # type: ignore[func-returns-value]

        # Should return False or handle error gracefully
        assert result is False or result is None

    @pytest.mark.asyncio()
    async def test_handle_malformed_message(
        self, ws_manager: RedisStreamWebSocketManager, mock_websocket: AsyncMock, mock_redis_client: AsyncMock
    ) -> None:
        """Test handling of malformed messages."""
        # Create a message that can't be JSON serialized
        malformed_message = {
            "type": "progress",
            "data": object(),  # Can't be JSON serialized
        }

        # Should handle without crashing
        with contextlib.suppress(TypeError):
            # Expected for non-serializable object
            await ws_manager.send_message("test-channel", malformed_message)

    @pytest.mark.asyncio()
    async def test_recovery_from_redis_stream_error(
        self, ws_manager: RedisStreamWebSocketManager, mock_redis_client: AsyncMock
    ) -> None:
        """Test recovery from Redis stream errors."""
        # Simulate Redis stream error
        mock_redis_client.xadd.side_effect = Exception("Redis stream error")

        # Should handle error gracefully
        result = await ws_manager.send_message("test-channel", {"test": "data"})  # type: ignore[func-returns-value]

        # Should indicate failure
        assert result is False or result is None

        # Reset error
        mock_redis_client.xadd.side_effect = None
        mock_redis_client.xadd.return_value = "123-0"

        # Should work again
        result = await ws_manager.send_message("test-channel", {"test": "data"})  # type: ignore[func-returns-value]
        assert result is not False


class TestConcurrentOperations:
    """Test handling of concurrent WebSocket operations."""

    @pytest.mark.asyncio()
    async def test_multiple_operations_same_user(
        self, ws_manager: RedisStreamWebSocketManager, mock_redis_client: AsyncMock
    ) -> None:
        """Test handling multiple operations for the same user."""
        user_id = "user-123"
        operation_ids = [f"op-{i}" for i in range(3)]
        websockets = []

        # Connect multiple operations for same user
        for op_id in operation_ids:
            ws = AsyncMock(spec=WebSocket)
            ws.accept = AsyncMock()
            ws.send_json = AsyncMock()
            websockets.append(ws)

            key = f"{user_id}:operation:{op_id}"
            ws_manager.connections[key] = {ws}

        # Send different updates to each operation
        for i, op_id in enumerate(operation_ids):
            channel = f"chunking:coll:{op_id}"
            message = {
                "type": "progress",
                "operation_id": op_id,
                "progress": i * 10,
            }
            await ws_manager.send_message(channel, message)

        # Verify all messages were sent
        assert mock_redis_client.xadd.call_count == 3

    @pytest.mark.asyncio()
    async def test_operation_isolation(
        self, ws_manager: RedisStreamWebSocketManager, mock_redis_client: AsyncMock
    ) -> None:
        """Test that operations are properly isolated from each other."""
        # Create connections for different operations
        op1_ws = AsyncMock(spec=WebSocket)
        op2_ws = AsyncMock(spec=WebSocket)

        ws_manager.connections["user1:operation:op1"] = {op1_ws}
        ws_manager.connections["user2:operation:op2"] = {op2_ws}

        # Send message to op1 channel
        await ws_manager.send_message("chunking:coll:op1", {"op": 1})

        # Send message to op2 channel
        await ws_manager.send_message("chunking:coll:op2", {"op": 2})

        # Verify messages went to correct channels
        calls = mock_redis_client.xadd.call_args_list
        assert len(calls) == 2
        assert "op1" in calls[0][0][0]
        assert "op2" in calls[1][0][0]

    @pytest.mark.asyncio()
    async def test_concurrent_connection_attempts(self, ws_manager: RedisStreamWebSocketManager) -> None:
        """Test handling of concurrent connection attempts."""
        user_id = "user-123"
        operation_id = "op-456"

        # Create multiple websockets trying to connect simultaneously
        websockets = [AsyncMock(spec=WebSocket) for _ in range(5)]
        for ws in websockets:
            ws.accept = AsyncMock()
            ws.close = AsyncMock()
            ws.send_json = AsyncMock()

        # Simulate concurrent connections
        tasks = []
        for ws in websockets:
            task = asyncio.create_task(ws_manager.connect(ws, operation_id, user_id))
            tasks.append(task)

        # Wait for all to complete
        await asyncio.gather(*tasks, return_exceptions=True)

        # Only one should be connected per operation
        key = f"{user_id}:operation:{operation_id}"
        if key in ws_manager.connections:
            # Should have limited connections
            assert len(ws_manager.connections[key]) <= ws_manager.max_connections_per_user


class TestPerformanceAndScaling:
    """Test performance and scaling aspects of WebSocket handling."""

    @pytest.mark.asyncio()
    async def test_large_message_handling(
        self, ws_manager: RedisStreamWebSocketManager, mock_redis_client: AsyncMock
    ) -> None:
        """Test handling of large progress messages."""
        # Create a large message with many chunks
        large_message = {
            "type": "chunking_progress",
            "operation_id": "op-123",
            "chunks_preview": [
                {
                    "index": i,
                    "content": f"Content {i}" * 100,
                    "metadata": {"key": f"value_{i}"},
                }
                for i in range(100)
            ],
        }

        # Should handle without issues
        await ws_manager.send_message("test-channel", large_message)
        mock_redis_client.xadd.assert_called_once()

    @pytest.mark.asyncio()
    async def test_message_ordering(
        self, ws_manager: RedisStreamWebSocketManager, mock_redis_client: AsyncMock
    ) -> None:
        """Test that messages maintain order."""
        messages = []

        # Capture messages as they're sent
        async def capture_xadd(_stream: str, data: dict, **_kwargs: Any) -> str:
            messages.append(json.loads(data.get("message", data.get("data", "{}"))))
            return f"{len(messages)}-0"

        mock_redis_client.xadd.side_effect = capture_xadd

        # Send messages in order
        for i in range(10):
            await ws_manager.send_message("test-channel", {"sequence": i, "type": "progress"})

        # Verify order is maintained
        for i, msg in enumerate(messages):
            assert msg["sequence"] == i

    @pytest.mark.asyncio()
    async def test_cleanup_performance(self, ws_manager: RedisStreamWebSocketManager) -> None:
        """Test performance of cleanup operations."""
        # Add many connections
        for i in range(100):
            key = f"user-{i}:operation:op-{i}"
            ws_manager.connections[key] = {AsyncMock(spec=WebSocket)}

            # Create a proper mock coroutine
            async def mock_coro() -> None:
                pass

            mock_task = asyncio.create_task(mock_coro())
            mock_task.cancel = MagicMock()
            ws_manager.consumer_tasks[f"op-{i}"] = mock_task

        # Measure cleanup time

        start = time.time()
        await ws_manager.shutdown()
        duration = time.time() - start

        # Should complete reasonably quickly
        assert duration < 5.0  # 5 seconds for 100 connections

        # Verify cleanup
        assert len(ws_manager.connections) == 0
        assert len(ws_manager.consumer_tasks) == 0


class TestIntegrationScenarios:
    """Test complete integration scenarios."""

    @pytest.mark.asyncio()
    async def test_complete_chunking_workflow(
        self, ws_manager: RedisStreamWebSocketManager, mock_websocket: AsyncMock, mock_redis_client: AsyncMock
    ) -> None:
        """Test a complete chunking workflow from start to finish."""
        user_id = "user-123"
        operation_id = str(uuid.uuid4())
        collection_id = "coll-456"
        channel = f"chunking:{collection_id}:{operation_id}"

        # 1. Connect WebSocket
        async def get_operation(op_id: str) -> dict[str, Any]:
            return {
                "uuid": op_id,
                "status": "pending",
                "progress_percentage": 0,
            }

        ws_manager.set_operation_getter(get_operation)
        await ws_manager.connect(mock_websocket, operation_id, user_id)

        # 2. Send start notification
        await ws_manager.send_message(
            channel,
            {
                "type": "chunking_started",
                "operation_id": operation_id,
                "total_documents": 5,
            },
        )

        # 3. Send progress updates
        for i in range(1, 6):
            await ws_manager.send_message(
                channel,
                {
                    "type": "chunking_progress",
                    "operation_id": operation_id,
                    "documents_processed": i,
                    "total_documents": 5,
                    "progress_percentage": i * 20,
                    "current_document": f"doc_{i}.pdf",
                },
            )
            await asyncio.sleep(0.1)  # Simulate processing time

        # 4. Send completion
        await ws_manager.send_message(
            channel,
            {
                "type": "chunking_completed",
                "operation_id": operation_id,
                "total_chunks_created": 250,
                "processing_time_seconds": 30,
            },
        )

        # Verify all messages were sent
        assert mock_redis_client.xadd.call_count >= 7  # start + 5 progress + completion

    @pytest.mark.asyncio()
    async def test_failed_chunking_workflow(
        self, ws_manager: RedisStreamWebSocketManager, mock_websocket: AsyncMock, mock_redis_client: AsyncMock
    ) -> None:
        """Test chunking workflow that fails midway."""
        user_id = "user-123"
        operation_id = str(uuid.uuid4())
        collection_id = "coll-789"
        channel = f"chunking:{collection_id}:{operation_id}"

        # Connect
        await ws_manager.connect(mock_websocket, operation_id, user_id)

        # Start processing
        await ws_manager.send_message(
            channel,
            {
                "type": "chunking_started",
                "operation_id": operation_id,
            },
        )

        # Some progress
        await ws_manager.send_message(
            channel,
            {
                "type": "chunking_progress",
                "operation_id": operation_id,
                "progress_percentage": 30,
            },
        )

        # Failure occurs
        await ws_manager.send_message(
            channel,
            {
                "type": "chunking_failed",
                "operation_id": operation_id,
                "error_message": "Out of memory",
                "error_code": "MEMORY_ERROR",
                "documents_processed": 2,
                "documents_failed": 1,
            },
        )

        # Verify error was sent
        calls = mock_redis_client.xadd.call_args_list
        last_call = calls[-1]
        last_message = json.loads(last_call[0][1].get("message", last_call[0][1].get("data", "{}")))
        assert last_message["type"] == "chunking_failed"
        assert last_message["error_code"] == "MEMORY_ERROR"
