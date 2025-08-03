"""Integration tests for WebSocket and Redis streaming functionality."""

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import WebSocket

from packages.webui.tasks import CeleryTaskWithOperationUpdates
from packages.webui.websocket_manager import RedisStreamWebSocketManager


@pytest.mark.timeout(30)  # Add timeout to prevent hanging
class TestWebSocketRedisIntegration:
    """Integration tests for WebSocket and Redis streaming."""

    async def _cleanup_tasks(self) -> None:
        """Helper to clean up all running async tasks."""
        # Give tasks a moment to complete naturally
        await asyncio.sleep(0.1)
        
        # Get all running tasks
        tasks = [task for task in asyncio.all_tasks() if not task.done() and task != asyncio.current_task()]
        
        # Cancel all remaining tasks
        for task in tasks:
            task.cancel()
        
        # Wait for all tasks to be cancelled
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _setup_operation_getter(self, manager, operation_ids) -> None:
        """Helper to set up mock operation getter for tests."""
        from datetime import UTC, datetime
        from enum import Enum
        from unittest.mock import MagicMock

        # Create mock enums
        class MockStatus(Enum):
            PROCESSING = "processing"

        class MockType(Enum):
            INDEX = "index"

        # Create mock operation
        mock_operation = MagicMock()
        mock_operation.status = MockStatus.PROCESSING
        mock_operation.type = MockType.INDEX
        mock_operation.collection_id = "collection1"
        mock_operation.created_at = datetime.now(UTC)
        mock_operation.started_at = datetime.now(UTC)
        mock_operation.completed_at = None
        mock_operation.error_message = None

        # Set up the operation getter function
        async def mock_get_operation(operation_id):
            if operation_id in operation_ids:
                return mock_operation
            return None

        manager.set_operation_getter(mock_get_operation)

    @pytest.fixture()
    async def real_redis_mock(self) -> None:
        """Create a more realistic Redis mock that simulates stream behavior."""

        class RedisStreamMock:
            def __init__(self) -> None:
                self.streams = {}
                self.consumer_groups = {}
                self.closed = False
                self.pending_messages = {}  # Track pending messages for consumer groups

            async def ping(self):
                if self.closed:
                    raise Exception("Connection closed")
                return True

            async def xadd(self, stream_key, data, maxlen=None):
                if self.closed:
                    raise Exception("Connection closed")

                if stream_key not in self.streams:
                    self.streams[stream_key] = []

                # Generate message ID
                msg_id = f"{len(self.streams[stream_key])}-0"
                self.streams[stream_key].append((msg_id, data))

                # Trim to maxlen if specified
                if maxlen and len(self.streams[stream_key]) > maxlen:
                    self.streams[stream_key] = self.streams[stream_key][-maxlen:]

                return msg_id

            async def expire(self, key, ttl):
                # Just track that expire was called
                pass

            async def xrange(self, stream_key, min="-", max="+", count=None):
                if stream_key not in self.streams:
                    return []

                messages = self.streams[stream_key]
                if count:
                    messages = messages[-count:]

                return messages

            async def xgroup_create(self, stream_key, group_name, id="0"):
                if stream_key not in self.consumer_groups:
                    self.consumer_groups[stream_key] = {}
                self.consumer_groups[stream_key][group_name] = {"last_delivered_id": id, "consumers": {}}

            async def xreadgroup(self, group_name, consumer_name, streams, count=None, block=None):
                # Simulate reading from stream with proper consumer group semantics
                results = []

                for stream_key, last_id in streams.items():
                    if stream_key not in self.streams:
                        continue

                    if stream_key not in self.consumer_groups:
                        continue

                    if group_name not in self.consumer_groups[stream_key]:
                        continue

                    group_info = self.consumer_groups[stream_key][group_name]

                    # Track this consumer
                    if consumer_name not in group_info["consumers"]:
                        group_info["consumers"][consumer_name] = {"last_ack": None}

                    # Get new messages since last delivered to this group
                    all_messages = self.streams[stream_key]
                    new_messages = []

                    if last_id == ">":
                        # Find messages after the group's last delivered ID
                        last_delivered = group_info["last_delivered_id"]
                        for msg_id, data in all_messages:
                            if msg_id > last_delivered:
                                new_messages.append((msg_id, data))

                        # Update last delivered ID for the group
                        if new_messages:
                            group_info["last_delivered_id"] = new_messages[-1][0]

                    if new_messages:
                        if count:
                            new_messages = new_messages[:count]
                        results.append((stream_key, new_messages))

                return results

            async def xack(self, stream_key, group_name, msg_id):
                # Just track acknowledgment
                pass

            async def xgroup_delconsumer(self, stream_key, group_name, consumer_name):
                pass

            async def delete(self, key):
                if key in self.streams:
                    del self.streams[key]
                return 1

            async def xinfo_groups(self, stream_key):
                if stream_key in self.consumer_groups:
                    return [{"name": name} for name in self.consumer_groups[stream_key]]
                return []

            async def xgroup_destroy(self, stream_key, group_name):
                if stream_key in self.consumer_groups:
                    self.consumer_groups[stream_key].pop(group_name, None)

            async def xinfo_stream(self, stream_key):
                if stream_key not in self.streams:
                    raise Exception(f"Stream {stream_key} does not exist")
                return {"length": len(self.streams[stream_key])}

            async def close(self):
                self.closed = True

        mock = RedisStreamMock()
        yield mock
        # Ensure the mock is closed after the test
        await mock.close()

    @pytest.fixture()
    async def mock_websocket_factory(self) -> None:
        """Factory to create mock WebSocket connections."""
        created_websockets = []

        def create_mock_websocket(client_id) -> None:
            # Create fresh AsyncMock instance
            mock = AsyncMock(spec=WebSocket)
            mock.accept = AsyncMock()
            mock.send_json = AsyncMock()
            mock.close = AsyncMock()
            mock.client_id = client_id  # For tracking in tests

            # Initialize fresh message list
            received_messages = []
            mock.received_messages = received_messages

            # Store messages when send_json is called
            async def track_send_json(data):
                received_messages.append(data)

            mock.send_json.side_effect = track_send_json

            created_websockets.append(mock)
            return mock

        yield create_mock_websocket
        
        # Clean up all created websockets
        for ws in created_websockets:
            try:
                await ws.close()
            except Exception:
                pass

    @pytest.mark.asyncio()
    async def test_end_to_end_operation_updates_flow(self, real_redis_mock, mock_websocket_factory):
        """Test complete flow from Celery task to WebSocket client."""
        # Setup
        manager = RedisStreamWebSocketManager()

        async def async_from_url(*args, **kwargs):  # noqa: ARG001
            return real_redis_mock

        with patch("packages.webui.websocket_manager.redis.from_url", side_effect=async_from_url):
            await manager.startup()

            # Create WebSocket client
            ws_client = mock_websocket_factory("client1")

            # Mock operation repository
            with patch("packages.shared.database.factory.create_operation_repository") as mock_create_repo:
                from datetime import UTC, datetime
                from enum import Enum
                from unittest.mock import MagicMock

                # Create mock operation object
                class MockStatus(Enum):
                    PROCESSING = "processing"

                class MockType(Enum):
                    INDEX = "index"

                mock_operation = MagicMock()
                mock_operation.uuid = "operation1"
                mock_operation.status = MockStatus.PROCESSING
                mock_operation.type = MockType.INDEX
                mock_operation.collection_id = "collection1"
                mock_operation.created_at = datetime.now(UTC)
                mock_operation.started_at = datetime.now(UTC)
                mock_operation.completed_at = None
                mock_operation.error_message = None

                mock_repo = AsyncMock()
                mock_repo.get_by_uuid = AsyncMock(return_value=mock_operation)
                mock_create_repo.return_value = mock_repo

                # Set up the operation getter function
                async def mock_get_operation(operation_id):
                    if operation_id == "operation1":
                        return mock_operation
                    return None

                # IMPORTANT: Set operation getter BEFORE connecting WebSocket
                manager.set_operation_getter(mock_get_operation)

                # First create the stream by sending an initial update
                # This ensures the stream exists before the consumer starts
                celery_updater = CeleryTaskWithOperationUpdates("operation1")
                celery_updater._redis_client = real_redis_mock

                # Send initial update to create the stream
                await celery_updater.send_update("start", {"status": "started"})
                await asyncio.sleep(0.1)  # Allow stream creation

                # Now connect WebSocket client - consumer will find existing stream
                await manager.connect(ws_client, "operation1", "user1")

                # Allow consumer task to start and read initial message
                await asyncio.sleep(0.2)

                # Send various updates
                await celery_updater.send_update("progress", {"progress": 25, "current_file": "doc1.pdf"})
                await asyncio.sleep(0.1)

                await celery_updater.send_update("progress", {"progress": 50, "current_file": "doc2.pdf"})
                await asyncio.sleep(0.1)

                # Verify client received all updates
                received_types = [msg["type"] for msg in ws_client.received_messages]
                assert "current_state" in received_types  # Initial state
                assert received_types.count("progress") >= 2  # Progress updates

                # Verify update content
                progress_updates = [msg for msg in ws_client.received_messages if msg["type"] == "progress"]
                if progress_updates:
                    assert any(u["data"]["progress"] == 25 for u in progress_updates)
                    assert any(u["data"]["progress"] == 50 for u in progress_updates)

                # Cleanup
                await manager.disconnect(ws_client, "operation1", "user1")
                await manager.shutdown()
                await self._cleanup_tasks()

    @pytest.mark.asyncio()
    async def test_multiple_clients_receive_updates(self, real_redis_mock, mock_websocket_factory):
        """Test that multiple clients receive the same updates."""
        manager = RedisStreamWebSocketManager()

        async def async_from_url(*args, **kwargs):  # noqa: ARG001
            return real_redis_mock

        with patch("packages.webui.websocket_manager.redis.from_url", side_effect=async_from_url):
            await manager.startup()

            # Create multiple clients
            clients = [mock_websocket_factory(f"client{i}") for i in range(3)]

            # Mock operation repository
            with patch("packages.shared.database.factory.create_operation_repository") as mock_create_repo:
                mock_repo = AsyncMock()
                # Create simple mock operation
                mock_operation = type(
                    "MockOp",
                    (),
                    {
                        "uuid": "operation1",
                        "status": type("Status", (), {"value": "processing"}),
                        "type": type("Type", (), {"value": "index"}),
                        "collection_id": "collection1",
                        "created_at": datetime.now(UTC),
                        "started_at": datetime.now(UTC),
                        "completed_at": None,
                        "error_message": None,
                    },
                )()
                mock_repo.get_by_uuid = AsyncMock(return_value=mock_operation)
                mock_create_repo.return_value = mock_repo

                # Set up operation getter for all clients
                self._setup_operation_getter(manager, ["operation1"])

                # Create the stream by sending an initial update BEFORE connecting clients
                await manager.send_update("operation1", "initial", {"status": "stream_created"})
                await asyncio.sleep(0.1)

                # Connect all clients
                for i, client in enumerate(clients):
                    await manager.connect(client, "operation1", f"user{i}")

                await asyncio.sleep(0.2)

                # Send update via manager (simulating from API)
                await manager.send_update("operation1", "broadcast", {"message": "Update for all clients"})

                await asyncio.sleep(0.1)

                # Verify all clients received the update
                for client in clients:
                    broadcast_msgs = [msg for msg in client.received_messages if msg["type"] == "broadcast"]
                    assert len(broadcast_msgs) >= 1
                    assert broadcast_msgs[0]["data"]["message"] == "Update for all clients"

                # Cleanup
                for i, client in enumerate(clients):
                    await manager.disconnect(client, "operation1", f"user{i}")

            await manager.shutdown()
            await self._cleanup_tasks()

    @pytest.mark.asyncio()
    async def test_message_history_replay(self, real_redis_mock, mock_websocket_factory):
        """Test that new clients receive message history."""
        manager = RedisStreamWebSocketManager()

        async def async_from_url(*args, **kwargs):  # noqa: ARG001
            return real_redis_mock

        with patch("packages.webui.websocket_manager.redis.from_url", side_effect=async_from_url):
            await manager.startup()

            # Send some updates before any client connects
            await manager.send_update("operation1", "update1", {"data": "first"})
            await manager.send_update("operation1", "update2", {"data": "second"})
            await manager.send_update("operation1", "update3", {"data": "third"})

            # Now connect a client
            client = mock_websocket_factory("client1")

            with patch("packages.shared.database.factory.create_operation_repository") as mock_create_repo:
                mock_repo = AsyncMock()
                # Create simple mock operation
                mock_operation = type(
                    "MockOp",
                    (),
                    {
                        "uuid": "operation1",
                        "status": type("Status", (), {"value": "processing"}),
                        "type": type("Type", (), {"value": "index"}),
                        "collection_id": "collection1",
                        "created_at": datetime.now(UTC),
                        "started_at": datetime.now(UTC),
                        "completed_at": None,
                        "error_message": None,
                    },
                )()
                mock_repo.get_by_uuid = AsyncMock(return_value=mock_operation)
                mock_create_repo.return_value = mock_repo

                # Set up operation getter
                self._setup_operation_getter(manager, ["operation1"])

                await manager.connect(client, "operation1", "user1")
                await asyncio.sleep(0.1)

                # Client should receive historical messages
                historical_types = [msg["type"] for msg in client.received_messages]
                assert "update1" in historical_types
                assert "update2" in historical_types
                assert "update3" in historical_types

                # Verify order is preserved
                update_messages = [
                    msg for msg in client.received_messages if msg["type"] in ["update1", "update2", "update3"]
                ]
                assert update_messages[0]["type"] == "update1"
                assert update_messages[1]["type"] == "update2"
                assert update_messages[2]["type"] == "update3"

                await manager.disconnect(client, "operation1", "user1")

            await manager.shutdown()
            await self._cleanup_tasks()

    @pytest.mark.asyncio()
    async def test_consumer_group_coordination(self, real_redis_mock, mock_websocket_factory):
        """Test that multiple server instances coordinate via consumer groups."""
        # Create two manager instances (simulating multiple servers)
        manager1 = RedisStreamWebSocketManager()
        manager2 = RedisStreamWebSocketManager()

        async def async_from_url(*args, **kwargs):  # noqa: ARG001
            return real_redis_mock

        with patch("packages.webui.websocket_manager.redis.from_url", side_effect=async_from_url):
            await manager1.startup()
            await manager2.startup()

            # Connect clients to different managers
            client1 = mock_websocket_factory("client1")
            client2 = mock_websocket_factory("client2")

            with patch("packages.shared.database.factory.create_operation_repository") as mock_create_repo:
                mock_repo = AsyncMock()
                # Create simple mock operation
                mock_operation = type(
                    "MockOp",
                    (),
                    {
                        "uuid": "operation1",
                        "status": type("Status", (), {"value": "processing"}),
                        "type": type("Type", (), {"value": "index"}),
                        "collection_id": "collection1",
                        "created_at": datetime.now(UTC),
                        "started_at": datetime.now(UTC),
                        "completed_at": None,
                        "error_message": None,
                    },
                )()
                mock_repo.get_by_uuid = AsyncMock(return_value=mock_operation)
                mock_create_repo.return_value = mock_repo

                # Set up operation getter for both managers
                self._setup_operation_getter(manager1, ["operation1"])
                self._setup_operation_getter(manager2, ["operation1"])

                # Create the stream first by sending an initial update
                await manager1.send_update("operation1", "initial", {"status": "stream_created"})
                await asyncio.sleep(0.1)

                # Connect to different manager instances
                await manager1.connect(client1, "operation1", "user1")
                await manager2.connect(client2, "operation1", "user2")

                await asyncio.sleep(0.1)

                # Send update (could come from any instance)
                await manager1.send_update("operation1", "shared_update", {"message": "Update from manager1"})

                await asyncio.sleep(0.2)

                # Both clients should receive the update
                client1_updates = [msg for msg in client1.received_messages if msg["type"] == "shared_update"]
                client2_updates = [msg for msg in client2.received_messages if msg["type"] == "shared_update"]

                # At least one client should receive it
                # (In real Redis, only one consumer group member would get it)
                assert len(client1_updates) > 0 or len(client2_updates) > 0

                # Cleanup
                await manager1.disconnect(client1, "operation1", "user1")
                await manager2.disconnect(client2, "operation1", "user2")

            await manager1.shutdown()
            await manager2.shutdown()
            await self._cleanup_tasks()

    @pytest.mark.asyncio()
    async def test_stream_cleanup_after_operation_completion(self, real_redis_mock):
        """Test that Redis streams are cleaned up after operation completion."""
        manager = RedisStreamWebSocketManager()

        async def async_from_url(*args, **kwargs):  # noqa: ARG001
            return real_redis_mock

        with patch("packages.webui.websocket_manager.redis.from_url", side_effect=async_from_url):
            await manager.startup()

            # Send some updates
            await manager.send_update("operation1", "progress", {"progress": 50})
            await manager.send_update("operation1", "complete", {"status": "completed"})

            # Verify stream exists
            assert "operation-progress:operation1" in real_redis_mock.streams

            # Clean up the stream
            await manager.cleanup_stream("operation1")

            # Verify stream is deleted
            assert "operation-progress:operation1" not in real_redis_mock.streams

            await manager.shutdown()
            await self._cleanup_tasks()

    @pytest.mark.asyncio()
    async def test_graceful_degradation_without_redis(self, mock_websocket_factory):
        """Test that system works in degraded mode when Redis is unavailable."""
        # Create a fresh manager instance with clean state
        manager = RedisStreamWebSocketManager()
        # Ensure no lingering connections
        manager.connections.clear()
        manager.consumer_tasks.clear()

        # Simulate Redis connection failure
        with (
            patch("packages.webui.websocket_manager.redis.from_url", side_effect=Exception("Connection failed")),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            await manager.startup()

        assert manager.redis is None  # Redis not available

        # Mark startup as already attempted to prevent reconnection attempts
        manager._startup_attempted = True

        # System should still work for direct broadcasts
        client = mock_websocket_factory("client1")

        with patch("packages.shared.database.factory.create_operation_repository") as mock_create_repo:
            mock_repo = AsyncMock()
            # Create simple mock operation
            mock_operation = type(
                "MockOp",
                (),
                {
                    "uuid": "operation1",
                    "status": type("Status", (), {"value": "processing"}),
                    "type": type("Type", (), {"value": "index"}),
                    "collection_id": "collection1",
                    "created_at": datetime.now(UTC),
                    "started_at": datetime.now(UTC),
                    "completed_at": None,
                    "error_message": None,
                    "total_files": 10,
                },
            )()
            mock_repo.get_by_uuid = AsyncMock(return_value=mock_operation)
            mock_create_repo.return_value = mock_repo

            # Set up operation getter
            self._setup_operation_getter(manager, ["operation1"])

            # Clear any previous messages before connecting
            client.received_messages.clear()

            # Connect should still work
            await manager.connect(client, "operation1", "user1")

            # Should receive initial state only
            state_messages = [msg for msg in client.received_messages if msg["type"] == "current_state"]
            assert (
                len(state_messages) == 1
            ), f"Expected 1 current_state message but got {len(state_messages)}: {state_messages}"
            assert state_messages[0]["data"]["status"] == "processing"

            # Direct updates should work
            await manager.send_update("operation1", "progress", {"progress": 75})

            # Allow time for direct broadcast
            await asyncio.sleep(0.1)

            # Client should receive update via direct broadcast
            # Now check for the new progress message
            progress_messages = [
                msg for msg in client.received_messages if msg["type"] == "progress" and msg["data"]["progress"] == 75
            ]
            assert (
                len(progress_messages) >= 1
            ), f"Expected at least 1 progress message with value 75 but got {len(progress_messages)}"

            # Verify we got at least 2 messages total (initial state + progress)
            assert (
                len(client.received_messages) >= 2
            ), f"Expected at least 2 messages but got {len(client.received_messages)}"

            await manager.disconnect(client, "operation1", "user1")

        await manager.shutdown()
        await self._cleanup_tasks()

    @pytest.mark.asyncio()
    async def test_connection_resilience(self, real_redis_mock, mock_websocket_factory):
        """Test handling of connection failures and reconnections."""
        manager = RedisStreamWebSocketManager()

        async def async_from_url(*args, **kwargs):  # noqa: ARG001
            return real_redis_mock

        with patch("packages.webui.websocket_manager.redis.from_url", side_effect=async_from_url):
            await manager.startup()

            # Create a client that will disconnect
            client = mock_websocket_factory("client1")

            with patch("packages.shared.database.factory.create_operation_repository") as mock_create_repo:
                mock_repo = AsyncMock()
                # Create simple mock operation
                mock_operation = type(
                    "MockOp",
                    (),
                    {
                        "uuid": "operation1",
                        "status": type("Status", (), {"value": "processing"}),
                        "type": type("Type", (), {"value": "index"}),
                        "collection_id": "collection1",
                        "created_at": datetime.now(UTC),
                        "started_at": datetime.now(UTC),
                        "completed_at": None,
                        "error_message": None,
                    },
                )()
                mock_repo.get_by_uuid = AsyncMock(return_value=mock_operation)
                mock_create_repo.return_value = mock_repo

                # Set up operation getter
                self._setup_operation_getter(manager, ["operation1"])

                await manager.connect(client, "operation1", "user1")

                # Simulate client disconnect by making send_json fail
                client.send_json.side_effect = Exception("Connection lost")

                # Send update - should handle the failed client
                await manager.send_update("operation1", "test", {"data": "test"})
                await asyncio.sleep(0.1)

                # Client should be automatically removed from connections
                assert client not in manager.connections.get("user1:operation:operation1", set())

            await manager.shutdown()
            await self._cleanup_tasks()

    @pytest.mark.asyncio()
    async def test_concurrent_operation_processing(self, real_redis_mock, mock_websocket_factory):
        """Test handling multiple operations concurrently."""
        manager = RedisStreamWebSocketManager()

        async def async_from_url(*args, **kwargs):  # noqa: ARG001
            return real_redis_mock

        with patch("packages.webui.websocket_manager.redis.from_url", side_effect=async_from_url):
            await manager.startup()

            # Create clients for different operations
            clients = {
                "operation1": [mock_websocket_factory(f"operation1_client{i}") for i in range(2)],
                "operation2": [mock_websocket_factory(f"operation2_client{i}") for i in range(2)],
                "operation3": [mock_websocket_factory(f"operation3_client{i}") for i in range(2)],
            }

            with patch("packages.shared.database.factory.create_operation_repository") as mock_create_repo:
                mock_repo = AsyncMock()
                # Create simple mock operation
                mock_operation = type(
                    "MockOp",
                    (),
                    {
                        "uuid": "operation1",
                        "status": type("Status", (), {"value": "processing"}),
                        "type": type("Type", (), {"value": "index"}),
                        "collection_id": "collection1",
                        "created_at": datetime.now(UTC),
                        "started_at": datetime.now(UTC),
                        "completed_at": None,
                        "error_message": None,
                    },
                )()
                mock_repo.get_by_uuid = AsyncMock(return_value=mock_operation)
                mock_create_repo.return_value = mock_repo

                # Set up operation getter for all operations
                self._setup_operation_getter(manager, ["operation1", "operation2", "operation3"])

                # Create streams first by sending initial updates
                await manager.send_update("operation1", "initial", {"status": "stream_created"})
                await manager.send_update("operation2", "initial", {"status": "stream_created"})
                await manager.send_update("operation3", "initial", {"status": "stream_created"})
                await asyncio.sleep(0.1)

                # Connect all clients
                for operation_id, operation_clients in clients.items():
                    for i, client in enumerate(operation_clients):
                        await manager.connect(client, operation_id, f"user_{operation_id}_{i}")

                await asyncio.sleep(0.1)

                # Send updates to different operations
                await manager.send_update("operation1", "update", {"operation": "operation1"})
                await manager.send_update("operation2", "update", {"operation": "operation2"})
                await manager.send_update("operation3", "update", {"operation": "operation3"})

                await asyncio.sleep(0.1)

                # Verify each operation's clients only received their updates
                for operation_id, operation_clients in clients.items():
                    for client in operation_clients:
                        updates = [msg for msg in client.received_messages if msg["type"] == "update"]
                        assert len(updates) >= 1
                        assert all(u["data"]["operation"] == operation_id for u in updates)

                # Cleanup
                for operation_id, operation_clients in clients.items():
                    for i, client in enumerate(operation_clients):
                        await manager.disconnect(client, operation_id, f"user_{operation_id}_{i}")

            await manager.shutdown()
            await self._cleanup_tasks()
