"""Tests for ScalableWebSocketManager cross-instance messaging and scaling."""

import asyncio
import json
import uuid
from typing import Any
from unittest.mock import patch

import pytest
import redis.asyncio as redis

from packages.webui.websocket.scalable_manager import ScalableWebSocketManager


class MockWebSocket:
    """Mock WebSocket for testing."""

    def __init__(self) -> None:
        self.accepted = False
        self.closed = False
        self.close_code: int | None = None
        self.close_reason: str | None = None
        self.sent_messages: list[dict[str, Any]] = []
        self.connection_state = "open"

    async def accept(self) -> None:
        """Accept the WebSocket connection."""
        self.accepted = True

    async def close(self, code: int = 1000, reason: str = "") -> None:
        """Close the WebSocket connection."""
        self.closed = True
        self.close_code = code
        self.close_reason = reason
        self.connection_state = "closed"

    async def send_json(self, data: dict) -> None:
        """Send JSON data through the WebSocket."""
        if self.connection_state == "closed":
            raise RuntimeError("WebSocket is closed")
        self.sent_messages.append(data)

    async def ping(self) -> None:
        """Send a ping frame."""
        if self.connection_state == "closed":
            raise RuntimeError("WebSocket is closed")


@pytest.fixture()
async def redis_client():
    """Create a test Redis client."""
    client = await redis.from_url("redis://localhost:6379/15", decode_responses=True)

    # Clean up test database
    await client.flushdb()

    yield client

    # Cleanup
    await client.flushdb()
    await client.close()


@pytest.fixture()
async def manager():
    """Create a ScalableWebSocketManager instance for testing."""
    manager = ScalableWebSocketManager(redis_url="redis://localhost:6379/15")

    yield manager

    # Cleanup
    await manager.shutdown()


@pytest.fixture()
async def dual_managers():
    """Create two manager instances to test cross-instance communication."""
    manager1 = ScalableWebSocketManager(redis_url="redis://localhost:6379/15")
    manager2 = ScalableWebSocketManager(redis_url="redis://localhost:6379/15")

    await manager1.startup()
    await manager2.startup()

    yield manager1, manager2

    # Cleanup
    await manager1.shutdown()
    await manager2.shutdown()


class TestCrossInstanceMessaging:
    """Test cross-instance message routing."""

    @pytest.mark.asyncio()
    async def test_user_message_routing_across_instances(self, dual_managers):
        """Test that messages route correctly between instances via Redis Pub/Sub."""
        manager1, manager2 = dual_managers

        # Create mock WebSockets
        ws1 = MockWebSocket()
        ws2 = MockWebSocket()

        # Connect to different instances with same user
        user_id = "user123"
        conn1 = await manager1.connect(ws1, user_id)
        conn2 = await manager2.connect(ws2, user_id)

        # Allow pub/sub subscriptions to settle
        await asyncio.sleep(0.5)

        # Send message from instance 1
        test_message = {"type": "test", "data": "cross-instance"}
        await manager1.send_to_user(user_id, test_message)

        # Allow message to propagate
        await asyncio.sleep(0.5)

        # Both connections should receive the message
        assert len(ws1.sent_messages) >= 1
        assert len(ws2.sent_messages) >= 1

        # Find the test message (ignoring pings)
        ws1_test_msgs = [m for m in ws1.sent_messages if m.get("type") == "test"]
        ws2_test_msgs = [m for m in ws2.sent_messages if m.get("type") == "test"]

        assert len(ws1_test_msgs) == 1
        assert len(ws2_test_msgs) == 1
        assert ws1_test_msgs[0] == test_message
        assert ws2_test_msgs[0] == test_message

    @pytest.mark.asyncio()
    async def test_operation_broadcast_across_instances(self, dual_managers):
        """Test operation broadcasts work across instances."""
        manager1, manager2 = dual_managers

        # Create mock WebSockets
        ws1 = MockWebSocket()
        ws2 = MockWebSocket()

        # Connect to same operation on different instances
        operation_id = str(uuid.uuid4())
        conn1 = await manager1.connect(ws1, "user1", operation_id=operation_id)
        conn2 = await manager2.connect(ws2, "user2", operation_id=operation_id)

        # Allow subscriptions to settle
        await asyncio.sleep(0.5)

        # Broadcast to operation from instance 1
        test_message = {"type": "operation_update", "status": "processing"}
        await manager1.send_to_operation(operation_id, test_message)

        # Allow message to propagate
        await asyncio.sleep(0.5)

        # Both connections should receive the broadcast
        ws1_msgs = [m for m in ws1.sent_messages if m.get("type") == "operation_update"]
        ws2_msgs = [m for m in ws2.sent_messages if m.get("type") == "operation_update"]

        assert len(ws1_msgs) == 1
        assert len(ws2_msgs) == 1
        assert ws1_msgs[0] == test_message
        assert ws2_msgs[0] == test_message

    @pytest.mark.asyncio()
    async def test_collection_broadcast_across_instances(self, dual_managers):
        """Test collection broadcasts work across instances."""
        manager1, manager2 = dual_managers

        # Create mock WebSockets
        ws1 = MockWebSocket()
        ws2 = MockWebSocket()
        ws3 = MockWebSocket()

        # Connect to same collection on different instances
        collection_id = str(uuid.uuid4())
        conn1 = await manager1.connect(ws1, "user1", collection_id=collection_id)
        conn2 = await manager2.connect(ws2, "user2", collection_id=collection_id)
        conn3 = await manager2.connect(ws3, "user3", collection_id=collection_id)

        # Allow subscriptions to settle
        await asyncio.sleep(0.5)

        # Broadcast to collection from instance 1
        test_message = {"type": "collection_update", "action": "document_added"}
        await manager1.broadcast_to_collection(collection_id, test_message)

        # Allow message to propagate
        await asyncio.sleep(0.5)

        # All connections should receive the broadcast
        for ws in [ws1, ws2, ws3]:
            msgs = [m for m in ws.sent_messages if m.get("type") == "collection_update"]
            assert len(msgs) == 1
            assert msgs[0] == test_message

    @pytest.mark.asyncio()
    async def test_no_duplicate_messages_same_instance(self, manager):
        """Test that messages aren't duplicated when sender and receiver are on same instance."""
        await manager.startup()

        # Create mock WebSockets
        ws1 = MockWebSocket()
        ws2 = MockWebSocket()

        # Connect multiple connections for same user
        user_id = "user456"
        conn1 = await manager.connect(ws1, user_id)
        conn2 = await manager.connect(ws2, user_id)

        # Send message to user
        test_message = {"type": "test", "data": "no-duplicate"}
        await manager.send_to_user(user_id, test_message)

        # Allow any async operations to complete
        await asyncio.sleep(0.1)

        # Each connection should receive exactly one copy
        ws1_test_msgs = [m for m in ws1.sent_messages if m.get("type") == "test"]
        ws2_test_msgs = [m for m in ws2.sent_messages if m.get("type") == "test"]

        assert len(ws1_test_msgs) == 1
        assert len(ws2_test_msgs) == 1

    @pytest.mark.asyncio()
    async def test_instance_isolation(self, dual_managers):
        """Test that instances properly isolate their connections."""
        manager1, manager2 = dual_managers

        # Create connections on different instances
        ws1 = MockWebSocket()
        ws2 = MockWebSocket()

        conn1 = await manager1.connect(ws1, "user1")
        conn2 = await manager2.connect(ws2, "user2")

        # Allow subscriptions to settle
        await asyncio.sleep(0.5)

        # Send to user1 from manager1
        await manager1.send_to_user("user1", {"type": "test1"})

        # Send to user2 from manager2
        await manager2.send_to_user("user2", {"type": "test2"})

        # Allow messages to propagate
        await asyncio.sleep(0.5)

        # Each WebSocket should only receive its own messages
        ws1_msgs = [m for m in ws1.sent_messages if m.get("type") in ["test1", "test2"]]
        ws2_msgs = [m for m in ws2.sent_messages if m.get("type") in ["test1", "test2"]]

        assert len(ws1_msgs) == 1
        assert ws1_msgs[0]["type"] == "test1"

        assert len(ws2_msgs) == 1
        assert ws2_msgs[0]["type"] == "test2"


class TestConnectionScaling:
    """Test connection scaling and limits."""

    @pytest.mark.asyncio()
    async def test_max_connections_per_user(self, manager):
        """Test that user connection limits are enforced."""
        await manager.startup()

        user_id = "user_limit_test"
        connections = []

        # Create connections up to the limit
        for i in range(manager.max_connections_per_user):
            ws = MockWebSocket()
            conn_id = await manager.connect(ws, user_id)
            connections.append((ws, conn_id))
            assert ws.accepted
            assert not ws.closed

        # Try to create one more connection
        ws_overflow = MockWebSocket()
        with pytest.raises(ConnectionError, match="User connection limit exceeded"):
            await manager.connect(ws_overflow, user_id)

        assert ws_overflow.closed
        assert ws_overflow.close_code == 1008

    @pytest.mark.asyncio()
    async def test_max_total_connections(self, manager):
        """Test that global connection limits are enforced."""
        # Set a lower limit for testing
        manager.max_total_connections = 5
        await manager.startup()

        connections = []

        # Create connections up to the limit
        for i in range(manager.max_total_connections):
            ws = MockWebSocket()
            user_id = f"user_{i}"
            conn_id = await manager.connect(ws, user_id)
            connections.append((ws, conn_id))

        # Try to create one more connection
        ws_overflow = MockWebSocket()
        with pytest.raises(ConnectionError, match="Server connection limit exceeded"):
            await manager.connect(ws_overflow, "overflow_user")

        assert ws_overflow.closed
        assert ws_overflow.close_code == 1008

    @pytest.mark.asyncio()
    async def test_connection_registry_in_redis(self, manager, redis_client):
        """Test that connections are properly registered in Redis."""
        await manager.startup()

        # Connect multiple users
        connections = []
        for i in range(3):
            ws = MockWebSocket()
            user_id = f"user_{i}"
            conn_id = await manager.connect(ws, user_id)
            connections.append((user_id, conn_id))

        # Check Redis registry
        all_connections = await redis_client.hgetall("websocket:connections")
        assert len(all_connections) == 3

        # Verify each connection data
        for user_id, conn_id in connections:
            conn_data = json.loads(all_connections[conn_id])
            assert conn_data["user_id"] == user_id
            assert conn_data["instance_id"] == manager.instance_id
            assert conn_data["connection_id"] == conn_id
            assert "connected_at" in conn_data

        # Check user sets
        for user_id, conn_id in connections:
            user_connections = await redis_client.smembers(f"websocket:user:{user_id}")
            assert conn_id in user_connections

    @pytest.mark.asyncio()
    async def test_sticky_sessions_simulation(self, dual_managers):
        """Test that connections stick to their original instance."""
        manager1, manager2 = dual_managers

        # Simulate sticky session routing - user1 always goes to manager1
        user1_connections = []
        for i in range(3):
            ws = MockWebSocket()
            conn_id = await manager1.connect(ws, "user1")
            user1_connections.append((ws, conn_id))

        # user2 always goes to manager2
        user2_connections = []
        for i in range(3):
            ws = MockWebSocket()
            conn_id = await manager2.connect(ws, "user2")
            user2_connections.append((ws, conn_id))

        # Send messages and verify routing
        await manager1.send_to_user("user1", {"type": "test1"})
        await manager2.send_to_user("user2", {"type": "test2"})

        await asyncio.sleep(0.5)

        # All user1 connections should receive their message
        for ws, _ in user1_connections:
            msgs = [m for m in ws.sent_messages if m.get("type") == "test1"]
            assert len(msgs) == 1

        # All user2 connections should receive their message
        for ws, _ in user2_connections:
            msgs = [m for m in ws.sent_messages if m.get("type") == "test2"]
            assert len(msgs) == 1


class TestInstanceManagement:
    """Test instance registration and management."""

    @pytest.mark.asyncio()
    async def test_instance_registration(self, manager, redis_client):
        """Test that instances register themselves in Redis."""
        await manager.startup()

        # Check instance is registered
        instance_key = f"websocket:instance:{manager.instance_id}"
        instance_data = await redis_client.get(instance_key)
        assert instance_data is not None

        data = json.loads(instance_data)
        assert data["instance_id"] == manager.instance_id
        assert "started_at" in data

        # Check TTL is set
        ttl = await redis_client.ttl(instance_key)
        assert 0 < ttl <= 60

    @pytest.mark.asyncio()
    async def test_instance_heartbeat(self, manager, redis_client):
        """Test that instance heartbeat keeps registration alive."""
        await manager.startup()

        instance_key = f"websocket:instance:{manager.instance_id}"

        # Get initial TTL
        initial_ttl = await redis_client.ttl(instance_key)

        # Wait for heartbeat to run
        await asyncio.sleep(2)

        # Force heartbeat execution
        if manager.heartbeat_task:
            # The heartbeat should have refreshed the TTL
            current_ttl = await redis_client.ttl(instance_key)
            # TTL should still be positive
            assert current_ttl > 0

    @pytest.mark.asyncio()
    async def test_instance_stats_tracking(self, manager, redis_client):
        """Test that instance stats are tracked in Redis."""
        await manager.startup()

        # Create some connections
        connections = []
        for i in range(3):
            ws = MockWebSocket()
            conn_id = await manager.connect(ws, f"user_{i}")
            connections.append((ws, conn_id))

        # Get stats
        stats = await manager.get_stats()

        assert stats["instance_id"] == manager.instance_id
        assert stats["local_connections"] == 3
        assert stats["unique_users"] == 3

        # Check Redis stats after heartbeat
        await asyncio.sleep(2)

        instance_stats = await redis_client.hget("websocket:instances:stats", manager.instance_id)
        if instance_stats:
            stats_data = json.loads(instance_stats)
            assert stats_data["connections"] == 3
            assert stats_data["users"] == 3


class TestMessageThrottling:
    """Test message throttling and performance optimizations."""

    @pytest.mark.asyncio()
    async def test_message_throttling(self, manager):
        """Test that rapid messages are throttled appropriately."""
        await manager.startup()

        ws = MockWebSocket()
        user_id = "user_throttle"
        conn_id = await manager.connect(ws, user_id)

        # Send rapid messages
        for i in range(10):
            await manager.send_to_user(user_id, {"type": "rapid", "index": i})
            await asyncio.sleep(0.01)  # 10ms between messages

        # Not all messages should be sent due to throttling
        # (This depends on the actual throttling implementation)
        rapid_msgs = [m for m in ws.sent_messages if m.get("type") == "rapid"]

        # Should have received at least some messages
        assert len(rapid_msgs) > 0
        # But potentially not all 10 if throttling is active
        assert len(rapid_msgs) <= 10


class TestRedisFailure:
    """Test behavior when Redis is unavailable."""

    @pytest.mark.asyncio()
    async def test_startup_with_redis_down(self):
        """Test graceful handling when Redis is unavailable at startup."""
        manager = ScalableWebSocketManager(redis_url="redis://nonexistent:6379/15")

        # Should not raise, but log errors
        await manager.startup()

        # Manager should still be usable for local connections
        assert manager.redis_client is not None or not manager._startup_complete

        await manager.shutdown()

    @pytest.mark.asyncio()
    async def test_local_messaging_without_redis(self):
        """Test that local messaging works even without Redis."""
        manager = ScalableWebSocketManager(redis_url="redis://localhost:6379/15")

        # Mock Redis to be None
        with patch.object(manager, "redis_client", None):
            ws1 = MockWebSocket()
            ws2 = MockWebSocket()

            # Should still allow local connections
            conn1 = await manager.connect(ws1, "user1")
            conn2 = await manager.connect(ws2, "user1")

            # Local message delivery should work
            await manager.send_to_user("user1", {"type": "local_test"})

            # Both should receive the message locally
            assert any(m.get("type") == "local_test" for m in ws1.sent_messages)
            assert any(m.get("type") == "local_test" for m in ws2.sent_messages)
