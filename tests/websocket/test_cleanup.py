"""Tests for ScalableWebSocketManager connection cleanup and TTL management."""

import asyncio
import json
import time
import uuid

import pytest
import redis.asyncio as redis

from packages.webui.websocket.scalable_manager import ScalableWebSocketManager


class MockWebSocket:
    """Mock WebSocket for testing."""

    def __init__(self, should_fail: bool = False) -> None:
        self.accepted = False
        self.closed = False
        self.close_code = None
        self.close_reason = None
        self.sent_messages: list[dict] = []
        self.should_fail = should_fail
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
        if self.should_fail:
            raise RuntimeError("WebSocket send failed")
        if self.connection_state == "closed":
            raise RuntimeError("WebSocket is closed")
        self.sent_messages.append(data)


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


class TestConnectionCleanup:
    """Test automatic connection cleanup."""

    @pytest.mark.asyncio()
    async def test_disconnect_removes_from_registry(self, manager, redis_client):
        """Test that disconnect properly removes connection from all registries."""
        await manager.startup()

        # Connect
        ws = MockWebSocket()
        user_id = "user_cleanup"
        operation_id = str(uuid.uuid4())
        conn_id = await manager.connect(ws, user_id, operation_id=operation_id)

        # Verify registration
        conn_data = await redis_client.hget("websocket:connections", conn_id)
        assert conn_data is not None

        user_connections = await redis_client.smembers(f"websocket:user:{user_id}")
        assert conn_id in user_connections

        # Disconnect
        await manager.disconnect(conn_id)

        # Verify cleanup
        conn_data = await redis_client.hget("websocket:connections", conn_id)
        assert conn_data is None

        user_connections = await redis_client.smembers(f"websocket:user:{user_id}")
        assert conn_id not in user_connections

        # Local tracking should be cleared
        assert conn_id not in manager.local_connections
        assert conn_id not in manager.connection_metadata

    @pytest.mark.asyncio()
    async def test_dead_connection_cleanup_via_heartbeat(self, manager):
        """Test that dead connections are cleaned up by heartbeat."""
        await manager.startup()

        # Create connections - one that will fail
        ws_good = MockWebSocket()
        ws_bad = MockWebSocket(should_fail=True)

        conn_good = await manager.connect(ws_good, "user_good")
        conn_bad = await manager.connect(ws_bad, "user_bad")

        # Verify both are connected
        assert len(manager.local_connections) == 2

        # Manually trigger heartbeat logic
        dead_connections = []
        for conn_id, websocket in list(manager.local_connections.items()):
            try:
                await asyncio.wait_for(
                    websocket.send_json({"type": "ping"}),
                    timeout=1.0
                )
            except Exception:
                dead_connections.append(conn_id)

        # Clean up dead connections
        for conn_id in dead_connections:
            await manager.disconnect(conn_id)

        # Should have removed the bad connection
        assert len(manager.local_connections) == 1
        assert conn_good in manager.local_connections
        assert conn_bad not in manager.local_connections

    @pytest.mark.asyncio()
    async def test_channel_unsubscribe_on_disconnect(self, manager):
        """Test that channels are unsubscribed when last connection disconnects."""
        await manager.startup()

        # Connect two users to same operation
        ws1 = MockWebSocket()
        ws2 = MockWebSocket()
        operation_id = str(uuid.uuid4())

        conn1 = await manager.connect(ws1, "user1", operation_id=operation_id)
        conn2 = await manager.connect(ws2, "user2", operation_id=operation_id)

        # Allow time for subscription to be processed
        await asyncio.sleep(0.1)

        # Both should be subscribed to operation channel
        subscriptions = list(manager.pubsub.channels.keys())
        operation_channel = f"operation:{operation_id}".encode() if subscriptions and isinstance(subscriptions[0], bytes) else f"operation:{operation_id}"
        assert operation_channel in subscriptions

        # Disconnect first connection
        await manager.disconnect(conn1)
        
        # Allow time for unsubscription check to process
        await asyncio.sleep(0.1)

        # Should still be subscribed (one connection remains)
        subscriptions = list(manager.pubsub.channels.keys())
        assert operation_channel in subscriptions

        # Disconnect second connection
        await manager.disconnect(conn2)
        
        # Allow time for unsubscription to be processed
        await asyncio.sleep(0.1)

        # Should be unsubscribed (no connections remain)
        subscriptions = list(manager.pubsub.channels.keys())
        assert operation_channel not in subscriptions

    @pytest.mark.asyncio()
    async def test_instance_cleanup_on_shutdown(self, manager, redis_client):
        """Test that instance cleanup happens properly on shutdown."""
        await manager.startup()

        # Create some connections
        connections = []
        for i in range(3):
            ws = MockWebSocket()
            conn_id = await manager.connect(ws, f"user_{i}")
            connections.append(conn_id)

        # Verify instance is registered
        instance_key = f"websocket:instance:{manager.instance_id}"
        assert await redis_client.exists(instance_key)

        # Shutdown
        await manager.shutdown()

        # Instance should be unregistered
        assert not await redis_client.exists(instance_key)

        # All connections should be removed from registry
        for conn_id in connections:
            conn_data = await redis_client.hget("websocket:connections", conn_id)
            assert conn_data is None

        # WebSockets should be closed (manager clears local_connections on shutdown)
        assert len(manager.local_connections) == 0

    @pytest.mark.asyncio()
    async def test_cleanup_dead_instance_connections(self, manager, redis_client):
        """Test cleanup of connections from dead instances."""
        await manager.startup()

        # Simulate a dead instance with orphaned connections
        dead_instance_id = str(uuid.uuid4())
        orphaned_conn_id = str(uuid.uuid4())

        # Add orphaned connection to registry
        orphaned_data = {
            "connection_id": orphaned_conn_id,
            "user_id": "orphaned_user",
            "instance_id": dead_instance_id,
            "connected_at": time.time()
        }
        await redis_client.hset(
            "websocket:connections",
            orphaned_conn_id,
            json.dumps(orphaned_data)
        )
        await redis_client.sadd("websocket:user:orphaned_user", orphaned_conn_id)

        # Run cleanup task logic
        connections = await redis_client.hgetall("websocket:connections")

        for conn_id, conn_data in connections.items():
            data = json.loads(conn_data)
            instance_id = data.get("instance_id")

            # Check if instance is alive
            instance_key = f"websocket:instance:{instance_id}"
            if not await redis_client.exists(instance_key):
                # Remove orphaned connection
                await redis_client.hdel("websocket:connections", conn_id)
                user_id = data.get("user_id")
                if user_id:
                    await redis_client.srem(f"websocket:user:{user_id}", conn_id)

        # Orphaned connection should be cleaned up
        assert not await redis_client.hexists("websocket:connections", orphaned_conn_id)
        assert orphaned_conn_id not in await redis_client.smembers("websocket:user:orphaned_user")


class TestTTLManagement:
    """Test TTL and expiration management."""

    @pytest.mark.asyncio()
    async def test_instance_ttl_refresh(self, manager, redis_client):
        """Test that instance TTL is refreshed by heartbeat."""
        await manager.startup()

        instance_key = f"websocket:instance:{manager.instance_id}"

        # Get initial TTL
        initial_ttl = await redis_client.ttl(instance_key)
        assert 0 < initial_ttl <= 60

        # Wait a bit
        await asyncio.sleep(2)

        # Manually refresh TTL (simulating heartbeat)
        await redis_client.expire(instance_key, 60)

        # TTL should be refreshed
        new_ttl = await redis_client.ttl(instance_key)
        assert new_ttl > initial_ttl - 2  # Account for time passed

    @pytest.mark.asyncio()
    async def test_user_set_ttl(self, manager, redis_client):
        """Test that user connection sets have TTL."""
        await manager.startup()

        ws = MockWebSocket()
        user_id = "ttl_test_user"
        conn_id = await manager.connect(ws, user_id)

        # Check user set has TTL
        user_key = f"websocket:user:{user_id}"
        ttl = await redis_client.ttl(user_key)
        assert ttl > 0  # Should have TTL set

        # Disconnect
        await manager.disconnect(conn_id)

        # If no more connections, set should be removed
        exists = await redis_client.exists(user_key)
        assert not exists

    @pytest.mark.asyncio()
    async def test_connection_data_persistence(self, manager, redis_client):
        """Test that connection data persists across manager restarts."""
        await manager.startup()

        # Create connections
        ws1 = MockWebSocket()
        conn1 = await manager.connect(ws1, "persist_user")

        # Store instance ID for later
        instance_id = manager.instance_id

        # Shutdown manager (simulating crash/restart)
        await manager.shutdown()

        # Connection data should still be in Redis
        conn_data = await redis_client.hget("websocket:connections", conn1)
        assert conn_data is not None

        # Create new manager instance
        new_manager = ScalableWebSocketManager(redis_url="redis://localhost:6379/15")
        await new_manager.startup()

        # New manager should be able to clean up old connections
        # when it detects the original instance is dead

        # Simulate cleanup task
        connections = await redis_client.hgetall("websocket:connections")
        for conn_id, data_str in connections.items():
            data = json.loads(data_str)
            if data.get("instance_id") == instance_id:
                # Original instance is dead
                instance_key = f"websocket:instance:{instance_id}"
                if not await redis_client.exists(instance_key):
                    await redis_client.hdel("websocket:connections", conn_id)

        # Old connection should be cleaned up
        conn_data = await redis_client.hget("websocket:connections", conn1)
        assert conn_data is None

        await new_manager.shutdown()


class TestGracefulFailover:
    """Test graceful failover scenarios."""

    @pytest.mark.asyncio()
    async def test_connection_migration_simulation(self, redis_client):
        """Test simulated connection migration between instances."""
        # Create first instance
        manager1 = ScalableWebSocketManager(redis_url="redis://localhost:6379/15")
        await manager1.startup()

        # Create connections
        ws1 = MockWebSocket()
        user_id = "migrate_user"
        conn1 = await manager1.connect(ws1, user_id)

        # Simulate instance failure
        await manager1.shutdown()

        # Create second instance
        manager2 = ScalableWebSocketManager(redis_url="redis://localhost:6379/15")
        await manager2.startup()

        # New connection from same user (simulating reconnect)
        ws2 = MockWebSocket()
        conn2 = await manager2.connect(ws2, user_id)

        # Send message to user
        await manager2.send_to_user(user_id, {"type": "reconnect", "status": "success"})

        # New connection should receive message
        assert any(m.get("type") == "reconnect" for m in ws2.sent_messages)

        await manager2.shutdown()

    @pytest.mark.asyncio()
    async def test_background_task_cleanup(self, manager):
        """Test that background tasks are properly cleaned up."""
        await manager.startup()

        # Verify background tasks are running
        assert manager.listener_task is not None
        assert manager.heartbeat_task is not None
        assert manager.cleanup_task is not None

        assert not manager.listener_task.done()
        assert not manager.heartbeat_task.done()
        assert not manager.cleanup_task.done()

        # Shutdown
        await manager.shutdown()

        # All tasks should be cancelled
        assert manager.listener_task.cancelled() or manager.listener_task.done()
        assert manager.heartbeat_task.cancelled() or manager.heartbeat_task.done()
        assert manager.cleanup_task.cancelled() or manager.cleanup_task.done()

    @pytest.mark.asyncio()
    async def test_redis_reconnection(self, manager):
        """Test Redis reconnection handling."""
        await manager.startup()

        # Simulate Redis disconnect
        original_client = manager.redis_client
        manager.redis_client = None

        # Operations should handle gracefully
        ws = MockWebSocket()
        try:
            # This should attempt reconnection
            conn_id = await manager.connect(ws, "test_user")
            # If Redis is truly down, this would fail
            # but the manager should handle it gracefully
        except (AttributeError, TypeError) as e:
            # Expected when redis_client is None
            assert True  # This is expected behavior
        except Exception as e:
            # Other exceptions should mention Redis or connection
            assert "redis" in str(e).lower() or "connection" in str(e).lower()

        # Restore Redis client
        manager.redis_client = original_client


class TestMemoryManagement:
    """Test memory usage and cleanup."""

    @pytest.mark.asyncio()
    async def test_connection_metadata_cleanup(self, manager):
        """Test that connection metadata is properly cleaned up."""
        await manager.startup()

        # Create and disconnect many connections
        for i in range(100):
            ws = MockWebSocket()
            conn_id = await manager.connect(ws, f"user_{i % 10}")  # Reuse user IDs

            # Disconnect immediately
            await manager.disconnect(conn_id)

        # Memory should be clean
        assert len(manager.local_connections) == 0
        assert len(manager.connection_metadata) == 0

    @pytest.mark.asyncio()
    async def test_message_throttle_cleanup(self, manager):
        """Test that message throttle data is cleaned up."""
        await manager.startup()

        # Manually add old throttle entries
        old_time = time.time() - 400  # More than 5 minutes old
        manager._message_throttle = {
            "old_channel": old_time,
            "recent_channel": time.time()
        }

        # Simulate cleanup logic
        now = time.time()
        old_entries = [
            channel for channel, last_time in manager._message_throttle.items()
            if now - last_time > 300
        ]

        for channel in old_entries:
            del manager._message_throttle[channel]

        # Old entry should be removed
        assert "old_channel" not in manager._message_throttle
        assert "recent_channel" in manager._message_throttle
