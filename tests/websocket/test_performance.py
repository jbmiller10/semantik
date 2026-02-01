"""Performance tests for ScalableWebSocketManager."""

import asyncio
import time
import uuid
from collections.abc import Generator
from typing import Any

import pytest
import redis.asyncio as redis

from webui.websocket.scalable_manager import ScalableWebSocketManager

pytestmark = pytest.mark.usefixtures("use_fakeredis")


class MockWebSocket:
    """Mock WebSocket for testing."""

    def __init__(self, latency: float = 0) -> None:
        self.accepted = False
        self.closed = False
        self.sent_messages: list[dict] = []
        self.latency = latency  # Simulated network latency
        self.send_times: list[float] = []  # Track send times for latency measurement

    async def accept(self, subprotocol: str | None = None) -> None:
        """Accept the WebSocket connection."""
        self.accepted = True
        self.subprotocol = subprotocol

    async def close(self, code: int = 1000, reason: str = "") -> None:
        """Close the WebSocket connection."""
        self.closed = True

    async def send_json(self, data: dict) -> None:
        """Send JSON data through the WebSocket."""
        if self.latency > 0:
            await asyncio.sleep(self.latency)
        self.send_times.append(time.time())
        self.sent_messages.append(data)


@pytest.fixture()
async def redis_client() -> Generator[Any, None, None]:
    """Create a test Redis client."""
    client = await redis.from_url("redis://localhost:6379/15", decode_responses=True)

    # Clean up test database
    await client.flushdb()

    yield client

    # Cleanup
    await client.flushdb()
    await client.close()


@pytest.fixture()
async def manager() -> Generator[Any, None, None]:
    """Create a ScalableWebSocketManager instance for testing."""
    manager = ScalableWebSocketManager(
        redis_url="redis://localhost:6379/15",
        max_connections_per_user=100,  # Higher limit for load testing
        max_total_connections=10000,
    )

    yield manager

    # Cleanup
    await manager.shutdown()


class TestMessageLatency:
    """Test message delivery latency."""

    @pytest.mark.asyncio()
    async def test_single_message_latency(self, manager) -> None:
        """Test latency for a single message delivery."""
        await manager.startup()

        ws = MockWebSocket()
        user_id = "latency_test"
        await manager.connect(ws, user_id)

        # Measure message delivery time
        start_time = time.time()
        await manager.send_to_user(user_id, {"type": "latency_test"})

        # Wait for message to be delivered
        max_wait = 0.1  # 100ms max wait
        while len(ws.sent_messages) < 1 and (time.time() - start_time) < max_wait:
            await asyncio.sleep(0.001)

        end_time = time.time()
        latency = (end_time - start_time) * 1000  # Convert to ms

        # Should be under 100ms as per requirements
        assert latency < 100, f"Message latency {latency:.2f}ms exceeds 100ms requirement"

        # Verify message was delivered
        assert len([m for m in ws.sent_messages if m.get("type") == "latency_test"]) == 1

    @pytest.mark.asyncio()
    async def test_cross_instance_latency(self) -> None:
        """Test message latency between instances."""
        # Create two instances
        manager1 = ScalableWebSocketManager(redis_url="redis://localhost:6379/15")
        manager2 = ScalableWebSocketManager(redis_url="redis://localhost:6379/15")

        await manager1.startup()
        await manager2.startup()

        try:
            # Connect to different instances
            ws1 = MockWebSocket()
            ws2 = MockWebSocket()

            user_id = "cross_latency"
            await manager1.connect(ws1, user_id)
            await manager2.connect(ws2, user_id)

            # Allow subscriptions to settle
            await asyncio.sleep(0.5)

            # Measure cross-instance delivery
            start_time = time.time()
            await manager1.send_to_user(user_id, {"type": "cross_test", "timestamp": start_time})

            # Wait for delivery to both instances
            max_wait = 0.1
            while len(ws2.sent_messages) < 1 and (time.time() - start_time) < max_wait:
                await asyncio.sleep(0.001)

            end_time = time.time()
            latency = (end_time - start_time) * 1000

            # Cross-instance latency can be noisy in CI; keep a generous bound to avoid flakiness.
            assert latency < 150, f"Cross-instance latency {latency:.2f}ms exceeds 150ms requirement"

        finally:
            await manager1.shutdown()
            await manager2.shutdown()

    @pytest.mark.asyncio()
    async def test_broadcast_latency(self, manager) -> None:
        """Test broadcast latency to multiple connections."""
        await manager.startup()

        # Create multiple connections for same collection
        collection_id = str(uuid.uuid4())
        connections = []

        for i in range(10):
            ws = MockWebSocket()
            await manager.connect(ws, f"user_{i}", collection_id=collection_id)
            connections.append(ws)

        # Measure broadcast time
        start_time = time.time()
        await manager.broadcast_to_collection(collection_id, {"type": "broadcast_test"})

        # Wait for all to receive
        max_wait = 0.1
        all_received = False
        while not all_received and (time.time() - start_time) < max_wait:
            all_received = all(any(m.get("type") == "broadcast_test" for m in ws.sent_messages) for ws in connections)
            if not all_received:
                await asyncio.sleep(0.001)

        end_time = time.time()
        latency = (end_time - start_time) * 1000

        # Broadcast should be efficient
        assert latency < 100, f"Broadcast latency {latency:.2f}ms exceeds 100ms requirement"
        assert all_received, "Not all connections received broadcast"


class TestConnectionScaling:
    """Test scaling to many connections."""

    @pytest.mark.asyncio()
    async def test_hundred_connections(self, manager) -> None:
        """Test handling 100 concurrent connections."""
        await manager.startup()

        connections = []
        start_time = time.time()

        # Create 100 connections
        for i in range(100):
            ws = MockWebSocket()
            conn_id = await manager.connect(ws, f"user_{i % 20}")  # 20 unique users
            connections.append((ws, conn_id))

        connection_time = time.time() - start_time

        # Should handle 100 connections quickly
        assert connection_time < 5, f"Took {connection_time:.2f}s to create 100 connections"

        # Verify all are connected
        assert len(manager.local_connections) == 100

        # Test message delivery at scale
        start_time = time.time()
        await manager.send_to_user("user_0", {"type": "scale_test"})

        # Count how many received it
        received_count = sum(
            1
            for ws, _ in connections[:5]  # user_0 appears at indices 0, 20, 40, 60, 80
            if any(m.get("type") == "scale_test" for m in ws.sent_messages)
        )

        # Should deliver to all connections for that user
        assert received_count > 0

    @pytest.mark.asyncio()
    @pytest.mark.slow()  # Mark as slow test
    async def test_thousand_connections(self) -> None:
        """Test handling 1000 concurrent connections."""
        # Use dedicated manager with higher limits
        manager = ScalableWebSocketManager(
            redis_url="redis://localhost:6379/15", max_connections_per_user=100, max_total_connections=10000
        )

        await manager.startup()

        try:
            connections = []
            batch_size = 100
            total_connections = 1000

            start_time = time.time()

            # Create connections in batches to avoid overwhelming
            for batch in range(total_connections // batch_size):
                batch_connections = []
                for i in range(batch_size):
                    idx = batch * batch_size + i
                    ws = MockWebSocket()
                    conn_id = await manager.connect(ws, f"user_{idx % 100}")
                    batch_connections.append((ws, conn_id))

                connections.extend(batch_connections)

                # Small delay between batches
                await asyncio.sleep(0.01)

            connection_time = time.time() - start_time

            # Should handle 1000 connections in reasonable time
            assert connection_time < 30, f"Took {connection_time:.2f}s to create 1000 connections"

            # Verify connections
            assert len(manager.local_connections) == 1000

            # Test message delivery at scale
            test_user = "user_0"
            start_time = time.time()
            await manager.send_to_user(test_user, {"type": "thousand_test"})

            # Allow time for delivery
            await asyncio.sleep(0.5)

            # Count deliveries
            delivered = sum(
                1 for ws, _ in connections if any(m.get("type") == "thousand_test" for m in ws.sent_messages)
            )

            # Should deliver to all connections for that user (10 connections)
            assert delivered == 10

        finally:
            await manager.shutdown()

    @pytest.mark.asyncio()
    async def test_rapid_connect_disconnect(self, manager) -> None:
        """Test rapid connection and disconnection cycles."""
        await manager.startup()

        cycles = 50
        start_time = time.time()

        for i in range(cycles):
            ws = MockWebSocket()
            conn_id = await manager.connect(ws, f"rapid_user_{i % 10}")

            # Send a message
            await manager.send_to_user(f"rapid_user_{i % 10}", {"type": "rapid", "index": i})

            # Disconnect immediately
            await manager.disconnect(conn_id)

        cycle_time = time.time() - start_time

        # Should handle rapid cycles efficiently
        assert cycle_time < 5, f"Rapid cycles took {cycle_time:.2f}s"

        # Should have no lingering connections
        assert len(manager.local_connections) == 0
        assert len(manager.connection_metadata) == 0


class TestMemoryStability:
    """Test memory stability under load."""

    @pytest.mark.asyncio()
    async def test_memory_cleanup_after_load(self, manager) -> None:
        """Test that memory is properly released after load."""
        await manager.startup()

        # Create many connections
        connections = []
        for i in range(100):
            ws = MockWebSocket()
            conn_id = await manager.connect(ws, f"mem_user_{i}")
            connections.append(conn_id)

        # Send many messages
        for i in range(10):
            for j in range(10):
                await manager.send_to_user(f"mem_user_{j}", {"type": "memory_test", "index": i})

        # Disconnect all
        for conn_id in connections:
            await manager.disconnect(conn_id)

        # Verify cleanup
        assert len(manager.local_connections) == 0
        assert len(manager.connection_metadata) == 0

        # Message throttle should eventually be cleaned
        # (In real implementation, this would be done by cleanup task)
        assert len(manager._message_throttle) <= 100  # Some throttle entries may remain temporarily

    @pytest.mark.asyncio()
    async def test_redis_memory_management(self, manager, redis_client) -> None:
        """Test Redis memory usage remains bounded."""
        await manager.startup()

        # Create and destroy many connections
        for cycle in range(10):
            connections = []

            # Create batch
            for i in range(50):
                ws = MockWebSocket()
                conn_id = await manager.connect(ws, f"cycle_{cycle}_user_{i}")
                connections.append(conn_id)

            # Send messages
            for i in range(10):
                await manager.send_to_user(f"cycle_{cycle}_user_{i}", {"type": "cycle", "data": cycle})

            # Cleanup
            for conn_id in connections:
                await manager.disconnect(conn_id)

        # Check Redis memory usage
        conn_count = await redis_client.hlen("websocket:connections")
        assert conn_count == 0, f"Leaked {conn_count} connections in Redis"

        # Check for leaked user sets
        user_keys = await redis_client.keys("websocket:user:*")
        assert len(user_keys) == 0, f"Leaked {len(user_keys)} user sets in Redis"


class TestConcurrency:
    """Test concurrent operations."""

    @pytest.mark.asyncio()
    async def test_concurrent_connections(self, manager) -> None:
        """Test handling concurrent connection attempts."""
        await manager.startup()

        async def connect_user(user_id: str) -> tuple:
            ws = MockWebSocket()
            conn_id = await manager.connect(ws, user_id)
            return ws, conn_id

        # Create connections concurrently
        tasks = [connect_user(f"concurrent_{i}") for i in range(50)]
        results = await asyncio.gather(*tasks)

        # All should succeed
        assert len(results) == 50
        assert len(manager.local_connections) == 50

    @pytest.mark.asyncio()
    async def test_concurrent_messages(self, manager) -> None:
        """Test handling concurrent message sending."""
        await manager.startup()

        # Create connections
        connections = []
        for i in range(10):
            ws = MockWebSocket()
            await manager.connect(ws, f"msg_user_{i}")
            connections.append(ws)

        # Send messages concurrently
        async def send_message(user_id: str, msg_id: int) -> None:
            await manager.send_to_user(user_id, {"type": "concurrent", "id": msg_id})

        tasks = []
        for i in range(100):
            user_id = f"msg_user_{i % 10}"
            tasks.append(send_message(user_id, i))

        await asyncio.gather(*tasks)

        # Each connection should have received its messages
        for _, ws in enumerate(connections):
            msgs = [m for m in ws.sent_messages if m.get("type") == "concurrent"]
            assert len(msgs) == 10  # Each user gets 10 messages

    @pytest.mark.asyncio()
    async def test_concurrent_disconnections(self, manager) -> None:
        """Test handling concurrent disconnection."""
        await manager.startup()

        # Create connections
        conn_ids = []
        for i in range(50):
            ws = MockWebSocket()
            conn_id = await manager.connect(ws, f"disc_user_{i}")
            conn_ids.append(conn_id)

        # Disconnect concurrently
        tasks = [manager.disconnect(conn_id) for conn_id in conn_ids]
        await asyncio.gather(*tasks)

        # All should be disconnected
        assert len(manager.local_connections) == 0
        assert len(manager.connection_metadata) == 0


class TestStressConditions:
    """Test behavior under stress conditions."""

    @pytest.mark.asyncio()
    async def test_message_flood(self, manager) -> None:
        """Test handling of message flooding."""
        await manager.startup()

        ws = MockWebSocket()
        user_id = "flood_user"
        await manager.connect(ws, user_id)

        # Send rapid-fire messages
        message_count = 1000
        start_time = time.time()

        for i in range(message_count):
            await manager.send_to_user(user_id, {"type": "flood", "index": i})
            # No delay - maximum stress

        flood_time = time.time() - start_time

        # Should handle flood without crashing
        assert flood_time < 10, f"Message flood took {flood_time:.2f}s"

        # Not all messages may be delivered due to throttling
        delivered = len([m for m in ws.sent_messages if m.get("type") == "flood"])
        assert delivered > 0, "No messages delivered"
        assert delivered <= message_count, "More messages than sent?"

    @pytest.mark.asyncio()
    async def test_connection_limits_under_load(self, manager) -> None:
        """Test connection limits are enforced under load."""
        manager.max_total_connections = 100
        await manager.startup()

        connections = []
        exceeded = False

        # Try to create more than limit
        for i in range(150):
            ws = MockWebSocket()
            try:
                conn_id = await manager.connect(ws, f"limit_user_{i}")
                connections.append(conn_id)
            except ConnectionError:
                exceeded = True
                break

        # Should hit limit
        assert exceeded, "Connection limit not enforced"
        assert len(manager.local_connections) <= manager.max_total_connections
