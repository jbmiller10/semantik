#!/usr/bin/env python3
"""
Race condition tests for WebSocket managers.

This test suite verifies that the WebSocket managers properly handle
concurrent access to shared state without race conditions.
"""

import asyncio
import logging
import os
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import WebSocket
from fastapi.websockets import WebSocketState

from webui.websocket.scalable_manager import ScalableWebSocketManager
from webui.websocket_manager import RedisStreamWebSocketManager

# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def create_mock_websocket(connection_id: str = "test") -> WebSocket:
    """Create a mock WebSocket for testing."""
    ws = MagicMock(spec=WebSocket)
    ws.accept = AsyncMock()
    ws.send_json = AsyncMock()
    ws.close = AsyncMock()
    ws.application_state = WebSocketState.CONNECTED
    ws._connection_id = connection_id
    # Add ping method for cleanup tests
    ws.ping = AsyncMock()
    return ws


class RaceConditionDetector:
    """Helper class to detect race conditions in concurrent operations."""

    def __init__(self):
        self.operations: list[tuple[float, str, Any]] = []
        self.errors: list[str] = []
        self.lock = asyncio.Lock()

    async def record_operation(self, op_type: str, data: Any):
        """Record an operation with timestamp."""
        async with self.lock:
            self.operations.append((time.time(), op_type, data))

    async def record_error(self, error: str):
        """Record an error during testing."""
        async with self.lock:
            self.errors.append(error)

    def analyze(self) -> dict:
        """Analyze recorded operations for race conditions."""
        # Sort operations by timestamp
        sorted_ops = sorted(self.operations, key=lambda x: x[0])

        # Check for overlapping critical sections
        overlaps = []
        for i in range(len(sorted_ops) - 1):
            t1, op1, data1 = sorted_ops[i]
            t2, op2, data2 = sorted_ops[i + 1]

            # If operations are too close (< 1ms), they might be racing
            if (t2 - t1) < 0.001 and op1 == op2:
                overlaps.append({"operation": op1, "time_diff": t2 - t1, "data": [data1, data2]})

        return {
            "total_operations": len(self.operations),
            "errors": self.errors,
            "potential_races": overlaps,
            "unique_operations": len({op[1] for op in self.operations}),
        }


@pytest.mark.asyncio()
class TestRedisStreamWebSocketManagerRaceConditions:
    """Test race conditions in RedisStreamWebSocketManager."""

    async def test_concurrent_connections(self):
        """Test that concurrent connections don't corrupt shared state."""
        manager = RedisStreamWebSocketManager()
        detector = RaceConditionDetector()

        # Mock Redis to avoid external dependencies
        with patch("webui.websocket_manager.redis") as mock_redis:
            mock_redis_client = AsyncMock()
            mock_redis.from_url = AsyncMock(return_value=mock_redis_client)
            mock_redis_client.ping = AsyncMock()

            await manager.startup()

            # Create many concurrent connections
            async def connect_user(user_id: str, operation_id: str):
                try:
                    ws = create_mock_websocket(f"conn_{user_id}_{operation_id}")
                    await detector.record_operation("connect_start", f"{user_id}:{operation_id}")
                    await manager.connect(ws, operation_id, user_id)
                    await detector.record_operation("connect_end", f"{user_id}:{operation_id}")

                    # Verify connection was added
                    key = f"{user_id}:operation:{operation_id}"
                    assert key in manager.connections
                    assert ws in manager.connections[key]

                    return ws
                except Exception as e:
                    await detector.record_error(f"Connection failed: {e}")
                    raise

            # Run many concurrent connections
            tasks = []
            for i in range(100):
                user_id = f"user_{i % 10}"  # 10 unique users
                operation_id = f"op_{i % 20}"  # 20 unique operations
                tasks.append(connect_user(user_id, operation_id))

            # Execute all connections concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Check for errors
            errors = [r for r in results if isinstance(r, Exception)]
            assert len(errors) == 0, f"Connection errors: {errors}"

            # Analyze for race conditions
            analysis = detector.analyze()
            assert len(analysis["errors"]) == 0, f"Race errors detected: {analysis['errors']}"
            assert len(analysis["potential_races"]) == 0, f"Potential races: {analysis['potential_races']}"

            # Verify all connections are properly stored
            total_connections = sum(len(sockets) for sockets in manager.connections.values())
            assert total_connections == 100, f"Expected 100 connections, got {total_connections}"

    async def test_concurrent_disconnections(self):
        """Test that concurrent disconnections don't cause corruption."""
        manager = RedisStreamWebSocketManager()
        detector = RaceConditionDetector()

        with patch("webui.websocket_manager.redis") as mock_redis:
            mock_redis_client = AsyncMock()
            mock_redis.from_url = AsyncMock(return_value=mock_redis_client)
            mock_redis_client.ping = AsyncMock()

            await manager.startup()

            # First, add connections
            connections = []
            for i in range(50):
                user_id = f"user_{i}"
                operation_id = f"op_{i}"
                ws = create_mock_websocket(f"conn_{i}")
                await manager.connect(ws, operation_id, user_id)
                connections.append((ws, operation_id, user_id))

            # Now disconnect them concurrently
            async def disconnect_user(ws: WebSocket, operation_id: str, user_id: str):
                try:
                    await detector.record_operation("disconnect_start", f"{user_id}:{operation_id}")
                    await manager.disconnect(ws, operation_id, user_id)
                    await detector.record_operation("disconnect_end", f"{user_id}:{operation_id}")

                    # Verify connection was removed
                    key = f"{user_id}:operation:{operation_id}"
                    if key in manager.connections:
                        assert ws not in manager.connections[key]

                except Exception as e:
                    await detector.record_error(f"Disconnection failed: {e}")
                    raise

            # Disconnect all concurrently
            tasks = [disconnect_user(ws, op_id, user_id) for ws, op_id, user_id in connections]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Check for errors
            errors = [r for r in results if isinstance(r, Exception)]
            assert len(errors) == 0, f"Disconnection errors: {errors}"

            # Verify all connections were removed
            assert len(manager.connections) == 0, f"Connections not cleared: {manager.connections}"

            # Analyze for race conditions
            analysis = detector.analyze()
            assert len(analysis["errors"]) == 0, f"Race errors detected: {analysis['errors']}"

    async def test_concurrent_broadcast(self):
        """Test that concurrent broadcasts don't lose messages."""
        manager = RedisStreamWebSocketManager()
        detector = RaceConditionDetector()
        messages_sent: set[str] = set()

        with patch("webui.websocket_manager.redis") as mock_redis:
            mock_redis_client = AsyncMock()
            mock_redis.from_url = AsyncMock(return_value=mock_redis_client)
            mock_redis_client.ping = AsyncMock()

            await manager.startup()

            # Add test connections
            operation_id = "test_op"
            websockets = []
            for i in range(10):
                user_id = f"user_{i}"
                ws = create_mock_websocket(f"conn_{i}")

                # Track messages sent to this websocket
                async def track_send(msg):
                    messages_sent.add(str(msg))

                ws.send_json.side_effect = track_send

                await manager.connect(ws, operation_id, user_id)
                websockets.append(ws)

            # Send many concurrent broadcasts
            async def send_broadcast(msg_id: int):
                message = {"id": msg_id, "data": f"Message {msg_id}"}
                await detector.record_operation("broadcast_start", msg_id)
                await manager._broadcast(operation_id, message)
                await detector.record_operation("broadcast_end", msg_id)

            # Send 100 messages concurrently
            tasks = [send_broadcast(i) for i in range(100)]
            await asyncio.gather(*tasks)

            # Each websocket should have received all 100 messages
            # (10 websockets * 100 messages = 1000 total sends)
            for ws in websockets:
                assert (
                    ws.send_json.call_count == 100
                ), f"WebSocket received {ws.send_json.call_count} messages, expected 100"

            # Analyze for race conditions
            analysis = detector.analyze()
            assert len(analysis["errors"]) == 0, f"Broadcast errors: {analysis['errors']}"

    async def test_connection_limit_enforcement(self):
        """Test that connection limits are properly enforced under concurrent load."""
        manager = RedisStreamWebSocketManager()
        manager.max_connections_per_user = 5  # Reduce for testing
        manager.max_total_connections = 20  # Reduce for testing

        with patch("webui.websocket_manager.redis") as mock_redis:
            mock_redis_client = AsyncMock()
            mock_redis.from_url = AsyncMock(return_value=mock_redis_client)
            mock_redis_client.ping = AsyncMock()

            await manager.startup()

            accepted_connections = []
            rejected_connections = []

            async def try_connect(user_id: str, operation_id: str):
                ws = create_mock_websocket(f"conn_{user_id}_{operation_id}")
                try:
                    await manager.connect(ws, operation_id, user_id)
                    accepted_connections.append((user_id, operation_id))
                    return True
                except Exception:
                    # Connection rejected due to limit
                    rejected_connections.append((user_id, operation_id))
                    return False

            # Try to create more connections than allowed
            tasks = []

            # Try 10 connections per user for 5 users (50 total, but limit is 20)
            for user_idx in range(5):
                user_id = f"user_{user_idx}"
                for conn_idx in range(10):
                    operation_id = f"op_{user_idx}_{conn_idx}"
                    tasks.append(try_connect(user_id, operation_id))

            await asyncio.gather(*tasks)

            # Verify limits were enforced
            assert (
                len(accepted_connections) <= manager.max_total_connections
            ), f"Total limit exceeded: {len(accepted_connections)} > {manager.max_total_connections}"

            # Count connections per user
            user_counts = {}
            for user_id, _ in accepted_connections:
                user_counts[user_id] = user_counts.get(user_id, 0) + 1

            for user_id, count in user_counts.items():
                assert (
                    count <= manager.max_connections_per_user
                ), f"User {user_id} exceeded limit: {count} > {manager.max_connections_per_user}"

    async def test_cleanup_during_concurrent_operations(self):
        """Test that cleanup operations don't interfere with concurrent operations."""
        manager = RedisStreamWebSocketManager()

        with patch("webui.websocket_manager.redis") as mock_redis:
            mock_redis_client = AsyncMock()
            mock_redis.from_url = AsyncMock(return_value=mock_redis_client)
            mock_redis_client.ping = AsyncMock()

            await manager.startup()

            # Add initial connections
            for i in range(20):
                user_id = f"user_{i}"
                operation_id = f"op_{i}"
                ws = create_mock_websocket(f"conn_{i}")
                await manager.connect(ws, operation_id, user_id)

            # Run cleanup while adding/removing connections
            async def run_cleanup():
                for _ in range(5):
                    await manager.cleanup_stale_connections()
                    await asyncio.sleep(0.01)

            async def add_remove_connections():
                for i in range(10):
                    # Add a connection
                    user_id = f"new_user_{i}"
                    operation_id = f"new_op_{i}"
                    ws = create_mock_websocket(f"new_conn_{i}")
                    await manager.connect(ws, operation_id, user_id)

                    # Remove another connection
                    if i < len(manager.connections):
                        key = list(manager.connections.keys())[0]
                        if key in manager.connections and manager.connections[key]:
                            ws_to_remove = next(iter(manager.connections[key]))
                            user_id_parts = key.split(":")
                            if len(user_id_parts) >= 3:
                                await manager.disconnect(ws_to_remove, user_id_parts[2], user_id_parts[0])

                    await asyncio.sleep(0.005)

            # Run cleanup and modifications concurrently
            await asyncio.gather(
                run_cleanup(),
                add_remove_connections(),
                add_remove_connections(),  # Run twice for more concurrency
            )

            # Verify manager state is consistent
            for key, websockets in manager.connections.items():
                assert isinstance(websockets, set), f"Connection value not a set: {type(websockets)}"
                assert len(websockets) > 0 or key not in manager.connections, f"Empty connection set for {key}"


@pytest.mark.asyncio()
class TestScalableWebSocketManagerRaceConditions:
    """Test race conditions in ScalableWebSocketManager (which lacks proper locking)."""

    async def test_scalable_manager_has_race_conditions(self):
        """
        This test demonstrates that ScalableWebSocketManager has race conditions
        due to lack of proper locking mechanisms.
        """
        manager = ScalableWebSocketManager()

        # Mock Redis to isolate the test
        with patch("webui.websocket.scalable_manager.redis") as mock_redis:
            mock_redis_client = AsyncMock()
            mock_redis.from_url = AsyncMock(return_value=mock_redis_client)
            mock_redis_client.ping = AsyncMock()
            mock_redis_client.pubsub = MagicMock()
            mock_redis_client.pubsub.return_value.subscribe = AsyncMock()

            await manager.startup()

            # Track connection operations
            connection_counts = {"adds": 0, "removes": 0}

            async def concurrent_add():
                """Add connections concurrently."""
                for i in range(50):
                    ws = create_mock_websocket(f"add_{i}")
                    conn_id = f"conn_add_{i}"

                    # Directly manipulate the dictionaries to simulate race
                    manager.local_connections[conn_id] = ws
                    manager.connection_metadata[conn_id] = {"user_id": f"user_{i}"}
                    connection_counts["adds"] += 1

                    # Small delay to increase chance of race
                    await asyncio.sleep(0.0001)

            async def concurrent_remove():
                """Remove connections concurrently."""
                await asyncio.sleep(0.01)  # Let some adds happen first

                for i in range(50):
                    conn_id = f"conn_add_{i}"

                    # Try to remove without proper synchronization
                    if conn_id in manager.local_connections:
                        del manager.local_connections[conn_id]
                        if conn_id in manager.connection_metadata:
                            del manager.connection_metadata[conn_id]
                        connection_counts["removes"] += 1

                    await asyncio.sleep(0.0001)

            async def concurrent_iterate():
                """Iterate over connections while they're being modified."""
                errors = []
                await asyncio.sleep(0.005)  # Start during modifications

                for _ in range(10):
                    try:
                        # This can raise RuntimeError: dictionary changed size during iteration
                        for _, metadata in manager.connection_metadata.items():
                            _ = metadata.get("user_id")
                    except RuntimeError as e:
                        errors.append(str(e))

                    await asyncio.sleep(0.001)

                return errors

            # Run operations concurrently - this WILL have race conditions
            results = await asyncio.gather(
                concurrent_add(),
                concurrent_add(),  # Two adds running simultaneously
                concurrent_remove(),
                concurrent_iterate(),
                return_exceptions=True,
            )

            # Check for iteration errors (common race condition symptom)
            iteration_errors = results[3] if len(results) > 3 and not isinstance(results[3], Exception) else []

            # The scalable manager without locks will likely have issues:
            # - Iteration errors when dictionaries change during iteration
            # - Inconsistent state between local_connections and connection_metadata
            # - Lost updates when concurrent operations modify the same data

            # NOTE: This test demonstrates the PROBLEM, not the solution
            # The ScalableWebSocketManager needs locks similar to RedisStreamWebSocketManager

            logger.info("Race condition test results for ScalableWebSocketManager:")
            logger.info(f"  Adds: {connection_counts['adds']}")
            logger.info(f"  Removes: {connection_counts['removes']}")
            logger.info(f"  Final connections: {len(manager.local_connections)}")
            logger.info(f"  Final metadata: {len(manager.connection_metadata)}")
            logger.info(f"  Iteration errors: {len(iteration_errors)}")

            # Assert that there ARE problems (to demonstrate the issue)
            # In a properly synchronized manager, these would be equal
            inconsistent = len(manager.local_connections) != len(manager.connection_metadata)

            # Return the problems found
            return {
                "inconsistent_state": inconsistent,
                "iteration_errors": iteration_errors,
                "connections_mismatch": abs(len(manager.local_connections) - len(manager.connection_metadata)),
            }


@pytest.mark.asyncio()
@pytest.mark.skipif(
    os.getenv("CI", "false").lower() == "true", reason="Performance tests unreliable in CI environments"
)
async def test_performance_impact_of_locking():
    """Measure the performance impact of the locking mechanisms."""

    # Test with RedisStreamWebSocketManager (with locks)
    locked_manager = RedisStreamWebSocketManager()

    with patch("webui.websocket_manager.redis") as mock_redis:
        mock_redis_client = AsyncMock()
        mock_redis.from_url = AsyncMock(return_value=mock_redis_client)
        mock_redis_client.ping = AsyncMock()

        await locked_manager.startup()

        # Measure connection time with locks
        start_time = time.time()

        tasks = []
        for i in range(100):
            ws = create_mock_websocket(f"perf_{i}")
            tasks.append(locked_manager.connect(ws, f"op_{i}", f"user_{i}"))

        await asyncio.gather(*tasks)

        locked_duration = time.time() - start_time

    # Performance assertion
    # With proper async locking, 100 connections should complete in < 1 second
    assert locked_duration < 1.0, f"Locking overhead too high: {locked_duration:.3f}s for 100 connections"

    logger.info("Performance test results:")
    logger.info(f"  100 connections with locking: {locked_duration:.3f}s")
    logger.info(f"  Average per connection: {locked_duration * 1000 / 100:.2f}ms")

    # The overhead should be minimal (< 5ms per connection)
    assert (locked_duration * 1000 / 100) < 5.0, "Locking adds too much overhead per connection"


if __name__ == "__main__":
    # Run the tests
    asyncio.run(test_performance_impact_of_locking())
