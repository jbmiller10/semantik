"""Test WebSocket manager for race conditions under concurrent load."""

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from packages.webui.websocket_manager import RedisStreamWebSocketManager


class MockWebSocket:
    """Mock WebSocket for testing."""

    def __init__(self):
        self.closed = False
        self.close_code = None
        self.close_reason = None
        self.accepted = False
        self.messages = []
        self.lock = asyncio.Lock()

    async def accept(self):
        self.accepted = True

    async def close(self, code=1000, reason=""):
        async with self.lock:
            self.closed = True
            self.close_code = code
            self.close_reason = reason

    async def send_json(self, data):
        async with self.lock:
            if self.closed:
                raise RuntimeError("WebSocket is closed")
            self.messages.append(data)

    async def ping(self):
        """Mock ping for testing stale connection cleanup."""
        if self.closed:
            raise RuntimeError("Connection is closed")


@pytest.fixture()
async def manager():
    """Create a WebSocket manager instance for testing."""
    manager = RedisStreamWebSocketManager()
    # Mock Redis to focus on race condition testing
    manager.redis = AsyncMock()
    manager.redis.xadd = AsyncMock()
    manager.redis.expire = AsyncMock()
    manager.redis.xrange = AsyncMock(return_value=[])
    manager.redis.xreadgroup = AsyncMock(return_value=[])
    manager.redis.xgroup_create = AsyncMock()
    manager.redis.xinfo_stream = AsyncMock(return_value={"length": 0})
    yield manager
    await manager.shutdown()


@pytest.mark.asyncio()
async def test_concurrent_connections_race_condition(manager):
    """Test that concurrent connections don't cause race conditions."""
    user_id = "test_user"
    num_connections = 100

    async def connect_client(client_id):
        websocket = MockWebSocket()
        operation_id = f"op_{client_id}"
        try:
            await manager.connect(websocket, operation_id, user_id)
            return websocket, operation_id
        except ConnectionRefusedError:
            # Expected when limit is exceeded
            return None

    # Create many concurrent connections
    tasks = [connect_client(i) for i in range(num_connections)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Check for unexpected exceptions (not ConnectionRefusedError)
    unexpected_exceptions = [
        r for r in results if isinstance(r, Exception) and not isinstance(r, ConnectionRefusedError)
    ]
    assert (
        len(unexpected_exceptions) == 0
    ), f"Unexpected exceptions during concurrent connections: {unexpected_exceptions}"

    # Count successful connections
    successful_results = [r for r in results if r is not None and not isinstance(r, Exception)]
    accepted_count = sum(1 for r in successful_results if r[0].accepted)

    # Due to per-user limit, only first 10 should be accepted
    assert accepted_count == manager.max_connections_per_user

    # Verify no corruption in connections dictionary
    total_connections = sum(len(sockets) for sockets in manager.connections.values())
    assert total_connections == manager.max_connections_per_user


@pytest.mark.asyncio()
async def test_concurrent_disconnections_race_condition(manager):
    """Test that concurrent disconnections don't cause race conditions."""
    user_id = "test_user"
    num_connections = 10

    # Setup connections
    connections = []
    for i in range(num_connections):
        websocket = MockWebSocket()
        operation_id = f"op_{i}"
        await manager.connect(websocket, operation_id, user_id)
        connections.append((websocket, operation_id))

    # Disconnect all concurrently
    async def disconnect_client(websocket, operation_id):
        await manager.disconnect(websocket, operation_id, user_id)

    tasks = [disconnect_client(ws, op_id) for ws, op_id in connections]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Check for any exceptions
    exceptions = [r for r in results if isinstance(r, Exception)]
    assert len(exceptions) == 0, f"Exceptions during concurrent disconnections: {exceptions}"

    # Verify all connections removed
    assert len(manager.connections) == 0
    assert len(manager.consumer_tasks) == 0


@pytest.mark.asyncio()
async def test_concurrent_broadcast_race_condition(manager):
    """Test that concurrent broadcasts don't cause race conditions."""
    user_id = "test_user"
    operation_id = "test_op"
    num_connections = 5
    num_messages = 100

    # Setup connections
    websockets = []
    for _ in range(num_connections):
        websocket = MockWebSocket()
        await manager.connect(websocket, operation_id, user_id)
        websockets.append(websocket)

    # Send many concurrent broadcasts
    async def send_broadcast(msg_id):
        message = {"type": "update", "id": msg_id}
        await manager._broadcast(operation_id, message)

    tasks = [send_broadcast(i) for i in range(num_messages)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Check for any exceptions
    exceptions = [r for r in results if isinstance(r, Exception)]
    assert len(exceptions) == 0, f"Exceptions during concurrent broadcasts: {exceptions}"

    # Verify all websockets received messages (order may vary due to concurrency)
    for ws in websockets:
        assert len(ws.messages) == num_messages
        received_ids = {msg["id"] for msg in ws.messages}
        expected_ids = set(range(num_messages))
        assert received_ids == expected_ids


@pytest.mark.asyncio()
async def test_concurrent_cleanup_race_condition(manager):
    """Test that concurrent cleanup operations don't cause race conditions."""
    num_operations = 20

    # Setup operations with connections and tasks
    for i in range(num_operations):
        operation_id = f"op_{i}"
        user_id = f"user_{i}"
        websocket = MockWebSocket()
        await manager.connect(websocket, operation_id, user_id)

        # Mock consumer task
        manager.consumer_tasks[operation_id] = asyncio.create_task(asyncio.sleep(0))

    # Cleanup all operations concurrently
    async def cleanup_operation(op_id):
        await manager.cleanup_operation_channel(op_id)

    tasks = [cleanup_operation(f"op_{i}") for i in range(num_operations)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Check for any exceptions
    exceptions = [r for r in results if isinstance(r, Exception)]
    assert len(exceptions) == 0, f"Exceptions during concurrent cleanup: {exceptions}"

    # Verify everything was cleaned up
    assert len(manager.connections) == 0
    assert len(manager.consumer_tasks) == 0


@pytest.mark.asyncio()
async def test_throttling_race_condition(manager):
    """Test that concurrent throttling checks don't cause race conditions."""
    operation_id = "test_op"
    num_concurrent_checks = 100

    # Run many concurrent throttle checks
    async def check_throttle(check_id):
        message = {"type": "chunking_progress", "id": check_id}
        return await manager._should_send_progress_update(operation_id, message)

    tasks = [check_throttle(i) for i in range(num_concurrent_checks)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Check for any exceptions
    exceptions = [r for r in results if isinstance(r, Exception)]
    assert len(exceptions) == 0, f"Exceptions during concurrent throttle checks: {exceptions}"

    # At least one should have been allowed (the first one)
    allowed_count = sum(1 for r in results if r is True)
    assert allowed_count >= 1

    # Verify throttle state is consistent
    assert operation_id in manager._chunking_progress_throttle


@pytest.mark.asyncio()
async def test_stale_connection_cleanup_race_condition(manager):
    """Test that concurrent stale connection cleanup doesn't cause race conditions."""
    num_connections = 20

    # Setup mix of alive and dead connections
    for i in range(num_connections):
        websocket = MockWebSocket()
        if i % 2 == 0:
            # Simulate dead connection
            websocket.closed = True
        operation_id = f"op_{i}"
        user_id = f"user_{i}"
        key = f"{user_id}:operation:{operation_id}"

        if key not in manager.connections:
            manager.connections[key] = set()
        manager.connections[key].add(websocket)

    initial_count = sum(len(sockets) for sockets in manager.connections.values())

    # Run multiple concurrent cleanup operations
    async def run_cleanup():
        await manager.cleanup_stale_connections()

    tasks = [run_cleanup() for _ in range(5)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Check for any exceptions
    exceptions = [r for r in results if isinstance(r, Exception)]
    assert len(exceptions) == 0, f"Exceptions during concurrent cleanup: {exceptions}"

    # Verify dead connections were removed
    final_count = sum(len(sockets) for sockets in manager.connections.values())
    assert final_count < initial_count


@pytest.mark.asyncio()
async def test_connection_limit_enforcement_race_condition(manager):
    """Test that connection limits are properly enforced under concurrent load."""
    # Test global limit
    manager.max_total_connections = 50
    manager.max_connections_per_user = 100  # High per-user limit

    async def connect_user(user_id, conn_id):
        websocket = MockWebSocket()
        operation_id = f"op_{conn_id}"
        await manager.connect(websocket, operation_id, user_id)
        return websocket

    # Try to create more connections than the global limit
    tasks = []
    for i in range(60):  # More than global limit
        user_id = f"user_{i}"
        tasks.append(connect_user(user_id, i))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count accepted connections
    accepted = sum(1 for r in results if not isinstance(r, Exception) and r.accepted)

    # Should not exceed global limit
    assert accepted <= manager.max_total_connections

    # Verify dictionary consistency
    total_connections = sum(len(sockets) for sockets in manager.connections.values())
    assert total_connections <= manager.max_total_connections


@pytest.mark.asyncio()
async def test_concurrent_connect_disconnect_race_condition(manager):
    """Test that simultaneous connects and disconnects don't cause race conditions."""
    user_id = "test_user"
    operation_id = "test_op"
    num_iterations = 50

    async def connect_disconnect_cycle(iteration):
        websocket = MockWebSocket()

        try:
            # Connect
            await manager.connect(websocket, operation_id, user_id)

            # Small random delay
            await asyncio.sleep(0.001 * (iteration % 3))

            # Disconnect
            await manager.disconnect(websocket, operation_id, user_id)

            return iteration
        except ConnectionRefusedError:
            # Expected when we hit connection limits during concurrent attempts
            return None

    # Run many concurrent connect/disconnect cycles
    tasks = [connect_disconnect_cycle(i) for i in range(num_iterations)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Check for unexpected exceptions (not ConnectionRefusedError)
    unexpected_exceptions = [
        r for r in results if isinstance(r, Exception) and not isinstance(r, ConnectionRefusedError)
    ]
    assert (
        len(unexpected_exceptions) == 0
    ), f"Unexpected exceptions during connect/disconnect cycles: {unexpected_exceptions}"

    # Final state should be clean
    assert len(manager.connections) == 0

    # Consumer tasks should be cleaned up
    assert len(manager.consumer_tasks) == 0


@pytest.mark.asyncio()
async def test_concurrent_channel_operations_race_condition(manager):
    """Test that concurrent channel operations don't cause race conditions."""
    channel = "test_channel"
    num_users = 20

    async def channel_operation(user_id):
        websocket = MockWebSocket()

        # Connect to channel
        await manager.connect_to_channel(websocket, channel, user_id)

        # Send message
        message = {"type": "channel_msg", "user": user_id}
        await manager._broadcast_to_channel(channel, message)

        # Disconnect
        await manager.disconnect_from_channel(websocket, channel, user_id)

        return user_id

    # Run concurrent channel operations
    tasks = [channel_operation(f"user_{i}") for i in range(num_users)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Check for any exceptions
    exceptions = [r for r in results if isinstance(r, Exception)]
    assert len(exceptions) == 0, f"Exceptions during channel operations: {exceptions}"

    # Verify clean final state
    assert len(manager.connections) == 0


@pytest.mark.asyncio()
async def test_stress_test_all_operations(manager):
    """Comprehensive stress test combining all operations concurrently."""
    num_operations = 10
    num_users = 5
    num_messages = 20

    async def user_session(user_id):
        operations = []

        # Connect to multiple operations
        for op_num in range(num_operations):
            websocket = MockWebSocket()
            operation_id = f"op_{user_id}_{op_num}"
            await manager.connect(websocket, operation_id, user_id)
            operations.append((websocket, operation_id))

        # Send broadcasts
        for msg_num in range(num_messages):
            for _, operation_id in operations:
                message = {"type": "update", "msg": msg_num}
                await manager._broadcast(operation_id, message)

        # Cleanup some operations
        for websocket, operation_id in operations[: num_operations // 2]:
            await manager.disconnect(websocket, operation_id, user_id)

        # Run stale cleanup
        await manager.cleanup_stale_connections()

        # Cleanup remaining
        for _, operation_id in operations[num_operations // 2 :]:
            await manager.cleanup_operation_channel(operation_id)

        return user_id

    # Run concurrent user sessions
    tasks = [user_session(f"user_{i}") for i in range(num_users)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Check for any exceptions
    exceptions = [r for r in results if isinstance(r, Exception)]
    assert len(exceptions) == 0, f"Exceptions during stress test: {exceptions}"

    # Verify clean final state
    assert len(manager.connections) == 0
    assert len(manager.consumer_tasks) == 0

    # Verify no lingering throttle entries for cleaned operations
    for op_id in manager._chunking_progress_throttle:
        # Should only have recent entries, not from cleaned operations
        assert (
            "op_user" not in op_id
            or datetime.now(UTC).timestamp() - manager._chunking_progress_throttle[op_id].timestamp() < 10
        )
