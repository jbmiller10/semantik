"""Unit tests for ScalableWebSocketManager."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from webui.websocket.scalable_manager import (
    ScalableWebSocketManager,
    TooManyConnectionsError,
)


@pytest.fixture
def manager():
    """Create a ScalableWebSocketManager instance."""
    mgr = ScalableWebSocketManager(
        redis_url="redis://localhost:6379/2",
        max_connections_per_user=5,
        max_total_connections=100,
    )
    return mgr


@pytest.fixture
def mock_redis_client():
    """Create a mock Redis client."""
    client = AsyncMock()
    client.ping = AsyncMock()
    client.setex = AsyncMock()
    client.delete = AsyncMock()
    client.hgetall = AsyncMock(return_value={})
    client.hget = AsyncMock()
    client.hexists = AsyncMock()
    client.hlen = AsyncMock(return_value=0)
    client.smembers = AsyncMock(return_value=set())
    client.scard = AsyncMock(return_value=0)
    client.srem = AsyncMock()
    client.keys = AsyncMock(return_value=[])
    client.expire = AsyncMock()
    client.exists = AsyncMock(return_value=True)
    client.publish = AsyncMock()
    client.eval = AsyncMock(return_value=1)
    client.hset = AsyncMock()
    client.close = AsyncMock()

    # Pipeline mock
    pipeline = AsyncMock()
    pipeline.__aenter__ = AsyncMock(return_value=pipeline)
    pipeline.__aexit__ = AsyncMock(return_value=None)
    pipeline.setex = MagicMock()
    pipeline.expire = MagicMock()
    pipeline.execute = AsyncMock(return_value=[])
    client.pipeline = MagicMock(return_value=pipeline)

    # Scan iter mock
    async def scan_iter_mock(*args, **kwargs):
        return
        yield  # Make it an async generator

    client.scan_iter = scan_iter_mock

    return client


@pytest.fixture
def mock_pubsub():
    """Create a mock PubSub client."""
    pubsub = AsyncMock()
    pubsub.subscribe = AsyncMock()
    pubsub.unsubscribe = AsyncMock()
    pubsub.aclose = AsyncMock()

    async def listen_mock():
        # Yield nothing, just make it an async generator
        return
        yield

    pubsub.listen = listen_mock
    return pubsub


@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket."""
    ws = AsyncMock()
    ws.accept = AsyncMock()
    ws.close = AsyncMock()
    ws.send_json = AsyncMock()
    return ws


# -----------------------------------------------------------------------------
# Test initialization
# -----------------------------------------------------------------------------


def test_init_sets_defaults(manager):
    """Test manager initializes with correct defaults."""
    assert manager.max_connections_per_user == 5
    assert manager.max_total_connections == 100
    assert manager.redis_url == "redis://localhost:6379/2"
    assert manager.local_connections == {}
    assert manager.connection_metadata == {}


def test_init_creates_unique_instance_id(manager):
    """Test each manager gets a unique instance ID."""
    manager2 = ScalableWebSocketManager()
    assert manager.instance_id != manager2.instance_id


# -----------------------------------------------------------------------------
# Test startup
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_startup_connects_to_redis(manager, mock_redis_client, mock_pubsub):
    """Test startup connects to Redis and starts background tasks."""
    with patch("redis.asyncio.from_url", new_callable=AsyncMock) as mock_from_url:
        mock_from_url.return_value = mock_redis_client
        # pubsub() returns a sync value, but subscribe is async
        mock_redis_client.pubsub = MagicMock(return_value=mock_pubsub)

        await manager.startup()

        assert manager._startup_complete is True
        assert manager.redis_client is mock_redis_client
        mock_redis_client.ping.assert_awaited_once()


@pytest.mark.asyncio
async def test_startup_idempotent(manager, mock_redis_client, mock_pubsub):
    """Test calling startup twice only connects once."""
    with patch("redis.asyncio.from_url", new_callable=AsyncMock) as mock_from_url:
        mock_from_url.return_value = mock_redis_client
        mock_redis_client.pubsub = MagicMock(return_value=mock_pubsub)

        await manager.startup()
        await manager.startup()

        # Should only be called once
        assert mock_from_url.await_count == 1


@pytest.mark.asyncio
async def test_startup_retries_on_failure(manager, mock_redis_client, mock_pubsub):
    """Test startup retries on connection failure."""
    call_count = 0

    async def failing_then_succeeding(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise Exception("Connection failed")
        return mock_redis_client

    with patch("redis.asyncio.from_url", side_effect=failing_then_succeeding):
        mock_redis_client.pubsub.return_value = mock_pubsub
        mock_redis_client.ping = AsyncMock()

        await manager.startup()

        assert call_count == 3


@pytest.mark.asyncio
async def test_startup_falls_back_to_local_mode(manager):
    """Test startup falls back to local mode after max retries."""
    with patch("redis.asyncio.from_url", side_effect=Exception("Connection failed")):
        await manager.startup()

        assert manager.redis_client is None
        assert manager._startup_complete is False


# -----------------------------------------------------------------------------
# Test shutdown
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shutdown_closes_connections(manager, mock_websocket):
    """Test shutdown closes all connections."""
    manager.local_connections["conn-1"] = mock_websocket
    manager.connection_metadata["conn-1"] = {"user_id": "user-1"}
    manager.redis_client = None  # Local mode

    await manager.shutdown()

    mock_websocket.close.assert_awaited_once()
    assert manager.local_connections == {}
    assert manager.connection_metadata == {}


@pytest.mark.asyncio
async def test_shutdown_cancels_background_tasks(manager, mock_redis_client):
    """Test shutdown cancels background tasks."""
    manager.redis_client = mock_redis_client

    # Create mock tasks
    async def never_ending():
        while True:
            await asyncio.sleep(1)

    manager.listener_task = asyncio.create_task(never_ending())
    manager.heartbeat_task = asyncio.create_task(never_ending())
    manager.cleanup_task = asyncio.create_task(never_ending())

    await manager.shutdown()

    assert manager.listener_task.cancelled() or manager.listener_task.done()
    assert manager.heartbeat_task.cancelled() or manager.heartbeat_task.done()
    assert manager.cleanup_task.cancelled() or manager.cleanup_task.done()


# -----------------------------------------------------------------------------
# Test connect
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_connect_accepts_websocket(manager, mock_websocket):
    """Test connect accepts the websocket and returns connection ID."""
    manager._startup_complete = True
    manager.redis_client = None  # Local mode

    conn_id = await manager.connect(
        websocket=mock_websocket,
        user_id="user-1",
    )

    assert conn_id is not None
    mock_websocket.accept.assert_awaited_once_with(subprotocol=None)
    assert conn_id in manager.local_connections
    assert manager.connection_metadata[conn_id]["user_id"] == "user-1"


@pytest.mark.asyncio
async def test_connect_with_operation_id(manager, mock_websocket):
    """Test connect stores operation_id in metadata."""
    manager._startup_complete = True
    manager.redis_client = None

    conn_id = await manager.connect(
        websocket=mock_websocket,
        user_id="user-1",
        operation_id="op-123",
    )

    assert manager.connection_metadata[conn_id]["operation_id"] == "op-123"


@pytest.mark.asyncio
async def test_connect_with_collection_id(manager, mock_websocket):
    """Test connect stores collection_id in metadata."""
    manager._startup_complete = True
    manager.redis_client = None

    conn_id = await manager.connect(
        websocket=mock_websocket,
        user_id="user-1",
        collection_id="col-456",
    )

    assert manager.connection_metadata[conn_id]["collection_id"] == "col-456"


@pytest.mark.asyncio
async def test_connect_with_subprotocol(manager, mock_websocket):
    """Test connect passes subprotocol to accept."""
    manager._startup_complete = True
    manager.redis_client = None

    await manager.connect(
        websocket=mock_websocket,
        user_id="user-1",
        subprotocol="graphql-ws",
    )

    mock_websocket.accept.assert_awaited_once_with(subprotocol="graphql-ws")


@pytest.mark.asyncio
async def test_connect_rejects_at_total_limit(manager, mock_websocket):
    """Test connect rejects when total connection limit reached."""
    manager._startup_complete = True
    manager.redis_client = None
    manager.max_total_connections = 1

    # Add one connection to reach limit
    manager.local_connections["existing"] = MagicMock()

    with pytest.raises(ConnectionError, match="Server connection limit exceeded"):
        await manager.connect(websocket=mock_websocket, user_id="user-1")

    mock_websocket.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_connect_rejects_at_user_limit(manager, mock_websocket):
    """Test connect rejects when user connection limit reached."""
    manager._startup_complete = True
    manager.redis_client = None
    manager.max_connections_per_user = 2

    # Add connections for user-1 to reach limit
    for i in range(2):
        manager.local_connections[f"conn-{i}"] = MagicMock()
        manager.connection_metadata[f"conn-{i}"] = {"user_id": "user-1"}

    with pytest.raises(ConnectionError, match="User connection limit exceeded"):
        await manager.connect(websocket=mock_websocket, user_id="user-1")


@pytest.mark.asyncio
async def test_connect_registers_in_redis(manager, mock_websocket, mock_redis_client):
    """Test connect registers connection in Redis."""
    manager._startup_complete = True
    manager.redis_client = mock_redis_client

    conn_id = await manager.connect(
        websocket=mock_websocket,
        user_id="user-1",
    )

    mock_redis_client.eval.assert_awaited_once()
    assert conn_id in manager.local_connections


@pytest.mark.asyncio
async def test_connect_handles_redis_too_many_connections(manager, mock_websocket, mock_redis_client):
    """Test connect handles TooManyConnectionsError from Redis."""
    manager._startup_complete = True
    manager.redis_client = mock_redis_client
    mock_redis_client.eval.return_value = 0  # Indicates limit exceeded

    with pytest.raises(ConnectionError, match="User connection limit exceeded"):
        await manager.connect(websocket=mock_websocket, user_id="user-1")


@pytest.mark.asyncio
async def test_connect_handles_redis_registration_failure(manager, mock_websocket, mock_redis_client):
    """Test connect continues when Redis registration fails."""
    manager._startup_complete = True
    manager.redis_client = mock_redis_client
    mock_redis_client.eval.side_effect = Exception("Redis error")

    # Should still succeed in local mode
    conn_id = await manager.connect(
        websocket=mock_websocket,
        user_id="user-1",
    )

    assert conn_id in manager.local_connections


# -----------------------------------------------------------------------------
# Test disconnect
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disconnect_removes_connection(manager, mock_websocket):
    """Test disconnect removes connection from local tracking."""
    conn_id = "conn-123"
    manager.local_connections[conn_id] = mock_websocket
    manager.connection_metadata[conn_id] = {"user_id": "user-1"}
    manager.redis_client = None

    await manager.disconnect(conn_id)

    assert conn_id not in manager.local_connections
    assert conn_id not in manager.connection_metadata


@pytest.mark.asyncio
async def test_disconnect_nonexistent_connection(manager):
    """Test disconnect handles nonexistent connection gracefully."""
    await manager.disconnect("nonexistent")
    # Should not raise


@pytest.mark.asyncio
async def test_disconnect_unregisters_from_redis(manager, mock_websocket, mock_redis_client):
    """Test disconnect removes connection from Redis."""
    conn_id = "conn-123"
    manager.local_connections[conn_id] = mock_websocket
    manager.connection_metadata[conn_id] = {"user_id": "user-1"}
    manager.redis_client = mock_redis_client
    manager.pubsub = AsyncMock()
    manager.pubsub.unsubscribe = AsyncMock()

    await manager.disconnect(conn_id)

    mock_redis_client.eval.assert_awaited()


# -----------------------------------------------------------------------------
# Test send_to_user
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_to_user_sends_to_local(manager, mock_websocket):
    """Test send_to_user sends to local connections."""
    conn_id = "conn-123"
    manager.local_connections[conn_id] = mock_websocket
    manager.connection_metadata[conn_id] = {"user_id": "user-1"}
    manager.redis_client = None

    await manager.send_to_user("user-1", {"type": "test", "data": "hello"})

    mock_websocket.send_json.assert_awaited_once_with({"type": "test", "data": "hello"})


@pytest.mark.asyncio
async def test_send_to_user_handles_send_failure(manager, mock_websocket):
    """Test send_to_user disconnects on send failure."""
    conn_id = "conn-123"
    manager.local_connections[conn_id] = mock_websocket
    manager.connection_metadata[conn_id] = {"user_id": "user-1"}
    manager.redis_client = None
    mock_websocket.send_json.side_effect = Exception("Send failed")

    await manager.send_to_user("user-1", {"type": "test"})

    # Connection should be removed
    assert conn_id not in manager.local_connections


@pytest.mark.asyncio
async def test_send_to_user_publishes_to_redis(manager, mock_redis_client):
    """Test send_to_user publishes to Redis for remote connections."""
    manager.redis_client = mock_redis_client
    mock_redis_client.smembers.return_value = {"remote-conn"}
    mock_redis_client.hget.return_value = json.dumps({"instance_id": "other-instance", "user_id": "user-1"})

    await manager.send_to_user("user-1", {"type": "test"})

    mock_redis_client.publish.assert_awaited()


# -----------------------------------------------------------------------------
# Test send_to_operation
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_to_operation_sends_to_local(manager, mock_websocket):
    """Test send_to_operation sends to local connections watching operation."""
    conn_id = "conn-123"
    manager.local_connections[conn_id] = mock_websocket
    manager.connection_metadata[conn_id] = {
        "user_id": "user-1",
        "operation_id": "op-456",
    }
    manager.redis_client = None

    await manager.send_to_operation("op-456", {"type": "progress", "percent": 50})

    mock_websocket.send_json.assert_awaited_once()


@pytest.mark.asyncio
async def test_send_to_operation_publishes_to_redis(manager, mock_redis_client):
    """Test send_to_operation publishes to Redis."""
    manager.redis_client = mock_redis_client

    await manager.send_to_operation("op-456", {"type": "progress"})

    mock_redis_client.publish.assert_awaited()
    call_args = mock_redis_client.publish.call_args
    assert "operation:op-456" in str(call_args)


# -----------------------------------------------------------------------------
# Test broadcast_to_collection
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_broadcast_to_collection_sends_to_local(manager, mock_websocket):
    """Test broadcast_to_collection sends to local connections."""
    conn_id = "conn-123"
    manager.local_connections[conn_id] = mock_websocket
    manager.connection_metadata[conn_id] = {
        "user_id": "user-1",
        "collection_id": "col-789",
    }
    manager.redis_client = None

    await manager.broadcast_to_collection("col-789", {"type": "update"})

    mock_websocket.send_json.assert_awaited_once()


@pytest.mark.asyncio
async def test_broadcast_to_collection_publishes_to_redis(manager, mock_redis_client):
    """Test broadcast_to_collection publishes to Redis."""
    manager.redis_client = mock_redis_client

    await manager.broadcast_to_collection("col-789", {"type": "update"})

    mock_redis_client.publish.assert_awaited()
    call_args = mock_redis_client.publish.call_args
    assert "collection:col-789" in str(call_args)


# -----------------------------------------------------------------------------
# Test get_stats
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_stats_returns_local_stats(manager, mock_websocket):
    """Test get_stats returns local connection statistics."""
    manager.local_connections["conn-1"] = mock_websocket
    manager.local_connections["conn-2"] = mock_websocket
    manager.connection_metadata["conn-1"] = {
        "user_id": "user-1",
        "operation_id": "op-1",
        "collection_id": "col-1",
    }
    manager.connection_metadata["conn-2"] = {
        "user_id": "user-2",
        "operation_id": "op-2",
        "collection_id": "col-1",
    }
    manager.redis_client = None

    stats = await manager.get_stats()

    assert stats["local_connections"] == 2
    assert stats["unique_users"] == 2
    assert stats["operations"] == 2
    assert stats["collections"] == 1


@pytest.mark.asyncio
async def test_get_stats_includes_redis_stats(manager, mock_redis_client):
    """Test get_stats includes global Redis statistics."""
    manager.redis_client = mock_redis_client
    mock_redis_client.hlen.return_value = 50
    mock_redis_client.keys.return_value = ["instance:1", "instance:2", "instance:3"]

    stats = await manager.get_stats()

    assert stats["total_connections"] == 50
    assert stats["active_instances"] == 3


@pytest.mark.asyncio
async def test_get_stats_handles_redis_error(manager, mock_redis_client):
    """Test get_stats handles Redis errors gracefully."""
    manager.redis_client = mock_redis_client
    mock_redis_client.hlen.side_effect = Exception("Redis error")

    # Should not raise
    stats = await manager.get_stats()

    assert "local_connections" in stats


# -----------------------------------------------------------------------------
# Test _handle_instance_message
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_instance_message_ping(manager):
    """Test handling ping command."""
    # Should not raise
    await manager._handle_instance_message({"command": "ping"})


@pytest.mark.asyncio
async def test_handle_instance_message_disconnect_user(manager, mock_websocket):
    """Test handling disconnect_user command."""
    conn_id = "conn-123"
    manager.local_connections[conn_id] = mock_websocket
    manager.connection_metadata[conn_id] = {"user_id": "user-1"}
    manager.redis_client = None

    await manager._handle_instance_message({"command": "disconnect_user", "user_id": "user-1"})

    mock_websocket.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_handle_instance_message_stats(manager, mock_websocket):
    """Test handling stats command."""
    manager.local_connections["conn-1"] = mock_websocket
    manager.connection_metadata["conn-1"] = {
        "user_id": "user-1",
        "connected_at": 1000,
    }

    # Should not raise
    await manager._handle_instance_message({"command": "stats"})


# -----------------------------------------------------------------------------
# Test helper methods
# -----------------------------------------------------------------------------


def test_user_connections_key(manager):
    """Test _user_connections_key returns correct key."""
    key = manager._user_connections_key("user-123")
    assert key == "websocket:user:user-123"


def test_heartbeat_key(manager):
    """Test _heartbeat_key returns correct key."""
    key = manager._heartbeat_key("conn-456")
    assert key == "websocket:connection:heartbeat:conn-456"


@pytest.mark.asyncio
async def test_get_hostname(manager):
    """Test _get_hostname returns hostname."""
    hostname = await manager._get_hostname()
    assert isinstance(hostname, str)


@pytest.mark.asyncio
async def test_get_hostname_handles_error(manager):
    """Test _get_hostname returns 'unknown' on error."""
    with patch("socket.gethostname", side_effect=Exception("Error")):
        hostname = await manager._get_hostname()
        assert hostname == "unknown"


# -----------------------------------------------------------------------------
# Test _register_connection result parsing
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_register_connection_handles_bytes_result(manager, mock_redis_client):
    """Test _register_connection handles bytes result from eval."""
    manager.redis_client = mock_redis_client
    mock_redis_client.eval.return_value = b"1"

    # Should not raise
    await manager._register_connection("conn-1", "user-1", None, None, 1000.0)


@pytest.mark.asyncio
async def test_register_connection_handles_string_result(manager, mock_redis_client):
    """Test _register_connection handles string result from eval."""
    manager.redis_client = mock_redis_client
    mock_redis_client.eval.return_value = "1"

    # Should not raise
    await manager._register_connection("conn-1", "user-1", None, None, 1000.0)


@pytest.mark.asyncio
async def test_register_connection_handles_bool_result(manager, mock_redis_client):
    """Test _register_connection handles bool result from eval."""
    manager.redis_client = mock_redis_client
    mock_redis_client.eval.return_value = True

    # Should not raise
    await manager._register_connection("conn-1", "user-1", None, None, 1000.0)


@pytest.mark.asyncio
async def test_register_connection_raises_on_limit_exceeded(manager, mock_redis_client):
    """Test _register_connection raises TooManyConnectionsError."""
    manager.redis_client = mock_redis_client
    mock_redis_client.eval.return_value = 0

    with pytest.raises(TooManyConnectionsError):
        await manager._register_connection("conn-1", "user-1", None, None, 1000.0)


@pytest.mark.asyncio
async def test_register_connection_no_redis(manager):
    """Test _register_connection does nothing without Redis."""
    manager.redis_client = None

    # Should not raise
    await manager._register_connection("conn-1", "user-1", None, None, 1000.0)


@pytest.mark.asyncio
async def test_unregister_connection_no_redis(manager):
    """Test _unregister_connection does nothing without Redis."""
    manager.redis_client = None

    # Should not raise
    await manager._unregister_connection("conn-1", "user-1")


@pytest.mark.asyncio
async def test_unregister_connection_with_unknown_user(manager, mock_redis_client):
    """Test _unregister_connection handles None user_id."""
    manager.redis_client = mock_redis_client

    await manager._unregister_connection("conn-1", None)

    mock_redis_client.eval.assert_awaited_once()


# -----------------------------------------------------------------------------
# Test _close_redis_connections
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_close_redis_connections(manager, mock_redis_client, mock_pubsub):
    """Test _close_redis_connections closes both clients."""
    manager.redis_client = mock_redis_client
    manager.pubsub = mock_pubsub

    await manager._close_redis_connections()

    mock_pubsub.aclose.assert_awaited_once()
    mock_redis_client.close.assert_awaited_once()
    assert manager.redis_client is None
    assert manager.pubsub is None


@pytest.mark.asyncio
async def test_close_redis_connections_handles_errors(manager, mock_redis_client, mock_pubsub):
    """Test _close_redis_connections handles errors gracefully."""
    manager.redis_client = mock_redis_client
    manager.pubsub = mock_pubsub
    mock_pubsub.aclose.side_effect = Exception("Close error")
    mock_redis_client.close.side_effect = Exception("Close error")

    # Should not raise
    await manager._close_redis_connections()

    assert manager.redis_client is None
    assert manager.pubsub is None
