"""Tests for ScalableWebSocketManager.

This module tests the horizontally scalable WebSocket manager that uses Redis
for cross-instance communication.
"""

import asyncio
import contextlib
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from webui.websocket.scalable_manager import ScalableWebSocketManager, TooManyConnectionsError


# Mock that properly simulates the awaitable Redis client
class AwaitableRedisMock(MagicMock):
    """Mock that can be awaited like the real Redis client."""

    def __await__(self):
        async def _coro():
            return self

        return _coro().__await__()


class TestScalableWebSocketManagerStartup:
    """Tests for startup and initialization."""

    @pytest.mark.asyncio()
    async def test_startup_connects_to_redis(self):
        """Test that startup() establishes Redis connection and starts background tasks."""
        manager = ScalableWebSocketManager()

        with patch("webui.websocket.scalable_manager.redis.from_url") as mock_from_url:
            mock_pubsub = AsyncMock()
            mock_pubsub.subscribe = AsyncMock()
            mock_pubsub.listen = AsyncMock(return_value=AsyncMock(__aiter__=lambda _: iter([])))
            mock_pubsub.aclose = AsyncMock()

            # Use AwaitableRedisMock because redis.from_url returns an awaitable client
            mock_redis = AwaitableRedisMock()
            mock_redis.ping = AsyncMock(return_value=True)
            mock_redis.pubsub = MagicMock(return_value=mock_pubsub)
            mock_redis.setex = AsyncMock()
            mock_redis.close = AsyncMock()
            mock_redis.expire = AsyncMock()
            mock_redis.hgetall = AsyncMock(return_value={})
            mock_redis.delete = AsyncMock()
            mock_from_url.return_value = mock_redis

            await manager.startup()

            # Verify Redis connection was established
            mock_from_url.assert_called_once()
            mock_redis.ping.assert_called_once()

            # Verify background tasks were started
            assert manager.listener_task is not None
            assert manager.heartbeat_task is not None
            assert manager.cleanup_task is not None
            assert manager._startup_complete is True

            # Cleanup
            await manager.shutdown()

    @pytest.mark.asyncio()
    async def test_startup_already_running_returns_early(self):
        """Test that startup() returns immediately when already started."""
        manager = ScalableWebSocketManager()
        manager._startup_complete = True

        # Should return immediately without connecting
        await manager.startup()

        # No Redis connection should be established
        assert manager.redis_client is None

    @pytest.mark.asyncio()
    async def test_startup_retry_on_connection_failure(self):
        """Test that startup() retries on Redis connection failure."""
        manager = ScalableWebSocketManager()

        with patch("webui.websocket.scalable_manager.redis.from_url") as mock_from_url:
            # First two calls fail, third succeeds
            mock_redis_fail = AwaitableRedisMock()
            mock_redis_fail.ping = AsyncMock(side_effect=Exception("Connection failed"))
            mock_redis_fail.close = AsyncMock()

            mock_pubsub_success = AsyncMock()
            mock_pubsub_success.subscribe = AsyncMock()
            mock_pubsub_success.listen = AsyncMock(return_value=AsyncMock(__aiter__=lambda _: iter([])))
            mock_pubsub_success.aclose = AsyncMock()

            mock_redis_success = AwaitableRedisMock()
            mock_redis_success.ping = AsyncMock(return_value=True)
            mock_redis_success.pubsub = MagicMock(return_value=mock_pubsub_success)
            mock_redis_success.setex = AsyncMock()
            mock_redis_success.close = AsyncMock()
            mock_redis_success.expire = AsyncMock()
            mock_redis_success.hgetall = AsyncMock(return_value={})
            mock_redis_success.delete = AsyncMock()

            mock_from_url.side_effect = [
                mock_redis_fail,
                mock_redis_fail,
                mock_redis_success,
            ]

            with patch("asyncio.sleep", new_callable=AsyncMock):
                await manager.startup()

            # Should have tried 3 times
            assert mock_from_url.call_count == 3
            assert manager._startup_complete is True

            await manager.shutdown()

    @pytest.mark.asyncio()
    async def test_startup_falls_back_to_local_mode_after_max_retries(self):
        """Test that startup() falls back to local-only mode after max retries."""
        manager = ScalableWebSocketManager()

        with patch("webui.websocket.scalable_manager.redis.from_url") as mock_from_url:
            mock_redis = AwaitableRedisMock()
            mock_redis.ping = AsyncMock(side_effect=Exception("Connection failed"))
            mock_redis.close = AsyncMock()
            mock_from_url.return_value = mock_redis

            with patch("asyncio.sleep", new_callable=AsyncMock):
                await manager.startup()

            # Should fall back to local-only mode
            assert manager.redis_client is None
            assert manager._startup_complete is True
            assert manager._local_only_mode is True


class TestScalableWebSocketManagerShutdown:
    """Tests for shutdown and cleanup."""

    @pytest.mark.asyncio()
    async def test_shutdown_cancels_tasks_and_closes_connections(self, mock_redis_client, mock_websocket):
        """Test shutdown() properly cancels tasks and closes WebSocket connections."""
        manager = ScalableWebSocketManager()
        manager.redis_client = mock_redis_client
        manager._startup_complete = True

        # Add a mock connection
        conn_id = "test-conn-1"
        manager.local_connections[conn_id] = mock_websocket
        manager.connection_metadata[conn_id] = {"user_id": "user-1"}

        # Create mock background tasks
        manager.listener_task = asyncio.create_task(asyncio.sleep(100))
        manager.heartbeat_task = asyncio.create_task(asyncio.sleep(100))
        manager.cleanup_task = asyncio.create_task(asyncio.sleep(100))

        await manager.shutdown()

        # Verify WebSocket was closed
        mock_websocket.close.assert_called_once()

        # Verify local tracking was cleared
        assert len(manager.local_connections) == 0
        assert len(manager.connection_metadata) == 0

    @pytest.mark.asyncio()
    async def test_shutdown_handles_websocket_close_errors(self, mock_redis_client):
        """Test shutdown() handles errors when closing WebSocket connections."""
        manager = ScalableWebSocketManager()
        manager.redis_client = mock_redis_client
        manager._startup_complete = True

        # Add a mock connection that raises on close
        mock_websocket = AsyncMock()
        mock_websocket.close = AsyncMock(side_effect=Exception("Close failed"))

        conn_id = "test-conn-1"
        manager.local_connections[conn_id] = mock_websocket
        manager.connection_metadata[conn_id] = {"user_id": "user-1"}

        # Should not raise, just log warning
        await manager.shutdown()

        # Connections should still be cleared
        assert len(manager.local_connections) == 0


class TestScalableWebSocketManagerConnect:
    """Tests for connection handling."""

    @pytest.mark.asyncio()
    async def test_connect_registers_websocket(self, mock_scalable_ws_manager, mock_websocket):
        """Test connect() registers WebSocket and returns connection ID."""
        manager = mock_scalable_ws_manager
        manager.redis_client.eval = AsyncMock(return_value=1)

        conn_id = await manager.connect(
            websocket=mock_websocket,
            user_id="user-1",
            operation_id="op-1",
            collection_id="col-1",
        )

        # Verify connection was accepted
        mock_websocket.accept.assert_called_once()

        # Verify connection was stored locally
        assert conn_id in manager.local_connections
        assert manager.local_connections[conn_id] == mock_websocket

        # Verify metadata was stored
        metadata = manager.connection_metadata[conn_id]
        assert metadata["user_id"] == "user-1"
        assert metadata["operation_id"] == "op-1"
        assert metadata["collection_id"] == "col-1"

    @pytest.mark.asyncio()
    async def test_connect_rejects_when_user_limit_exceeded(self, mock_scalable_ws_manager, mock_websocket):
        """Test connect() rejects when user exceeds connection limit."""
        manager = mock_scalable_ws_manager

        # Add max connections for user
        for i in range(manager.max_connections_per_user):
            manager.connection_metadata[f"conn-{i}"] = {"user_id": "user-1"}

        with pytest.raises(ConnectionError, match="User connection limit exceeded"):
            await manager.connect(
                websocket=mock_websocket,
                user_id="user-1",
            )

        # Verify WebSocket was closed with appropriate code
        mock_websocket.close.assert_called_once()
        call_args = mock_websocket.close.call_args
        assert call_args.kwargs.get("code") == 1008

    @pytest.mark.asyncio()
    async def test_connect_rejects_when_total_limit_exceeded(self, mock_scalable_ws_manager, mock_websocket):
        """Test connect() rejects when total connection limit exceeded."""
        manager = mock_scalable_ws_manager
        manager.max_total_connections = 5

        # Add max connections
        for i in range(5):
            manager.local_connections[f"conn-{i}"] = AsyncMock()

        with pytest.raises(ConnectionError, match="Server connection limit exceeded"):
            await manager.connect(
                websocket=mock_websocket,
                user_id="user-new",
            )

    @pytest.mark.asyncio()
    async def test_connect_subscribes_to_channels(self, mock_scalable_ws_manager, mock_websocket):
        """Test connect() subscribes to appropriate pub/sub channels."""
        manager = mock_scalable_ws_manager
        manager.redis_client.eval = AsyncMock(return_value=1)

        await manager.connect(
            websocket=mock_websocket,
            user_id="user-1",
            operation_id="op-1",
            collection_id="col-1",
        )

        # Verify subscriptions
        subscribe_calls = manager.pubsub.subscribe.call_args_list
        subscribed_channels = [call.args[0] for call in subscribe_calls]

        assert "user:user-1" in subscribed_channels
        assert "operation:op-1" in subscribed_channels
        assert "collection:col-1" in subscribed_channels

    @pytest.mark.asyncio()
    async def test_connect_with_subprotocol(self, mock_scalable_ws_manager, mock_websocket):
        """Test connect() accepts with subprotocol when provided."""
        manager = mock_scalable_ws_manager
        manager.redis_client.eval = AsyncMock(return_value=1)

        await manager.connect(
            websocket=mock_websocket,
            user_id="user-1",
            subprotocol="graphql-ws",
        )

        mock_websocket.accept.assert_called_once_with(subprotocol="graphql-ws")

    @pytest.mark.asyncio()
    async def test_connect_handles_redis_registration_failure(self, mock_scalable_ws_manager, mock_websocket):
        """Test connect() handles Redis registration failure (TooManyConnectionsError)."""
        manager = mock_scalable_ws_manager
        manager.redis_client.eval = AsyncMock(return_value=0)  # Registration rejected

        with pytest.raises(ConnectionError, match="User connection limit exceeded"):
            await manager.connect(
                websocket=mock_websocket,
                user_id="user-1",
            )


class TestScalableWebSocketManagerDisconnect:
    """Tests for disconnection handling."""

    @pytest.mark.asyncio()
    async def test_disconnect_removes_connection(self, mock_scalable_ws_manager, mock_websocket):
        """Test disconnect() removes connection from local storage and Redis."""
        manager = mock_scalable_ws_manager

        # Setup connection
        conn_id = "test-conn-1"
        manager.local_connections[conn_id] = mock_websocket
        manager.connection_metadata[conn_id] = {
            "user_id": "user-1",
            "operation_id": "op-1",
            "collection_id": "col-1",
        }

        await manager.disconnect(conn_id)

        # Verify connection was removed
        assert conn_id not in manager.local_connections
        assert conn_id not in manager.connection_metadata

    @pytest.mark.asyncio()
    async def test_disconnect_nonexistent_connection(self, mock_scalable_ws_manager):
        """Test disconnect() handles non-existent connection gracefully."""
        manager = mock_scalable_ws_manager

        # Should not raise
        await manager.disconnect("nonexistent-conn")

    @pytest.mark.asyncio()
    async def test_disconnect_unsubscribes_from_channels(self, mock_scalable_ws_manager, mock_websocket):
        """Test disconnect() unsubscribes from channels when last user connection."""
        manager = mock_scalable_ws_manager

        # Setup single connection for user
        conn_id = "test-conn-1"
        manager.local_connections[conn_id] = mock_websocket
        manager.connection_metadata[conn_id] = {
            "user_id": "user-1",
            "operation_id": "op-1",
            "collection_id": None,
        }

        await manager.disconnect(conn_id)

        # Verify unsubscriptions
        unsubscribe_calls = manager.pubsub.unsubscribe.call_args_list
        unsubscribed_channels = [call.args[0] for call in unsubscribe_calls]

        assert "user:user-1" in unsubscribed_channels
        assert "operation:op-1" in unsubscribed_channels


class TestScalableWebSocketManagerMessaging:
    """Tests for message sending."""

    @pytest.mark.asyncio()
    async def test_send_to_user_sends_to_local_connections(self, mock_scalable_ws_manager, mock_websocket):
        """Test send_to_user() sends message to all local user connections."""
        manager = mock_scalable_ws_manager
        manager.redis_client.smembers = AsyncMock(return_value=set())

        # Setup connections for user
        manager.local_connections["conn-1"] = mock_websocket
        manager.local_connections["conn-2"] = mock_websocket
        manager.connection_metadata["conn-1"] = {"user_id": "user-1"}
        manager.connection_metadata["conn-2"] = {"user_id": "user-1"}

        message = {"type": "test", "data": "hello"}
        await manager.send_to_user("user-1", message)

        # Should send to both connections
        assert mock_websocket.send_json.call_count == 2

    @pytest.mark.asyncio()
    async def test_send_to_user_publishes_for_remote_connections(self, mock_scalable_ws_manager):
        """Test send_to_user() publishes to Redis for remote connections."""
        manager = mock_scalable_ws_manager

        # Mock remote connection
        manager.redis_client.smembers = AsyncMock(return_value={"remote-conn-1"})
        manager.redis_client.exists = AsyncMock(return_value=True)
        manager.redis_client.hget = AsyncMock(
            return_value=json.dumps({"instance_id": "other-instance", "user_id": "user-1"})
        )
        manager.redis_client.publish = AsyncMock()

        message = {"type": "test", "data": "hello"}
        await manager.send_to_user("user-1", message)

        # Should publish to Redis
        manager.redis_client.publish.assert_called_once()
        call_args = manager.redis_client.publish.call_args
        assert call_args.args[0] == "user:user-1"

    @pytest.mark.asyncio()
    async def test_send_to_operation_sends_to_subscribers(self, mock_scalable_ws_manager, mock_websocket):
        """Test send_to_operation() sends to connections watching the operation."""
        manager = mock_scalable_ws_manager
        manager.redis_client.publish = AsyncMock()

        # Setup connection watching operation
        manager.local_connections["conn-1"] = mock_websocket
        manager.connection_metadata["conn-1"] = {
            "user_id": "user-1",
            "operation_id": "op-1",
        }

        message = {"type": "progress", "percent": 50}
        await manager.send_to_operation("op-1", message)

        mock_websocket.send_json.assert_called_once_with(message)

    @pytest.mark.asyncio()
    async def test_send_to_operation_handles_send_failure(self, mock_scalable_ws_manager):
        """Test send_to_operation() handles send failure by disconnecting."""
        manager = mock_scalable_ws_manager
        manager.redis_client.publish = AsyncMock()

        # Setup connection that fails to send
        mock_ws = AsyncMock()
        mock_ws.send_json = AsyncMock(side_effect=Exception("Send failed"))

        manager.local_connections["conn-1"] = mock_ws
        manager.connection_metadata["conn-1"] = {
            "user_id": "user-1",
            "operation_id": "op-1",
        }

        message = {"type": "progress", "percent": 50}
        await manager.send_to_operation("op-1", message)

        # Connection should be disconnected
        assert "conn-1" not in manager.local_connections

    @pytest.mark.asyncio()
    async def test_broadcast_to_collection_sends_to_subscribers(self, mock_scalable_ws_manager, mock_websocket):
        """Test broadcast_to_collection() sends to all collection subscribers."""
        manager = mock_scalable_ws_manager
        manager.redis_client.publish = AsyncMock()

        # Setup connection watching collection
        manager.local_connections["conn-1"] = mock_websocket
        manager.connection_metadata["conn-1"] = {
            "user_id": "user-1",
            "collection_id": "col-1",
        }

        message = {"type": "update", "status": "ready"}
        await manager.broadcast_to_collection("col-1", message)

        mock_websocket.send_json.assert_called_once_with(message)
        manager.redis_client.publish.assert_called_once()


class TestScalableWebSocketManagerHeartbeat:
    """Tests for heartbeat functionality."""

    @pytest.mark.asyncio()
    async def test_heartbeat_pings_connections(self, mock_scalable_ws_manager, mock_websocket):
        """Test _heartbeat() pings all local connections."""
        manager = mock_scalable_ws_manager
        manager.heartbeat_interval_seconds = 0.05

        # Setup connection
        manager.local_connections["conn-1"] = mock_websocket
        manager.connection_metadata["conn-1"] = {"user_id": "user-1"}

        # Mock pipeline
        mock_pipeline = AsyncMock()
        mock_pipeline.__aenter__ = AsyncMock(return_value=mock_pipeline)
        mock_pipeline.__aexit__ = AsyncMock()
        mock_pipeline.setex = MagicMock()
        mock_pipeline.expire = MagicMock()
        mock_pipeline.execute = AsyncMock(return_value=[])
        manager.redis_client.pipeline = MagicMock(return_value=mock_pipeline)

        # Run one iteration
        heartbeat_task = asyncio.create_task(manager._heartbeat())
        await asyncio.sleep(0.15)
        heartbeat_task.cancel()

        # Wait for task to finish, ignoring CancelledError
        with contextlib.suppress(asyncio.CancelledError):
            await heartbeat_task

        # Verify ping was sent
        mock_websocket.send_json.assert_called()
        call_args = mock_websocket.send_json.call_args
        assert call_args.args[0]["type"] == "ping"

    @pytest.mark.asyncio()
    async def test_heartbeat_removes_dead_connections(self, mock_scalable_ws_manager):
        """Test _heartbeat() removes connections that fail to respond."""
        manager = mock_scalable_ws_manager
        manager.heartbeat_interval_seconds = 0.05

        # Setup connection that times out
        mock_ws = AsyncMock()
        mock_ws.send_json = AsyncMock(side_effect=TimeoutError())

        manager.local_connections["conn-1"] = mock_ws
        manager.connection_metadata["conn-1"] = {"user_id": "user-1"}

        # Mock pipeline
        mock_pipeline = AsyncMock()
        mock_pipeline.__aenter__ = AsyncMock(return_value=mock_pipeline)
        mock_pipeline.__aexit__ = AsyncMock()
        mock_pipeline.setex = MagicMock()
        mock_pipeline.expire = MagicMock()
        mock_pipeline.execute = AsyncMock(return_value=[])
        manager.redis_client.pipeline = MagicMock(return_value=mock_pipeline)

        # Run one iteration
        heartbeat_task = asyncio.create_task(manager._heartbeat())
        await asyncio.sleep(0.15)
        heartbeat_task.cancel()

        # Wait for task to finish, ignoring CancelledError
        with contextlib.suppress(asyncio.CancelledError):
            await heartbeat_task

        # Connection should be removed
        assert "conn-1" not in manager.local_connections


class TestScalableWebSocketManagerCleanup:
    """Tests for dead connection cleanup."""

    @pytest.mark.asyncio()
    async def test_cleanup_dead_connections_removes_stale(self, mock_scalable_ws_manager):
        """Test _cleanup_dead_connections() removes connections with expired heartbeats."""
        manager = mock_scalable_ws_manager
        manager.cleanup_interval_seconds = 0.05

        # Mock stale connection in Redis
        stale_conn_data = json.dumps(
            {
                "connection_id": "stale-conn",
                "user_id": "user-1",
                "instance_id": "dead-instance",
            }
        )
        manager.redis_client.hgetall = AsyncMock(return_value={"stale-conn": stale_conn_data})
        manager.redis_client.exists = AsyncMock(return_value=False)  # Heartbeat expired

        # Mock scan_iter to return empty async generator
        async def mock_scan_iter(**_kwargs):
            if False:  # noqa: SIM223
                yield  # Make it an async generator that yields nothing

        manager.redis_client.scan_iter = mock_scan_iter
        manager.redis_client.eval = AsyncMock()

        # Run one iteration
        cleanup_task = asyncio.create_task(manager._cleanup_dead_connections())
        await asyncio.sleep(0.15)
        cleanup_task.cancel()

        # Wait for task to finish, ignoring CancelledError
        with contextlib.suppress(asyncio.CancelledError):
            await cleanup_task

        # Verify unregister was called
        manager.redis_client.eval.assert_called()

    @pytest.mark.asyncio()
    async def test_cleanup_handles_json_decode_error(self, mock_scalable_ws_manager):
        """Test _cleanup_dead_connections() handles malformed JSON gracefully."""
        manager = mock_scalable_ws_manager
        manager.cleanup_interval_seconds = 0.05

        # Mock malformed connection data
        manager.redis_client.hgetall = AsyncMock(return_value={"bad-conn": "not-json"})

        # Mock scan_iter to return empty async generator
        async def mock_scan_iter(**_kwargs):
            if False:  # noqa: SIM223
                yield  # Make it an async generator that yields nothing

        manager.redis_client.scan_iter = mock_scan_iter

        # Should not raise
        cleanup_task = asyncio.create_task(manager._cleanup_dead_connections())
        await asyncio.sleep(0.15)
        cleanup_task.cancel()

        # Wait for task to finish, ignoring CancelledError
        with contextlib.suppress(asyncio.CancelledError):
            await cleanup_task


class TestScalableWebSocketManagerStats:
    """Tests for statistics."""

    @pytest.mark.asyncio()
    async def test_get_stats_returns_local_stats(self, mock_scalable_ws_manager, mock_websocket):
        """Test get_stats() returns correct local connection stats."""
        manager = mock_scalable_ws_manager

        # Setup connections
        manager.local_connections["conn-1"] = mock_websocket
        manager.local_connections["conn-2"] = mock_websocket
        manager.connection_metadata["conn-1"] = {
            "user_id": "user-1",
            "operation_id": "op-1",
            "collection_id": "col-1",
        }
        manager.connection_metadata["conn-2"] = {
            "user_id": "user-2",
            "operation_id": "op-1",
            "collection_id": None,
        }

        # Mock Redis stats
        manager.redis_client.hlen = AsyncMock(return_value=5)
        manager.redis_client.keys = AsyncMock(return_value=["instance:1", "instance:2"])

        stats = await manager.get_stats()

        assert stats["local_connections"] == 2
        assert stats["unique_users"] == 2
        assert stats["operations"] == 1
        assert stats["collections"] == 1
        assert stats["total_connections"] == 5
        assert stats["active_instances"] == 2

    @pytest.mark.asyncio()
    async def test_get_stats_handles_redis_failure(self, mock_scalable_ws_manager):
        """Test get_stats() handles Redis failure gracefully."""
        manager = mock_scalable_ws_manager
        manager.redis_client.hlen = AsyncMock(side_effect=Exception("Redis error"))

        stats = await manager.get_stats()

        # Should still return local stats
        assert "local_connections" in stats
        assert "total_connections" not in stats  # Redis stats not available


class TestTooManyConnectionsError:
    """Tests for TooManyConnectionsError."""

    def test_exception_message(self):
        """Test TooManyConnectionsError has correct message."""
        error = TooManyConnectionsError("User exceeded limit")
        assert str(error) == "User exceeded limit"
