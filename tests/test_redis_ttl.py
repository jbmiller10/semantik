"""Tests for Redis TTL and memory management functionality.

This module tests that Redis keys have proper TTL set and that
memory usage remains stable over time.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from webui.background_tasks import STREAM_MAX_LENGTH, TTL_CONFIG, RedisCleanupTask
from webui.websocket.legacy_stream_manager import RedisStreamWebSocketManager


@pytest.mark.asyncio()
async def test_operation_stream_ttl() -> None:
    """Test that operation streams have proper TTL set."""
    # Mock Redis client
    mock_redis = AsyncMock()
    mock_redis.xadd = AsyncMock()
    mock_redis.expire = AsyncMock()
    mock_redis.exists = AsyncMock(return_value=True)
    mock_redis.ttl = AsyncMock()
    mock_redis.close = AsyncMock()

    # Create WebSocket manager with mock Redis
    ws_manager = RedisStreamWebSocketManager()
    ws_manager.redis = mock_redis

    # Send an update for an active operation
    operation_id = "test-operation-123"
    await ws_manager.send_update(operation_id, "status_update", {"status": "active", "progress": 50})

    # Check that xadd was called with maxlen
    stream_key = f"operation-progress:{operation_id}"
    mock_redis.xadd.assert_called()
    call_args = mock_redis.xadd.call_args
    assert call_args[0][0] == stream_key
    assert call_args[1]["maxlen"] == 1000

    # Check that expire was called with 24 hours TTL for active
    mock_redis.expire.assert_called_with(stream_key, 86400)

    # Send a completion update
    await ws_manager.send_update(operation_id, "status_update", {"status": "completed", "progress": 100})

    # Check that expire was called with 5 minutes TTL for completed
    mock_redis.expire.assert_called_with(stream_key, 300)

    # Send a failure update
    await ws_manager.send_update(operation_id, "status_update", {"status": "failed", "error": "Test error"})

    # Check that expire was called with 1 minute TTL for failed
    mock_redis.expire.assert_called_with(stream_key, 60)


@pytest.mark.asyncio()
async def test_stream_max_length() -> None:
    """Test that streams are limited to maximum length."""
    mock_redis = AsyncMock()
    mock_redis.xadd = AsyncMock()
    mock_redis.expire = AsyncMock()

    ws_manager = RedisStreamWebSocketManager()
    ws_manager.redis = mock_redis

    operation_id = "test-operation-456"

    # Send a message
    await ws_manager.send_update(operation_id, "progress_update", {"progress": 50})

    # Verify xadd was called with maxlen parameter
    mock_redis.xadd.assert_called()
    call_args = mock_redis.xadd.call_args
    assert "maxlen" in call_args[1]
    assert call_args[1]["maxlen"] == STREAM_MAX_LENGTH


@pytest.mark.asyncio()
async def test_cleanup_task_sets_ttl() -> None:
    """Test that cleanup task sets TTL on keys without TTL."""
    # Mock Redis with specific behavior for our test
    mock_redis = AsyncMock()

    # Mock scan to return our test keys
    mock_redis.scan = AsyncMock()
    mock_redis.scan.side_effect = [
        (0, ["operation:test1"]),  # First pattern
        (0, ["preview:test2"]),  # Second pattern
        (0, ["chunking:preview:test3"]),  # Third pattern
        (0, []),  # WebSocket pattern
    ]

    # Mock TTL to return -1 (no TTL set)
    mock_redis.ttl = AsyncMock(return_value=-1)

    # Mock expire
    mock_redis.expire = AsyncMock()

    # Mock info for metrics
    mock_redis.info = AsyncMock(return_value={"used_memory_human": "100MB"})
    mock_redis.dbsize = AsyncMock(return_value=3)

    # Run cleanup task
    cleanup_task = RedisCleanupTask(mock_redis)
    await cleanup_task._perform_cleanup()

    # Check that expire was called for keys without TTL
    assert mock_redis.expire.call_count >= 3

    # Verify appropriate TTLs were set
    expire_calls = mock_redis.expire.call_args_list
    for call in expire_calls:
        key = call[0][0]
        ttl = call[0][1]

        if "operation:" in key:
            assert ttl == TTL_CONFIG["operation_active"]
        elif "preview:" in key or "chunking:preview:" in key:
            assert ttl == TTL_CONFIG["preview_cache"]


@pytest.mark.asyncio()
async def test_cleanup_task_trims_streams() -> None:
    """Test that cleanup task trims streams to max length."""
    # Mock Redis
    mock_redis = AsyncMock()

    # Mock scan to return stream keys
    mock_redis.scan = AsyncMock()
    mock_redis.scan.side_effect = [
        (0, ["operation-progress:test1", "operation-progress:test2"]),
        (0, ["stream:test3"]),
    ]

    # Mock xtrim to simulate trimming
    mock_redis.xtrim = AsyncMock(return_value=100)  # Simulate 100 entries trimmed

    # Run cleanup task
    cleanup_task = RedisCleanupTask(mock_redis)
    metrics = {"streams_trimmed": 0}
    await cleanup_task._trim_streams(metrics)

    # Check that xtrim was called
    assert mock_redis.xtrim.call_count >= 3

    # Check that metrics were updated
    assert metrics["streams_trimmed"] >= 3  # All streams should be counted as trimmed

    # Verify xtrim was called with correct parameters
    for call in mock_redis.xtrim.call_args_list:
        assert call[1]["maxlen"] == STREAM_MAX_LENGTH
        assert call[1]["approximate"] is True


@pytest.mark.asyncio()
async def test_cleanup_task_metrics_logging() -> None:
    """Test that cleanup task logs appropriate metrics."""
    # Mock Redis
    mock_redis = AsyncMock()

    # Mock Redis operations
    mock_redis.info = AsyncMock(return_value={"used_memory_human": "100MB"})
    mock_redis.dbsize = AsyncMock(return_value=5)
    mock_redis.scan = AsyncMock(return_value=(0, ["operation:test1", "preview:test2"]))
    mock_redis.ttl = AsyncMock(return_value=-1)
    mock_redis.expire = AsyncMock()
    mock_redis.xtrim = AsyncMock(return_value=0)

    # Mock logger to capture metrics
    with patch("webui.background_tasks.logger") as mock_logger:
        cleanup_task = RedisCleanupTask(mock_redis)
        await cleanup_task._perform_cleanup()

        # Check that metrics were logged
        assert mock_logger.info.called

        # Find the metrics log call
        for call in mock_logger.info.call_args_list:
            if "Redis cleanup metrics" in str(call):
                metrics_log = str(call)
                assert "keys_checked=" in metrics_log
                assert "ttl_set=" in metrics_log
                assert "total_keys=" in metrics_log
                break


@pytest.mark.asyncio()
async def test_websocket_channel_ttl() -> None:
    """Test that WebSocket channel messages have proper TTL."""
    # Mock Redis
    mock_redis = AsyncMock()
    mock_redis.xadd = AsyncMock()
    mock_redis.expire = AsyncMock()

    ws_manager = RedisStreamWebSocketManager()
    ws_manager.redis = mock_redis

    # Send a message to a channel
    channel = "chunking:collection-123:operation-456"
    await ws_manager.send_message(channel, {"type": "progress", "data": {"progress": 50}})

    # Check that xadd was called with correct maxlen
    stream_key = f"stream:{channel}"
    mock_redis.xadd.assert_called()
    call_args = mock_redis.xadd.call_args
    assert call_args[0][0] == stream_key
    assert call_args[1]["maxlen"] == 1000  # Updated to 1000 in our fix

    # Check that expire was called with 15 minutes TTL
    mock_redis.expire.assert_called_with(stream_key, 900)


@pytest.mark.asyncio()
async def test_cleanup_task_lifecycle() -> None:
    """Test starting and stopping the cleanup task."""
    # Mock Redis with proper async mock behavior
    mock_redis = AsyncMock()
    mock_redis.ping = AsyncMock()
    mock_redis.close = AsyncMock()
    mock_redis.info = AsyncMock(return_value={"used_memory_human": "100MB"})
    mock_redis.dbsize = AsyncMock(return_value=0)
    mock_redis.scan = AsyncMock(return_value=(0, []))

    cleanup_task = RedisCleanupTask(mock_redis)

    # Start the task
    await cleanup_task.start()
    assert cleanup_task.running
    assert cleanup_task._task is not None

    # Store the task reference
    task = cleanup_task._task

    # Let it run briefly
    await asyncio.sleep(0.1)

    # Stop the task
    await cleanup_task.stop()
    assert not cleanup_task.running

    # Check that task was cancelled or completed
    assert task.done()  # Task should be done (either cancelled or completed)


@pytest.mark.asyncio()
async def test_memory_stability() -> None:
    """Test that memory usage remains stable with TTL cleanup.

    This is a simulation test that verifies the concept of memory stability.
    In production, this would run for longer and measure actual Redis memory.
    """
    # Mock Redis with memory info
    mock_redis = AsyncMock()

    # Simulate memory info before and after
    mock_redis.info = AsyncMock()
    mock_redis.info.side_effect = [
        {"used_memory_human": "200MB"},  # Before cleanup
        {"used_memory_human": "150MB"},  # After cleanup
    ]

    mock_redis.dbsize = AsyncMock()
    mock_redis.dbsize.side_effect = [100, 80]  # Keys before and after

    # Mock scan to simulate finding keys
    mock_redis.scan = AsyncMock()
    mock_redis.scan.side_effect = [
        (0, [f"operation-progress:op-{i}" for i in range(10)]),
        (0, []),  # End of scan
    ] * 5  # Multiple patterns

    # Mock TTL and expire
    mock_redis.ttl = AsyncMock(return_value=-1)  # No TTL
    mock_redis.expire = AsyncMock()
    mock_redis.xtrim = AsyncMock(return_value=10)  # Some entries trimmed

    # Run cleanup
    cleanup_task = RedisCleanupTask(mock_redis)
    await cleanup_task._perform_cleanup()

    # Verify cleanup operations occurred
    assert mock_redis.expire.called
    assert mock_redis.info.call_count == 2  # Called for before and after metrics

    # Check that memory metrics would be logged
    # In production, we'd verify actual memory reduction


@pytest.mark.asyncio()
async def test_cleanup_handles_redis_errors() -> None:
    """Test that cleanup task handles Redis errors gracefully."""
    # Mock Redis with errors
    mock_redis = AsyncMock()
    mock_redis.ping = AsyncMock()
    mock_redis.info = AsyncMock(return_value={"used_memory_human": "100MB"})
    mock_redis.dbsize = AsyncMock(return_value=1000)
    mock_redis.scan = AsyncMock(side_effect=Exception("Redis connection error"))

    cleanup_task = RedisCleanupTask(mock_redis)

    # Should not raise exception despite Redis error
    await cleanup_task._perform_cleanup()

    # Verify error was handled (would be logged in real scenario)
    assert mock_redis.scan.called


@pytest.mark.asyncio()
async def test_websocket_cleanup_stale_connections() -> None:
    """Test that WebSocket manager can clean up stale connections."""
    ws_manager = RedisStreamWebSocketManager()

    # Create mock websockets
    mock_ws1 = AsyncMock()
    mock_ws1.ping = AsyncMock()  # Alive connection

    mock_ws2 = AsyncMock()
    mock_ws2.ping = AsyncMock(side_effect=Exception("Connection dead"))  # Dead connection

    # Add connections
    ws_manager.connections["user1:operation:op1"] = {mock_ws1, mock_ws2}

    # Run cleanup
    await ws_manager.cleanup_stale_connections()

    # Check that dead connection was removed
    assert mock_ws1 in ws_manager.connections.get("user1:operation:op1", set())
    assert mock_ws2 not in ws_manager.connections.get("user1:operation:op1", set())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
