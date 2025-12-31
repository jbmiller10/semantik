"""Tests for background tasks module.

This module tests the RedisCleanupTask and circuit breaker functionality.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from webui.background_tasks import (
    STREAM_MAX_LENGTH,
    TTL_CONFIG,
    RedisCleanupTask,
    start_background_tasks,
    stop_background_tasks,
)


class TestRedisCleanupTaskInit:
    """Tests for RedisCleanupTask initialization."""

    def test_init_with_default_values(self):
        """Test initialization with default values."""
        task = RedisCleanupTask()

        assert task.redis is None
        assert task.running is False
        assert task._task is None
        assert task._consecutive_failures == 0
        assert task._circuit_open is False

    def test_init_with_custom_redis_client(self):
        """Test initialization with custom Redis client."""
        mock_redis = AsyncMock()
        task = RedisCleanupTask(redis_client=mock_redis)

        assert task.redis is mock_redis


class TestRedisCleanupTaskStart:
    """Tests for RedisCleanupTask start method."""

    @pytest.mark.asyncio()
    async def test_start_connects_to_redis(self):
        """Test start() establishes Redis connection when not provided."""
        task = RedisCleanupTask()

        with patch("webui.background_tasks.redis.from_url", new_callable=AsyncMock) as mock_from_url:
            mock_redis = AsyncMock()
            mock_redis.ping = AsyncMock(return_value=True)
            mock_from_url.return_value = mock_redis

            await task.start()

            # Verify Redis connection
            mock_from_url.assert_called_once()
            mock_redis.ping.assert_called_once()
            assert task.running is True
            assert task._task is not None

            # Cleanup
            await task.stop()

    @pytest.mark.asyncio()
    async def test_start_when_already_running_logs_warning(self):
        """Test start() logs warning if already running."""
        task = RedisCleanupTask()
        task.running = True

        # Should return immediately without connecting
        await task.start()

        # No task should be created
        assert task._task is None

    @pytest.mark.asyncio()
    async def test_start_handles_connection_failure(self):
        """Test start() handles Redis connection failure."""
        task = RedisCleanupTask()

        with patch("webui.background_tasks.redis.from_url") as mock_from_url:
            mock_from_url.side_effect = Exception("Connection failed")

            await task.start()

            # Should not be running
            assert task.running is False
            assert task._task is None

    @pytest.mark.asyncio()
    async def test_start_with_existing_redis_client(self):
        """Test start() uses existing Redis client if provided."""
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)
        task = RedisCleanupTask(redis_client=mock_redis)

        with patch("webui.background_tasks.redis.from_url") as mock_from_url:
            await task.start()

            # Should not create new connection
            mock_from_url.assert_not_called()
            assert task.running is True

            await task.stop()


class TestRedisCleanupTaskStop:
    """Tests for RedisCleanupTask stop method."""

    @pytest.mark.asyncio()
    async def test_stop_cancels_task_and_closes_redis(self):
        """Test stop() properly cancels cleanup task and closes Redis."""
        mock_redis = AsyncMock()
        mock_redis.close = AsyncMock()
        task = RedisCleanupTask(redis_client=mock_redis)
        task.running = True
        task._task = asyncio.create_task(asyncio.sleep(100))

        await task.stop()

        assert task.running is False
        mock_redis.close.assert_called_once()

    @pytest.mark.asyncio()
    async def test_stop_when_not_running(self):
        """Test stop() when task is not running."""
        task = RedisCleanupTask()

        # Should not raise
        await task.stop()


class TestRedisCleanupLoop:
    """Tests for the cleanup loop."""

    @pytest.mark.asyncio()
    async def test_cleanup_loop_calls_perform_cleanup(self):
        """Test _cleanup_loop() calls _perform_cleanup() on each iteration."""
        mock_redis = AsyncMock()
        task = RedisCleanupTask(redis_client=mock_redis)
        task.running = True
        task._cleanup_interval = 0.05

        # Mock _perform_cleanup
        perform_cleanup_called = []

        async def mock_perform_cleanup():
            perform_cleanup_called.append(True)
            task.running = False  # Stop after first iteration

        task._perform_cleanup = mock_perform_cleanup

        await task._cleanup_loop()

        assert len(perform_cleanup_called) == 1

    @pytest.mark.asyncio()
    async def test_cleanup_loop_handles_circuit_breaker_open(self):
        """Test _cleanup_loop() pauses when circuit breaker is open."""
        mock_redis = AsyncMock()
        task = RedisCleanupTask(redis_client=mock_redis)
        task.running = True
        task._circuit_open = True
        task._current_backoff = 0.05

        iteration_count = [0]

        async def mock_perform_cleanup():
            iteration_count[0] += 1
            task.running = False

        task._perform_cleanup = mock_perform_cleanup

        await task._cleanup_loop()

        # Circuit breaker should have been reset to half-open
        assert task._circuit_open is False
        assert iteration_count[0] == 1


class TestCircuitBreaker:
    """Tests for circuit breaker logic."""

    @pytest.mark.asyncio()
    async def test_on_success_resets_failure_counter(self):
        """Test _on_success() resets consecutive_failures to 0."""
        task = RedisCleanupTask()
        task._consecutive_failures = 3
        task._current_backoff = 120
        task._cleanup_interval = 0.01

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await task._on_success()

        assert task._consecutive_failures == 0
        assert task._current_backoff == task._cleanup_interval

    @pytest.mark.asyncio()
    async def test_on_failure_increments_failure_counter(self):
        """Test _on_failure() increments consecutive_failures."""
        task = RedisCleanupTask()
        task._consecutive_failures = 0
        task._cleanup_interval = 0.01

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await task._on_failure(Exception("Test error"))

        assert task._consecutive_failures == 1

    @pytest.mark.asyncio()
    async def test_on_failure_opens_circuit_breaker_after_max_failures(self):
        """Test circuit breaker opens after max consecutive failures."""
        task = RedisCleanupTask()
        task._max_consecutive_failures = 3
        task._consecutive_failures = 2  # One more will trigger
        task._cleanup_interval = 0.01
        task._backoff_multiplier = 2.0

        await task._on_failure(Exception("Test error"))

        assert task._consecutive_failures == 3
        assert task._circuit_open is True
        assert task._current_backoff > task._cleanup_interval

    @pytest.mark.asyncio()
    async def test_on_failure_applies_exponential_backoff(self):
        """Test backoff multiplier is applied to current_backoff."""
        task = RedisCleanupTask()
        task._max_consecutive_failures = 3
        task._consecutive_failures = 2
        task._current_backoff = 10
        task._backoff_multiplier = 2.0
        task._max_backoff_seconds = 3600

        await task._on_failure(Exception("Test error"))

        assert task._current_backoff == 20  # 10 * 2.0

    @pytest.mark.asyncio()
    async def test_on_failure_respects_max_backoff(self):
        """Test backoff does not exceed max_backoff_seconds."""
        task = RedisCleanupTask()
        task._max_consecutive_failures = 3
        task._consecutive_failures = 2
        task._current_backoff = 1000
        task._backoff_multiplier = 10.0
        task._max_backoff_seconds = 300  # Would be 10000, capped at 300

        await task._on_failure(Exception("Test error"))

        assert task._current_backoff == 300


class TestPerformCleanup:
    """Tests for the actual cleanup operations."""

    @pytest.mark.asyncio()
    async def test_perform_cleanup_when_redis_not_available(self):
        """Test _perform_cleanup() handles missing Redis gracefully."""
        task = RedisCleanupTask()
        task.redis = None

        # Should not raise
        await task._perform_cleanup()

    @pytest.mark.asyncio()
    async def test_perform_cleanup_sets_ttl_on_keys_without_ttl(self):
        """Test _perform_cleanup() sets TTL on keys with TTL=-1."""
        mock_redis = AsyncMock()
        mock_redis.info = AsyncMock(return_value={"used_memory_human": "10M"})
        mock_redis.scan = AsyncMock(
            side_effect=[
                (0, ["operation:test-1"]),  # First call returns keys, cursor 0 means done
            ]
        )
        mock_redis.ttl = AsyncMock(return_value=-1)  # No TTL set
        mock_redis.expire = AsyncMock()
        mock_redis.dbsize = AsyncMock(return_value=10)

        task = RedisCleanupTask(redis_client=mock_redis)

        await task._perform_cleanup()

        # TTL should have been set
        mock_redis.expire.assert_called()

    @pytest.mark.asyncio()
    async def test_perform_cleanup_skips_keys_with_ttl(self):
        """Test _perform_cleanup() skips keys that already have TTL."""
        mock_redis = AsyncMock()
        mock_redis.info = AsyncMock(return_value={"used_memory_human": "10M"})
        mock_redis.scan = AsyncMock(
            side_effect=[
                (0, ["operation:test-1"]),
            ]
        )
        mock_redis.ttl = AsyncMock(return_value=300)  # Has TTL
        mock_redis.expire = AsyncMock()
        mock_redis.dbsize = AsyncMock(return_value=10)

        task = RedisCleanupTask(redis_client=mock_redis)

        await task._perform_cleanup()

        # TTL should NOT have been set (already has one)
        mock_redis.expire.assert_not_called()

    @pytest.mark.asyncio()
    async def test_trim_streams_trims_to_max_length(self):
        """Test _trim_streams() trims streams to STREAM_MAX_LENGTH."""
        mock_redis = AsyncMock()
        mock_redis.scan = AsyncMock(
            side_effect=[
                (0, ["operation-progress:test-1"]),
            ]
        )
        mock_redis.xtrim = AsyncMock(return_value=100)  # 100 entries deleted

        task = RedisCleanupTask(redis_client=mock_redis)

        metrics = {"streams_trimmed": 0}
        await task._trim_streams(metrics)

        # Verify xtrim was called with correct parameters
        mock_redis.xtrim.assert_called_with(
            "operation-progress:test-1",
            maxlen=STREAM_MAX_LENGTH,
            approximate=True,
        )
        assert metrics["streams_trimmed"] == 1

    @pytest.mark.asyncio()
    async def test_cleanup_pattern_handles_scan_iteration(self):
        """Test _cleanup_pattern() iterates through keys with SCAN."""
        mock_redis = AsyncMock()
        # Simulate multiple SCAN iterations
        mock_redis.scan = AsyncMock(
            side_effect=[
                (100, ["preview:1", "preview:2"]),  # First batch, more to come
                (0, ["preview:3"]),  # Last batch
            ]
        )
        mock_redis.ttl = AsyncMock(return_value=-1)
        mock_redis.expire = AsyncMock()

        task = RedisCleanupTask(redis_client=mock_redis)

        metrics = {"keys_checked": 0, "ttl_set": 0, "expired_removed": 0}
        await task._cleanup_pattern("preview:*", lambda _: 300, metrics)

        assert metrics["keys_checked"] == 3
        assert metrics["ttl_set"] == 3


class TestGetHealthStatus:
    """Tests for health status reporting."""

    def test_get_health_status_returns_circuit_breaker_state(self):
        """Test get_health_status() returns correct health info."""
        task = RedisCleanupTask()
        task._consecutive_failures = 2
        task._circuit_open = True
        task._current_backoff = 120
        task._max_consecutive_failures = 5

        status = task.get_health_status()

        assert status["consecutive_failures"] == 2
        assert status["circuit_open"] is True
        assert status["current_backoff_seconds"] == 120
        assert status["max_consecutive_failures"] == 5
        assert status["healthy"] is True  # 2 < 5

    def test_get_health_status_unhealthy_when_at_max_failures(self):
        """Test health status reports unhealthy at max failures."""
        task = RedisCleanupTask()
        task._consecutive_failures = 5
        task._max_consecutive_failures = 5

        status = task.get_health_status()

        assert status["healthy"] is False


class TestLogMetrics:
    """Tests for metrics logging."""

    def test_log_metrics_calculates_key_change(self):
        """Test _log_metrics() calculates key change from last run."""
        task = RedisCleanupTask()
        task._last_metrics = {"total_keys": 100}

        metrics = {
            "keys_checked": 50,
            "ttl_set": 10,
            "expired_removed": 5,
            "streams_trimmed": 2,
            "total_keys": 95,
        }

        task._log_metrics(metrics)

        assert metrics["keys_change"] == -5
        assert task._last_metrics == metrics


class TestBackgroundTaskHelpers:
    """Tests for start/stop background task helpers."""

    @pytest.mark.asyncio()
    async def test_start_background_tasks(self):
        """Test start_background_tasks() starts cleanup task."""
        with patch("webui.background_tasks.redis_cleanup_task") as mock_task:
            mock_task.start = AsyncMock()

            await start_background_tasks()

            mock_task.start.assert_called_once()

    @pytest.mark.asyncio()
    async def test_stop_background_tasks(self):
        """Test stop_background_tasks() stops cleanup task."""
        with patch("webui.background_tasks.redis_cleanup_task") as mock_task:
            mock_task.stop = AsyncMock()

            await stop_background_tasks()

            mock_task.stop.assert_called_once()


class TestTTLConfig:
    """Tests for TTL configuration."""

    def test_ttl_config_values(self):
        """Test TTL_CONFIG has expected values."""
        assert TTL_CONFIG["operation_active"] == 86400  # 24 hours
        assert TTL_CONFIG["operation_completed"] == 300  # 5 minutes
        assert TTL_CONFIG["operation_failed"] == 60  # 1 minute
        assert TTL_CONFIG["websocket_state"] == 900  # 15 minutes
        assert TTL_CONFIG["preview_cache"] == 300  # 5 minutes

    def test_stream_max_length(self):
        """Test STREAM_MAX_LENGTH constant."""
        assert STREAM_MAX_LENGTH == 1000
