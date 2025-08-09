"""Background tasks for Redis memory management and cleanup.

This module contains background tasks that run periodically to clean up
expired data from Redis and prevent memory leaks in pre-release environment.
"""

import asyncio
import contextlib
import logging
from datetime import UTC, datetime
from typing import Any

import redis.asyncio as redis

from packages.shared.config import settings

logger = logging.getLogger(__name__)

# TTL Configuration for different data types (in seconds)
TTL_CONFIG = {
    # Operation-related TTLs
    "operation_active": 3600,  # Active operations: 1 hour
    "operation_completed": 300,  # Completed operations: 5 minutes
    "operation_failed": 60,  # Failed operations: 1 minute
    # WebSocket state TTLs
    "websocket_state": 900,  # WebSocket state: 15 minutes
    "websocket_stream": 900,  # WebSocket streams: 15 minutes
    # Cache TTLs
    "cache_default": 300,  # Default cache: 5 minutes
    "preview_cache": 300,  # Preview cache: 5 minutes
    "progress_cache": 300,  # Progress cache: 5 minutes
}

# Stream length limits
STREAM_MAX_LENGTH = 1000  # Maximum events per stream


class RedisCleanupTask:
    """Background task for cleaning up expired Redis data."""

    def __init__(self, redis_client: redis.Redis | None = None):
        """Initialize the cleanup task.

        Args:
            redis_client: Optional Redis client to use
        """
        self.redis = redis_client
        self.running = False
        self._task: asyncio.Task | None = None
        self._cleanup_interval = 60  # Run every minute
        self._last_metrics: dict[str, Any] = {}

    async def start(self) -> None:
        """Start the background cleanup task."""
        if self.running:
            logger.warning("Redis cleanup task already running")
            return

        # Initialize Redis if not provided
        if self.redis is None:
            try:
                self.redis = await redis.from_url(
                    settings.REDIS_URL,
                    decode_responses=True,
                    health_check_interval=30,
                    socket_keepalive=True,
                )
                await self.redis.ping()
                logger.info(f"Redis cleanup task connected to {settings.REDIS_URL}")
            except Exception as e:
                logger.error(f"Failed to connect to Redis for cleanup task: {e}")
                return

        self.running = True
        self._task = asyncio.create_task(self._cleanup_loop())
        logger.info("Redis cleanup task started")

    async def stop(self) -> None:
        """Stop the background cleanup task."""
        if not self.running:
            return

        self.running = False

        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

        if self.redis:
            await self.redis.close()

        logger.info("Redis cleanup task stopped")

    async def _cleanup_loop(self) -> None:
        """Main cleanup loop that runs periodically."""
        while self.running:
            try:
                await self._perform_cleanup()
                await asyncio.sleep(self._cleanup_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(self._cleanup_interval)

    async def _perform_cleanup(self) -> None:
        """Perform the actual cleanup operations."""
        if not self.redis:
            logger.warning("Redis not available for cleanup")
            return

        start_time = datetime.now(UTC)
        metrics = {
            "timestamp": start_time.isoformat(),
            "keys_checked": 0,
            "ttl_set": 0,
            "expired_removed": 0,
            "streams_trimmed": 0,
        }

        try:
            # Get initial memory stats
            info = await self.redis.info("memory")
            metrics["memory_used_before"] = info.get("used_memory_human", "unknown")

            # Pattern-based cleanup for different key types
            patterns = [
                # Operation patterns
                ("operation:*", self._get_operation_ttl),
                ("operation-progress:*", self._get_operation_stream_ttl),
                # Cache patterns
                ("preview:*", lambda _: TTL_CONFIG["preview_cache"]),
                ("chunking:preview:*", lambda _: TTL_CONFIG["preview_cache"]),
                ("operation:progress:*", lambda _: TTL_CONFIG["progress_cache"]),
                # WebSocket patterns
                ("stream:*", lambda _: TTL_CONFIG["websocket_stream"]),
            ]

            for pattern, ttl_func in patterns:
                await self._cleanup_pattern(pattern, ttl_func, metrics)

            # Trim streams to prevent unbounded growth
            await self._trim_streams(metrics)

            # Get final memory stats
            info = await self.redis.info("memory")
            metrics["memory_used_after"] = info.get("used_memory_human", "unknown")

            # Calculate total keys
            metrics["total_keys"] = await self.redis.dbsize()

            # Log metrics
            self._log_metrics(metrics)

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            metrics["error"] = str(e)
            self._log_metrics(metrics)

    async def _cleanup_pattern(self, pattern: str, ttl_func: Any, metrics: dict[str, Any]) -> None:
        """Clean up keys matching a pattern.

        Args:
            pattern: Redis key pattern to match
            ttl_func: Function to determine TTL for a key
            metrics: Metrics dictionary to update
        """
        if not self.redis:
            return

        try:
            cursor = 0
            while True:
                # Use SCAN to iterate through keys without blocking
                cursor, keys = await self.redis.scan(cursor, match=pattern, count=100)

                for key in keys:
                    metrics["keys_checked"] += 1

                    # Check if key has TTL
                    ttl = await self.redis.ttl(key)

                    if ttl == -1:  # No TTL set
                        # Determine appropriate TTL
                        new_ttl = ttl_func(key)

                        if new_ttl > 0:
                            await self.redis.expire(key, new_ttl)
                            metrics["ttl_set"] += 1
                            logger.debug(f"Set TTL={new_ttl}s on key: {key}")
                    elif ttl == -2:  # Key doesn't exist (already expired)
                        metrics["expired_removed"] += 1

                if cursor == 0:
                    break

        except Exception as e:
            logger.error(f"Error cleaning pattern {pattern}: {e}")

    async def _trim_streams(self, metrics: dict[str, Any]) -> None:
        """Trim Redis streams to prevent unbounded growth.

        Args:
            metrics: Metrics dictionary to update
        """
        if not self.redis:
            return

        try:
            # Find all stream keys
            stream_patterns = [
                "operation-progress:*",
                "stream:*",
            ]

            for pattern in stream_patterns:
                cursor = 0
                while True:
                    cursor, keys = await self.redis.scan(cursor, match=pattern, count=100)

                    for key in keys:
                        try:
                            # Trim stream to maximum length
                            # XTRIM returns the number of entries deleted
                            deleted = await self.redis.xtrim(
                                key,
                                maxlen=STREAM_MAX_LENGTH,
                                approximate=True,  # More efficient
                            )
                            if deleted > 0:
                                metrics["streams_trimmed"] += 1
                                logger.debug(f"Trimmed {deleted} entries from stream: {key}")
                        except Exception as e:
                            logger.warning(f"Failed to trim stream {key}: {e}")

                    if cursor == 0:
                        break

        except Exception as e:
            logger.error(f"Error trimming streams: {e}")

    def _get_operation_ttl(self, key: str) -> int:  # noqa: ARG002
        """Determine TTL for an operation key based on its status.

        Args:
            key: The Redis key

        Returns:
            TTL in seconds
        """
        # For now, use a default TTL
        # In production, we would check the operation status
        return TTL_CONFIG["operation_active"]

    def _get_operation_stream_ttl(self, key: str) -> int:  # noqa: ARG002
        """Determine TTL for an operation stream.

        Args:
            key: The Redis key

        Returns:
            TTL in seconds
        """
        # Operation streams should expire after completion
        return TTL_CONFIG["operation_completed"]

    def _log_metrics(self, metrics: dict[str, Any]) -> None:
        """Log cleanup metrics.

        Args:
            metrics: Metrics dictionary to log
        """
        # Calculate changes from last run
        if self._last_metrics:
            metrics["keys_change"] = metrics.get("total_keys", 0) - self._last_metrics.get("total_keys", 0)

        # Log metrics
        logger.info(
            f"Redis cleanup metrics: "
            f"keys_checked={metrics['keys_checked']}, "
            f"ttl_set={metrics['ttl_set']}, "
            f"expired={metrics['expired_removed']}, "
            f"streams_trimmed={metrics['streams_trimmed']}, "
            f"total_keys={metrics.get('total_keys', 'unknown')}, "
            f"memory_before={metrics.get('memory_used_before', 'unknown')}, "
            f"memory_after={metrics.get('memory_used_after', 'unknown')}"
        )

        # Store for next comparison
        self._last_metrics = metrics


# Global cleanup task instance
redis_cleanup_task = RedisCleanupTask()


async def start_background_tasks() -> None:
    """Start all background tasks."""
    logger.info("Starting background tasks...")

    # Start Redis cleanup task
    await redis_cleanup_task.start()

    logger.info("Background tasks started")


async def stop_background_tasks() -> None:
    """Stop all background tasks."""
    logger.info("Stopping background tasks...")

    # Stop Redis cleanup task
    await redis_cleanup_task.stop()

    logger.info("Background tasks stopped")
