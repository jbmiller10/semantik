#!/usr/bin/env python3
"""
Redis Stream Manager for job progress updates.

This module provides a RedisStreamManager class that handles publishing
job progress updates to Redis streams. It's used by Celery workers to
send real-time updates that are consumed by WebSocket handlers.
"""

import json
import logging
from datetime import UTC, datetime
from typing import Any

import redis
import redis.asyncio as aioredis

logger = logging.getLogger(__name__)


class RedisStreamManager:
    """Manager for publishing job updates to Redis streams."""

    def __init__(self, redis_url: str):
        """Initialize the Redis Stream Manager.

        Args:
            redis_url: Redis connection URL
        """
        self.redis_url = redis_url
        self._sync_client: redis.Redis | None = None
        self._async_client: aioredis.Redis | None = None

    @property
    def sync_client(self) -> redis.Redis:
        """Get or create synchronous Redis client."""
        if self._sync_client is None:
            self._sync_client = redis.from_url(self.redis_url, decode_responses=True)
        return self._sync_client

    @property
    async def async_client(self) -> aioredis.Redis:
        """Get or create asynchronous Redis client."""
        if self._async_client is None:
            self._async_client = await aioredis.from_url(self.redis_url, decode_responses=True)
        return self._async_client

    def get_stream_key(self, job_id: str) -> str:
        """Get the Redis stream key for a job.

        Args:
            job_id: The job ID

        Returns:
            The Redis stream key
        """
        return f"job:stream:{job_id}"

    def publish_update(self, job_id: str, update_type: str, data: dict[str, Any]) -> None:
        """Publish an update to a job's Redis stream (synchronous).

        Args:
            job_id: The job ID
            update_type: Type of update (e.g., 'progress', 'status', 'error')
            data: Update data
        """
        try:
            stream_key = self.get_stream_key(job_id)
            message = {"type": update_type, "timestamp": datetime.now(UTC).isoformat(), "data": json.dumps(data)}

            # Add to stream with automatic ID
            self.sync_client.xadd(stream_key, message, maxlen=1000)

            # Set expiration on the stream (24 hours)
            self.sync_client.expire(stream_key, 86400)

            logger.debug(f"Published {update_type} update for job {job_id}")
        except Exception as e:
            logger.error(f"Failed to publish update for job {job_id}: {e}")

    async def publish_update_async(self, job_id: str, update_type: str, data: dict[str, Any]) -> None:
        """Publish an update to a job's Redis stream (asynchronous).

        Args:
            job_id: The job ID
            update_type: Type of update (e.g., 'progress', 'status', 'error')
            data: Update data
        """
        try:
            client = await self.async_client
            stream_key = self.get_stream_key(job_id)
            message = {"type": update_type, "timestamp": datetime.now(UTC).isoformat(), "data": json.dumps(data)}

            # Add to stream with automatic ID
            await client.xadd(stream_key, message, maxlen=1000)

            # Set expiration on the stream (24 hours)
            await client.expire(stream_key, 86400)

            logger.debug(f"Published {update_type} update for job {job_id}")
        except Exception as e:
            logger.error(f"Failed to publish update for job {job_id}: {e}")

    def cleanup_stream(self, job_id: str) -> None:
        """Clean up a job's Redis stream.

        Args:
            job_id: The job ID
        """
        try:
            stream_key = self.get_stream_key(job_id)
            self.sync_client.delete(stream_key)
            logger.debug(f"Cleaned up stream for job {job_id}")
        except Exception as e:
            logger.error(f"Failed to cleanup stream for job {job_id}: {e}")

    async def cleanup_stream_async(self, job_id: str) -> None:
        """Clean up a job's Redis stream (asynchronous).

        Args:
            job_id: The job ID
        """
        try:
            client = await self.async_client
            stream_key = self.get_stream_key(job_id)
            await client.delete(stream_key)
            logger.debug(f"Cleaned up stream for job {job_id}")
        except Exception as e:
            logger.error(f"Failed to cleanup stream for job {job_id}: {e}")

    def close(self) -> None:
        """Close Redis connections."""
        if self._sync_client:
            self._sync_client.close()
            self._sync_client = None

    async def close_async(self) -> None:
        """Close Redis connections (asynchronous)."""
        if self._async_client:
            await self._async_client.close()
            self._async_client = None


# Convenience functions for Celery tasks
_manager: RedisStreamManager | None = None


def get_redis_stream_manager(redis_url: str) -> RedisStreamManager:
    """Get or create the global Redis stream manager.

    Args:
        redis_url: Redis connection URL

    Returns:
        The Redis stream manager instance
    """
    global _manager
    if _manager is None:
        _manager = RedisStreamManager(redis_url)
    return _manager


def publish_job_update(job_id: str, update_type: str, data: dict[str, Any], redis_url: str) -> None:
    """Convenience function to publish job updates from Celery tasks.

    Args:
        job_id: The job ID
        update_type: Type of update
        data: Update data
        redis_url: Redis connection URL
    """
    manager = get_redis_stream_manager(redis_url)
    manager.publish_update(job_id, update_type, data)
