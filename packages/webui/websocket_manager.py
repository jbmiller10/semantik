#!/usr/bin/env python3
"""
WebSocket Redis Stream Consumer for real-time job updates.

This module provides a RedisStreamWebSocketManager that replaces the old
ConnectionManager. It handles WebSocket connections and consumes job updates
from Redis streams, ensuring clients receive real-time updates even when
connecting mid-job.
"""

import asyncio
import json
import logging
from datetime import UTC, datetime
from typing import Any

import redis.asyncio as aioredis
from fastapi import WebSocket

from packages.webui.redis_streams import RedisStreamManager

logger = logging.getLogger(__name__)


class RedisStreamWebSocketManager:
    """Manages WebSocket connections and Redis stream consumption for job updates."""

    def __init__(self, redis_url: str):
        """Initialize the WebSocket manager.

        Args:
            redis_url: Redis connection URL
        """
        self.redis_url = redis_url
        self.redis_manager = RedisStreamManager(redis_url)
        self.active_connections: dict[str, set[WebSocket]] = {}
        self.consumer_tasks: dict[str, asyncio.Task] = {}
        self._redis_client: aioredis.Redis | None = None

    async def get_redis_client(self) -> aioredis.Redis:
        """Get or create Redis client."""
        if self._redis_client is None:
            self._redis_client = await aioredis.from_url(self.redis_url, decode_responses=True)
        return self._redis_client

    async def connect(self, websocket: WebSocket, job_id: str) -> None:
        """Connect a WebSocket client for a specific job.

        Args:
            websocket: The WebSocket connection
            job_id: The job ID to subscribe to
        """
        await websocket.accept()

        # Add to active connections
        if job_id not in self.active_connections:
            self.active_connections[job_id] = set()
        self.active_connections[job_id].add(websocket)

        # Start consumer task if not already running
        if job_id not in self.consumer_tasks or self.consumer_tasks[job_id].done():
            self.consumer_tasks[job_id] = asyncio.create_task(self._consume_stream(job_id))

        logger.info(f"WebSocket connected for job {job_id}")

    async def disconnect(self, websocket: WebSocket, job_id: str) -> None:
        """Disconnect a WebSocket client.

        Args:
            websocket: The WebSocket connection
            job_id: The job ID
        """
        if job_id in self.active_connections:
            self.active_connections[job_id].discard(websocket)

            # If no more connections for this job, cancel the consumer task
            if not self.active_connections[job_id]:
                del self.active_connections[job_id]
                if job_id in self.consumer_tasks:
                    self.consumer_tasks[job_id].cancel()
                    del self.consumer_tasks[job_id]

        logger.info(f"WebSocket disconnected for job {job_id}")

    async def send_to_job(self, job_id: str, message: dict[str, Any]) -> None:
        """Send a message to all WebSocket clients for a job.

        Args:
            job_id: The job ID
            message: The message to send
        """
        if job_id not in self.active_connections:
            return

        # Send to all connected clients for this job
        disconnected = set()
        for websocket in self.active_connections[job_id]:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error sending to WebSocket: {e}")
                disconnected.add(websocket)

        # Remove disconnected clients
        for websocket in disconnected:
            self.active_connections[job_id].discard(websocket)

    async def _consume_stream(self, job_id: str) -> None:
        """Consume updates from Redis stream for a job.

        Args:
            job_id: The job ID
        """
        redis_client = await self.get_redis_client()
        stream_key = self.redis_manager.get_stream_key(job_id)
        consumer_group = f"websocket-{job_id}"
        consumer_name = f"consumer-{id(self)}"

        try:
            # Create consumer group (ignore if already exists)
            try:
                await redis_client.xgroup_create(stream_key, consumer_group, id="0")
            except Exception:
                pass  # Group already exists

            # First, read any existing messages from the beginning
            existing_messages = await redis_client.xread({stream_key: "0"}, count=100)
            for stream_name, messages in existing_messages:
                for message_id, data in messages:
                    await self._process_message(job_id, data)

            # Now consume new messages
            while job_id in self.active_connections:
                try:
                    # Read new messages with blocking
                    messages = await redis_client.xreadgroup(
                        consumer_group, consumer_name, {stream_key: ">"}, count=10, block=1000  # Block for 1 second
                    )

                    for stream_name, stream_messages in messages:
                        for message_id, data in stream_messages:
                            await self._process_message(job_id, data)
                            # Acknowledge the message
                            await redis_client.xack(stream_key, consumer_group, message_id)

                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(f"Error consuming stream for job {job_id}: {e}")
                    await asyncio.sleep(1)

        except asyncio.CancelledError:
            logger.info(f"Stream consumer cancelled for job {job_id}")
        finally:
            # Clean up consumer group
            try:
                await redis_client.xgroup_destroy(stream_key, consumer_group)
            except Exception:
                pass

    async def _process_message(self, job_id: str, data: dict[str, str]) -> None:
        """Process a message from Redis stream.

        Args:
            job_id: The job ID
            data: The message data
        """
        try:
            message = {
                "type": data.get("type", "update"),
                "timestamp": data.get("timestamp", datetime.now(UTC).isoformat()),
                "data": json.loads(data.get("data", "{}")),
            }
            await self.send_to_job(job_id, message)
        except Exception as e:
            logger.error(f"Error processing message for job {job_id}: {e}")

    async def send_initial_state(self, websocket: WebSocket, job_id: str, job_data: dict[str, Any]) -> None:
        """Send initial job state to a newly connected client.

        Args:
            websocket: The WebSocket connection
            job_id: The job ID
            job_data: The current job data from database
        """
        try:
            initial_message = {"type": "initial_state", "timestamp": datetime.now(UTC).isoformat(), "data": job_data}
            await websocket.send_json(initial_message)
        except Exception as e:
            logger.error(f"Error sending initial state for job {job_id}: {e}")

    async def close(self) -> None:
        """Close all connections and clean up resources."""
        # Cancel all consumer tasks
        for task in self.consumer_tasks.values():
            task.cancel()

        # Close all WebSocket connections
        for job_id, connections in list(self.active_connections.items()):
            for websocket in list(connections):
                try:
                    await websocket.close()
                except Exception:
                    pass

        # Clear data structures
        self.active_connections.clear()
        self.consumer_tasks.clear()

        # Close Redis connection
        if self._redis_client:
            await self._redis_client.close()
            self._redis_client = None

        await self.redis_manager.close_async()


# Global instance
_websocket_manager: RedisStreamWebSocketManager | None = None


def get_websocket_manager(redis_url: str) -> RedisStreamWebSocketManager:
    """Get or create the global WebSocket manager.

    Args:
        redis_url: Redis connection URL

    Returns:
        The WebSocket manager instance
    """
    global _websocket_manager
    if _websocket_manager is None:
        _websocket_manager = RedisStreamWebSocketManager(redis_url)
    return _websocket_manager
