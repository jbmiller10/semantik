"""WebSocket manager with Redis Streams for distributed state synchronization."""

import asyncio
import contextlib
import json
import logging
import uuid
from datetime import UTC, datetime

import redis.asyncio as redis
from fastapi import WebSocket
from shared.config import settings

logger = logging.getLogger(__name__)


class RedisStreamWebSocketManager:
    """WebSocket manager that uses Redis Streams for distributed state synchronization."""

    def __init__(self):
        """Initialize the WebSocket manager."""
        self.redis: redis.Redis | None = None
        self.connections: dict[str, set[WebSocket]] = {}
        self.consumer_tasks: dict[str, asyncio.Task] = {}
        self.consumer_group = f"webui-{uuid.uuid4().hex[:8]}"
        self.redis_url = settings.REDIS_URL
        self.max_connections_per_user = 10  # Prevent DOS attacks

    async def startup(self) -> None:
        """Initialize Redis connection on application startup with retry logic."""
        max_retries = 3
        retry_delay = 1.0  # Initial delay in seconds

        for attempt in range(max_retries):
            try:
                self.redis = await redis.from_url(
                    self.redis_url,
                    decode_responses=True,
                    health_check_interval=30,
                    socket_keepalive=True,
                    retry_on_timeout=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                )
                # Test connection
                await self.redis.ping()
                logger.info(f"WebSocket manager connected to Redis at {self.redis_url}")
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2**attempt)  # Exponential backoff
                    logger.warning(
                        f"Failed to connect to Redis (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {wait_time:.1f} seconds..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Failed to connect to Redis after {max_retries} attempts: {e}")
                    # Don't raise - allow graceful degradation
                    self.redis = None

    async def shutdown(self) -> None:
        """Clean up resources on application shutdown."""
        # Cancel all consumer tasks
        for job_id, task in list(self.consumer_tasks.items()):
            logger.info(f"Cancelling consumer task for job {job_id}")
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        # Close all WebSocket connections
        for _, websockets in list(self.connections.items()):
            for websocket in list(websockets):
                with contextlib.suppress(Exception):
                    await websocket.close()

        # Close Redis connection
        if self.redis:
            await self.redis.close()
            logger.info("WebSocket manager Redis connection closed")

    async def connect(self, websocket: WebSocket, job_id: str, user_id: str) -> None:
        """Handle new WebSocket connection with connection limit enforcement."""
        # Check connection limit for this user
        user_connections = sum(
            len(sockets) for key, sockets in self.connections.items() if key.startswith(f"{user_id}:")
        )

        if user_connections >= self.max_connections_per_user:
            logger.warning(f"User {user_id} exceeded connection limit ({self.max_connections_per_user})")
            await websocket.close(code=1008, reason="Connection limit exceeded")
            return

        await websocket.accept()

        # Store connection
        key = f"{user_id}:{job_id}"
        if key not in self.connections:
            self.connections[key] = set()
        self.connections[key].add(websocket)

        logger.info(
            f"WebSocket connected: user={user_id}, job={job_id} (total user connections: {user_connections + 1})"
        )

        # Get current job state from database and send it
        try:
            from shared.database.factory import create_job_repository

            job_repo = create_job_repository()
            job = await job_repo.get_job(job_id)

            if job:
                # Send current state
                state_message = {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "type": "current_state",
                    "data": {
                        "status": job["status"],
                        "total_files": job.get("total_files", 0),
                        "processed_files": job.get("processed_files", 0),
                        "failed_files": job.get("failed_files", 0),
                        "current_file": job.get("current_file"),
                        "error": job.get("error"),
                    },
                }
                await websocket.send_json(state_message)
                logger.info(f"Sent current state to client for job {job_id}")
        except Exception as e:
            logger.error(f"Failed to send current state for job {job_id}: {e}")

        # Only start Redis consumer if Redis is available
        if self.redis is not None:
            # Start consumer task if not exists
            if job_id not in self.consumer_tasks:
                task = asyncio.create_task(self._consume_updates(job_id))
                self.consumer_tasks[job_id] = task
                logger.info(f"Started consumer task for job {job_id}")

            # Send message history
            await self._send_history(websocket, job_id)
        else:
            logger.warning(
                f"Redis not available for job {job_id}. WebSocket will work in degraded mode "
                "(initial state only, no real-time updates)"
            )

    async def disconnect(self, websocket: WebSocket, job_id: str, user_id: str) -> None:
        """Handle WebSocket disconnection."""
        key = f"{user_id}:{job_id}"
        if key in self.connections:
            self.connections[key].discard(websocket)
            if not self.connections[key]:
                del self.connections[key]

        logger.info(f"WebSocket disconnected: user={user_id}, job={job_id}")

        # Stop consumer if no more connections for this job
        if not any(job_id in k for k in self.connections) and job_id in self.consumer_tasks:
            logger.info(f"Stopping consumer task for job {job_id} (no more connections)")
            self.consumer_tasks[job_id].cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.consumer_tasks[job_id]
            del self.consumer_tasks[job_id]

    async def send_job_update(self, job_id: str, update_type: str, data: dict) -> None:
        """Send an update to Redis Stream for a specific job.

        This method is called by Celery tasks to send updates.
        If Redis is not available, updates are sent directly to connected clients.
        """
        message = {"timestamp": datetime.now(UTC).isoformat(), "type": update_type, "data": data}

        if self.redis:
            # Redis available - use streams for persistence
            stream_key = f"job:updates:{job_id}"

            try:
                # Add to stream with automatic ID
                await self.redis.xadd(
                    stream_key, {"message": json.dumps(message)}, maxlen=1000  # Keep last 1000 messages
                )

                # Set TTL on first message (24 hours)
                await self.redis.expire(stream_key, 86400)

                logger.debug(f"Sent update to stream {stream_key}: type={update_type}")
            except Exception as e:
                logger.error(f"Failed to send update to Redis stream: {e}")
                # Fall back to direct broadcast
                await self._broadcast_to_job(job_id, message)
        else:
            # Redis not available - send directly to connected clients
            logger.debug(f"Redis not available, broadcasting directly for job {job_id}")
            await self._broadcast_to_job(job_id, message)

    async def _consume_updates(self, job_id: str) -> None:
        """Consume updates from Redis Stream for a specific job."""
        stream_key = f"job:updates:{job_id}"

        try:
            # Create consumer group
            try:
                await self.redis.xgroup_create(stream_key, self.consumer_group, id="0")
                logger.info(f"Created consumer group {self.consumer_group} for stream {stream_key}")
            except Exception as e:
                # Group might already exist
                logger.debug(f"Consumer group might already exist: {e}")

            consumer_name = f"consumer-{job_id}"
            last_id = ">"  # Start reading new messages

            while True:
                try:
                    # Read from stream with blocking
                    messages = await self.redis.xreadgroup(
                        self.consumer_group,
                        consumer_name,
                        {stream_key: last_id},
                        count=10,
                        block=1000,  # 1 second timeout
                    )

                    if messages:
                        for _, stream_messages in messages:
                            for msg_id, data in stream_messages:
                                try:
                                    # Parse message
                                    message = json.loads(data["message"])

                                    # Send to all connected clients for this job
                                    await self._broadcast_to_job(job_id, message)

                                    # Acknowledge message
                                    await self.redis.xack(stream_key, self.consumer_group, msg_id)

                                    logger.debug(f"Processed message {msg_id} for job {job_id}")
                                except Exception as e:
                                    logger.error(f"Error processing message {msg_id}: {e}")

                    await asyncio.sleep(0.1)  # Small delay between reads

                except asyncio.CancelledError:
                    # Clean up consumer
                    try:
                        await self.redis.xgroup_delconsumer(stream_key, self.consumer_group, consumer_name)
                        logger.info(f"Cleaned up consumer {consumer_name}")
                    except Exception:
                        pass
                    raise
                except Exception as e:
                    logger.error(f"Error in consumer loop for job {job_id}: {e}")
                    await asyncio.sleep(5)  # Wait before retry

        except asyncio.CancelledError:
            logger.info(f"Consumer task cancelled for job {job_id}")
            raise
        except Exception as e:
            logger.error(f"Fatal error in consumer for job {job_id}: {e}")

    async def _send_history(self, websocket: WebSocket, job_id: str) -> None:
        """Send historical messages to newly connected client."""
        if not self.redis:
            logger.debug("Redis not available, skipping message history")
            return

        stream_key = f"job:updates:{job_id}"

        try:
            # Read last 100 messages
            messages = await self.redis.xrange(stream_key, min="-", max="+", count=100)

            for _, data in messages:
                try:
                    message = json.loads(data["message"])
                    await websocket.send_json(message)
                except Exception as e:
                    logger.warning(f"Failed to send historical message: {e}")

            if messages:
                logger.info(f"Sent {len(messages)} historical messages to client for job {job_id}")

        except Exception as e:
            logger.warning(f"Failed to send history for job {job_id}: {e}")

    async def _broadcast_to_job(self, job_id: str, message: dict) -> None:
        """Broadcast message to all connections for a job."""
        disconnected = []

        for key, websockets in list(self.connections.items()):
            if job_id in key:
                for websocket in list(websockets):
                    try:
                        await websocket.send_json(message)
                    except Exception as e:
                        logger.warning(f"Failed to send message to websocket: {e}")
                        disconnected.append((key, websocket))

        # Clean up disconnected clients
        for key, websocket in disconnected:
            self.connections[key].discard(websocket)
            if not self.connections[key]:
                del self.connections[key]

    async def cleanup_job_stream(self, job_id: str) -> None:
        """Clean up Redis stream for a completed job.

        This should be called when a job is completed or deleted to free up Redis memory.
        """
        if not self.redis:
            logger.debug("Redis not available, skipping stream cleanup")
            return

        stream_key = f"job:updates:{job_id}"

        try:
            # Delete the stream
            deleted = await self.redis.delete(stream_key)
            if deleted:
                logger.info(f"Cleaned up Redis stream for job {job_id}")

            # Also try to delete consumer groups associated with this stream
            try:
                # Get all consumer groups for this stream
                groups = await self.redis.xinfo_groups(stream_key)
                for group in groups:
                    group_name = group.get("name", "")
                    if group_name:
                        await self.redis.xgroup_destroy(stream_key, group_name)
                        logger.debug(f"Deleted consumer group {group_name} for job {job_id}")
            except Exception:
                # Stream might already be deleted or have no groups
                pass

        except Exception as e:
            logger.warning(f"Failed to clean up stream for job {job_id}: {e}")


# Global instance
ws_manager = RedisStreamWebSocketManager()
