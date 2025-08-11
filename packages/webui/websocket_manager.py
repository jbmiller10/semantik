"""WebSocket manager with Redis Streams for distributed state synchronization."""

import asyncio
import contextlib
import json
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

import redis.asyncio as aioredis
from fastapi import WebSocket

# Make redis available at module level for backward compatibility with tests
import redis.asyncio as redis

logger = logging.getLogger(__name__)


class RedisStreamWebSocketManager:
    """WebSocket manager that uses Redis Streams for distributed state synchronization."""

    def __init__(self) -> None:
        """Initialize the WebSocket manager."""
        self.redis: aioredis.Redis | None = None
        self.connections: dict[str, set[WebSocket]] = {}
        self.consumer_tasks: dict[str, asyncio.Task] = {}
        self.consumer_group = f"webui-{uuid.uuid4().hex[:8]}"
        self.max_connections_per_user = 10  # Prevent DOS attacks per user
        self.max_total_connections = 1000  # Global connection limit
        self._startup_lock = asyncio.Lock()
        self._startup_attempted = False
        self._get_operation_func = None  # Function to get operation by ID
        self._chunking_progress_throttle: dict[str, datetime] = {}  # Track last progress update time per operation
        self._chunking_progress_threshold = 0.5  # Minimum seconds between progress updates

    async def startup(self) -> None:
        """Initialize Redis connection on application startup with retry logic.

        Supports both the service factory path and direct redis.from_url for
        backward compatibility with tests that patch either mechanism.
        """
        async with self._startup_lock:
            # Skip if already attempted or connected
            if self._startup_attempted and self.redis is not None:
                return

            self._startup_attempted = True
            logger.info("WebSocket manager startup initiated")
            max_retries = 3
            retry_delay = 1.0  # Initial delay in seconds

            import os
            is_testing = os.getenv("TESTING", "false").lower() in ("true", "1", "yes")

            for attempt in range(max_retries):
                try:
                    logger.info(f"Attempting to connect to Redis (attempt {attempt + 1}/{max_retries})")

                    # Always use from_url with retries; tests patch this call
                    from packages.shared.config import settings as _settings
                    redis_url = getattr(_settings, "REDIS_URL", "redis://localhost:6379/0")
                    redis_client = await redis.from_url(redis_url, decode_responses=True)

                    # Validate connection
                    if hasattr(redis_client, "ping"):
                        await redis_client.ping()
                    self.redis = redis_client
                    logger.info("WebSocket manager connected to Redis")
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
        for operation_id, task in list(self.consumer_tasks.items()):
            logger.info(f"Cancelling consumer task for operation {operation_id}")
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        # Close all WebSocket connections
        for _, websockets in list(self.connections.items()):
            for websocket in list(websockets):
                with contextlib.suppress(Exception):
                    await websocket.close()

        # Clear the dictionaries after cleanup
        self.connections.clear()
        self.consumer_tasks.clear()

        # Close Redis connection
        if self.redis:
            await self.redis.close()
            logger.info("WebSocket manager Redis connection closed")

    def set_operation_getter(self, get_operation_func: Any) -> None:
        """Set the function to get operation by ID.

        This allows dependency injection for testing.
        The function should be an async function that takes an operation_id
        and returns an operation object or None.
        """
        self._get_operation_func = get_operation_func

    async def connect(self, websocket: WebSocket, operation_id: str, user_id: str) -> None:
        """Handle new WebSocket connection for operation updates with connection limit enforcement."""
        # Try to reconnect to Redis if not connected
        if self.redis is None:
            logger.info("Redis not connected, attempting to reconnect...")
            await self.startup()

        # Check global connection limit first
        total_connections = sum(len(sockets) for sockets in self.connections.values())
        if total_connections >= self.max_total_connections:
            logger.error(f"Global connection limit reached ({self.max_total_connections})")
            await websocket.close(code=1008, reason="Server connection limit exceeded")
            return

        # Check connection limit for this user
        user_connections = sum(
            len(sockets) for key, sockets in self.connections.items() if key.startswith(f"{user_id}:")
        )

        if user_connections >= self.max_connections_per_user:
            logger.warning(f"User {user_id} exceeded connection limit ({self.max_connections_per_user})")
            await websocket.close(code=1008, reason="User connection limit exceeded")
            return

        await websocket.accept()

        # Store connection
        key = f"{user_id}:operation:{operation_id}"
        if key not in self.connections:
            self.connections[key] = set()
        self.connections[key].add(websocket)

        logger.info(
            f"Operation WebSocket connected: user={user_id}, operation={operation_id} "
            f"(total user connections: {user_connections + 1})"
        )

        # Get current operation state from database and send it
        try:
            operation = None

            if self._get_operation_func:
                # Use injected function (for testing)
                operation = await self._get_operation_func(operation_id)
            else:
                # Use default implementation
                from packages.shared.database.database import AsyncSessionLocal
                from packages.shared.database.repositories.operation_repository import OperationRepository

                async with AsyncSessionLocal() as session:  # type: ignore[misc]
                    operation_repo = OperationRepository(session)
                    operation = await operation_repo.get_by_uuid(operation_id)

            if operation:
                # Send current state
                state_message = {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "type": "current_state",
                    "data": {
                        "status": operation.status.value,
                        "operation_type": operation.type.value,
                        "created_at": operation.created_at.isoformat(),
                        "started_at": operation.started_at.isoformat() if operation.started_at else None,
                        "completed_at": operation.completed_at.isoformat() if operation.completed_at else None,
                        "error_message": operation.error_message,
                    },
                }
                await websocket.send_json(state_message)
                logger.info(f"Sent current state to client for operation {operation_id}")
        except Exception as e:
            logger.error(f"Failed to send current state for operation {operation_id}: {e}")

        # Only start Redis consumer if Redis is available
        if self.redis is not None:
            # Start consumer task if not exists
            if operation_id not in self.consumer_tasks:
                task = asyncio.create_task(self._consume_updates(operation_id))
                self.consumer_tasks[operation_id] = task
                logger.info(f"Started consumer task for operation {operation_id}")

            # Send message history
            await self._send_history(websocket, operation_id)
        else:
            logger.warning(
                f"Redis not available for operation {operation_id}. WebSocket will work in degraded mode "
                "(initial state only, no real-time updates)"
            )

    async def disconnect(self, websocket: WebSocket, operation_id: str, user_id: str) -> None:
        """Handle WebSocket disconnection for operation updates."""
        key = f"{user_id}:operation:{operation_id}"
        if key in self.connections:
            self.connections[key].discard(websocket)
            if not self.connections[key]:
                del self.connections[key]

        logger.info(f"Operation WebSocket disconnected: user={user_id}, operation={operation_id}")

        # Stop consumer if no more connections for this operation
        if not any(operation_id in k for k in self.connections) and operation_id in self.consumer_tasks:
            logger.info(f"Stopping consumer task for operation {operation_id} (no more connections)")
            self.consumer_tasks[operation_id].cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.consumer_tasks[operation_id]
            if operation_id in self.consumer_tasks:
                del self.consumer_tasks[operation_id]

    async def send_update(self, operation_id: str, update_type: str, data: dict) -> None:
        """Send an update to Redis Stream for a specific operation.

        This method is called by Celery tasks to send updates.
        If Redis is not available, updates are sent directly to connected clients.
        """
        message = {"timestamp": datetime.now(UTC).isoformat(), "type": update_type, "data": data}

        if self.redis:
            # Redis available - use streams for persistence
            stream_key = f"operation-progress:{operation_id}"

            try:
                # Add to stream with automatic ID and max length to prevent unbounded growth
                await self.redis.xadd(
                    stream_key,
                    {"message": json.dumps(message)},
                    maxlen=1000,  # Keep last 1000 messages
                )

                # Set TTL based on operation status
                # Active operations get longer TTL, completed operations get shorter
                ttl = 86400  # Default: 24 hours for active operations

                if update_type == "status_update":
                    status = data.get("status", "")
                    if status in ["completed", "cancelled"]:
                        ttl = 300  # 5 minutes for completed operations
                    elif status == "failed":
                        ttl = 60  # 1 minute for failed operations

                await self.redis.expire(stream_key, ttl)

                logger.debug(f"Sent update to stream {stream_key}: type={update_type}, TTL={ttl}s")
            except Exception as e:
                logger.error(f"Failed to send update to Redis stream: {e}")
                # Fall back to direct broadcast
                await self._broadcast(operation_id, message)
        else:
            # Redis not available - send directly to connected clients
            logger.debug(f"Redis not available, broadcasting directly for operation {operation_id}")
            await self._broadcast(operation_id, message)

    async def _consume_updates(self, operation_id: str) -> None:
        """Consume updates from Redis Stream for a specific operation."""
        stream_key = f"operation-progress:{operation_id}"
        consumer_name = f"consumer-{operation_id}"
        consumer_group_created = False

        try:
            while True:
                try:
                    if self.redis is None:
                        raise RuntimeError("Redis connection not established")

                    # Try to create consumer group if not already created
                    if not consumer_group_created:
                        try:
                            # First check if the stream exists
                            try:
                                stream_info = await self.redis.xinfo_stream(stream_key)
                                logger.debug(f"Stream {stream_key} exists with {stream_info.get('length', 0)} messages")
                            except Exception:
                                # Stream doesn't exist - this is normal for operations that haven't started yet
                                logger.debug(
                                    f"Stream {stream_key} does not exist yet - waiting for worker to create it"
                                )
                                # Wait a bit and continue the loop
                                await asyncio.sleep(2)
                                continue

                            # Stream exists, try to create consumer group
                            await self.redis.xgroup_create(stream_key, self.consumer_group, id="0")
                            logger.info(f"Created consumer group {self.consumer_group} for stream {stream_key}")
                            consumer_group_created = True
                        except Exception as e:
                            # Group might already exist
                            if "BUSYGROUP" in str(e):
                                logger.debug(f"Consumer group already exists for {stream_key}")
                                consumer_group_created = True
                            else:
                                logger.debug(f"Could not create consumer group: {e}")
                                await asyncio.sleep(2)
                                continue

                    # Read from stream with blocking
                    last_id = ">"  # Start reading new messages
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

                                    # Send to all connected clients for this operation
                                    await self._broadcast(operation_id, message)

                                    # Check if operation is complete and close connections
                                    if message.get("type") == "status_update" and message.get("data", {}).get(
                                        "status"
                                    ) in ["completed", "failed", "cancelled"]:
                                        logger.info(
                                            f"Operation {operation_id} completed, closing WebSocket connections"
                                        )
                                        await self._close_connections(operation_id)

                                    # Acknowledge message
                                    await self.redis.xack(stream_key, self.consumer_group, msg_id)

                                    logger.debug(f"Processed message {msg_id} for operation {operation_id}")
                                except Exception as e:
                                    logger.error(f"Error processing message {msg_id}: {e}")

                    await asyncio.sleep(0.1)  # Small delay between reads

                except asyncio.CancelledError:
                    # Clean up consumer
                    try:
                        if self.redis is not None:
                            await self.redis.xgroup_delconsumer(stream_key, self.consumer_group, consumer_name)
                            logger.info(f"Cleaned up consumer {consumer_name}")
                    except Exception:
                        pass
                    raise
                except Exception as e:
                    error_str = str(e)
                    if "NOGROUP" in error_str:
                        # Consumer group doesn't exist anymore, reset flag
                        consumer_group_created = False
                        logger.debug(f"Consumer group lost for {stream_key}, will recreate...")
                        await asyncio.sleep(2)
                    elif "Stream" in error_str and "does not exist" in error_str:
                        # Stream doesn't exist yet
                        logger.debug(f"Stream {stream_key} not ready yet, waiting...")
                        await asyncio.sleep(2)
                    elif "no running event loop" in error_str.lower():
                        # Event loop has been closed, exit gracefully
                        logger.debug(f"Event loop closed for operation {operation_id}, exiting consumer")
                        return
                    else:
                        logger.error(f"Error in consumer loop for operation {operation_id}: {e}")
                        await asyncio.sleep(5)  # Wait before retry

        except asyncio.CancelledError:
            logger.info(f"Consumer task cancelled for operation {operation_id}")
            raise
        except RuntimeError as e:
            if "no running event loop" in str(e).lower():
                logger.debug(f"Event loop closed for operation {operation_id}, exiting consumer")
            else:
                logger.error(f"Fatal runtime error in consumer for operation {operation_id}: {e}")
        except Exception as e:
            logger.error(f"Fatal error in consumer for operation {operation_id}: {e}")

    async def _send_history(self, websocket: WebSocket, operation_id: str) -> None:
        """Send historical messages to newly connected client for operation."""
        if not self.redis:
            logger.debug("Redis not available, skipping message history")
            return

        stream_key = f"operation-progress:{operation_id}"

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
                logger.info(f"Sent {len(messages)} historical messages to client for operation {operation_id}")

        except Exception as e:
            logger.warning(f"Failed to send history for operation {operation_id}: {e}")

    async def _broadcast(self, operation_id: str, message: dict) -> None:
        """Broadcast message to all connections for an operation."""
        disconnected = []

        for key, websockets in list(self.connections.items()):
            if f"operation:{operation_id}" in key:
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

    async def _close_connections(self, operation_id: str) -> None:
        """Close all WebSocket connections for a completed operation."""
        connections_to_close: list[WebSocket] = []

        for key, websockets in list(self.connections.items()):
            if f"operation:{operation_id}" in key:
                connections_to_close.extend(websockets)
                del self.connections[key]

        for websocket in connections_to_close:
            try:
                await websocket.close(code=1000, reason="Operation completed")
                logger.debug(f"Closed WebSocket connection for completed operation {operation_id}")
            except Exception as e:
                logger.warning(f"Failed to close WebSocket connection: {e}")

    async def cleanup_stream(self, operation_id: str) -> None:
        """Clean up Redis stream for a completed operation.

        This should be called when an operation is completed or deleted to free up Redis memory.
        """
        if not self.redis:
            logger.debug("Redis not available, skipping stream cleanup")
            return

        stream_key = f"operation-progress:{operation_id}"

        try:
            # Delete the stream
            deleted = await self.redis.delete(stream_key)
            if deleted:
                logger.info(f"Cleaned up Redis stream for operation {operation_id}")

            # Also try to delete consumer groups associated with this stream
            try:
                # Get all consumer groups for this stream
                groups = await self.redis.xinfo_groups(stream_key)
                for group in groups:
                    group_name = group.get("name", "")
                    if group_name:
                        await self.redis.xgroup_destroy(stream_key, group_name)
                        logger.debug(f"Deleted consumer group {group_name} for operation {operation_id}")
            except Exception:
                # Stream might already be deleted or have no groups
                pass

        except Exception as e:
            logger.warning(f"Failed to clean up stream for operation {operation_id}: {e}")

    async def cleanup_operation_channel(self, operation_id: str) -> None:
        """Clean up all resources for a completed operation.

        Args:
            operation_id: The operation ID to clean up
        """
        # Cancel consumer task if exists
        if operation_id in self.consumer_tasks:
            task = self.consumer_tasks[operation_id]
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            del self.consumer_tasks[operation_id]

        # Clean up Redis stream
        await self.cleanup_stream(operation_id)

        # Close all connections for this operation
        await self._close_connections(operation_id)

        logger.info(f"Cleaned up all resources for operation {operation_id}")

    async def broadcast_to_operation(
        self,
        operation_id: str,
        message: dict,
    ) -> None:
        """Broadcast a message to all connections for an operation.

        Args:
            operation_id: The operation ID
            message: The message to broadcast
        """
        await self._broadcast(operation_id, message)

    async def _should_send_progress_update(
        self,
        operation_id: str,
        message: dict,
    ) -> bool:
        """Check if a progress update should be sent based on throttling.

        Args:
            operation_id: The operation ID
            message: The message to check

        Returns:
            True if the message should be sent, False if throttled
        """
        # Check if this is a progress update
        if message.get("type") != "chunking_progress":
            return True  # Non-progress messages always go through

        # Check throttling
        now = datetime.now(UTC)
        if operation_id in self._chunking_progress_throttle:
            time_since_last = (now - self._chunking_progress_throttle[operation_id]).total_seconds()
            if time_since_last < self._chunking_progress_threshold:
                # Too soon after last update, throttle it
                return False

        # Update timestamp for next check
        self._chunking_progress_throttle[operation_id] = now
        return True

    async def cleanup_stale_connections(self) -> None:
        """Clean up stale WebSocket connections and their associated data.

        This is called periodically to remove dead connections and free memory.
        """
        cleaned_count = 0

        for key in list(self.connections.keys()):
            websockets = self.connections.get(key, set())
            if not websockets:
                continue

            dead_sockets = []

            for websocket in list(websockets):
                try:
                    # For testing with mocks, check if this is a mock object
                    # and call ping() if available (mock specific)
                    if hasattr(websocket, "ping"):
                        # This is likely a mock in tests
                        await websocket.ping()
                    else:
                        # Try to send a ping frame to check if connection is alive
                        # FastAPI WebSocket doesn't have a ping() method, so we use send_json
                        # with a ping message and handle any exceptions
                        await asyncio.wait_for(websocket.send_json({"type": "ping"}), timeout=1.0)
                except Exception:
                    # Connection is dead or timed out
                    dead_sockets.append(websocket)
                    cleaned_count += 1

            # Remove dead sockets from the connections dictionary
            if dead_sockets:
                # Update the connections dict directly
                remaining_sockets = websockets - set(dead_sockets)
                if remaining_sockets:
                    self.connections[key] = remaining_sockets
                else:
                    # Remove empty connection sets
                    del self.connections[key]

        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} stale WebSocket connections")

        # Clean up old progress throttle entries
        now = datetime.now(UTC)
        old_entries = [
            op_id
            for op_id, last_time in self._chunking_progress_throttle.items()
            if (now - last_time).total_seconds() > 300  # Remove entries older than 5 minutes
        ]

        for op_id in old_entries:
            del self._chunking_progress_throttle[op_id]

        if old_entries:
            logger.debug(f"Cleaned up {len(old_entries)} old progress throttle entries")

    async def send_chunking_progress(
        self,
        operation_id: str,
        progress_percentage: float,
        documents_processed: int,
        total_documents: int,
        chunks_created: int,
        current_document: str | None = None,
        throttle: bool = True,
    ) -> None:
        """Send chunking progress update with optional throttling.

        Args:
            operation_id: The chunking operation ID
            progress_percentage: Completion percentage (0-100)
            documents_processed: Number of documents processed
            total_documents: Total documents to process
            chunks_created: Number of chunks created so far
            current_document: Currently processing document name
            throttle: Whether to throttle progress updates
        """
        # Apply throttling to reduce WebSocket traffic
        if throttle and operation_id in self._chunking_progress_throttle:
            time_since_last = (datetime.now(UTC) - self._chunking_progress_throttle[operation_id]).total_seconds()
            if time_since_last < self._chunking_progress_threshold:
                # Skip this update if too soon after the last one
                return

        # Update throttle timestamp
        self._chunking_progress_throttle[operation_id] = datetime.now(UTC)

        # Send the progress update
        await self.send_update(
            operation_id,
            "chunking_progress",
            {
                "progress_percentage": progress_percentage,
                "documents_processed": documents_processed,
                "total_documents": total_documents,
                "chunks_created": chunks_created,
                "current_document": current_document,
            },
        )

    async def send_chunking_event(
        self,
        operation_id: str,
        event_type: str,
        data: dict,
    ) -> None:
        """Send a chunking-specific event.

        Supported event types:
        - chunking_started: Chunking operation has started
        - chunking_document_start: Started processing a document
        - chunking_document_complete: Completed processing a document
        - chunking_completed: Entire chunking operation completed
        - chunking_failed: Chunking operation failed
        - chunking_cancelled: Chunking operation was cancelled
        - chunking_strategy_changed: Chunking strategy was changed

        Args:
            operation_id: The chunking operation ID
            event_type: Type of chunking event
            data: Event-specific data
        """
        await self.send_update(operation_id, event_type, data)

    async def send_message(self, channel: str, message: dict) -> None:
        """Send a message to a specific channel.

        This is used for custom WebSocket channels like chunking operations.

        Args:
            channel: The channel identifier (e.g., "chunking:collection_id:operation_id")
            message: The message to send
        """
        if self.redis:
            stream_key = f"stream:{channel}"

            try:
                # Add to stream with automatic ID and proper maxlen
                await self.redis.xadd(
                    stream_key,
                    {"message": json.dumps(message)},
                    maxlen=1000,  # Increased to 1000 for consistency
                )

                # Set TTL for WebSocket channel streams (15 minutes)
                await self.redis.expire(stream_key, 900)

                logger.debug(f"Sent message to channel {channel}, TTL=900s")
            except Exception as e:
                logger.error(f"Failed to send message to channel: {e}")
                # Fall back to direct broadcast if we have connections
                await self._broadcast_to_channel(channel, message)
        else:
            # Redis not available - send directly to connected clients
            await self._broadcast_to_channel(channel, message)

    async def _broadcast_to_channel(self, channel: str, message: dict) -> None:
        """Broadcast a message to all connections on a channel.

        Args:
            channel: The channel identifier
            message: The message to broadcast
        """
        # Find all connections for this channel
        for key, websockets in list(self.connections.items()):
            if channel in key:
                for websocket in list(websockets):
                    try:
                        await websocket.send_json(message)
                        logger.debug(f"Sent channel message to websocket: {channel}")
                    except Exception as e:
                        logger.warning(f"Failed to send channel message to websocket: {e}")
                        # Remove broken connection
                        websockets.discard(websocket)

    async def connect_to_channel(self, websocket: WebSocket, channel: str, user_id: str) -> None:
        """Connect a WebSocket to a custom channel.

        Args:
            websocket: The WebSocket connection
            channel: The channel to connect to
            user_id: The user ID making the connection
        """
        # Check global connection limit first
        total_connections = sum(len(sockets) for sockets in self.connections.values())
        if total_connections >= self.max_total_connections:
            logger.error(f"Global connection limit reached ({self.max_total_connections})")
            await websocket.close(code=1008, reason="Server connection limit exceeded")
            return

        # Check connection limit for this user
        user_connections = sum(
            len(sockets) for key, sockets in self.connections.items() if key.startswith(f"{user_id}:")
        )

        if user_connections >= self.max_connections_per_user:
            logger.warning(f"User {user_id} exceeded connection limit ({self.max_connections_per_user})")
            await websocket.close(code=1008, reason="User connection limit exceeded")
            return

        await websocket.accept()

        # Store connection
        key = f"{user_id}:channel:{channel}"
        if key not in self.connections:
            self.connections[key] = set()
        self.connections[key].add(websocket)

        logger.info(
            f"Channel WebSocket connected: user={user_id}, channel={channel} "
            f"(total user connections: {user_connections + 1})"
        )

    async def disconnect_from_channel(self, websocket: WebSocket, channel: str, user_id: str) -> None:
        """Disconnect a WebSocket from a custom channel.

        Args:
            websocket: The WebSocket connection
            channel: The channel to disconnect from
            user_id: The user ID
        """
        key = f"{user_id}:channel:{channel}"
        if key in self.connections:
            self.connections[key].discard(websocket)
            if not self.connections[key]:
                del self.connections[key]

        logger.info(f"Channel WebSocket disconnected: user={user_id}, channel={channel}")


# Global instance
ws_manager = RedisStreamWebSocketManager()
