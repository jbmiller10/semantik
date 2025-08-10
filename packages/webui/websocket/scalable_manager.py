"""Horizontally scalable WebSocket manager using Redis Pub/Sub.

This implementation supports:
- 10,000+ concurrent connections
- <100ms message latency
- Cross-instance message routing
- Automatic connection cleanup
- Graceful failover
"""

import asyncio
import contextlib
import json
import logging
import time
import uuid
from datetime import UTC, datetime
from typing import Any

import redis.asyncio as redis
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ScalableWebSocketManager:
    """Horizontally scalable WebSocket manager with Redis Pub/Sub backend.

    Features:
    - Local connection tracking per instance
    - Redis Pub/Sub for cross-instance messaging
    - Connection registry in Redis with TTL
    - Automatic dead connection cleanup
    - Support for user channels and collection broadcasts
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/2",
                 max_connections_per_user: int = 10,
                 max_total_connections: int = 10000) -> None:
        """Initialize the scalable WebSocket manager.

        Args:
            redis_url: Redis connection URL (defaults to DB 2 for WebSocket state)
            max_connections_per_user: Maximum connections allowed per user
            max_total_connections: Maximum total connections per instance
        """
        # Instance identification
        self.instance_id = str(uuid.uuid4())
        self.redis_url = redis_url

        # Connection limits
        self.max_connections_per_user = max_connections_per_user
        self.max_total_connections = max_total_connections

        # Local connection tracking
        self.local_connections: dict[str, WebSocket] = {}
        self.connection_metadata: dict[str, dict[str, Any]] = {}

        # Redis clients
        self.redis_client: redis.Redis | None = None
        self.pubsub: redis.PubSub | None = None

        # Background tasks
        self.listener_task: asyncio.Task | None = None
        self.heartbeat_task: asyncio.Task | None = None
        self.cleanup_task: asyncio.Task | None = None

        # Startup management
        self._startup_lock = asyncio.Lock()
        self._startup_complete = False

        # Message throttling
        self._message_throttle: dict[str, datetime] = {}
        self._throttle_threshold = 0.05  # 50ms between messages per channel

    async def startup(self) -> None:
        """Initialize Redis connections and start background tasks."""
        async with self._startup_lock:
            if self._startup_complete:
                return

            logger.info(f"Starting ScalableWebSocketManager instance {self.instance_id}")

            # Connect to Redis with retry logic
            max_retries = 3
            retry_delay = 1.0

            for attempt in range(max_retries):
                try:
                    logger.info(f"Connecting to Redis (attempt {attempt + 1}/{max_retries})")

                    # Create main Redis client
                    self.redis_client = await redis.from_url(
                        self.redis_url,
                        decode_responses=True,
                        health_check_interval=30,
                        socket_keepalive=True,
                        retry_on_timeout=True,
                        socket_connect_timeout=5,
                        socket_timeout=5,
                    )

                    # Test connection
                    await self.redis_client.ping()

                    # Create pub/sub client
                    self.pubsub = self.redis_client.pubsub()

                    # Subscribe to instance channel
                    await self.pubsub.subscribe(f"instance:{self.instance_id}")

                    # Register instance in Redis
                    await self._register_instance()

                    # Start background tasks
                    self.listener_task = asyncio.create_task(self._listen_for_messages())
                    self.heartbeat_task = asyncio.create_task(self._heartbeat())
                    self.cleanup_task = asyncio.create_task(self._cleanup_dead_connections())

                    logger.info(f"ScalableWebSocketManager started successfully on instance {self.instance_id}")
                    self._startup_complete = True
                    return

                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)
                        logger.warning(f"Failed to start manager: {e}. Retrying in {wait_time:.1f}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Failed to start ScalableWebSocketManager after {max_retries} attempts: {e}")
                        raise

    async def shutdown(self) -> None:
        """Clean up resources and shut down gracefully."""
        logger.info(f"Shutting down ScalableWebSocketManager instance {self.instance_id}")

        # Cancel background tasks
        tasks = [self.listener_task, self.heartbeat_task, self.cleanup_task]
        for task in tasks:
            if task and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        # Close all local WebSocket connections
        for conn_id, websocket in list(self.local_connections.items()):
            try:
                await websocket.close(code=1001, reason="Server shutdown")
            except Exception as e:
                logger.warning(f"Error closing WebSocket {conn_id}: {e}")

        # Clear local tracking
        self.local_connections.clear()
        self.connection_metadata.clear()

        # Unregister instance from Redis
        if self.redis_client:
            try:
                await self.redis_client.delete(f"websocket:instance:{self.instance_id}")

                # Remove instance connections from registry
                connections = await self.redis_client.hgetall("websocket:connections")
                for conn_id, conn_data in connections.items():
                    data = json.loads(conn_data)
                    if data.get("instance_id") == self.instance_id:
                        await self.redis_client.hdel("websocket:connections", conn_id)

            except Exception as e:
                logger.error(f"Error cleaning up Redis state: {e}")

        # Close Redis connections
        if self.pubsub:
            await self.pubsub.aclose()
        if self.redis_client:
            await self.redis_client.aclose()

        logger.info(f"ScalableWebSocketManager instance {self.instance_id} shut down complete")

    async def connect(self, websocket: WebSocket, user_id: str,
                     operation_id: str | None = None,
                     collection_id: str | None = None) -> str:
        """Handle new WebSocket connection.

        Args:
            websocket: The WebSocket connection
            user_id: User identifier
            operation_id: Optional operation ID to subscribe to
            collection_id: Optional collection ID to subscribe to

        Returns:
            Connection ID for this connection
        """
        # Ensure manager is started
        if not self._startup_complete:
            await self.startup()

        # Check global connection limit
        if len(self.local_connections) >= self.max_total_connections:
            logger.error(f"Instance connection limit reached ({self.max_total_connections})")
            await websocket.close(code=1008, reason="Server connection limit exceeded")
            raise ConnectionError("Server connection limit exceeded")

        # Check user connection limit
        user_conn_count = sum(
            1 for metadata in self.connection_metadata.values()
            if metadata.get("user_id") == user_id
        )

        if user_conn_count >= self.max_connections_per_user:
            logger.warning(f"User {user_id} exceeded connection limit ({self.max_connections_per_user})")
            await websocket.close(code=1008, reason="User connection limit exceeded")
            raise ConnectionError("User connection limit exceeded")

        # Accept connection
        await websocket.accept()

        # Generate connection ID
        connection_id = str(uuid.uuid4())

        # Store locally
        self.local_connections[connection_id] = websocket
        self.connection_metadata[connection_id] = {
            "user_id": user_id,
            "operation_id": operation_id,
            "collection_id": collection_id,
            "connected_at": time.time(),
        }

        # Register in Redis
        await self._register_connection(connection_id, user_id, operation_id, collection_id)

        # Subscribe to relevant channels
        channels = [f"user:{user_id}"]
        if operation_id:
            channels.append(f"operation:{operation_id}")
        if collection_id:
            channels.append(f"collection:{collection_id}")

        for channel in channels:
            await self.pubsub.subscribe(channel)

        logger.info(
            f"WebSocket connected: connection={connection_id}, user={user_id}, "
            f"operation={operation_id}, collection={collection_id}, "
            f"instance={self.instance_id}"
        )

        # Send initial state if operation_id provided
        if operation_id:
            await self._send_operation_state(websocket, operation_id)

        return connection_id

    async def disconnect(self, connection_id: str) -> None:
        """Handle WebSocket disconnection.

        Args:
            connection_id: The connection ID to disconnect
        """
        if connection_id not in self.local_connections:
            return

        metadata = self.connection_metadata.get(connection_id, {})
        user_id = metadata.get("user_id")
        operation_id = metadata.get("operation_id")
        collection_id = metadata.get("collection_id")

        # Remove from local tracking
        del self.local_connections[connection_id]
        del self.connection_metadata[connection_id]

        # Remove from Redis registry
        if self.redis_client:
            await self.redis_client.hdel("websocket:connections", connection_id)

            # Remove from user set
            if user_id:
                await self.redis_client.srem(f"websocket:user:{user_id}", connection_id)

                # Check if user has any remaining connections on this instance
                remaining_local = any(
                    m.get("user_id") == user_id
                    for m in self.connection_metadata.values()
                )

                if not remaining_local:
                    # Unsubscribe from user channel if no local connections
                    await self.pubsub.unsubscribe(f"user:{user_id}")

            # Handle operation channel
            if operation_id:
                remaining_op = any(
                    m.get("operation_id") == operation_id
                    for m in self.connection_metadata.values()
                )
                if not remaining_op:
                    await self.pubsub.unsubscribe(f"operation:{operation_id}")

            # Handle collection channel
            if collection_id:
                remaining_coll = any(
                    m.get("collection_id") == collection_id
                    for m in self.connection_metadata.values()
                )
                if not remaining_coll:
                    await self.pubsub.unsubscribe(f"collection:{collection_id}")

        logger.info(
            f"WebSocket disconnected: connection={connection_id}, user={user_id}, "
            f"instance={self.instance_id}"
        )

    async def send_to_user(self, user_id: str, message: dict) -> None:
        """Send message to all connections for a specific user.

        Args:
            user_id: Target user ID
            message: Message to send
        """
        # Check for local connections first
        local_sent = await self._send_to_local_user(user_id, message)

        # If user has connections on other instances, publish to Redis
        if self.redis_client:
            # Check if user has connections on other instances
            all_connections = await self.redis_client.smembers(f"websocket:user:{user_id}")

            # Filter out local connections
            remote_connections = [
                conn_id for conn_id in all_connections
                if conn_id not in self.local_connections
            ]

            if remote_connections:
                # Publish to Redis for other instances
                await self.redis_client.publish(
                    f"user:{user_id}",
                    json.dumps({
                        "message": message,
                        "from_instance": self.instance_id,
                        "timestamp": time.time(),
                    })
                )
                logger.debug(f"Published message to user {user_id} channel for {len(remote_connections)} remote connections")

    async def send_to_operation(self, operation_id: str, message: dict) -> None:
        """Send message to all connections watching an operation.

        Args:
            operation_id: Target operation ID
            message: Message to send
        """
        # Send to local connections
        local_sent = 0
        for conn_id, metadata in self.connection_metadata.items():
            if metadata.get("operation_id") == operation_id:
                websocket = self.local_connections.get(conn_id)
                if websocket:
                    try:
                        await websocket.send_json(message)
                        local_sent += 1
                    except Exception as e:
                        logger.warning(f"Failed to send to connection {conn_id}: {e}")
                        await self.disconnect(conn_id)

        # Publish to Redis for other instances
        if self.redis_client:
            await self.redis_client.publish(
                f"operation:{operation_id}",
                json.dumps({
                    "message": message,
                    "from_instance": self.instance_id,
                    "timestamp": time.time(),
                })
            )

        logger.debug(f"Sent operation message to {local_sent} local connections, published to Redis")

    async def broadcast_to_collection(self, collection_id: str, message: dict) -> None:
        """Broadcast message to all connections watching a collection.

        Args:
            collection_id: Target collection ID
            message: Message to broadcast
        """
        # Send to local connections
        local_sent = 0
        for conn_id, metadata in self.connection_metadata.items():
            if metadata.get("collection_id") == collection_id:
                websocket = self.local_connections.get(conn_id)
                if websocket:
                    try:
                        await websocket.send_json(message)
                        local_sent += 1
                    except Exception as e:
                        logger.warning(f"Failed to send to connection {conn_id}: {e}")
                        await self.disconnect(conn_id)

        # Publish to Redis for other instances
        if self.redis_client:
            await self.redis_client.publish(
                f"collection:{collection_id}",
                json.dumps({
                    "message": message,
                    "from_instance": self.instance_id,
                    "timestamp": time.time(),
                })
            )

        logger.debug(f"Broadcast to collection {collection_id}: {local_sent} local, published to Redis")

    async def _register_instance(self) -> None:
        """Register this instance in Redis with TTL."""
        instance_data = {
            "instance_id": self.instance_id,
            "started_at": time.time(),
            "hostname": await self._get_hostname(),
            "pid": asyncio.get_event_loop()._thread_id if hasattr(asyncio.get_event_loop(), '_thread_id') else None,
        }

        await self.redis_client.setex(
            f"websocket:instance:{self.instance_id}",
            60,  # 60 second TTL
            json.dumps(instance_data)
        )

        logger.info(f"Registered instance {self.instance_id} in Redis")

    async def _register_connection(self, connection_id: str, user_id: str,
                                  operation_id: str | None = None,
                                  collection_id: str | None = None) -> None:
        """Register connection in Redis registry."""
        connection_data = {
            "connection_id": connection_id,
            "user_id": user_id,
            "instance_id": self.instance_id,
            "operation_id": operation_id,
            "collection_id": collection_id,
            "connected_at": time.time(),
        }

        # Store in connections hash
        await self.redis_client.hset(
            "websocket:connections",
            connection_id,
            json.dumps(connection_data)
        )

        # Add to user set
        await self.redis_client.sadd(f"websocket:user:{user_id}", connection_id)

        # Set TTL on user set (refresh on activity)
        await self.redis_client.expire(f"websocket:user:{user_id}", 3600)  # 1 hour

    async def _listen_for_messages(self) -> None:
        """Listen for Redis pub/sub messages and route to local connections."""
        try:
            async for message in self.pubsub.listen():
                if message["type"] == "message":
                    try:
                        channel = message["channel"]
                        data = json.loads(message["data"])

                        # Skip messages from this instance
                        if data.get("from_instance") == self.instance_id:
                            continue

                        msg_content = data.get("message", {})

                        # Route based on channel type
                        if channel.startswith("user:"):
                            user_id = channel.split(":", 1)[1]
                            await self._send_to_local_user(user_id, msg_content)

                        elif channel.startswith("operation:"):
                            operation_id = channel.split(":", 1)[1]
                            await self._send_to_local_operation(operation_id, msg_content)

                        elif channel.startswith("collection:"):
                            collection_id = channel.split(":", 1)[1]
                            await self._send_to_local_collection(collection_id, msg_content)

                        elif channel == f"instance:{self.instance_id}":
                            # Direct instance message
                            await self._handle_instance_message(msg_content)

                    except Exception as e:
                        logger.error(f"Error processing pub/sub message: {e}")

        except asyncio.CancelledError:
            logger.info("Message listener task cancelled")
            raise
        except Exception as e:
            logger.error(f"Fatal error in message listener: {e}")

    async def _send_to_local_user(self, user_id: str, message: dict) -> bool:
        """Send message to local connections for a user.

        Returns:
            True if message was sent to at least one local connection
        """
        sent = False
        for conn_id, metadata in list(self.connection_metadata.items()):
            if metadata.get("user_id") == user_id:
                websocket = self.local_connections.get(conn_id)
                if websocket:
                    try:
                        await websocket.send_json(message)
                        sent = True
                    except Exception as e:
                        logger.warning(f"Failed to send to connection {conn_id}: {e}")
                        await self.disconnect(conn_id)

        return sent

    async def _send_to_local_operation(self, operation_id: str, message: dict) -> None:
        """Send message to local connections watching an operation."""
        for conn_id, metadata in list(self.connection_metadata.items()):
            if metadata.get("operation_id") == operation_id:
                websocket = self.local_connections.get(conn_id)
                if websocket:
                    try:
                        await websocket.send_json(message)
                    except Exception as e:
                        logger.warning(f"Failed to send to connection {conn_id}: {e}")
                        await self.disconnect(conn_id)

    async def _send_to_local_collection(self, collection_id: str, message: dict) -> None:
        """Send message to local connections watching a collection."""
        for conn_id, metadata in list(self.connection_metadata.items()):
            if metadata.get("collection_id") == collection_id:
                websocket = self.local_connections.get(conn_id)
                if websocket:
                    try:
                        await websocket.send_json(message)
                    except Exception as e:
                        logger.warning(f"Failed to send to connection {conn_id}: {e}")
                        await self.disconnect(conn_id)

    async def _handle_instance_message(self, message: dict) -> None:
        """Handle direct instance messages (e.g., admin commands)."""
        command = message.get("command")

        if command == "ping":
            # Health check
            logger.debug("Received ping command")

        elif command == "disconnect_user":
            # Disconnect all connections for a user
            user_id = message.get("user_id")
            if user_id:
                for conn_id, metadata in list(self.connection_metadata.items()):
                    if metadata.get("user_id") == user_id:
                        websocket = self.local_connections.get(conn_id)
                        if websocket:
                            await websocket.close(code=1000, reason="Admin disconnect")
                        await self.disconnect(conn_id)

        elif command == "stats":
            # Return instance statistics
            stats = {
                "instance_id": self.instance_id,
                "connections": len(self.local_connections),
                "users": len(set(m.get("user_id") for m in self.connection_metadata.values())),
                "uptime": time.time() - self.connection_metadata.get(
                    next(iter(self.connection_metadata), ""), {}
                ).get("connected_at", time.time()),
            }
            logger.info(f"Instance stats: {stats}")

    async def _heartbeat(self) -> None:
        """Keep instance registration alive and clean up dead connections."""
        try:
            while True:
                await asyncio.sleep(30)  # Run every 30 seconds

                # Refresh instance TTL
                if self.redis_client:
                    await self.redis_client.expire(
                        f"websocket:instance:{self.instance_id}",
                        60
                    )

                    # Update instance stats
                    stats = {
                        "connections": len(self.local_connections),
                        "users": len(set(m.get("user_id") for m in self.connection_metadata.values())),
                        "updated_at": time.time(),
                    }
                    await self.redis_client.hset(
                        "websocket:instances:stats",
                        self.instance_id,
                        json.dumps(stats)
                    )

                # Ping all local connections
                dead_connections = []
                for conn_id, websocket in list(self.local_connections.items()):
                    try:
                        await asyncio.wait_for(
                            websocket.send_json({"type": "ping"}),
                            timeout=5.0
                        )
                    except Exception:
                        dead_connections.append(conn_id)

                # Clean up dead connections
                for conn_id in dead_connections:
                    logger.info(f"Removing dead connection {conn_id}")
                    await self.disconnect(conn_id)

                logger.debug(
                    f"Heartbeat: {len(self.local_connections)} connections, "
                    f"{len(dead_connections)} removed"
                )

        except asyncio.CancelledError:
            logger.info("Heartbeat task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in heartbeat: {e}")

    async def _cleanup_dead_connections(self) -> None:
        """Periodically clean up dead connections from Redis registry."""
        try:
            while True:
                await asyncio.sleep(60)  # Run every minute

                if not self.redis_client:
                    continue

                # Get all registered connections
                connections = await self.redis_client.hgetall("websocket:connections")

                for conn_id, conn_data in connections.items():
                    try:
                        data = json.loads(conn_data)
                        instance_id = data.get("instance_id")

                        # Check if instance is still alive
                        instance_key = f"websocket:instance:{instance_id}"
                        if not await self.redis_client.exists(instance_key):
                            # Instance is dead, remove connection
                            logger.info(f"Removing connection {conn_id} from dead instance {instance_id}")

                            await self.redis_client.hdel("websocket:connections", conn_id)

                            # Remove from user set
                            user_id = data.get("user_id")
                            if user_id:
                                await self.redis_client.srem(f"websocket:user:{user_id}", conn_id)

                    except Exception as e:
                        logger.warning(f"Error cleaning up connection {conn_id}: {e}")

                logger.debug("Dead connection cleanup completed")

        except asyncio.CancelledError:
            logger.info("Cleanup task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")

    async def _send_operation_state(self, websocket: WebSocket, operation_id: str) -> None:
        """Send current operation state to newly connected client."""
        try:
            # Import here to avoid circular dependencies
            from packages.shared.database.database import AsyncSessionLocal
            from packages.shared.database.repositories.operation_repository import OperationRepository

            async with AsyncSessionLocal() as session:
                operation_repo = OperationRepository(session)
                operation = await operation_repo.get_by_uuid(operation_id)

                if operation:
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
                    logger.debug(f"Sent operation state for {operation_id}")

        except Exception as e:
            logger.error(f"Failed to send operation state: {e}")

    async def _get_hostname(self) -> str:
        """Get hostname for instance identification."""
        import socket
        try:
            return socket.gethostname()
        except Exception:
            return "unknown"

    async def get_stats(self) -> dict:
        """Get current manager statistics.

        Returns:
            Dictionary with connection and performance stats
        """
        stats = {
            "instance_id": self.instance_id,
            "local_connections": len(self.local_connections),
            "unique_users": len(set(m.get("user_id") for m in self.connection_metadata.values())),
            "operations": len(set(m.get("operation_id") for m in self.connection_metadata.values() if m.get("operation_id"))),
            "collections": len(set(m.get("collection_id") for m in self.connection_metadata.values() if m.get("collection_id"))),
        }

        # Get global stats from Redis
        if self.redis_client:
            try:
                # Count total connections across all instances
                total_connections = await self.redis_client.hlen("websocket:connections")

                # Count active instances
                instance_keys = await self.redis_client.keys("websocket:instance:*")
                active_instances = len(instance_keys)

                stats.update({
                    "total_connections": total_connections,
                    "active_instances": active_instances,
                })

            except Exception as e:
                logger.warning(f"Failed to get Redis stats: {e}")

        return stats


# Global instance - will be initialized by FastAPI on startup
scalable_ws_manager = ScalableWebSocketManager()
