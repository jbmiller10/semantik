"""Scalable WebSocket manager with Redis Pub/Sub for distributed messaging."""

import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any

import redis.asyncio as redis
from fastapi import WebSocket
from pydantic import BaseModel

from packages.shared.config import settings

from .registry import ConnectionRegistry
from .router import MessageRouter

logger = logging.getLogger(__name__)


class WebSocketConfig(BaseModel):
    """WebSocket configuration parameters."""

    max_connections_per_user: int = 10
    max_total_connections: int = 1000
    heartbeat_interval: int = 30
    message_rate_limit: int = 100  # per second
    max_message_size: int = 1048576  # 1MB
    reconnect_timeout: int = 60
    instance_id: str = ""

    def __init__(self, **data: Any) -> None:
        """Initialize with instance ID."""
        if not data.get("instance_id"):
            data["instance_id"] = f"webui-{uuid.uuid4().hex[:8]}"
        super().__init__(**data)


class ConnectionInfo(BaseModel):
    """Information about a WebSocket connection."""

    connection_id: str
    user_id: str
    channel: str
    instance_id: str
    connected_at: datetime
    last_heartbeat: datetime
    metadata: dict[str, Any] = {}


class ScalableWebSocketManager:
    """
    Horizontally scalable WebSocket manager using Redis Pub/Sub.

    Features:
    - Distributed message routing via Redis Pub/Sub
    - Connection registry for cross-instance awareness
    - Automatic failover and recovery
    - Rate limiting and backpressure handling
    - Graceful degradation without Redis
    """

    def __init__(self, config: WebSocketConfig | None = None) -> None:
        """Initialize the scalable WebSocket manager."""
        self.config = config or WebSocketConfig()
        self.redis_client: redis.Redis | None = None
        self.pubsub: redis.client.PubSub | None = None
        self.registry = ConnectionRegistry(instance_id=self.config.instance_id)
        self.router = MessageRouter()

        # Local connection tracking
        self.connections: dict[str, WebSocket] = {}
        self.connection_info: dict[str, ConnectionInfo] = {}
        self.user_connections: defaultdict[str, set[str]] = defaultdict(set)

        # Rate limiting
        self.message_counts: defaultdict[str, list[float]] = defaultdict(list)

        # Background tasks
        self.tasks: dict[str, asyncio.Task] = {}
        self._running = False
        self._startup_lock = asyncio.Lock()

    async def startup(self) -> None:
        """Initialize Redis connections and start background tasks."""
        async with self._startup_lock:
            if self._running:
                return

            try:
                # Connect to Redis
                self.redis_client = await redis.from_url(
                    settings.REDIS_URL,
                    decode_responses=True,
                    max_connections=100,
                    health_check_interval=30,
                    socket_keepalive=True,
                    socket_keepalive_options={
                        1: 1,  # TCP_KEEPIDLE
                        2: 2,  # TCP_KEEPINTVL
                        3: 3,  # TCP_KEEPCNT
                    },
                )

                # Test connection
                await self.redis_client.ping()

                # Initialize Pub/Sub
                self.pubsub = self.redis_client.pubsub()

                # Initialize registry
                await self.registry.initialize(self.redis_client)

                # Start background tasks
                self.tasks["heartbeat"] = asyncio.create_task(self._heartbeat_loop())
                self.tasks["pubsub"] = asyncio.create_task(self._pubsub_listener())
                self.tasks["cleanup"] = asyncio.create_task(self._cleanup_loop())

                self._running = True
                logger.info(f"WebSocket manager started with instance ID: {self.config.instance_id}")

            except Exception as e:
                logger.error(f"Failed to start WebSocket manager: {e}")
                # Allow degraded mode without Redis
                self._running = True

    async def shutdown(self) -> None:
        """Shutdown manager and cleanup resources."""
        self._running = False

        # Cancel background tasks
        for task_name, task in self.tasks.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                logger.debug(f"Cancelled task: {task_name}")

        # Close all connections
        for connection_id in list(self.connections.keys()):
            await self._close_connection(connection_id, reason="Server shutdown")

        # Cleanup registry
        if self.registry:
            await self.registry.cleanup()

        # Close Redis connections
        if self.pubsub:
            await self.pubsub.close()
        if self.redis_client:
            await self.redis_client.close()

        logger.info(f"WebSocket manager shutdown complete: {self.config.instance_id}")

    async def connect(
        self, websocket: WebSocket, user_id: str, channel: str, metadata: dict[str, Any] | None = None
    ) -> str | None:
        """
        Handle new WebSocket connection.

        Args:
            websocket: The WebSocket connection
            user_id: User identifier
            channel: Channel to subscribe to
            metadata: Optional connection metadata

        Returns:
            Connection ID if successful, None otherwise
        """
        # Check global connection limit
        if len(self.connections) >= self.config.max_total_connections:
            logger.warning(f"Global connection limit reached: {self.config.max_total_connections}")
            await websocket.close(code=1008, reason="Server capacity exceeded")
            return None

        # Check per-user connection limit
        if len(self.user_connections[user_id]) >= self.config.max_connections_per_user:
            logger.warning(f"User {user_id} exceeded connection limit: {self.config.max_connections_per_user}")
            await websocket.close(code=1008, reason="User connection limit exceeded")
            return None

        # Accept connection
        await websocket.accept()

        # Generate connection ID
        connection_id = f"{self.config.instance_id}:{uuid.uuid4().hex}"

        # Store connection
        self.connections[connection_id] = websocket
        self.connection_info[connection_id] = ConnectionInfo(
            connection_id=connection_id,
            user_id=user_id,
            channel=channel,
            instance_id=self.config.instance_id,
            connected_at=datetime.now(UTC),
            last_heartbeat=datetime.now(UTC),
            metadata=metadata or {},
        )
        self.user_connections[user_id].add(connection_id)

        # Register with registry
        if self.registry:
            await self.registry.register_connection(connection_id, user_id, channel)

        # Subscribe to channel
        if self.pubsub and self.redis_client:
            await self.pubsub.subscribe(channel)
            await self.pubsub.subscribe(f"user:{user_id}")
            await self.pubsub.subscribe(f"instance:{self.config.instance_id}")

        # Send welcome message
        await self._send_to_connection(
            connection_id,
            {
                "type": "connected",
                "connection_id": connection_id,
                "channel": channel,
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )

        logger.info(f"WebSocket connected: {connection_id} (user={user_id}, channel={channel})")
        return connection_id

    async def disconnect(self, connection_id: str) -> None:
        """
        Handle WebSocket disconnection.

        Args:
            connection_id: Connection identifier
        """
        if connection_id not in self.connections:
            return

        info = self.connection_info.get(connection_id)
        if info:
            # Unsubscribe from channels
            if self.pubsub:
                try:
                    await self.pubsub.unsubscribe(info.channel)
                    await self.pubsub.unsubscribe(f"user:{info.user_id}")
                except Exception as e:
                    logger.debug(f"Error unsubscribing: {e}")

            # Remove from tracking
            self.user_connections[info.user_id].discard(connection_id)
            if not self.user_connections[info.user_id]:
                del self.user_connections[info.user_id]

            # Unregister from registry
            if self.registry:
                await self.registry.unregister_connection(connection_id)

        # Close and remove connection
        await self._close_connection(connection_id, reason="Disconnected")

        logger.info(f"WebSocket disconnected: {connection_id}")

    async def send_message(
        self, channel: str, message: dict[str, Any], target_user: str | None = None, target_instance: str | None = None
    ) -> int:
        """
        Send message to channel subscribers.

        Args:
            channel: Target channel
            message: Message to send
            target_user: Optional specific user
            target_instance: Optional specific instance

        Returns:
            Number of messages sent
        """
        # Add metadata
        message["timestamp"] = datetime.now(UTC).isoformat()
        message["source_instance"] = self.config.instance_id

        # Route message
        if self.redis_client:
            # Distributed routing via Redis
            routing_key = self.router.get_routing_key(channel, target_user, target_instance)

            try:
                # Publish to Redis
                await self.redis_client.publish(routing_key, json.dumps(message))

                # Also send to local connections
                return await self._send_to_local_channel(channel, message)

            except Exception as e:
                logger.error(f"Failed to publish message: {e}")
                # Fallback to local delivery
                return await self._send_to_local_channel(channel, message)
        else:
            # Local delivery only (degraded mode)
            return await self._send_to_local_channel(channel, message)

    async def broadcast(self, message: dict[str, Any]) -> int:
        """
        Broadcast message to all connections.

        Args:
            message: Message to broadcast

        Returns:
            Number of messages sent
        """
        message["type"] = "broadcast"
        message["timestamp"] = datetime.now(UTC).isoformat()

        if self.redis_client:
            # Publish to system channel
            await self.redis_client.publish("system:broadcast", json.dumps(message))

        # Send to all local connections
        count = 0
        for connection_id in list(self.connections.keys()):
            if await self._send_to_connection(connection_id, message):
                count += 1

        return count

    async def handle_message(self, connection_id: str, message: dict[str, Any]) -> None:
        """
        Handle incoming message from client.

        Args:
            connection_id: Connection identifier
            message: Received message
        """
        info = self.connection_info.get(connection_id)
        if not info:
            return

        # Rate limiting
        if not await self._check_rate_limit(connection_id):
            await self._send_to_connection(
                connection_id,
                {"type": "error", "error": "Rate limit exceeded", "timestamp": datetime.now(UTC).isoformat()},
            )
            return

        # Update heartbeat
        info.last_heartbeat = datetime.now(UTC)

        # Handle different message types
        message_type = message.get("type")

        if message_type == "ping":
            # Respond with pong
            await self._send_to_connection(connection_id, {"type": "pong", "timestamp": datetime.now(UTC).isoformat()})

        elif message_type == "subscribe":
            # Subscribe to additional channel
            channel = message.get("channel")
            if channel and self.pubsub:
                await self.pubsub.subscribe(channel)
                info.metadata.setdefault("subscriptions", []).append(channel)

        elif message_type == "unsubscribe":
            # Unsubscribe from channel
            channel = message.get("channel")
            if channel and self.pubsub:
                await self.pubsub.unsubscribe(channel)
                if "subscriptions" in info.metadata:
                    info.metadata["subscriptions"].remove(channel)

        else:
            # Forward message to channel
            await self.send_message(info.channel, {"type": "message", "from": info.user_id, "data": message})

    async def _send_to_connection(self, connection_id: str, message: dict[str, Any]) -> bool:
        """
        Send message to specific connection.

        Args:
            connection_id: Connection identifier
            message: Message to send

        Returns:
            True if successful
        """
        websocket = self.connections.get(connection_id)
        if not websocket:
            return False

        try:
            await websocket.send_json(message)
            return True
        except Exception as e:
            logger.debug(f"Failed to send to {connection_id}: {e}")
            await self.disconnect(connection_id)
            return False

    async def _send_to_local_channel(self, channel: str, message: dict[str, Any]) -> int:
        """
        Send message to local channel subscribers.

        Args:
            channel: Target channel
            message: Message to send

        Returns:
            Number of messages sent
        """
        count = 0
        for connection_id, info in self.connection_info.items():
            if (
                info.channel == channel or channel in info.metadata.get("subscriptions", [])
            ) and await self._send_to_connection(connection_id, message):
                count += 1
        return count

    async def _close_connection(self, connection_id: str, reason: str = "Connection closed") -> None:
        """
        Close a WebSocket connection.

        Args:
            connection_id: Connection identifier
            reason: Close reason
        """
        websocket = self.connections.pop(connection_id, None)
        self.connection_info.pop(connection_id, None)

        if websocket:
            try:
                await websocket.close(code=1000, reason=reason)
            except Exception as e:
                logger.debug(f"Error closing connection {connection_id}: {e}")

    async def _check_rate_limit(self, connection_id: str) -> bool:
        """
        Check if connection is within rate limit.

        Args:
            connection_id: Connection identifier

        Returns:
            True if within limit
        """
        now = time.time()
        counts = self.message_counts[connection_id]

        # Remove old timestamps
        counts[:] = [t for t in counts if now - t < 1.0]

        # Check limit
        if len(counts) >= self.config.message_rate_limit:
            return False

        # Add current timestamp
        counts.append(now)
        return True

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats and check connection health."""
        while self._running:
            try:
                # Send heartbeat to registry
                if self.registry:
                    await self.registry.heartbeat()

                # Check connection health
                now = datetime.now(UTC)
                timeout = timedelta(seconds=self.config.heartbeat_interval * 2)

                for connection_id, info in list(self.connection_info.items()):
                    if now - info.last_heartbeat > timeout:
                        logger.info(f"Connection {connection_id} timed out")
                        await self.disconnect(connection_id)

                await asyncio.sleep(self.config.heartbeat_interval)

            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(5)

    async def _pubsub_listener(self) -> None:
        """Listen for Redis Pub/Sub messages."""
        if not self.pubsub:
            return

        while self._running:
            try:
                message = await self.pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)

                if message and message["type"] == "message":
                    # Parse message
                    try:
                        data = json.loads(message["data"])
                        channel = message["channel"]

                        # Route to local connections
                        await self._send_to_local_channel(channel, data)

                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid message format: {e}")

            except Exception as e:
                logger.error(f"Error in pubsub listener: {e}")
                await asyncio.sleep(1)

    async def _cleanup_loop(self) -> None:
        """Periodic cleanup of stale data."""
        while self._running:
            try:
                # Clean up old rate limit data
                now = time.time()
                for connection_id in list(self.message_counts.keys()):
                    if connection_id not in self.connections:
                        del self.message_counts[connection_id]
                    else:
                        # Remove old timestamps
                        counts = self.message_counts[connection_id]
                        counts[:] = [t for t in counts if now - t < 60]

                # Clean up registry
                if self.registry:
                    await self.registry.cleanup_stale_connections()

                await asyncio.sleep(60)  # Run every minute

            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(60)


# Global instance
scalable_ws_manager = ScalableWebSocketManager()
