"""Connection registry for tracking WebSocket connections across instances."""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import redis.asyncio as redis
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class InstanceInfo(BaseModel):
    """Information about a WebUI instance."""

    instance_id: str
    hostname: str
    started_at: datetime
    last_heartbeat: datetime
    connection_count: int
    metadata: dict[str, Any] = {}


class ConnectionRecord(BaseModel):
    """Record of a WebSocket connection."""

    connection_id: str
    user_id: str
    channel: str
    instance_id: str
    connected_at: datetime
    last_seen: datetime
    metadata: dict[str, Any] = {}


class ConnectionRegistry:
    """
    Distributed registry for WebSocket connections.

    Tracks all active connections across instances for:
    - Connection routing
    - Load balancing
    - Failover handling
    - Monitoring
    """

    def __init__(self, instance_id: str) -> None:
        """
        Initialize the connection registry.

        Args:
            instance_id: Unique identifier for this instance
        """
        self.instance_id = instance_id
        self.redis_client: redis.Redis | None = None
        self.hostname = self._get_hostname()
        self.started_at = datetime.now(UTC)

        # Registry keys
        self.instance_key = f"ws:instance:{instance_id}"
        self.instances_set = "ws:instances"
        self.connections_hash = "ws:connections"
        self.user_index = "ws:user_index"
        self.channel_index = "ws:channel_index"

        # Heartbeat settings
        self.heartbeat_interval = 30  # seconds
        self.heartbeat_timeout = 90  # seconds

        self._initialized = False

    def _get_hostname(self) -> str:
        """Get the hostname of this instance."""
        import socket

        return socket.gethostname()

    async def initialize(self, redis_client: redis.Redis) -> None:
        """
        Initialize the registry with Redis connection.

        Args:
            redis_client: Redis client for distributed state
        """
        if self._initialized:
            return

        self.redis_client = redis_client

        # Register this instance
        await self._register_instance()

        self._initialized = True
        logger.info(f"Connection registry initialized for instance: {self.instance_id}")

    async def cleanup(self) -> None:
        """Cleanup registry on shutdown."""
        if not self.redis_client:
            return

        try:
            # Remove all connections for this instance
            connections = await self._get_instance_connections()
            for connection_id in connections:
                await self.unregister_connection(connection_id)

            # Remove instance registration
            await self.redis_client.srem(self.instances_set, self.instance_id)
            await self.redis_client.delete(self.instance_key)

            logger.info(f"Connection registry cleaned up for instance: {self.instance_id}")

        except Exception as e:
            logger.error(f"Error cleaning up registry: {e}")

    async def register_connection(
        self, connection_id: str, user_id: str, channel: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """
        Register a new connection.

        Args:
            connection_id: Unique connection identifier
            user_id: User identifier
            channel: Channel subscribed to
            metadata: Optional connection metadata
        """
        if not self.redis_client:
            return

        record = ConnectionRecord(
            connection_id=connection_id,
            user_id=user_id,
            channel=channel,
            instance_id=self.instance_id,
            connected_at=datetime.now(UTC),
            last_seen=datetime.now(UTC),
            metadata=metadata or {},
        )

        try:
            # Store connection record
            await self.redis_client.hset(self.connections_hash, connection_id, record.model_dump_json())

            # Update indexes
            await self.redis_client.sadd(f"{self.user_index}:{user_id}", connection_id)
            await self.redis_client.sadd(f"{self.channel_index}:{channel}", connection_id)
            await self.redis_client.sadd(f"{self.instance_key}:connections", connection_id)

            # Update instance connection count
            await self._update_instance_stats()

            logger.debug(f"Registered connection: {connection_id}")

        except Exception as e:
            logger.error(f"Failed to register connection {connection_id}: {e}")

    async def unregister_connection(self, connection_id: str) -> None:
        """
        Unregister a connection.

        Args:
            connection_id: Connection identifier to remove
        """
        if not self.redis_client:
            return

        try:
            # Get connection record
            record_json = await self.redis_client.hget(self.connections_hash, connection_id)
            if not record_json:
                return

            record = ConnectionRecord.model_validate_json(record_json)

            # Remove from indexes
            await self.redis_client.srem(f"{self.user_index}:{record.user_id}", connection_id)
            await self.redis_client.srem(f"{self.channel_index}:{record.channel}", connection_id)
            await self.redis_client.srem(f"{self.instance_key}:connections", connection_id)

            # Remove connection record
            await self.redis_client.hdel(self.connections_hash, connection_id)

            # Update instance connection count
            await self._update_instance_stats()

            logger.debug(f"Unregistered connection: {connection_id}")

        except Exception as e:
            logger.error(f"Failed to unregister connection {connection_id}: {e}")

    async def get_connection(self, connection_id: str) -> ConnectionRecord | None:
        """
        Get connection information.

        Args:
            connection_id: Connection identifier

        Returns:
            Connection record if found
        """
        if not self.redis_client:
            return None

        try:
            record_json = await self.redis_client.hget(self.connections_hash, connection_id)
            if record_json:
                return ConnectionRecord.model_validate_json(record_json)
        except Exception as e:
            logger.error(f"Failed to get connection {connection_id}: {e}")

        return None

    async def get_user_connections(self, user_id: str) -> list[ConnectionRecord]:
        """
        Get all connections for a user.

        Args:
            user_id: User identifier

        Returns:
            List of connection records
        """
        if not self.redis_client:
            return []

        connections = []
        try:
            connection_ids = await self.redis_client.smembers(f"{self.user_index}:{user_id}")
            for connection_id in connection_ids:
                record = await self.get_connection(connection_id)
                if record:
                    connections.append(record)
        except Exception as e:
            logger.error(f"Failed to get user connections for {user_id}: {e}")

        return connections

    async def get_channel_connections(self, channel: str) -> list[ConnectionRecord]:
        """
        Get all connections for a channel.

        Args:
            channel: Channel name

        Returns:
            List of connection records
        """
        if not self.redis_client:
            return []

        connections = []
        try:
            connection_ids = await self.redis_client.smembers(f"{self.channel_index}:{channel}")
            for connection_id in connection_ids:
                record = await self.get_connection(connection_id)
                if record:
                    connections.append(record)
        except Exception as e:
            logger.error(f"Failed to get channel connections for {channel}: {e}")

        return connections

    async def get_instance_connections(self, instance_id: str | None = None) -> list[ConnectionRecord]:
        """
        Get all connections for an instance.

        Args:
            instance_id: Instance identifier (default: current instance)

        Returns:
            List of connection records
        """
        if not self.redis_client:
            return []

        instance_id = instance_id or self.instance_id
        connections = []

        try:
            connection_ids = await self._get_instance_connections(instance_id)
            for connection_id in connection_ids:
                record = await self.get_connection(connection_id)
                if record:
                    connections.append(record)
        except Exception as e:
            logger.error(f"Failed to get instance connections for {instance_id}: {e}")

        return connections

    async def get_active_instances(self) -> list[InstanceInfo]:
        """
        Get all active WebUI instances.

        Returns:
            List of active instance information
        """
        if not self.redis_client:
            return []

        instances = []
        try:
            instance_ids = await self.redis_client.smembers(self.instances_set)
            for instance_id in instance_ids:
                info = await self._get_instance_info(instance_id)
                if info:
                    instances.append(info)
        except Exception as e:
            logger.error(f"Failed to get active instances: {e}")

        return instances

    async def heartbeat(self) -> None:
        """Send heartbeat to maintain instance registration."""
        if not self.redis_client:
            return

        try:
            # Update instance heartbeat
            info = InstanceInfo(
                instance_id=self.instance_id,
                hostname=self.hostname,
                started_at=self.started_at,
                last_heartbeat=datetime.now(UTC),
                connection_count=await self._count_instance_connections(),
            )

            await self.redis_client.setex(self.instance_key, self.heartbeat_timeout, info.model_dump_json())

            # Ensure instance is in set
            await self.redis_client.sadd(self.instances_set, self.instance_id)

        except Exception as e:
            logger.error(f"Failed to send heartbeat: {e}")

    async def cleanup_stale_connections(self) -> int:
        """
        Remove connections from failed instances.

        Returns:
            Number of connections cleaned up
        """
        if not self.redis_client:
            return 0

        cleaned = 0
        try:
            # Check all registered instances
            instance_ids = await self.redis_client.smembers(self.instances_set)

            for instance_id in instance_ids:
                # Check if instance is alive
                info = await self._get_instance_info(instance_id)
                if not info:
                    # Instance is dead, clean up its connections
                    connections = await self._get_instance_connections(instance_id)
                    for connection_id in connections:
                        await self.unregister_connection(connection_id)
                        cleaned += 1

                    # Remove instance from set
                    await self.redis_client.srem(self.instances_set, instance_id)
                    logger.info(f"Cleaned up {len(connections)} connections from dead instance: {instance_id}")

                elif datetime.now(UTC) - info.last_heartbeat > timedelta(seconds=self.heartbeat_timeout):
                    # Instance heartbeat expired
                    connections = await self._get_instance_connections(instance_id)
                    for connection_id in connections:
                        await self.unregister_connection(connection_id)
                        cleaned += 1

                    # Remove instance
                    await self.redis_client.srem(self.instances_set, instance_id)
                    await self.redis_client.delete(f"ws:instance:{instance_id}")
                    logger.info(f"Cleaned up {len(connections)} connections from expired instance: {instance_id}")

        except Exception as e:
            logger.error(f"Failed to cleanup stale connections: {e}")

        return cleaned

    async def get_stats(self) -> dict[str, Any]:
        """
        Get registry statistics.

        Returns:
            Dictionary of statistics
        """
        if not self.redis_client:
            return {}

        try:
            # Count connections
            total_connections = await self.redis_client.hlen(self.connections_hash)

            # Count instances
            instance_count = await self.redis_client.scard(self.instances_set)

            # Get instance details
            instances = await self.get_active_instances()

            # Count unique users
            user_keys = await self.redis_client.keys(f"{self.user_index}:*")
            unique_users = len(user_keys)

            # Count unique channels
            channel_keys = await self.redis_client.keys(f"{self.channel_index}:*")
            unique_channels = len(channel_keys)

            return {
                "total_connections": total_connections,
                "instance_count": instance_count,
                "unique_users": unique_users,
                "unique_channels": unique_channels,
                "instances": [
                    {
                        "instance_id": inst.instance_id,
                        "hostname": inst.hostname,
                        "connection_count": inst.connection_count,
                        "uptime": (datetime.now(UTC) - inst.started_at).total_seconds(),
                    }
                    for inst in instances
                ],
            }

        except Exception as e:
            logger.error(f"Failed to get registry stats: {e}")
            return {}

    async def _register_instance(self) -> None:
        """Register this instance in the registry."""
        try:
            # Create instance info
            info = InstanceInfo(
                instance_id=self.instance_id,
                hostname=self.hostname,
                started_at=self.started_at,
                last_heartbeat=datetime.now(UTC),
                connection_count=0,
            )

            # Store instance info
            await self.redis_client.setex(self.instance_key, self.heartbeat_timeout, info.model_dump_json())

            # Add to instances set
            await self.redis_client.sadd(self.instances_set, self.instance_id)

            logger.info(f"Registered instance: {self.instance_id} on {self.hostname}")

        except Exception as e:
            logger.error(f"Failed to register instance: {e}")

    async def _get_instance_info(self, instance_id: str) -> InstanceInfo | None:
        """Get information about an instance."""
        try:
            info_json = await self.redis_client.get(f"ws:instance:{instance_id}")
            if info_json:
                return InstanceInfo.model_validate_json(info_json)
        except Exception as e:
            logger.error(f"Failed to get instance info for {instance_id}: {e}")
        return None

    async def _get_instance_connections(self, instance_id: str | None = None) -> set[str]:
        """Get connection IDs for an instance."""
        instance_id = instance_id or self.instance_id
        try:
            return await self.redis_client.smembers(f"ws:instance:{instance_id}:connections")
        except Exception:
            return set()

    async def _count_instance_connections(self) -> int:
        """Count connections for this instance."""
        try:
            return await self.redis_client.scard(f"{self.instance_key}:connections")
        except Exception:
            return 0

    async def _update_instance_stats(self) -> None:
        """Update instance statistics."""
        await self.heartbeat()
