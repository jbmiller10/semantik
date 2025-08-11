"""
Redis client manager for handling both async and sync Redis clients.

This module provides a centralized manager for Redis connections that ensures
proper separation between async and sync clients, preventing type mismatches
that can cause authentication bypasses and event loop conflicts.
"""

import logging
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field

import redis
import redis.asyncio as aioredis
from redis.backoff import ExponentialBackoff
from redis.exceptions import BusyLoadingError, ConnectionError, TimeoutError
from redis.retry import Retry

logger = logging.getLogger(__name__)


@dataclass
class RedisConfig:
    """Configuration for Redis connections."""

    url: str
    max_connections: int = 50
    decode_responses: bool = True
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    retry_on_timeout: bool = True
    retry_on_error: list = field(default_factory=lambda: [ConnectionError, TimeoutError, BusyLoadingError])
    health_check_interval: int = 30
    socket_keepalive: bool = True


class RedisManager:
    """Manages both sync and async Redis clients with proper typing.

    This class provides a single point of management for Redis connections,
    ensuring that services receive the correct client type (async or sync)
    based on their execution context. This prevents type mismatches that
    can cause silent failures in Redis operations.

    Attributes:
        config: Redis configuration settings
        _async_pool: Connection pool for async Redis clients
        _sync_pool: Connection pool for sync Redis clients
    """

    def __init__(self, config: RedisConfig):
        """Initialize the Redis manager with configuration.

        Args:
            config: Redis configuration settings
        """
        self.config = config
        self._async_pool: aioredis.ConnectionPool | None = None
        self._sync_pool: redis.ConnectionPool | None = None
        self._sync_client: redis.Redis | None = None
        self._async_client: aioredis.Redis | None = None

    @property
    def sync_client(self) -> redis.Redis:
        """Get synchronous Redis client for Celery tasks.

        This property returns a sync Redis client that can be used in
        synchronous contexts like Celery tasks. The client uses connection
        pooling for efficiency and includes retry logic for resilience.

        Returns:
            Synchronous Redis client

        Note:
            This client should NEVER be used with asyncio.run() in Celery tasks
            as it will cause event loop conflicts.
        """
        if not self._sync_client:
            if not self._sync_pool:
                # Create retry strategy
                retry = Retry(
                    ExponentialBackoff(),
                    retries=3,
                )

                self._sync_pool = redis.ConnectionPool.from_url(
                    self.config.url,
                    max_connections=self.config.max_connections,
                    decode_responses=self.config.decode_responses,
                    socket_timeout=self.config.socket_timeout,
                    socket_connect_timeout=self.config.socket_connect_timeout,
                    retry_on_timeout=self.config.retry_on_timeout,
                    retry_on_error=self.config.retry_on_error,
                    retry=retry,
                    health_check_interval=self.config.health_check_interval,
                    socket_keepalive=self.config.socket_keepalive,
                )
                logger.info(
                    "Created sync Redis connection pool with %d max connections",
                    self.config.max_connections,
                )

            self._sync_client = redis.Redis(connection_pool=self._sync_pool)

        return self._sync_client

    async def async_client(self) -> aioredis.Redis:
        """Get asynchronous Redis client for FastAPI services.

        This method returns an async Redis client that can be used in
        asynchronous contexts like FastAPI endpoints and services.
        The client uses connection pooling for efficiency.

        Returns:
            Asynchronous Redis client

        Note:
            This client should ONLY be used in async contexts with proper
            await syntax.
        """
        if not self._async_client:
            if not self._async_pool:
                # Create retry strategy
                retry = Retry(
                    ExponentialBackoff(),
                    retries=3,
                )

                self._async_pool = aioredis.ConnectionPool.from_url(
                    self.config.url,
                    max_connections=self.config.max_connections,
                    decode_responses=self.config.decode_responses,
                    socket_timeout=self.config.socket_timeout,
                    socket_connect_timeout=self.config.socket_connect_timeout,
                    retry_on_timeout=self.config.retry_on_timeout,
                    retry_on_error=self.config.retry_on_error,
                    retry=retry,
                    health_check_interval=self.config.health_check_interval,
                    socket_keepalive=self.config.socket_keepalive,
                )
                logger.info(
                    "Created async Redis connection pool with %d max connections",
                    self.config.max_connections,
                )

            self._async_client = aioredis.Redis(connection_pool=self._async_pool)

        return self._async_client

    @asynccontextmanager
    async def async_transaction(self):
        """Context manager for async Redis transactions.

        This provides a transactional context for async Redis operations,
        ensuring atomicity of multiple operations.

        Yields:
            Async Redis pipeline for transactional operations

        Example:
            async with redis_manager.async_transaction() as pipe:
                pipe.set("key1", "value1")
                pipe.set("key2", "value2")
                # Both operations committed atomically
        """
        client = await self.async_client()
        async with client.pipeline(transaction=True) as pipe:
            try:
                yield pipe
                await pipe.execute()
            except Exception as e:
                logger.error("Async Redis transaction failed: %s", e)
                raise

    @contextmanager
    def sync_transaction(self):
        """Context manager for sync Redis transactions.

        This provides a transactional context for sync Redis operations,
        ensuring atomicity of multiple operations.

        Yields:
            Sync Redis pipeline for transactional operations

        Example:
            with redis_manager.sync_transaction() as pipe:
                pipe.set("key1", "value1")
                pipe.set("key2", "value2")
                # Both operations committed atomically
        """
        client = self.sync_client
        with client.pipeline(transaction=True) as pipe:
            try:
                yield pipe
                pipe.execute()
            except Exception as e:
                logger.error("Sync Redis transaction failed: %s", e)
                raise

    async def close_async(self):
        """Close async connections gracefully.

        This method should be called during application shutdown to
        properly close all async Redis connections.
        """
        if self._async_client:
            await self._async_client.close()
            self._async_client = None

        if self._async_pool:
            await self._async_pool.disconnect()
            self._async_pool = None
            logger.info("Closed async Redis connection pool")

    def close_sync(self):
        """Close sync connections gracefully.

        This method should be called during application shutdown to
        properly close all sync Redis connections.
        """
        if self._sync_client:
            self._sync_client.close()
            self._sync_client = None

        if self._sync_pool:
            self._sync_pool.disconnect()
            self._sync_pool = None
            logger.info("Closed sync Redis connection pool")

    async def health_check_async(self) -> bool:
        """Check if async Redis connection is healthy.

        Returns:
            True if connection is healthy, False otherwise
        """
        try:
            client = await self.async_client()
            await client.ping()
            return True
        except Exception as e:
            logger.warning("Async Redis health check failed: %s", e)
            return False

    def health_check_sync(self) -> bool:
        """Check if sync Redis connection is healthy.

        Returns:
            True if connection is healthy, False otherwise
        """
        try:
            self.sync_client.ping()
            return True
        except Exception as e:
            logger.warning("Sync Redis health check failed: %s", e)
            return False
