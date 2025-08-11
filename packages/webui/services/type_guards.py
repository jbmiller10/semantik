"""
Type guards for runtime type checking of Redis clients.

This module provides type guards and runtime type checking utilities to ensure
that services receive the correct Redis client type, preventing type mismatches
that can cause silent failures and security issues.
"""

import logging
from typing import Any, TypeGuard

import redis
import redis.asyncio as aioredis
from packages.shared.utils.testing_utils import is_testing

logger = logging.getLogger(__name__)


def is_async_redis(client: Any) -> TypeGuard[aioredis.Redis]:
    """Type guard for async Redis client.

    This function checks if the provided client is an async Redis client,
    which is required for FastAPI services and other async contexts.

    Args:
        client: The client to check

    Returns:
        True if the client is an async Redis client, False otherwise

    Example:
        if is_async_redis(client):
            # client is guaranteed to be aioredis.Redis here
            await client.set("key", "value")
    """
    if is_testing():
        # In tests, accept both real and fake Redis
        try:
            import fakeredis.aioredis
            return isinstance(client, (aioredis.Redis, fakeredis.aioredis.FakeRedis))
        except ImportError:
            return isinstance(client, aioredis.Redis)
    return isinstance(client, aioredis.Redis)


def is_sync_redis(client: Any) -> TypeGuard[redis.Redis]:
    """Type guard for sync Redis client.

    This function checks if the provided client is a sync Redis client,
    which is required for Celery tasks and other synchronous contexts.

    Args:
        client: The client to check

    Returns:
        True if the client is a sync Redis client, False otherwise

    Example:
        if is_sync_redis(client):
            # client is guaranteed to be redis.Redis here
            client.set("key", "value")
    """
    if is_testing():
        # In tests, accept both real and fake Redis
        try:
            import fakeredis
            return isinstance(client, (redis.Redis, fakeredis.FakeRedis))
        except ImportError:
            return isinstance(client, redis.Redis)
    return isinstance(client, redis.Redis)


def ensure_async_redis(client: Any) -> aioredis.Redis:
    """Ensure client is async Redis, raise TypeError if not.

    This function validates that the provided client is an async Redis client
    and raises a descriptive error if not. This helps catch type mismatches
    early in the application lifecycle.

    Args:
        client: The client to validate

    Returns:
        The validated async Redis client

    Raises:
        TypeError: If the client is not an async Redis client

    Example:
        def __init__(self, redis_client: Any):
            self.redis = ensure_async_redis(redis_client)
            # Now self.redis is guaranteed to be aioredis.Redis
    """
    if not is_async_redis(client):
        error_msg = (
            f"Expected aioredis.Redis, got {type(client).__name__}. "
            "This service requires an async Redis client for non-blocking operations. "
            "Please use ServiceFactory.create_*_service() methods to ensure proper client types."
        )
        logger.error(error_msg)
        raise TypeError(error_msg)
    return client


def ensure_sync_redis(client: Any) -> redis.Redis:
    """Ensure client is sync Redis, raise TypeError if not.

    This function validates that the provided client is a sync Redis client
    and raises a descriptive error if not. This is critical for Celery tasks
    which cannot use async clients without causing event loop conflicts.

    Args:
        client: The client to validate

    Returns:
        The validated sync Redis client

    Raises:
        TypeError: If the client is not a sync Redis client

    Example:
        def process_task(redis_client: Any):
            redis = ensure_sync_redis(redis_client)
            # Now redis is guaranteed to be redis.Redis
            redis.set("key", "value")  # No await needed
    """
    if not is_sync_redis(client):
        error_msg = (
            f"Expected redis.Redis, got {type(client).__name__}. "
            "Celery tasks require a sync Redis client to avoid event loop conflicts. "
            "Do not use asyncio.run() in Celery tasks! "
            "Use ServiceFactory.create_celery_*_service() methods for proper client types."
        )
        logger.error(error_msg)
        raise TypeError(error_msg)
    return client


def validate_redis_response(response: Any, operation: str) -> Any:
    """Validate Redis operation response.

    This function checks if a Redis operation returned a valid response
    and logs appropriate warnings or errors if not.

    Args:
        response: The response from a Redis operation
        operation: Description of the operation for logging

    Returns:
        The response if valid, None otherwise

    Example:
        result = redis.get("key")
        validated = validate_redis_response(result, "get user session")
        if validated is None:
            # Handle missing data case
    """
    if response is None:
        logger.debug(f"Redis operation '{operation}' returned None (key might not exist)")
        return None

    if isinstance(response, Exception):
        logger.error(f"Redis operation '{operation}' failed: {response}")
        return None

    return response


def is_redis_available(client: Any) -> bool:
    """Check if a Redis client (sync or async) is available and connected.

    This function performs a basic availability check without actually
    pinging the server, just checking if the client object exists and
    appears to be properly initialized.

    Args:
        client: The Redis client to check

    Returns:
        True if the client appears available, False otherwise

    Note:
        This does not perform an actual connection test. Use the
        health_check methods on RedisManager for actual connectivity tests.
    """
    if client is None:
        return False

    # Check if it's either type of Redis client
    if not (is_async_redis(client) or is_sync_redis(client)):
        logger.warning(f"Unknown Redis client type: {type(client).__name__}")
        return False

    return True

