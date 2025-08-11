"""Utilities for testing support."""

import os


def is_testing() -> bool:
    """Check if running in test environment."""
    return os.getenv("TESTING", "false").lower() in ("true", "1", "yes")


def is_redis_mock_allowed() -> bool:
    """Check if Redis mocks are allowed in current context."""
    return is_testing()


def validate_redis_client(client, client_type="async") -> bool:
    """Validate Redis client type with test support.

    Args:
        client: Redis client to validate
        client_type: "async" or "sync"

    Returns:
        True if valid, False otherwise
    """
    if is_testing():
        # In tests, accept both real and fake Redis
        if client_type == "async":
            import redis.asyncio as aioredis

            try:
                import fakeredis.aioredis

                return isinstance(client, aioredis.Redis | fakeredis.aioredis.FakeRedis)
            except ImportError:
                return isinstance(client, aioredis.Redis)
        else:
            import redis

            try:
                import fakeredis

                return isinstance(client, redis.Redis | fakeredis.FakeRedis)
            except ImportError:
                return isinstance(client, redis.Redis)
    else:
        # In production, only accept real Redis
        if client_type == "async":
            import redis.asyncio as aioredis

            return isinstance(client, aioredis.Redis)
        else:
            import redis

            return isinstance(client, redis.Redis)
