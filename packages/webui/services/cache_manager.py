"""
Cache manager for query result caching with Redis.

This module provides caching functionality for expensive database queries,
helping to improve performance and reduce database load.
"""

import hashlib
import json
import logging
from collections.abc import Callable
from functools import wraps
from typing import Any

import redis.asyncio as aioredis

from shared.config import settings

logger = logging.getLogger(__name__)


class CacheManager:
    """Manage query result caching with Redis."""

    def __init__(self, redis_client: aioredis.Redis):
        """Initialize cache manager.

        Args:
            redis_client: Redis client for cache operations
        """
        self.redis = redis_client
        self.default_ttl = settings.CACHE_DEFAULT_TTL_SECONDS

        # Cache statistics
        self.hits = 0
        self.misses = 0

    def _generate_cache_key(self, prefix: str, params: dict[str, Any]) -> str:
        """Generate deterministic cache key from parameters.

        Args:
            prefix: Cache key prefix
            params: Parameters to include in key

        Returns:
            Cache key string
        """
        # Sort params for consistent key generation
        sorted_params = json.dumps(params, sort_keys=True)
        param_hash = hashlib.sha256(sorted_params.encode()).hexdigest()
        return f"cache:{prefix}:{param_hash}"

    async def get(self, key: str, deserializer: Callable = json.loads) -> Any | None:
        """Get value from cache.

        Args:
            key: Cache key
            deserializer: Function to deserialize cached value

        Returns:
            Cached value or None if not found
        """
        try:
            value = await self.redis.get(key)

            if value is None:
                self.misses += 1
                return None

            self.hits += 1
            return deserializer(value)
        except Exception as e:
            logger.warning("Cache get error for key %s: %s", key, e, exc_info=True)
            self.misses += 1
            return None

    async def set(self, key: str, value: Any, ttl: int | None = None, serializer: Callable = json.dumps) -> None:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            serializer: Function to serialize value
        """
        try:
            ttl = ttl or self.default_ttl
            serialized = serializer(value)
            await self.redis.setex(key, ttl, serialized)
        except Exception as e:
            logger.warning("Cache set error for key %s: %s", key, e, exc_info=True)

    async def delete(self, pattern: str) -> None:
        """Delete keys matching pattern.

        Args:
            pattern: Pattern to match keys (e.g., "cache:*:collection_id")
        """
        try:
            cursor = 0
            while True:
                cursor, keys = await self.redis.scan(cursor, match=pattern, count=100)

                if keys:
                    await self.redis.delete(*keys)

                if cursor == 0:
                    break
        except Exception as e:
            logger.warning("Cache delete error for pattern %s: %s", pattern, e, exc_info=True)

    async def invalidate_collection(self, collection_id: str) -> None:
        """Invalidate all cache entries for a collection.

        Args:
            collection_id: Collection ID to invalidate
        """
        await self.delete(f"cache:*:{collection_id}:*")
        await self.delete(f"cache:statistics:{collection_id}")
        await self.delete(f"cache:chunks:{collection_id}:*")

    def cache_result(self, prefix: str, ttl: int | None = None, key_params: list[str] | None = None) -> Callable:
        """Decorator for caching async function results.

        Args:
            prefix: Cache key prefix
            ttl: Time to live in seconds
            key_params: Parameters to include in cache key

        Returns:
            Decorator function
        """

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                # Build cache key from specified params
                cache_params = {k: kwargs.get(k) for k in key_params if k in kwargs} if key_params else kwargs

                cache_key = self._generate_cache_key(prefix, cache_params)

                # Try to get from cache
                cached = await self.get(cache_key)
                if cached is not None:
                    logger.debug(f"Cache hit for {prefix} with params {cache_params}")
                    return cached

                # Execute function
                result = await func(*args, **kwargs)

                # Cache result
                await self.set(cache_key, result, ttl)
                logger.debug(f"Cached result for {prefix} with params {cache_params}")

                return result

            # Store original function for testing
            wrapper._original = func  # type: ignore[attr-defined]
            return wrapper

        return decorator

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0

        return {"hits": self.hits, "misses": self.misses, "total_requests": total, "hit_rate": hit_rate}

    async def clear_all(self) -> None:
        """Clear all cache entries (use with caution)."""
        try:
            await self.delete("cache:*")
            logger.info("Cleared all cache entries")
        except Exception as e:
            logger.error("Failed to clear cache: %s", e, exc_info=True)

    async def warmup_collection_cache(self, collection_id: str, service: Any) -> None:
        """Pre-populate cache for a collection.

        Args:
            collection_id: Collection ID to warm up
            service: Service instance with methods to call
        """
        try:
            # Pre-fetch commonly accessed data
            if hasattr(service, "get_chunking_statistics"):
                await service.get_chunking_statistics(collection_id)

            logger.info(f"Warmed up cache for collection {collection_id}")
        except Exception as e:
            logger.warning("Cache warmup failed for collection %s: %s", collection_id, e, exc_info=True)


class QueryMonitor:
    """Monitor database query performance."""

    def __init__(self) -> None:
        """Initialize query monitor."""
        self.slow_query_threshold = 1.0  # 1 second
        self.query_times: list[dict[str, Any]] = []
        self.slow_queries: list[dict[str, Any]] = []

    async def __aenter__(self) -> "QueryMonitor":
        """Context manager entry."""
        import time

        self.start_time = time.time()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        import time

        execution_time = time.time() - self.start_time

        if execution_time > self.slow_query_threshold:
            logger.warning(f"Slow query detected: {execution_time:.2f}s", extra={"execution_time": execution_time})

    @staticmethod
    def monitor(query_name: str) -> Callable:
        """Decorator to monitor query performance.

        Args:
            query_name: Name of the query for logging

        Returns:
            Decorator function
        """

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                import time

                start_time = time.time()

                try:
                    return await func(*args, **kwargs)
                finally:
                    execution_time = time.time() - start_time

                    if execution_time > 1.0:  # Log slow queries
                        logger.warning(
                            f"Slow query '{query_name}': {execution_time:.2f}s",
                            extra={"query_name": query_name, "execution_time": execution_time, "params": kwargs},
                        )
                    else:
                        logger.debug(f"Query '{query_name}' executed in {execution_time:.2f}s")

            return wrapper

        return decorator
