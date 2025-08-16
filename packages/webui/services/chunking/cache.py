"""
Chunking cache service.

Handles caching of chunking preview results and other cacheable operations.
"""

import hashlib
import json
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)


class ChunkingCache:
    """Service responsible for chunking result caching."""

    DEFAULT_TTL = 1800  # 30 minutes
    PREVIEW_CACHE_PREFIX = "chunking:preview"
    METRICS_CACHE_PREFIX = "chunking:metrics"

    def __init__(self, redis_client: aioredis.Redis | None = None):
        """
        Initialize the chunking cache service.

        Args:
            redis_client: Redis client for caching operations
        """
        self.redis = redis_client
        self.enabled = redis_client is not None

    async def get_cached_preview(
        self,
        content_hash: str,
        strategy: str,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """
        Retrieve cached preview result.

        Args:
            content_hash: Hash of the content
            strategy: Chunking strategy used
            config: Strategy configuration

        Returns:
            Cached preview data or None if not found
        """
        if not self.enabled:
            return None

        cache_key = self._generate_preview_key(content_hash, strategy, config)

        try:
            cached_data = await self.redis.get(cache_key)
            if cached_data:
                logger.debug("Cache hit for preview: %s", cache_key)
                return json.loads(cached_data)
        except Exception as e:
            logger.error("Error retrieving cached preview: %s", str(e))

        return None

    async def cache_preview(
        self,
        content_hash: str,
        strategy: str,
        config: dict[str, Any] | None,
        preview_data: dict[str, Any],
        ttl: int | None = None,
    ) -> str:
        """
        Cache preview result.

        Args:
            content_hash: Hash of the content
            strategy: Chunking strategy used
            config: Strategy configuration
            preview_data: Preview data to cache
            ttl: Time to live in seconds

        Returns:
            Cache key used
        """
        if not self.enabled:
            return ""

        cache_key = self._generate_preview_key(content_hash, strategy, config)
        ttl = ttl or self.DEFAULT_TTL

        try:
            # Add timestamp to cached data
            preview_data["cached_at"] = datetime.now(UTC).isoformat()
            preview_data["cache_key"] = cache_key

            await self.redis.setex(
                cache_key,
                ttl,
                json.dumps(preview_data),
            )
            logger.debug("Cached preview with key: %s (TTL: %ds)", cache_key, ttl)
            return cache_key
        except Exception as e:
            logger.error("Error caching preview: %s", str(e))
            return ""

    async def get_cached_by_id(self, cache_id: str) -> dict[str, Any] | None:
        """
        Retrieve cached data by cache ID.

        Args:
            cache_id: Cache identifier

        Returns:
            Cached data or None if not found
        """
        if not self.enabled:
            return None

        cache_key = f"{self.PREVIEW_CACHE_PREFIX}:id:{cache_id}"

        try:
            cached_data = await self.redis.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.error("Error retrieving cached data by ID: %s", str(e))

        return None

    async def cache_with_id(
        self,
        data: dict[str, Any],
        ttl: int | None = None,
    ) -> str:
        """
        Cache data with a generated ID.

        Args:
            data: Data to cache
            ttl: Time to live in seconds

        Returns:
            Cache ID for retrieval
        """
        if not self.enabled:
            return ""

        cache_id = str(uuid.uuid4())
        cache_key = f"{self.PREVIEW_CACHE_PREFIX}:id:{cache_id}"
        ttl = ttl or self.DEFAULT_TTL

        try:
            data["cache_id"] = cache_id
            data["cached_at"] = datetime.now(UTC).isoformat()

            await self.redis.setex(
                cache_key,
                ttl,
                json.dumps(data),
            )
            return cache_id
        except Exception as e:
            logger.error("Error caching with ID: %s", str(e))
            return ""

    async def clear_cache(self, pattern: str | None = None) -> int:
        """
        Clear cache entries matching pattern.

        Args:
            pattern: Pattern to match cache keys (default: all preview cache)

        Returns:
            Number of keys deleted
        """
        if not self.enabled:
            return 0

        pattern = pattern or f"{self.PREVIEW_CACHE_PREFIX}:*"

        try:
            keys = []
            async for key in self.redis.scan_iter(match=pattern):
                keys.append(key)

            if keys:
                deleted = await self.redis.delete(*keys)
                logger.info("Cleared %d cache entries matching pattern: %s", deleted, pattern)
                return deleted
        except Exception as e:
            logger.error("Error clearing cache: %s", str(e))

        return 0

    async def track_usage(
        self,
        user_id: int,
        strategy: str,
        cache_hit: bool,
    ) -> None:
        """
        Track cache usage metrics.

        Args:
            user_id: User identifier
            strategy: Strategy used
            cache_hit: Whether it was a cache hit
        """
        if not self.enabled:
            return

        try:
            # Track overall metrics
            metric_key = f"{self.METRICS_CACHE_PREFIX}:{strategy}"
            field = "hits" if cache_hit else "misses"
            await self.redis.hincrby(metric_key, field, 1)

            # Track user-specific metrics
            user_key = f"{self.METRICS_CACHE_PREFIX}:user:{user_id}"
            await self.redis.hincrby(user_key, strategy, 1)

            # Set expiry for metrics (7 days)
            await self.redis.expire(metric_key, 604800)
            await self.redis.expire(user_key, 604800)
        except Exception as e:
            logger.error("Error tracking cache usage: %s", str(e))

    async def get_usage_metrics(self, strategy: str | None = None) -> dict[str, Any]:
        """
        Get cache usage metrics.

        Args:
            strategy: Specific strategy to get metrics for (optional)

        Returns:
            Dictionary of metrics
        """
        if not self.enabled:
            return {}

        metrics = {}

        try:
            if strategy:
                metric_key = f"{self.METRICS_CACHE_PREFIX}:{strategy}"
                data = await self.redis.hgetall(metric_key)
                metrics[strategy] = {
                    "hits": int(data.get("hits", 0)),
                    "misses": int(data.get("misses", 0)),
                }
            else:
                # Get all strategy metrics
                async for key in self.redis.scan_iter(match=f"{self.METRICS_CACHE_PREFIX}:*"):
                    if ":user:" not in key:  # Skip user metrics
                        strategy_name = key.split(":")[-1]
                        data = await self.redis.hgetall(key)
                        metrics[strategy_name] = {
                            "hits": int(data.get("hits", 0)),
                            "misses": int(data.get("misses", 0)),
                        }
        except Exception as e:
            logger.error("Error getting usage metrics: %s", str(e))

        return metrics

    def _generate_preview_key(
        self,
        content_hash: str,
        strategy: str,
        config: dict[str, Any] | None,
    ) -> str:
        """Generate a deterministic cache key for preview."""
        config = config or {}
        config_str = json.dumps(config, sort_keys=True)
        combined = f"{content_hash}:{strategy}:{config_str}"
        key_hash = hashlib.sha256(combined.encode()).hexdigest()[:16]
        return f"{self.PREVIEW_CACHE_PREFIX}:{strategy}:{key_hash}"

    def generate_content_hash(self, content: str) -> str:
        """Generate hash for content."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
