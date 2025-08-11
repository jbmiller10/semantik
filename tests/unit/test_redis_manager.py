#!/usr/bin/env python3

"""
Unit tests for RedisManager and type guards.

This module tests the Redis client management functionality.
"""

import pytest
import redis
import redis.asyncio as aioredis

from packages.webui.services.redis_manager import RedisConfig, RedisManager
from packages.webui.services.type_guards import (
    ensure_async_redis,
    ensure_sync_redis,
    is_async_redis,
    is_redis_available,
    is_sync_redis,
)


class TestRedisManager:
    """Tests for RedisManager."""

    @pytest.fixture()
    def redis_config(self) -> RedisConfig:
        """Create test Redis configuration."""
        return RedisConfig(
            url="redis://localhost:6379/0",
            max_connections=10,
            decode_responses=True,
        )

    @pytest.fixture()
    def redis_manager(self, redis_config: RedisConfig) -> RedisManager:
        """Create RedisManager instance."""
        return RedisManager(redis_config)

    def test_sync_client_creation(self, redis_manager: RedisManager) -> None:
        """Test that sync Redis client is created correctly."""
        client = redis_manager.sync_client
        
        # Verify it's a sync Redis client
        assert isinstance(client, redis.Redis)
        assert not isinstance(client, aioredis.Redis)
        
        # Verify connection pool is created
        assert redis_manager._sync_pool is not None
        
        # Multiple calls should return the same client
        client2 = redis_manager.sync_client
        assert client is client2

    async def test_async_client_creation(self, redis_manager: RedisManager) -> None:
        """Test that async Redis client is created correctly."""
        client = await redis_manager.async_client()
        
        # Verify it's an async Redis client
        assert isinstance(client, aioredis.Redis)
        
        # Verify connection pool is created
        assert redis_manager._async_pool is not None
        
        # Multiple calls should return the same client
        client2 = await redis_manager.async_client()
        assert client is client2

    async def test_async_transaction(self, redis_manager: RedisManager) -> None:
        """Test async transaction context manager."""
        # This test just verifies the context manager works without errors
        # In a real test environment with Redis running, we'd test actual transactions
        try:
            async with redis_manager.async_transaction() as pipe:
                assert pipe is not None
        except (ConnectionError, OSError):
            # Redis might not be available in test environment
            pytest.skip("Redis not available")

    def test_sync_transaction(self, redis_manager: RedisManager) -> None:
        """Test sync transaction context manager."""
        # This test just verifies the context manager works without errors
        try:
            with redis_manager.sync_transaction() as pipe:
                assert pipe is not None
        except (ConnectionError, OSError):
            # Redis might not be available in test environment
            pytest.skip("Redis not available")


class TestTypeGuards:
    """Tests for Redis type guard functions."""

    def test_is_sync_redis(self) -> None:
        """Test sync Redis type guard."""
        sync_client = redis.Redis()
        async_client = aioredis.Redis()
        
        assert is_sync_redis(sync_client) is True
        assert is_sync_redis(async_client) is False
        assert is_sync_redis(None) is False
        assert is_sync_redis("not a redis client") is False

    def test_is_async_redis(self) -> None:
        """Test async Redis type guard."""
        sync_client = redis.Redis()
        async_client = aioredis.Redis()
        
        assert is_async_redis(async_client) is True
        assert is_async_redis(sync_client) is False
        assert is_async_redis(None) is False
        assert is_async_redis("not a redis client") is False

    def test_ensure_sync_redis(self) -> None:
        """Test sync Redis type enforcement."""
        sync_client = redis.Redis()
        async_client = aioredis.Redis()
        
        # Should work with sync client
        result = ensure_sync_redis(sync_client)
        assert result is sync_client
        
        # Should raise TypeError with async client
        with pytest.raises(TypeError, match="Expected redis.Redis"):
            ensure_sync_redis(async_client)
        
        # Should raise TypeError with None
        with pytest.raises(TypeError, match="Expected redis.Redis"):
            ensure_sync_redis(None)

    def test_ensure_async_redis(self) -> None:
        """Test async Redis type enforcement."""
        sync_client = redis.Redis()
        async_client = aioredis.Redis()
        
        # Should work with async client
        result = ensure_async_redis(async_client)
        assert result is async_client
        
        # Should raise TypeError with sync client
        with pytest.raises(TypeError, match="Expected aioredis.Redis"):
            ensure_async_redis(sync_client)
        
        # Should raise TypeError with None
        with pytest.raises(TypeError, match="Expected aioredis.Redis"):
            ensure_async_redis(None)

    def test_is_redis_available(self) -> None:
        """Test Redis availability check."""
        sync_client = redis.Redis()
        async_client = aioredis.Redis()
        
        assert is_redis_available(sync_client) is True
        assert is_redis_available(async_client) is True
        assert is_redis_available(None) is False
        assert is_redis_available("not a redis client") is False