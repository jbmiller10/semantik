#!/usr/bin/env python3
"""
Unit tests for Redis Stream Manager.

Tests the RedisStreamManager class for publishing and managing
job progress updates via Redis streams.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from packages.webui.redis_streams import RedisStreamManager, get_redis_stream_manager, publish_job_update


class TestRedisStreamManager:
    """Test cases for RedisStreamManager."""

    @pytest.fixture()
    def redis_url(self):
        """Test Redis URL."""
        return "redis://localhost:6379/0"

    @pytest.fixture()
    def manager(self, redis_url):
        """Create a RedisStreamManager instance."""
        return RedisStreamManager(redis_url)

    def test_get_stream_key(self, manager):
        """Test stream key generation."""
        job_id = "test-job-123"
        expected_key = "job:stream:test-job-123"
        assert manager.get_stream_key(job_id) == expected_key

    @patch("redis.from_url")
    def test_sync_client_creation(self, mock_redis, manager):
        """Test synchronous Redis client creation."""
        mock_client = Mock()
        mock_redis.return_value = mock_client

        client = manager.sync_client
        mock_redis.assert_called_once_with(manager.redis_url, decode_responses=True)
        assert client == mock_client

        # Test that client is cached
        client2 = manager.sync_client
        assert client == client2
        mock_redis.assert_called_once()  # Still only called once

    @pytest.mark.asyncio
    @patch("redis.asyncio.from_url")
    async def test_async_client_creation(self, mock_redis, manager):
        """Test asynchronous Redis client creation."""
        mock_client = AsyncMock()
        mock_redis.return_value = mock_client

        client = await manager.async_client
        mock_redis.assert_called_once_with(manager.redis_url, decode_responses=True)
        assert client == mock_client

    @patch("redis.from_url")
    def test_publish_update(self, mock_redis, manager):
        """Test publishing updates to Redis stream."""
        mock_client = Mock()
        mock_redis.return_value = mock_client

        job_id = "test-job-123"
        update_type = "progress"
        data = {"processed": 10, "total": 100}

        manager.publish_update(job_id, update_type, data)

        # Verify xadd was called
        stream_key = manager.get_stream_key(job_id)
        mock_client.xadd.assert_called_once()
        call_args = mock_client.xadd.call_args

        assert call_args[0][0] == stream_key
        assert call_args[1]["maxlen"] == 1000

        # Verify message format
        message = call_args[0][1]
        assert message["type"] == update_type
        assert "timestamp" in message
        assert json.loads(message["data"]) == data

        # Verify expiration was set
        mock_client.expire.assert_called_once_with(stream_key, 86400)

    @patch("redis.from_url")
    def test_publish_update_error_handling(self, mock_redis, manager):
        """Test error handling in publish_update."""
        mock_client = Mock()
        mock_client.xadd.side_effect = Exception("Redis error")
        mock_redis.return_value = mock_client

        # Should not raise exception
        manager.publish_update("test-job", "progress", {"test": "data"})

    @pytest.mark.asyncio
    @patch("redis.asyncio.from_url")
    async def test_publish_update_async(self, mock_redis, manager):
        """Test asynchronous update publishing."""
        mock_client = AsyncMock()
        mock_client.xadd = AsyncMock()
        mock_client.expire = AsyncMock()
        mock_redis.return_value = mock_client

        job_id = "test-job-123"
        update_type = "file_completed"
        data = {"file": "test.txt", "chunks": 50}

        await manager.publish_update_async(job_id, update_type, data)

        # Verify async xadd was called
        mock_client.xadd.assert_called_once()

        # Verify async expire was called
        mock_client.expire.assert_called_once()

    @patch("redis.from_url")
    def test_cleanup_stream(self, mock_redis, manager):
        """Test stream cleanup."""
        mock_client = Mock()
        mock_redis.return_value = mock_client

        job_id = "test-job-123"
        manager.cleanup_stream(job_id)

        stream_key = manager.get_stream_key(job_id)
        mock_client.delete.assert_called_once_with(stream_key)

    @pytest.mark.asyncio
    @patch("redis.asyncio.from_url")
    async def test_cleanup_stream_async(self, mock_redis, manager):
        """Test asynchronous stream cleanup."""
        mock_client = AsyncMock()
        mock_client.delete = AsyncMock()
        mock_redis.return_value = mock_client

        job_id = "test-job-123"
        await manager.cleanup_stream_async(job_id)

        expected_key = manager.get_stream_key(job_id)
        mock_client.delete.assert_called_once_with(expected_key)

    @patch("redis.from_url")
    def test_close(self, mock_redis, manager):
        """Test closing Redis connections."""
        mock_client = Mock()
        mock_redis.return_value = mock_client

        # Force client creation
        _ = manager.sync_client

        manager.close()
        mock_client.close.assert_called_once()
        assert manager._sync_client is None

    @pytest.mark.asyncio
    @patch("redis.asyncio.from_url")
    async def test_close_async(self, mock_redis, manager):
        """Test closing async Redis connections."""
        mock_client = AsyncMock()
        mock_client.close = AsyncMock()
        mock_redis.return_value = mock_client

        # Force client creation
        _ = await manager.async_client

        await manager.close_async()
        mock_client.close.assert_called_once()
        assert manager._async_client is None


class TestConvenienceFunctions:
    """Test convenience functions for Celery tasks."""

    @patch("packages.webui.redis_streams.RedisStreamManager")
    def test_get_redis_stream_manager(self, mock_manager_class):
        """Test getting global Redis stream manager."""
        redis_url = "redis://localhost:6379/0"
        mock_instance = Mock()
        mock_manager_class.return_value = mock_instance

        # Clear global manager
        import packages.webui.redis_streams

        packages.webui.redis_streams._manager = None

        manager1 = get_redis_stream_manager(redis_url)
        assert manager1 == mock_instance
        mock_manager_class.assert_called_once_with(redis_url)

        # Test caching
        manager2 = get_redis_stream_manager(redis_url)
        assert manager1 == manager2
        mock_manager_class.assert_called_once()  # Still only called once

    @patch("packages.webui.redis_streams.get_redis_stream_manager")
    def test_publish_job_update(self, mock_get_manager):
        """Test convenience function for publishing updates."""
        mock_manager = Mock()
        mock_get_manager.return_value = mock_manager

        job_id = "test-job"
        update_type = "progress"
        data = {"test": "data"}
        redis_url = "redis://localhost:6379/0"

        publish_job_update(job_id, update_type, data, redis_url)

        mock_get_manager.assert_called_once_with(redis_url)
        mock_manager.publish_update.assert_called_once_with(job_id, update_type, data)
