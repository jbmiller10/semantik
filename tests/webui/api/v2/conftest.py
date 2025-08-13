"""Configuration for directory scan v2 API tests."""

import os
from unittest.mock import AsyncMock, patch

import pytest

# Mock Redis and WebSocket manager before any imports
mock_redis = AsyncMock()
mock_redis.ping = AsyncMock(return_value=True)
mock_redis.close = AsyncMock()
mock_redis.xadd = AsyncMock()
mock_redis.expire = AsyncMock()
mock_redis.xrange = AsyncMock(return_value=[])
mock_redis.xreadgroup = AsyncMock(return_value=[])
mock_redis.xgroup_create = AsyncMock()
mock_redis.xack = AsyncMock()
mock_redis.xgroup_delconsumer = AsyncMock()
mock_redis.delete = AsyncMock(return_value=1)
mock_redis.xinfo_groups = AsyncMock(return_value=[])
mock_redis.xgroup_destroy = AsyncMock()
mock_redis.get = AsyncMock(return_value=None)
mock_redis.set = AsyncMock()
mock_redis.setex = AsyncMock()
mock_redis.hget = AsyncMock(return_value=None)
mock_redis.hset = AsyncMock()
mock_redis.hdel = AsyncMock()
mock_redis.hgetall = AsyncMock(return_value={})
mock_redis.exists = AsyncMock(return_value=False)
mock_redis.ttl = AsyncMock(return_value=-1)
mock_redis.pipeline = AsyncMock()

# Apply patches before any webui imports
with patch("redis.asyncio.from_url", return_value=mock_redis):
    with patch("redis.from_url", return_value=mock_redis):
        # Force reimport of the WebSocket manager with mocked Redis
        import sys
        # Remove any already imported modules that use Redis
        modules_to_remove = [
            key for key in sys.modules.keys() 
            if 'websocket' in key or 'scalable_manager' in key
        ]
        for module in modules_to_remove:
            del sys.modules[module]


@pytest.fixture(autouse=True)
async def mock_websocket_manager():
    """Mock WebSocket manager for all tests."""
    mock_ws_manager = AsyncMock()
    mock_ws_manager.startup = AsyncMock()
    mock_ws_manager.shutdown = AsyncMock()
    mock_ws_manager.send_to_user = AsyncMock()
    mock_ws_manager.broadcast_to_collection = AsyncMock()
    mock_ws_manager.connect = AsyncMock(return_value="connection_id")
    mock_ws_manager.disconnect = AsyncMock()
    mock_ws_manager.redis_client = mock_redis
    mock_ws_manager.redis_url = "redis://localhost:6379/2"
    mock_ws_manager.instance_id = "test-instance"
    mock_ws_manager._startup_complete = True
    
    with (
        patch("packages.webui.websocket.scalable_manager.scalable_ws_manager", mock_ws_manager),
        patch("packages.webui.api.v2.directory_scan.ws_manager", mock_ws_manager),
        patch("packages.webui.services.directory_scan_service.ws_manager", mock_ws_manager),
    ):
        yield mock_ws_manager