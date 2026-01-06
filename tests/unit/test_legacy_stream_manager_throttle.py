"""Unit tests for LegacyStreamManager throttling helpers."""

from __future__ import annotations

import pytest

from webui.websocket.legacy_stream_manager import RedisStreamWebSocketManager


def test_resolve_stream_ttl_status_updates() -> None:
    assert RedisStreamWebSocketManager._resolve_stream_ttl("status_update", {"status": "completed"}) == 300
    assert RedisStreamWebSocketManager._resolve_stream_ttl("status_update", {"status": "cancelled"}) == 300
    assert RedisStreamWebSocketManager._resolve_stream_ttl("status_update", {"status": "failed"}) == 60
    assert RedisStreamWebSocketManager._resolve_stream_ttl("status_update", {"status": "processing"}) == 86400


@pytest.mark.asyncio()
async def test_should_send_progress_update_throttle() -> None:
    manager = RedisStreamWebSocketManager()
    message = {"type": "chunking_progress", "data": {"progress": 10}}

    assert await manager._should_send_progress_update("op-1", message) is True
    assert await manager._should_send_progress_update("op-1", message) is False


@pytest.mark.asyncio()
async def test_should_send_progress_update_non_progress() -> None:
    manager = RedisStreamWebSocketManager()
    message = {"type": "status_update", "data": {"status": "processing"}}

    assert await manager._should_send_progress_update("op-1", message) is True
