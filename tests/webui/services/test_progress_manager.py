"""Unit tests for the ProgressUpdateManager abstraction."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from packages.webui.services.progress_manager import (
    ProgressPayload,
    ProgressSendResult,
    ProgressUpdateManager,
)


def _payload(**overrides: object) -> ProgressPayload:
    defaults: dict[str, object] = {
        "operation_id": "op-123",
        "correlation_id": "corr-456",
        "progress": 50,
        "message": "processing",
    }
    defaults.update(overrides)
    return ProgressPayload(**defaults)


def test_send_sync_update_with_hash_and_ttl() -> None:
    """Sync updates should publish to stream, set TTL, and update the hash."""

    sync_client = MagicMock()
    manager = ProgressUpdateManager(
        sync_redis=sync_client,
        default_stream_template="stream:{operation_id}",
        default_ttl=120,
        default_maxlen=200,
    )

    payload = _payload()
    result = manager.send_sync_update(payload, hash_key_template="hash:{operation_id}")

    assert result is ProgressSendResult.SENT
    sync_client.xadd.assert_called_once()
    stream_key, fields = sync_client.xadd.call_args[0]
    assert stream_key == "stream:op-123"
    assert fields["operation_id"] == "op-123"
    assert fields["progress"] == "50"
    assert sync_client.xadd.call_args.kwargs["maxlen"] == 200

    sync_client.expire.assert_called_once_with("stream:op-123", 120)
    sync_client.hset.assert_called_once()
    hash_key = sync_client.hset.call_args[0][0]
    mapping = sync_client.hset.call_args.kwargs["mapping"]
    assert hash_key == "hash:op-123"
    assert mapping["progress"] == "50"


def test_send_sync_update_without_ttl_or_maxlen() -> None:
    """Passing explicit overrides should suppress expire/maxlen logic."""

    sync_client = MagicMock()
    manager = ProgressUpdateManager(
        sync_redis=sync_client,
        default_stream_template="stream:{operation_id}",
        default_ttl=None,
        default_maxlen=50,
    )

    payload = _payload()
    result = manager.send_sync_update(payload, ttl=None, maxlen=0)

    assert result is ProgressSendResult.SENT
    _, kwargs = sync_client.xadd.call_args
    assert "maxlen" not in kwargs
    sync_client.expire.assert_not_called()


def test_send_sync_update_respects_throttle() -> None:
    """Second call within the throttle window should skip publishing."""

    sync_client = MagicMock()
    manager = ProgressUpdateManager(
        sync_redis=sync_client,
        default_stream_template="stream:{operation_id}",
        default_ttl=None,
        sync_throttle_interval=60,
    )

    payload = _payload()
    first = manager.send_sync_update(payload, use_throttle=True)
    second = manager.send_sync_update(payload, use_throttle=True)

    assert first is ProgressSendResult.SENT
    assert second is ProgressSendResult.SKIPPED
    assert sync_client.xadd.call_count == 1


def test_send_sync_update_handles_exceptions() -> None:
    """Failures should return FAILED and avoid raising to the caller."""

    sync_client = MagicMock()
    sync_client.xadd.side_effect = RuntimeError("redis down")
    manager = ProgressUpdateManager(sync_redis=sync_client)

    payload = _payload()
    result = manager.send_sync_update(payload)

    assert result is ProgressSendResult.FAILED
    sync_client.expire.assert_not_called()


@pytest.mark.asyncio()
async def test_send_async_update_success() -> None:
    """Async updates should publish to the stream and set TTL when provided."""

    async_client = AsyncMock()
    manager = ProgressUpdateManager(
        async_redis=async_client,
        default_stream_template="stream:{operation_id}",
        default_ttl=90,
        default_maxlen=500,
    )

    payload = _payload()
    result = await manager.send_async_update(payload)

    assert result is ProgressSendResult.SENT
    async_client.xadd.assert_awaited_once()
    stream_key, fields = async_client.xadd.call_args[0]
    assert stream_key == "stream:op-123"
    assert fields["message"] == "processing"
    assert async_client.xadd.call_args.kwargs["maxlen"] == 500
    async_client.expire.assert_awaited_once_with("stream:op-123", 90)


@pytest.mark.asyncio()
async def test_send_async_update_throttle_skip() -> None:
    """Async throttling should bypass subsequent sends within the window."""

    async_client = AsyncMock()
    manager = ProgressUpdateManager(
        async_redis=async_client,
        default_stream_template="stream:{operation_id}",
        default_ttl=90,
        async_throttle_interval=120,
    )

    payload = _payload()
    first = await manager.send_async_update(payload, use_throttle=True)
    second = await manager.send_async_update(payload, use_throttle=True)

    assert first is ProgressSendResult.SENT
    assert second is ProgressSendResult.SKIPPED
    assert async_client.xadd.await_count == 1


@pytest.mark.asyncio()
async def test_send_async_update_without_client() -> None:
    """When no async client is configured the call should fail gracefully."""

    manager = ProgressUpdateManager(async_redis=None)
    result = await manager.send_async_update(_payload())
    assert result is ProgressSendResult.FAILED


@pytest.mark.asyncio()
async def test_send_async_update_handles_exceptions() -> None:
    """Runtime errors in async publishing should be reported as FAILED."""

    async_client = AsyncMock()
    async_client.xadd.side_effect = RuntimeError("xadd failure")
    manager = ProgressUpdateManager(async_redis=async_client)

    result = await manager.send_async_update(_payload())

    assert result is ProgressSendResult.FAILED
    async_client.expire.assert_not_awaited()
