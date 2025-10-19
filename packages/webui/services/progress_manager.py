"""Shared progress update utilities for Redis stream publishing.

This module centralises the logic for building and publishing progress updates
to Redis streams from both synchronous (Celery) and asynchronous (FastAPI /
WebSocket) contexts.  It ensures consistent payload formatting, optional TTL
handling, error reporting, and throttling.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Mapping

    from redis import Redis
    from redis.asyncio import Redis as AsyncRedis
else:  # pragma: no cover - runtime duck typing suffices
    Mapping = Any  # type: ignore[assignment]
    Redis = Any
    AsyncRedis = Any

logger = logging.getLogger(__name__)


class ProgressSendResult(Enum):
    """Outcome returned by progress publishing operations."""

    SENT = "sent"
    SKIPPED = "skipped"
    FAILED = "failed"


def _stringify(value: Any) -> str:
    """Convert a value to a Redis-compatible string representation."""

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (int, float, bool)):  # noqa: UP038 - tuple needed for runtime compatibility
        return str(value)
    return json.dumps(value, default=str)


@dataclass(slots=True)
class ProgressPayload:
    """Normalised description of a progress update."""

    operation_id: str
    correlation_id: str | None = None
    progress: float | int | None = None
    message: str | None = None
    status: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_stream_fields(self) -> dict[str, str]:
        """Render the payload as Redis stream fields."""

        fields: dict[str, Any] = {"operation_id": self.operation_id}

        if self.correlation_id is not None:
            fields["correlation_id"] = self.correlation_id
        if self.progress is not None:
            fields["progress"] = self.progress
        if self.message is not None:
            fields["message"] = self.message
        if self.status is not None:
            fields["status"] = self.status

        # Include timestamp for consumers that expect it in the stream entry.
        fields.setdefault("timestamp", self.timestamp.isoformat())

        for key, value in self.extra.items():
            fields[key] = value

        return {key: _stringify(value) for key, value in fields.items() if value is not None}

    def to_hash_mapping(self) -> dict[str, str]:
        """Build a sensible default hash mapping for progress tracking."""

        mapping: dict[str, Any] = {"updated_at": self.timestamp.isoformat()}

        if self.progress is not None:
            mapping["progress"] = self.progress
        if self.message is not None:
            mapping["message"] = self.message
        if self.status is not None:
            mapping["status"] = self.status
        if self.correlation_id is not None:
            mapping["correlation_id"] = self.correlation_id

        return {key: _stringify(value) for key, value in mapping.items() if value is not None}


class _ThrottleController:
    """Throttle helper that supports both sync and async producers."""

    def __init__(
        self,
        *,
        sync_interval_seconds: float | None = None,
        async_interval_seconds: float | None = None,
    ) -> None:
        self._sync_interval = sync_interval_seconds
        self._async_interval = async_interval_seconds
        self._sync_last_sent: dict[str, datetime] = {}
        self._async_last_sent: dict[str, datetime] = {}
        self._sync_lock = threading.Lock()
        self._async_lock = asyncio.Lock()

    def allow_sync(self, key: str) -> bool:
        """Return True if a sync update should be sent for *key*."""

        if self._sync_interval is None or self._sync_interval <= 0:
            return True

        now = datetime.now(UTC)
        with self._sync_lock:
            last_sent = self._sync_last_sent.get(key)
            if last_sent and (now - last_sent).total_seconds() < self._sync_interval:
                return False
            self._sync_last_sent[key] = now
        return True

    async def allow_async(self, key: str) -> bool:
        """Return True if an async update should be sent for *key*."""

        if self._async_interval is None or self._async_interval <= 0:
            return True

        now = datetime.now(UTC)
        async with self._async_lock:
            last_sent = self._async_last_sent.get(key)
            if last_sent and (now - last_sent).total_seconds() < self._async_interval:
                return False
            self._async_last_sent[key] = now
        return True

    def reset_sync(self, key: str) -> None:
        with self._sync_lock:
            self._sync_last_sent.pop(key, None)

    async def reset_async(self, key: str) -> None:
        async with self._async_lock:
            self._async_last_sent.pop(key, None)


class ProgressUpdateManager:
    """Unified manager for sending progress updates to Redis streams."""

    def __init__(
        self,
        *,
        sync_redis: Redis | None = None,
        async_redis: AsyncRedis | None = None,
        default_stream_template: str = "operation-progress:{operation_id}",
        default_ttl: int | None = 86400,
        default_maxlen: int | None = 1000,
        sync_throttle_interval: float | None = None,
        async_throttle_interval: float | None = None,
        logger_: logging.Logger | None = None,
    ) -> None:
        self._sync_redis = sync_redis
        self._async_redis = async_redis
        self._default_stream_template = default_stream_template
        self._default_ttl = default_ttl
        self._default_maxlen = default_maxlen
        self._throttle = _ThrottleController(
            sync_interval_seconds=sync_throttle_interval,
            async_interval_seconds=async_throttle_interval,
        )
        self._logger = logger_ or logger

    def set_sync_client(self, client: Redis) -> None:
        """Attach or replace the synchronous Redis client."""

        self._sync_redis = client

    def set_async_client(self, client: AsyncRedis) -> None:
        """Attach or replace the asynchronous Redis client."""

        self._async_redis = client

    def _resolve_maxlen(self, override: int | None) -> int | None:
        """Determine the effective maxlen to apply to XADD."""

        if override is None:
            return self._default_maxlen
        if override <= 0:
            return None
        return override

    def send_sync_update(
        self,
        payload: ProgressPayload,
        *,
        stream_fields: Mapping[str, Any] | None = None,
        stream_template: str | None = None,
        ttl: int | None = None,
        maxlen: int | None = None,
        hash_key_template: str | None = None,
        hash_mapping: Mapping[str, Any] | None = None,
        use_throttle: bool = False,
        throttle_key: str | None = None,
        redis_client: Redis | None = None,
    ) -> ProgressSendResult:
        """Publish a progress update using a synchronous Redis client."""

        client = redis_client or self._sync_redis
        if client is None:
            self._logger.debug("No sync Redis client configured; skipping progress update")
            return ProgressSendResult.FAILED

        key = throttle_key or payload.operation_id
        if use_throttle and not self._throttle.allow_sync(key):
            return ProgressSendResult.SKIPPED

        stream_key = (stream_template or self._default_stream_template).format(operation_id=payload.operation_id)
        fields = (
            {name: _stringify(value) for name, value in stream_fields.items()}
            if stream_fields is not None
            else payload.to_stream_fields()
        )

        try:
            effective_maxlen = self._resolve_maxlen(maxlen)
            if effective_maxlen is None:
                client.xadd(stream_key, fields)
            else:
                client.xadd(stream_key, fields, maxlen=effective_maxlen)

            ttl_value = ttl if ttl is not None else self._default_ttl
            if ttl_value:
                client.expire(stream_key, ttl_value)

            if hash_key_template:
                hash_key = hash_key_template.format(operation_id=payload.operation_id)
                mapping_raw = (
                    {name: _stringify(value) for name, value in hash_mapping.items()}
                    if hash_mapping is not None
                    else payload.to_hash_mapping()
                )
                if mapping_raw:
                    mapping_typed = cast(
                        "Mapping[str | bytes, bytes | float | int | str]",
                        mapping_raw,
                    )
                    client.hset(hash_key, mapping=mapping_typed)

            return ProgressSendResult.SENT
        except Exception as exc:  # pragma: no cover - defensive logging
            self._logger.warning(
                "Failed to send sync progress update for %s: %s",
                payload.operation_id,
                exc,
            )
            try:
                # If an exception occurred before throttling recorded the send,
                # ensure future attempts are not blocked.
                if use_throttle:
                    self._throttle.reset_sync(key)
            except Exception:  # pragma: no cover - safety
                pass
            return ProgressSendResult.FAILED

    async def send_async_update(
        self,
        payload: ProgressPayload,
        *,
        stream_fields: Mapping[str, Any] | None = None,
        stream_template: str | None = None,
        ttl: int | None = None,
        maxlen: int | None = None,
        hash_key_template: str | None = None,
        hash_mapping: Mapping[str, Any] | None = None,
        use_throttle: bool = False,
        throttle_key: str | None = None,
        redis_client: AsyncRedis | None = None,
    ) -> ProgressSendResult:
        """Publish a progress update using an async Redis client."""

        client = redis_client or self._async_redis
        if client is None:
            self._logger.debug("No async Redis client configured; skipping progress update")
            return ProgressSendResult.FAILED

        key = throttle_key or payload.operation_id
        if use_throttle and not await self._throttle.allow_async(key):
            return ProgressSendResult.SKIPPED

        stream_key = (stream_template or self._default_stream_template).format(operation_id=payload.operation_id)
        fields = (
            {name: _stringify(value) for name, value in stream_fields.items()}
            if stream_fields is not None
            else payload.to_stream_fields()
        )

        try:
            effective_maxlen = self._resolve_maxlen(maxlen)
            if effective_maxlen is None:
                await client.xadd(stream_key, fields)
            else:
                await client.xadd(stream_key, fields, maxlen=effective_maxlen)

            ttl_value = ttl if ttl is not None else self._default_ttl
            if ttl_value:
                await client.expire(stream_key, ttl_value)

            if hash_key_template:
                hash_key = hash_key_template.format(operation_id=payload.operation_id)
                mapping_raw = (
                    {name: _stringify(value) for name, value in hash_mapping.items()}
                    if hash_mapping is not None
                    else payload.to_hash_mapping()
                )
                if mapping_raw:
                    mapping_typed = cast(
                        "Mapping[str | bytes, bytes | float | int | str]",
                        mapping_raw,
                    )
                    await client.hset(hash_key, mapping=mapping_typed)

            return ProgressSendResult.SENT
        except Exception as exc:  # pragma: no cover - defensive logging
            self._logger.warning(
                "Failed to send async progress update for %s: %s",
                payload.operation_id,
                exc,
            )
            try:
                if use_throttle:
                    await self._throttle.reset_async(key)
            except Exception:  # pragma: no cover - safety
                pass
            return ProgressSendResult.FAILED
