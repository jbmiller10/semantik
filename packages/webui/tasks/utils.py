"""Shared utilities for Celery task modules.

This module centralizes cross-cutting helpers, constants, and infrastructure that
other task modules rely on. Keeping these pieces here avoids circular imports once
domain-specific task implementations are split across multiple files.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
import re
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from importlib import import_module
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, Mock

import redis.asyncio as redis
from shared.config import settings
from shared.config.internal_api_key import ensure_internal_api_key
from shared.managers.qdrant_manager import QdrantManager
from shared.metrics.collection_metrics import update_collection_stats
from webui.celery_app import celery_app
from webui.utils.qdrant_manager import qdrant_manager

logger = logging.getLogger(__name__)

try:
    ensure_internal_api_key(settings)
except RuntimeError as exc:
    logger.error("Internal API key not configured: %s", exc)
    raise


# Re-export ChunkingService for tests that patch packages.webui.tasks.ChunkingService
try:  # Prefer packages.* import path to match test patch targets
    from packages.webui.services.chunking_service import ChunkingService
except Exception:  # Fallback for runtime usage paths
    try:
        from webui.services.chunking_service import ChunkingService  # type: ignore
    except Exception:  # As a last resort, define a placeholder
        ChunkingService = None  # type: ignore


# Task timeout constants
OPERATION_SOFT_TIME_LIMIT = 3600  # 1 hour soft limit
OPERATION_HARD_TIME_LIMIT = 7200  # 2 hour hard limit

# Retry configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 60  # seconds

# Batch processing constants
EMBEDDING_BATCH_SIZE = 100
VECTOR_UPLOAD_BATCH_SIZE = 100
DOCUMENT_REMOVAL_BATCH_SIZE = 100

# Validation thresholds
REINDEX_VECTOR_COUNT_VARIANCE = 0.1  # 10% variance allowed
REINDEX_SEARCH_MISMATCH_THRESHOLD = 0.3  # 30% mismatch threshold
REINDEX_SCORE_DIFF_THRESHOLD = 0.1  # 0.1 score difference threshold

# Redis configuration
REDIS_STREAM_MAX_LEN = 1000  # Keep last 1000 messages
REDIS_STREAM_TTL = 86400  # 24 hours

# Cleanup configuration
DEFAULT_DAYS_TO_KEEP = 7  # Days to keep old results
CLEANUP_DELAY_SECONDS = 300  # 5 minutes default delay before cleaning up old collections
CLEANUP_DELAY_MIN_SECONDS = 300  # 5 minutes minimum
CLEANUP_DELAY_MAX_SECONDS = 1800  # 30 minutes maximum
CLEANUP_DELAY_PER_10K_VECTORS = 60  # Additional 1 minute per 10k vectors

# Background task executor
executor = ThreadPoolExecutor(max_workers=8)


class CeleryTaskWithOperationUpdates:
    """Helper class to send operation updates to Redis Stream from Celery tasks.

    Uses operation-progress:{operation_id} stream format for real-time updates and
    implements the async context manager protocol for automatic resource cleanup.
    """

    def __init__(self, operation_id: str):
        """Initialize with operation ID."""
        self.operation_id = operation_id
        self.redis_url = settings.REDIS_URL
        self.stream_key = f"operation-progress:{operation_id}"
        self._redis_client: redis.Redis | None = None
        self._publisher_id = (
            getattr(settings, "PROGRESS_PUBLISHER_ID", None)
            or getattr(settings, "INSTANCE_ID", None)
            or f"celery-worker:{socket.gethostname()}"
        )

    async def _get_redis(self) -> redis.Redis:
        """Get or create Redis client."""
        if self._redis_client is None:
            self._redis_client = await redis.from_url(self.redis_url, decode_responses=True)
        assert self._redis_client is not None
        return self._redis_client

    async def send_update(self, update_type: str, data: dict) -> None:
        """Send update to Redis Stream and Pub/Sub."""
        try:
            redis_client = await self._get_redis()
            message = {"timestamp": datetime.now(UTC).isoformat(), "type": update_type, "data": data}
            message_json = json.dumps(message)

            # Add to stream with automatic ID
            await redis_client.xadd(
                self.stream_key,
                {"message": message_json},
                maxlen=REDIS_STREAM_MAX_LEN,
            )

            # Publish to pub/sub listeners so WebSocket clients receive live updates
            publish_payload = {
                "message": message,
                "from_instance": self._publisher_id,
                "timestamp": time.time(),
            }
            publish_result = redis_client.publish(
                f"operation:{self.operation_id}",
                json.dumps(publish_payload),
            )
            await await_if_awaitable(publish_result)

            # Set TTL for stream
            await redis_client.expire(self.stream_key, REDIS_STREAM_TTL)

        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to send operation update: %s", exc)

    async def __aenter__(self) -> CeleryTaskWithOperationUpdates:
        redis_client = await self._get_redis()
        await redis_client.ping()
        return self

    async def __aexit__(self, *_args: Any) -> None:
        close_result = self.close()
        await await_if_awaitable(close_result)

    def close(self) -> Any:
        """Close the Redis client gracefully, supporting sync and async callers."""

        async def _close_async() -> None:
            if self._redis_client is None:
                return

            client = self._redis_client
            self._redis_client = None

            try:
                close_result = client.close()
                await await_if_awaitable(close_result)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Failed to close Redis client: %s", exc)

        close_coro = _close_async()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            resolve_awaitable_sync(close_coro)
            return None

        if loop.is_running():
            return close_coro

        resolve_awaitable_sync(close_coro)
        return None


def _get_internal_api_key() -> str:
    """Return the configured internal API key, raising if unavailable."""
    key = cast(str | None, getattr(settings, "INTERNAL_API_KEY", None))
    if not key:
        raise RuntimeError("Internal API key is not configured")
    return key


def _build_internal_api_headers() -> dict[str, str]:
    """Construct headers for internal API calls that require authentication."""
    return {
        "X-Internal-API-Key": _get_internal_api_key(),
        "Content-Type": "application/json",
    }


def extract_and_serialize_thread_safe(filepath: str) -> list[tuple[str, dict[str, Any]]]:
    """Thread-safe wrapper around extract_and_serialize that preserves metadata."""
    from shared.text_processing.extraction import extract_and_serialize

    result: list[tuple[str, dict[str, Any]]] = extract_and_serialize(filepath)
    return result


def calculate_cleanup_delay(vector_count: int) -> int:
    """Calculate cleanup delay based on collection size."""
    safe_vector_count = max(0, vector_count)

    additional_delay = (safe_vector_count // 10000) * CLEANUP_DELAY_PER_10K_VECTORS
    total_delay = CLEANUP_DELAY_MIN_SECONDS + additional_delay

    cleanup_delay = min(total_delay, CLEANUP_DELAY_MAX_SECONDS)

    logger.info(
        "Calculated cleanup delay: %ss for %s vectors (base: %ss, additional: %ss)",
        cleanup_delay,
        safe_vector_count,
        CLEANUP_DELAY_MIN_SECONDS,
        additional_delay,
    )

    return cleanup_delay


def _sanitize_error_message(error_msg: str) -> str:
    """Sanitize error messages to remove PII."""
    # Replace user home paths
    sanitized = re.sub(r"/home/[^/]+", "/home/~", error_msg)
    sanitized = re.sub(r"/Users/[^/]+", "/Users/~", sanitized)
    sanitized = re.sub(r"C:\\Users\\[^\\]+", r"C:\\Users\\~", sanitized)

    # Redact email addresses
    sanitized = re.sub(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "[email]", sanitized)

    # Remove potential usernames from common paths
    return re.sub(r"(/tmp/|/var/folders/)[^/]+/[^/]+", r"\1[redacted]", sanitized)


def _sanitize_audit_details(details: dict[str, Any] | None, _seen: set[int] | None = None) -> dict[str, Any] | None:
    """Sanitize audit log details to ensure no PII is logged."""
    if not details:
        return details

    if _seen is None:
        _seen = set()

    obj_id = id(details)
    if obj_id in _seen:
        return {"__circular_reference__": True}
    _seen.add(obj_id)

    sanitized: dict[str, Any] = {}

    for key, value in details.items():
        if any(sensitive in key.lower() for sensitive in ["password", "secret", "token", "key"]):
            continue

        if isinstance(value, str):
            sanitized[key] = _sanitize_error_message(value)
        elif isinstance(value, dict):
            sanitized_value = _sanitize_audit_details(value, _seen)
            if sanitized_value is not None:
                sanitized[key] = sanitized_value
        elif isinstance(value, list):
            sanitized[key] = [
                (
                    _sanitize_error_message(item)
                    if isinstance(item, str)
                    else _sanitize_audit_details(item, _seen) if isinstance(item, dict) else item
                )
                for item in value
            ]
        else:
            sanitized[key] = value

    _seen.discard(obj_id)
    return sanitized


async def _audit_log_operation(
    collection_id: str,
    operation_id: int,
    user_id: int | None,
    action: str,
    details: dict[str, Any] | None = None,
) -> None:
    """Create an audit log entry for a collection operation."""
    try:
        from shared.database.database import AsyncSessionLocal
        from shared.database.models import CollectionAuditLog

        sanitized_details = _sanitize_audit_details(details)

        async with AsyncSessionLocal() as session:
            audit_log = CollectionAuditLog(
                collection_id=collection_id,
                operation_id=operation_id,
                user_id=user_id,
                action=action,
                details=sanitized_details,
            )
            add_result = session.add(audit_log)
            await await_if_awaitable(add_result)
            await session.commit()
    except Exception as exc:
        logger.warning("Failed to create audit log: %s", exc)


async def _record_operation_metrics(operation_repo: Any, operation_id: str, metrics: dict[str, Any]) -> None:
    """Record operation metrics in the database."""
    try:
        # Get operation ID (database ID, not UUID)
        operation = await operation_repo.get_by_uuid(operation_id)
        if operation:
            from shared.database.database import AsyncSessionLocal
            from shared.database.models import OperationMetrics

            async with AsyncSessionLocal() as session:
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, int | float):
                        metric = OperationMetrics(
                            operation_id=operation.id,
                            metric_name=metric_name,
                            metric_value=float(metric_value),
                        )
                        add_result = session.add(metric)
                        await await_if_awaitable(add_result)
                await session.commit()
    except Exception as exc:
        logger.warning("Failed to record operation metrics: %s", exc)


async def _update_collection_metrics(collection_id: str, documents: int, vectors: int, size_bytes: int) -> None:
    """Update collection metrics in Prometheus."""
    try:
        tasks_module = import_module("packages.webui.tasks")
        update_stats = getattr(tasks_module, "update_collection_stats", update_collection_stats)
        update_stats(collection_id, documents, vectors, size_bytes)
    except Exception as exc:
        logger.warning("Failed to update collection metrics: %s", exc)


def _is_mock_like(obj: Any) -> bool:
    """Return True when the object is a unittest.mock instance."""
    return isinstance(obj, Mock | MagicMock | AsyncMock) or getattr(obj, "_mock_parent", None) is not None


def _select_patchable(tasks_module: Any, attr: str, fallback_module: Any) -> Any:
    """Return an attribute preferring patched versions and keep modules in sync."""
    candidate = getattr(tasks_module, attr, None)
    fallback = getattr(fallback_module, attr, None)

    if fallback is not None and _is_mock_like(fallback):
        selected = fallback
    elif candidate is not None:
        selected = candidate
    else:
        selected = fallback

    if selected is not None:
        setattr(tasks_module, attr, selected)

    return selected


def resolve_qdrant_manager() -> Any:
    """Return the current qdrant_manager, honoring patches on tasks or utils."""
    tasks_module = import_module("packages.webui.tasks")
    utils_module = import_module("webui.utils.qdrant_manager")

    return _select_patchable(tasks_module, "qdrant_manager", utils_module)


def resolve_qdrant_manager_class() -> Any:
    """Return the QdrantManager class, honoring patches on tasks or shared libs."""
    tasks_module = import_module("packages.webui.tasks")
    managers_module = import_module("shared.managers.qdrant_manager")

    return _select_patchable(tasks_module, "QdrantManager", managers_module)


async def await_if_awaitable(value: Any) -> Any:
    """Await coroutine-like values while passing through synchronous results."""
    if inspect.isawaitable(value):
        return await value
    return value


_WORKER_EVENT_LOOP_ATTR = "_celery_worker_event_loop"


def resolve_awaitable_sync(value: Any) -> Any:
    """Resolve coroutine-like values from synchronous contexts without recreating loops."""
    if inspect.isawaitable(value):
        tasks_module = import_module("packages.webui.tasks")
        asyncio_module = tasks_module.asyncio
        loop = getattr(tasks_module, _WORKER_EVENT_LOOP_ATTR, None)
        if loop is None or loop.is_closed():
            loop = asyncio_module.new_event_loop()
            setattr(tasks_module, _WORKER_EVENT_LOOP_ATTR, loop)
            try:
                asyncio_module.set_event_loop(loop)
            except RuntimeError:
                # set_event_loop may fail if policy forbids setting outside main thread.
                # In that case the loop will still be passed explicitly to ensure_future/run_until_complete.
                pass
        else:
            try:
                asyncio_module.set_event_loop(loop)
            except RuntimeError:
                pass

        task = asyncio_module.ensure_future(value, loop=loop)
        try:
            return loop.run_until_complete(task)
        finally:
            if not task.done():
                task.cancel()
                with contextlib.suppress(Exception):
                    loop.run_until_complete(task)
            close = getattr(value, "close", None)
            if callable(close):
                with contextlib.suppress(Exception):
                    close()
    return value


__all__ = [
    "celery_app",
    "qdrant_manager",
    "QdrantManager",
    "ChunkingService",
    "CeleryTaskWithOperationUpdates",
    "DEFAULT_DAYS_TO_KEEP",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_RETRY_DELAY",
    "DOCUMENT_REMOVAL_BATCH_SIZE",
    "EMBEDDING_BATCH_SIZE",
    "OPERATION_HARD_TIME_LIMIT",
    "OPERATION_SOFT_TIME_LIMIT",
    "REINDEX_SCORE_DIFF_THRESHOLD",
    "REINDEX_SEARCH_MISMATCH_THRESHOLD",
    "REINDEX_VECTOR_COUNT_VARIANCE",
    "REDIS_STREAM_MAX_LEN",
    "REDIS_STREAM_TTL",
    "VECTOR_UPLOAD_BATCH_SIZE",
    "CLEANUP_DELAY_PER_10K_VECTORS",
    "CLEANUP_DELAY_MAX_SECONDS",
    "CLEANUP_DELAY_MIN_SECONDS",
    "CLEANUP_DELAY_SECONDS",
    "executor",
    "extract_and_serialize_thread_safe",
    "calculate_cleanup_delay",
    "_audit_log_operation",
    "_build_internal_api_headers",
    "_get_internal_api_key",
    "_record_operation_metrics",
    "_sanitize_audit_details",
    "_sanitize_error_message",
    "_update_collection_metrics",
    "logger",
    "settings",
    "resolve_qdrant_manager",
    "resolve_qdrant_manager_class",
    "await_if_awaitable",
    "resolve_awaitable_sync",
]
