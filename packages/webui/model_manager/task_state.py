"""Redis task state management for model download/delete operations.

This module manages Redis keys for:
1. Active operation tracking (cross-operation exclusion)
2. Task progress tracking (status, bytes, errors)

Key Structure:
- Active key: `model-manager:active:{model_id}` -> `{operation}:{task_id}`
- Progress hash: `model-manager:task:{task_id}` -> hash with status, bytes, etc.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from webui.model_manager.constants import ACTIVE_KEY_TTL, PROGRESS_KEY_TTL

if TYPE_CHECKING:
    import redis
    import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

# Redis key prefixes
ACTIVE_KEY_PREFIX = "model-manager:active:"
PROGRESS_KEY_PREFIX = "model-manager:task:"

RELEASE_IF_OWNER_SCRIPT = """
local key = KEYS[1]
local expected = ARGV[1]
local current = redis.call('GET', key)
if current == expected then
    redis.call('DEL', key)
    return 1
end
return 0
"""


class CrossOpConflictError(Exception):
    """Raised when attempting an operation while a conflicting operation is active."""

    def __init__(self, model_id: str, active_operation: str, active_task_id: str):
        self.model_id = model_id
        self.active_operation = active_operation
        self.active_task_id = active_task_id
        super().__init__(
            f"Cannot start operation: {active_operation} already active "
            f"for model {model_id} (task: {active_task_id})"
        )


def _active_key(model_id: str) -> str:
    """Generate Redis key for active operation tracking."""
    return f"{ACTIVE_KEY_PREFIX}{model_id}"


def _progress_key(task_id: str) -> str:
    """Generate Redis key for task progress hash."""
    return f"{PROGRESS_KEY_PREFIX}{task_id}"


def _parse_active_value(value: str | bytes | None) -> tuple[str, str] | None:
    """Parse active key value into (operation, task_id) or None."""
    if not value:
        return None
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            logger.warning("Failed to decode Redis value: %r", value)
            return None
    parts = value.split(":", 1)
    if len(parts) != 2:
        return None
    return (parts[0], parts[1])


# =============================================================================
# Async functions (for FastAPI endpoints)
# =============================================================================


async def claim_model_operation(
    redis_client: aioredis.Redis,
    model_id: str,
    operation: str,
    task_id: str,
) -> tuple[bool, str | None]:
    """Atomically claim an operation slot for a model.

    Uses SET NX EX for atomic claim. Returns (success, existing_task_id).
    If same operation is already active, returns (False, existing_task_id).
    If different operation is active, raises CrossOpConflictError.

    Args:
        redis_client: Async Redis client
        model_id: HuggingFace model ID
        operation: Operation type ("download" or "delete")
        task_id: Unique task identifier

    Returns:
        Tuple of (claimed: bool, existing_task_id: str | None)
        - (True, None) if claim succeeded
        - (False, existing_task_id) if same operation already active

    Raises:
        CrossOpConflictError: If a different operation is active
    """
    key = _active_key(model_id)
    value = f"{operation}:{task_id}"

    # Try atomic SET NX EX
    claimed = await redis_client.set(key, value, nx=True, ex=ACTIVE_KEY_TTL)

    if claimed:
        logger.debug("Claimed %s operation for %s (task: %s)", operation, model_id, task_id)
        return (True, None)

    # Claim failed - check existing operation
    existing_value = await redis_client.get(key)
    existing = _parse_active_value(existing_value)

    if existing is None:
        # Race condition: key was deleted between SET and GET
        # Try once more
        claimed = await redis_client.set(key, value, nx=True, ex=ACTIVE_KEY_TTL)
        if claimed:
            return (True, None)
        # Still failed, fetch again
        existing_value = await redis_client.get(key)
        existing = _parse_active_value(existing_value)

    if existing is None:
        # Shouldn't happen, but handle gracefully
        logger.warning("Unexpected state: claim failed but no existing operation for %s", model_id)
        return (False, None)

    existing_op, existing_task_id = existing

    if existing_op != operation:
        # Cross-operation conflict
        raise CrossOpConflictError(model_id, existing_op, existing_task_id)

    # Same operation already active - return for de-duplication
    logger.debug(
        "De-dupe: %s already active for %s (task: %s)",
        operation,
        model_id,
        existing_task_id,
    )
    return (False, existing_task_id)


async def release_model_operation(
    redis_client: aioredis.Redis,
    model_id: str,
) -> None:
    """Release the active operation slot for a model.

    Should be called on task completion or failure.

    Args:
        redis_client: Async Redis client
        model_id: HuggingFace model ID
    """
    key = _active_key(model_id)
    await redis_client.delete(key)
    logger.debug("Released operation slot for %s", model_id)


async def release_model_operation_if_owner(
    redis_client: aioredis.Redis,
    model_id: str,
    operation: str,
    task_id: str,
) -> bool:
    """Release the active operation slot for a model if owned by (operation, task_id).

    Returns:
        True if the key was released, False otherwise.
    """
    key = _active_key(model_id)
    expected_value = f"{operation}:{task_id}"
    released = await redis_client.eval(RELEASE_IF_OWNER_SCRIPT, 1, key, expected_value)
    if released == 1:
        logger.debug("Released operation slot for %s (owned by %s:%s)", model_id, operation, task_id)
        return True
    return False


async def get_active_operation(
    redis_client: aioredis.Redis,
    model_id: str,
) -> tuple[str, str] | None:
    """Get the active operation for a model, if any.

    Args:
        redis_client: Async Redis client
        model_id: HuggingFace model ID

    Returns:
        Tuple of (operation, task_id) or None if no active operation
    """
    key = _active_key(model_id)
    value = await redis_client.get(key)
    return _parse_active_value(value)


async def init_task_progress(
    redis_client: aioredis.Redis,
    task_id: str,
    model_id: str,
    operation: str,
) -> None:
    """Initialize a task progress hash with pending status.

    Args:
        redis_client: Async Redis client
        task_id: Unique task identifier
        model_id: HuggingFace model ID
        operation: Operation type ("download" or "delete")
    """
    key = _progress_key(task_id)
    now = time.time()

    await redis_client.hset(
        key,
        mapping={
            "task_id": task_id,
            "model_id": model_id,
            "operation": operation,
            "status": "pending",
            "bytes_downloaded": "0",
            "bytes_total": "0",
            "error": "",
            "updated_at": str(now),
        },
    )
    await redis_client.expire(key, PROGRESS_KEY_TTL)
    logger.debug("Initialized progress for task %s", task_id)


async def update_task_progress(
    redis_client: aioredis.Redis,
    task_id: str,
    status: str,
    bytes_downloaded: int | None = None,
    bytes_total: int | None = None,
    error: str | None = None,
) -> None:
    """Update task progress hash.

    Also refreshes the active key TTL if the task is still running.

    Args:
        redis_client: Async Redis client
        task_id: Unique task identifier
        status: Current status (pending, running, completed, failed)
        bytes_downloaded: Bytes downloaded so far
        bytes_total: Total bytes to download
        error: Error message if failed
    """
    key = _progress_key(task_id)
    now = time.time()

    mapping: dict[str | bytes, bytes | float | int | str] = {
        "status": status,
        "updated_at": str(now),
    }
    if bytes_downloaded is not None:
        mapping["bytes_downloaded"] = str(bytes_downloaded)
    if bytes_total is not None:
        mapping["bytes_total"] = str(bytes_total)
    if error is not None:
        mapping["error"] = error

    await redis_client.hset(key, mapping=mapping)

    # Refresh progress TTL
    await redis_client.expire(key, PROGRESS_KEY_TTL)

    # If still running, refresh active key TTL via model_id lookup
    if status == "running":
        progress = await redis_client.hgetall(key)
        if progress and "model_id" in progress:
            model_id = progress["model_id"]
            active_key = _active_key(model_id)
            await redis_client.expire(active_key, ACTIVE_KEY_TTL)


async def get_task_progress(
    redis_client: aioredis.Redis,
    task_id: str,
) -> dict[str, Any] | None:
    """Get task progress as a dictionary.

    Args:
        redis_client: Async Redis client
        task_id: Unique task identifier

    Returns:
        Dictionary with progress fields or None if not found
    """
    key = _progress_key(task_id)
    data = await redis_client.hgetall(key)

    if not data:
        return None

    # Convert string values to appropriate types
    return {
        "task_id": data.get("task_id", task_id),
        "model_id": data.get("model_id", ""),
        "operation": data.get("operation", ""),
        "status": data.get("status", "pending"),
        "bytes_downloaded": int(data.get("bytes_downloaded", "0")),
        "bytes_total": int(data.get("bytes_total", "0")),
        "error": data.get("error") or None,
        "updated_at": float(data.get("updated_at", "0")),
    }


# =============================================================================
# Sync functions (for Celery tasks)
# =============================================================================


def task_progress_exists_sync(
    redis_client: redis.Redis,
    task_id: str,
) -> bool:
    """Return True if the task progress hash exists."""
    return bool(redis_client.exists(_progress_key(task_id)))


def claim_model_operation_sync(
    redis_client: redis.Redis,
    model_id: str,
    operation: str,
    task_id: str,
) -> tuple[bool, str | None]:
    """Synchronous version of claim_model_operation for Celery tasks.

    Args:
        redis_client: Sync Redis client
        model_id: HuggingFace model ID
        operation: Operation type ("download" or "delete")
        task_id: Unique task identifier

    Returns:
        Tuple of (claimed: bool, existing_task_id: str | None)

    Raises:
        CrossOpConflictError: If a different operation is active
    """
    key = _active_key(model_id)
    value = f"{operation}:{task_id}"

    claimed = redis_client.set(key, value, nx=True, ex=ACTIVE_KEY_TTL)

    if claimed:
        logger.debug("Claimed %s operation for %s (task: %s)", operation, model_id, task_id)
        return (True, None)

    existing_value = redis_client.get(key)
    existing = _parse_active_value(existing_value)

    if existing is None:
        claimed = redis_client.set(key, value, nx=True, ex=ACTIVE_KEY_TTL)
        if claimed:
            return (True, None)
        existing_value = redis_client.get(key)
        existing = _parse_active_value(existing_value)

    if existing is None:
        logger.warning("Unexpected state: claim failed but no existing operation for %s", model_id)
        return (False, None)

    existing_op, existing_task_id = existing

    if existing_op != operation:
        raise CrossOpConflictError(model_id, existing_op, existing_task_id)

    logger.debug(
        "De-dupe: %s already active for %s (task: %s)",
        operation,
        model_id,
        existing_task_id,
    )
    return (False, existing_task_id)


def release_model_operation_sync(
    redis_client: redis.Redis,
    model_id: str,
) -> None:
    """Synchronous version of release_model_operation for Celery tasks."""
    key = _active_key(model_id)
    redis_client.delete(key)
    logger.debug("Released operation slot for %s", model_id)


def release_model_operation_if_owner_sync(
    redis_client: redis.Redis,
    model_id: str,
    operation: str,
    task_id: str,
) -> bool:
    """Release the active operation slot for a model if owned by (operation, task_id).

    Returns:
        True if the key was released, False otherwise.
    """
    key = _active_key(model_id)
    expected_value = f"{operation}:{task_id}"
    released = redis_client.eval(RELEASE_IF_OWNER_SCRIPT, 1, key, expected_value)
    if released == 1:
        logger.debug("Released operation slot for %s (owned by %s:%s)", model_id, operation, task_id)
        return True
    return False


def get_active_operation_sync(
    redis_client: redis.Redis,
    model_id: str,
) -> tuple[str, str] | None:
    """Synchronous version of get_active_operation for Celery tasks."""
    key = _active_key(model_id)
    value = redis_client.get(key)
    return _parse_active_value(value)


def update_task_progress_sync(
    redis_client: redis.Redis,
    task_id: str,
    status: str,
    bytes_downloaded: int | None = None,
    bytes_total: int | None = None,
    error: str | None = None,
) -> None:
    """Synchronous version of update_task_progress for Celery tasks."""
    key = _progress_key(task_id)
    now = time.time()

    mapping: dict[str | bytes, bytes | float | int | str] = {
        "status": status,
        "updated_at": str(now),
    }
    if bytes_downloaded is not None:
        mapping["bytes_downloaded"] = str(bytes_downloaded)
    if bytes_total is not None:
        mapping["bytes_total"] = str(bytes_total)
    if error is not None:
        mapping["error"] = error

    redis_client.hset(key, mapping=mapping)
    redis_client.expire(key, PROGRESS_KEY_TTL)

    if status == "running":
        progress = redis_client.hgetall(key)
        if progress and "model_id" in progress:
            model_id = progress["model_id"]
            active_key = _active_key(model_id)
            redis_client.expire(active_key, ACTIVE_KEY_TTL)


def init_task_progress_sync(
    redis_client: redis.Redis,
    task_id: str,
    model_id: str,
    operation: str,
) -> None:
    """Synchronous version of init_task_progress for Celery tasks."""
    key = _progress_key(task_id)
    now = time.time()

    redis_client.hset(
        key,
        mapping={
            "task_id": task_id,
            "model_id": model_id,
            "operation": operation,
            "status": "pending",
            "bytes_downloaded": "0",
            "bytes_total": "0",
            "error": "",
            "updated_at": str(now),
        },
    )
    redis_client.expire(key, PROGRESS_KEY_TTL)
    logger.debug("Initialized progress for task %s", task_id)
