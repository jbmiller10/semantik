"""Celery tasks for model download and delete operations.

This module provides background tasks for:
- Downloading HuggingFace models via snapshot_download
- Deleting models from the HuggingFace cache
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from celery.exceptions import SoftTimeLimitExceeded

from shared.config import settings
from shared.model_manager.hf_cache import resolve_hf_cache_dir
from webui.celery_app import celery_app
from webui.model_manager import task_state
from webui.services.redis_manager import RedisConfig, RedisManager

logger = logging.getLogger(__name__)

# Queue configuration
MODEL_MANAGER_QUEUE = "model-manager"

# Time limits
DOWNLOAD_SOFT_TIME_LIMIT = 14400  # 4 hours
DOWNLOAD_HARD_TIME_LIMIT = 21600  # 6 hours
DELETE_SOFT_TIME_LIMIT = 600  # 10 minutes
DELETE_HARD_TIME_LIMIT = 900  # 15 minutes


def _get_sync_redis_client() -> Any:
    """Get synchronous Redis client for Celery tasks."""
    config = RedisConfig(url=settings.REDIS_URL)
    manager = RedisManager(config)
    return manager.sync_client


class _TaskHeartbeat:
    """Heartbeat thread that refreshes progress + active-key TTL while a task runs."""

    def __init__(self, redis_client: Any, task_id: str, *, interval_seconds: int = 30) -> None:
        self._redis_client = redis_client
        self._task_id = task_id
        self._interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"model-manager-heartbeat:{task_id}", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval_seconds):
            try:
                task_state.update_task_progress_sync(
                    self._redis_client,
                    self._task_id,
                    status="running",
                    bytes_downloaded=0,
                    bytes_total=0,
                )
            except Exception as e:
                logger.debug("Heartbeat progress update failed: %s", e)


def _is_retryable_error(exc: Exception) -> bool:
    """Check if an exception is retryable."""
    import requests

    # Network errors are retryable
    if isinstance(exc, ConnectionError | TimeoutError | requests.exceptions.ConnectionError):
        return True

    # HTTP errors - some are retryable
    if isinstance(exc, requests.exceptions.HTTPError):
        status_code = getattr(exc.response, "status_code", None)
        if status_code:
            # 5xx server errors are retryable
            if 500 <= status_code < 600:
                return True
            # 429 rate limit is retryable
            if status_code == 429:
                return True
        return False

    return False


def _is_fatal_error(exc: Exception) -> bool:
    """Check if an exception should fail immediately without retry."""
    import requests

    if isinstance(exc, requests.exceptions.HTTPError):
        status_code = getattr(exc.response, "status_code", None)
        if status_code:
            # 401/403 auth errors
            if status_code in (401, 403):
                return True
            # 404 not found
            if status_code == 404:
                return True
        return False

    # Disk errors are fatal
    if isinstance(exc, OSError):
        # ENOSPC - no space left on device
        if getattr(exc, "errno", None) == 28:
            return True
        # EACCES - permission denied
        if getattr(exc, "errno", None) == 13:
            return True

    return False


@celery_app.task(
    bind=True,
    name="webui.tasks.model_manager.download_model",
    max_retries=3,
    default_retry_delay=30,
    acks_late=True,
    soft_time_limit=DOWNLOAD_SOFT_TIME_LIMIT,
    time_limit=DOWNLOAD_HARD_TIME_LIMIT,
    queue=MODEL_MANAGER_QUEUE,
)
def download_model(self: Any, model_id: str, task_id: str) -> dict[str, Any]:
    """Download a HuggingFace model using snapshot_download.

    This task:
    1. Claims the operation slot (or returns if already claimed)
    2. Downloads the model with progress updates
    3. Updates final status
    4. Releases the operation slot

    Args:
        self: Celery task instance (bound)
        model_id: HuggingFace model ID (e.g., "BAAI/bge-small-en-v1.5")
        task_id: Unique task identifier for progress tracking

    Returns:
        Dict with task_id, model_id, status, and optional error
    """
    redis_client = _get_sync_redis_client()
    heartbeat: _TaskHeartbeat | None = None
    operation = "download"
    should_release = True

    try:
        # Ensure we own the active operation slot.
        # The API claims the op before enqueuing. This check prevents tasks from
        # "stealing" locks or clearing another task's lock in finally.
        active = task_state.get_active_operation_sync(redis_client, model_id)
        if active is None:
            # Fallback for direct task invocation (e.g., manual admin ops).
            claimed, existing_task_id = task_state.claim_model_operation_sync(
                redis_client, model_id, operation, task_id
            )
            if not claimed and existing_task_id and existing_task_id != task_id:
                return {
                    "task_id": task_id,
                    "model_id": model_id,
                    "status": "deduplicated",
                    "existing_task_id": existing_task_id,
                }
        else:
            active_operation, active_task_id = active
            if active_operation != operation:
                raise task_state.CrossOpConflictError(model_id, active_operation, active_task_id)
            if active_task_id != task_id:
                return {
                    "task_id": task_id,
                    "model_id": model_id,
                    "status": "deduplicated",
                    "existing_task_id": active_task_id,
                }

        # Ensure progress exists (API usually initializes it before enqueue).
        if not task_state.task_progress_exists_sync(redis_client, task_id):
            task_state.init_task_progress_sync(redis_client, task_id, model_id, operation)

        # Update status to running
        task_state.update_task_progress_sync(redis_client, task_id, status="running", bytes_downloaded=0, bytes_total=0)

        logger.info("Starting download for model: %s (task: %s)", model_id, task_id)

        # Import huggingface_hub here to avoid loading it in all workers
        from huggingface_hub import snapshot_download

        # Keep Redis state fresh while snapshot_download runs (may take hours).
        heartbeat = _TaskHeartbeat(redis_client, task_id, interval_seconds=30)
        heartbeat.start()

        # Download the model
        # Note: HF_HOME environment variable controls cache location
        local_dir = snapshot_download(
            repo_id=model_id,
            # Progress bytes are best-effort (status updates via heartbeat).
        )

        logger.info("Download complete for %s: %s", model_id, local_dir)

        # Update final status
        task_state.update_task_progress_sync(
            redis_client,
            task_id,
            status="completed",
            bytes_downloaded=0,
            bytes_total=0,
        )

        return {
            "task_id": task_id,
            "model_id": model_id,
            "status": "completed",
            "local_dir": local_dir,
        }

    except task_state.CrossOpConflictError as e:
        logger.warning("Cross-operation conflict for %s: %s", model_id, e)
        task_state.update_task_progress_sync(
            redis_client,
            task_id,
            status="failed",
            error=f"Conflict: {e.active_operation} already active",
        )
        return {
            "task_id": task_id,
            "model_id": model_id,
            "status": "conflict",
            "error": str(e),
        }

    except SoftTimeLimitExceeded:
        logger.error("Download timeout for %s (task: %s)", model_id, task_id)
        task_state.update_task_progress_sync(
            redis_client,
            task_id,
            status="failed",
            error="Download timed out",
        )
        raise

    except Exception as e:
        logger.exception("Download failed for %s (task: %s): %s", model_id, task_id, e)

        # Check if we should retry
        if _is_fatal_error(e):
            task_state.update_task_progress_sync(
                redis_client,
                task_id,
                status="failed",
                error=str(e),
            )
            return {
                "task_id": task_id,
                "model_id": model_id,
                "status": "failed",
                "error": str(e),
            }

        if _is_retryable_error(e) and self.request.retries < self.max_retries:
            # Retry with exponential backoff
            should_release = False
            raise self.retry(exc=e, countdown=30 * (2**self.request.retries)) from e

        # Final failure
        task_state.update_task_progress_sync(
            redis_client,
            task_id,
            status="failed",
            error=str(e),
        )
        return {
            "task_id": task_id,
            "model_id": model_id,
            "status": "failed",
            "error": str(e),
        }

    finally:
        if heartbeat is not None:
            heartbeat.stop()
        if should_release:
            try:
                task_state.release_model_operation_if_owner_sync(redis_client, model_id, operation, task_id)
            except Exception as e:
                logger.warning("Failed to release operation slot for %s: %s", model_id, e)


@celery_app.task(
    bind=True,
    name="webui.tasks.model_manager.delete_model",
    max_retries=1,
    acks_late=True,
    soft_time_limit=DELETE_SOFT_TIME_LIMIT,
    time_limit=DELETE_HARD_TIME_LIMIT,
    queue=MODEL_MANAGER_QUEUE,
)
def delete_model(self: Any, model_id: str, task_id: str) -> dict[str, Any]:  # noqa: ARG001
    """Delete a model from the HuggingFace cache.

    This task:
    1. Claims the operation slot (or returns if already claimed)
    2. Locates the model in the HF cache
    3. Deletes all revisions
    4. Updates final status
    5. Releases the operation slot

    Args:
        self: Celery task instance (bound)
        model_id: HuggingFace model ID (e.g., "BAAI/bge-small-en-v1.5")
        task_id: Unique task identifier for progress tracking

    Returns:
        Dict with task_id, model_id, status, and optional error
    """
    redis_client = _get_sync_redis_client()
    operation = "delete"

    try:
        # Ensure we own the active operation slot (API claims before enqueue).
        active = task_state.get_active_operation_sync(redis_client, model_id)
        if active is None:
            claimed, existing_task_id = task_state.claim_model_operation_sync(
                redis_client, model_id, operation, task_id
            )
            if not claimed and existing_task_id and existing_task_id != task_id:
                return {
                    "task_id": task_id,
                    "model_id": model_id,
                    "status": "deduplicated",
                    "existing_task_id": existing_task_id,
                }
        else:
            active_operation, active_task_id = active
            if active_operation != operation:
                raise task_state.CrossOpConflictError(model_id, active_operation, active_task_id)
            if active_task_id != task_id:
                return {
                    "task_id": task_id,
                    "model_id": model_id,
                    "status": "deduplicated",
                    "existing_task_id": active_task_id,
                }

        if not task_state.task_progress_exists_sync(redis_client, task_id):
            task_state.init_task_progress_sync(redis_client, task_id, model_id, operation)

        # Update status to running
        task_state.update_task_progress_sync(redis_client, task_id, status="running")

        logger.info("Starting delete for model: %s (task: %s)", model_id, task_id)

        # Import huggingface_hub here to avoid loading it in all workers
        from huggingface_hub import scan_cache_dir

        # Scan the cache
        cache_info = scan_cache_dir(resolve_hf_cache_dir())

        # Find the repo for this model
        target_repo = None
        for repo in cache_info.repos:
            if repo.repo_id == model_id:
                target_repo = repo
                break

        if target_repo is None:
            logger.info("Model %s not found in cache (task: %s)", model_id, task_id)
            task_state.update_task_progress_sync(
                redis_client,
                task_id,
                status="completed",
                error="Model not found in cache",
            )
            return {
                "task_id": task_id,
                "model_id": model_id,
                "status": "not_found",
            }

        # Get all revision hashes
        revision_hashes = [rev.commit_hash for rev in target_repo.revisions]

        if not revision_hashes:
            logger.info("No revisions found for %s (task: %s)", model_id, task_id)
            task_state.update_task_progress_sync(
                redis_client,
                task_id,
                status="completed",
            )
            return {
                "task_id": task_id,
                "model_id": model_id,
                "status": "completed",
                "revisions_deleted": 0,
            }

        # Delete all revisions
        logger.info(
            "Deleting %d revisions for %s (task: %s)",
            len(revision_hashes),
            model_id,
            task_id,
        )

        delete_strategy = cache_info.delete_revisions(*revision_hashes)
        freed_size = delete_strategy.expected_freed_size
        delete_strategy.execute()

        logger.info(
            "Deleted %s, freed %d bytes (task: %s)",
            model_id,
            freed_size,
            task_id,
        )

        # Update final status
        task_state.update_task_progress_sync(
            redis_client,
            task_id,
            status="completed",
        )

        return {
            "task_id": task_id,
            "model_id": model_id,
            "status": "completed",
            "revisions_deleted": len(revision_hashes),
            "freed_bytes": freed_size,
        }

    except task_state.CrossOpConflictError as e:
        logger.warning("Cross-operation conflict for %s: %s", model_id, e)
        task_state.update_task_progress_sync(
            redis_client,
            task_id,
            status="failed",
            error=f"Conflict: {e.active_operation} already active",
        )
        return {
            "task_id": task_id,
            "model_id": model_id,
            "status": "conflict",
            "error": str(e),
        }

    except SoftTimeLimitExceeded:
        logger.error("Delete timeout for %s (task: %s)", model_id, task_id)
        task_state.update_task_progress_sync(
            redis_client,
            task_id,
            status="failed",
            error="Delete timed out",
        )
        raise

    except Exception as e:
        logger.exception("Delete failed for %s (task: %s): %s", model_id, task_id, e)
        task_state.update_task_progress_sync(
            redis_client,
            task_id,
            status="failed",
            error=str(e),
        )
        return {
            "task_id": task_id,
            "model_id": model_id,
            "status": "failed",
            "error": str(e),
        }

    finally:
        # Release the operation slot only if we still own it.
        try:
            task_state.release_model_operation_if_owner_sync(redis_client, model_id, operation, task_id)
        except Exception as e:
            logger.warning("Failed to release operation slot for %s: %s", model_id, e)
