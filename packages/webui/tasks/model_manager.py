"""Celery tasks for model download and delete operations.

This module provides background tasks for:
- Downloading HuggingFace models via snapshot_download
- Deleting models from the HuggingFace cache
"""

from __future__ import annotations

import logging
import threading
import time
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

# Process-wide Redis manager reuse for Celery workers.
# Creating a new RedisManager per task wastes connections and memory under load.
_redis_manager: RedisManager | None = None
_redis_manager_lock = threading.Lock()

# Time limits
DOWNLOAD_SOFT_TIME_LIMIT = 14400  # 4 hours
DOWNLOAD_HARD_TIME_LIMIT = 21600  # 6 hours
DELETE_SOFT_TIME_LIMIT = 600  # 10 minutes
DELETE_HARD_TIME_LIMIT = 900  # 15 minutes


def _get_sync_redis_client() -> Any:
    """Get synchronous Redis client for Celery tasks."""
    global _redis_manager
    if _redis_manager is None:
        with _redis_manager_lock:
            if _redis_manager is None:
                config = RedisConfig(url=settings.REDIS_URL)
                _redis_manager = RedisManager(config)
    return _redis_manager.sync_client


class _TaskHeartbeat:
    """Heartbeat thread that refreshes progress + active-key TTL while a task runs."""

    def __init__(self, redis_client: Any, task_id: str, *, interval_seconds: int = 30) -> None:
        self._redis_client = redis_client
        self._task_id = task_id
        self._interval_seconds = interval_seconds
        self._consecutive_failures = 0
        self._warned_after_failures = False
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
                )
                self._consecutive_failures = 0
                self._warned_after_failures = False
            except Exception as e:
                self._consecutive_failures += 1
                if self._consecutive_failures >= 3 and (
                    not self._warned_after_failures or self._consecutive_failures % 30 == 0
                ):
                    logger.warning(
                        "Heartbeat progress update failed %d times: %s",
                        self._consecutive_failures,
                        e,
                    )
                    self._warned_after_failures = True
                else:
                    logger.debug("Heartbeat progress update failed: %s", e)


class _DownloadProgressAggregator:
    """Aggregate byte progress across potentially multiple HF progress bars."""

    def __init__(
        self,
        redis_client: Any,
        task_id: str,
        *,
        min_update_interval_seconds: float = 0.5,
    ) -> None:
        self._redis_client = redis_client
        self._task_id = task_id
        self._min_update_interval_seconds = min_update_interval_seconds

        self._lock = threading.Lock()
        self._enabled = True
        self._bars: dict[int, tuple[int, int]] = {}
        self._last_emit_monotonic: float = 0.0

        self._latest_downloaded: int = 0
        self._latest_total: int = 0
        self._consecutive_failures = 0
        self._warned_after_failures = False

    def set_bar(self, bar_id: int, *, downloaded: int, total: int) -> None:
        with self._lock:
            if not self._enabled:
                return
            self._bars[bar_id] = (max(downloaded, 0), max(total, 0))
        self._maybe_emit(force=False)

    def remove_bar(self, bar_id: int) -> None:
        with self._lock:
            self._bars.pop(bar_id, None)
        self._maybe_emit(force=True)

    def clear_all_bars(self) -> None:
        with self._lock:
            self._bars.clear()

    def flush(self) -> tuple[int, int]:
        self._maybe_emit(force=True)
        return self.latest()

    def disable(self) -> None:
        with self._lock:
            self._enabled = False
            self._bars.clear()

    def latest(self) -> tuple[int, int]:
        with self._lock:
            return (self._latest_downloaded, self._latest_total)

    def _maybe_emit(self, *, force: bool) -> None:
        now = time.monotonic()
        if not force and (now - self._last_emit_monotonic) < self._min_update_interval_seconds:
            return

        with self._lock:
            bytes_downloaded = sum(downloaded for downloaded, _total in self._bars.values())
            bytes_total = sum(total for _downloaded, total in self._bars.values() if total > 0)
            self._latest_downloaded = bytes_downloaded
            self._latest_total = bytes_total
            self._last_emit_monotonic = now
            enabled = self._enabled

        if enabled:
            try:
                task_state.update_task_progress_sync(
                    self._redis_client,
                    self._task_id,
                    status="running",
                    bytes_downloaded=bytes_downloaded,
                    bytes_total=bytes_total,
                )
                self._consecutive_failures = 0
                self._warned_after_failures = False
            except Exception as e:
                self._consecutive_failures += 1
                if self._consecutive_failures >= 3 and (
                    not self._warned_after_failures or self._consecutive_failures % 30 == 0
                ):
                    logger.warning(
                        "Progress update failed %d times: %s",
                        self._consecutive_failures,
                        e,
                    )
                    self._warned_after_failures = True
                else:
                    logger.debug("Progress update failed: %s", e)


def _is_retryable_error(exc: Exception) -> bool:
    """Check if an exception is retryable."""
    try:
        import httpx
    except Exception:  # pragma: no cover - optional dependency in some deployments
        httpx = None  # type: ignore[assignment]

    try:
        import requests
    except Exception:  # pragma: no cover - requests is not a hard dependency anymore
        requests = None  # type: ignore[assignment]

    # Network errors are retryable
    if isinstance(exc, ConnectionError | TimeoutError):
        return True
    if httpx is not None and isinstance(exc, httpx.TransportError | httpx.TimeoutException):
        return True
    if requests is not None and isinstance(exc, requests.exceptions.ConnectionError):
        return True

    # HTTP errors - some are retryable
    status_code = _extract_http_status_code(exc)
    if status_code is not None:
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
    status_code = _extract_http_status_code(exc)
    if status_code is not None:
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


def _extract_http_status_code(exc: Exception) -> int | None:
    """Try to extract an HTTP status code from common exception types.

    With `huggingface_hub>=1.0.0`, network calls are powered by `httpx`, but other
    dependencies may still raise `requests` exceptions. This helper keeps our
    retry/fatal logic robust across both backends.
    """
    # Direct status_code attribute (rare but easy)
    direct = getattr(exc, "status_code", None)
    if isinstance(direct, int):
        return direct

    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None) if response is not None else None
    if isinstance(status_code, int):
        return status_code

    # Some exception wrappers store the originating exception as __cause__/__context__.
    cause = getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
    if isinstance(cause, Exception) and cause is not exc:
        return _extract_http_status_code(cause)

    return None


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
    progress_aggregator: _DownloadProgressAggregator | None = None
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

        download_progress_aggregator = _DownloadProgressAggregator(redis_client, task_id)
        progress_aggregator = download_progress_aggregator

        tqdm_class: Any | None = None
        try:
            from tqdm.auto import tqdm as _tqdm

            class _RedisBytesTqdm(_tqdm):
                def __init__(self, *args: Any, **kwargs: Any) -> None:
                    self._track_bytes = kwargs.get("unit") == "B"
                    super().__init__(*args, **kwargs)
                    if self._track_bytes:
                        total = int(self.total) if self.total is not None else 0
                        download_progress_aggregator.set_bar(id(self), downloaded=int(self.n), total=total)

                def update(self, n: float = 1) -> Any:  # noqa: ANN401
                    result = super().update(n)
                    if self._track_bytes:
                        total = int(self.total) if self.total is not None else 0
                        download_progress_aggregator.set_bar(id(self), downloaded=int(self.n), total=total)
                    return result

                def close(self) -> None:
                    try:
                        if getattr(self, "_track_bytes", False):
                            download_progress_aggregator.remove_bar(id(self))
                    finally:
                        super().close()

            tqdm_class = _RedisBytesTqdm
        except ImportError as e:
            # tqdm isn't strictly required; we'll fall back to heartbeat-only updates.
            logger.debug("tqdm not available (%s) - download progress bytes will be unavailable", e)
            tqdm_class = None
        except Exception as e:
            # tqdm isn't strictly required; we'll fall back to heartbeat-only updates.
            logger.warning("Failed to initialize tqdm for download progress: %s", e)
            tqdm_class = None

        # Keep Redis state fresh while snapshot_download runs (may take hours).
        heartbeat = _TaskHeartbeat(redis_client, task_id, interval_seconds=30)
        heartbeat.start()

        # Download the model
        # Note: HF_HOME environment variable controls cache location
        snapshot_kwargs: dict[str, Any] = {"repo_id": model_id}
        if tqdm_class is not None:
            try:
                import inspect

                sig = inspect.signature(snapshot_download)
                accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
                if "tqdm_class" in sig.parameters or accepts_var_kw:
                    snapshot_kwargs["tqdm_class"] = tqdm_class
            except Exception as e:
                # Be conservative: don't pass tqdm_class if we can't determine support.
                logger.warning("Unable to determine snapshot_download tqdm_class support: %s", e)

        local_dir = snapshot_download(**snapshot_kwargs)

        logger.info("Download complete for %s: %s", model_id, local_dir)

        # Stop heartbeat BEFORE writing terminal status to avoid a race where a heartbeat
        # tick overwrites the terminal status with "running".
        if heartbeat is not None:
            heartbeat.stop()
            heartbeat = None

        final_downloaded, final_total = (0, 0)
        if progress_aggregator is not None:
            final_downloaded, final_total = progress_aggregator.flush()
            progress_aggregator.disable()

        # Update final status
        task_state.update_task_progress_sync(
            redis_client,
            task_id,
            status="completed",
            bytes_downloaded=final_downloaded,
            bytes_total=final_total,
        )

        return {
            "task_id": task_id,
            "model_id": model_id,
            "status": "completed",
            "local_dir": local_dir,
        }

    except task_state.CrossOpConflictError as e:
        if heartbeat is not None:
            heartbeat.stop()
            heartbeat = None
        if progress_aggregator is not None:
            progress_aggregator.disable()
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
        if heartbeat is not None:
            heartbeat.stop()
            heartbeat = None
        if progress_aggregator is not None:
            progress_aggregator.disable()
        logger.error("Download timeout for %s (task: %s)", model_id, task_id)
        task_state.update_task_progress_sync(
            redis_client,
            task_id,
            status="failed",
            error="Download timed out",
        )
        raise

    except Exception as e:
        if heartbeat is not None:
            heartbeat.stop()
            heartbeat = None
        if progress_aggregator is not None:
            progress_aggregator.disable()
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
            for attempt in range(3):
                try:
                    task_state.release_model_operation_if_owner_sync(redis_client, model_id, operation, task_id)
                    break
                except Exception as e:
                    logger.warning(
                        "Failed to release operation slot for %s (attempt %d/3): %s",
                        model_id,
                        attempt + 1,
                        e,
                    )
                    if attempt < 2:
                        time.sleep(0.2 * (2**attempt))
            else:
                # All retries exhausted
                logger.error(
                    "Failed to release operation slot for %s after 3 attempts. " "Slot will auto-expire after TTL.",
                    model_id,
                )


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
            if repo.repo_id == model_id and repo.repo_type == "model":
                target_repo = repo
                break

        if target_repo is None:
            logger.info(
                "Model %s not found in cache (task: %s) - treating as no-op",
                model_id,
                task_id,
            )
            task_state.update_task_progress_sync(
                redis_client,
                task_id,
                status="not_installed",  # No error field - this is a successful no-op
            )
            return {
                "task_id": task_id,
                "model_id": model_id,
                "status": "not_installed",
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
        for attempt in range(3):
            try:
                task_state.release_model_operation_if_owner_sync(redis_client, model_id, operation, task_id)
                break
            except Exception as e:
                logger.warning(
                    "Failed to release operation slot for %s (attempt %d/3): %s",
                    model_id,
                    attempt + 1,
                    e,
                )
                if attempt < 2:
                    time.sleep(0.2 * (2**attempt))
        else:
            # All retries exhausted
            logger.error(
                "Failed to release operation slot for %s after 3 attempts. " "Slot will auto-expire after TTL.",
                model_id,
            )
