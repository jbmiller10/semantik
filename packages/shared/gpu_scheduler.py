"""GPU scheduling and resource management for Celery workers.

This module provides GPU scheduling to prevent resource contention when
multiple workers need GPU access.
"""

import logging
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

import redis
import torch
from redis.lock import Lock

logger = logging.getLogger(__name__)


class GPUScheduler:
    """Manages GPU allocation across multiple workers using Redis locks."""

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        """Initialize the GPU scheduler.

        Args:
            redis_url: Redis connection URL for distributed locking
        """
        self.redis_client = redis.from_url(redis_url)
        self.gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.lock_timeout = 7200  # 2 hours max for any GPU task
        self.blocking_timeout = 300  # Wait up to 5 minutes for GPU

        logger.info(f"GPU Scheduler initialized with {self.gpu_count} GPUs available")

    def get_available_gpu(self, preferred_gpu: int | None = None) -> int | None:
        """Get an available GPU ID.

        Args:
            preferred_gpu: Preferred GPU ID if available

        Returns:
            GPU ID if available, None otherwise
        """
        if self.gpu_count == 0:
            return None

        # Try preferred GPU first
        if preferred_gpu is not None and 0 <= preferred_gpu < self.gpu_count and self._try_acquire_gpu(preferred_gpu):
            return preferred_gpu

        # Try all GPUs
        for gpu_id in range(self.gpu_count):
            if self._try_acquire_gpu(gpu_id):
                return gpu_id

        return None

    def _try_acquire_gpu(self, gpu_id: int) -> bool:
        """Try to acquire a lock on a specific GPU.

        Args:
            gpu_id: GPU ID to acquire

        Returns:
            True if acquired, False otherwise
        """
        lock_key = f"gpu_lock:{gpu_id}"
        # Check if lock exists and is held
        if self.redis_client.get(lock_key):
            return False
        return True

    @contextmanager
    def allocate_gpu(self, task_id: str, preferred_gpu: int | None = None) -> Iterator[int | None]:
        """Allocate a GPU for a task.

        Args:
            task_id: Unique task identifier
            preferred_gpu: Preferred GPU ID if available

        Yields:
            GPU ID if allocated, None if no GPU available
        """
        gpu_id = None
        lock = None

        try:
            # Wait for an available GPU
            start_time = time.time()
            while time.time() - start_time < self.blocking_timeout:
                for candidate_gpu in range(self.gpu_count):
                    if preferred_gpu is not None and candidate_gpu != preferred_gpu:
                        continue

                    lock_key = f"gpu_lock:{candidate_gpu}"
                    lock = Lock(self.redis_client, lock_key, timeout=self.lock_timeout)

                    if lock.acquire(blocking=False):
                        gpu_id = candidate_gpu
                        logger.info(f"Task {task_id} acquired GPU {gpu_id}")

                        # Set CUDA_VISIBLE_DEVICES for this process
                        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

                        # Log GPU memory before task
                        if torch.cuda.is_available():
                            torch.cuda.set_device(0)  # Device 0 in our restricted view
                            memory_allocated = torch.cuda.memory_allocated() / 1024**3
                            memory_reserved = torch.cuda.memory_reserved() / 1024**3
                            logger.info(
                                f"GPU {gpu_id} memory before task: "
                                f"allocated={memory_allocated:.2f}GB, reserved={memory_reserved:.2f}GB"
                            )

                        yield gpu_id
                        return

                # No GPU available, wait before retrying
                logger.debug(f"Task {task_id} waiting for GPU availability...")
                time.sleep(5)

            # Timeout - no GPU available
            logger.warning(f"Task {task_id} could not acquire GPU within {self.blocking_timeout}s")
            yield None

        finally:
            if lock and gpu_id is not None:
                try:
                    # Clear GPU memory
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()

                        # Log GPU memory after task
                        memory_allocated = torch.cuda.memory_allocated() / 1024**3
                        memory_reserved = torch.cuda.memory_reserved() / 1024**3
                        logger.info(
                            f"GPU {gpu_id} memory after task: "
                            f"allocated={memory_allocated:.2f}GB, reserved={memory_reserved:.2f}GB"
                        )

                    lock.release()
                    logger.info(f"Task {task_id} released GPU {gpu_id}")
                except Exception as e:
                    logger.error(f"Error releasing GPU {gpu_id}: {e}")

    def get_gpu_status(self) -> dict[str, any]:
        """Get current GPU allocation status.

        Returns:
            Dictionary with GPU status information
        """
        status = {"total_gpus": self.gpu_count, "gpus": {}}

        for gpu_id in range(self.gpu_count):
            lock_key = f"gpu_lock:{gpu_id}"
            is_locked = bool(self.redis_client.get(lock_key))

            gpu_info = {"allocated": is_locked, "lock_key": lock_key}

            # Get GPU memory info if available
            if torch.cuda.is_available() and not is_locked:
                try:
                    torch.cuda.set_device(gpu_id)
                    gpu_info["memory_allocated_gb"] = torch.cuda.memory_allocated() / 1024**3
                    gpu_info["memory_reserved_gb"] = torch.cuda.memory_reserved() / 1024**3
                    gpu_info["memory_total_gb"] = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
                except Exception as e:
                    logger.warning(f"Could not get memory info for GPU {gpu_id}: {e}")

            status["gpus"][gpu_id] = gpu_info

        return status


# Global GPU scheduler instance
_gpu_scheduler = None
_scheduler_lock = threading.Lock()


def get_gpu_scheduler(redis_url: str | None = None) -> GPUScheduler:
    """Get or create the global GPU scheduler.

    Args:
        redis_url: Redis URL for distributed locking

    Returns:
        The global GPUScheduler instance
    """
    global _gpu_scheduler

    if _gpu_scheduler is None:
        with _scheduler_lock:
            if _gpu_scheduler is None:
                redis_url = redis_url or os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
                _gpu_scheduler = GPUScheduler(redis_url)

    return _gpu_scheduler


@contextmanager
def gpu_task(task_id: str, preferred_gpu: int | None = None) -> Iterator[int | None]:
    """Context manager for GPU-accelerated tasks.

    This is the main interface for Celery tasks that need GPU access.

    Args:
        task_id: Unique task identifier (e.g., Celery task ID)
        preferred_gpu: Preferred GPU ID if available

    Yields:
        GPU ID if allocated, None if no GPU available

    Example:
        @celery_app.task(bind=True)
        def gpu_processing_task(self, data):
            with gpu_task(self.request.id) as gpu_id:
                if gpu_id is None:
                    # Fallback to CPU processing
                    return process_on_cpu(data)
                else:
                    # GPU processing
                    return process_on_gpu(data, gpu_id)
    """
    scheduler = get_gpu_scheduler()
    with scheduler.allocate_gpu(task_id, preferred_gpu) as gpu_id:
        yield gpu_id
