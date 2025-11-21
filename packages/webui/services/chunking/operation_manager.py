"""Operational management helpers for chunking workflows.

This module centralises cross-cutting operational concerns that were
previously embedded inside Celery task implementations.  The
``ChunkingOperationManager`` owns circuit breaker state, retry
bookkeeping, dead-letter queue publishing, and resource monitoring so
service/business layers can focus on chunking logic.
"""

from __future__ import annotations

import asyncio
import json
import traceback
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from webui.api.chunking_exceptions import (
    ChunkingDependencyError,
    ChunkingMemoryError,
    ChunkingResourceLimitError,
    ChunkingTimeoutError,
    ResourceType,
)

try:  # pragma: no cover - psutil always available in production, but tests can mock it
    import psutil
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal environments
    psutil = None

import time as _time

DEFAULT_CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5
DEFAULT_CIRCUIT_BREAKER_RECOVERY_TIMEOUT = 300
DEFAULT_DEAD_LETTER_TTL_SECONDS = 7 * 24 * 60 * 60


class CircuitBreakerState(str, Enum):
    """Finite states tracked by the circuit breaker."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class ChunkingOperationManager:
    """Manage operational concerns for chunking workflows.

    Parameters are dependency-injected to keep the manager easy to test and
    replace in different execution contexts (Celery workers vs. API layer).
    """

    def __init__(
        self,
        *,
        redis_client: Any | None,
        error_handler: Any,
        error_classifier: Any,
        logger: Any,
        expected_circuit_breaker_exceptions: tuple[type[Exception], ...] | None = None,
        failure_threshold: int = DEFAULT_CIRCUIT_BREAKER_FAILURE_THRESHOLD,
        recovery_timeout: int = DEFAULT_CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
        dead_letter_ttl_seconds: int = DEFAULT_DEAD_LETTER_TTL_SECONDS,
        memory_usage_gauge: Any | None = None,
        psutil_module: Any | None = None,
        time_module: Any | None = None,
        memory_limit_gb: int = 4,
        cpu_time_limit_seconds: int = 1800,
    ) -> None:
        self._redis_client = redis_client
        self._error_handler = error_handler
        self._error_classifier = error_classifier
        self._logger = logger
        self._expected_circuit_breaker_exceptions: tuple[type[Exception], ...] = (
            tuple(expected_circuit_breaker_exceptions)
            if expected_circuit_breaker_exceptions
            else (ChunkingDependencyError,)
        )
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._dead_letter_ttl_seconds = dead_letter_ttl_seconds
        self._memory_usage_gauge = memory_usage_gauge
        self._psutil = psutil_module or psutil
        self._time = time_module or _time
        self._memory_limit_bytes = memory_limit_gb * 1024**3
        self._cpu_time_limit_seconds = cpu_time_limit_seconds

        self._circuit_breaker_state = CircuitBreakerState.CLOSED
        self._circuit_breaker_failures = 0
        self._circuit_breaker_last_failure_time: float | None = None

    # ------------------------------------------------------------------
    # Circuit breaker helpers
    # ------------------------------------------------------------------
    def allow_execution(self) -> bool:
        """Return ``True`` when the circuit breaker allows execution."""

        if self._circuit_breaker_state is CircuitBreakerState.CLOSED:
            return True

        current_time = self._time.time()

        if (
            self._circuit_breaker_state is CircuitBreakerState.OPEN
            and self._circuit_breaker_last_failure_time is not None
            and current_time - self._circuit_breaker_last_failure_time > self._recovery_timeout
        ):
            self._circuit_breaker_state = CircuitBreakerState.HALF_OPEN
            self._logger.info("Circuit breaker entering half-open state")
            return True

        if self._circuit_breaker_state is CircuitBreakerState.HALF_OPEN:
            return True

        return False

    def handle_success(self, *, task_id: str, operation_id: str, result: Any | None = None) -> None:
        """Reset circuit breaker state after a successful execution."""

        self._circuit_breaker_failures = 0
        self._circuit_breaker_state = CircuitBreakerState.CLOSED
        self._circuit_breaker_last_failure_time = None

        self._logger.info(
            "Chunking task %s completed successfully",
            task_id,
            extra={"task_id": task_id, "operation_id": operation_id, "result": result},
        )

    def handle_failure(
        self,
        *,
        exc: Exception,
        task_id: str,
        operation_id: str,
        correlation_id: str,
        retry_count: int,
        max_retries: int,
        args: tuple[Any, ...] | None,
        kwargs: dict[str, Any] | None,
    ) -> str:
        """Log failure, update circuit breaker, and publish to DLQ if needed.

        Returns the classified error code suitable for metrics.
        """

        error_code = self.classify_error(exc)

        if isinstance(exc, self._expected_circuit_breaker_exceptions):
            self._record_circuit_breaker_failure()

        self._logger.error(
            "Chunking task %s failed after %s retries",
            task_id,
            retry_count,
            extra={
                "task_id": task_id,
                "operation_id": operation_id,
                "correlation_id": correlation_id,
                "error_type": error_code,
                "retries": retry_count,
                "max_retries": max_retries,
                "traceback": traceback.format_exc(),
            },
            exc_info=exc,
        )

        if retry_count >= max_retries:
            self._send_to_dead_letter_queue(
                task_id=task_id,
                operation_id=operation_id,
                correlation_id=correlation_id,
                error=exc,
                error_type=error_code,
                args=args or (),
                kwargs=dict(kwargs or {}),
                retry_count=retry_count,
            )

        return error_code

    def handle_retry(
        self,
        *,
        exc: Exception,
        task_id: str,
        operation_id: str,
        correlation_id: str,
        retry_count: int,
    ) -> str:
        """Persist retry state for observability and future recovery."""

        error_code = self.classify_error(exc)

        self._logger.warning(
            "Retrying chunking task %s",
            task_id,
            extra={
                "task_id": task_id,
                "operation_id": operation_id,
                "correlation_id": correlation_id,
                "retry_count": retry_count,
                "error": str(exc),
            },
        )

        if not self._redis_client:
            return error_code

        try:
            retry_state: dict[str, Any] = {
                "operation_id": operation_id,
                "correlation_id": correlation_id,
                "task_id": task_id,
                "retry_count": str(retry_count),
                "last_error": str(exc),
                "error_type": error_code,
                "timestamp": datetime.now(UTC).isoformat(),
            }
            self._redis_client.hset(f"operation:{operation_id}:retry_state", mapping=retry_state)
            self._redis_client.expire(f"operation:{operation_id}:retry_state", 86400)
        except Exception as redis_error:  # pragma: no cover - defensive
            self._logger.warning("Failed to save retry state: %s", redis_error)

        return error_code

    def classify_error(self, exc: Exception) -> str:
        """Return the stable error code for metrics/logging."""

        return str(self._error_classifier.as_code(exc))

    # ------------------------------------------------------------------
    # Resource monitoring helpers
    # ------------------------------------------------------------------
    async def check_resource_limits(
        self,
        *,
        operation_id: str,
        correlation_id: str,
    ) -> None:
        """Verify system-wide resource limits before processing."""

        if self._psutil is None:  # pragma: no cover - psutil should be available in runtime
            return

        memory = self._psutil.virtual_memory()
        if memory.percent > 90:
            recovery_action = await self._error_handler.handle_resource_exhaustion(
                operation_id=operation_id,
                resource_type=ResourceType.MEMORY,
                current_usage=memory.percent,
                limit=100,
            )
            if recovery_action.action == "fail":
                raise ChunkingResourceLimitError(
                    detail="System memory exhausted",
                    correlation_id=correlation_id,
                    resource_type=ResourceType.MEMORY,
                    current_usage=memory.percent,
                    limit=100,
                    operation_id=operation_id,
                )

        cpu_percent = self._psutil.cpu_percent(interval=0.1)
        if cpu_percent > 90:
            recovery_action = await self._error_handler.handle_resource_exhaustion(
                operation_id=operation_id,
                resource_type=ResourceType.CPU,
                current_usage=cpu_percent,
                limit=100,
            )
            if recovery_action.action == "wait_and_retry":
                await asyncio.sleep(recovery_action.wait_time or 30)

    async def monitor_resources(
        self,
        *,
        process: Any,
        operation_id: str,
        initial_memory: int,
        initial_cpu_time: float,
        correlation_id: str,
    ) -> None:
        """Monitor per-operation resource usage while processing batches."""

        if self._psutil is None:  # pragma: no cover
            return

        current_memory = process.memory_info().rss
        memory_increase = current_memory - initial_memory

        if self._memory_usage_gauge:
            self._memory_usage_gauge.labels(operation_id=operation_id).set(current_memory)

        if memory_increase > self._memory_limit_bytes:
            raise ChunkingMemoryError(
                detail="Operation memory usage exceeded limit",
                correlation_id=correlation_id,
                operation_id=operation_id,
                memory_used=current_memory,
                memory_limit=self._memory_limit_bytes,
            )

        current_cpu_time = process.cpu_times().user + process.cpu_times().system
        cpu_time_used = current_cpu_time - initial_cpu_time

        if cpu_time_used > self._cpu_time_limit_seconds:
            raise ChunkingTimeoutError(
                detail="Operation CPU time exceeded limit",
                correlation_id=correlation_id,
                operation_id=operation_id,
                elapsed_time=cpu_time_used,
                timeout_limit=self._cpu_time_limit_seconds,
            )

    async def calculate_batch_size(self) -> int:
        """Return an adaptive batch size based on current memory usage."""

        if self._psutil is None:  # pragma: no cover
            return 1

        memory = self._psutil.virtual_memory()
        batch_size = self._error_handler._calculate_adaptive_batch_size(
            current_usage=memory.percent,
            limit=100,
        )
        return int(batch_size)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _record_circuit_breaker_failure(self) -> None:
        self._circuit_breaker_failures += 1
        self._circuit_breaker_last_failure_time = self._time.time()

        if (
            self._circuit_breaker_state is CircuitBreakerState.CLOSED
            and self._circuit_breaker_failures >= self._failure_threshold
        ):
            self._circuit_breaker_state = CircuitBreakerState.OPEN
            self._logger.warning(
                "Circuit breaker opened after %s failures",
                self._circuit_breaker_failures,
            )

    def _send_to_dead_letter_queue(
        self,
        *,
        task_id: str,
        operation_id: str,
        correlation_id: str,
        error: Exception,
        error_type: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        retry_count: int,
    ) -> None:
        if not self._redis_client:
            self._logger.error("Cannot send to DLQ: Redis client not available")
            return

        try:
            dlq_entry = {
                "task_id": task_id,
                "operation_id": operation_id,
                "correlation_id": correlation_id,
                "error_type": error_type,
                "error_message": str(error),
                "error_class": type(error).__name__,
                "args": list(args),
                "kwargs": dict(kwargs),
                "timestamp": datetime.now(UTC).isoformat(),
                "retries": retry_count,
            }
            dlq_key = "chunking:dlq:tasks"
            self._redis_client.rpush(dlq_key, json.dumps(dlq_entry))
            self._redis_client.expire(dlq_key, self._dead_letter_ttl_seconds)
            self._logger.error(
                "Task %s sent to dead letter queue",
                task_id,
                extra={
                    "task_id": task_id,
                    "operation_id": operation_id,
                    "correlation_id": correlation_id,
                    "dlq_key": dlq_key,
                },
            )
        except Exception as redis_error:  # pragma: no cover - defensive logging path
            self._logger.error("Failed to send task to DLQ: %s", redis_error, exc_info=redis_error)


__all__ = [
    "ChunkingOperationManager",
    "CircuitBreakerState",
    "DEFAULT_CIRCUIT_BREAKER_FAILURE_THRESHOLD",
    "DEFAULT_CIRCUIT_BREAKER_RECOVERY_TIMEOUT",
    "DEFAULT_DEAD_LETTER_TTL_SECONDS",
]
