"""Celery task for benchmark execution.

This module provides the run_benchmark Celery task which dispatches
benchmark evaluation to the BenchmarkExecutor service.
"""

from __future__ import annotations

import uuid
from importlib import import_module
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from shared.database.database import AsyncSessionLocal, ensure_async_sessionmaker
from shared.database.models import BenchmarkStatus, OperationStatus

from .utils import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_DELAY,
    OPERATION_HARD_TIME_LIMIT,
    OPERATION_SOFT_TIME_LIMIT,
    CeleryTaskWithOperationUpdates,
    celery_app,
    logger,
    resolve_awaitable_sync,
)

if TYPE_CHECKING:
    from types import ModuleType


def _tasks_namespace() -> ModuleType:
    """Return the top-level tasks module for accessing patched attributes."""
    return import_module("webui.tasks")


async def _resolve_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get or create the async session factory."""
    factory = AsyncSessionLocal
    if factory is None:
        factory = await ensure_async_sessionmaker()
    assert factory is not None
    return cast(async_sessionmaker[AsyncSession], factory)


@celery_app.task(
    bind=True,
    name="webui.tasks.benchmark.run_benchmark",
    max_retries=DEFAULT_MAX_RETRIES,
    default_retry_delay=DEFAULT_RETRY_DELAY,
    acks_late=True,
    soft_time_limit=OPERATION_SOFT_TIME_LIMIT,
    time_limit=OPERATION_HARD_TIME_LIMIT,
)
def run_benchmark(
    self: Any,
    operation_uuid: str,
    benchmark_id: str,
) -> dict[str, Any]:
    """Execute a benchmark evaluation.

    This task orchestrates the full benchmark execution:
    1. Loads the operation and benchmark records
    2. Initializes the BenchmarkExecutor with dependencies
    3. Executes all benchmark runs sequentially
    4. Updates operation and benchmark status on completion

    Args:
        self: Celery task instance (for retries and task_id)
        operation_uuid: UUID of the backing Operation record
        benchmark_id: UUID of the benchmark to execute

    Returns:
        Dictionary with execution results
    """
    # Extract Celery task_id for operation tracking
    task_id = self.request.id if hasattr(self, "request") and self.request else str(uuid.uuid4())
    return cast(dict[str, Any], resolve_awaitable_sync(_run_benchmark_async(operation_uuid, benchmark_id, task_id)))


async def _run_benchmark_async(
    operation_uuid: str,
    benchmark_id: str,
    task_id: str,
) -> dict[str, Any]:
    """Async implementation of benchmark execution.

    Args:
        operation_uuid: UUID of the backing Operation record
        benchmark_id: UUID of the benchmark to execute
        task_id: Celery task ID for operation tracking

    Returns:
        Dictionary with execution results
    """
    tasks_ns = _tasks_namespace()
    log = getattr(tasks_ns, "logger", logger)

    log.info("Starting benchmark %s with operation %s (task_id=%s)", benchmark_id, operation_uuid, task_id)

    session_factory = await _resolve_session_factory()

    # Import repositories and services
    from shared.database.repositories.benchmark_dataset_repository import BenchmarkDatasetRepository
    from shared.database.repositories.benchmark_repository import BenchmarkRepository
    from shared.database.repositories.collection_repository import CollectionRepository
    from shared.database.repositories.operation_repository import OperationRepository
    from webui.services.benchmark_executor import BenchmarkExecutor
    from webui.services.search_service import SearchService

    result: dict[str, Any] = {
        "benchmark_id": benchmark_id,
        "operation_uuid": operation_uuid,
        "status": "unknown",
        "error": None,
    }

    async with session_factory() as session:
        # Initialize repositories
        operation_repo = OperationRepository(session)
        benchmark_repo = BenchmarkRepository(session)
        benchmark_dataset_repo = BenchmarkDatasetRepository(session)
        collection_repo = CollectionRepository(session)

        # Load and update operation status
        operation = await operation_repo.get_by_uuid(operation_uuid)
        if not operation:
            log.error("Operation not found: %s", operation_uuid)
            result["error"] = f"Operation not found: {operation_uuid}"
            result["status"] = "failed"
            return result

        # Store task_id for operation tracking (enables task cancellation)
        await operation_repo.set_task_id(operation_uuid, task_id)
        log.info("Set task_id %s for operation %s", task_id, operation_uuid)

        await operation_repo.update_status(
            operation_uuid,
            OperationStatus.PROCESSING,
        )
        await session.commit()

        # Initialize progress reporter
        progress_reporter = CeleryTaskWithOperationUpdates(operation_uuid)
        progress_reporter.set_user_id(cast(int | None, operation.user_id))

        try:
            # Initialize search service
            search_service = SearchService(
                db_session=session,
                collection_repo=collection_repo,
            )

            # Initialize executor
            executor = BenchmarkExecutor(
                db_session=session,
                benchmark_repo=benchmark_repo,
                benchmark_dataset_repo=benchmark_dataset_repo,
                collection_repo=collection_repo,
                search_service=search_service,
                progress_reporter=progress_reporter,
            )

            # Execute benchmark
            async with progress_reporter:
                exec_result = await executor.execute_benchmark(benchmark_id)

            result.update(exec_result)

            # Update operation status based on benchmark result
            benchmark_status = exec_result.get("status")
            if benchmark_status == BenchmarkStatus.COMPLETED.value:
                await operation_repo.update_status(
                    operation_uuid,
                    OperationStatus.COMPLETED,
                )
            elif benchmark_status == BenchmarkStatus.CANCELLED.value:
                await operation_repo.update_status(
                    operation_uuid,
                    OperationStatus.CANCELLED,
                )
            else:
                await operation_repo.update_status(
                    operation_uuid,
                    OperationStatus.FAILED,
                    error_message=exec_result.get("error"),
                )

            await session.commit()

            log.info(
                "Benchmark %s completed with status %s",
                benchmark_id,
                benchmark_status,
            )

        except Exception as exc:
            log.error(
                "Benchmark %s failed with error: %s",
                benchmark_id,
                exc,
                exc_info=True,
            )

            result["status"] = "failed"
            result["error"] = str(exc)

            # Update benchmark status to FAILED
            try:
                await benchmark_repo.update_status(
                    benchmark_id,
                    BenchmarkStatus.FAILED,
                )
            except Exception as update_exc:
                log.warning(
                    "Failed to update benchmark status: %s",
                    update_exc,
                )

            # Update operation status to FAILED
            try:
                await operation_repo.update_status(
                    operation_uuid,
                    OperationStatus.FAILED,
                    error_message=str(exc)[:1000],
                )
                await session.commit()
            except Exception as op_update_exc:
                log.warning(
                    "Failed to update operation status: %s",
                    op_update_exc,
                )

        finally:
            # Ensure progress reporter is closed
            progress_reporter.close()

    return result


__all__ = ["run_benchmark"]
