"""Celery task for benchmark dataset mapping resolution.

This module provides the resolve_mapping Celery task which resolves benchmark dataset
doc_refs against a mapped collection's documents, updating BenchmarkRelevance rows and
emitting progress events on the backing Operation channel.
"""

from __future__ import annotations

import uuid
from importlib import import_module
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from shared.database.database import AsyncSessionLocal, ensure_async_sessionmaker
from shared.database.models import OperationStatus

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
    name="webui.tasks.benchmark_mapping.resolve_mapping",
    max_retries=DEFAULT_MAX_RETRIES,
    default_retry_delay=DEFAULT_RETRY_DELAY,
    acks_late=True,
    soft_time_limit=OPERATION_SOFT_TIME_LIMIT,
    time_limit=OPERATION_HARD_TIME_LIMIT,
)
def resolve_mapping(
    self: Any,
    operation_uuid: str,
    mapping_id: int,
    user_id: int,
) -> dict[str, Any]:
    """Resolve benchmark mapping document references.

    Args:
        self: Celery task instance (for retries and task_id)
        operation_uuid: UUID of the backing Operation record
        mapping_id: ID of the dataset-collection mapping to resolve
        user_id: ID of the user who initiated the resolution (auth / progress channel)

    Returns:
        Dictionary with resolution results
    """
    task_id = self.request.id if hasattr(self, "request") and self.request else str(uuid.uuid4())
    return cast(
        dict[str, Any],
        resolve_awaitable_sync(_resolve_mapping_async(operation_uuid, mapping_id, user_id, task_id)),
    )


async def _resolve_mapping_async(
    operation_uuid: str,
    mapping_id: int,
    user_id: int,
    task_id: str,
) -> dict[str, Any]:
    tasks_ns = _tasks_namespace()
    log = getattr(tasks_ns, "logger", logger)

    log.info("Starting mapping resolution %s with operation %s (task_id=%s)", mapping_id, operation_uuid, task_id)

    session_factory = await _resolve_session_factory()

    from shared.database.repositories.benchmark_dataset_repository import BenchmarkDatasetRepository
    from shared.database.repositories.collection_repository import CollectionRepository
    from shared.database.repositories.document_repository import DocumentRepository
    from shared.database.repositories.operation_repository import OperationRepository
    from webui.services.benchmark_dataset_service import BenchmarkDatasetService

    result: dict[str, Any] = {
        "mapping_id": mapping_id,
        "operation_uuid": operation_uuid,
        "status": "unknown",
        "error": None,
    }

    async with session_factory() as session:
        operation_repo = OperationRepository(session)
        benchmark_dataset_repo = BenchmarkDatasetRepository(session)
        collection_repo = CollectionRepository(session)
        document_repo = DocumentRepository(session)

        operation = await operation_repo.get_by_uuid(operation_uuid)
        if not operation:
            log.error("Operation not found: %s", operation_uuid)
            result["error"] = f"Operation not found: {operation_uuid}"
            result["status"] = "failed"
            return result

        await operation_repo.set_task_id(operation_uuid, task_id)
        await operation_repo.update_status(operation_uuid, OperationStatus.PROCESSING)
        await session.commit()

        progress_reporter = CeleryTaskWithOperationUpdates(operation_uuid)
        progress_reporter.set_user_id(user_id)
        progress_reporter.set_collection_id(cast(str | None, operation.collection_id))

        service = BenchmarkDatasetService(
            db_session=session,
            benchmark_dataset_repo=benchmark_dataset_repo,
            collection_repo=collection_repo,
            document_repo=document_repo,
            operation_repo=operation_repo,
        )

        try:
            async with progress_reporter:
                resolution_result = await service.resolve_mapping_with_progress(
                    mapping_id=mapping_id,
                    user_id=user_id,
                    operation_uuid=operation_uuid,
                    progress_reporter=progress_reporter,
                )

            await operation_repo.update_status(operation_uuid, OperationStatus.COMPLETED)
            await session.commit()

            result.update(resolution_result)
            result["status"] = "completed"
            log.info("Mapping resolution %s completed (operation=%s)", mapping_id, operation_uuid)

        except Exception as exc:
            log.error("Mapping resolution %s failed: %s", mapping_id, exc, exc_info=True)
            result["status"] = "failed"
            result["error"] = str(exc)

            try:
                await operation_repo.update_status(
                    operation_uuid, OperationStatus.FAILED, error_message=str(exc)[:1000]
                )
                await session.commit()
            except Exception as op_exc:  # pragma: no cover - defensive logging
                log.warning("Failed to update operation status: %s", op_exc)

        finally:
            progress_reporter.close()

    return result


__all__ = ["resolve_mapping"]
