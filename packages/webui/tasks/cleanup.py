"""Cleanup and maintenance Celery tasks.

These tasks cover lifecycle management of Celery results, Qdrant collections, and
periodic maintenance such as refreshing materialized views or monitoring
partitions. They import shared helpers from ``webui.tasks.utils`` to stay
aligned with the refactored module structure.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from importlib import import_module
from typing import TYPE_CHECKING, Any, cast

from shared.database.database import AsyncSessionLocal, ensure_async_sessionmaker
from shared.metrics.collection_metrics import QdrantOperationTimer, record_metric_safe

if TYPE_CHECKING:
    from types import ModuleType

from .utils import (
    DEFAULT_DAYS_TO_KEEP,
    celery_app,
    logger,
    resolve_awaitable_sync,
    resolve_qdrant_manager,
    resolve_qdrant_manager_class,
)


def _tasks_namespace() -> ModuleType:
    """Return the top-level tasks module for accessing patched attributes."""
    return import_module("webui.tasks")


async def _resolve_session_factory() -> Any:
    factory = AsyncSessionLocal
    if factory is None:
        factory = await ensure_async_sessionmaker()
    return factory


@celery_app.task(name="webui.tasks.cleanup_old_results")
def cleanup_old_results(days_to_keep: int = DEFAULT_DAYS_TO_KEEP) -> dict[str, Any]:
    """Clean up old Celery results and operation records."""
    stats: dict[str, Any] = {"celery_results_deleted": 0, "old_operations_marked": 0, "errors": []}

    tasks_ns = _tasks_namespace()
    log = getattr(tasks_ns, "logger", logger)
    datetime_module = getattr(tasks_ns, "datetime", datetime)

    try:
        cutoff_time = datetime_module.now(UTC) - timedelta(days=days_to_keep)

        log.info("Starting cleanup of results older than %s days", days_to_keep)
        log.info("Cleanup of operations older than %s is handled by operation lifecycle", cutoff_time)

        log.info("Cleanup completed: %s", stats)
        return stats

    except Exception as exc:  # pragma: no cover - defensive logging
        log.error("Cleanup task failed: %s", exc)
        stats["errors"].append(str(exc))
        return stats


@celery_app.task(name="webui.tasks.refresh_collection_chunking_stats")
def refresh_collection_chunking_stats() -> dict[str, Any]:
    """Refresh the collection_chunking_stats materialized view."""
    from sqlalchemy import text

    stats: dict[str, Any] = {"status": "success", "duration_seconds": 0.0, "error": None}
    start_time = time.time()

    tasks_ns = _tasks_namespace()
    log = getattr(tasks_ns, "logger", logger)

    try:
        log.info("Starting refresh of collection_chunking_stats materialized view")

        async def _refresh_view() -> None:
            session_factory = await _resolve_session_factory()
            async with session_factory() as session:
                await session.execute(text("SELECT refresh_collection_chunking_stats()"))
                await session.commit()

        resolve_awaitable_sync(_refresh_view())

        stats["duration_seconds"] = time.time() - start_time
        log.info(
            "Successfully refreshed collection_chunking_stats in %.2f seconds",
            stats["duration_seconds"],
        )

    except Exception as exc:
        stats["status"] = "failed"
        stats["error"] = str(exc)
        stats["duration_seconds"] = time.time() - start_time
        log.error(f"Failed to refresh collection_chunking_stats: {exc}")
        raise

    return stats


@celery_app.task(name="webui.tasks.monitor_partition_health")
def monitor_partition_health() -> dict[str, Any]:
    """Monitor partition health and alert on imbalances."""
    from webui.services.partition_monitoring_service import PartitionMonitoringService

    tasks_ns = _tasks_namespace()
    log = getattr(tasks_ns, "logger", logger)

    try:
        log.info("Starting partition health monitoring")

        async def _check_health() -> dict[str, Any]:
            session_factory = await _resolve_session_factory()
            async with session_factory() as session:
                service = PartitionMonitoringService(session)
                monitoring_result = await service.check_partition_health()

                return {
                    "status": monitoring_result.status,
                    "timestamp": monitoring_result.timestamp,
                    "alerts": monitoring_result.alerts,
                    "metrics": monitoring_result.metrics,
                    "error": monitoring_result.error,
                }

        results = cast(dict[str, Any], resolve_awaitable_sync(_check_health()))

    except Exception as exc:  # pragma: no cover - defensive logging
        log.error(f"Partition health monitoring failed: {exc}")
        raise

    return results


@celery_app.task(
    name="webui.tasks.cleanup_old_collections",
    max_retries=3,
    default_retry_delay=60,
    retry_backoff=True,
    retry_backoff_max=600,
)
def cleanup_old_collections(old_collection_names: list[str], collection_id: str) -> dict[str, Any]:
    """Clean up old Qdrant collections after a successful reindex."""
    stats: dict[str, Any] = {
        "collections_deleted": 0,
        "collections_failed": 0,
        "errors": [],
        "collection_id": collection_id,
    }

    if not old_collection_names:
        logger.info("No old collections to clean up for collection %s", collection_id)
        return stats

    logger.info(
        "Starting cleanup of %s old collections for collection %s",
        len(old_collection_names),
        collection_id,
    )

    try:
        manager = resolve_qdrant_manager()
        qdrant_client = manager.get_client()

        for collection_name in old_collection_names:
            try:
                with QdrantOperationTimer("check_collection_exists"):
                    collections_response = qdrant_client.get_collections()
                    collection_entries = getattr(collections_response, "collections", []) or []
                    exists = any(getattr(col, "name", None) == collection_name for col in collection_entries)

                if not exists:
                    logger.warning("Collection %s does not exist, skipping", collection_name)
                    continue

                with QdrantOperationTimer("delete_old_collection"):
                    qdrant_client.delete_collection(collection_name)

                stats["collections_deleted"] += 1
                logger.info("Successfully deleted old collection: %s", collection_name)

            except Exception as exc:
                error_msg = f"Failed to delete collection {collection_name}: {str(exc)}"
                logger.error(error_msg)
                stats["collections_failed"] += 1
                stats["errors"].append(error_msg)

        logger.info(
            "Cleanup completed for collection %s: deleted=%s, failed=%s",
            collection_id,
            stats["collections_deleted"],
            stats["collections_failed"],
        )

        status = "success" if stats["collections_failed"] == 0 else "partial"
        record_metric_safe("collection_cleanup_total", {"status": status})

        return stats

    except Exception as exc:
        logger.error("Cleanup task failed for collection %s: %s", collection_id, exc)
        stats["errors"].append(str(exc))
        record_metric_safe("collection_cleanup_total", {"status": "failed"})
        return stats


@celery_app.task(
    name="webui.tasks.cleanup_qdrant_collections",
    max_retries=3,
    default_retry_delay=60,
    retry_backoff=True,
    retry_backoff_max=600,
)
def cleanup_qdrant_collections(collection_names: list[str], staging_age_hours: int = 1) -> dict[str, Any]:
    """Clean up orphaned Qdrant collections with enhanced safety checks."""
    stats: dict[str, Any] = {
        "collections_deleted": 0,
        "collections_skipped": 0,
        "collections_failed": 0,
        "safety_checks": {},
        "errors": [],
        "timestamp": datetime.now(UTC).isoformat(),
    }

    tasks_ns = _tasks_namespace()
    log = getattr(tasks_ns, "logger", logger)

    if not collection_names:
        log.info("No collections provided for cleanup")
        return stats

    log.info("Starting enhanced cleanup of %s collections", len(collection_names))

    try:
        active_collections = resolve_awaitable_sync(_get_active_collections())
        stats["safety_checks"]["active_collections_found"] = len(active_collections)

        manager = resolve_qdrant_manager()
        qdrant_client = manager.get_client()
        qdrant_manager_class = resolve_qdrant_manager_class()
        qdrant_manager_instance = qdrant_manager_class(qdrant_client)

        deletions_for_audit: list[tuple[str, int]] = []

        for collection_name in collection_names:
            try:
                if collection_name.startswith("_"):
                    log.warning("Skipping system collection: %s", collection_name)
                    stats["collections_skipped"] += 1
                    stats["safety_checks"][collection_name] = "system_collection"
                    continue

                with QdrantOperationTimer("check_collection_exists"):
                    exists_result = qdrant_manager_instance.collection_exists(collection_name)
                    exists = bool(resolve_awaitable_sync(exists_result))
                    log.debug("collection_exists(%s) -> %s", collection_name, exists)
                    if not exists:
                        log.info("Collection %s does not exist, skipping", collection_name)
                        stats["collections_skipped"] += 1
                        stats["safety_checks"][collection_name] = "not_found"
                        continue

                if collection_name in active_collections:
                    log.warning("Skipping active collection: %s", collection_name)
                    stats["collections_skipped"] += 1
                    stats["safety_checks"][collection_name] = "active_collection"
                    continue

                collection_info = resolve_awaitable_sync(qdrant_manager_instance.get_collection_info(collection_name))
                vector_count = collection_info.vectors_count if collection_info else 0

                staging_is_old = bool(
                    resolve_awaitable_sync(
                        qdrant_manager_instance._is_staging_collection_old(collection_name, hours=staging_age_hours)
                    )
                )

                if collection_name.startswith("staging_") and not staging_is_old:
                    log.warning("Skipping recent staging collection: %s", collection_name)
                    stats["collections_skipped"] += 1
                    stats["safety_checks"][collection_name] = "staging_too_recent"
                    continue

                log.info("Deleting collection %s with %s vectors", collection_name, vector_count)

                with QdrantOperationTimer("delete_collection_safe"):
                    qdrant_client.delete_collection(collection_name)

                stats["collections_deleted"] += 1
                stats["safety_checks"][collection_name] = "deleted"
                deletions_for_audit.append((collection_name, vector_count))
                time.sleep(0.1)

            except Exception as exc:
                error_msg = f"Failed to delete collection {collection_name}: {str(exc)}"
                log.error(error_msg)
                stats["collections_failed"] += 1
                stats["errors"].append(error_msg)
                stats["safety_checks"][collection_name] = f"error: {str(exc)}"

        if deletions_for_audit:
            resolve_awaitable_sync(_audit_collection_deletions_batch(deletions_for_audit))

        log.info(
            "Enhanced cleanup completed: deleted=%s, skipped=%s, failed=%s",
            stats["collections_deleted"],
            stats["collections_skipped"],
            stats["collections_failed"],
        )

        status = "success" if stats["collections_failed"] == 0 else "partial"
        record_metric_safe("qdrant_cleanup_total", {"status": status, "type": "enhanced"})

        return stats

    except Exception as exc:
        log.error("Enhanced cleanup task failed: %s", exc)
        stats["errors"].append(str(exc))
        record_metric_safe("qdrant_cleanup_total", {"status": "failed", "type": "enhanced"})

        return stats


async def _get_active_collections() -> set[str]:
    """Get all active Qdrant collection names from the database."""
    from shared.database.repositories.collection_repository import CollectionRepository

    session_factory = await _resolve_session_factory()
    async with session_factory() as session:
        collection_repo = CollectionRepository(session)

        collections = await collection_repo.list_all()

        active_collections = set()
        for collection in collections:
            if collection.get("vector_store_name"):
                active_collections.add(collection["vector_store_name"])

            if collection.get("qdrant_collections"):
                active_collections.update(collection["qdrant_collections"])

            if collection.get("qdrant_staging"):
                staging_info = collection["qdrant_staging"]
                if isinstance(staging_info, dict) and "collection_name" in staging_info:
                    active_collections.add(staging_info["collection_name"])

        return active_collections


async def _audit_collection_deletion(collection_name: str, vector_count: int) -> None:
    """Create audit log entry for collection deletion."""
    try:
        from shared.database.models import CollectionAuditLog

        session_factory = await _resolve_session_factory()
        async with session_factory() as session:
            audit_log = CollectionAuditLog(
                collection_id=None,
                operation_id=None,
                user_id=None,
                action="qdrant_collection_deleted",
                details={
                    "collection_name": collection_name,
                    "vector_count": vector_count,
                    "deleted_at": datetime.now(UTC).isoformat(),
                },
            )
            session.add(audit_log)
            await session.commit()
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Failed to create audit log for collection deletion: %s", exc)


async def _audit_collection_deletions_batch(deletions: list[tuple[str, int]]) -> None:
    """Create audit log entries for multiple collection deletions in a single transaction."""
    if not deletions:
        return

    try:
        from shared.database.models import CollectionAuditLog

        session_factory = await _resolve_session_factory()
        async with session_factory() as session:
            deleted_at = datetime.now(UTC).isoformat()

            for collection_name, vector_count in deletions:
                audit_log = CollectionAuditLog(
                    collection_id=None,
                    operation_id=None,
                    user_id=None,
                    action="qdrant_collection_deleted",
                    details={
                        "collection_name": collection_name,
                        "vector_count": vector_count,
                        "deleted_at": deleted_at,
                    },
                )
                session.add(audit_log)

            await session.commit()
            logger.info("Created %s audit log entries for collection deletions", len(deletions))
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Failed to create batch audit logs for collection deletions: %s", exc)


__all__ = [
    "cleanup_old_results",
    "cleanup_old_collections",
    "cleanup_qdrant_collections",
    "refresh_collection_chunking_stats",
    "monitor_partition_health",
]
