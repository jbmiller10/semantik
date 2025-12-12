"""Sync dispatcher task for continuous source synchronization.

This module provides the Celery Beat task that dispatches sync operations
for sources that are due for synchronization.
"""

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from shared.database.models import CollectionStatus, OperationType
from shared.database.postgres_database import pg_connection_manager
from shared.database.repositories.collection_repository import CollectionRepository
from shared.database.repositories.collection_source_repository import CollectionSourceRepository
from shared.database.repositories.operation_repository import OperationRepository
from webui.celery_app import celery_app

logger = logging.getLogger(__name__)

# Maximum number of sources to dispatch in a single run
MAX_SOURCES_PER_RUN = 50


async def _dispatch_due_syncs_async() -> dict[str, Any]:
    """Async implementation of the sync dispatcher.

    Returns:
        Dictionary with dispatch statistics
    """
    stats = {
        "checked_at": datetime.now(UTC).isoformat(),
        "due_count": 0,
        "dispatched_count": 0,
        "skipped_count": 0,
        "errors": [],
    }

    # Initialize database connection if needed
    if pg_connection_manager._engine is None:
        await pg_connection_manager.initialize()

    async with pg_connection_manager.session() as session:
        source_repo = CollectionSourceRepository(session)
        operation_repo = OperationRepository(session)
        collection_repo = CollectionRepository(session)

        # Get sources due for sync
        due_sources = await source_repo.get_due_for_sync(limit=MAX_SOURCES_PER_RUN)
        stats["due_count"] = len(due_sources)

        if not due_sources:
            logger.debug("No sources due for sync")
            return stats

        logger.info(f"Found {len(due_sources)} sources due for sync")

        for source in due_sources:
            try:
                # Check if collection exists and is in a valid state
                collection = await collection_repo.get_by_id(source.collection_id)
                if collection is None:
                    logger.warning(f"Collection {source.collection_id} not found for source {source.id}")
                    stats["skipped_count"] += 1
                    stats["errors"].append(
                        {
                            "source_id": source.id,
                            "error": f"Collection {source.collection_id} not found",
                        }
                    )
                    continue

                # Check collection status - only dispatch for READY or DEGRADED
                if collection.status not in (CollectionStatus.READY, CollectionStatus.DEGRADED):
                    logger.debug(
                        f"Skipping source {source.id} - collection {source.collection_id} "
                        f"status is {collection.status}"
                    )
                    stats["skipped_count"] += 1
                    continue

                # Check for active operations on the collection (operation gating)
                active_ops = await operation_repo.get_active_operations(source.collection_id)
                if active_ops:
                    logger.debug(
                        f"Skipping source {source.id} - collection {source.collection_id} "
                        f"has {len(active_ops)} active operations"
                    )
                    stats["skipped_count"] += 1
                    continue

                # Create APPEND operation for the source
                operation = await operation_repo.create(
                    collection_id=source.collection_id,
                    operation_type=OperationType.APPEND,
                    user_id=collection.user_id,
                    config={
                        "source_id": source.id,
                        "source_type": source.source_type,
                        "source_path": source.source_path,
                        "source_config": source.source_config or {},
                        "triggered_by": "sync_dispatcher",
                    },
                )

                # Update last_run_started_at to track sync initiation
                # (The actual status will be updated by the ingestion task on completion)
                await source_repo.update(
                    source_id=source.id,
                    last_run_started_at=datetime.now(UTC),
                )

                # Calculate next run time
                next_run = datetime.now(UTC) + timedelta(minutes=source.interval_minutes or 60)
                await source_repo.set_next_run(source.id, next_run)

                # Commit the transaction before dispatching the task
                await session.commit()

                # Dispatch the ingestion task
                from webui.tasks.ingestion import process_collection_operation

                process_collection_operation.delay(str(operation.id))

                logger.info(
                    f"Dispatched sync for source {source.id} (collection {source.collection_id}), "
                    f"operation {operation.id}, next run at {next_run.isoformat()}"
                )
                stats["dispatched_count"] += 1

            except Exception as e:
                logger.error(f"Error dispatching sync for source {source.id}: {e}")
                stats["errors"].append(
                    {
                        "source_id": source.id,
                        "error": str(e),
                    }
                )
                # Rollback this source's transaction and continue with others
                await session.rollback()

    return stats


@celery_app.task(name="webui.tasks.dispatch_due_syncs", bind=True)
def dispatch_due_syncs(self: Any) -> dict[str, Any]:  # noqa: ARG001
    """Celery task to dispatch sync operations for due sources.

    This task runs every 60 seconds via Celery Beat and:
    1. Queries sources where next_run_at <= now AND paused_at IS NULL
    2. Checks each source's collection for active operations (operation gating)
    3. Creates APPEND operations for eligible sources
    4. Updates next_run_at based on interval_minutes

    Returns:
        Dictionary with dispatch statistics
    """
    logger.info("Running sync dispatcher task")

    try:
        # Run the async implementation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(_dispatch_due_syncs_async())
        finally:
            loop.close()

        logger.info(
            f"Sync dispatcher completed: "
            f"due={result['due_count']}, "
            f"dispatched={result['dispatched_count']}, "
            f"skipped={result['skipped_count']}, "
            f"errors={len(result['errors'])}"
        )
        return result

    except Exception as e:
        logger.error(f"Sync dispatcher task failed: {e}")
        raise
