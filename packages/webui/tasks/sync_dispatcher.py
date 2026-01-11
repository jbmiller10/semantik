"""Sync dispatcher task for continuous collection synchronization.

This module provides the Celery Beat task that dispatches sync operations
for collections that are due for synchronization. Sync policy is managed at
the collection level, with fan-out to individual sources.
"""

import asyncio
import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from shared.database.models import CollectionStatus, OperationType
from shared.database.postgres_database import pg_connection_manager
from shared.database.repositories.collection_repository import CollectionRepository
from shared.database.repositories.collection_source_repository import CollectionSourceRepository
from shared.database.repositories.collection_sync_run_repository import CollectionSyncRunRepository
from shared.database.repositories.operation_repository import OperationRepository
from webui.celery_app import celery_app

logger = logging.getLogger(__name__)

# Maximum number of collections to dispatch in a single run
MAX_COLLECTIONS_PER_RUN = 20


async def _dispatch_due_syncs_async() -> dict[str, Any]:
    """Async implementation of the sync dispatcher.

    Dispatches sync runs for collections that are due based on collection-level
    sync policy (sync_mode='continuous' and sync_next_run_at <= now).

    Returns:
        Dictionary with dispatch statistics
    """
    stats: dict[str, Any] = {
        "checked_at": datetime.now(UTC).isoformat(),
        "collections_checked": 0,
        "collections_dispatched": 0,
        "sources_dispatched": 0,
        "collections_skipped": 0,
        "errors": [],
    }

    # Initialize database connection if needed
    if pg_connection_manager._engine is None:
        await pg_connection_manager.initialize()

    async with pg_connection_manager.get_session() as session:
        collection_repo = CollectionRepository(session)
        source_repo = CollectionSourceRepository(session)
        sync_run_repo = CollectionSyncRunRepository(session)
        operation_repo = OperationRepository(session)

        # Get collections due for sync (collection-level sync scheduling)
        # WHERE sync_mode='continuous' AND sync_paused_at IS NULL
        #   AND sync_next_run_at <= now() AND status IN (READY, DEGRADED)
        due_collections = await collection_repo.get_due_for_sync(limit=MAX_COLLECTIONS_PER_RUN)
        stats["collections_checked"] = len(due_collections)

        if not due_collections:
            logger.debug("No collections due for sync")
            return stats

        logger.info(f"Found {len(due_collections)} collections due for sync")

        for collection in due_collections:
            try:
                # Check for active operations (collection-level gating)
                active_ops = await operation_repo.get_active_operations(collection.id)
                if active_ops:
                    logger.debug(f"Skipping collection {collection.id} - has {len(active_ops)} active operations")
                    stats["collections_skipped"] += 1
                    continue

                # Get all sources for this collection
                sources, total = await source_repo.list_by_collection(
                    collection_id=collection.id,
                    offset=0,
                    limit=1000,  # Assume reasonable upper bound
                )

                if not sources:
                    logger.debug(f"Skipping collection {collection.id} - no sources configured")
                    stats["collections_skipped"] += 1
                    # Still update next_run to avoid checking again immediately
                    next_run = datetime.now(UTC) + timedelta(minutes=collection.sync_interval_minutes or 60)
                    await collection_repo.set_next_sync_run(collection.id, next_run)
                    await session.commit()
                    continue

                # Create sync run record
                sync_run = await sync_run_repo.create(
                    collection_id=collection.id,
                    triggered_by="scheduler",
                    expected_sources=len(sources),
                )

                # Update collection sync status
                now = datetime.now(UTC)
                await collection_repo.update_sync_status(
                    collection_uuid=collection.id,
                    status="running",
                    started_at=now,
                )

                # Create APPEND operations for each source
                operations = []
                for source in sources:
                    operation = await operation_repo.create(
                        collection_id=collection.id,
                        operation_type=OperationType.APPEND,
                        user_id=collection.owner_id,
                        config={
                            "source_id": source.id,
                            "source_type": source.source_type,
                            "source_path": source.source_path,
                            "source_config": source.source_config or {},
                            "sync_run_id": sync_run.id,  # CRITICAL: Link to sync run
                            "triggered_by": "scheduler",
                        },
                    )
                    operations.append(operation)

                # Update collection status to PROCESSING
                await collection_repo.update_status(collection.id, CollectionStatus.PROCESSING)

                # Calculate and set next sync run
                next_run = now + timedelta(minutes=collection.sync_interval_minutes or 60)
                await collection_repo.set_next_sync_run(collection.id, next_run)

                # Commit the transaction before dispatching tasks
                await session.commit()

                # Dispatch all operations
                for operation in operations:
                    celery_app.send_task(
                        "webui.tasks.process_collection_operation",
                        args=[operation.uuid],
                        task_id=str(uuid.uuid4()),
                    )

                logger.info(
                    f"Dispatched sync run {sync_run.id} for collection {collection.id} "
                    f"with {len(sources)} sources, next run at {next_run.isoformat()}"
                )
                stats["collections_dispatched"] += 1
                stats["sources_dispatched"] += len(sources)

            except Exception as e:
                logger.error("Error dispatching sync for collection %s: %s", collection.id, e, exc_info=True)
                stats["errors"].append(
                    {
                        "collection_id": collection.id,
                        "error": str(e),
                    }
                )
                # Rollback this collection's transaction and continue with others
                await session.rollback()

    return stats


@celery_app.task(name="webui.tasks.dispatch_due_syncs", bind=True)
def dispatch_due_syncs(self: Any) -> dict[str, Any]:  # noqa: ARG001
    """Celery task to dispatch sync operations for due collections.

    This task runs every 60 seconds via Celery Beat and:
    1. Queries collections where sync_mode='continuous' AND sync_paused_at IS NULL
       AND sync_next_run_at <= now AND status IN (READY, DEGRADED)
    2. Checks each collection for active operations (collection-level gating)
    3. Creates a CollectionSyncRun record for completion tracking
    4. Fans out APPEND operations for each source with sync_run_id
    5. Updates sync_next_run_at based on sync_interval_minutes

    Returns:
        Dictionary with dispatch statistics
    """
    logger.info("Running sync dispatcher task")

    try:
        # Run the async implementation
        # Reset connection manager to ensure fresh connections for new event loop
        pg_connection_manager.reset()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(_dispatch_due_syncs_async())
        finally:
            loop.close()

        logger.info(
            f"Sync dispatcher completed: "
            f"checked={result['collections_checked']}, "
            f"dispatched={result['collections_dispatched']} collections "
            f"({result['sources_dispatched']} sources), "
            f"skipped={result['collections_skipped']}, "
            f"errors={len(result['errors'])}"
        )
        return result

    except Exception as e:
        logger.error("Sync dispatcher task failed: %s", e, exc_info=True)
        raise
