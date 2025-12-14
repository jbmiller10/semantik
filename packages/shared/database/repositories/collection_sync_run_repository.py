"""Repository implementation for CollectionSyncRun model."""

import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from shared.database.exceptions import DatabaseOperationError, EntityNotFoundError
from shared.database.models import CollectionSyncRun

logger = logging.getLogger(__name__)


class CollectionSyncRunRepository:
    """Repository for CollectionSyncRun model operations.

    Manages sync run lifecycle including creation, status tracking,
    source completion aggregation, and run completion.
    """

    def __init__(self, session: AsyncSession):
        """Initialize with database session.

        Args:
            session: AsyncSession instance for database operations
        """
        self.session = session

    async def create(
        self,
        collection_id: str,
        triggered_by: str,
        expected_sources: int,
        meta: dict[str, Any] | None = None,
    ) -> CollectionSyncRun:
        """Create a new sync run record.

        Args:
            collection_id: UUID of the collection
            triggered_by: How the sync was initiated ('scheduler', 'manual')
            expected_sources: Number of sources expected to complete
            meta: Optional metadata

        Returns:
            Created CollectionSyncRun instance

        Raises:
            DatabaseOperationError: For database errors
        """
        try:
            sync_run = CollectionSyncRun(
                collection_id=collection_id,
                triggered_by=triggered_by,
                started_at=datetime.now(UTC),
                status="running",
                expected_sources=expected_sources,
                completed_sources=0,
                failed_sources=0,
                partial_sources=0,
                meta=meta or {},
            )

            self.session.add(sync_run)
            await self.session.flush()

            logger.info(
                f"Created sync run {sync_run.id} for collection {collection_id} "
                f"(triggered_by={triggered_by}, expected_sources={expected_sources})"
            )
            return sync_run

        except Exception as e:
            logger.error(f"Failed to create sync run for collection {collection_id}: {e}")
            raise DatabaseOperationError("create", "collection_sync_run", str(e)) from e

    async def get_by_id(self, run_id: int) -> CollectionSyncRun | None:
        """Get a sync run by ID.

        Args:
            run_id: Primary key of the sync run

        Returns:
            CollectionSyncRun instance or None if not found
        """
        try:
            result = await self.session.execute(select(CollectionSyncRun).where(CollectionSyncRun.id == run_id))
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Failed to get sync run {run_id}: {e}")
            raise DatabaseOperationError("get", "collection_sync_run", str(e)) from e

    async def get_active_run(self, collection_id: str) -> CollectionSyncRun | None:
        """Get active (running) sync run for a collection.

        Args:
            collection_id: UUID of the collection

        Returns:
            Active CollectionSyncRun instance or None if none running
        """
        try:
            result = await self.session.execute(
                select(CollectionSyncRun).where(
                    CollectionSyncRun.collection_id == collection_id,
                    CollectionSyncRun.status == "running",
                )
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Failed to get active sync run for collection {collection_id}: {e}")
            raise DatabaseOperationError("get", "collection_sync_run", str(e)) from e

    async def update_source_completion(
        self,
        run_id: int,
        status: str,
    ) -> CollectionSyncRun:
        """Increment the appropriate counter when a source completes.

        Args:
            run_id: ID of the sync run
            status: Source completion status ('success', 'failed', 'partial')

        Returns:
            Updated CollectionSyncRun instance

        Raises:
            EntityNotFoundError: If sync run not found
            DatabaseOperationError: For database errors
        """
        try:
            sync_run = await self.get_by_id(run_id)
            if not sync_run:
                raise EntityNotFoundError("collection_sync_run", str(run_id))

            # Increment the appropriate counter based on source status
            if status == "success":
                sync_run.completed_sources += 1
            elif status == "failed":
                sync_run.failed_sources += 1
            elif status == "partial":
                sync_run.partial_sources += 1
            else:
                # Treat unknown status as partial
                sync_run.partial_sources += 1
                logger.warning(f"Unknown source status '{status}' for sync run {run_id}, treating as partial")

            await self.session.flush()

            logger.debug(
                f"Updated sync run {run_id} source completion: "
                f"completed={sync_run.completed_sources}, "
                f"failed={sync_run.failed_sources}, "
                f"partial={sync_run.partial_sources}"
            )
            return sync_run

        except EntityNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to update source completion for sync run {run_id}: {e}")
            raise DatabaseOperationError("update", "collection_sync_run", str(e)) from e

    async def complete_run(
        self,
        run_id: int,
        error_summary: str | None = None,
    ) -> CollectionSyncRun:
        """Mark a sync run as complete and calculate final status.

        The final status is determined by:
        - success: All sources completed successfully (failed=0, partial=0)
        - partial: Some sources failed or had partial success
        - failed: All sources failed (completed=0, partial=0)

        Args:
            run_id: ID of the sync run
            error_summary: Optional error summary

        Returns:
            Updated CollectionSyncRun instance

        Raises:
            EntityNotFoundError: If sync run not found
            DatabaseOperationError: For database errors
        """
        try:
            sync_run = await self.get_by_id(run_id)
            if not sync_run:
                raise EntityNotFoundError("collection_sync_run", str(run_id))

            # Calculate final status
            if sync_run.failed_sources == sync_run.expected_sources:
                final_status = "failed"
            elif sync_run.failed_sources > 0 or sync_run.partial_sources > 0:
                final_status = "partial"
            else:
                final_status = "success"

            sync_run.status = final_status
            sync_run.completed_at = datetime.now(UTC)
            if error_summary:
                sync_run.error_summary = error_summary

            await self.session.flush()

            logger.info(
                f"Completed sync run {run_id} with status={final_status} "
                f"(completed={sync_run.completed_sources}, "
                f"failed={sync_run.failed_sources}, "
                f"partial={sync_run.partial_sources})"
            )
            return sync_run

        except EntityNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to complete sync run {run_id}: {e}")
            raise DatabaseOperationError("update", "collection_sync_run", str(e)) from e

    async def list_for_collection(
        self,
        collection_id: str,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[CollectionSyncRun], int]:
        """List sync runs for a collection with pagination.

        Args:
            collection_id: UUID of the collection
            offset: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            Tuple of (list of sync runs, total count)

        Raises:
            DatabaseOperationError: For database errors
        """
        try:
            # Get total count
            count_result = await self.session.execute(
                select(func.count(CollectionSyncRun.id)).where(CollectionSyncRun.collection_id == collection_id)
            )
            total = count_result.scalar() or 0

            # Get paginated results
            result = await self.session.execute(
                select(CollectionSyncRun)
                .where(CollectionSyncRun.collection_id == collection_id)
                .order_by(desc(CollectionSyncRun.started_at))
                .offset(offset)
                .limit(limit)
            )
            sync_runs = list(result.scalars().all())

            return sync_runs, total

        except Exception as e:
            logger.error(f"Failed to list sync runs for collection {collection_id}: {e}")
            raise DatabaseOperationError("list", "collection_sync_run", str(e)) from e

    async def get_latest_for_collection(self, collection_id: str) -> CollectionSyncRun | None:
        """Get the most recent sync run for a collection.

        Args:
            collection_id: UUID of the collection

        Returns:
            Most recent CollectionSyncRun instance or None
        """
        try:
            result = await self.session.execute(
                select(CollectionSyncRun)
                .where(CollectionSyncRun.collection_id == collection_id)
                .order_by(desc(CollectionSyncRun.started_at))
                .limit(1)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Failed to get latest sync run for collection {collection_id}: {e}")
            raise DatabaseOperationError("get", "collection_sync_run", str(e)) from e
