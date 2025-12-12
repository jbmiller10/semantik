"""Repository implementation for CollectionSource model."""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import and_, delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from shared.database.exceptions import (
    DatabaseOperationError,
    EntityAlreadyExistsError,
    EntityNotFoundError,
    ValidationError,
)
from shared.database.models import CollectionSource

logger = logging.getLogger(__name__)

# Minimum sync interval in minutes
MIN_SYNC_INTERVAL_MINUTES = 15


class CollectionSourceRepository:
    """Repository for CollectionSource model operations."""

    def __init__(self, session: AsyncSession):
        """Initialize with database session.

        Args:
            session: AsyncSession instance for database operations
        """
        self.session = session

    async def create(
        self,
        collection_id: str,
        source_type: str,
        source_path: str,
        source_config: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
        sync_mode: str = "one_time",
        interval_minutes: int | None = None,
    ) -> CollectionSource:
        """Create a new collection source.

        Args:
            collection_id: UUID of the parent collection
            source_type: Type of source (directory, web, slack, etc.)
            source_path: Display path or identifier for the source
            source_config: Connector-specific configuration
            meta: Optional metadata
            sync_mode: 'one_time' or 'continuous' (default: 'one_time')
            interval_minutes: Sync interval for continuous mode (min 15)

        Returns:
            Created CollectionSource instance

        Raises:
            EntityAlreadyExistsError: If source with same path already exists
            ValidationError: If required fields are invalid
            DatabaseOperationError: For database errors
        """
        try:
            # Validate inputs
            if not collection_id:
                raise ValidationError("Collection ID is required", "collection_id")
            if not source_type:
                raise ValidationError("Source type is required", "source_type")
            if not source_path:
                raise ValidationError("Source path is required", "source_path")

            # Validate sync settings
            if sync_mode not in ("one_time", "continuous"):
                raise ValidationError("sync_mode must be 'one_time' or 'continuous'", "sync_mode")

            # Keep repository validation aligned with the DB constraint:
            # interval_minutes IS NULL OR interval_minutes >= MIN_SYNC_INTERVAL_MINUTES.
            # For one_time sources, interval_minutes is ignored and normalized to NULL.
            if sync_mode == "one_time":
                interval_minutes = None
            else:
                if interval_minutes is None:
                    raise ValidationError("interval_minutes is required for continuous sync", "interval_minutes")
                if interval_minutes < MIN_SYNC_INTERVAL_MINUTES:
                    raise ValidationError(
                        f"interval_minutes must be at least {MIN_SYNC_INTERVAL_MINUTES}",
                        "interval_minutes",
                    )

            # Calculate next_run_at for continuous sync
            next_run_at = None
            if sync_mode == "continuous" and interval_minutes:
                # Schedule first run immediately
                next_run_at = datetime.now(UTC)

            source = CollectionSource(
                collection_id=collection_id,
                source_type=source_type,
                source_path=source_path,
                source_config=source_config or {},
                meta=meta or {},
                sync_mode=sync_mode,
                interval_minutes=interval_minutes,
                next_run_at=next_run_at,
            )

            self.session.add(source)
            await self.session.flush()

            logger.info(
                f"Created collection source {source.id} for collection {collection_id} "
                f"(type={source_type}, path={source_path}, sync_mode={sync_mode})"
            )
            return source

        except IntegrityError as e:
            logger.error(f"Integrity error creating collection source: {e}")
            raise EntityAlreadyExistsError("collection_source", source_path) from e
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Failed to create collection source: {e}")
            raise DatabaseOperationError("create", "collection_source", str(e)) from e

    async def get_by_id(self, source_id: int) -> CollectionSource | None:
        """Get a collection source by ID.

        Args:
            source_id: Integer ID of the source

        Returns:
            CollectionSource instance or None if not found
        """
        try:
            result = await self.session.execute(select(CollectionSource).where(CollectionSource.id == source_id))
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Failed to get collection source {source_id}: {e}")
            raise DatabaseOperationError("get", "collection_source", str(e)) from e

    async def get_by_collection_and_path(
        self,
        collection_id: str,
        source_path: str,
    ) -> CollectionSource | None:
        """Get a collection source by collection ID and source path.

        Args:
            collection_id: UUID of the parent collection
            source_path: Path/identifier of the source

        Returns:
            CollectionSource instance or None if not found
        """
        try:
            result = await self.session.execute(
                select(CollectionSource).where(
                    CollectionSource.collection_id == collection_id,
                    CollectionSource.source_path == source_path,
                )
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Failed to get collection source for collection {collection_id} with path {source_path}: {e}")
            raise DatabaseOperationError("get", "collection_source", str(e)) from e

    async def get_or_create(
        self,
        collection_id: str,
        source_type: str,
        source_path: str,
        source_config: dict[str, Any] | None = None,
    ) -> tuple[CollectionSource, bool]:
        """Get an existing source or create a new one.

        Uses (collection_id, source_path) as the unique lookup key.
        If found, updates source_type and source_config if different.

        Args:
            collection_id: UUID of the parent collection
            source_type: Type of source (directory, web, slack, etc.)
            source_path: Display path or identifier for the source
            source_config: Connector-specific configuration

        Returns:
            Tuple of (CollectionSource, is_new) where is_new indicates creation

        Raises:
            ValidationError: If required fields are invalid
            DatabaseOperationError: For database errors
        """
        try:
            # Try to find existing source
            existing = await self.get_by_collection_and_path(collection_id, source_path)

            if existing:
                # Update source_type and source_config if they've changed
                updated = False
                if existing.source_type != source_type:
                    existing.source_type = source_type
                    updated = True
                if source_config and existing.source_config != source_config:
                    existing.source_config = source_config
                    updated = True

                if updated:
                    existing.updated_at = datetime.now(UTC)
                    await self.session.flush()
                    logger.info(f"Updated existing collection source {existing.id} for collection {collection_id}")

                return existing, False

            # Create new source
            source = await self.create(
                collection_id=collection_id,
                source_type=source_type,
                source_path=source_path,
                source_config=source_config,
            )
            return source, True

        except (ValidationError, EntityAlreadyExistsError):
            raise
        except Exception as e:
            logger.error(f"Failed to get_or_create collection source: {e}")
            raise DatabaseOperationError("get_or_create", "collection_source", str(e)) from e

    async def list_by_collection(
        self,
        collection_id: str,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[CollectionSource], int]:
        """List sources for a collection.

        Args:
            collection_id: UUID of the parent collection
            offset: Pagination offset
            limit: Maximum number of results

        Returns:
            Tuple of (sources list, total count)
        """
        try:
            # Build base query
            query = select(CollectionSource).where(CollectionSource.collection_id == collection_id)

            # Get total count
            count_query = select(func.count()).select_from(query.subquery())
            total = await self.session.scalar(count_query)

            # Get paginated results
            query = query.order_by(CollectionSource.created_at.desc()).offset(offset).limit(limit)
            result = await self.session.execute(query)
            sources = result.scalars().all()

            return list(sources), total or 0

        except Exception as e:
            logger.error(f"Failed to list sources for collection {collection_id}: {e}")
            raise DatabaseOperationError("list", "collection_source", str(e)) from e

    async def update_stats(
        self,
        source_id: int,
        document_count: int | None = None,
        size_bytes: int | None = None,
        last_indexed_at: datetime | None = None,
    ) -> CollectionSource:
        """Update source statistics.

        Args:
            source_id: Integer ID of the source
            document_count: New document count
            size_bytes: New size in bytes
            last_indexed_at: Last indexing timestamp

        Returns:
            Updated CollectionSource instance

        Raises:
            EntityNotFoundError: If source not found
            ValidationError: If any count is negative
        """
        try:
            # Validate non-negative counts
            if document_count is not None and document_count < 0:
                raise ValidationError("Document count cannot be negative", "document_count")
            if size_bytes is not None and size_bytes < 0:
                raise ValidationError("Size bytes cannot be negative", "size_bytes")

            source = await self.get_by_id(source_id)
            if not source:
                raise EntityNotFoundError("collection_source", str(source_id))

            if document_count is not None:
                source.document_count = document_count
            if size_bytes is not None:
                source.size_bytes = size_bytes
            if last_indexed_at is not None:
                source.last_indexed_at = last_indexed_at

            source.updated_at = datetime.now(UTC)
            await self.session.flush()

            logger.debug(f"Updated collection source {source_id} stats")
            return source

        except (EntityNotFoundError, ValidationError):
            raise
        except Exception as e:
            logger.error(f"Failed to update collection source stats: {e}")
            raise DatabaseOperationError("update", "collection_source", str(e)) from e

    async def delete(self, source_id: int) -> None:
        """Delete a collection source.

        Args:
            source_id: Integer ID of the source to delete

        Raises:
            EntityNotFoundError: If source not found
        """
        try:
            source = await self.get_by_id(source_id)
            if not source:
                raise EntityNotFoundError("collection_source", str(source_id))

            await self.session.execute(delete(CollectionSource).where(CollectionSource.id == source.id))
            await self.session.flush()

            logger.info(f"Deleted collection source {source_id}")

        except EntityNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to delete collection source: {e}")
            raise DatabaseOperationError("delete", "collection_source", str(e)) from e

    # -------------------------------------------------------------------------
    # Sync-related methods
    # -------------------------------------------------------------------------

    async def update(
        self,
        source_id: int,
        source_config: dict[str, Any] | None = None,
        sync_mode: str | None = None,
        interval_minutes: int | None = None,
        meta: dict[str, Any] | None = None,
        last_run_started_at: datetime | None = None,
    ) -> CollectionSource:
        """Update a collection source.

        Args:
            source_id: Integer ID of the source
            source_config: New connector-specific configuration
            sync_mode: New sync mode ('one_time' or 'continuous')
            interval_minutes: New sync interval (min 15)
            meta: New metadata
            last_run_started_at: When the current run started

        Returns:
            Updated CollectionSource instance

        Raises:
            EntityNotFoundError: If source not found
            ValidationError: If validation fails
        """
        try:
            source = await self.get_by_id(source_id)
            if not source:
                raise EntityNotFoundError("collection_source", str(source_id))

            # Validate sync settings if changing
            if sync_mode is not None and sync_mode not in ("one_time", "continuous"):
                raise ValidationError("sync_mode must be 'one_time' or 'continuous'", "sync_mode")

            # Use provided interval or existing one for validation
            effective_interval = interval_minutes if interval_minutes is not None else source.interval_minutes
            effective_sync_mode = sync_mode if sync_mode is not None else source.sync_mode

            # Keep repository validation aligned with the DB constraint:
            # interval_minutes IS NULL OR interval_minutes >= MIN_SYNC_INTERVAL_MINUTES.
            # For one_time sources, interval_minutes is ignored and normalized to NULL.
            if effective_sync_mode == "continuous":
                if effective_interval is None:
                    raise ValidationError("interval_minutes is required for continuous sync", "interval_minutes")
                if effective_interval < MIN_SYNC_INTERVAL_MINUTES:
                    raise ValidationError(
                        f"interval_minutes must be at least {MIN_SYNC_INTERVAL_MINUTES}",
                        "interval_minutes",
                    )

            # Apply updates
            if source_config is not None:
                source.source_config = source_config
            if sync_mode is not None:
                source.sync_mode = sync_mode
            if effective_sync_mode == "one_time":
                # If caller is setting one_time, always clear interval_minutes.
                # If caller only provides interval_minutes while staying one_time, ignore it.
                if sync_mode == "one_time" or interval_minutes is not None:
                    source.interval_minutes = None
            elif interval_minutes is not None:
                source.interval_minutes = interval_minutes
            if meta is not None:
                source.meta = meta
            if last_run_started_at is not None:
                source.last_run_started_at = last_run_started_at

            # If switching to continuous mode and not paused, schedule next run
            if sync_mode == "continuous" and source.paused_at is None and source.next_run_at is None:
                source.next_run_at = datetime.now(UTC)

            source.updated_at = datetime.now(UTC)
            await self.session.flush()

            logger.info(f"Updated collection source {source_id}")
            return source

        except (EntityNotFoundError, ValidationError):
            raise
        except Exception as e:
            logger.error(f"Failed to update collection source: {e}")
            raise DatabaseOperationError("update", "collection_source", str(e)) from e

    async def get_due_for_sync(self, limit: int = 100) -> list[CollectionSource]:
        """Get sources that are due for sync.

        Returns continuous sync sources where:
        - next_run_at <= now()
        - paused_at IS NULL
        - sync_mode = 'continuous'

        Args:
            limit: Maximum number of sources to return

        Returns:
            List of CollectionSource instances due for sync
        """
        try:
            now = datetime.now(UTC)
            query = (
                select(CollectionSource)
                .where(
                    and_(
                        CollectionSource.sync_mode == "continuous",
                        CollectionSource.paused_at.is_(None),
                        CollectionSource.next_run_at <= now,
                    )
                )
                .order_by(CollectionSource.next_run_at.asc())
                .limit(limit)
            )

            result = await self.session.execute(query)
            sources = result.scalars().all()

            logger.debug(f"Found {len(sources)} sources due for sync")
            return list(sources)

        except Exception as e:
            logger.error(f"Failed to get sources due for sync: {e}")
            raise DatabaseOperationError("get_due_for_sync", "collection_source", str(e)) from e

    async def update_sync_status(
        self,
        source_id: int,
        status: str,
        error: str | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
    ) -> CollectionSource:
        """Update sync status after a run.

        Args:
            source_id: Integer ID of the source
            status: Run status ('success', 'failed', 'partial')
            error: Error message if status is 'failed'
            started_at: When the run started
            completed_at: When the run completed

        Returns:
            Updated CollectionSource instance

        Raises:
            EntityNotFoundError: If source not found
            ValidationError: If status is invalid
        """
        try:
            if status not in ("success", "failed", "partial"):
                raise ValidationError("status must be 'success', 'failed', or 'partial'", "status")

            source = await self.get_by_id(source_id)
            if not source:
                raise EntityNotFoundError("collection_source", str(source_id))

            source.last_run_status = status
            source.last_error = error if status == "failed" else None

            if started_at:
                source.last_run_started_at = started_at
            if completed_at:
                source.last_run_completed_at = completed_at

            source.updated_at = datetime.now(UTC)
            await self.session.flush()

            logger.info(f"Updated sync status for source {source_id}: {status}")
            return source

        except (EntityNotFoundError, ValidationError):
            raise
        except Exception as e:
            logger.error(f"Failed to update sync status: {e}")
            raise DatabaseOperationError("update_sync_status", "collection_source", str(e)) from e

    async def set_next_run(
        self,
        source_id: int,
        next_run_at: datetime | None = None,
    ) -> CollectionSource:
        """Schedule the next sync run.

        If next_run_at is None, calculates based on interval_minutes.

        Args:
            source_id: Integer ID of the source
            next_run_at: When to run next (or None to calculate)

        Returns:
            Updated CollectionSource instance

        Raises:
            EntityNotFoundError: If source not found
        """
        try:
            source = await self.get_by_id(source_id)
            if not source:
                raise EntityNotFoundError("collection_source", str(source_id))

            if next_run_at is not None:
                source.next_run_at = next_run_at
            elif source.interval_minutes:
                source.next_run_at = datetime.now(UTC) + timedelta(minutes=source.interval_minutes)
            else:
                source.next_run_at = None

            source.updated_at = datetime.now(UTC)
            await self.session.flush()

            logger.debug(f"Set next run for source {source_id}: {source.next_run_at}")
            return source

        except EntityNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to set next run: {e}")
            raise DatabaseOperationError("set_next_run", "collection_source", str(e)) from e

    async def pause(self, source_id: int) -> CollectionSource:
        """Pause a source's sync schedule.

        Args:
            source_id: Integer ID of the source

        Returns:
            Updated CollectionSource instance

        Raises:
            EntityNotFoundError: If source not found
            ValidationError: If source is not continuous sync
        """
        try:
            source = await self.get_by_id(source_id)
            if not source:
                raise EntityNotFoundError("collection_source", str(source_id))

            if source.sync_mode != "continuous":
                raise ValidationError("Can only pause continuous sync sources", "sync_mode")

            if source.paused_at is not None:
                # Already paused
                return source

            source.paused_at = datetime.now(UTC)
            source.updated_at = datetime.now(UTC)
            await self.session.flush()

            logger.info(f"Paused sync for source {source_id}")
            return source

        except (EntityNotFoundError, ValidationError):
            raise
        except Exception as e:
            logger.error(f"Failed to pause source: {e}")
            raise DatabaseOperationError("pause", "collection_source", str(e)) from e

    async def resume(self, source_id: int) -> CollectionSource:
        """Resume a paused source's sync schedule.

        Args:
            source_id: Integer ID of the source

        Returns:
            Updated CollectionSource instance

        Raises:
            EntityNotFoundError: If source not found
            ValidationError: If source is not continuous sync
        """
        try:
            source = await self.get_by_id(source_id)
            if not source:
                raise EntityNotFoundError("collection_source", str(source_id))

            if source.sync_mode != "continuous":
                raise ValidationError("Can only resume continuous sync sources", "sync_mode")

            if source.paused_at is None:
                # Not paused
                return source

            source.paused_at = None
            # Schedule next run immediately
            source.next_run_at = datetime.now(UTC)
            source.updated_at = datetime.now(UTC)
            await self.session.flush()

            logger.info(f"Resumed sync for source {source_id}")
            return source

        except (EntityNotFoundError, ValidationError):
            raise
        except Exception as e:
            logger.error(f"Failed to resume source: {e}")
            raise DatabaseOperationError("resume", "collection_source", str(e)) from e
