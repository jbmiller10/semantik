"""Source Service for managing collection source operations."""

import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from shared.database.exceptions import (
    AccessDeniedError,
    EntityNotFoundError,
    InvalidStateError,
)
from shared.database.models import CollectionSource, OperationType
from shared.database.repositories.collection_repository import CollectionRepository
from shared.database.repositories.collection_source_repository import CollectionSourceRepository
from shared.database.repositories.operation_repository import OperationRepository
from webui.celery_app import celery_app

logger = logging.getLogger(__name__)


class SourceService:
    """Service for managing collection source lifecycle operations.

    This service handles:
    - Creating, updating, and deleting sources
    - Triggering sync runs (manual and scheduled)
    - Pausing and resuming continuous syncs
    - Listing sources with sync status

    Uses the existing operation infrastructure for triggering ingestion.
    """

    def __init__(
        self,
        db_session: AsyncSession,
        collection_repo: CollectionRepository,
        source_repo: CollectionSourceRepository,
        operation_repo: OperationRepository,
    ):
        """Initialize the source service.

        Args:
            db_session: Database session for transactions
            collection_repo: Collection repository for access checks
            source_repo: Source repository for CRUD operations
            operation_repo: Operation repository for dispatching operations
        """
        self.db_session = db_session
        self.collection_repo = collection_repo
        self.source_repo = source_repo
        self.operation_repo = operation_repo

    async def create_source(
        self,
        user_id: int,
        collection_id: str,
        source_type: str,
        source_path: str,
        source_config: dict[str, Any],
        sync_mode: str = "one_time",
        interval_minutes: int | None = None,
    ) -> CollectionSource:
        """Create a new source for a collection.

        Args:
            user_id: ID of the user creating the source
            collection_id: UUID of the parent collection
            source_type: Type of source (directory, git, imap)
            source_path: Display path or identifier for the source
            source_config: Connector-specific configuration
            sync_mode: 'one_time' or 'continuous' (default: 'one_time')
            interval_minutes: Sync interval for continuous mode (min 15)

        Returns:
            Created CollectionSource instance

        Raises:
            EntityNotFoundError: If collection not found
            AccessDeniedError: If user doesn't have access
            ValidationError: If validation fails
        """
        # Verify user has access to collection
        collection = await self.collection_repo.get_by_id(collection_id)
        if not collection:
            raise EntityNotFoundError("collection", collection_id)

        if collection.owner_id != user_id:
            raise AccessDeniedError(str(user_id), "collection", collection_id)

        # Create the source
        source = await self.source_repo.create(
            collection_id=collection_id,
            source_type=source_type,
            source_path=source_path,
            source_config=source_config,
            sync_mode=sync_mode,
            interval_minutes=interval_minutes,
        )

        logger.info(
            f"Created source {source.id} for collection {collection_id} (type={source_type}, sync_mode={sync_mode})"
        )

        return source

    async def update_source(
        self,
        user_id: int,
        source_id: int,
        source_config: dict[str, Any] | None = None,
        sync_mode: str | None = None,
        interval_minutes: int | None = None,
    ) -> CollectionSource:
        """Update a source's configuration.

        Args:
            user_id: ID of the user updating the source
            source_id: ID of the source to update
            source_config: New connector-specific configuration
            sync_mode: New sync mode ('one_time' or 'continuous')
            interval_minutes: New sync interval (min 15)

        Returns:
            Updated CollectionSource instance

        Raises:
            EntityNotFoundError: If source not found
            AccessDeniedError: If user doesn't have access
            ValidationError: If validation fails
        """
        # Get source and verify access
        source = await self.source_repo.get_by_id(source_id)
        if not source:
            raise EntityNotFoundError("collection_source", str(source_id))

        collection = await self.collection_repo.get_by_id(source.collection_id)
        if not collection or collection.owner_id != user_id:
            raise AccessDeniedError(str(user_id), "collection_source", str(source_id))

        # Update the source
        updated_source = await self.source_repo.update(
            source_id=source_id,
            source_config=source_config,
            sync_mode=sync_mode,
            interval_minutes=interval_minutes,
        )

        logger.info(f"Updated source {source_id}")
        return updated_source

    async def delete_source(
        self,
        user_id: int,
        source_id: int,
    ) -> dict[str, Any]:
        """Delete a source and its documents.

        This triggers a REMOVE_SOURCE operation to delete the source's
        documents and vectors before removing the source record.

        Args:
            user_id: ID of the user deleting the source
            source_id: ID of the source to delete

        Returns:
            Operation dictionary for the REMOVE_SOURCE operation

        Raises:
            EntityNotFoundError: If source not found
            AccessDeniedError: If user doesn't have access
            InvalidStateError: If collection has active operation
        """
        # Get source and verify access
        source = await self.source_repo.get_by_id(source_id)
        if not source:
            raise EntityNotFoundError("collection_source", str(source_id))

        collection = await self.collection_repo.get_by_id(source.collection_id)
        if not collection or collection.owner_id != user_id:
            raise AccessDeniedError(str(user_id), "collection_source", str(source_id))

        # Check for active operations
        active_ops = await self.operation_repo.get_active_operations(source.collection_id)
        if active_ops:
            raise InvalidStateError(
                "Collection has active operation(s). Please wait for them to complete before deleting sources."
            )

        # Create REMOVE_SOURCE operation
        operation = await self.operation_repo.create(
            collection_id=source.collection_id,
            user_id=user_id,
            operation_type=OperationType.REMOVE_SOURCE,
            config={"source_id": source_id},
        )

        # Commit before dispatching Celery task
        await self.db_session.commit()

        # Dispatch the task
        celery_app.send_task(
            "webui.tasks.process_collection_operation",
            args=[operation.uuid],
            queue="default",
        )

        logger.info(f"Dispatched REMOVE_SOURCE operation {operation.uuid} for source {source_id}")

        return {
            "id": operation.id,
            "uuid": operation.uuid,
            "type": operation.type.value,
            "status": operation.status.value,
        }

    async def get_source(
        self,
        user_id: int,
        source_id: int,
    ) -> CollectionSource:
        """Get a source by ID.

        Args:
            user_id: ID of the user requesting the source
            source_id: ID of the source

        Returns:
            CollectionSource instance

        Raises:
            EntityNotFoundError: If source not found
            AccessDeniedError: If user doesn't have access
        """
        source = await self.source_repo.get_by_id(source_id)
        if not source:
            raise EntityNotFoundError("collection_source", str(source_id))

        collection = await self.collection_repo.get_by_id(source.collection_id)
        if not collection or collection.owner_id != user_id:
            raise AccessDeniedError(str(user_id), "collection_source", str(source_id))

        return source

    async def list_sources(
        self,
        user_id: int,
        collection_id: str,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[CollectionSource], int]:
        """List sources for a collection.

        Args:
            user_id: ID of the user requesting sources
            collection_id: UUID of the collection
            offset: Pagination offset
            limit: Maximum results

        Returns:
            Tuple of (sources list, total count)

        Raises:
            EntityNotFoundError: If collection not found
            AccessDeniedError: If user doesn't have access
        """
        # Verify user has access to collection
        collection = await self.collection_repo.get_by_id(collection_id)
        if not collection:
            raise EntityNotFoundError("collection", collection_id)

        if collection.owner_id != user_id:
            raise AccessDeniedError(str(user_id), "collection", collection_id)

        return await self.source_repo.list_by_collection(
            collection_id=collection_id,
            offset=offset,
            limit=limit,
        )

    async def run_now(
        self,
        user_id: int,
        source_id: int,
    ) -> dict[str, Any]:
        """Trigger an immediate sync run for a source.

        Creates an APPEND operation for the source and dispatches it.
        For continuous sync sources, also updates next_run_at.

        Args:
            user_id: ID of the user triggering the run
            source_id: ID of the source to sync

        Returns:
            Operation dictionary

        Raises:
            EntityNotFoundError: If source not found
            AccessDeniedError: If user doesn't have access
            InvalidStateError: If collection has active operation
        """
        # Get source and verify access
        source = await self.source_repo.get_by_id(source_id)
        if not source:
            raise EntityNotFoundError("collection_source", str(source_id))

        collection = await self.collection_repo.get_by_id(source.collection_id)
        if not collection or collection.owner_id != user_id:
            raise AccessDeniedError(str(user_id), "collection_source", str(source_id))

        # Check for active operations
        active_ops = await self.operation_repo.get_active_operations(source.collection_id)
        if active_ops:
            raise InvalidStateError(
                "Collection has active operation(s). Please wait for them to complete before running sync."
            )

        # Mark sync as started
        await self.source_repo.update_sync_status(
            source_id=source_id,
            status="partial",  # Will be updated on completion
            started_at=datetime.now(UTC),
        )

        # Create APPEND operation
        operation = await self.operation_repo.create(
            collection_id=source.collection_id,
            user_id=user_id,
            operation_type=OperationType.APPEND,
            config={
                "source_id": source_id,
                "source_type": source.source_type,
                "source_config": source.source_config,
                "source_path": source.source_path,
            },
        )

        # Update next_run_at for continuous sync
        if source.sync_mode == "continuous" and source.interval_minutes:
            await self.source_repo.set_next_run(source_id)

        # Commit before dispatching Celery task
        await self.db_session.commit()

        # Dispatch the task
        celery_app.send_task(
            "webui.tasks.process_collection_operation",
            args=[operation.uuid],
            queue="default",
        )

        logger.info(f"Dispatched APPEND operation {operation.uuid} for source {source_id}")

        return {
            "id": operation.id,
            "uuid": operation.uuid,
            "type": operation.type.value,
            "status": operation.status.value,
        }

    async def pause(
        self,
        user_id: int,
        source_id: int,
    ) -> CollectionSource:
        """Pause a source's continuous sync.

        Args:
            user_id: ID of the user pausing the source
            source_id: ID of the source to pause

        Returns:
            Updated CollectionSource instance

        Raises:
            EntityNotFoundError: If source not found
            AccessDeniedError: If user doesn't have access
            ValidationError: If source is not continuous sync
        """
        # Get source and verify access
        source = await self.source_repo.get_by_id(source_id)
        if not source:
            raise EntityNotFoundError("collection_source", str(source_id))

        collection = await self.collection_repo.get_by_id(source.collection_id)
        if not collection or collection.owner_id != user_id:
            raise AccessDeniedError(str(user_id), "collection_source", str(source_id))

        return await self.source_repo.pause(source_id)

    async def resume(
        self,
        user_id: int,
        source_id: int,
    ) -> CollectionSource:
        """Resume a paused source's continuous sync.

        Args:
            user_id: ID of the user resuming the source
            source_id: ID of the source to resume

        Returns:
            Updated CollectionSource instance

        Raises:
            EntityNotFoundError: If source not found
            AccessDeniedError: If user doesn't have access
            ValidationError: If source is not continuous sync
        """
        # Get source and verify access
        source = await self.source_repo.get_by_id(source_id)
        if not source:
            raise EntityNotFoundError("collection_source", str(source_id))

        collection = await self.collection_repo.get_by_id(source.collection_id)
        if not collection or collection.owner_id != user_id:
            raise AccessDeniedError(str(user_id), "collection_source", str(source_id))

        return await self.source_repo.resume(source_id)
