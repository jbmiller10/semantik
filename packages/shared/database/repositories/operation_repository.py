"""Repository implementation for Operation model."""

import logging
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import desc, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from shared.database.exceptions import AccessDeniedError, DatabaseOperationError, EntityNotFoundError, ValidationError
from shared.database.models import Collection, Operation, OperationStatus, OperationType

logger = logging.getLogger(__name__)


class OperationRepository:
    """Repository for Operation model operations."""

    def __init__(self, session: AsyncSession):
        """Initialize with database session.

        Args:
            session: AsyncSession instance for database operations
        """
        self.session = session

    async def create(
        self,
        collection_id: str,
        user_id: int,
        operation_type: OperationType,
        config: dict[str, Any],
        meta: dict[str, Any] | None = None,
    ) -> Operation:
        """Create a new operation.

        Args:
            collection_id: UUID of the collection
            user_id: ID of the user creating the operation
            operation_type: Type of operation
            config: Operation configuration
            meta: Optional metadata

        Returns:
            Created Operation instance

        Raises:
            EntityNotFoundError: If collection not found
            AccessDeniedError: If user doesn't have access to collection
            ValidationError: If config is invalid
            DatabaseOperationError: For database errors
        """
        try:
            # Validate config is not empty
            if not config:
                raise ValidationError("Operation config cannot be empty", "config")

            # Check collection exists and user has access
            collection_result = await self.session.execute(select(Collection).where(Collection.id == collection_id))
            collection = collection_result.scalar_one_or_none()

            if not collection:
                raise EntityNotFoundError("collection", collection_id)

            # Check user has access (owner or public collection)
            if collection.owner_id != user_id and not collection.is_public:
                # TODO: Check CollectionPermission table for write access
                raise AccessDeniedError(str(user_id), "collection", collection_id)

            # Generate UUID for the operation
            operation_uuid = str(uuid4())

            operation = Operation(
                uuid=operation_uuid,
                collection_id=collection_id,
                user_id=user_id,
                type=operation_type,
                status=OperationStatus.PENDING,  # Pass enum object directly
                config=config,
                meta=meta or {},
            )

            self.session.add(operation)
            await self.session.flush()

            logger.info(
                f"Created operation {operation.uuid} of type {operation_type} "
                f"for collection {collection_id} by user {user_id}"
            )
            return operation

        except (EntityNotFoundError, AccessDeniedError, ValidationError):
            raise
        except Exception as e:
            logger.error("Failed to create operation: %s", e, exc_info=True)
            raise DatabaseOperationError("create", "operation", str(e)) from e

    async def get_by_uuid(self, operation_uuid: str) -> Operation | None:
        """Get an operation by UUID.

        Args:
            operation_uuid: UUID of the operation

        Returns:
            Operation instance or None if not found
        """
        try:
            result = await self.session.execute(
                select(Operation).where(Operation.uuid == operation_uuid).options(selectinload(Operation.collection))
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error("Failed to get operation %s: %s", operation_uuid, e, exc_info=True)
            raise DatabaseOperationError("get", "operation", str(e)) from e

    async def get_by_uuid_with_permission_check(self, operation_uuid: str, user_id: int) -> Operation:
        """Get an operation by UUID with permission check.

        Args:
            operation_uuid: UUID of the operation
            user_id: ID of the user requesting access

        Returns:
            Operation instance

        Raises:
            EntityNotFoundError: If operation not found
            AccessDeniedError: If user doesn't have access
        """
        operation = await self.get_by_uuid(operation_uuid)

        if not operation:
            raise EntityNotFoundError("operation", operation_uuid)

        # Load collection for permission check
        await self.session.refresh(operation, ["collection"])

        # Check if user owns the operation or the collection
        if (
            operation.user_id != user_id
            and operation.collection.owner_id != user_id
            and not operation.collection.is_public
        ):
            # TODO: Check CollectionPermission table for access
            raise AccessDeniedError(str(user_id), "operation", operation_uuid)

        return operation

    async def set_task_id(self, operation_uuid: str, task_id: str) -> Operation:
        """Set the Celery task ID for an operation.

        This is a high-priority update that should be done immediately
        after submitting the task to Celery.

        Args:
            operation_uuid: UUID of the operation
            task_id: Celery task ID

        Returns:
            Updated Operation instance

        Raises:
            EntityNotFoundError: If operation not found
        """
        try:
            # Use update query for immediate effect
            result = await self.session.execute(
                update(Operation).where(Operation.uuid == operation_uuid).values(task_id=task_id).returning(Operation)
            )
            operation = result.scalar_one_or_none()

            if not operation:
                raise EntityNotFoundError("operation", operation_uuid)

            await self.session.flush()

            logger.debug(f"Set task_id {task_id} for operation {operation_uuid}")
            return operation

        except EntityNotFoundError:
            raise
        except Exception as e:
            logger.error("Failed to set task_id for operation: %s", e, exc_info=True)
            raise DatabaseOperationError("update", "operation", str(e)) from e

    async def update_status(
        self,
        operation_uuid: str,
        status: OperationStatus,
        error_message: str | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
    ) -> Operation:
        """Update operation status.

        Args:
            operation_uuid: UUID of the operation
            status: New status
            error_message: Optional error message (for failed status)
            started_at: When the operation started processing
            completed_at: When the operation completed

        Returns:
            Updated Operation instance

        Raises:
            EntityNotFoundError: If operation not found
        """
        try:
            operation = await self.get_by_uuid(operation_uuid)
            if not operation:
                raise EntityNotFoundError("operation", operation_uuid)

            operation.status = status  # Pass enum object directly

            if error_message is not None:
                operation.error_message = error_message
            if started_at is not None:
                operation.started_at = started_at
            if completed_at is not None:
                operation.completed_at = completed_at

            # Auto-set timestamps based on status
            if status == OperationStatus.PROCESSING and not operation.started_at:
                operation.started_at = datetime.now(UTC)
            elif (
                status in (OperationStatus.COMPLETED, OperationStatus.FAILED, OperationStatus.CANCELLED)
                and not operation.completed_at
            ):
                operation.completed_at = datetime.now(UTC)

            await self.session.flush()

            logger.info(f"Updated operation {operation_uuid} status to {status}")
            return operation

        except EntityNotFoundError:
            raise
        except Exception as e:
            logger.error("Failed to update operation status: %s", e, exc_info=True)
            raise DatabaseOperationError("update", "operation", str(e)) from e

    async def list_for_collection(
        self,
        collection_id: str,
        user_id: int,
        status: OperationStatus | None = None,
        operation_type: OperationType | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[Operation], int]:
        """List operations for a collection.

        Args:
            collection_id: UUID of the collection
            user_id: ID of the user requesting the list
            status: Optional status filter
            operation_type: Optional type filter
            offset: Pagination offset
            limit: Maximum number of results

        Returns:
            Tuple of (operations list, total count)

        Raises:
            AccessDeniedError: If user doesn't have access to collection
        """
        try:
            # Check collection access
            collection_result = await self.session.execute(select(Collection).where(Collection.id == collection_id))
            collection = collection_result.scalar_one_or_none()

            if not collection:
                raise EntityNotFoundError("collection", collection_id)

            if collection.owner_id != user_id and not collection.is_public:
                # TODO: Check CollectionPermission table
                raise AccessDeniedError(str(user_id), "collection", collection_id)

            # Build query
            query = select(Operation).where(Operation.collection_id == collection_id)

            if status is not None:
                query = query.where(Operation.status == status)
            if operation_type is not None:
                query = query.where(Operation.type == operation_type)

            # Get total count
            count_query = select(func.count()).select_from(query.subquery())
            total = await self.session.scalar(count_query)

            # Get paginated results
            query = query.order_by(desc(Operation.created_at)).offset(offset).limit(limit)
            result = await self.session.execute(query)
            operations = result.scalars().all()

            return list(operations), total or 0

        except (EntityNotFoundError, AccessDeniedError):
            raise
        except Exception as e:
            logger.error("Failed to list operations for collection: %s", e, exc_info=True)
            raise DatabaseOperationError("list", "operations", str(e)) from e

    async def list_for_user(
        self,
        user_id: int,
        status: OperationStatus | None = None,
        status_list: list[OperationStatus] | None = None,
        operation_type: OperationType | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[Operation], int]:
        """List operations created by a user.

        Args:
            user_id: ID of the user
            status: Optional single status filter (deprecated, use status_list)
            status_list: Optional list of statuses to filter by
            operation_type: Optional type filter
            offset: Pagination offset
            limit: Maximum number of results

        Returns:
            Tuple of (operations list, total count)
        """
        try:
            # Build query
            query = select(Operation).where(Operation.user_id == user_id)

            # Handle status filtering
            if status_list is not None and len(status_list) > 0:
                # Use enum objects directly with native_enum
                query = query.where(Operation.status.in_(status_list))
            elif status is not None:
                # Backwards compatibility - use enum object directly
                query = query.where(Operation.status == status)

            if operation_type is not None:
                # Use enum object directly
                query = query.where(Operation.type == operation_type)

            # Get total count
            count_query = select(func.count()).select_from(query.subquery())
            total = await self.session.scalar(count_query)

            # Get paginated results with collection loaded
            query = (
                query.options(selectinload(Operation.collection))
                .order_by(desc(Operation.created_at))
                .offset(offset)
                .limit(limit)
            )
            result = await self.session.execute(query)
            operations = result.scalars().all()

            return list(operations), total or 0

        except Exception as e:
            logger.error("Failed to list operations for user: %s", e, exc_info=True)
            raise DatabaseOperationError("list", "operations", str(e)) from e

    async def cancel(self, operation_uuid: str, user_id: int) -> Operation:
        """Cancel a pending or processing operation.

        Args:
            operation_uuid: UUID of the operation
            user_id: ID of the user cancelling the operation

        Returns:
            Updated Operation instance

        Raises:
            EntityNotFoundError: If operation not found
            AccessDeniedError: If user doesn't have permission
            ValidationError: If operation cannot be cancelled
        """
        try:
            # Get operation with permission check (handles all authorization)
            operation = await self.get_by_uuid_with_permission_check(operation_uuid, user_id)

            # Can only cancel pending or processing operations
            if operation.status not in (OperationStatus.PENDING, OperationStatus.PROCESSING):
                raise ValidationError(f"Cannot cancel operation in {operation.status} status", "status")

            operation.status = OperationStatus.CANCELLED  # Pass enum object directly
            operation.completed_at = datetime.now(UTC)

            await self.session.flush()

            logger.info(f"Cancelled operation {operation_uuid}")
            return operation

        except (EntityNotFoundError, AccessDeniedError, ValidationError):
            raise
        except Exception as e:
            logger.error("Failed to cancel operation: %s", e, exc_info=True)
            raise DatabaseOperationError("cancel", "operation", str(e)) from e

    async def get_active_operations_count(self, collection_id: str) -> int:
        """Get count of active (pending/processing) operations for a collection.

        Args:
            collection_id: UUID of the collection

        Returns:
            Number of active operations
        """
        try:
            result = await self.session.scalar(
                select(func.count(Operation.id)).where(
                    Operation.collection_id == collection_id,
                    Operation.status.in_([OperationStatus.PENDING, OperationStatus.PROCESSING]),
                )
            )
            return result or 0
        except Exception as e:
            logger.error("Failed to get active operations count: %s", e, exc_info=True)
            raise DatabaseOperationError("count", "operations", str(e)) from e

    async def get_active_operations(self, collection_id: str) -> list[Operation]:
        """Get active (pending/processing) operations for a collection.

        Args:
            collection_id: UUID of the collection

        Returns:
            List of active operations
        """
        try:
            stmt = select(Operation).where(
                Operation.collection_id == collection_id,
                Operation.status.in_([OperationStatus.PENDING, OperationStatus.PROCESSING]),
            )
            result = await self.session.execute(stmt)
            return list(result.scalars().all())
        except Exception as e:
            logger.error("Failed to get active operations: %s", e, exc_info=True)
            raise DatabaseOperationError("list", "operations", str(e)) from e

    async def get_stuck_operations(
        self,
        stuck_threshold_minutes: int = 15,
        limit: int = 100,
    ) -> list[Operation]:
        """Find operations stuck in PENDING/PROCESSING longer than threshold.

        These are candidates for cleanup - actual cleanup should verify
        Celery task status before marking as failed.

        Causes of stuck operations:
        - Celery dispatch failure after DB commit
        - Worker crash during processing
        - Task timeout without status update

        Args:
            stuck_threshold_minutes: Minutes after which an operation is considered stuck
            limit: Maximum number of operations to return

        Returns:
            List of stuck operations, oldest first
        """
        from datetime import timedelta

        try:
            threshold = datetime.now(UTC) - timedelta(minutes=stuck_threshold_minutes)
            stmt = (
                select(Operation)
                .where(
                    Operation.status.in_([OperationStatus.PENDING, OperationStatus.PROCESSING]),
                    Operation.created_at < threshold,
                )
                .order_by(Operation.created_at.asc())
                .limit(limit)
            )
            result = await self.session.execute(stmt)
            return list(result.scalars().all())
        except Exception as e:
            logger.error("Failed to get stuck operations: %s", e, exc_info=True)
            raise DatabaseOperationError("list", "operations", str(e)) from e

    async def mark_operations_failed(
        self,
        operation_ids: list[int],
        error_message: str,
    ) -> int:
        """Mark multiple operations as FAILED with error message.

        Args:
            operation_ids: List of operation IDs to mark as failed
            error_message: Error message to set

        Returns:
            Number of operations updated
        """
        if not operation_ids:
            return 0

        try:
            stmt = (
                update(Operation)
                .where(Operation.id.in_(operation_ids))
                .values(
                    status=OperationStatus.FAILED,
                    error_message=error_message,
                    completed_at=datetime.now(UTC),
                )
            )
            result = await self.session.execute(stmt)
            await self.session.flush()

            logger.info(
                "Marked %d operations as failed: %s",
                result.rowcount,
                error_message,
            )
            return result.rowcount
        except Exception as e:
            logger.error("Failed to mark operations as failed: %s", e, exc_info=True)
            raise DatabaseOperationError("update", "operations", str(e)) from e
