"""Operation Service for managing operation-related business logic."""

import logging
from typing import Any

from shared.database.exceptions import AccessDeniedError, EntityNotFoundError, ValidationError
from shared.database.models import Operation, OperationStatus, OperationType
from shared.database.repositories.operation_repository import OperationRepository
from sqlalchemy.ext.asyncio import AsyncSession
from webui.celery_app import celery_app

logger = logging.getLogger(__name__)


class OperationService:
    """Service for managing operation-related business logic."""

    def __init__(
        self,
        db_session: AsyncSession,
        operation_repo: OperationRepository,
    ):
        """Initialize the operation service."""
        self.db_session = db_session
        self.operation_repo = operation_repo

    async def get_operation(self, operation_uuid: str, user_id: int) -> Operation:
        """Get an operation by UUID with permission check.

        Args:
            operation_uuid: UUID of the operation
            user_id: ID of the user requesting the operation

        Returns:
            Operation object

        Raises:
            EntityNotFoundError: If operation not found
            AccessDeniedError: If user doesn't have permission
        """
        return await self.operation_repo.get_by_uuid_with_permission_check(
            operation_uuid=operation_uuid,
            user_id=user_id,
        )

    async def cancel_operation(self, operation_uuid: str, user_id: int) -> Operation:
        """Cancel a pending or processing operation.

        Args:
            operation_uuid: UUID of the operation to cancel
            user_id: ID of the user cancelling the operation

        Returns:
            Cancelled operation object

        Raises:
            EntityNotFoundError: If operation not found
            AccessDeniedError: If user doesn't have permission
            ValidationError: If operation cannot be cancelled
        """
        # Cancel the operation in database
        operation = await self.operation_repo.cancel(
            operation_uuid=operation_uuid,
            user_id=user_id,
        )

        # If operation has a Celery task ID, attempt to revoke it
        if operation.task_id:
            try:
                celery_app.control.revoke(operation.task_id, terminate=True)
                logger.info(f"Revoked Celery task {operation.task_id} for operation {operation_uuid}")
            except Exception as e:
                logger.warning(f"Failed to revoke Celery task {operation.task_id}: {e}")
                # Continue even if Celery revoke fails

        # Commit the transaction
        await self.db_session.commit()

        return operation

    async def list_operations(
        self,
        user_id: int,
        status_list: list[OperationStatus] | None = None,
        operation_type: OperationType | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[Operation], int]:
        """List operations for a user with optional filtering.

        Args:
            user_id: ID of the user
            status_list: Optional list of statuses to filter by
            operation_type: Optional operation type to filter by
            offset: Pagination offset
            limit: Pagination limit

        Returns:
            Tuple of (operations list, total count)
        """
        return await self.operation_repo.list_for_user(
            user_id=user_id,
            status_list=status_list,
            operation_type=operation_type,
            offset=offset,
            limit=limit,
        )

    async def verify_websocket_access(self, operation_uuid: str, user_id: int) -> Operation:
        """Verify user has access to operation for WebSocket connection.

        Args:
            operation_uuid: UUID of the operation
            user_id: ID of the user

        Returns:
            Operation object if access is granted

        Raises:
            EntityNotFoundError: If operation not found
            AccessDeniedError: If user doesn't have permission
        """
        # This method is essentially the same as get_operation, but we keep it
        # separate for clarity and potential future WebSocket-specific logic
        return await self.get_operation(operation_uuid, user_id)