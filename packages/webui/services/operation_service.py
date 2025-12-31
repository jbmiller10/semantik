"""Operation Service for managing operation-related business logic."""

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from shared.database.models import Operation, OperationStatus, OperationType
from shared.database.repositories.operation_repository import OperationRepository
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
                logger.info("Revoked Celery task %s for operation %s", operation.task_id, operation_uuid)
            except Exception as e:
                logger.warning("Failed to revoke Celery task %s: %s", operation.task_id, e, exc_info=True)
                # Continue even if Celery revoke fails

        # Commit the transaction
        await self.db_session.commit()

        return operation

    async def parse_status_filter(self, status: str | None) -> list[OperationStatus] | None:
        """Parse and validate status filter string.

        This method contains the parsing logic that was previously
        in the router, ensuring proper separation of concerns.

        Args:
            status: Comma-separated status string or None

        Returns:
            List of OperationStatus enums or None

        Raises:
            ValueError: If invalid status value provided
        """
        if not status:
            return None

        status_list = []
        # Split comma-separated statuses
        for s in status.split(","):
            s = s.strip()
            try:
                status_list.append(OperationStatus(s))
            except ValueError:
                raise ValueError(
                    f"Invalid status: {s}. Valid values are: {[st.value for st in OperationStatus]}"
                ) from None

        return status_list

    async def parse_type_filter(self, operation_type: str | None) -> OperationType | None:
        """Parse and validate operation type filter.

        This method contains the parsing logic that was previously
        in the router, ensuring proper separation of concerns.

        Args:
            operation_type: Operation type string or None

        Returns:
            OperationType enum or None

        Raises:
            ValueError: If invalid operation type provided
        """
        if not operation_type:
            return None

        try:
            return OperationType(operation_type)
        except ValueError:
            raise ValueError(
                f"Invalid operation type: {operation_type}. Valid values are: {[t.value for t in OperationType]}"
            ) from None

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
        return await self.operation_repo.list_for_user(  # type: ignore[no-any-return]
            user_id=user_id,
            status_list=status_list,
            operation_type=operation_type,
            offset=offset,
            limit=limit,
        )

    async def list_operations_with_filters(
        self,
        user_id: int,
        status: str | None = None,
        operation_type: str | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[Operation], int]:
        """List operations with string-based filters.

        This method combines filter parsing and listing, handling
        all business logic that was previously in the router.

        Args:
            user_id: ID of the user
            status: Comma-separated status string
            operation_type: Operation type string
            offset: Pagination offset
            limit: Pagination limit

        Returns:
            Tuple of (operations list, total count)

        Raises:
            ValueError: If invalid filter values provided
        """
        # Parse filters
        status_list = await self.parse_status_filter(status)
        type_enum = await self.parse_type_filter(operation_type)

        # List with parsed filters
        return await self.list_operations(
            user_id=user_id,
            status_list=status_list,
            operation_type=type_enum,
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
