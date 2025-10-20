"""
Cancel Operation Use Case.

Handles cancellation of in-progress chunking operations with proper cleanup.
"""

from datetime import UTC, datetime
from typing import Any

from packages.shared.chunking.application.dto.requests import CancelOperationRequest
from packages.shared.chunking.application.dto.responses import CancelOperationResponse, OperationStatus
from packages.shared.chunking.application.interfaces.services import NotificationService, UnitOfWork


class CancelOperationUseCase:
    """
    Use case for cancelling chunking operations.

    This use case:
    - Validates cancellation is allowed
    - Updates operation status
    - Optionally cleans up created chunks
    - Removes checkpoints
    - Ensures transactional consistency
    """

    def __init__(self, unit_of_work: UnitOfWork, notification_service: NotificationService):
        """
        Initialize the use case with dependencies.

        Args:
            unit_of_work: UoW for transaction management
            notification_service: Service for notifications
        """
        self.unit_of_work = unit_of_work
        self.notification_service = notification_service

    async def execute(self, request: CancelOperationRequest) -> CancelOperationResponse:
        """
        Execute the cancellation use case.

        Manages proper cleanup and state transitions.

        Args:
            request: Cancellation request with operation ID

        Returns:
            CancelOperationResponse with cancellation details

        Raises:
            ValueError: If operation not found or cannot be cancelled
        """
        cancellation_time = datetime.now(tz=UTC)
        chunks_deleted = 0

        # Transaction boundary
        async with self.unit_of_work:
            try:
                # 1. Validate request
                request.validate()

                # 2. Find operation
                operation = await self.unit_of_work.operations.find_by_id(request.operation_id)
                if not operation:
                    raise ValueError(f"Operation not found: {request.operation_id}")

                # 3. Check if operation can be cancelled
                previous_status = self._map_status(operation.status)
                if not self._can_cancel(previous_status, request.force):
                    raise ValueError(
                        f"Operation cannot be cancelled. Current status: {previous_status.value}. "
                        f"Use force=True to override."
                    )

                # 4. Determine new status
                new_status = self._determine_new_status(previous_status, operation)

                # 5. Clean up chunks if requested
                if request.cleanup_chunks:
                    chunks_deleted = await self.unit_of_work.chunks.delete_by_operation(request.operation_id)

                # 6. Clean up checkpoints
                checkpoint_count = await self.unit_of_work.checkpoints.delete_checkpoints(request.operation_id)

                # 7. Update operation status based on determined new status
                if new_status == OperationStatus.PARTIALLY_COMPLETED:
                    # Update to partially completed status
                    await self.unit_of_work.operations.update_status(
                        operation_id=request.operation_id,
                        status=OperationStatus.PARTIALLY_COMPLETED,
                        error_message=request.reason,
                    )
                else:
                    # Mark as cancelled
                    await self.unit_of_work.operations.mark_cancelled(
                        operation_id=request.operation_id, reason=request.reason
                    )

                # 8. Update document status if needed
                if hasattr(operation, "document_id") and operation.document_id:
                    # Check if there are other operations for this document
                    other_ops = await self.unit_of_work.operations.find_by_document(operation.document_id)
                    active_ops = [
                        op for op in other_ops if op.id != request.operation_id and op.status == "in_progress"
                    ]

                    # If no other active operations, update document status
                    if not active_ops:
                        await self.unit_of_work.documents.update_chunking_status(
                            document_id=operation.document_id,
                            status="cancelled" if new_status == OperationStatus.CANCELLED else "partial",
                        )

                # Commit transaction
                await self.unit_of_work.commit()

                # 9. Send cancellation notification (after commit)
                await self.notification_service.notify_operation_cancelled(
                    operation_id=request.operation_id, reason=request.reason
                )

                # 10. Log cancellation details
                await self.notification_service.notify_error(
                    error=RuntimeError("Operation cancelled"),
                    context={
                        "event": "operation_cancelled",
                        "operation_id": request.operation_id,
                        "previous_status": previous_status.value,
                        "new_status": new_status.value,
                        "chunks_deleted": chunks_deleted,
                        "checkpoints_deleted": checkpoint_count,
                        "reason": request.reason,
                        "forced": request.force,
                    },
                )

                # 11. Return response
                return CancelOperationResponse(
                    operation_id=request.operation_id,
                    previous_status=previous_status,
                    new_status=new_status,
                    chunks_deleted=chunks_deleted,
                    cancellation_time=cancellation_time,
                    cancellation_reason=request.reason,
                    cleanup_performed=request.cleanup_chunks,
                )

            except Exception as e:
                # Rollback on error
                await self.unit_of_work.rollback()

                # Notify error
                await self.notification_service.notify_error(
                    error=e,
                    context={
                        "operation_id": request.operation_id,
                        "use_case": "cancel_operation",
                        "action": "cancellation_failed",
                    },
                )

                raise

    def _map_status(self, status_string: str) -> OperationStatus:
        """
        Map string status to enum.

        Args:
            status_string: Status as string

        Returns:
            OperationStatus enum value
        """
        status_mapping = {
            "pending": OperationStatus.PENDING,
            "in_progress": OperationStatus.IN_PROGRESS,
            "processing": OperationStatus.IN_PROGRESS,  # Handle both forms
            "completed": OperationStatus.COMPLETED,
            "failed": OperationStatus.FAILED,
            "cancelled": OperationStatus.CANCELLED,
            "partially_completed": OperationStatus.PARTIALLY_COMPLETED,
        }
        return status_mapping.get(status_string.lower(), OperationStatus.PENDING)

    def _can_cancel(self, status: OperationStatus, force: bool) -> bool:
        """
        Check if operation can be cancelled.

        Args:
            status: Current operation status
            force: Whether to force cancellation

        Returns:
            True if cancellation is allowed
        """
        # Always allow if forcing
        if force:
            return True

        # Can cancel pending and in-progress operations
        cancellable_statuses = [OperationStatus.PENDING, OperationStatus.IN_PROGRESS]
        return status in cancellable_statuses

    def _determine_new_status(self, previous_status: OperationStatus, operation: Any) -> OperationStatus:
        """
        Determine the new status after cancellation.

        Args:
            previous_status: Previous operation status
            operation: Operation entity

        Returns:
            New operation status
        """
        # If operation was pending, it's simply cancelled
        if previous_status == OperationStatus.PENDING:
            return OperationStatus.CANCELLED

        # If in progress, check if any chunks were created
        if previous_status == OperationStatus.IN_PROGRESS:
            if hasattr(operation, "chunks_processed") and operation.chunks_processed > 0:
                # Some chunks were created before cancellation
                return OperationStatus.PARTIALLY_COMPLETED
            # No chunks created yet
            return OperationStatus.CANCELLED

        # For completed or failed, status doesn't change (forced cancellation)
        if previous_status in [OperationStatus.COMPLETED, OperationStatus.FAILED]:
            return previous_status

        # Default to cancelled
        return OperationStatus.CANCELLED
