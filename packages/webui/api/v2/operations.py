"""
Operation API v2 endpoints.

This module provides RESTful API endpoints for operation management
in the new collection-centric architecture.
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from packages.shared.database import get_db
from packages.shared.database.exceptions import AccessDeniedError, EntityNotFoundError, ValidationError
from packages.shared.database.models import OperationStatus, OperationType
from packages.shared.database.repositories.operation_repository import OperationRepository
from packages.webui.api.schemas import ErrorResponse, OperationResponse
from packages.webui.auth import get_current_user
from packages.webui.celery_app import celery_app
from packages.webui.dependencies import get_operation_repository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/operations", tags=["operations-v2"])


@router.get(
    "/{operation_uuid}",
    response_model=OperationResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Operation not found"},
        403: {"model": ErrorResponse, "description": "Access denied"},
    },
)
async def get_operation(
    operation_uuid: str,
    current_user: dict[str, Any] = Depends(get_current_user),
    repo: OperationRepository = Depends(get_operation_repository),
) -> OperationResponse:
    """Get detailed information about a specific operation.

    Returns full details about an operation including its status, configuration,
    and any error messages.
    """
    try:
        operation = await repo.get_by_uuid_with_permission_check(
            operation_uuid=operation_uuid,
            user_id=int(current_user["id"]),
        )

        return OperationResponse(
            id=operation.uuid,
            collection_id=operation.collection_id,
            type=operation.type.value,
            status=operation.status.value,
            config=operation.config,
            error_message=operation.error_message,
            created_at=operation.created_at,
            started_at=operation.started_at,
            completed_at=operation.completed_at,
        )

    except EntityNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Operation '{operation_uuid}' not found",
        ) from e
    except AccessDeniedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this operation",
        ) from e
    except Exception as e:
        logger.error(f"Failed to get operation: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get operation",
        ) from e


@router.delete(
    "/{operation_uuid}",
    response_model=OperationResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Operation not found"},
        403: {"model": ErrorResponse, "description": "Access denied"},
        400: {"model": ErrorResponse, "description": "Cannot cancel operation"},
    },
)
async def cancel_operation(
    operation_uuid: str,
    current_user: dict[str, Any] = Depends(get_current_user),
    repo: OperationRepository = Depends(get_operation_repository),
    db: AsyncSession = Depends(get_db),
) -> OperationResponse:
    """Cancel a pending or processing operation.

    Attempts to cancel an operation. Only operations in PENDING or PROCESSING
    state can be cancelled. The actual cancellation depends on the task
    implementation and may not be immediate.
    """
    try:
        # Cancel the operation in database
        operation = await repo.cancel(
            operation_uuid=operation_uuid,
            user_id=int(current_user["id"]),
        )

        # If operation has a Celery task ID, attempt to revoke it
        if operation.task_id:
            try:
                celery_app.control.revoke(operation.task_id, terminate=True)
                logger.info(f"Revoked Celery task {operation.task_id} for operation {operation_uuid}")
            except Exception as e:
                logger.warning(f"Failed to revoke Celery task {operation.task_id}: {e}")
                # Continue even if Celery revoke fails

        await db.commit()

        return OperationResponse(
            id=operation.uuid,
            collection_id=operation.collection_id,
            type=operation.type.value,
            status=operation.status.value,
            config=operation.config,
            error_message=operation.error_message,
            created_at=operation.created_at,
            started_at=operation.started_at,
            completed_at=operation.completed_at,
        )

    except EntityNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Operation '{operation_uuid}' not found",
        ) from e
    except AccessDeniedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to cancel this operation",
        ) from e
    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error(f"Failed to cancel operation: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to cancel operation",
        ) from e


@router.get(
    "",
    response_model=list[OperationResponse],
    responses={
        400: {"model": ErrorResponse, "description": "Invalid parameters"},
    },
)
async def list_operations(
    status: str | None = Query(None, description="Filter by operation status (comma-separated for multiple)"),
    operation_type: str | None = Query(None, description="Filter by operation type"),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(50, ge=1, le=100, description="Items per page"),
    current_user: dict[str, Any] = Depends(get_current_user),
    repo: OperationRepository = Depends(get_operation_repository),
) -> list[OperationResponse]:
    """List operations for the current user.

    Returns a paginated list of all operations created by the current user,
    ordered by creation date (newest first).
    """
    try:
        offset = (page - 1) * per_page

        # Convert string parameters to enums if provided
        status_list = None
        if status:
            status_list = []
            # Split comma-separated statuses
            for s in status.split(","):
                s = s.strip()
                try:
                    status_list.append(OperationStatus(s))
                except ValueError:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid status: {s}. Valid values are: {[st.value for st in OperationStatus]}",
                    ) from None

        type_enum = None
        if operation_type:
            try:
                type_enum = OperationType(operation_type)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid operation type: {operation_type}. Valid values are: {[t.value for t in OperationType]}",
                ) from None

        operations, total = await repo.list_for_user(
            user_id=int(current_user["id"]),
            status_list=status_list,
            operation_type=type_enum,
            offset=offset,
            limit=per_page,
        )

        # Convert ORM objects to response models
        return [
            OperationResponse(
                id=op.uuid,
                collection_id=op.collection_id,
                type=op.type.value,
                status=op.status.value,
                config=op.config,
                error_message=op.error_message,
                created_at=op.created_at,
                started_at=op.started_at,
                completed_at=op.completed_at,
            )
            for op in operations
        ]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list operations: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to list operations",
        ) from e
