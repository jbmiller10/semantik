"""
Operation API v2 endpoints.

This module provides RESTful API endpoints for operation management
in the new collection-centric architecture.
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect, status

from packages.shared.database import get_db
from packages.shared.database.exceptions import AccessDeniedError, EntityNotFoundError, ValidationError
from packages.shared.database.models import OperationStatus, OperationType
from packages.shared.database.repositories.operation_repository import OperationRepository
from packages.webui.api.schemas import ErrorResponse, OperationResponse
from packages.webui.auth import get_current_user, get_current_user_websocket
from packages.webui.services.factory import get_operation_service
from packages.webui.services.operation_service import OperationService
from packages.webui.websocket_manager import ws_manager

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
    service: OperationService = Depends(get_operation_service),
) -> OperationResponse:
    """Get detailed information about a specific operation.

    Returns full details about an operation including its status, configuration,
    and any error messages.
    """
    try:
        operation = await service.get_operation(
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
    service: OperationService = Depends(get_operation_service),
) -> OperationResponse:
    """Cancel a pending or processing operation.

    Attempts to cancel an operation. Only operations in PENDING or PROCESSING
    state can be cancelled. The actual cancellation depends on the task
    implementation and may not be immediate.
    """
    try:
        operation = await service.cancel_operation(
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
    service: OperationService = Depends(get_operation_service),
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

        operations, total = await service.list_operations(
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


# WebSocket handler for operation progress - export this separately so it can be mounted at the app level
async def operation_websocket(websocket: WebSocket, operation_id: str) -> None:
    """WebSocket for real-time operation progress updates.

    Authentication is handled via JWT token passed as query parameter.
    The token should be passed as ?token=<jwt_token> in the WebSocket URL.

    The WebSocket will:
    1. Authenticate the user via JWT token
    2. Verify user has permission to access the operation
    3. Subscribe to Redis updates for the operation
    4. Stream progress updates until the operation completes or the connection closes
    """
    # Extract token from query parameters
    token = websocket.query_params.get("token")

    try:
        # Authenticate the user
        user = await get_current_user_websocket(token)
        user_id = str(user["id"])
    except ValueError as e:
        # Authentication failed
        await websocket.close(code=1008, reason=str(e))
        return
    except Exception as e:
        logger.error(f"WebSocket authentication error: {e}")
        await websocket.close(code=1011, reason="Internal server error")
        return

    # Verify user has permission to access this operation
    try:
        # Get a database session and create service
        async for db in get_db():
            operation_repo = OperationRepository(db)
            service = OperationService(db, operation_repo)
            await service.verify_websocket_access(
                operation_uuid=operation_id,
                user_id=int(user["id"]),
            )
            break  # Exit after first iteration
    except EntityNotFoundError:
        await websocket.close(code=1008, reason=f"Operation '{operation_id}' not found")
        return
    except AccessDeniedError:
        await websocket.close(code=1008, reason="You don't have access to this operation")
        return
    except Exception as e:
        logger.error(f"Error verifying operation access: {e}")
        await websocket.close(code=1011, reason="Internal server error")
        return

    # Authentication and authorization successful, connect the WebSocket
    await ws_manager.connect(websocket, operation_id, user_id)

    try:
        # Keep the connection alive and handle any incoming messages
        while True:
            # We don't expect the client to send data, but we need to keep receiving
            # to detect disconnections properly
            try:
                data = await websocket.receive_json()
                # Handle ping messages to keep connection alive
                if data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
            except Exception:
                # If receiving fails, the connection is likely closed
                break
    except WebSocketDisconnect:
        pass
    finally:
        # Ensure we always disconnect properly to clean up resources
        await ws_manager.disconnect(websocket, operation_id, user_id)
