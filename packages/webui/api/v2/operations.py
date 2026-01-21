"""
Operation API v2 endpoints.

This module provides RESTful API endpoints for operation management
in the new collection-centric architecture.

Error Handling:
    REST endpoints rely on global exception handlers registered in
    middleware/exception_handlers.py. WebSocket endpoints use WebSocket
    close codes and maintain their own error handling.
"""

import ipaddress
import json
import logging
from typing import Any
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect

from shared.config import settings as shared_settings
from shared.database import get_db
from shared.database.exceptions import AccessDeniedError, EntityNotFoundError
from shared.database.repositories.operation_repository import OperationRepository
from webui.api.schemas import ErrorResponse, OperationListResponse, OperationResponse
from webui.auth import get_current_user, get_current_user_websocket
from webui.services.factory import get_operation_service
from webui.services.operation_service import OperationService

# Use the scalable WebSocket manager for horizontal scaling
from webui.websocket.scalable_manager import scalable_ws_manager as ws_manager

logger = logging.getLogger(__name__)


def _get_allowed_websocket_origins() -> list[str]:
    """Get list of allowed origins for WebSocket connections.

    Uses CORS_ORIGINS from settings with fallback defaults for development.
    """
    origins = [origin.strip() for origin in shared_settings.CORS_ORIGINS.split(",") if origin.strip()]
    # Fallback for development if no origins configured
    if not origins:
        origins = [
            "http://localhost:5173",
            "http://localhost:3000",
            "http://localhost:8080",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:8080",
        ]
    return origins


def _is_ip_same_origin(websocket: WebSocket, origin: str) -> bool:
    """Return True when the Origin is an IP literal matching the request host.

    This is a safe escape hatch for LAN/dev deployments where the UI is served from
    an IP (e.g. http://192.168.x.y:8080) but the configured CORS_ORIGINS only include
    localhost. It intentionally does *not* allow arbitrary hostnames to avoid
    weakening origin checks via DNS rebinding.
    """

    try:
        parsed_origin = urlparse(origin)
    except Exception:
        return False

    origin_host = parsed_origin.hostname
    if not origin_host:
        return False

    try:
        ipaddress.ip_address(origin_host)
    except ValueError:
        return False

    request_host = websocket.url.hostname
    if not request_host:
        return False

    try:
        ipaddress.ip_address(request_host)
    except ValueError:
        return False

    origin_port = parsed_origin.port
    if origin_port is None:
        if parsed_origin.scheme == "http":
            origin_port = 80
        elif parsed_origin.scheme == "https":
            origin_port = 443
        else:
            return False

    request_port = websocket.url.port
    if request_port is None:
        if websocket.url.scheme == "ws":
            request_port = 80
        elif websocket.url.scheme == "wss":
            request_port = 443
        else:
            return False

    return origin_host == request_host and origin_port == request_port


async def _validate_websocket_origin(websocket: WebSocket) -> bool:
    """Validate WebSocket Origin header against allowed origins.

    Returns True if origin is valid or not present.
    Returns False if origin is present but not in allowed list.
    """
    origin = websocket.headers.get("origin")
    if not origin:
        # Non-browser clients may omit Origin header.
        return True

    allowed_origins = _get_allowed_websocket_origins()
    if origin not in allowed_origins:
        if _is_ip_same_origin(websocket, origin):
            return True
        logger.warning(
            "WebSocket connection rejected: origin %s not in allowed origins %s",
            origin,
            allowed_origins,
        )
        return False
    return True


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


@router.get(
    "",
    response_model=OperationListResponse,
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
) -> OperationListResponse:
    """List operations for the current user.

    Returns a paginated list of all operations created by the current user,
    ordered by creation date (newest first).
    """
    offset = (page - 1) * per_page

    # Delegate all parsing and filtering logic to service
    operations, total = await service.list_operations_with_filters(
        user_id=int(current_user["id"]),
        status=status,
        operation_type=operation_type,
        offset=offset,
        limit=per_page,
    )

    operations_response = [
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
    return OperationListResponse(
        operations=operations_response,
        total=total,
        page=page,
        per_page=per_page,
    )


# WebSocket handler for operation progress - export this separately so it can be mounted at the app level
async def operation_websocket(websocket: WebSocket, operation_id: str) -> None:
    """WebSocket for real-time operation progress updates.

    Authentication is handled via JWT token passed as query parameter.
    The token should be passed as ?token=<jwt_token> in the WebSocket URL.

    The WebSocket will:
    1. Validate Origin header against allowed origins
    2. Authenticate the user via JWT token
    3. Verify user has permission to access the operation
    4. Subscribe to Redis updates for the operation
    5. Stream progress updates until the operation completes or the connection closes
    """
    # Validate Origin header to prevent cross-site WebSocket hijacking
    if not await _validate_websocket_origin(websocket):
        await websocket.close(code=4003, reason="Origin not allowed")
        return

    # Extract token from Sec-WebSocket-Protocol header (preferred, more secure)
    # Format: "access_token.<jwt_token>"
    # Falls back to query param for backward compatibility (deprecated)
    token = None
    accepted_subprotocol = None
    protocol_header = websocket.headers.get("sec-websocket-protocol", "")
    for protocol in protocol_header.split(","):
        protocol = protocol.strip()
        if protocol.startswith("access_token."):
            token = protocol[len("access_token.") :]
            accepted_subprotocol = protocol  # Echo back the full subprotocol
            break

    # Fallback to query param (deprecated, will be removed in future)
    if not token:
        token = websocket.query_params.get("token")
        if token:
            logger.warning("WebSocket using deprecated query param authentication - migrate to subprotocol")

    try:
        # Authenticate the user
        user = await get_current_user_websocket(token)
        user_id = str(user["id"])
    except ValueError as e:
        # Authentication failed
        await websocket.close(code=1008, reason=str(e))
        return
    except Exception as e:
        logger.error("WebSocket authentication error: %s", e, exc_info=True)
        await websocket.close(code=1011, reason="Internal server error")
        return

    # Verify user has permission to access this operation
    try:
        db_gen = get_db()
        try:
            db = await anext(db_gen)
            operation_repo = OperationRepository(db)
            service = OperationService(db, operation_repo)
            await service.verify_websocket_access(
                operation_uuid=operation_id,
                user_id=int(user["id"]),
            )
        finally:
            await db_gen.aclose()
    except EntityNotFoundError:
        await websocket.close(code=1008, reason=f"Operation '{operation_id}' not found")
        return
    except AccessDeniedError:
        await websocket.close(code=1008, reason="You don't have access to this operation")
        return
    except Exception as e:
        logger.error("Error verifying operation access: %s", e, exc_info=True)
        await websocket.close(code=1011, reason="Internal server error")
        return

    # Authentication and authorization successful, connect the WebSocket
    # Pass the subprotocol so the server echoes it back (required by WebSocket spec)
    connection_id = await ws_manager.connect(websocket, user_id, operation_id, subprotocol=accepted_subprotocol)

    try:
        # Keep the connection alive and handle any incoming messages
        while True:
            # We don't expect the client to send data, but we need to keep receiving
            # to detect disconnections properly. Some legacy clients send plain
            # "ping" strings instead of JSON – handle those gracefully so we don't
            # tear down the socket and trigger reconnect loops.
            try:
                message = await websocket.receive()

                if message.get("type") == "websocket.disconnect":
                    break

                raw_text = message.get("text")
                if raw_text is None:
                    # Ignore binary payloads or unexpected message formats
                    continue

                if not raw_text:
                    continue

                try:
                    data = json.loads(raw_text)
                except json.JSONDecodeError:
                    if raw_text.strip().lower() == "ping":
                        await websocket.send_json({"type": "pong"})
                    else:
                        logger.debug(
                            "Ignoring non-JSON WebSocket payload for operation %s: %s",
                            operation_id,
                            raw_text[:128],
                        )
                    continue

                if data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
            except Exception:
                # If receiving fails, the connection is likely closed
                break
    except WebSocketDisconnect:
        pass
    finally:
        # Ensure we always disconnect properly to clean up resources
        await ws_manager.disconnect(connection_id)


# WebSocket handler for global operation updates
async def operation_websocket_global(websocket: WebSocket) -> None:
    """WebSocket for real-time updates of ALL operations for a user.

    Authentication is handled via JWT token passed as query parameter.
    The token should be passed as ?token=<jwt_token> in the WebSocket URL.

    The WebSocket will:
    1. Validate Origin header against allowed origins
    2. Authenticate the user via JWT token
    3. Subscribe to Redis updates for the user channel
    4. Stream progress updates for all operations belonging to the user
    """
    # Validate Origin header to prevent cross-site WebSocket hijacking
    if not await _validate_websocket_origin(websocket):
        await websocket.close(code=4003, reason="Origin not allowed")
        return

    # Extract token from Sec-WebSocket-Protocol header (preferred, more secure)
    # Format: "access_token.<jwt_token>"
    # Falls back to query param for backward compatibility (deprecated)
    token = None
    accepted_subprotocol = None
    protocol_header = websocket.headers.get("sec-websocket-protocol", "")
    for protocol in protocol_header.split(","):
        protocol = protocol.strip()
        if protocol.startswith("access_token."):
            token = protocol[len("access_token.") :]
            accepted_subprotocol = protocol  # Echo back the full subprotocol
            break

    # Fallback to query param (deprecated, will be removed in future)
    if not token:
        token = websocket.query_params.get("token")
        if token:
            logger.warning("WebSocket using deprecated query param authentication - migrate to subprotocol")

    try:
        # Authenticate the user
        user = await get_current_user_websocket(token)
        user_id = str(user["id"])
    except ValueError as e:
        # Authentication failed
        await websocket.close(code=1008, reason=str(e))
        return
    except Exception as e:
        logger.error("WebSocket authentication error: %s", e, exc_info=True)
        await websocket.close(code=1011, reason="Internal server error")
        return

    # Connect the WebSocket using only user_id (no operation_id)
    # This will subscribe to the user:{user_id} channel
    # Pass the subprotocol so the server echoes it back (required by WebSocket spec)
    connection_id = await ws_manager.connect(websocket, user_id, subprotocol=accepted_subprotocol)

    try:
        # Keep the connection alive and handle any incoming messages
        while True:
            try:
                message = await websocket.receive()

                if message.get("type") == "websocket.disconnect":
                    break

                raw_text = message.get("text")
                if raw_text is None:
                    continue

                if not raw_text:
                    continue

                try:
                    data = json.loads(raw_text)
                except json.JSONDecodeError:
                    if raw_text.strip().lower() == "ping":
                        await websocket.send_json({"type": "pong"})
                    continue

                if data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
            except Exception:
                break
    except WebSocketDisconnect:
        pass
    finally:
        await ws_manager.disconnect(connection_id)
