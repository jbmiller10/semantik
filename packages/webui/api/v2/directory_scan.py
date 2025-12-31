"""
Directory scan API v2 endpoints.

This module provides RESTful API endpoints for scanning directories
without creating collections, useful for previewing directory contents.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect, status

from shared.database.exceptions import ValidationError
from webui.api.schemas import DirectoryScanProgress, DirectoryScanRequest, DirectoryScanResponse, ErrorResponse
from webui.auth import get_current_user, get_current_user_websocket
from webui.rate_limiter import limiter
from webui.services.directory_scan_service import DirectoryScanService
from webui.services.factory import get_directory_scan_service

# Use the scalable WebSocket manager for horizontal scaling
from webui.websocket.scalable_manager import scalable_ws_manager as ws_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/directory-scan", tags=["directory-scan-v2"])


@router.post(
    "/preview",
    response_model=DirectoryScanResponse,
    status_code=200,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request data"},
        403: {"model": ErrorResponse, "description": "Access denied to directory"},
        404: {"model": ErrorResponse, "description": "Directory not found"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
@limiter.limit("30/minute")
async def scan_directory_preview(
    request: Request,  # noqa: ARG001
    scan_request: DirectoryScanRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
    service: DirectoryScanService = Depends(get_directory_scan_service),
) -> DirectoryScanResponse:
    """Scan a directory and return a preview of supported documents.

    This endpoint scans the specified directory for supported file types without
    creating a collection. It's useful for previewing what documents would be
    included before creating a collection.

    The scan is performed asynchronously with progress updates available via
    WebSocket connection using the provided scan_id.

    Supported file types:
    - PDF (.pdf)
    - Word documents (.docx, .doc)
    - Text files (.txt, .text)
    - PowerPoint (.pptx)
    - Email (.eml)
    - Markdown (.md)
    - HTML (.html)
    """
    try:
        # Validate path
        scan_path = Path(scan_request.path)
        if not scan_path.is_absolute():
            raise ValidationError("Path must be absolute")

        # Start the scan asynchronously
        scan_task = asyncio.create_task(
            service.scan_directory_preview(
                path=scan_request.path,
                scan_id=scan_request.scan_id,
                user_id=int(current_user["id"]),
                recursive=scan_request.recursive,
                include_patterns=scan_request.include_patterns,
                exclude_patterns=scan_request.exclude_patterns,
            )
        )

        # Wait a short time to see if there are immediate errors
        try:
            # If we get here, the scan completed very quickly (likely empty directory)
            return await asyncio.wait_for(scan_task, timeout=0.5)
        except TimeoutError:
            # Normal case - scan is still running
            # Send initial "started" message via WebSocket to the user
            progress_msg = DirectoryScanProgress(
                type="started",
                scan_id=scan_request.scan_id,
                data={
                    "path": scan_request.path,
                    "recursive": scan_request.recursive,
                    "message": "Directory scan started",
                },
            )
            # Send to the user who initiated the scan
            await ws_manager.send_to_user(str(current_user["id"]), progress_msg.model_dump())

            # Return a partial response indicating scan is in progress
            return DirectoryScanResponse(
                scan_id=scan_request.scan_id,
                path=scan_request.path,
                files=[],
                total_files=0,
                total_size=0,
                warnings=["Scan in progress - connect to WebSocket for real-time updates"],
            )

    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Directory not found: {scan_request.path}",
        ) from e
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied to directory: {scan_request.path}",
        ) from e
    except Exception as e:
        logger.error("Failed to scan directory: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to scan directory",
        ) from e


# WebSocket handler for directory scan progress - export this separately so it can be mounted at the app level
async def directory_scan_websocket(websocket: WebSocket, scan_id: str) -> None:
    """WebSocket for real-time directory scan progress updates.

    Authentication is handled via JWT token passed as query parameter.
    The token should be passed as ?token=<jwt_token> in the WebSocket URL.

    The WebSocket will:
    1. Authenticate the user via JWT token
    2. Subscribe to Redis updates for the scan
    3. Stream progress updates until the scan completes or the connection closes

    Message types sent:
    - started: Scan has begun
    - counting: Counting files in directory
    - progress: Periodic progress updates with files_scanned, total_files, percentage
    - warning: Non-fatal warnings (e.g., permission denied on subdirectory)
    - error: Fatal errors that stop the scan
    - completed: Scan finished successfully
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
        logger.error("WebSocket authentication error: %s", e, exc_info=True)
        await websocket.close(code=1011, reason="Internal server error")
        return

    # Authentication successful, connect the WebSocket
    # Use scan_id as operation_id for directory scan tracking
    connection_id = await ws_manager.connect(websocket, user_id, operation_id=scan_id)

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
        if "connection_id" in locals():
            await ws_manager.disconnect(connection_id)
