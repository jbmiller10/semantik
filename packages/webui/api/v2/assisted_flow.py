"""Assisted flow API endpoints.

Provides endpoints for the Claude Agent SDK-powered pipeline
configuration assistant.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from shared.database import get_db
from webui.api.v2.assisted_flow_schemas import (
    SendMessageRequest,
    StartFlowRequest,
    StartFlowResponse,
)
from webui.auth import get_current_user
from webui.services.assisted_flow.sdk_service import (
    SDKNotAvailableError,
    SDKSessionError,
    get_session_client,
    send_message,
)
from webui.services.assisted_flow.source_stats import get_source_stats

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/assisted-flow", tags=["assisted-flow"])


@router.post("/start", response_model=StartFlowResponse)
async def start_assisted_flow(
    request: StartFlowRequest,
    db: AsyncSession = Depends(get_db),
    user: dict[str, Any] = Depends(get_current_user),
) -> StartFlowResponse:
    """Start a new assisted flow session.

    Creates a new SDK session with the pipeline configuration tools
    and returns a session ID for subsequent message requests.

    Args:
        request: Contains source_id to configure
        db: Database session
        user: Authenticated user

    Returns:
        Session ID and source info
    """
    try:
        # Get source stats for initial context
        stats = await get_source_stats(db, request.source_id)

        # TODO: Initialize SDK session with tools and prompts
        # For now, return a placeholder
        session_id = f"session_{request.source_id}"

        return StartFlowResponse(
            session_id=session_id,
            source_name=stats["source_name"],
        )

    except Exception as e:
        logger.error(f"Failed to start assisted flow: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.post(
    "/{session_id}/messages/stream",
    responses={
        200: {"content": {"text/event-stream": {}}},
        404: {"description": "Session not found"},
        503: {"description": "SDK service unavailable"},
    },
)
async def send_message_stream(
    session_id: str,
    request: SendMessageRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
) -> StreamingResponse:
    """Send a message and stream the agent's response via SSE.

    Returns Server-Sent Events as the agent processes the request.
    Each event has the format:
        event: {type}
        data: {json}

    Event types:
    - text: Text content from the agent
    - tool_use: Tool being executed
    - tool_result: Tool execution result
    - done: Stream complete
    - error: Error occurred
    """
    # Verify session exists
    client = await get_session_client(session_id)
    if not client:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found or expired",
        )

    message = request.message

    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate SSE events from SDK client streaming."""
        try:
            # Send message to SDK
            sdk_client = await send_message(session_id, message)

            # Stream response
            async for msg in sdk_client.receive_response():
                # Format message as SSE event
                event_type = getattr(msg, "type", "message")
                event_data = {
                    "type": event_type,
                }

                # Extract content based on message type
                if hasattr(msg, "content"):
                    event_data["content"] = msg.content
                if hasattr(msg, "tool_name"):
                    event_data["tool_name"] = msg.tool_name
                if hasattr(msg, "result"):
                    event_data["result"] = msg.result

                yield f"event: {event_type}\ndata: {json.dumps(event_data)}\n\n"

            # Send done event
            yield f"event: done\ndata: {json.dumps({'status': 'complete'})}\n\n"

        except SDKNotAvailableError as e:
            logger.error(f"SDK not available: {e}")
            error_data = json.dumps({
                "message": "Claude Code CLI is not installed",
                "code": "sdk_not_available",
            })
            yield f"event: error\ndata: {error_data}\n\n"

        except SDKSessionError as e:
            logger.error(f"SDK session error: {e}")
            error_data = json.dumps({
                "message": str(e),
                "code": "session_error",
            })
            yield f"event: error\ndata: {error_data}\n\n"

        except Exception as e:
            logger.exception(f"Streaming error for session {session_id}: {e}")
            error_data = json.dumps({"message": str(e)})
            yield f"event: error\ndata: {error_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
