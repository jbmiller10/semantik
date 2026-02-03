"""Assisted flow API endpoints.

Provides endpoints for the Claude Agent SDK-powered pipeline
configuration assistant.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from shared.database import get_db
from shared.database.exceptions import AccessDeniedError, EntityNotFoundError
from webui.api.v2.assisted_flow_schemas import (
    SendMessageRequest,
    StartFlowRequest,
    StartFlowResponse,
)
from webui.auth import get_current_user
from webui.services.assisted_flow.sdk_service import (
    SDKNotAvailableError,
    SDKSessionError,
    create_sdk_session,
    get_session_client,
    send_message,
)
from webui.services.assisted_flow.source_stats import get_source_stats

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/assisted-flow", tags=["assisted-flow"])


@router.post("/start", response_model=StartFlowResponse)
async def start_assisted_flow(
    request: StartFlowRequest,
    db: AsyncSession = Depends(get_db),
    _user: dict[str, Any] = Depends(get_current_user),
) -> StartFlowResponse:
    """Start a new assisted flow session.

    Creates a new SDK session with the pipeline configuration tools
    and returns a session ID for subsequent message requests.

    Supports two modes:
    - source_id: Configure an existing source from the database
    - inline_source: Configure a new source (will be created when pipeline is applied)

    Args:
        request: Contains source_id OR inline_source config
        db: Database session
        user: Authenticated user

    Returns:
        Session ID and source info
    """
    try:
        user_id = _user.get("id")
        if not isinstance(user_id, int):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid user session")

        if request.source_id is not None:
            # Existing source mode
            stats = await get_source_stats(db, user_id=user_id, source_id=request.source_id)
            source_name = stats["source_name"]

            session_id, _client = await create_sdk_session(
                user_id=user_id,
                source_id=request.source_id,
                source_stats=stats,
            )

        else:
            # Inline source mode
            inline = request.inline_source
            assert inline is not None  # Validated by schema

            # Derive a display name from the config
            config = inline.source_config
            if inline.source_type == "directory":
                source_name = str(config.get("path", "New Directory Source"))
            elif inline.source_type == "git":
                source_name = str(config.get("repo_url", config.get("repository_url", "New Git Source")))
            elif inline.source_type == "imap":
                username = str(config.get("username", ""))
                host = str(config.get("host", ""))
                source_name = f"{username}@{host}" if username and host else "New IMAP Source"
            else:
                source_name = f"New {inline.source_type.title()} Source"

            # For inline source mode, treat inline config as source stats context (redact secrets).
            stats = {
                "source_name": source_name,
                "source_type": inline.source_type,
                "source_path": source_name,
                "source_config": {k: v for k, v in config.items() if k not in (request.secrets or {})},
            }
            session_id, _client = await create_sdk_session(
                user_id=user_id,
                source_id=None,
                source_stats=stats,
            )

        return StartFlowResponse(
            session_id=session_id,
            source_name=source_name,
        )

    except SDKNotAvailableError as e:
        logger.error("Claude Code CLI not available: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        ) from e
    except SDKSessionError as e:
        logger.error("Failed to create assisted flow session: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e
    except EntityNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e
    except AccessDeniedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        ) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start assisted flow: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e


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
    _current_user: dict[str, Any] = Depends(get_current_user),
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
    user_id = _current_user.get("id")
    if not isinstance(user_id, int):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid user session")

    client = await get_session_client(session_id, user_id=user_id)
    if not client:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found or expired",
        )

    message = request.message

    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate SSE events from SDK client streaming."""
        tool_names_by_id: dict[str, str] = {}

        def _sse(event: str, data: dict[str, Any]) -> str:
            return f"event: {event}\ndata: {json.dumps(data)}\n\n"

        from claude_agent_sdk.types import (
            AssistantMessage,
            ResultMessage,
            StreamEvent,
            TextBlock,
            ToolResultBlock,
            ToolUseBlock,
        )

        try:
            # Send message to SDK
            sdk_client = await send_message(session_id, message, user_id=user_id)

            # Stream response
            async for msg in sdk_client.receive_response():
                if isinstance(msg, StreamEvent):
                    # Best-effort incremental text streaming.
                    event = msg.event or {}
                    if event.get("type") == "content_block_delta":
                        delta = event.get("delta") or {}
                        text = delta.get("text")
                        if isinstance(text, str) and text:
                            yield _sse("text", {"content": text})
                    continue

                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            if block.text:
                                yield _sse("text", {"content": block.text})
                        elif isinstance(block, ToolUseBlock):
                            tool_names_by_id[block.id] = block.name
                            yield _sse(
                                "tool_use",
                                {
                                    "tool_use_id": block.id,
                                    "tool_name": block.name,
                                    "arguments": block.input,
                                },
                            )
                        elif isinstance(block, ToolResultBlock):
                            yield _sse(
                                "tool_result",
                                {
                                    "tool_use_id": block.tool_use_id,
                                    "tool_name": tool_names_by_id.get(block.tool_use_id),
                                    "result": block.content,
                                    "success": not bool(block.is_error),
                                },
                            )
                    continue

                if isinstance(msg, ResultMessage):
                    yield _sse(
                        "done",
                        {
                            "status": "complete",
                            "is_error": msg.is_error,
                            "duration_ms": msg.duration_ms,
                            "total_cost_usd": msg.total_cost_usd,
                            "num_turns": msg.num_turns,
                        },
                    )
                    return

        except SDKNotAvailableError as e:
            logger.error(f"SDK not available: {e}")
            error_data = json.dumps(
                {
                    "message": "Claude Code CLI is not installed",
                    "code": "sdk_not_available",
                }
            )
            yield f"event: error\ndata: {error_data}\n\n"

        except SDKSessionError as e:
            logger.error(f"SDK session error: {e}")
            error_data = json.dumps(
                {
                    "message": str(e),
                    "code": "session_error",
                }
            )
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
