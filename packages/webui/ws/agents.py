"""WebSocket endpoint for agent streaming.

This module provides real-time WebSocket streaming for agent executions,
allowing clients to receive partial responses as they're generated.

Protocol:
    Client → Server:
        {"type": "execute", "prompt": "...", "tools": [...], "config": {...}}
        {"type": "interrupt"}

    Server → Client:
        {"type": "message", "message": {...}}  # AgentMessage dict
        {"type": "complete"}
        {"type": "interrupted"}
        {"type": "error", "error": {"message": "...", "code": "..."}}
        {"type": "pong"}  # Response to "ping"
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from fastapi import WebSocket  # noqa: TCH002 - used at runtime
from starlette.websockets import WebSocketDisconnect, WebSocketState

from shared.agents.exceptions import AgentInterruptedError, SessionNotFoundError
from shared.agents.types import AgentContext
from shared.config import settings as shared_settings
from shared.database.exceptions import AccessDeniedError
from webui.auth import get_current_user_websocket
from webui.dependencies import get_db
from webui.services.factory import create_agent_service

if TYPE_CHECKING:
    from shared.database.models.agent_session import AgentSession
    from webui.services.agent_service import AgentService

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


async def _validate_websocket_origin(websocket: WebSocket) -> bool:
    """Validate WebSocket Origin header against allowed origins.

    Returns True if origin is valid or not present (same-origin requests may not send Origin).
    Returns False if origin is present but not in allowed list.
    """
    origin = websocket.headers.get("origin")
    if not origin:
        # Same-origin requests may not include Origin header
        return True

    allowed_origins = _get_allowed_websocket_origins()
    if origin not in allowed_origins:
        logger.warning(
            "WebSocket connection rejected: origin %s not in allowed origins %s",
            origin,
            allowed_origins,
        )
        return False
    return True


async def agent_websocket_endpoint(websocket: WebSocket, session_id: str) -> None:
    """WebSocket endpoint for real-time agent streaming.

    Args:
        websocket: The WebSocket connection.
        session_id: External session ID to connect to.

    Authentication:
        Token is passed via Sec-WebSocket-Protocol header as "access_token.<jwt>"
        or via query parameter (deprecated).

    Protocol:
        1. Client connects with session_id in URL path
        2. Client sends execute message with prompt
        3. Server streams messages until complete
        4. Client can send interrupt to stop execution
    """
    # 1. Validate origin (prevent CSWSH attacks)
    if not await _validate_websocket_origin(websocket):
        await websocket.close(code=4003, reason="Origin not allowed")
        return

    # 2. Extract token from Sec-WebSocket-Protocol header (preferred, more secure)
    # Format: "access_token.<jwt_token>"
    token = None
    accepted_subprotocol = None
    protocol_header = websocket.headers.get("sec-websocket-protocol", "")
    for protocol in protocol_header.split(","):
        protocol = protocol.strip()
        if protocol.startswith("access_token."):
            token = protocol[len("access_token.") :]
            # Use safe identifier instead of echoing token in response header
            accepted_subprotocol = "v1.authenticated"
            break

    # Fallback to query param (deprecated)
    if not token:
        token = websocket.query_params.get("token")
        if token:
            logger.warning("WebSocket using deprecated query param authentication - migrate to subprotocol")

    # 3. Authenticate user
    try:
        user = await get_current_user_websocket(token)
        user_id = int(user["id"])
    except ValueError as e:
        await websocket.close(code=1008, reason=str(e))
        return
    except Exception as e:
        logger.error("WebSocket auth error: %s", e, exc_info=True)
        await websocket.close(code=1011, reason="Internal server error")
        return

    # 4. Verify session access
    db_gen = get_db()
    try:
        db = await anext(db_gen)
        agent_service = create_agent_service(db)
        session = await agent_service.verify_websocket_access(session_id, user_id)
    except SessionNotFoundError:
        await websocket.close(code=1008, reason=f"Session not found: {session_id}")
        await db_gen.aclose()
        return
    except AccessDeniedError:
        await websocket.close(code=1008, reason="Access denied to session")
        await db_gen.aclose()
        return
    except Exception as e:
        logger.error("Session verification error: %s", e, exc_info=True)
        await websocket.close(code=1011, reason="Internal server error")
        await db_gen.aclose()
        return

    # 5. Accept connection with subprotocol
    await websocket.accept(subprotocol=accepted_subprotocol)

    # 6. Handle messages
    try:
        while True:
            raw = await websocket.receive()
            if raw.get("type") == "websocket.disconnect":
                break

            text = raw.get("text")
            if not text:
                continue

            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                # Handle plain text ping
                if text.strip().lower() == "ping":
                    await websocket.send_json({"type": "pong"})
                else:
                    logger.debug("Ignoring non-JSON WebSocket payload: %s", text[:128])
                continue

            msg_type = data.get("type")
            if msg_type == "execute":
                await _handle_execute(websocket, data, session, agent_service, user_id)
            elif msg_type == "interrupt":
                await _handle_interrupt(websocket, session_id, agent_service)
            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})
            else:
                await websocket.send_json(
                    {
                        "type": "error",
                        "error": {"message": f"Unknown message type: {msg_type}"},
                    }
                )

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("WebSocket error: %s", e, exc_info=True)
    finally:
        await db_gen.aclose()


async def _handle_execute(
    websocket: WebSocket,
    data: dict[str, Any],
    session: AgentSession,
    agent_service: AgentService,
    user_id: int,
) -> None:
    """Execute agent and stream responses.

    Args:
        websocket: The WebSocket connection.
        data: Execute message data containing prompt and options.
        session: The agent session.
        agent_service: Service for agent operations.
        user_id: Authenticated user ID.
    """
    prompt = data.get("prompt", "")
    tools = data.get("tools")
    config = data.get("config")
    model = data.get("model")
    temperature = data.get("temperature")
    max_tokens = data.get("max_tokens")

    context = AgentContext(
        request_id=str(session.id),
        user_id=str(user_id),
        session_id=session.external_id,
        collection_id=str(session.collection_id) if session.collection_id else None,
    )

    try:
        async for message in agent_service.execute(
            plugin_id=session.agent_plugin_id,
            prompt=prompt,
            context=context,
            session_id=session.external_id,
            config=config or session.agent_config,
            tools=tools,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        ):
            if websocket.client_state != WebSocketState.CONNECTED:
                break
            await websocket.send_json({"type": "message", "message": message.to_dict()})

        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json({"type": "complete"})

    except AgentInterruptedError:
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json({"type": "interrupted"})
    except Exception as e:
        logger.error("Execute error: %s", e, exc_info=True)
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json(
                {
                    "type": "error",
                    "error": {"message": str(e), "code": type(e).__name__},
                }
            )


async def _handle_interrupt(
    websocket: WebSocket,
    session_id: str,
    agent_service: AgentService,
) -> None:
    """Handle interrupt request.

    Args:
        websocket: The WebSocket connection.
        session_id: Session to interrupt.
        agent_service: Service for agent operations.
    """
    try:
        await agent_service.interrupt(session_id)
        # Interrupted message will be sent by _handle_execute when it catches AgentInterruptedError
    except Exception as e:
        logger.error("Interrupt error: %s", e, exc_info=True)
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json(
                {
                    "type": "error",
                    "error": {"message": str(e), "code": type(e).__name__},
                }
            )
