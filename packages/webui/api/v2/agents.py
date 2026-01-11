"""
Agent API v2 endpoints.

This module provides RESTful API endpoints for agent plugin operations,
session management, and tool discovery.

IMPORTANT: Route order matters in FastAPI! Specific paths (/sessions, /tools)
must be defined BEFORE wildcard paths (/{agent_id}) to avoid incorrect matching.
"""

import logging
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from shared.agents.exceptions import (
    AgentError,
    AgentExecutionError,
    AgentInterruptedError,
    AgentTimeoutError,
    SessionNotFoundError,
    ToolNotFoundError,
)
from shared.agents.tools.registry import get_tool_registry
from shared.agents.types import AgentContext
from webui.api.v2.agent_schemas import (
    AgentDetailResponse,
    AgentErrorResponse,
    AgentListItem,
    AgentListResponse,
    ExecuteRequest,
    ExecuteResponse,
    ForkResponse,
    MessageListResponse,
    MessageResponse,
    SessionListResponse,
    SessionResponse,
    SessionUpdateRequest,
    TokenUsageResponse,
    ToolDefinitionResponse,
    ToolListResponse,
    ToolParameterResponse,
)
from webui.auth import get_current_user
from webui.rate_limiter import limiter
from webui.services.agent_service import AgentService
from webui.services.factory import get_agent_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/agents", tags=["agents-v2"])


# =============================================================================
# Agent List Endpoint (base path)
# =============================================================================


@router.get(
    "",
    response_model=AgentListResponse,
    responses={
        429: {"model": AgentErrorResponse, "description": "Rate limit exceeded"},
    },
)
@limiter.limit("200/minute")
async def list_agents(
    request: Request,  # noqa: ARG001
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
    service: AgentService = Depends(get_agent_service),
) -> AgentListResponse:
    """List all available agent plugins.

    Returns metadata and capabilities for each registered agent plugin.
    """
    agents = await service.list_agents()

    return AgentListResponse(
        agents=[
            AgentListItem(
                id=a["id"],
                display_name=a["manifest"]["display_name"] if a.get("manifest") else a["id"],
                description=a["manifest"]["description"] if a.get("manifest") else "",
                capabilities=a["capabilities"],
                use_cases=a["use_cases"],
            )
            for a in agents
        ],
        total=len(agents),
    )


# =============================================================================
# Session Endpoints (MUST come before /{agent_id} wildcard)
# =============================================================================


@router.get(
    "/sessions",
    response_model=SessionListResponse,
    responses={
        429: {"model": AgentErrorResponse, "description": "Rate limit exceeded"},
    },
)
@limiter.limit("100/minute")
async def list_sessions(
    request: Request,  # noqa: ARG001
    collection_id: str | None = Query(None, description="Filter by collection"),
    status: str | None = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=100, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    current_user: dict[str, Any] = Depends(get_current_user),
    service: AgentService = Depends(get_agent_service),
) -> SessionListResponse:
    """List agent sessions for the current user.

    Returns paginated list of sessions with metadata.
    """
    user_id = int(current_user["id"])

    sessions, total = await service.list_sessions(
        user_id,
        status=status,
        collection_id=collection_id,
        limit=limit,
        offset=offset,
    )

    return SessionListResponse(
        sessions=[_session_to_response(s) for s in sessions],
        total=total,
    )


@router.get(
    "/sessions/{session_id}",
    response_model=SessionResponse,
    responses={
        403: {"model": AgentErrorResponse, "description": "Access denied"},
        404: {"model": AgentErrorResponse, "description": "Session not found"},
        429: {"model": AgentErrorResponse, "description": "Rate limit exceeded"},
    },
)
@limiter.limit("200/minute")
async def get_session(
    request: Request,  # noqa: ARG001
    session_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),
    service: AgentService = Depends(get_agent_service),
) -> SessionResponse:
    """Get session details by external ID.

    Returns session metadata including message count and token usage.
    """
    session = await _get_session_for_user(
        session_id=session_id,
        user_id=int(current_user["id"]),
        service=service,
    )

    return _session_to_response(session)


@router.patch(
    "/sessions/{session_id}",
    response_model=SessionResponse,
    responses={
        403: {"model": AgentErrorResponse, "description": "Access denied"},
        404: {"model": AgentErrorResponse, "description": "Session not found"},
        429: {"model": AgentErrorResponse, "description": "Rate limit exceeded"},
    },
)
@limiter.limit("100/minute")
async def update_session(
    request: Request,  # noqa: ARG001
    session_id: str,
    update_request: SessionUpdateRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
    service: AgentService = Depends(get_agent_service),
) -> SessionResponse:
    """Update session metadata.

    Currently supports updating the session title.
    """
    # Verify ownership first
    await _get_session_for_user(
        session_id=session_id,
        user_id=int(current_user["id"]),
        service=service,
    )

    # Update title if provided
    if update_request.title is not None:
        session = await service.update_session_title(session_id, update_request.title)
        return _session_to_response(session)

    # No updates - return current session
    session = await service.get_session(session_id)
    return (
        _session_to_response(session)
        if session
        else SessionResponse(
            id="",
            external_id=session_id,
            title=None,
            agent_plugin_id="",
            message_count=0,
            total_input_tokens=0,
            total_output_tokens=0,
            total_cost_usd=0.0,
            status="unknown",
            created_at="",
            last_activity_at="",
        )
    )


@router.get(
    "/sessions/{session_id}/messages",
    response_model=MessageListResponse,
    responses={
        403: {"model": AgentErrorResponse, "description": "Access denied"},
        404: {"model": AgentErrorResponse, "description": "Session not found"},
        429: {"model": AgentErrorResponse, "description": "Rate limit exceeded"},
    },
)
@limiter.limit("200/minute")
async def get_session_messages(
    request: Request,  # noqa: ARG001
    session_id: str,
    limit: int = Query(100, ge=1, le=500, description="Maximum messages"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    current_user: dict[str, Any] = Depends(get_current_user),
    service: AgentService = Depends(get_agent_service),
) -> MessageListResponse:
    """Get messages for a session.

    Returns paginated list of messages in sequence order.
    """
    # Verify ownership first
    await _get_session_for_user(
        session_id=session_id,
        user_id=int(current_user["id"]),
        service=service,
    )

    messages = await service.get_messages(
        session_id,
        limit=limit,
        offset=offset,
    )

    return MessageListResponse(
        messages=[
            MessageResponse(
                id=m["id"],
                sequence=m["sequence"],
                role=m["role"],
                type=m["type"],
                content=m["content"],
                tool_name=m.get("tool_name"),
                tool_call_id=m.get("tool_call_id"),
                tool_input=m.get("tool_input"),
                tool_output=m.get("tool_output"),
                model=m.get("model"),
                created_at=m.get("created_at", ""),
            )
            for m in messages
        ]
    )


@router.post(
    "/sessions/{session_id}/fork",
    response_model=ForkResponse,
    responses={
        403: {"model": AgentErrorResponse, "description": "Access denied"},
        404: {"model": AgentErrorResponse, "description": "Session not found"},
        429: {"model": AgentErrorResponse, "description": "Rate limit exceeded"},
    },
)
@limiter.limit("30/minute")
async def fork_session(
    request: Request,  # noqa: ARG001
    session_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),
    service: AgentService = Depends(get_agent_service),
) -> ForkResponse:
    """Fork a session to create a branch in the conversation.

    Creates a new session with the same messages as the parent,
    enabling exploration of alternative paths.
    """
    # Verify ownership first
    await _get_session_for_user(
        session_id=session_id,
        user_id=int(current_user["id"]),
        service=service,
    )

    new_session_id = await service.fork_session(session_id)

    return ForkResponse(session_id=new_session_id)


@router.post(
    "/sessions/{session_id}/interrupt",
    responses={
        403: {"model": AgentErrorResponse, "description": "Access denied"},
        404: {"model": AgentErrorResponse, "description": "Session not found"},
        429: {"model": AgentErrorResponse, "description": "Rate limit exceeded"},
    },
)
@limiter.limit("60/minute")
async def interrupt_session(
    request: Request,  # noqa: ARG001
    session_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),
    service: AgentService = Depends(get_agent_service),
) -> dict[str, str]:
    """Interrupt an active execution on a session.

    Safe to call if no execution is running.
    """
    # Verify ownership first
    await _get_session_for_user(
        session_id=session_id,
        user_id=int(current_user["id"]),
        service=service,
    )

    await service.interrupt(session_id)

    return {"status": "interrupted"}


@router.delete(
    "/sessions/{session_id}",
    responses={
        403: {"model": AgentErrorResponse, "description": "Access denied"},
        404: {"model": AgentErrorResponse, "description": "Session not found"},
        429: {"model": AgentErrorResponse, "description": "Rate limit exceeded"},
    },
)
@limiter.limit("60/minute")
async def delete_session(
    request: Request,  # noqa: ARG001
    session_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),
    service: AgentService = Depends(get_agent_service),
) -> dict[str, str]:
    """Soft-delete a session.

    The session data is preserved but marked as deleted.
    """
    # Verify ownership first
    await _get_session_for_user(
        session_id=session_id,
        user_id=int(current_user["id"]),
        service=service,
    )

    await service.delete_session(session_id)

    return {"status": "deleted"}


# =============================================================================
# Tool Endpoints (MUST come before /{agent_id} wildcard)
# =============================================================================


@router.get(
    "/tools",
    response_model=ToolListResponse,
    responses={
        429: {"model": AgentErrorResponse, "description": "Rate limit exceeded"},
    },
)
@limiter.limit("200/minute")
async def list_tools(
    request: Request,  # noqa: ARG001
    category: str | None = Query(None, description="Filter by category"),
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
) -> ToolListResponse:
    """List available tools for agents.

    Returns tool definitions that can be passed to agent execution.
    """
    registry = get_tool_registry()
    records = registry.list_all(category=category)

    return ToolListResponse(
        tools=[
            ToolDefinitionResponse(
                name=r.tool.name,
                description=r.tool.description,
                parameters=[
                    ToolParameterResponse(
                        name=p.name,
                        type=p.type,
                        description=p.description,
                        required=p.required,
                        default=p.default,
                    )
                    for p in r.tool.definition.parameters
                ],
                category=r.tool.definition.category,
                requires_context=r.tool.definition.requires_context,
                is_destructive=r.tool.definition.is_destructive,
            )
            for r in records
            if r.enabled
        ]
    )


@router.get(
    "/tools/{tool_name}",
    response_model=ToolDefinitionResponse,
    responses={
        404: {"model": AgentErrorResponse, "description": "Tool not found"},
        429: {"model": AgentErrorResponse, "description": "Rate limit exceeded"},
    },
)
@limiter.limit("200/minute")
async def get_tool(
    request: Request,  # noqa: ARG001
    tool_name: str,
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
) -> ToolDefinitionResponse:
    """Get details for a specific tool.

    Returns the tool definition including parameters and constraints.
    """
    registry = get_tool_registry()
    tool = registry.get(tool_name)

    if not tool:
        raise HTTPException(
            status_code=404,
            detail=f"Tool not found: {tool_name}",
        )

    return ToolDefinitionResponse(
        name=tool.name,
        description=tool.description,
        parameters=[
            ToolParameterResponse(
                name=p.name,
                type=p.type,
                description=p.description,
                required=p.required,
                default=p.default,
            )
            for p in tool.definition.parameters
        ],
        category=tool.definition.category,
        requires_context=tool.definition.requires_context,
        is_destructive=tool.definition.is_destructive,
    )


# =============================================================================
# Agent Detail Endpoint (wildcard - MUST come after /sessions and /tools)
# =============================================================================


@router.get(
    "/{agent_id}",
    response_model=AgentDetailResponse,
    responses={
        404: {"model": AgentErrorResponse, "description": "Agent not found"},
        429: {"model": AgentErrorResponse, "description": "Rate limit exceeded"},
    },
)
@limiter.limit("200/minute")
async def get_agent(
    request: Request,  # noqa: ARG001
    agent_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
    service: AgentService = Depends(get_agent_service),
) -> AgentDetailResponse:
    """Get detailed information about a specific agent.

    Returns full metadata, capabilities, and configuration schema.
    """
    agent = await service.get_agent(agent_id)
    if not agent:
        raise HTTPException(
            status_code=404,
            detail=f"Agent not found: {agent_id}",
        )

    return AgentDetailResponse(
        id=agent["id"],
        manifest=agent.get("manifest"),
        capabilities=agent["capabilities"],
        use_cases=agent["use_cases"],
        config_schema=agent.get("config_schema"),
    )


# =============================================================================
# Execution Endpoint (wildcard - MUST come after /sessions and /tools)
# =============================================================================


@router.post(
    "/{agent_id}/execute",
    response_model=ExecuteResponse,
    responses={
        400: {"model": AgentErrorResponse, "description": "Invalid request"},
        404: {"model": AgentErrorResponse, "description": "Agent or session not found"},
        429: {"model": AgentErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": AgentErrorResponse, "description": "Execution error"},
        504: {"model": AgentErrorResponse, "description": "Execution timeout"},
    },
)
@limiter.limit("30/minute")
async def execute_agent(
    request: Request,  # noqa: ARG001
    agent_id: str,
    execute_request: ExecuteRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
    service: AgentService = Depends(get_agent_service),
) -> ExecuteResponse:
    """Execute an agent with the given prompt.

    Creates a new session or resumes an existing one. Returns all response
    messages synchronously (no streaming).
    """
    # Build execution context
    context = AgentContext(
        request_id=str(uuid4()),
        user_id=str(current_user["id"]),
        collection_id=execute_request.collection_id,
    )

    try:
        # Collect all messages from execution
        messages: list[dict[str, Any]] = []
        session_id: str | None = None

        async for msg in service.execute(
            agent_id,
            execute_request.prompt,
            context=context,
            session_id=execute_request.session_id,
            config=execute_request.config,
            tools=execute_request.tools,
            system_prompt=execute_request.system_prompt,
            model=execute_request.model,
            temperature=execute_request.temperature,
            max_tokens=execute_request.max_tokens,
            stream=False,  # Sync execution only in Phase 7
        ):
            if not msg.is_partial:
                messages.append(msg.to_dict())

            # Capture session ID from context
            if context.session_id and not session_id:
                session_id = context.session_id

        # If no session_id in context, get it from the service by looking up session
        if not session_id and execute_request.session_id:
            session_id = execute_request.session_id
        elif not session_id:
            # Session was created - find it from the context
            session_id = context.session_id or ""

        # Calculate total usage from all messages
        usage = _calculate_total_usage(messages)

        # Get external session ID for response
        session = await service.get_session(session_id) if session_id else None
        external_id = session["external_id"] if session else session_id or ""

        return ExecuteResponse(
            session_id=external_id,
            messages=messages,
            usage=usage,
        )

    except SessionNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e),
        ) from e

    except ToolNotFoundError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        ) from e

    except AgentTimeoutError as e:
        raise HTTPException(
            status_code=504,
            detail=str(e),
        ) from e

    except AgentInterruptedError as e:
        raise HTTPException(
            status_code=499,  # Client Closed Request
            detail=str(e),
        ) from e

    except AgentExecutionError as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        ) from e

    except AgentError as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        ) from e

    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e),
        ) from e


# =============================================================================
# Helper Functions
# =============================================================================


def _calculate_total_usage(messages: list[dict[str, Any]]) -> TokenUsageResponse | None:
    """Calculate aggregate token usage from all messages."""
    total_input = 0
    total_output = 0
    total_cache_read = 0
    total_cache_write = 0
    has_usage = False

    for msg in messages:
        usage = msg.get("usage")
        if usage:
            has_usage = True
            total_input += usage.get("input_tokens", 0)
            total_output += usage.get("output_tokens", 0)
            total_cache_read += usage.get("cache_read_tokens", 0)
            total_cache_write += usage.get("cache_write_tokens", 0)

    if not has_usage:
        return None

    return TokenUsageResponse(
        input_tokens=total_input,
        output_tokens=total_output,
        total_tokens=total_input + total_output,
        cache_read_tokens=total_cache_read,
        cache_write_tokens=total_cache_write,
    )


async def _get_session_for_user(
    session_id: str,
    user_id: int,
    service: AgentService,
) -> dict[str, Any]:
    """Get session and verify ownership.

    Args:
        session_id: External session ID.
        user_id: Current user's ID.
        service: Agent service instance.

    Returns:
        Session dictionary.

    Raises:
        HTTPException: 404 if not found, 403 if access denied.
    """
    session = await service.get_session(session_id)

    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {session_id}",
        )

    # Check ownership (allow if user_id is None for backward compatibility)
    session_user_id = session.get("user_id")
    if session_user_id is not None and session_user_id != user_id:
        raise HTTPException(
            status_code=403,
            detail="Access denied to this session",
        )

    # Type assertion for mypy - session is guaranteed to be dict[str, Any] at this point
    result: dict[str, Any] = session
    return result


def _session_to_response(session: dict[str, Any]) -> SessionResponse:
    """Convert session dictionary to response model."""
    return SessionResponse(
        id=session.get("id", ""),
        external_id=session.get("external_id", ""),
        title=session.get("title"),
        agent_plugin_id=session.get("agent_plugin_id", ""),
        message_count=session.get("message_count", 0),
        total_input_tokens=session.get("total_input_tokens", 0),
        total_output_tokens=session.get("total_output_tokens", 0),
        total_cost_usd=session.get("total_cost_usd", 0.0),
        status=session.get("status", "unknown"),
        created_at=session.get("created_at", ""),
        last_activity_at=session.get("last_activity_at", ""),
        collection_id=session.get("collection_id"),
        user_id=session.get("user_id"),
    )
