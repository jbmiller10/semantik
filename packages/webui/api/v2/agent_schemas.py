"""
Pydantic schemas for the Agent Plugin API.

This module defines request/response models for agent execution,
session management, and tool discovery endpoints.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# =============================================================================
# Agent Discovery Schemas
# =============================================================================


class AgentListItem(BaseModel):
    """Summary information about an available agent."""

    id: str = Field(description="Agent plugin ID")
    display_name: str = Field(description="Human-readable agent name")
    description: str = Field(description="Agent description")
    capabilities: dict[str, Any] = Field(description="Agent capabilities object")
    use_cases: list[str] = Field(description="Supported use cases")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "claude-agent",
                "display_name": "Claude Agent",
                "description": "LLM agent powered by Claude",
                "capabilities": {
                    "supports_streaming": True,
                    "supports_tools": True,
                    "supports_sessions": True,
                },
                "use_cases": ["assistant", "tool_use", "reasoning"],
            }
        }
    )


class AgentListResponse(BaseModel):
    """Response for listing available agents."""

    agents: list[AgentListItem]
    total: int

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "agents": [
                    {
                        "id": "claude-agent",
                        "display_name": "Claude Agent",
                        "description": "LLM agent powered by Claude",
                        "capabilities": {"supports_streaming": True},
                        "use_cases": ["assistant"],
                    }
                ],
                "total": 1,
            }
        }
    )


class AgentDetailResponse(BaseModel):
    """Detailed information about a specific agent."""

    id: str = Field(description="Agent plugin ID")
    manifest: dict[str, Any] | None = Field(description="Plugin manifest metadata")
    capabilities: dict[str, Any] = Field(description="Agent capabilities")
    use_cases: list[str] = Field(description="Supported use cases")
    config_schema: dict[str, Any] | None = Field(description="JSON Schema for configuration")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "claude-agent",
                "manifest": {
                    "display_name": "Claude Agent",
                    "description": "LLM agent powered by Claude",
                    "version": "1.0.0",
                },
                "capabilities": {
                    "supports_streaming": True,
                    "supports_tools": True,
                },
                "use_cases": ["assistant", "tool_use"],
                "config_schema": {
                    "type": "object",
                    "properties": {
                        "model": {"type": "string"},
                        "temperature": {"type": "number"},
                    },
                },
            }
        }
    )


# =============================================================================
# Execution Schemas
# =============================================================================


class ExecuteRequest(BaseModel):
    """Request to execute an agent."""

    prompt: str = Field(..., min_length=1, max_length=100000, description="User message or task")
    session_id: str | None = Field(None, description="Resume existing session (external ID)")
    collection_id: str | None = Field(None, description="Collection context for search tools")
    tools: list[str] | None = Field(None, description="Tool names to enable")
    config: dict[str, Any] | None = Field(None, description="Agent configuration overrides")
    model: str | None = Field(None, description="Model override")
    temperature: float | None = Field(None, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int | None = Field(None, ge=1, le=100000, description="Maximum output tokens")
    system_prompt: str | None = Field(None, description="System prompt override")

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "prompt": "Search for documents about authentication",
                "session_id": None,
                "collection_id": "123e4567-e89b-12d3-a456-426614174000",
                "tools": ["semantic_search"],
                "model": "claude-sonnet-4-20250514",
                "temperature": 0.7,
            }
        },
    )


class TokenUsageResponse(BaseModel):
    """Token usage statistics."""

    input_tokens: int = Field(description="Number of input tokens")
    output_tokens: int = Field(description="Number of output tokens")
    total_tokens: int = Field(description="Total tokens used")
    cache_read_tokens: int = Field(default=0, description="Tokens read from cache")
    cache_write_tokens: int = Field(default=0, description="Tokens written to cache")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "input_tokens": 150,
                "output_tokens": 250,
                "total_tokens": 400,
                "cache_read_tokens": 50,
                "cache_write_tokens": 0,
            }
        }
    )


class ExecuteResponse(BaseModel):
    """Response from agent execution."""

    session_id: str = Field(description="Session external ID")
    messages: list[dict[str, Any]] = Field(description="Response messages")
    usage: TokenUsageResponse | None = Field(description="Aggregate token usage")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "session_id": "abc12345",
                "messages": [
                    {
                        "id": "msg_001",
                        "role": "assistant",
                        "type": "text",
                        "content": "I found several documents about authentication...",
                    }
                ],
                "usage": {
                    "input_tokens": 150,
                    "output_tokens": 250,
                    "total_tokens": 400,
                },
            }
        }
    )


# =============================================================================
# Session Schemas
# =============================================================================


class SessionResponse(BaseModel):
    """Agent session metadata."""

    id: str = Field(description="Internal session UUID")
    external_id: str = Field(description="Public session ID (URL-safe)")
    title: str | None = Field(description="Session title")
    agent_plugin_id: str = Field(description="Agent plugin ID")
    message_count: int = Field(description="Number of messages in session")
    total_input_tokens: int = Field(description="Total input tokens used")
    total_output_tokens: int = Field(description="Total output tokens used")
    total_cost_usd: float = Field(description="Estimated cost in USD")
    status: str = Field(description="Session status: active, archived, deleted")
    created_at: str = Field(description="ISO timestamp of creation")
    last_activity_at: str = Field(description="ISO timestamp of last activity")
    collection_id: str | None = Field(None, description="Associated collection ID")
    user_id: int | None = Field(None, description="Owner user ID")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "external_id": "abc12345",
                "title": "Authentication Search Session",
                "agent_plugin_id": "claude-agent",
                "message_count": 5,
                "total_input_tokens": 1500,
                "total_output_tokens": 2500,
                "total_cost_usd": 0.0125,
                "status": "active",
                "created_at": "2025-01-06T10:00:00Z",
                "last_activity_at": "2025-01-06T10:05:00Z",
            }
        }
    )


class SessionListResponse(BaseModel):
    """Response for listing sessions."""

    sessions: list[SessionResponse]
    total: int

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "sessions": [],
                "total": 0,
            }
        }
    )


class SessionUpdateRequest(BaseModel):
    """Request to update session metadata."""

    title: str | None = Field(None, min_length=1, max_length=255, description="New session title")

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "title": "Updated Session Title",
            }
        },
    )


class ForkResponse(BaseModel):
    """Response from forking a session."""

    session_id: str = Field(description="New session external ID")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "session_id": "def67890",
            }
        }
    )


# =============================================================================
# Message Schemas
# =============================================================================


class MessageResponse(BaseModel):
    """Individual message in a session."""

    id: str = Field(description="Message ID")
    sequence: int = Field(description="Message sequence number")
    role: str = Field(description="Role: user, assistant, system, tool_call, tool_result, error")
    type: str = Field(description="Type: text, thinking, tool_use, tool_output, error")
    content: str = Field(description="Message content")
    tool_name: str | None = Field(None, description="Tool name if tool-related")
    tool_call_id: str | None = Field(None, description="Tool call ID")
    tool_input: dict[str, Any] | None = Field(None, description="Tool input arguments")
    tool_output: dict[str, Any] | None = Field(None, description="Tool execution output")
    model: str | None = Field(None, description="Model used for generation")
    created_at: str = Field(description="ISO timestamp")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "msg_001",
                "sequence": 1,
                "role": "assistant",
                "type": "text",
                "content": "I'll search for authentication documents.",
                "created_at": "2025-01-06T10:00:00Z",
            }
        }
    )


class MessageListResponse(BaseModel):
    """Response for listing session messages."""

    messages: list[MessageResponse]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "messages": [
                    {
                        "id": "msg_001",
                        "sequence": 0,
                        "role": "user",
                        "type": "text",
                        "content": "Find authentication docs",
                        "created_at": "2025-01-06T10:00:00Z",
                    }
                ]
            }
        }
    )


# =============================================================================
# Tool Schemas
# =============================================================================


class ToolParameterResponse(BaseModel):
    """Tool parameter definition."""

    name: str = Field(description="Parameter name")
    type: str = Field(description="Parameter type")
    description: str = Field(description="Parameter description")
    required: bool = Field(default=True, description="Whether parameter is required")
    default: Any | None = Field(None, description="Default value")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "query",
                "type": "string",
                "description": "The search query",
                "required": True,
            }
        }
    )


class ToolDefinitionResponse(BaseModel):
    """Tool definition."""

    name: str = Field(description="Tool name")
    description: str = Field(description="Tool description")
    parameters: list[ToolParameterResponse] = Field(description="Tool parameters")
    category: str = Field(description="Tool category")
    requires_context: bool = Field(default=False, description="Whether tool requires context")
    is_destructive: bool = Field(default=False, description="Whether tool makes destructive changes")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "semantic_search",
                "description": "Search documents by semantic similarity",
                "parameters": [
                    {
                        "name": "query",
                        "type": "string",
                        "description": "The search query",
                        "required": True,
                    }
                ],
                "category": "search",
                "requires_context": True,
                "is_destructive": False,
            }
        }
    )


class ToolListResponse(BaseModel):
    """Response for listing available tools."""

    tools: list[ToolDefinitionResponse]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "tools": [
                    {
                        "name": "semantic_search",
                        "description": "Search documents by semantic similarity",
                        "parameters": [],
                        "category": "search",
                    }
                ]
            }
        }
    )


# =============================================================================
# Error Schemas
# =============================================================================


class AgentErrorResponse(BaseModel):
    """Error response for agent operations."""

    detail: str = Field(description="Error message")
    code: str | None = Field(None, description="Error code")
    session_id: str | None = Field(None, description="Session ID if applicable")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "detail": "Session not found",
                "code": "SESSION_NOT_FOUND",
                "session_id": "abc12345",
            }
        }
    )
