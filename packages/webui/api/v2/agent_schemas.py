"""Pydantic schemas for agent conversation endpoints.

This module defines the request/response schemas for the agent service API,
which handles pipeline builder conversations.
"""

from __future__ import annotations

from datetime import datetime  # noqa: TCH003 - Required at runtime for Pydantic
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# =============================================================================
# SSE Streaming Types
# =============================================================================


class AgentPhase(str, Enum):
    """Phase of agent processing for status bar display."""

    IDLE = "idle"
    ANALYZING = "analyzing"
    SAMPLING = "sampling"
    BUILDING = "building"
    VALIDATING = "validating"
    READY = "ready"


class AgentStreamEventType(str, Enum):
    """Event types for SSE streaming responses.

    Events are emitted as the agent processes a request:
    - TOOL_CALL_START: Before a tool starts executing
    - TOOL_CALL_END: After a tool completes (success or failure)
    - SUBAGENT_START: When a sub-agent is spawned
    - SUBAGENT_END: When a sub-agent completes
    - UNCERTAINTY: When an uncertainty is flagged
    - PIPELINE_UPDATE: When the pipeline configuration changes
    - CONTENT: Final response content from the agent
    - DONE: Stream complete with full response metadata
    - ERROR: An error occurred during processing
    - STATUS: Status update for status bar (phase, message, progress)
    - ACTIVITY: Activity log entry with timestamp
    """

    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_END = "tool_call_end"
    SUBAGENT_START = "subagent_start"
    SUBAGENT_END = "subagent_end"
    UNCERTAINTY = "uncertainty"
    PIPELINE_UPDATE = "pipeline_update"
    CONTENT = "content"
    DONE = "done"
    ERROR = "error"
    STATUS = "status"
    ACTIVITY = "activity"
    QUESTION = "question"


class AgentStreamEvent(BaseModel):
    """SSE event payload for streaming agent responses.

    Each event has a type and associated data. Events are serialized
    as Server-Sent Events in the format:
        event: {event_type}
        data: {json_data}
    """

    event: AgentStreamEventType = Field(
        ...,
        description="Type of event being streamed",
    )
    data: dict[str, Any] = Field(
        default_factory=dict,
        description="Event-specific data payload",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "event": "tool_call_start",
                    "data": {"tool": "list_plugins", "arguments": {"type": "embedding"}},
                },
                {
                    "event": "tool_call_end",
                    "data": {
                        "tool": "list_plugins",
                        "success": True,
                        "result": {"plugins": []},
                    },
                },
                {
                    "event": "content",
                    "data": {"text": "Here are the available plugins..."},
                },
                {
                    "event": "done",
                    "data": {
                        "pipeline_updated": False,
                        "uncertainties_added": [],
                        "tool_calls": [],
                    },
                },
            ]
        },
    )


# =============================================================================
# Request Schemas
# =============================================================================


class InlineSourceConfig(BaseModel):
    """Configuration for a new source to be created with the conversation.

    Used when the user wants to configure a source directly in the guided setup
    rather than using a pre-existing source.
    """

    source_type: str = Field(
        ...,
        description="Type of connector (e.g., 'directory', 'git', 'imap')",
        json_schema_extra={"example": "directory"},
    )
    source_config: dict[str, Any] = Field(
        ...,
        description="Connector-specific configuration",
        json_schema_extra={"example": {"path": "/home/user/documents"}},
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "source_type": "directory",
                "source_config": {"path": "/home/user/documents"},
            }
        },
    )


class CreateConversationRequest(BaseModel):
    """Request schema for creating a new agent conversation.

    Either source_id OR inline_source must be provided, but not both.
    When using inline_source, secrets can be provided separately for security.
    """

    source_id: int | None = Field(
        default=None,
        description="ID of an existing collection source to configure",
        json_schema_extra={"example": 42},
    )
    inline_source: InlineSourceConfig | None = Field(
        default=None,
        description="Configuration for a new source (created when pipeline is applied)",
    )
    secrets: dict[str, str] | None = Field(
        default=None,
        description="Secrets for inline_source (e.g., passwords, tokens). Only used with inline_source.",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "source_id": 42,
                },
                {
                    "inline_source": {
                        "source_type": "directory",
                        "source_config": {"path": "/home/user/documents"},
                    },
                },
                {
                    "inline_source": {
                        "source_type": "git",
                        "source_config": {
                            "repository_url": "https://github.com/user/repo.git",
                            "branch": "main",
                        },
                    },
                    "secrets": {"password": "github_token"},
                },
            ]
        },
    )

    @model_validator(mode="after")
    def validate_source_specification(self) -> CreateConversationRequest:
        """Validate that exactly one of source_id or inline_source is provided."""
        if self.source_id is None and self.inline_source is None:
            raise ValueError("Either source_id or inline_source must be provided")
        if self.source_id is not None and self.inline_source is not None:
            raise ValueError("Cannot specify both source_id and inline_source")
        if self.secrets is not None and self.inline_source is None:
            raise ValueError("secrets can only be provided with inline_source")
        return self


class SendMessageRequest(BaseModel):
    """Request schema for sending a message to the agent."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The message to send to the agent",
        json_schema_extra={"example": "Use semantic chunking with 512 tokens max"},
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "message": "Use semantic chunking with 512 tokens max",
            }
        },
    )


class ApplyPipelineRequest(BaseModel):
    """Request schema for applying a configured pipeline to create a collection."""

    collection_name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Name for the new collection",
        json_schema_extra={"example": "My Documents Collection"},
    )
    force: bool = Field(
        default=False,
        description="Apply despite blocking uncertainties",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "collection_name": "My Documents Collection",
                "force": False,
            }
        },
    )


class ResolveUncertaintyRequest(BaseModel):
    """Request schema for resolving an uncertainty."""

    uncertainty_id: str = Field(
        ...,
        description="UUID of the uncertainty to resolve",
    )
    resolution: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="How the uncertainty was resolved",
        json_schema_extra={"example": "user_confirmed"},
    )

    model_config = ConfigDict(extra="forbid")


class AnswerQuestionRequest(BaseModel):
    """Request to answer an agent question."""

    question_id: str = Field(..., description="ID of the question being answered")
    option_id: str | None = Field(None, description="ID of selected option (if choosing from options)")
    custom_response: str | None = Field(None, description="Custom text response (if not using options)")

    @model_validator(mode="after")
    def validate_response(self) -> AnswerQuestionRequest:
        """Ensure either option_id or custom_response is provided."""
        if not self.option_id and not self.custom_response:
            raise ValueError("Either option_id or custom_response must be provided")
        return self

    model_config = ConfigDict(extra="forbid")


class AnswerQuestionResponse(BaseModel):
    """Response after answering a question."""

    success: bool
    message: str | None = None

    model_config = ConfigDict(extra="forbid")


# =============================================================================
# Response Schemas
# =============================================================================


class UncertaintyResponse(BaseModel):
    """Response schema for a conversation uncertainty."""

    id: str = Field(..., description="Unique identifier (UUID)")
    severity: Literal["blocking", "notable", "info"] = Field(
        ...,
        description="Severity level of the uncertainty",
    )
    message: str = Field(..., description="Human-readable description")
    resolved: bool = Field(..., description="Whether this uncertainty has been addressed")
    context: dict[str, Any] | None = Field(
        default=None,
        description="Additional context data",
    )

    model_config = ConfigDict(from_attributes=True)


class MessageResponse(BaseModel):
    """Response schema for a conversation message."""

    role: Literal["user", "assistant", "tool", "subagent"] = Field(
        ...,
        description="Who sent this message",
    )
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(..., description="When the message was created")
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Additional message metadata",
    )

    model_config = ConfigDict(from_attributes=True)


class ConversationResponse(BaseModel):
    """Response schema for basic conversation info."""

    id: str = Field(..., description="Unique identifier (UUID)")
    status: Literal["active", "applied", "abandoned"] = Field(
        ...,
        description="Current conversation status",
    )
    source_id: int | None = Field(
        default=None,
        description="ID of the source being configured",
    )
    created_at: datetime = Field(..., description="Creation timestamp")

    model_config = ConfigDict(from_attributes=True)


class InlineSourceConfigResponse(BaseModel):
    """Response schema for inline source configuration.

    Note: _pending_secrets is intentionally NOT included in the response
    to prevent exposing secrets.
    """

    source_type: str = Field(..., description="Type of connector")
    source_config: dict[str, Any] = Field(..., description="Connector configuration")

    model_config = ConfigDict(from_attributes=True)


class ConversationDetailResponse(BaseModel):
    """Response schema for full conversation details."""

    id: str = Field(..., description="Unique identifier (UUID)")
    status: Literal["active", "applied", "abandoned"] = Field(
        ...,
        description="Current conversation status",
    )
    source_id: int | None = Field(
        default=None,
        description="ID of the source being configured (existing source)",
    )
    inline_source_config: InlineSourceConfigResponse | None = Field(
        default=None,
        description="Inline source configuration (new source to be created)",
    )
    collection_id: str | None = Field(
        default=None,
        description="ID of the created collection (if applied)",
    )
    current_pipeline: dict[str, Any] | None = Field(
        default=None,
        description="Current pipeline configuration",
    )
    source_analysis: dict[str, Any] | None = Field(
        default=None,
        description="Results from source analysis",
    )
    uncertainties: list[UncertaintyResponse] = Field(
        default_factory=list,
        description="Uncertainties flagged during conversation",
    )
    messages: list[MessageResponse] = Field(
        default_factory=list,
        description="Conversation message history",
    )
    message_load_error: str | None = Field(
        default=None,
        description="Error message if messages failed to load from Redis",
    )
    summary: str | None = Field(
        default=None,
        description="Conversation summary for recovery",
    )
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

    model_config = ConfigDict(from_attributes=True)


class ConversationListResponse(BaseModel):
    """Response schema for listing conversations."""

    conversations: list[ConversationResponse] = Field(
        ...,
        description="List of conversations",
    )
    total: int = Field(..., description="Total number of conversations")

    model_config = ConfigDict(extra="forbid")


class ApplyPipelineResponse(BaseModel):
    """Response schema for applying a pipeline."""

    collection_id: str = Field(..., description="UUID of the created collection")
    collection_name: str = Field(..., description="Name of the created collection")
    operation_id: str | None = Field(
        default=None,
        description="UUID of the indexing operation (if started)",
    )
    status: Literal["created", "indexing"] = Field(
        ...,
        description="Collection status after creation",
    )

    model_config = ConfigDict(extra="forbid")


class AgentMessageResponse(BaseModel):
    """Response from sending a message to the agent."""

    response: str = Field(
        ...,
        description="Agent's response text",
    )
    pipeline_updated: bool = Field(
        ...,
        description="Whether the pipeline was modified",
    )
    uncertainties_added: list[UncertaintyResponse] = Field(
        default_factory=list,
        description="New uncertainties flagged by this turn",
    )
    tool_calls: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Tools called during this turn (for transparency)",
    )

    model_config = ConfigDict(extra="forbid")
