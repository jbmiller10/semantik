"""Pydantic schemas for assisted flow API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class InlineSourceConfig(BaseModel):
    """Configuration for a new source to be created with the session."""

    source_type: str = Field(..., description="Type of source (directory, git, imap)")
    source_config: dict[str, Any] = Field(default_factory=dict, description="Source-specific configuration")


class StartFlowRequest(BaseModel):
    """Request to start an assisted flow session.

    Either source_id OR inline_source must be provided, not both.
    - source_id: Use an existing source from the database
    - inline_source: Configure a new source (created when pipeline is applied)
    """

    source_id: int | None = Field(default=None, description="Integer ID of an existing collection source")
    inline_source: InlineSourceConfig | None = Field(default=None, description="Configuration for a new source")
    secrets: dict[str, str] | None = Field(default=None, description="Secrets for inline_source (passwords, tokens)")

    @model_validator(mode="after")
    def validate_source(self) -> StartFlowRequest:
        """Validate that exactly one of source_id or inline_source is provided."""
        if self.source_id is not None and self.inline_source is not None:
            raise ValueError("Cannot specify both source_id and inline_source")
        if self.source_id is None and self.inline_source is None:
            raise ValueError("Must specify either source_id or inline_source")
        if self.secrets is not None and self.inline_source is None:
            raise ValueError("secrets can only be used with inline_source")
        return self


class StartFlowResponse(BaseModel):
    """Response from starting an assisted flow session."""

    session_id: str = Field(..., description="SDK session ID for resuming")
    source_name: str = Field(..., description="Name of the source being configured")


class SendMessageRequest(BaseModel):
    """Request to send a message in an assisted flow session."""

    message: str = Field(..., description="User message to send")


class MessageEvent(BaseModel):
    """SSE event from message processing."""

    type: str = Field(..., description="Event type")
    data: dict = Field(default_factory=dict, description="Event data")


class SubmitAnswerRequest(BaseModel):
    """Request to submit an answer to a pending question."""

    question_id: str = Field(..., description="ID of the question being answered")
    answers: dict[str, str] = Field(
        ...,
        description="Answers mapping question text to selected option label",
    )


class SubmitAnswerResponse(BaseModel):
    """Response from submitting an answer."""

    success: bool = Field(..., description="Whether the answer was accepted")
