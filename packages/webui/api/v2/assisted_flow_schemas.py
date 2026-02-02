"""Pydantic schemas for assisted flow API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class StartFlowRequest(BaseModel):
    """Request to start an assisted flow session."""

    source_id: int = Field(..., description="Integer ID of the collection source to configure")


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
