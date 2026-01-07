"""
SQLAlchemy models for agent session persistence.

Agent sessions store conversation history, usage statistics, and SDK session
references for LLM agent plugins. Sessions can be forked to create branching
conversation histories.

This module provides:
- AgentSession: Main session model with configuration and statistics
- AgentSessionMessage: Denormalized messages for efficient querying
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import relationship

from shared.database.models import Base


class AgentSession(Base):
    """
    Persistent agent conversation session.

    Stores:
    - Session metadata (user, collection, config)
    - Conversation history (messages as denormalized JSONB)
    - Execution statistics (tokens, costs)
    - Parent session for forks
    """

    __tablename__ = "agent_sessions"

    # Primary key (UUID as string)
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))

    # Session identification
    external_id = Column(String(64), unique=True, nullable=False, index=True)
    title = Column(String(255), nullable=True)

    # Ownership (optional - allows anonymous sessions)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)

    # Agent configuration
    agent_plugin_id = Column(String(64), nullable=False, index=True)
    agent_config = Column(JSON, nullable=False, default=dict)

    # Collection context (optional - for search-related sessions)
    collection_id = Column(String, ForeignKey("collections.id"), nullable=True, index=True)

    # Conversation state (denormalized for quick access)
    messages = Column(JSON, nullable=False, default=list)
    message_count = Column(Integer, nullable=False, default=0)

    # SDK session ID (for resume functionality)
    sdk_session_id = Column(String(255), nullable=True)

    # Fork tracking
    parent_session_id = Column(String, ForeignKey("agent_sessions.id"), nullable=True)
    fork_count = Column(Integer, nullable=False, default=0)

    # Usage statistics
    total_input_tokens = Column(Integer, nullable=False, default=0)
    total_output_tokens = Column(Integer, nullable=False, default=0)
    total_cost_usd = Column(Integer, nullable=False, default=0)  # Stored as cents * 100

    # Lifecycle
    status = Column(String(20), nullable=False, default="active")  # active, archived, deleted
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())
    last_activity_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    archived_at = Column(DateTime(timezone=True), nullable=True)

    # Self-referential relationship for forks
    parent_session = relationship(
        "AgentSession",
        remote_side=[id],
        backref="forks",
        foreign_keys=[parent_session_id],
    )

    # Indexes for common query patterns
    __table_args__ = (
        Index("ix_agent_sessions_user_status", "user_id", "status"),
        Index("ix_agent_sessions_last_activity", "last_activity_at"),
    )

    def add_message(self, message: dict[str, Any]) -> None:
        """Add a message to the session."""
        if self.messages is None:
            self.messages = []
        self.messages = [*self.messages, message]  # type: ignore[assignment]
        self.message_count = len(self.messages)  # type: ignore[assignment]
        self.last_activity_at = datetime.now(UTC)  # type: ignore[assignment]

    def update_stats(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        """Update session statistics."""
        self.total_input_tokens += input_tokens  # type: ignore[assignment]
        self.total_output_tokens += output_tokens  # type: ignore[assignment]
        self.total_cost_usd += int(cost_usd * 10000)  # type: ignore[assignment]

    def to_dict(self) -> dict[str, Any]:
        """Serialize for API responses."""
        return {
            "id": self.id,
            "external_id": self.external_id,
            "title": self.title,
            "user_id": self.user_id,
            "agent_plugin_id": self.agent_plugin_id,
            "collection_id": self.collection_id,
            "message_count": self.message_count,
            "parent_session_id": self.parent_session_id,
            "fork_count": self.fork_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": self.total_cost_usd / 10000 if self.total_cost_usd else 0.0,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_activity_at": self.last_activity_at.isoformat() if self.last_activity_at else None,
        }


class AgentSessionMessage(Base):
    """
    Individual message in a session (for efficient querying).

    Denormalized from session.messages for:
    - Pagination
    - Search
    - Individual message access
    """

    __tablename__ = "agent_session_messages"

    # Primary key (UUID as string)
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))

    # Session reference
    session_id = Column(
        String,
        ForeignKey("agent_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Message data
    message_id = Column(String(64), nullable=False)  # From AgentMessage.id
    sequence_number = Column(Integer, nullable=False)
    role = Column(String(20), nullable=False)  # user, assistant, system, tool_call, tool_result, error
    type = Column(String(20), nullable=False)  # text, thinking, tool_use, tool_output, error, metadata
    content = Column(Text, nullable=False)

    # Tool-related data
    tool_name = Column(String(64), nullable=True)
    tool_call_id = Column(String(64), nullable=True)
    tool_input = Column(JSON, nullable=True)
    tool_output = Column(JSON, nullable=True)

    # Execution metadata
    model = Column(String(64), nullable=True)
    input_tokens = Column(Integer, nullable=True)
    output_tokens = Column(Integer, nullable=True)
    cost_usd = Column(Integer, nullable=True)  # Stored as cents * 100

    # Timestamp
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())

    # Index for efficient message retrieval
    __table_args__ = (Index("ix_agent_session_messages_session_seq", "session_id", "sequence_number"),)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for API responses."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "message_id": self.message_id,
            "sequence_number": self.sequence_number,
            "role": self.role,
            "type": self.type,
            "content": self.content,
            "tool_name": self.tool_name,
            "tool_call_id": self.tool_call_id,
            "tool_input": self.tool_input,
            "tool_output": self.tool_output,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd / 10000 if self.cost_usd else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
