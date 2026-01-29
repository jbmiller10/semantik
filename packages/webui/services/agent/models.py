"""SQLAlchemy models for agent conversations.

This module defines the database models for the agent service:
- AgentConversation: Persistent conversation state
- ConversationUncertainty: Uncertainties flagged during conversation
"""

from __future__ import annotations

import enum
from datetime import datetime  # noqa: TCH003 - SQLAlchemy needs this at runtime
from typing import TYPE_CHECKING
from uuid import uuid4

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from shared.database.models import Base

if TYPE_CHECKING:
    from shared.database.models import Collection, CollectionSource, User


class ConversationStatus(str, enum.Enum):
    """Status of an agent conversation."""

    ACTIVE = "active"  # Conversation is in progress
    APPLIED = "applied"  # Pipeline was applied, collection created
    ABANDONED = "abandoned"  # User abandoned the conversation


class UncertaintySeverity(str, enum.Enum):
    """Severity levels for uncertainties flagged during conversation."""

    BLOCKING = "blocking"  # Must be resolved before applying
    NOTABLE = "notable"  # Should be surfaced but can proceed
    INFO = "info"  # Informational, for transparency


class AgentConversation(Base):
    """Persistent conversation state for the pipeline builder agent.

    Stores the durable state of a conversation, including the current
    pipeline configuration and source analysis results. Message history
    is stored in Redis with a 24-hour TTL for efficiency.

    Attributes:
        id: Unique conversation identifier
        user_id: ID of the user who owns this conversation
        source_id: Optional source being configured (from collection_sources)
        collection_id: Collection created when pipeline is applied
        status: Current conversation status (active/applied/abandoned)
        current_pipeline: Serialized PipelineDAG configuration
        source_analysis: Results from SourceAnalyzer sub-agent
        summary: Conversation summary for recovery when Redis expires
        created_at: When the conversation was created
        updated_at: Last modification time
    """

    __tablename__ = "agent_conversations"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    source_id: Mapped[int | None] = mapped_column(
        ForeignKey("collection_sources.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    collection_id: Mapped[str | None] = mapped_column(
        ForeignKey("collections.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Current state
    # Use values_callable to serialize enum values (lowercase) instead of names (uppercase)
    status: Mapped[ConversationStatus] = mapped_column(
        Enum(
            ConversationStatus,
            name="conversation_status",
            create_type=False,
            values_callable=lambda e: [m.value for m in e],
        ),
        nullable=False,
        default=ConversationStatus.ACTIVE,
    )
    current_pipeline: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    source_analysis: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Inline source configuration (for new sources created during conversation)
    # Stores: {"source_type": "directory", "source_config": {"path": "/docs"}, "_pending_secrets": {...}}
    inline_source_config: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # For conversation recovery when Redis messages expire
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Arbitrary extra data (e.g., is_paused for mode toggle)
    extra_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    # Relationships
    user: Mapped[User] = relationship("User", foreign_keys=[user_id])
    source: Mapped[CollectionSource | None] = relationship("CollectionSource", foreign_keys=[source_id])
    collection: Mapped[Collection | None] = relationship("Collection", foreign_keys=[collection_id])
    uncertainties: Mapped[list[ConversationUncertainty]] = relationship(
        "ConversationUncertainty",
        back_populates="conversation",
        cascade="all, delete-orphan",
    )


class ConversationUncertainty(Base):
    """Uncertainties flagged during an agent conversation.

    Tracks issues and concerns raised by sub-agents during analysis.
    Blocking uncertainties must be resolved before the pipeline can
    be applied.

    Attributes:
        id: Unique uncertainty identifier
        conversation_id: Parent conversation
        severity: How serious this uncertainty is (blocking/notable/info)
        message: Human-readable description of the uncertainty
        context: Additional data (file references, error details, etc.)
        resolved: Whether this uncertainty has been addressed
        resolved_by: How it was resolved (user_confirmed, pipeline_adjusted, etc.)
        created_at: When the uncertainty was flagged
    """

    __tablename__ = "conversation_uncertainties"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    conversation_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("agent_conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Use values_callable to serialize enum values (lowercase) instead of names (uppercase)
    severity: Mapped[UncertaintySeverity] = mapped_column(
        Enum(
            UncertaintySeverity,
            name="uncertainty_severity",
            create_type=False,
            values_callable=lambda e: [m.value for m in e],
        ),
        nullable=False,
    )
    message: Mapped[str] = mapped_column(Text, nullable=False)
    context: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    resolved: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    resolved_by: Mapped[str | None] = mapped_column(String(50), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # Relationships
    conversation: Mapped[AgentConversation] = relationship("AgentConversation", back_populates="uncertainties")

    # Note: Partial index ix_uncertainties_blocking_unresolved is created in migration
    # to avoid issues with create_all() validating enum values before migration runs
