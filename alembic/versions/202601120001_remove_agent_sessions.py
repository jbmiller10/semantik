"""Remove agent sessions tables.

The agent plugin system has been removed from Semantik. This migration
drops the agent_sessions and agent_session_messages tables that were
used for LLM agent conversation persistence.

Revision ID: 202601120001
Revises: 202601101400
Create Date: 2026-01-12 00:01:00.000000

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa

from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence

revision: str = "202601120001"
down_revision: str | Sequence[str] | None = "202601101400"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Remove agent sessions tables."""
    # Drop agent_session_messages first (has FK to agent_sessions)
    op.drop_index("ix_agent_session_messages_session_seq", table_name="agent_session_messages")
    op.drop_index("ix_agent_session_messages_session_id", table_name="agent_session_messages")
    op.drop_table("agent_session_messages")

    # Drop agent_sessions
    op.drop_index("ix_agent_sessions_last_activity", table_name="agent_sessions")
    op.drop_index("ix_agent_sessions_plugin_id", table_name="agent_sessions")
    op.drop_index("ix_agent_sessions_collection_id", table_name="agent_sessions")
    op.drop_index("ix_agent_sessions_user_status", table_name="agent_sessions")
    op.drop_index("ix_agent_sessions_user_id", table_name="agent_sessions")
    op.drop_index("ix_agent_sessions_external_id", table_name="agent_sessions")
    op.drop_table("agent_sessions")


def downgrade() -> None:
    """Recreate agent sessions tables (if rolling back)."""
    # Create agent_sessions table
    op.create_table(
        "agent_sessions",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("external_id", sa.String(64), nullable=False),
        sa.Column("title", sa.String(255), nullable=True),
        sa.Column(
            "user_id",
            sa.Integer(),
            sa.ForeignKey("users.id"),
            nullable=True,
        ),
        sa.Column("agent_plugin_id", sa.String(64), nullable=False),
        sa.Column("agent_config", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column(
            "collection_id",
            sa.String(),
            sa.ForeignKey("collections.id"),
            nullable=True,
        ),
        sa.Column("messages", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("message_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("sdk_session_id", sa.String(255), nullable=True),
        sa.Column(
            "parent_session_id",
            sa.String(),
            sa.ForeignKey("agent_sessions.id"),
            nullable=True,
        ),
        sa.Column("fork_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_input_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_output_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_cost_usd", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("status", sa.String(20), nullable=False, server_default="active"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "last_activity_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
    )

    # Create indexes for agent_sessions
    op.create_index(
        "ix_agent_sessions_external_id",
        "agent_sessions",
        ["external_id"],
        unique=True,
    )
    op.create_index(
        "ix_agent_sessions_user_id",
        "agent_sessions",
        ["user_id"],
    )
    op.create_index(
        "ix_agent_sessions_user_status",
        "agent_sessions",
        ["user_id", "status"],
    )
    op.create_index(
        "ix_agent_sessions_collection_id",
        "agent_sessions",
        ["collection_id"],
    )
    op.create_index(
        "ix_agent_sessions_plugin_id",
        "agent_sessions",
        ["agent_plugin_id"],
    )
    op.create_index(
        "ix_agent_sessions_last_activity",
        "agent_sessions",
        ["last_activity_at"],
    )

    # Create agent_session_messages table
    op.create_table(
        "agent_session_messages",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "session_id",
            sa.String(),
            sa.ForeignKey("agent_sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("message_id", sa.String(64), nullable=False),
        sa.Column("sequence_number", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("type", sa.String(20), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("tool_name", sa.String(64), nullable=True),
        sa.Column("tool_call_id", sa.String(64), nullable=True),
        sa.Column("tool_input", sa.JSON(), nullable=True),
        sa.Column("tool_output", sa.JSON(), nullable=True),
        sa.Column("model", sa.String(64), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("cost_usd", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    # Create indexes for agent_session_messages
    op.create_index(
        "ix_agent_session_messages_session_id",
        "agent_session_messages",
        ["session_id"],
    )
    op.create_index(
        "ix_agent_session_messages_session_seq",
        "agent_session_messages",
        ["session_id", "sequence_number"],
    )
