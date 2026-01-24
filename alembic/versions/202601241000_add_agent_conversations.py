"""Add agent conversation tables for Phase 3.

Revision ID: 202601241000
Revises: 202601231000
Create Date: 2026-01-24

This migration adds database support for the agent service:
- agent_conversations: Persistent conversation state
- conversation_uncertainties: Issues flagged during conversation
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text, text
from sqlalchemy.dialects.postgresql import JSON, UUID

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "202601241000"
down_revision: str | None = "202601231000"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add agent conversation tables."""
    # Create conversation_status enum if it doesn't exist
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'conversation_status') THEN
                CREATE TYPE conversation_status AS ENUM ('active', 'applied', 'abandoned');
            END IF;
        END
        $$;
    """
    )

    # Create uncertainty_severity enum if it doesn't exist
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'uncertainty_severity') THEN
                CREATE TYPE uncertainty_severity AS ENUM ('blocking', 'notable', 'info');
            END IF;
        END
        $$;
    """
    )

    # Create agent_conversations table
    op.create_table(
        "agent_conversations",
        Column("id", UUID, primary_key=True),
        Column("user_id", Integer, nullable=False, index=True),
        Column("source_id", Integer, nullable=True, index=True),
        Column("collection_id", String, nullable=True),
        # State
        Column(
            "status",
            sa.Enum("active", "applied", "abandoned", name="conversation_status", create_type=False),
            nullable=False,
            server_default="active",
        ),
        Column("current_pipeline", JSON, nullable=True),
        Column("source_analysis", JSON, nullable=True),
        Column("summary", Text, nullable=True),
        # Timestamps
        Column(
            "created_at",
            DateTime(timezone=True),
            nullable=False,
            server_default=text("now()"),
        ),
        Column(
            "updated_at",
            DateTime(timezone=True),
            nullable=False,
            server_default=text("now()"),
        ),
    )

    # Add foreign key constraints for agent_conversations
    op.create_foreign_key(
        "fk_agent_conversations_user_id",
        "agent_conversations",
        "users",
        ["user_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "fk_agent_conversations_source_id",
        "agent_conversations",
        "collection_sources",
        ["source_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "fk_agent_conversations_collection_id",
        "agent_conversations",
        "collections",
        ["collection_id"],
        ["id"],
        ondelete="SET NULL",
    )

    # Create conversation_uncertainties table
    op.create_table(
        "conversation_uncertainties",
        Column("id", UUID, primary_key=True),
        Column("conversation_id", UUID, nullable=False, index=True),
        Column(
            "severity",
            sa.Enum("blocking", "notable", "info", name="uncertainty_severity", create_type=False),
            nullable=False,
        ),
        Column("message", Text, nullable=False),
        Column("context", JSON, nullable=True),
        Column("resolved", Boolean, nullable=False, server_default="false"),
        Column("resolved_by", String(50), nullable=True),
        Column(
            "created_at",
            DateTime(timezone=True),
            nullable=False,
            server_default=text("now()"),
        ),
    )

    # Add foreign key constraint for conversation_uncertainties
    op.create_foreign_key(
        "fk_conversation_uncertainties_conversation_id",
        "conversation_uncertainties",
        "agent_conversations",
        ["conversation_id"],
        ["id"],
        ondelete="CASCADE",
    )

    # Create partial index for finding unresolved blocking issues
    op.create_index(
        "ix_uncertainties_blocking_unresolved",
        "conversation_uncertainties",
        ["conversation_id", "severity", "resolved"],
        postgresql_where=text("severity = 'blocking' AND resolved = false"),
    )


def downgrade() -> None:
    """Remove agent conversation tables."""
    # Drop foreign keys first
    op.drop_constraint(
        "fk_conversation_uncertainties_conversation_id",
        "conversation_uncertainties",
        type_="foreignkey",
    )
    op.drop_constraint(
        "fk_agent_conversations_collection_id",
        "agent_conversations",
        type_="foreignkey",
    )
    op.drop_constraint(
        "fk_agent_conversations_source_id",
        "agent_conversations",
        type_="foreignkey",
    )
    op.drop_constraint(
        "fk_agent_conversations_user_id",
        "agent_conversations",
        type_="foreignkey",
    )

    # Drop indexes and tables
    op.drop_index("ix_uncertainties_blocking_unresolved", "conversation_uncertainties")
    op.drop_table("conversation_uncertainties")
    op.drop_table("agent_conversations")

    # Drop enums if they exist
    op.execute("DROP TYPE IF EXISTS uncertainty_severity")
    op.execute("DROP TYPE IF EXISTS conversation_status")
