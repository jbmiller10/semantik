"""Add inline_source_config to agent_conversations.

Revision ID: 20260125120000
Revises: 202601241000
Create Date: 2026-01-25

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSON

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20260125120000"
down_revision: str | None = "202601241000"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add inline_source_config column to agent_conversations table.

    This column stores inline source configuration for sources that are
    configured during the conversation but not yet created in the database.
    The actual CollectionSource record is created when the pipeline is applied.

    Stores: {"source_type": "directory", "source_config": {...}, "_pending_secrets": {...}}
    """
    op.add_column(
        "agent_conversations",
        sa.Column("inline_source_config", JSON, nullable=True),
    )


def downgrade() -> None:
    """Remove inline_source_config column."""
    op.drop_column("agent_conversations", "inline_source_config")
