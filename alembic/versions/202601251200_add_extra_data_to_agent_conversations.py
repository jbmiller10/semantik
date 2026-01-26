"""Add extra_data column to agent_conversations for mode toggle.

Revision ID: 202601251200
Revises: 20260125120000
Create Date: 2026-01-25 12:00:00.000000

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSON

from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence

revision: str = "202601251200"
down_revision: str | Sequence[str] | None = "20260125120000"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add extra_data column to agent_conversations."""
    op.add_column(
        "agent_conversations",
        sa.Column("extra_data", JSON, nullable=True),
    )


def downgrade() -> None:
    """Remove extra_data column from agent_conversations."""
    op.drop_column("agent_conversations", "extra_data")
