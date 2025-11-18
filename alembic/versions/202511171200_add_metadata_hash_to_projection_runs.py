"""Add metadata_hash column to projection_runs for idempotent recompute.

Revision ID: 202511171200
Revises: 202510221045
Create Date: 2025-11-17 12:00:00.000000

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa

from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence

revision: str = "202511171200"
down_revision: str | Sequence[str] | None = "202510221045"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add metadata_hash column and index to projection_runs."""

    op.add_column(
        "projection_runs",
        sa.Column("metadata_hash", sa.String(), nullable=True),
    )
    op.create_index(
        "ix_projection_runs_metadata_hash",
        "projection_runs",
        ["metadata_hash"],
        unique=False,
    )


def downgrade() -> None:
    """Drop metadata_hash column and index from projection_runs."""

    op.drop_index("ix_projection_runs_metadata_hash", table_name="projection_runs")
    op.drop_column("projection_runs", "metadata_hash")

