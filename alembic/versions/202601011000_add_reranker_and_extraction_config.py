"""Add reranker and extraction config fields to collections.

Revision ID: 202601011000
Revises: 202601010900
Create Date: 2026-01-01 10:00:00.000000

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa

from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence

revision: str = "202601011000"
down_revision: str | Sequence[str] | None = "202601010900"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add reranker and extraction config columns to collections table."""
    # Add default_reranker_id column
    op.add_column(
        "collections",
        sa.Column("default_reranker_id", sa.String(), nullable=True),
    )

    # Add extraction_config column (JSON for flexible configuration)
    # Schema: {"enabled": bool, "extractor_ids": [str], "types": [str], "options": {}}
    op.add_column(
        "collections",
        sa.Column("extraction_config", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    """Remove reranker and extraction config columns from collections table."""
    op.drop_column("collections", "extraction_config")
    op.drop_column("collections", "default_reranker_id")
