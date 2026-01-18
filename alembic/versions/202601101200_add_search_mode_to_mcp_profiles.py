"""Add search_mode and rrf_k to mcp_profiles.

Adds new columns for sparse/hybrid search mode configuration:
- search_mode: dense, sparse, or hybrid (default: dense)
- rrf_k: RRF constant for hybrid mode (nullable, default: 60 at runtime)

Revision ID: 202601101200
Revises: 202601071000
Create Date: 2026-01-10 12:00:00.000000

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa

from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence

revision: str = "202601101200"
down_revision: str | Sequence[str] | None = "202601071000"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add search_mode and rrf_k columns to mcp_profiles."""
    # Add search_mode column with default value
    op.add_column(
        "mcp_profiles",
        sa.Column(
            "search_mode",
            sa.String(16),
            nullable=False,
            server_default="dense",
        ),
    )

    # Add rrf_k column (nullable)
    op.add_column(
        "mcp_profiles",
        sa.Column("rrf_k", sa.Integer(), nullable=True),
    )

    # Add check constraint for search_mode
    op.create_check_constraint(
        "ck_mcp_profiles_search_mode",
        "mcp_profiles",
        "search_mode IN ('dense', 'sparse', 'hybrid')",
    )

    # Add check constraint for rrf_k
    op.create_check_constraint(
        "ck_mcp_profiles_rrf_k",
        "mcp_profiles",
        "rrf_k IS NULL OR (rrf_k >= 1 AND rrf_k <= 1000)",
    )


def downgrade() -> None:
    """Remove search_mode and rrf_k columns from mcp_profiles."""
    # Drop check constraints first
    op.drop_constraint("ck_mcp_profiles_rrf_k", "mcp_profiles", type_="check")
    op.drop_constraint("ck_mcp_profiles_search_mode", "mcp_profiles", type_="check")

    # Drop columns
    op.drop_column("mcp_profiles", "rrf_k")
    op.drop_column("mcp_profiles", "search_mode")
