"""Add MCP profiles for LLM client integration.

Adds tables for MCP (Model Context Protocol) search profiles that allow
LLM clients like Claude Desktop to search Semantik collections via
standardized tool interfaces.

- mcp_profiles: Profile configuration with search defaults
- mcp_profile_collections: Many-to-many mapping with ordering

Revision ID: 202601061000
Revises: 202601021000
Create Date: 2026-01-06 10:00:00.000000

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa

from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence

revision: str = "202601061000"
down_revision: str | Sequence[str] | None = "202601021000"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create MCP profiles tables."""
    # Create mcp_profiles table
    op.create_table(
        "mcp_profiles",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("name", sa.String(64), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column(
            "owner_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("search_type", sa.String(32), nullable=False, server_default="semantic"),
        sa.Column("result_count", sa.Integer(), nullable=False, server_default=sa.text("10")),
        sa.Column("use_reranker", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("score_threshold", sa.Float(), nullable=True),
        sa.Column("hybrid_alpha", sa.Float(), nullable=True),
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
    )

    # Create index on owner_id for fast profile listing
    op.create_index("ix_mcp_profiles_owner_id", "mcp_profiles", ["owner_id"])

    # Create unique constraint on (owner_id, name)
    op.create_unique_constraint(
        "uq_mcp_profiles_owner_name",
        "mcp_profiles",
        ["owner_id", "name"],
    )

    # Create mcp_profile_collections junction table
    op.create_table(
        "mcp_profile_collections",
        sa.Column(
            "profile_id",
            sa.String(),
            sa.ForeignKey("mcp_profiles.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "collection_id",
            sa.String(),
            sa.ForeignKey("collections.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("order", sa.Integer(), nullable=False, server_default=sa.text("0")),
    )


def downgrade() -> None:
    """Remove MCP profiles tables."""
    op.drop_table("mcp_profile_collections")
    op.drop_index("ix_mcp_profiles_owner_id", table_name="mcp_profiles")
    op.drop_constraint("uq_mcp_profiles_owner_name", "mcp_profiles", type_="unique")
    op.drop_table("mcp_profiles")
