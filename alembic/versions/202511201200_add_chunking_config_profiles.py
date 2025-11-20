"""Add chunking_config_profiles table for saved configs.

Revision ID: 202511201200
Revises: 202511171200
Create Date: 2025-11-20 12:00:00.000000

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa
from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence


revision: str = "202511201200"
down_revision: str | Sequence[str] | None = "202511171200"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create chunking_config_profiles table."""

    op.create_table(
        "chunking_config_profiles",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("strategy", sa.String(), nullable=False),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column("created_by", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("is_default", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("usage_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("tags", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
            server_onupdate=sa.func.now(),
        ),
        sa.UniqueConstraint("created_by", "name", name="uq_chunking_config_profiles_user_name"),
    )

    op.create_index(
        "ix_chunking_config_profiles_strategy",
        "chunking_config_profiles",
        ["strategy"],
        unique=False,
    )
    op.create_index(
        "ix_chunking_config_profiles_created_by",
        "chunking_config_profiles",
        ["created_by"],
        unique=False,
    )
    op.create_index(
        "ix_chunking_config_profiles_is_default",
        "chunking_config_profiles",
        ["is_default"],
        unique=False,
    )


def downgrade() -> None:
    """Drop chunking_config_profiles table."""

    op.drop_index("ix_chunking_config_profiles_is_default", table_name="chunking_config_profiles")
    op.drop_index("ix_chunking_config_profiles_created_by", table_name="chunking_config_profiles")
    op.drop_index("ix_chunking_config_profiles_strategy", table_name="chunking_config_profiles")
    op.drop_table("chunking_config_profiles")
