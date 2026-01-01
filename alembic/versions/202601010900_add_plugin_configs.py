"""Add plugin_configs table for external plugin management.

Revision ID: 202601010900
Revises: 202512140100
Create Date: 2026-01-01 09:00:00.000000

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa

from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence

revision: str = "202601010900"
down_revision: str | Sequence[str] | None = "202512140100"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create plugin_configs table."""
    op.create_table(
        "plugin_configs",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("type", sa.String(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column("last_health_check", sa.DateTime(timezone=True), nullable=True),
        sa.Column("health_status", sa.String(20), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
            server_onupdate=sa.func.now(),
        ),
    )

    op.create_index("ix_plugin_configs_type", "plugin_configs", ["type"], unique=False)


def downgrade() -> None:
    """Drop plugin_configs table."""
    op.drop_index("ix_plugin_configs_type", table_name="plugin_configs")
    op.drop_table("plugin_configs")
