"""Add indexes to plugin_configs for performance.

Revision ID: 202601021000
Revises: 202601011000
Create Date: 2026-01-02 10:00:00.000000

"""

from __future__ import annotations

from typing import TYPE_CHECKING

from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence

revision: str = "202601021000"
down_revision: str | Sequence[str] | None = "202601011000"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add performance indexes to plugin_configs table."""
    # Index on enabled column for filtering
    op.create_index(
        "ix_plugin_configs_enabled",
        "plugin_configs",
        ["enabled"],
        unique=False,
    )

    # Composite index on type + enabled for common query pattern
    op.create_index(
        "ix_plugin_configs_type_enabled",
        "plugin_configs",
        ["type", "enabled"],
        unique=False,
    )

    # Index on health_status for monitoring queries
    op.create_index(
        "ix_plugin_configs_health_status",
        "plugin_configs",
        ["health_status"],
        unique=False,
    )

    # Partial index on last_health_check for active health monitoring
    # Only indexes rows that have been health-checked at least once
    op.create_index(
        "ix_plugin_configs_last_health_active",
        "plugin_configs",
        ["last_health_check"],
        unique=False,
        postgresql_where="last_health_check IS NOT NULL",
    )


def downgrade() -> None:
    """Remove performance indexes from plugin_configs table."""
    op.drop_index("ix_plugin_configs_last_health_active", table_name="plugin_configs")
    op.drop_index("ix_plugin_configs_health_status", table_name="plugin_configs")
    op.drop_index("ix_plugin_configs_type_enabled", table_name="plugin_configs")
    op.drop_index("ix_plugin_configs_enabled", table_name="plugin_configs")
