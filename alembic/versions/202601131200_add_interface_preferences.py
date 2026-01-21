"""Add interface preferences to user_preferences table.

Adds user interface settings for UI behavior:
- data_refresh_interval_ms: Polling interval (10s-60s, default 30s)
- visualization_sample_limit: Max points for UMAP/PCA (10k-500k, default 200k)
- animation_enabled: Enable UI animations (default true)

Revision ID: 202601131200
Revises: 202601140002
Create Date: 2026-01-13 12:00:00.000000

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa

from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence

revision: str = "202601131200"
down_revision: str | Sequence[str] | None = "202601140002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add interface preferences columns to user_preferences."""
    # Add data_refresh_interval_ms column
    op.add_column(
        "user_preferences",
        sa.Column(
            "data_refresh_interval_ms",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("30000"),
        ),
    )

    # Add visualization_sample_limit column
    op.add_column(
        "user_preferences",
        sa.Column(
            "visualization_sample_limit",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("200000"),
        ),
    )

    # Add animation_enabled column
    op.add_column(
        "user_preferences",
        sa.Column(
            "animation_enabled",
            sa.Boolean(),
            nullable=False,
            server_default=sa.true(),
        ),
    )

    # Add check constraints for valid ranges
    op.create_check_constraint(
        "ck_user_preferences_data_refresh_interval_ms",
        "user_preferences",
        "data_refresh_interval_ms >= 10000 AND data_refresh_interval_ms <= 60000",
    )

    op.create_check_constraint(
        "ck_user_preferences_visualization_sample_limit",
        "user_preferences",
        "visualization_sample_limit >= 10000 AND visualization_sample_limit <= 500000",
    )


def downgrade() -> None:
    """Remove interface preferences columns from user_preferences."""
    # Drop check constraints first
    op.drop_constraint(
        "ck_user_preferences_visualization_sample_limit",
        "user_preferences",
        type_="check",
    )
    op.drop_constraint(
        "ck_user_preferences_data_refresh_interval_ms",
        "user_preferences",
        type_="check",
    )

    # Drop columns
    op.drop_column("user_preferences", "animation_enabled")
    op.drop_column("user_preferences", "visualization_sample_limit")
    op.drop_column("user_preferences", "data_refresh_interval_ms")
