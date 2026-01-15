"""Add HyDE settings to user_preferences and mcp_profiles.

Revision ID: 202601140005
Revises: 202601140004
Create Date: 2026-01-14

Adds HyDE (Hypothetical Document Embeddings) configuration:
- user_preferences: search_use_hyde, search_hyde_quality_tier, search_hyde_timeout_seconds
- mcp_profiles: use_hyde
"""

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "202601140005"
down_revision = "202601140004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add HyDE settings columns."""
    # Add HyDE columns to user_preferences
    op.add_column(
        "user_preferences",
        sa.Column("search_use_hyde", sa.Boolean(), nullable=False, server_default="false"),
    )
    op.add_column(
        "user_preferences",
        sa.Column(
            "search_hyde_quality_tier",
            sa.String(4),
            nullable=False,
            server_default="low",
        ),
    )
    op.add_column(
        "user_preferences",
        sa.Column(
            "search_hyde_timeout_seconds",
            sa.Integer(),
            nullable=False,
            server_default="10",
        ),
    )

    # Add check constraints for user_preferences
    op.create_check_constraint(
        "ck_user_preferences_search_hyde_quality_tier",
        "user_preferences",
        "search_hyde_quality_tier IN ('high', 'low')",
    )
    op.create_check_constraint(
        "ck_user_preferences_search_hyde_timeout",
        "user_preferences",
        "search_hyde_timeout_seconds >= 3 AND search_hyde_timeout_seconds <= 60",
    )

    # Add use_hyde column to mcp_profiles
    op.add_column(
        "mcp_profiles",
        sa.Column("use_hyde", sa.Boolean(), nullable=False, server_default="false"),
    )


def downgrade() -> None:
    """Remove HyDE settings columns."""
    # Remove from mcp_profiles
    op.drop_column("mcp_profiles", "use_hyde")

    # Remove constraints first
    op.drop_constraint(
        "ck_user_preferences_search_hyde_timeout", "user_preferences", type_="check"
    )
    op.drop_constraint(
        "ck_user_preferences_search_hyde_quality_tier", "user_preferences", type_="check"
    )

    # Remove columns from user_preferences
    op.drop_column("user_preferences", "search_hyde_timeout_seconds")
    op.drop_column("user_preferences", "search_hyde_quality_tier")
    op.drop_column("user_preferences", "search_use_hyde")
