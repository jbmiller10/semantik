"""Update search_top_k constraint from 5-50 to 1-250.

Expands the allowed range for search results count to support
more flexible search configurations.

Revision ID: 202601140003
Revises: 202601131200
Create Date: 2026-01-13 20:00:00.000000

"""

from __future__ import annotations

from typing import TYPE_CHECKING

from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence

revision: str = "202601140003"
down_revision: str | Sequence[str] | None = "202601131200"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Update search_top_k constraint to allow 1-250."""
    # Drop old constraint
    op.drop_constraint(
        "ck_user_preferences_search_top_k",
        "user_preferences",
        type_="check",
    )

    # Create new constraint with expanded range
    op.create_check_constraint(
        "ck_user_preferences_search_top_k",
        "user_preferences",
        "search_top_k >= 1 AND search_top_k <= 250",
    )


def downgrade() -> None:
    """Revert search_top_k constraint to 5-50."""
    # Drop new constraint
    op.drop_constraint(
        "ck_user_preferences_search_top_k",
        "user_preferences",
        type_="check",
    )

    # Restore old constraint
    op.create_check_constraint(
        "ck_user_preferences_search_top_k",
        "user_preferences",
        "search_top_k >= 5 AND search_top_k <= 50",
    )
