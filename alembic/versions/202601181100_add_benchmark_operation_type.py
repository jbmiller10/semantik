"""Add benchmark operation type.

Revision ID: 202601181100
Revises: 202601181000
Create Date: 2026-01-18

Adds the 'benchmark' value to the operation_type PostgreSQL enum.
This enables benchmark execution to create proper Operation records.
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "202601181100"
down_revision: str | None = "202601181000"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add 'benchmark' to operation_type enum."""
    op.execute("ALTER TYPE operation_type ADD VALUE IF NOT EXISTS 'benchmark'")


def downgrade() -> None:
    """Cannot remove enum values in PostgreSQL."""
    # PostgreSQL doesn't support removing enum values once added.
    # This is a no-op to allow downgrade commands to proceed.
    pass
