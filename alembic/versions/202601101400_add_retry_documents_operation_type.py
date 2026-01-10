"""Add retry_documents operation type.

Revision ID: 202601101400
Revises: 202601101300
Create Date: 2026-01-10 14:00:00.000000+00:00

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "202601101400"
down_revision: str | None = "202601101300"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add retry_documents value to operationtype enum."""
    # Add the new enum value to the PostgreSQL enum type
    op.execute("ALTER TYPE operationtype ADD VALUE IF NOT EXISTS 'retry_documents'")


def downgrade() -> None:
    """Remove retry_documents value from operationtype enum.

    Note: PostgreSQL does not support removing enum values directly.
    This downgrade is a no-op as the value won't cause issues if unused.
    """
    # PostgreSQL doesn't support removing enum values
    # The value will simply be unused after downgrade
    pass
