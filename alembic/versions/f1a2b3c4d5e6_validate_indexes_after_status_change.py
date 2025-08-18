"""validate_indexes_after_status_change

Revision ID: f1a2b3c4d5e6
Revises: d3673ca73e81
Create Date: 2025-08-12 12:10:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f1a2b3c4d5e6"
down_revision: str | None = "d3673ca73e81"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Ensure partial indexes match current enum values and exist."""
    bind = op.get_bind()

    # Detect whether operation_status enum is lowercase mode
    is_lowercase = bind.execute(
        sa.text(
            """
            SELECT EXISTS (
                SELECT 1
                FROM pg_type t
                JOIN pg_enum e ON t.oid = e.enumtypid
                WHERE t.typname = 'operation_status' AND e.enumlabel = 'pending'
            )
            """
        )
    ).scalar()

    # Drop and recreate the partial index with the correct predicate
    op.execute("DROP INDEX IF EXISTS idx_operations_user_status")
    if is_lowercase:
        op.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_operations_user_status
            ON operations(user_id, status)
            WHERE status IN ('processing', 'pending');
            """
        )
    else:
        op.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_operations_user_status
            ON operations(user_id, status)
            WHERE status IN ('PROCESSING', 'PENDING');
            """
        )

    # Ensure a simple index on status exists (harmless if already present)
    op.execute("CREATE INDEX IF NOT EXISTS ix_operations_status ON operations(status)")


def downgrade() -> None:
    """Best-effort reversal: recreate index for uppercase values."""
    # On downgrade, prefer uppercase predicate to align with previous revision
    op.execute("DROP INDEX IF EXISTS idx_operations_user_status")
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_operations_user_status
        ON operations(user_id, status)
        WHERE status IN ('PROCESSING', 'PENDING');
        """
    )
