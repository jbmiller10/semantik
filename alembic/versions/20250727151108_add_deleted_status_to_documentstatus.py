"""Add DELETED status to DocumentStatus enum

Revision ID: 20250727151108
Revises: 005a8fe3aedc
Create Date: 2025-07-27 15:11:08

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20250727151108"
down_revision: str | Sequence[str] | None = "005a8fe3aedc"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add DELETED status to document_status enum."""
    # PostgreSQL requires special handling for enum alterations
    # We need to add the new value to the enum type
    op.execute("ALTER TYPE document_status ADD VALUE IF NOT EXISTS 'DELETED'")


def downgrade() -> None:
    """Remove DELETED status from document_status enum."""
    # Note: PostgreSQL doesn't support removing values from enums directly.
    # In a production environment, you would need to:
    # 1. Create a new enum type without DELETED
    # 2. Alter the column to use the new enum
    # 3. Drop the old enum
    # 4. Rename the new enum to the original name
    # For now, we'll leave this as a no-op since it's complex and rarely needed
    # PostgreSQL doesn't support removing enum values directly
    return
