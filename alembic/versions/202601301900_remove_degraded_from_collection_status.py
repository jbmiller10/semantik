"""Remove DEGRADED from collection_status enum.

Revision ID: 202601301900
Revises: 202601181000
Create Date: 2026-01-30

This migration:
1. Updates all collections with status 'DEGRADED' to 'READY'
2. Removes the 'DEGRADED' value from the collection_status enum
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "202601301900"
down_revision = "202601251200"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Step 1: Update all DEGRADED collections to READY
    op.execute("""
        UPDATE collections
        SET status = 'READY'
        WHERE status = 'DEGRADED'
    """)

    # Step 2: Remove DEGRADED from the enum
    # PostgreSQL requires recreating the enum type to remove a value
    # We need to:
    # 1. Create a new enum without DEGRADED
    # 2. Update the column to use the new enum
    # 3. Drop the old enum
    # 4. Rename the new enum

    # Create new enum type without DEGRADED
    op.execute("""
        CREATE TYPE collection_status_new AS ENUM ('PENDING', 'READY', 'PROCESSING', 'ERROR')
    """)

    # Update the column to use the new type
    op.execute("""
        ALTER TABLE collections
        ALTER COLUMN status TYPE collection_status_new
        USING status::text::collection_status_new
    """)

    # Drop the old enum
    op.execute("""
        DROP TYPE collection_status
    """)

    # Rename new enum to original name
    op.execute("""
        ALTER TYPE collection_status_new RENAME TO collection_status
    """)


def downgrade() -> None:
    # Re-add DEGRADED to the enum
    # Create new enum with DEGRADED
    op.execute("""
        CREATE TYPE collection_status_new AS ENUM ('PENDING', 'READY', 'PROCESSING', 'ERROR', 'DEGRADED')
    """)

    # Update the column to use the new type
    op.execute("""
        ALTER TABLE collections
        ALTER COLUMN status TYPE collection_status_new
        USING status::text::collection_status_new
    """)

    # Drop the old enum
    op.execute("""
        DROP TYPE collection_status
    """)

    # Rename new enum to original name
    op.execute("""
        ALTER TYPE collection_status_new RENAME TO collection_status
    """)
