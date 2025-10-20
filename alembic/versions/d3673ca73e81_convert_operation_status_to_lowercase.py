"""convert_operation_status_to_lowercase

Revision ID: d3673ca73e81
Revises: db004_add_chunking_indexes
Create Date: 2025-08-12 10:33:23.318557

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d3673ca73e81"
down_revision: str | None = "add_complete_chunking"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Convert OperationStatus enum values from uppercase to lowercase.

    This migration recreates the enum type with lowercase values.
    """

    # First, we need to remove the default constraint if it exists
    op.execute("ALTER TABLE operations ALTER COLUMN status DROP DEFAULT")

    # Create a temporary text column to hold the values during migration
    op.execute("ALTER TABLE operations ADD COLUMN status_temp TEXT")

    # Copy the current status values to the temp column
    op.execute(
        """
        UPDATE operations
        SET status_temp = status::text
    """
    )

    # Drop the old status column
    op.execute("ALTER TABLE operations DROP COLUMN status")

    # Create a new enum type with lowercase values
    op.execute("CREATE TYPE operation_status_new AS ENUM ('pending', 'processing', 'completed', 'failed', 'cancelled')")

    # Add the new status column with the new enum type
    op.execute("ALTER TABLE operations ADD COLUMN status operation_status_new")

    # Populate the new status column with converted values
    op.execute(
        """
        UPDATE operations
        SET status = CASE status_temp
            WHEN 'PENDING' THEN 'pending'::operation_status_new
            WHEN 'PROCESSING' THEN 'processing'::operation_status_new
            WHEN 'COMPLETED' THEN 'completed'::operation_status_new
            WHEN 'FAILED' THEN 'failed'::operation_status_new
            WHEN 'CANCELLED' THEN 'cancelled'::operation_status_new
            WHEN 'pending' THEN 'pending'::operation_status_new
            WHEN 'processing' THEN 'processing'::operation_status_new
            WHEN 'completed' THEN 'completed'::operation_status_new
            WHEN 'failed' THEN 'failed'::operation_status_new
            WHEN 'cancelled' THEN 'cancelled'::operation_status_new
        END
    """
    )

    # Make the column not null
    op.execute("ALTER TABLE operations ALTER COLUMN status SET NOT NULL")

    # Set the default value
    op.execute("ALTER TABLE operations ALTER COLUMN status SET DEFAULT 'pending'::operation_status_new")

    # Create index on the status column
    op.execute("CREATE INDEX IF NOT EXISTS ix_operations_status ON operations(status)")

    # Drop the temporary column
    op.execute("ALTER TABLE operations DROP COLUMN status_temp")

    # Drop the old enum type
    op.execute("DROP TYPE IF EXISTS operation_status")

    # Rename the new enum type to the original name
    op.execute("ALTER TYPE operation_status_new RENAME TO operation_status")

    # Recreate partial index to match new lowercase enum values
    # Previous migration created idx_operations_user_status with uppercase constants
    op.execute("DROP INDEX IF EXISTS idx_operations_user_status")
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_operations_user_status
        ON operations(user_id, status)
        WHERE status IN ('processing', 'pending');
        """
    )

    # Update any audit_logs that might reference operation status in JSONB, if table exists
    op.execute(
        """
        DO $$
        BEGIN
            IF to_regclass('public.audit_logs') IS NOT NULL THEN
                UPDATE audit_logs
                SET details = jsonb_set(
                    details,
                    '{status}',
                    to_jsonb(LOWER(details->>'status'))
                )
                WHERE details->>'status' IN ('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED', 'CANCELLED');
            END IF;
        END
        $$;
        """
    )


def downgrade() -> None:
    """Revert OperationStatus enum values back to uppercase."""

    # Remove the default constraint
    op.execute("ALTER TABLE operations ALTER COLUMN status DROP DEFAULT")

    # Create a temporary text column
    op.execute("ALTER TABLE operations ADD COLUMN status_temp TEXT")

    # Copy the current status values to the temp column
    op.execute(
        """
        UPDATE operations
        SET status_temp = status::text
    """
    )

    # Drop the status column
    op.execute("ALTER TABLE operations DROP COLUMN status")

    # Create a new enum type with uppercase values
    op.execute("CREATE TYPE operation_status_new AS ENUM ('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED', 'CANCELLED')")

    # Add the new status column with the new enum type
    op.execute("ALTER TABLE operations ADD COLUMN status operation_status_new")

    # Populate the new status column with converted values
    op.execute(
        """
        UPDATE operations
        SET status = CASE status_temp
            WHEN 'pending' THEN 'PENDING'::operation_status_new
            WHEN 'processing' THEN 'PROCESSING'::operation_status_new
            WHEN 'completed' THEN 'COMPLETED'::operation_status_new
            WHEN 'failed' THEN 'FAILED'::operation_status_new
            WHEN 'cancelled' THEN 'CANCELLED'::operation_status_new
            WHEN 'PENDING' THEN 'PENDING'::operation_status_new
            WHEN 'PROCESSING' THEN 'PROCESSING'::operation_status_new
            WHEN 'COMPLETED' THEN 'COMPLETED'::operation_status_new
            WHEN 'FAILED' THEN 'FAILED'::operation_status_new
            WHEN 'CANCELLED' THEN 'CANCELLED'::operation_status_new
        END
    """
    )

    # Make the column not null
    op.execute("ALTER TABLE operations ALTER COLUMN status SET NOT NULL")

    # Set the default value
    op.execute("ALTER TABLE operations ALTER COLUMN status SET DEFAULT 'PENDING'::operation_status_new")

    # Create index on the status column
    op.execute("CREATE INDEX IF NOT EXISTS ix_operations_status ON operations(status)")

    # Drop the temporary column
    op.execute("ALTER TABLE operations DROP COLUMN status_temp")

    # Drop the old enum type
    op.execute("DROP TYPE IF EXISTS operation_status")

    # Rename the new enum type to the original name
    op.execute("ALTER TYPE operation_status_new RENAME TO operation_status")

    # Recreate partial index with uppercase enum values to match downgrade
    op.execute("DROP INDEX IF EXISTS idx_operations_user_status")
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_operations_user_status
        ON operations(user_id, status)
        WHERE status IN ('PROCESSING', 'PENDING');
        """
    )

    # Revert audit_logs status values back to uppercase, if table exists
    op.execute(
        """
        DO $$
        BEGIN
            IF to_regclass('public.audit_logs') IS NOT NULL THEN
                UPDATE audit_logs
                SET details = jsonb_set(
                    details,
                    '{status}',
                    to_jsonb(UPPER(details->>'status'))
                )
                WHERE details->>'status' IN ('pending', 'processing', 'completed', 'failed', 'cancelled');
            END IF;
        END
        $$;
        """
    )
