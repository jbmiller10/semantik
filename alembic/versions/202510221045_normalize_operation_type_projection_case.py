"""Ensure projection operation enum uses lowercase variant.

Revision ID: 202510221045
Revises: 202510211200
Create Date: 2025-10-22 10:45:00.000000

"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "202510221045"
down_revision: str | Sequence[str] | None = "202510211200"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Normalize projection enum to use the lowercase value."""

    # Rename the uppercase enum value to lowercase if it exists.
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1
                FROM pg_type t
                JOIN pg_enum e ON e.enumtypid = t.oid
                WHERE t.typname = 'operation_type'
                  AND e.enumlabel = 'PROJECTION_BUILD'
            ) THEN
                EXECUTE 'ALTER TYPE operation_type RENAME VALUE ''PROJECTION_BUILD'' TO ''projection_build''';
            END IF;
        END
        $$;
        """
    )

    # Ensure the lowercase value exists (for databases that never had projection operations).
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1
                FROM pg_type t
                JOIN pg_enum e ON e.enumtypid = t.oid
                WHERE t.typname = 'operation_type'
                  AND e.enumlabel = 'projection_build'
            ) THEN
                EXECUTE 'ALTER TYPE operation_type ADD VALUE ''projection_build''';
            END IF;
        END
        $$;
        """
    )


def downgrade() -> None:
    """Restore the uppercase enum label used prior to this migration."""

    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1
                FROM pg_type t
                JOIN pg_enum e ON e.enumtypid = t.oid
                WHERE t.typname = 'operation_type'
                  AND e.enumlabel = 'projection_build'
            ) THEN
                EXECUTE 'ALTER TYPE operation_type RENAME VALUE ''projection_build'' TO ''PROJECTION_BUILD''';
            END IF;
        END
        $$;
        """
    )

    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1
                FROM pg_type t
                JOIN pg_enum e ON e.enumtypid = t.oid
                WHERE t.typname = 'operation_type'
                  AND e.enumlabel = 'PROJECTION_BUILD'
            ) THEN
                EXECUTE 'ALTER TYPE operation_type ADD VALUE ''PROJECTION_BUILD''';
            END IF;
        END
        $$;
        """
    )
