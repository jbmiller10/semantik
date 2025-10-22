"""Ensure projection operation enum uses uppercase variant.

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
    """Rename projection enum value to uppercase to match existing values."""

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


def downgrade() -> None:
    """Rename projection enum value back to lowercase."""

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
