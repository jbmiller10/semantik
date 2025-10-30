"""Ensure operation_type enum values use lowercase variants.

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
    """Normalize the operation_type enum to lower-case labels."""

    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1
                FROM pg_type t
                JOIN pg_enum e ON e.enumtypid = t.oid
                WHERE t.typname = 'operation_type'
                  AND e.enumlabel = 'INDEX'
            ) THEN
                EXECUTE 'ALTER TYPE operation_type RENAME VALUE ''INDEX'' TO ''index''';
            END IF;

            IF EXISTS (
                SELECT 1
                FROM pg_type t
                JOIN pg_enum e ON e.enumtypid = t.oid
                WHERE t.typname = 'operation_type'
                  AND e.enumlabel = 'APPEND'
            ) THEN
                EXECUTE 'ALTER TYPE operation_type RENAME VALUE ''APPEND'' TO ''append''';
            END IF;

            IF EXISTS (
                SELECT 1
                FROM pg_type t
                JOIN pg_enum e ON e.enumtypid = t.oid
                WHERE t.typname = 'operation_type'
                  AND e.enumlabel = 'REINDEX'
            ) THEN
                EXECUTE 'ALTER TYPE operation_type RENAME VALUE ''REINDEX'' TO ''reindex''';
            END IF;

            IF EXISTS (
                SELECT 1
                FROM pg_type t
                JOIN pg_enum e ON e.enumtypid = t.oid
                WHERE t.typname = 'operation_type'
                  AND e.enumlabel = 'REMOVE_SOURCE'
            ) THEN
                EXECUTE 'ALTER TYPE operation_type RENAME VALUE ''REMOVE_SOURCE'' TO ''remove_source''';
            END IF;

            IF EXISTS (
                SELECT 1
                FROM pg_type t
                JOIN pg_enum e ON e.enumtypid = t.oid
                WHERE t.typname = 'operation_type'
                  AND e.enumlabel = 'DELETE'
            ) THEN
                EXECUTE 'ALTER TYPE operation_type RENAME VALUE ''DELETE'' TO ''delete''';
            END IF;

            IF EXISTS (
                SELECT 1
                FROM pg_type t
                JOIN pg_enum e ON e.enumtypid = t.oid
                WHERE t.typname = 'operation_type'
                  AND e.enumlabel = 'PROJECTION_BUILD'
            )
            AND NOT EXISTS (
                SELECT 1
                FROM pg_type t
                JOIN pg_enum e ON e.enumtypid = t.oid
                WHERE t.typname = 'operation_type'
                  AND e.enumlabel = 'projection_build'
            ) THEN
                EXECUTE 'ALTER TYPE operation_type RENAME VALUE ''PROJECTION_BUILD'' TO ''projection_build''';
            END IF;

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
    """Restore the original uppercase enum labels."""

    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1
                FROM pg_type t
                JOIN pg_enum e ON e.enumtypid = t.oid
                WHERE t.typname = 'operation_type'
                  AND e.enumlabel = 'index'
            ) THEN
                EXECUTE 'ALTER TYPE operation_type RENAME VALUE ''index'' TO ''INDEX''';
            END IF;

            IF EXISTS (
                SELECT 1
                FROM pg_type t
                JOIN pg_enum e ON e.enumtypid  = t.oid
                WHERE t.typname = 'operation_type'
                  AND e.enumlabel = 'append'
            ) THEN
                EXECUTE 'ALTER TYPE operation_type RENAME VALUE ''append'' TO ''APPEND''';
            END IF;

            IF EXISTS (
                SELECT 1
                FROM pg_type t
                JOIN pg_enum e ON e.enumtypid = t.oid
                WHERE t.typname = 'operation_type'
                  AND e.enumlabel = 'reindex'
            ) THEN
                EXECUTE 'ALTER TYPE operation_type RENAME VALUE ''reindex'' TO ''REINDEX''';
            END IF;

            IF EXISTS (
                SELECT 1
                FROM pg_type t
                JOIN pg_enum e ON e.enumtypid = t.oid
                WHERE t.typname = 'operation_type'
                  AND e.enumlabel = 'remove_source'
            ) THEN
                EXECUTE 'ALTER TYPE operation_type RENAME VALUE ''remove_source'' TO ''REMOVE_SOURCE''';
            END IF;

            IF EXISTS (
                SELECT 1
                FROM pg_type t
                JOIN pg_enum e ON e.enumtypid = t.oid
                WHERE t.typname = 'operation_type'
                  AND e.enumlabel = 'delete'
            ) THEN
                EXECUTE 'ALTER TYPE operation_type RENAME VALUE ''delete'' TO ''DELETE''';
            END IF;

            IF EXISTS (
                SELECT 1
                FROM pg_type t
                JOIN pg_enum e ON e.enumtypid = t.oid
                WHERE t.typname = 'operation_type'
                  AND e.enumlabel = 'projection_build'
            ) THEN
                EXECUTE 'ALTER TYPE operation_type RENAME VALUE ''projection_build'' TO ''PROJECTION_BUILD''';
            END IF;

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
