"""Add sync fields to collection_sources for continuous sync support.

This migration adds scheduling and sync status tracking columns to
collection_sources to enable one-time import and continuous sync modes.

Changes to collection_sources:
- sync_mode: 'one_time' or 'continuous' (default: 'one_time')
- interval_minutes: sync interval for continuous mode (min 15)
- paused_at: timestamp when sync was paused (NULL = not paused)
- next_run_at: when the next sync should run
- last_run_started_at: when the last sync started
- last_run_completed_at: when the last sync completed
- last_run_status: 'success', 'failed', or 'partial'
- last_error: error message from last failed sync

Indexes:
- ix_sources_due: partial index for efficient due-source queries

Revision ID: 202512121200
Revises: 202511291200
Create Date: 2025-12-12 12:00:00.000000

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa

from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence


revision: str = "202512121200"
down_revision: str | Sequence[str] | None = "202511291200"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add sync fields to collection_sources table."""

    # Add sync_mode column (default 'one_time')
    op.add_column(
        "collection_sources",
        sa.Column(
            "sync_mode",
            sa.String(20),
            nullable=False,
            server_default="one_time",
        ),
    )

    # Add interval_minutes column (nullable, only used for continuous mode)
    op.add_column(
        "collection_sources",
        sa.Column("interval_minutes", sa.Integer(), nullable=True),
    )

    # Add paused_at column (NULL = not paused)
    op.add_column(
        "collection_sources",
        sa.Column("paused_at", sa.DateTime(timezone=True), nullable=True),
    )

    # Add next_run_at column (when the next sync should run)
    op.add_column(
        "collection_sources",
        sa.Column("next_run_at", sa.DateTime(timezone=True), nullable=True),
    )

    # Add last_run_started_at column
    op.add_column(
        "collection_sources",
        sa.Column("last_run_started_at", sa.DateTime(timezone=True), nullable=True),
    )

    # Add last_run_completed_at column
    op.add_column(
        "collection_sources",
        sa.Column("last_run_completed_at", sa.DateTime(timezone=True), nullable=True),
    )

    # Add last_run_status column ('success', 'failed', 'partial')
    op.add_column(
        "collection_sources",
        sa.Column("last_run_status", sa.String(20), nullable=True),
    )

    # Add last_error column (error message from last failed sync)
    op.add_column(
        "collection_sources",
        sa.Column("last_error", sa.Text(), nullable=True),
    )

    # Create partial index for efficient due-source queries
    # This index helps the sync dispatcher find sources that need to run
    op.create_index(
        "ix_sources_due",
        "collection_sources",
        ["next_run_at"],
        postgresql_where=sa.text("paused_at IS NULL AND sync_mode = 'continuous'"),
    )

    # Add check constraint for minimum interval
    op.execute(
        """
        ALTER TABLE collection_sources
        ADD CONSTRAINT ck_interval_minutes_minimum
        CHECK (interval_minutes IS NULL OR interval_minutes >= 15)
        """
    )

    # Add check constraint for sync_mode values
    op.execute(
        """
        ALTER TABLE collection_sources
        ADD CONSTRAINT ck_sync_mode_values
        CHECK (sync_mode IN ('one_time', 'continuous'))
        """
    )

    # Add check constraint for last_run_status values
    op.execute(
        """
        ALTER TABLE collection_sources
        ADD CONSTRAINT ck_last_run_status_values
        CHECK (last_run_status IS NULL OR last_run_status IN ('success', 'failed', 'partial'))
        """
    )


def downgrade() -> None:
    """Remove sync fields from collection_sources table."""

    # Drop check constraints
    op.execute("ALTER TABLE collection_sources DROP CONSTRAINT IF EXISTS ck_last_run_status_values")
    op.execute("ALTER TABLE collection_sources DROP CONSTRAINT IF EXISTS ck_sync_mode_values")
    op.execute("ALTER TABLE collection_sources DROP CONSTRAINT IF EXISTS ck_interval_minutes_minimum")

    # Drop the partial index
    op.drop_index("ix_sources_due", table_name="collection_sources")

    # Drop columns in reverse order
    op.drop_column("collection_sources", "last_error")
    op.drop_column("collection_sources", "last_run_status")
    op.drop_column("collection_sources", "last_run_completed_at")
    op.drop_column("collection_sources", "last_run_started_at")
    op.drop_column("collection_sources", "next_run_at")
    op.drop_column("collection_sources", "paused_at")
    op.drop_column("collection_sources", "interval_minutes")
    op.drop_column("collection_sources", "sync_mode")
