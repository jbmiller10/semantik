"""Move sync policy from sources to collections for collection-level sync.

This migration moves sync scheduling from the source level to the collection
level, enabling a unified sync policy that applies to all sources in a collection.

Changes to collections table:
- sync_mode: 'one_time' or 'continuous' (default: 'one_time')
- sync_interval_minutes: sync interval for continuous mode (min 15)
- sync_paused_at: timestamp when sync was paused (NULL = not paused)
- sync_next_run_at: when the next sync should run
- sync_last_run_started_at: when the last sync run started
- sync_last_run_completed_at: when the last sync run completed
- sync_last_run_status: 'running', 'success', 'failed', or 'partial'
- sync_last_error: error summary from last failed sync

New collection_sync_runs table:
- Tracks each sync run's progress across all sources
- Aggregates completion status from individual source operations

Changes to collection_sources table:
- Remove: sync_mode, interval_minutes, paused_at, next_run_at (scheduling)
- Keep: last_run_started_at, last_run_completed_at, last_run_status, last_error (telemetry)

Revision ID: 202512140100
Revises: 202512130200
Create Date: 2025-12-14 01:00:00.000000

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa

from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence


revision: str = "202512140100"
down_revision: str | Sequence[str] | None = "202512130200"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Move sync policy from sources to collections."""

    # =========================================================================
    # 1. Add sync policy fields to collections table
    # =========================================================================

    # sync_mode: 'one_time' or 'continuous'
    op.add_column(
        "collections",
        sa.Column(
            "sync_mode",
            sa.String(20),
            nullable=False,
            server_default="one_time",
        ),
    )

    # sync_interval_minutes: required for continuous mode (min 15)
    op.add_column(
        "collections",
        sa.Column("sync_interval_minutes", sa.Integer(), nullable=True),
    )

    # sync_paused_at: NULL means not paused
    op.add_column(
        "collections",
        sa.Column("sync_paused_at", sa.DateTime(timezone=True), nullable=True),
    )

    # sync_next_run_at: when the next sync should run
    op.add_column(
        "collections",
        sa.Column("sync_next_run_at", sa.DateTime(timezone=True), nullable=True),
    )

    # sync_last_run_started_at: when the last sync run started
    op.add_column(
        "collections",
        sa.Column("sync_last_run_started_at", sa.DateTime(timezone=True), nullable=True),
    )

    # sync_last_run_completed_at: when the last sync run completed
    op.add_column(
        "collections",
        sa.Column("sync_last_run_completed_at", sa.DateTime(timezone=True), nullable=True),
    )

    # sync_last_run_status: 'running', 'success', 'failed', 'partial'
    op.add_column(
        "collections",
        sa.Column("sync_last_run_status", sa.String(20), nullable=True),
    )

    # sync_last_error: error summary from last failed sync
    op.add_column(
        "collections",
        sa.Column("sync_last_error", sa.Text(), nullable=True),
    )

    # Create partial index for efficient due-collection queries
    op.create_index(
        "ix_collections_sync_due",
        "collections",
        ["sync_next_run_at"],
        postgresql_where=sa.text("sync_paused_at IS NULL AND sync_mode = 'continuous'"),
    )

    # Add check constraint for collection sync_mode values
    op.execute(
        """
        ALTER TABLE collections
        ADD CONSTRAINT ck_collection_sync_mode
        CHECK (sync_mode IN ('one_time', 'continuous'))
        """
    )

    # Add check constraint for minimum interval
    op.execute(
        """
        ALTER TABLE collections
        ADD CONSTRAINT ck_collection_sync_interval_min
        CHECK (sync_interval_minutes IS NULL OR sync_interval_minutes >= 15)
        """
    )

    # Add check constraint for sync_last_run_status values
    op.execute(
        """
        ALTER TABLE collections
        ADD CONSTRAINT ck_collection_sync_status
        CHECK (sync_last_run_status IS NULL OR sync_last_run_status IN ('running', 'success', 'failed', 'partial'))
        """
    )

    # =========================================================================
    # 2. Create collection_sync_runs table for run aggregation
    # =========================================================================

    op.create_table(
        "collection_sync_runs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "collection_id",
            sa.String(),
            sa.ForeignKey("collections.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("triggered_by", sa.String(50), nullable=False),  # 'scheduler', 'manual'
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "status",
            sa.String(20),
            nullable=False,
            server_default="running",
        ),  # running, success, failed, partial
        sa.Column("expected_sources", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("completed_sources", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("failed_sources", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("partial_sources", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error_summary", sa.Text(), nullable=True),
        sa.Column("meta", sa.JSON(), nullable=True),
    )

    # Create indexes for collection_sync_runs
    op.create_index(
        "ix_sync_runs_collection_id",
        "collection_sync_runs",
        ["collection_id"],
    )
    op.create_index(
        "ix_sync_runs_status",
        "collection_sync_runs",
        ["status"],
    )
    op.create_index(
        "ix_sync_runs_started_at",
        "collection_sync_runs",
        ["started_at"],
    )

    # Add check constraint for sync run status values
    op.execute(
        """
        ALTER TABLE collection_sync_runs
        ADD CONSTRAINT ck_sync_run_status
        CHECK (status IN ('running', 'success', 'failed', 'partial'))
        """
    )

    # =========================================================================
    # 3. Remove scheduling fields from collection_sources (keep telemetry)
    # =========================================================================

    # Drop the partial index first
    op.drop_index("ix_sources_due", table_name="collection_sources")

    # Drop check constraints for removed fields
    op.execute("ALTER TABLE collection_sources DROP CONSTRAINT IF EXISTS ck_interval_minutes_minimum")
    op.execute("ALTER TABLE collection_sources DROP CONSTRAINT IF EXISTS ck_sync_mode_values")

    # Drop scheduling columns (keep telemetry: last_run_*, last_error)
    op.drop_column("collection_sources", "sync_mode")
    op.drop_column("collection_sources", "interval_minutes")
    op.drop_column("collection_sources", "paused_at")
    op.drop_column("collection_sources", "next_run_at")


def downgrade() -> None:
    """Restore source-level sync policy."""

    # =========================================================================
    # 1. Restore scheduling fields on collection_sources
    # =========================================================================

    # Re-add sync_mode column
    op.add_column(
        "collection_sources",
        sa.Column(
            "sync_mode",
            sa.String(20),
            nullable=False,
            server_default="one_time",
        ),
    )

    # Re-add interval_minutes column
    op.add_column(
        "collection_sources",
        sa.Column("interval_minutes", sa.Integer(), nullable=True),
    )

    # Re-add paused_at column
    op.add_column(
        "collection_sources",
        sa.Column("paused_at", sa.DateTime(timezone=True), nullable=True),
    )

    # Re-add next_run_at column
    op.add_column(
        "collection_sources",
        sa.Column("next_run_at", sa.DateTime(timezone=True), nullable=True),
    )

    # Re-create partial index
    op.create_index(
        "ix_sources_due",
        "collection_sources",
        ["next_run_at"],
        postgresql_where=sa.text("paused_at IS NULL AND sync_mode = 'continuous'"),
    )

    # Re-add check constraints
    op.execute(
        """
        ALTER TABLE collection_sources
        ADD CONSTRAINT ck_interval_minutes_minimum
        CHECK (interval_minutes IS NULL OR interval_minutes >= 15)
        """
    )
    op.execute(
        """
        ALTER TABLE collection_sources
        ADD CONSTRAINT ck_sync_mode_values
        CHECK (sync_mode IN ('one_time', 'continuous'))
        """
    )

    # =========================================================================
    # 2. Drop collection_sync_runs table
    # =========================================================================

    op.drop_table("collection_sync_runs")

    # =========================================================================
    # 3. Remove sync fields from collections table
    # =========================================================================

    # Drop check constraints
    op.execute("ALTER TABLE collections DROP CONSTRAINT IF EXISTS ck_collection_sync_status")
    op.execute("ALTER TABLE collections DROP CONSTRAINT IF EXISTS ck_collection_sync_interval_min")
    op.execute("ALTER TABLE collections DROP CONSTRAINT IF EXISTS ck_collection_sync_mode")

    # Drop index
    op.drop_index("ix_collections_sync_due", table_name="collections")

    # Drop columns
    op.drop_column("collections", "sync_last_error")
    op.drop_column("collections", "sync_last_run_status")
    op.drop_column("collections", "sync_last_run_completed_at")
    op.drop_column("collections", "sync_last_run_started_at")
    op.drop_column("collections", "sync_next_run_at")
    op.drop_column("collections", "sync_paused_at")
    op.drop_column("collections", "sync_interval_minutes")
    op.drop_column("collections", "sync_mode")
