"""Add projection run tracking table.

Revision ID: 202510211200
Revises: 202510201015
Create Date: 2025-10-21 12:00:00.000000

"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "202510211200"
down_revision: str | Sequence[str] | None = "202510201015"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


projection_run_status = postgresql.ENUM(
    "pending",
    "running",
    "completed",
    "failed",
    "cancelled",
    name="projection_run_status",
)


def upgrade() -> None:
    """Create projection_runs table and extend operation_type enum."""

    op.execute("ALTER TYPE operation_type ADD VALUE IF NOT EXISTS 'projection_build'")

    bind = op.get_bind()
    projection_run_status.create(bind, checkfirst=True)

    op.create_table(
        "projection_runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("uuid", sa.String(), nullable=False),
        sa.Column("collection_id", sa.String(), nullable=False),
        sa.Column("operation_uuid", sa.String(), nullable=True),
        sa.Column("status", projection_run_status, nullable=False, server_default="pending"),
        sa.Column("dimensionality", sa.Integer(), nullable=False),
        sa.Column("reducer", sa.String(), nullable=False),
        sa.Column("storage_path", sa.String(), nullable=True),
        sa.Column("point_count", sa.Integer(), nullable=True),
        sa.Column("config", sa.JSON(), nullable=True),
        sa.Column("meta", sa.JSON(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("timezone('utc', now())"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("timezone('utc', now())"),
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["collection_id"], ["collections.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["operation_uuid"], ["operations.uuid"], ondelete="SET NULL"),
        sa.UniqueConstraint("uuid", name="uq_projection_runs_uuid"),
        sa.UniqueConstraint("operation_uuid", name="uq_projection_runs_operation_uuid"),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint("dimensionality > 0", name="ck_projection_runs_dimensionality_positive"),
        sa.CheckConstraint(
            "point_count IS NULL OR point_count >= 0",
            name="ck_projection_runs_point_count_non_negative",
        ),
    )

    op.create_index(
        "ix_projection_runs_collection_id", "projection_runs", ["collection_id"], unique=False
    )
    op.create_index("ix_projection_runs_status", "projection_runs", ["status"], unique=False)
    op.create_index("ix_projection_runs_created_at", "projection_runs", ["created_at"], unique=False)


def downgrade() -> None:
    """Drop projection_runs table and revert enum changes."""

    op.drop_index("ix_projection_runs_created_at", table_name="projection_runs")
    op.drop_index("ix_projection_runs_status", table_name="projection_runs")
    op.drop_index("ix_projection_runs_collection_id", table_name="projection_runs")
    op.drop_table("projection_runs")

    bind = op.get_bind()
    projection_run_status.drop(bind, checkfirst=True)

    op.execute("UPDATE operations SET type = 'index' WHERE type = 'projection_build'")
    op.execute("ALTER TYPE operation_type RENAME TO operation_type_old")
    op.execute(
        "CREATE TYPE operation_type AS ENUM ('index', 'append', 'reindex', 'remove_source', 'delete')"
    )
    op.execute(
        "ALTER TABLE operations ALTER COLUMN type TYPE operation_type USING type::text::operation_type"
    )
    op.execute("DROP TYPE operation_type_old")
