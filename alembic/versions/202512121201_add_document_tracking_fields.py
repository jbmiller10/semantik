"""Add document tracking fields for sync support.

This migration adds fields to the documents table to track document
freshness for sync operations with "keep last-known" behavior.

Changes to documents:
- last_seen_at: timestamp of when document was last seen during a sync
- is_stale: boolean flag indicating document was not seen in recent sync

Indexes:
- ix_documents_stale: partial index for querying stale documents

Revision ID: 202512121201
Revises: 202512121200
Create Date: 2025-12-12 12:01:00.000000

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa

from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence


revision: str = "202512121201"
down_revision: str | Sequence[str] | None = "202512121200"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add document tracking fields."""

    # Add last_seen_at column (tracks when document was last seen during sync)
    op.add_column(
        "documents",
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=True),
    )

    # Add is_stale column (marks documents not seen in recent sync)
    op.add_column(
        "documents",
        sa.Column("is_stale", sa.Boolean(), nullable=False, server_default="false"),
    )

    # Create partial index for efficiently querying stale documents
    # This helps identify documents that may have been deleted from source
    op.create_index(
        "ix_documents_stale",
        "documents",
        ["collection_id", "is_stale"],
        postgresql_where=sa.text("is_stale = TRUE"),
    )


def downgrade() -> None:
    """Remove document tracking fields."""

    # Drop the partial index
    op.drop_index("ix_documents_stale", table_name="documents")

    # Drop columns
    op.drop_column("documents", "is_stale")
    op.drop_column("documents", "last_seen_at")
