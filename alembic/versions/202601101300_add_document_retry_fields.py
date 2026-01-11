"""Add document retry tracking fields.

This migration adds fields to the documents table to track retry attempts
for failed documents, enabling automatic retry with backoff and manual retry.

Changes to documents:
- retry_count: number of retry attempts (default 0)
- last_retry_at: timestamp of last retry attempt
- error_category: classification of error ('transient', 'permanent', 'unknown')

Indexes:
- ix_documents_collection_failed_retryable: partial index for querying
  retryable failed documents efficiently

Revision ID: 202601101300
Revises: 202601101200
Create Date: 2026-01-10 13:00:00.000000

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa

from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence


revision: str = "202601101300"
down_revision: str | Sequence[str] | None = "202601101200"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add document retry tracking fields."""

    # Add retry_count column (tracks number of retry attempts)
    op.add_column(
        "documents",
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
    )

    # Add last_retry_at column (tracks when last retry was attempted)
    op.add_column(
        "documents",
        sa.Column("last_retry_at", sa.DateTime(timezone=True), nullable=True),
    )

    # Add error_category column (classifies error for retry decisions)
    op.add_column(
        "documents",
        sa.Column("error_category", sa.String(50), nullable=True),
    )

    # Create partial index for efficiently querying retryable failed documents
    # This helps identify documents that can be retried (transient/unknown errors)
    status_literal = "FAILED"
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        result = bind.execute(
            sa.text(
                """
                SELECT enumlabel
                FROM pg_enum
                JOIN pg_type ON pg_enum.enumtypid = pg_type.oid
                WHERE pg_type.typname = 'document_status'
                """
            )
        )
        labels = {row[0] for row in result}
        if "failed" in labels:
            status_literal = "failed"
        elif "FAILED" in labels:
            status_literal = "FAILED"
        else:
            raise RuntimeError("document_status enum missing FAILED/failed label")

    op.create_index(
        "ix_documents_collection_failed_retryable",
        "documents",
        ["collection_id", "status", "error_category", "retry_count"],
        postgresql_where=sa.text(f"status = '{status_literal}'"),
    )


def downgrade() -> None:
    """Remove document retry tracking fields."""

    # Drop the partial index
    op.drop_index("ix_documents_collection_failed_retryable", table_name="documents")

    # Drop columns
    op.drop_column("documents", "error_category")
    op.drop_column("documents", "last_retry_at")
    op.drop_column("documents", "retry_count")
