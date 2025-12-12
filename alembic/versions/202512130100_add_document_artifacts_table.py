"""Add document_artifacts table for non-file document content storage.

This migration adds a table to store canonical document content in the database
for non-file sources (Git, IMAP, web). This enables DocumentViewer to display
content from any source type, not just local files.

The content endpoint will check for artifacts first, then fall back to file
serving for backward compatibility.

Changes:
- Add document_artifacts table with:
  - document_id, collection_id (FKs with cascade delete)
  - artifact_kind ('primary', 'preview', 'thumbnail')
  - mime_type, charset for content type handling
  - content_text (TEXT) or content_bytes (BYTEA) for storage
  - content_hash for deduplication
  - size_bytes, is_truncated for size tracking
  - timestamps

Indexes:
- ix_document_artifacts_document: Fast lookup by document
- ix_document_artifacts_collection: Collection-level queries

Constraints:
- uq_document_artifact_kind: One artifact per kind per document
- ck_content_present: Either text or bytes must be set
- ck_artifact_kind_values: Restrict kind to known values

Revision ID: 202512130100
Revises: 202512121201
Create Date: 2025-12-13 01:00:00.000000

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa

from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence


revision: str = "202512130100"
down_revision: str | Sequence[str] | None = "202512121201"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add document_artifacts table."""

    op.create_table(
        "document_artifacts",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "document_id",
            sa.String(),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "collection_id",
            sa.String(),
            sa.ForeignKey("collections.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "artifact_kind",
            sa.String(20),
            nullable=False,
            server_default="primary",
        ),
        sa.Column("mime_type", sa.String(255), nullable=False),
        sa.Column("charset", sa.String(50), nullable=True),
        sa.Column("content_text", sa.Text(), nullable=True),
        sa.Column("content_bytes", sa.LargeBinary(), nullable=True),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column(
            "is_truncated",
            sa.Boolean(),
            nullable=False,
            server_default="false",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    # Create indexes
    op.create_index(
        "ix_document_artifacts_document",
        "document_artifacts",
        ["document_id"],
    )
    op.create_index(
        "ix_document_artifacts_collection",
        "document_artifacts",
        ["collection_id"],
    )

    # Create unique constraint (one artifact per kind per document)
    op.create_unique_constraint(
        "uq_document_artifact_kind",
        "document_artifacts",
        ["document_id", "artifact_kind"],
    )

    # Create check constraints
    op.execute(
        """
        ALTER TABLE document_artifacts
        ADD CONSTRAINT ck_content_present
        CHECK (content_text IS NOT NULL OR content_bytes IS NOT NULL)
        """
    )
    op.execute(
        """
        ALTER TABLE document_artifacts
        ADD CONSTRAINT ck_artifact_kind_values
        CHECK (artifact_kind IN ('primary', 'preview', 'thumbnail'))
        """
    )


def downgrade() -> None:
    """Remove document_artifacts table."""

    op.drop_table("document_artifacts")
