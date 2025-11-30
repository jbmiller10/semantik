"""Add flexible sources support for connectors.

This migration adds support for arbitrary source types (web, Slack, etc.)
by extending the collection_sources and documents tables with new columns.

Changes:
- collection_sources: Add source_config (JSON), remove source_type default
- documents: Add uri (String), source_metadata (JSON), unique index on (collection_id, uri)

Data migrations:
- Backfill source_config = {"path": source_path} for existing sources
- Backfill uri = file_path for existing documents

Revision ID: 202511291200
Revises: 202511201200
Create Date: 2025-11-29 12:00:00.000000

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa

from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence


revision: str = "202511291200"
down_revision: str | Sequence[str] | None = "202511201200"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add flexible sources support columns and backfill existing data."""

    # --- collection_sources table ---

    # Add source_config column (JSON, nullable)
    op.add_column(
        "collection_sources",
        sa.Column("source_config", sa.JSON(), nullable=True),
    )

    # Remove default from source_type column
    # This allows application code to control the default rather than the DB
    op.execute("ALTER TABLE collection_sources ALTER COLUMN source_type DROP DEFAULT")

    # Backfill source_config for existing rows
    # Creates {"path": source_path} for all existing directory sources
    op.execute("""
        UPDATE collection_sources
        SET source_config = jsonb_build_object('path', source_path)::json
        WHERE source_config IS NULL
    """)

    # --- documents table ---

    # Add uri column (String, nullable)
    # This is the logical identifier for the document (URL, Slack message ID, file path, etc.)
    op.add_column(
        "documents",
        sa.Column("uri", sa.String(), nullable=True),
    )

    # Add source_metadata column (JSON, nullable)
    # Stores connector-specific metadata (raw response, headers, Slack thread info, etc.)
    op.add_column(
        "documents",
        sa.Column("source_metadata", sa.JSON(), nullable=True),
    )

    # Backfill uri from file_path for existing documents
    # This ensures all existing documents remain addressable via uri
    op.execute("UPDATE documents SET uri = file_path WHERE uri IS NULL")

    # Create partial unique index on (collection_id, uri) WHERE uri IS NOT NULL
    # This enforces one document per URI per collection while allowing NULL URIs
    op.create_index(
        "ix_documents_collection_uri_unique",
        "documents",
        ["collection_id", "uri"],
        unique=True,
        postgresql_where=sa.text("uri IS NOT NULL"),
    )


def downgrade() -> None:
    """Remove flexible sources support columns."""

    # --- documents table (reverse order) ---

    # Drop the unique index first
    op.drop_index("ix_documents_collection_uri_unique", table_name="documents")

    # Drop source_metadata column
    op.drop_column("documents", "source_metadata")

    # Drop uri column
    op.drop_column("documents", "uri")

    # --- collection_sources table ---

    # Restore default on source_type for backward compatibility
    op.execute("ALTER TABLE collection_sources ALTER COLUMN source_type SET DEFAULT 'directory'")

    # Drop source_config column
    op.drop_column("collection_sources", "source_config")
