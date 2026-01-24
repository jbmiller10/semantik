"""Add Pipeline DAG support for Phase 1b.

Revision ID: 202601231000
Revises: 202601191100
Create Date: 2026-01-23

This migration adds database support for the Pipeline DAG abstraction:
- New columns on Collection: pipeline_config, pipeline_version, persist_originals
- New column on Document: pipeline_version
- New pipeline_failures table for tracking files that fail during pipeline execution

IMPORTANT: This migration deletes existing data from chunks, documents, operations,
and collections tables. This is acceptable because this is a pre-release version
with no production users.
"""

from collections.abc import Sequence

from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text, text
from sqlalchemy.dialects.postgresql import JSON

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "202601231000"
down_revision: str | None = "202601191100"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add Pipeline DAG support columns and tables."""
    # Clean slate - delete all existing data (pre-release, no users)
    # Order matters due to foreign key constraints - delete dependent tables first
    op.execute("DELETE FROM benchmark_query_results")
    op.execute("DELETE FROM benchmark_run_metrics")
    op.execute("DELETE FROM benchmark_runs")
    op.execute("DELETE FROM benchmarks")
    op.execute("DELETE FROM benchmark_relevance")
    op.execute("DELETE FROM benchmark_queries")
    op.execute("DELETE FROM benchmark_dataset_mappings")
    op.execute("DELETE FROM benchmark_datasets")
    op.execute("DELETE FROM chunks")
    op.execute("DELETE FROM document_artifacts")
    op.execute("DELETE FROM documents")
    op.execute("DELETE FROM collection_audit_log")
    op.execute("DELETE FROM operation_metrics")
    op.execute("DELETE FROM projection_runs")
    op.execute("DELETE FROM operations")
    op.execute("DELETE FROM collection_sync_runs")
    op.execute("DELETE FROM connector_secrets")
    op.execute("DELETE FROM collection_sources")
    op.execute("DELETE FROM collection_permissions")
    op.execute("DELETE FROM collection_resource_limits")
    op.execute("DELETE FROM mcp_profile_collections")
    op.execute("DELETE FROM collections")

    # Add new columns to collections
    op.add_column(
        "collections",
        Column("pipeline_config", JSON, nullable=True),
    )
    op.add_column(
        "collections",
        Column("pipeline_version", Integer, nullable=False, server_default="1"),
    )
    op.add_column(
        "collections",
        Column("persist_originals", Boolean, nullable=False, server_default="false"),
    )

    # Add new column to documents
    op.add_column(
        "documents",
        Column("pipeline_version", Integer, nullable=True),
    )

    # Create pipeline_failures table
    op.create_table(
        "pipeline_failures",
        Column("id", String, primary_key=True),
        Column(
            "collection_id",
            String,
            nullable=False,
            index=True,
        ),
        Column(
            "operation_id",
            String,
            nullable=True,
            index=True,
        ),
        Column("file_uri", String, nullable=False),
        Column("file_metadata", JSON, nullable=True),
        Column("stage_id", String, nullable=False),
        Column("stage_type", String, nullable=False),
        Column("error_type", String, nullable=False),
        Column("error_message", Text, nullable=False),
        Column("error_traceback", Text, nullable=True),
        Column("retry_count", Integer, nullable=False, server_default="0"),
        Column("last_retry_at", DateTime(timezone=True), nullable=True),
        Column(
            "created_at",
            DateTime(timezone=True),
            nullable=False,
            server_default=text("NOW()"),
        ),
    )

    # Add foreign key constraints
    op.create_foreign_key(
        "fk_pipeline_failures_collection_id",
        "pipeline_failures",
        "collections",
        ["collection_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "fk_pipeline_failures_operation_id",
        "pipeline_failures",
        "operations",
        ["operation_id"],
        ["uuid"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    """Remove Pipeline DAG support columns and tables."""
    # Drop foreign keys first
    op.drop_constraint(
        "fk_pipeline_failures_operation_id",
        "pipeline_failures",
        type_="foreignkey",
    )
    op.drop_constraint(
        "fk_pipeline_failures_collection_id",
        "pipeline_failures",
        type_="foreignkey",
    )

    # Drop pipeline_failures table
    op.drop_table("pipeline_failures")

    # Remove columns from documents
    op.drop_column("documents", "pipeline_version")

    # Remove columns from collections
    op.drop_column("collections", "persist_originals")
    op.drop_column("collections", "pipeline_version")
    op.drop_column("collections", "pipeline_config")
