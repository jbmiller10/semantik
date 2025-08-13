"""Add indexes for chunking performance optimization

Revision ID: db004_add_chunking_indexes
Revises: db003_replace_trigger
Create Date: 2025-08-11 14:30:00.000000

This migration adds composite and specialized indexes to optimize query performance
for chunking operations, eliminating N+1 queries and full table scans.

Indexes added:
- Composite indexes for common query patterns
- JSONB indexes for metadata queries
- BRIN indexes for time-series data
- Partition-specific indexes for chunks table
"""

import logging
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "db004_add_chunking_indexes"
down_revision: str | None = "db003_replace_trigger"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

logger = logging.getLogger(__name__)


def upgrade() -> None:
    """Add performance indexes for chunking operations."""

    logger.info("Adding composite indexes for operations table...")

    # Operations table indexes
    op.create_index(
        "idx_operations_collection_type_status",
        "operations",
        ["collection_id", "type", "status"],
        postgresql_using="btree",
    )

    op.create_index("idx_operations_created_desc", "operations", [sa.text("created_at DESC")], postgresql_using="btree")

    op.create_index(
        "idx_operations_user_status",
        "operations",
        ["user_id", "status"],
        postgresql_where=sa.text("status IN ('PROCESSING', 'PENDING')"),
        postgresql_using="btree",
    )

    # JSONB index for config queries
    op.create_index(
        "idx_operations_config_strategy",
        "operations",
        [sa.text("(config->>'strategy')")],
        postgresql_using="btree",
        postgresql_where=sa.text("config IS NOT NULL"),
    )

    logger.info("Adding indexes for collections table...")

    # Collections table indexes
    op.create_index("idx_collections_owner_status", "collections", ["owner_id", "status"], postgresql_using="btree")

    logger.info("Adding indexes for documents table...")

    # Documents table indexes
    op.create_index(
        "idx_documents_collection_status", "documents", ["collection_id", "status"], postgresql_using="btree"
    )

    logger.info("Adding partition-specific indexes for chunks table...")

    # Chunks table indexes (per partition)
    # We need to add indexes to each partition
    # Note: Partitions are named with zero-padding (00, 01, ... 99)
    # First check which partitions exist to avoid transaction abort
    conn = op.get_bind()
    result = conn.execute(
        sa.text(
            """
        SELECT tablename
        FROM pg_tables
        WHERE schemaname = 'public'
        AND tablename LIKE 'chunks_part_%'
    """
        )
    )
    existing_partitions = {row[0] for row in result}

    for i in range(100):
        partition_name = f"chunks_part_{i:02d}"

        # Skip if partition doesn't exist
        if partition_name not in existing_partitions:
            logger.info(f"Skipping partition {partition_name} - does not exist")
            continue

        try:
            # Collection + document index for common queries
            op.create_index(
                f"idx_{partition_name}_collection_document",
                partition_name,
                ["collection_id", "document_id"],
                postgresql_using="btree",
            )

            # Document + index for ordered chunk retrieval
            op.create_index(
                f"idx_{partition_name}_document_index",
                partition_name,
                ["document_id", "chunk_index"],
                postgresql_using="btree",
            )

            # BRIN index for created_at (efficient for time-series data)
            op.create_index(
                f"idx_{partition_name}_created_brin", partition_name, ["created_at"], postgresql_using="brin"
            )

            # Index for chunks without embeddings
            op.create_index(
                f"idx_{partition_name}_no_embedding",
                partition_name,
                ["collection_id", "id"],
                postgresql_where=sa.text("embedding_vector_id IS NULL"),
                postgresql_using="btree",
            )

        except Exception as e:
            logger.warning(f"Could not create indexes for partition {partition_name}: {e}")
            # Continue with other partitions even if one fails
            continue

    logger.info("Running ANALYZE to update table statistics...")

    # Analyze tables to update statistics for query planner
    op.execute("ANALYZE operations")
    op.execute("ANALYZE collections")
    op.execute("ANALYZE documents")

    # Analyze all chunk partitions that exist
    for i in range(100):
        partition_name = f"chunks_part_{i:02d}"
        if partition_name in existing_partitions:
            try:
                # Use PL/pgSQL with format for safe identifier quoting
                conn.execute(
                    sa.text(
                        "DO $$ BEGIN EXECUTE format('ANALYZE %I', :table_name); END $$;"
                    ).bindparams(table_name=partition_name)
                )
            except Exception as e:
                logger.warning(f"Could not analyze partition {partition_name}: {e}")
                continue

    logger.info("Index creation completed successfully")


def downgrade() -> None:
    """Remove performance indexes."""

    logger.info("Dropping operations table indexes...")

    # Drop operations table indexes
    op.drop_index("idx_operations_collection_type_status", table_name="operations")
    op.drop_index("idx_operations_created_desc", table_name="operations")
    op.drop_index("idx_operations_user_status", table_name="operations")
    op.drop_index("idx_operations_config_strategy", table_name="operations")

    logger.info("Dropping collections table indexes...")

    # Drop collections table indexes
    op.drop_index("idx_collections_owner_status", table_name="collections")

    logger.info("Dropping documents table indexes...")

    # Drop documents table indexes
    op.drop_index("idx_documents_collection_status", table_name="documents")

    logger.info("Dropping partition-specific indexes...")

    # Check which partitions exist before trying to drop indexes
    conn = op.get_bind()
    result = conn.execute(
        sa.text(
            """
        SELECT tablename
        FROM pg_tables
        WHERE schemaname = 'public'
        AND tablename LIKE 'chunks_part_%'
    """
        )
    )
    existing_partitions = {row[0] for row in result}

    # Drop chunk partition indexes
    for i in range(100):
        partition_name = f"chunks_part_{i:02d}"

        if partition_name not in existing_partitions:
            continue

        try:
            op.drop_index(f"idx_{partition_name}_collection_document", table_name=partition_name)
            op.drop_index(f"idx_{partition_name}_document_index", table_name=partition_name)
            op.drop_index(f"idx_{partition_name}_created_brin", table_name=partition_name)
            op.drop_index(f"idx_{partition_name}_no_embedding", table_name=partition_name)
        except Exception as e:
            logger.warning(f"Could not drop indexes for partition {partition_name}: {e}")
            continue

    logger.info("Index removal completed")
