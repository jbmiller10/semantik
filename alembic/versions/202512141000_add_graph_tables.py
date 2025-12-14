"""Add Entity and Relationship tables for GraphRAG.

This migration adds the infrastructure for GraphRAG features:

1. graph_enabled column on collections table:
   - Boolean flag to enable/disable graph features per collection
   - Defaults to false for existing collections

2. entities table (partitioned with 100 LIST partitions):
   - Stores named entities extracted from document chunks
   - Partitioned by partition_key (0-99) for query performance
   - Supports entity deduplication via name_hash and canonical_id

3. relationships table (partitioned with 100 LIST partitions):
   - Stores relationships between entities
   - Links source and target entities with typed relationships
   - Same partitioning scheme as entities for consistent performance

Partitioning Strategy:
- partition_key = abs(hash(collection_id)) % 100
- Matches the existing Chunks table partitioning pattern
- Enables efficient partition pruning when collection_id is in WHERE clause

Revision ID: 202512141000
Revises: 202512140100
Create Date: 2025-12-14 10:00:00.000000

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa
from sqlalchemy import text

from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence


revision: str = "202512141000"
down_revision: str | Sequence[str] | None = "202512140100"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add graph tables with partitioning."""
    conn = op.get_bind()

    # =========================================================================
    # 1. Add graph_enabled column to collections table
    # =========================================================================

    op.add_column(
        "collections",
        sa.Column(
            "graph_enabled",
            sa.Boolean(),
            nullable=False,
            server_default="false",
        ),
    )

    # =========================================================================
    # 2. Create entities table (partitioned by LIST on partition_key)
    # =========================================================================

    conn.execute(
        text(
            """
            CREATE TABLE entities (
                id BIGSERIAL,
                collection_id VARCHAR NOT NULL,
                partition_key INTEGER NOT NULL DEFAULT 0,
                document_id VARCHAR NOT NULL,
                chunk_id BIGINT,
                name VARCHAR(500) NOT NULL,
                name_normalized VARCHAR(500) NOT NULL,
                name_hash VARCHAR(64) NOT NULL,
                entity_type VARCHAR(100) NOT NULL,
                canonical_id BIGINT,
                start_offset INTEGER,
                end_offset INTEGER,
                confidence FLOAT DEFAULT 0.85,
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                PRIMARY KEY (id, collection_id, partition_key)
            ) PARTITION BY LIST (partition_key)
            """
        )
    )

    # Create 100 partitions for entities (entities_p0 to entities_p99)
    for i in range(100):
        conn.execute(
            text(f"CREATE TABLE entities_p{i} PARTITION OF entities FOR VALUES IN ({i})")
        )

    # =========================================================================
    # 3. Create relationships table (partitioned by LIST on partition_key)
    # =========================================================================

    conn.execute(
        text(
            """
            CREATE TABLE relationships (
                id BIGSERIAL,
                collection_id VARCHAR NOT NULL,
                partition_key INTEGER NOT NULL DEFAULT 0,
                source_entity_id BIGINT NOT NULL,
                target_entity_id BIGINT NOT NULL,
                relationship_type VARCHAR(200) NOT NULL,
                confidence FLOAT DEFAULT 0.7,
                extraction_method VARCHAR(50),
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                PRIMARY KEY (id, collection_id, partition_key)
            ) PARTITION BY LIST (partition_key)
            """
        )
    )

    # Create 100 partitions for relationships (relationships_p0 to relationships_p99)
    for i in range(100):
        conn.execute(
            text(f"CREATE TABLE relationships_p{i} PARTITION OF relationships FOR VALUES IN ({i})")
        )

    # =========================================================================
    # 4. Create indexes for entities table
    # =========================================================================

    op.create_index("idx_entities_collection_id", "entities", ["collection_id"])
    op.create_index("idx_entities_document_id", "entities", ["document_id"])
    op.create_index("idx_entities_entity_type", "entities", ["entity_type"])
    op.create_index("idx_entities_name_hash", "entities", ["name_hash"])
    op.create_index("idx_entities_canonical_id", "entities", ["canonical_id"])
    op.create_index("idx_entities_chunk_id", "entities", ["chunk_id"])

    # =========================================================================
    # 5. Create indexes for relationships table
    # =========================================================================

    op.create_index("idx_relationships_collection_id", "relationships", ["collection_id"])
    op.create_index("idx_relationships_source", "relationships", ["source_entity_id"])
    op.create_index("idx_relationships_target", "relationships", ["target_entity_id"])
    op.create_index("idx_relationships_type", "relationships", ["relationship_type"])


def downgrade() -> None:
    """Remove graph tables and column."""
    conn = op.get_bind()

    # =========================================================================
    # 1. Drop indexes for relationships table
    # =========================================================================

    op.drop_index("idx_relationships_type", table_name="relationships")
    op.drop_index("idx_relationships_target", table_name="relationships")
    op.drop_index("idx_relationships_source", table_name="relationships")
    op.drop_index("idx_relationships_collection_id", table_name="relationships")

    # =========================================================================
    # 2. Drop indexes for entities table
    # =========================================================================

    op.drop_index("idx_entities_chunk_id", table_name="entities")
    op.drop_index("idx_entities_canonical_id", table_name="entities")
    op.drop_index("idx_entities_name_hash", table_name="entities")
    op.drop_index("idx_entities_entity_type", table_name="entities")
    op.drop_index("idx_entities_document_id", table_name="entities")
    op.drop_index("idx_entities_collection_id", table_name="entities")

    # =========================================================================
    # 3. Drop partitioned tables (CASCADE drops all partitions automatically)
    # =========================================================================

    conn.execute(text("DROP TABLE IF EXISTS relationships CASCADE"))
    conn.execute(text("DROP TABLE IF EXISTS entities CASCADE"))

    # =========================================================================
    # 4. Remove graph_enabled column from collections table
    # =========================================================================

    op.drop_column("collections", "graph_enabled")
