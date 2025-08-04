"""add_chunking_tables_with_partitioning

Revision ID: 52db15bd2686
Revises: 20250727151108
Create Date: 2025-08-04 15:34:52.426728

"""

import os
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "52db15bd2686"
down_revision: str | Sequence[str] | None = "20250727151108"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add chunking tables with partitioning for scalable document processing."""

    # Get partition count from environment variable, default to 16
    partition_count = int(os.environ.get("CHUNK_PARTITION_COUNT", "16"))

    # Step 1: Create chunking_strategies table
    op.create_table(
        "chunking_strategies",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("version", sa.String(), nullable=False, server_default="1.0.0"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("meta", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_index(op.f("ix_chunking_strategies_is_active"), "chunking_strategies", ["is_active"], unique=False)

    # Step 2: Create chunking_configs table (deduplicated)
    op.create_table(
        "chunking_configs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("strategy_id", sa.Integer(), nullable=False),
        sa.Column("config_hash", sa.String(length=64), nullable=False),
        sa.Column("config_data", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("use_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["strategy_id"],
            ["chunking_strategies.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("config_hash"),
    )
    op.create_index(op.f("ix_chunking_configs_strategy_id"), "chunking_configs", ["strategy_id"], unique=False)
    op.create_index(op.f("ix_chunking_configs_config_hash"), "chunking_configs", ["config_hash"], unique=True)
    op.create_index(op.f("ix_chunking_configs_use_count"), "chunking_configs", ["use_count"], unique=False)

    # Step 3: Create chunks table with partitioning
    # Create the parent table
    op.execute(
        """
        CREATE TABLE chunks (
            id UUID DEFAULT gen_random_uuid() NOT NULL,
            collection_id VARCHAR NOT NULL,
            document_id VARCHAR NOT NULL,
            chunking_config_id INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            start_offset INTEGER NOT NULL,
            end_offset INTEGER NOT NULL,
            token_count INTEGER,
            embedding_vector_id VARCHAR,
            created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
            meta JSON,
            PRIMARY KEY (id, collection_id),
            FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE,
            FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
            FOREIGN KEY (chunking_config_id) REFERENCES chunking_configs(id)
        ) PARTITION BY HASH (collection_id);
    """
    )

    # Create partitions based on configuration
    for i in range(partition_count):
        op.execute(
            f"""
            CREATE TABLE chunks_p{i} PARTITION OF chunks
            FOR VALUES WITH (MODULUS {partition_count}, REMAINDER {i});
        """
        )

    # Create indexes on the parent table (will be inherited by partitions)
    op.create_index("ix_chunks_collection_id_document_id", "chunks", ["collection_id", "document_id"], unique=False)
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"], unique=False)
    op.create_index("ix_chunks_chunking_config_id", "chunks", ["chunking_config_id"], unique=False)
    op.create_index("ix_chunks_collection_id_chunk_index", "chunks", ["collection_id", "chunk_index"], unique=False)
    op.create_index("ix_chunks_created_at", "chunks", ["created_at"], unique=False)

    # Step 4: Add columns to existing tables
    # Update collections table
    op.add_column("collections", sa.Column("default_chunking_config_id", sa.Integer(), nullable=True))
    op.add_column("collections", sa.Column("chunks_total_count", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("collections", sa.Column("chunking_completed_at", sa.DateTime(timezone=True), nullable=True))

    op.create_foreign_key(
        "fk_collections_default_chunking_config",
        "collections",
        "chunking_configs",
        ["default_chunking_config_id"],
        ["id"],
    )
    op.create_index(
        "ix_collections_default_chunking_config_id", "collections", ["default_chunking_config_id"], unique=False
    )

    # Update documents table
    op.add_column("documents", sa.Column("chunking_config_id", sa.Integer(), nullable=True))
    op.add_column("documents", sa.Column("chunks_count", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("documents", sa.Column("chunking_started_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("documents", sa.Column("chunking_completed_at", sa.DateTime(timezone=True), nullable=True))

    op.create_foreign_key(
        "fk_documents_chunking_config", "documents", "chunking_configs", ["chunking_config_id"], ["id"]
    )
    op.create_index("ix_documents_chunking_config_id", "documents", ["chunking_config_id"], unique=False)
    op.create_index(
        "ix_documents_collection_id_chunking_completed_at",
        "documents",
        ["collection_id", "chunking_completed_at"],
        unique=False,
    )

    # Step 5: Create materialized view for collection statistics
    op.execute(
        """
        CREATE MATERIALIZED VIEW collection_chunking_stats AS
        SELECT
            c.id,
            c.name,
            COUNT(DISTINCT ch.document_id) as chunked_documents,
            COUNT(ch.id) as total_chunks,
            AVG(ch.token_count)::NUMERIC(10,2) as avg_tokens_per_chunk,
            MAX(ch.created_at) as last_chunk_created
        FROM collections c
        LEFT JOIN chunks ch ON c.id = ch.collection_id
        GROUP BY c.id, c.name
        WITH DATA;
    """
    )

    # Create index on materialized view
    op.create_index("ix_collection_chunking_stats_id", "collection_chunking_stats", ["id"], unique=True)

    # Create refresh function for materialized view
    op.execute(
        """
        CREATE OR REPLACE FUNCTION refresh_collection_chunking_stats()
        RETURNS void AS $$
        BEGIN
            REFRESH MATERIALIZED VIEW CONCURRENTLY collection_chunking_stats;
        END;
        $$ LANGUAGE plpgsql;
    """
    )

    # Step 6: Create regular view for active chunking configs
    op.execute(
        """
        CREATE VIEW active_chunking_configs AS
        SELECT
            cc.id,
            cc.strategy_id,
            cc.config_hash,
            cc.config_data,
            cc.created_at,
            cc.use_count,
            cc.last_used_at,
            cs.name as strategy_name,
            sub.collections_using
        FROM chunking_configs cc
        JOIN chunking_strategies cs ON cc.strategy_id = cs.id
        LEFT JOIN (
            SELECTchunking_config_id, COUNT(DISTINCT collection_id) as collections_using
            FROM chunks
            GROUP BY chunking_config_id
        ) sub ON cc.id = sub.chunking_config_id
        WHERE cc.use_count > 0;
    """
    )

    # Step 7: Insert default chunking strategies
    op.execute(
        """
        INSERT INTO chunking_strategies (name, description, is_active, meta) VALUES
        ('character', 'Simple fixed-size character-based chunking using TokenTextSplitter', true, '{"supports_streaming": true}'),
        ('recursive', 'Smart sentence-aware splitting using SentenceSplitter', true, '{"supports_streaming": true, "recommended_default": true}'),
        ('markdown', 'Respects markdown structure using MarkdownNodeParser', true, '{"supports_streaming": true, "file_types": [".md", ".mdx"]}'),
        ('semantic', 'Uses AI embeddings to find natural boundaries using SemanticSplitterNodeParser', false, '{"supports_streaming": false, "requires_embeddings": true}'),
        ('hierarchical', 'Creates parent-child chunks using HierarchicalNodeParser', false, '{"supports_streaming": false}'),
        ('hybrid', 'Automatically selects strategy based on content', false, '{"supports_streaming": false}');
    """
    )


def downgrade() -> None:
    """Rollback chunking tables and related changes."""

    # Drop views first
    op.execute("DROP VIEW IF EXISTS active_chunking_configs")
    op.execute("DROP FUNCTION IF EXISTS refresh_collection_chunking_stats()")
    op.execute("DROP MATERIALIZED VIEW IF EXISTS collection_chunking_stats")

    # Remove columns from documents table
    op.drop_index("ix_documents_collection_id_chunking_completed_at", table_name="documents")
    op.drop_index("ix_documents_chunking_config_id", table_name="documents")
    op.drop_constraint("fk_documents_chunking_config", "documents", type_="foreignkey")
    op.drop_column("documents", "chunking_completed_at")
    op.drop_column("documents", "chunking_started_at")
    op.drop_column("documents", "chunks_count")
    op.drop_column("documents", "chunking_config_id")

    # Remove columns from collections table
    op.drop_index("ix_collections_default_chunking_config_id", table_name="collections")
    op.drop_constraint("fk_collections_default_chunking_config", "collections", type_="foreignkey")
    op.drop_column("collections", "chunking_completed_at")
    op.drop_column("collections", "chunks_total_count")
    op.drop_column("collections", "default_chunking_config_id")

    # Drop chunks table and all partitions
    op.drop_index("ix_chunks_created_at", table_name="chunks")
    op.drop_index("ix_chunks_collection_id_chunk_index", table_name="chunks")
    op.drop_index("ix_chunks_chunking_config_id", table_name="chunks")
    op.drop_index("ix_chunks_document_id", table_name="chunks")
    op.drop_index("ix_chunks_collection_id_document_id", table_name="chunks")

    # Drop all partition tables (PostgreSQL will drop them with the parent)
    op.execute("DROP TABLE chunks CASCADE")

    # Drop chunking_configs table
    op.drop_index(op.f("ix_chunking_configs_use_count"), table_name="chunking_configs")
    op.drop_index(op.f("ix_chunking_configs_config_hash"), table_name="chunking_configs")
    op.drop_index(op.f("ix_chunking_configs_strategy_id"), table_name="chunking_configs")
    op.drop_table("chunking_configs")

    # Drop chunking_strategies table
    op.drop_index(op.f("ix_chunking_strategies_is_active"), table_name="chunking_strategies")
    op.drop_table("chunking_strategies")
