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


def table_exists(inspector, table_name: str) -> bool:
    """Check if a table exists in the database."""
    return table_name in inspector.get_table_names()


def index_exists(inspector, table_name: str, index_name: str) -> bool:
    """Check if an index exists on a table."""
    try:
        indexes = inspector.get_indexes(table_name)
        return any(idx['name'] == index_name for idx in indexes)
    except Exception:
        return False


def foreign_key_exists(inspector, table_name: str, fk_name: str) -> bool:
    """Check if a foreign key exists on a table."""
    try:
        fks = inspector.get_foreign_keys(table_name)
        return any(fk['name'] == fk_name for fk in fks)
    except Exception:
        return False


def column_exists(inspector, table_name: str, column_name: str) -> bool:
    """Check if a column exists in a table."""
    try:
        columns = inspector.get_columns(table_name)
        return any(col['name'] == column_name for col in columns)
    except Exception:
        return False


def upgrade() -> None:
    """Add chunking tables with partitioning for scalable document processing."""

    # Get partition count from environment variable, default to 16
    partition_count = int(os.environ.get("CHUNK_PARTITION_COUNT", "16"))
    
    # Check if tables already exist
    from sqlalchemy import inspect
    conn = op.get_bind()
    inspector = inspect(conn)
    existing_tables = inspector.get_table_names()

    # Step 1: Create chunking_strategies table
    if not table_exists(inspector, "chunking_strategies"):
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
    
    # Create indexes for chunking_strategies table
    if not index_exists(inspector, "chunking_strategies", "ix_chunking_strategies_is_active"):
        op.create_index(op.f("ix_chunking_strategies_is_active"), "chunking_strategies", ["is_active"], unique=False)

    # Step 2: Create chunking_configs table (deduplicated)
    if not table_exists(inspector, "chunking_configs"):
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
    
    # Create indexes for chunking_configs table
    if not index_exists(inspector, "chunking_configs", "ix_chunking_configs_strategy_id"):
        op.create_index(op.f("ix_chunking_configs_strategy_id"), "chunking_configs", ["strategy_id"], unique=False)
    if not index_exists(inspector, "chunking_configs", "ix_chunking_configs_config_hash"):
        op.create_index(op.f("ix_chunking_configs_config_hash"), "chunking_configs", ["config_hash"], unique=True)
    if not index_exists(inspector, "chunking_configs", "ix_chunking_configs_use_count"):
        op.create_index(op.f("ix_chunking_configs_use_count"), "chunking_configs", ["use_count"], unique=False)

    # Step 3: Create chunks table with partitioning
    # Create the parent table
    if not table_exists(inspector, "chunks"):
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
            # Check if partition already exists
            if not table_exists(inspector, f"chunks_p{i}"):
                op.execute(
                    f"""
                    CREATE TABLE chunks_p{i} PARTITION OF chunks
                    FOR VALUES WITH (MODULUS {partition_count}, REMAINDER {i});
                """
                )

    # Create indexes on the parent table (will be inherited by partitions)
    if not index_exists(inspector, "chunks", "ix_chunks_collection_id_document_id"):
        op.create_index("ix_chunks_collection_id_document_id", "chunks", ["collection_id", "document_id"], unique=False)
    if not index_exists(inspector, "chunks", "ix_chunks_document_id"):
        op.create_index("ix_chunks_document_id", "chunks", ["document_id"], unique=False)
    if not index_exists(inspector, "chunks", "ix_chunks_chunking_config_id"):
        op.create_index("ix_chunks_chunking_config_id", "chunks", ["chunking_config_id"], unique=False)
    if not index_exists(inspector, "chunks", "ix_chunks_collection_id_chunk_index"):
        op.create_index("ix_chunks_collection_id_chunk_index", "chunks", ["collection_id", "chunk_index"], unique=False)
    if not index_exists(inspector, "chunks", "ix_chunks_created_at"):
        op.create_index("ix_chunks_created_at", "chunks", ["created_at"], unique=False)

    # Step 4: Add columns to existing tables
    # Update collections table
    if not column_exists(inspector, "collections", "default_chunking_config_id"):
        op.add_column("collections", sa.Column("default_chunking_config_id", sa.Integer(), nullable=True))
    if not column_exists(inspector, "collections", "chunks_total_count"):
        op.add_column("collections", sa.Column("chunks_total_count", sa.Integer(), nullable=False, server_default="0"))
    if not column_exists(inspector, "collections", "chunking_completed_at"):
        op.add_column("collections", sa.Column("chunking_completed_at", sa.DateTime(timezone=True), nullable=True))

    # Create foreign key and index for collections table
    if not foreign_key_exists(inspector, "collections", "fk_collections_default_chunking_config"):
        op.create_foreign_key(
            "fk_collections_default_chunking_config",
            "collections",
            "chunking_configs",
            ["default_chunking_config_id"],
            ["id"],
        )
    if not index_exists(inspector, "collections", "ix_collections_default_chunking_config_id"):
        op.create_index(
            "ix_collections_default_chunking_config_id", "collections", ["default_chunking_config_id"], unique=False
        )

    # Update documents table
    if not column_exists(inspector, "documents", "chunking_config_id"):
        op.add_column("documents", sa.Column("chunking_config_id", sa.Integer(), nullable=True))
    if not column_exists(inspector, "documents", "chunks_count"):
        op.add_column("documents", sa.Column("chunks_count", sa.Integer(), nullable=False, server_default="0"))
    if not column_exists(inspector, "documents", "chunking_started_at"):
        op.add_column("documents", sa.Column("chunking_started_at", sa.DateTime(timezone=True), nullable=True))
    if not column_exists(inspector, "documents", "chunking_completed_at"):
        op.add_column("documents", sa.Column("chunking_completed_at", sa.DateTime(timezone=True), nullable=True))

    # Create foreign key and indexes for documents table
    if not foreign_key_exists(inspector, "documents", "fk_documents_chunking_config"):
        op.create_foreign_key(
            "fk_documents_chunking_config", "documents", "chunking_configs", ["chunking_config_id"], ["id"]
        )
    if not index_exists(inspector, "documents", "ix_documents_chunking_config_id"):
        op.create_index("ix_documents_chunking_config_id", "documents", ["chunking_config_id"], unique=False)
    if not index_exists(inspector, "documents", "ix_documents_collection_id_chunking_completed_at"):
        op.create_index(
            "ix_documents_collection_id_chunking_completed_at",
            "documents",
            ["collection_id", "chunking_completed_at"],
            unique=False,
        )

    # Get existing views and materialized views
    existing_matviews = []
    existing_views = []
    try:
        result = conn.execute(sa.text("SELECT matviewname FROM pg_matviews WHERE schemaname = 'public'"))
        existing_matviews = [row[0] for row in result]
        result = conn.execute(sa.text("SELECT viewname FROM pg_views WHERE schemaname = 'public'"))
        existing_views = [row[0] for row in result]
    except Exception:
        pass  # If query fails, assume views don't exist

    # Step 5: Create materialized view for collection statistics
    if "collection_chunking_stats" not in existing_matviews:
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

    # Create index on materialized view (check if it exists first)
    if not index_exists(inspector, "collection_chunking_stats", "ix_collection_chunking_stats_id"):
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
    if "active_chunking_configs" not in existing_views:
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
            SELECT chunking_config_id, COUNT(DISTINCT collection_id) as collections_using
            FROM chunks
            GROUP BY chunking_config_id
        ) sub ON cc.id = sub.chunking_config_id
            WHERE cc.use_count > 0;
        """
        )

    # Step 7: Default chunking strategies are now handled in a separate data migration
    # See migration: a1b2c3d4e5f6_add_default_chunking_strategies.py


def downgrade() -> None:
    """Rollback chunking tables and related changes."""
    
    # Get inspector for checking existence
    from sqlalchemy import inspect
    conn = op.get_bind()
    inspector = inspect(conn)

    # Drop views first
    op.execute("DROP VIEW IF EXISTS active_chunking_configs")
    op.execute("DROP FUNCTION IF EXISTS refresh_collection_chunking_stats()")
    op.execute("DROP MATERIALIZED VIEW IF EXISTS collection_chunking_stats")

    # Remove columns from documents table
    if index_exists(inspector, "documents", "ix_documents_collection_id_chunking_completed_at"):
        op.drop_index("ix_documents_collection_id_chunking_completed_at", table_name="documents")
    if index_exists(inspector, "documents", "ix_documents_chunking_config_id"):
        op.drop_index("ix_documents_chunking_config_id", table_name="documents")
    if foreign_key_exists(inspector, "documents", "fk_documents_chunking_config"):
        op.drop_constraint("fk_documents_chunking_config", "documents", type_="foreignkey")
    if column_exists(inspector, "documents", "chunking_completed_at"):
        op.drop_column("documents", "chunking_completed_at")
    if column_exists(inspector, "documents", "chunking_started_at"):
        op.drop_column("documents", "chunking_started_at")
    if column_exists(inspector, "documents", "chunks_count"):
        op.drop_column("documents", "chunks_count")
    if column_exists(inspector, "documents", "chunking_config_id"):
        op.drop_column("documents", "chunking_config_id")

    # Remove columns from collections table
    if index_exists(inspector, "collections", "ix_collections_default_chunking_config_id"):
        op.drop_index("ix_collections_default_chunking_config_id", table_name="collections")
    if foreign_key_exists(inspector, "collections", "fk_collections_default_chunking_config"):
        op.drop_constraint("fk_collections_default_chunking_config", "collections", type_="foreignkey")
    if column_exists(inspector, "collections", "chunking_completed_at"):
        op.drop_column("collections", "chunking_completed_at")
    if column_exists(inspector, "collections", "chunks_total_count"):
        op.drop_column("collections", "chunks_total_count")
    if column_exists(inspector, "collections", "default_chunking_config_id"):
        op.drop_column("collections", "default_chunking_config_id")

    # Drop chunks table and all partitions
    if table_exists(inspector, "chunks"):
        if index_exists(inspector, "chunks", "ix_chunks_created_at"):
            op.drop_index("ix_chunks_created_at", table_name="chunks")
        if index_exists(inspector, "chunks", "ix_chunks_collection_id_chunk_index"):
            op.drop_index("ix_chunks_collection_id_chunk_index", table_name="chunks")
        if index_exists(inspector, "chunks", "ix_chunks_chunking_config_id"):
            op.drop_index("ix_chunks_chunking_config_id", table_name="chunks")
        if index_exists(inspector, "chunks", "ix_chunks_document_id"):
            op.drop_index("ix_chunks_document_id", table_name="chunks")
        if index_exists(inspector, "chunks", "ix_chunks_collection_id_document_id"):
            op.drop_index("ix_chunks_collection_id_document_id", table_name="chunks")
        
        # Drop all partition tables (PostgreSQL will drop them with the parent)
        op.execute("DROP TABLE chunks CASCADE")

    # Drop chunking_configs table
    if table_exists(inspector, "chunking_configs"):
        if index_exists(inspector, "chunking_configs", "ix_chunking_configs_use_count"):
            op.drop_index(op.f("ix_chunking_configs_use_count"), table_name="chunking_configs")
        if index_exists(inspector, "chunking_configs", "ix_chunking_configs_config_hash"):
            op.drop_index(op.f("ix_chunking_configs_config_hash"), table_name="chunking_configs")
        if index_exists(inspector, "chunking_configs", "ix_chunking_configs_strategy_id"):
            op.drop_index(op.f("ix_chunking_configs_strategy_id"), table_name="chunking_configs")
        op.drop_table("chunking_configs")

    # Drop chunking_strategies table
    if table_exists(inspector, "chunking_strategies"):
        if index_exists(inspector, "chunking_strategies", "ix_chunking_strategies_is_active"):
            op.drop_index(op.f("ix_chunking_strategies_is_active"), table_name="chunking_strategies")
        op.drop_table("chunking_strategies")
