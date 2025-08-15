"""Add complete chunking infrastructure

Revision ID: add_complete_chunking
Revises: 20250727151108
Create Date: 2025-08-15 00:00:00.000000

This consolidated migration adds all chunking-related infrastructure in a single,
clean migration. It replaces multiple conflicting migrations with a unified approach.

Features:
- Chunking strategies and configs tables
- Chunks table with 100 LIST partitions
- All necessary indexes for performance
- Monitoring views and functions
- Collection and document column additions
- Default chunking strategies
"""

import logging
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy import text
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine import Connection

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "add_complete_chunking"
down_revision: str | Sequence[str] | None = "20250727151108"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

logger = logging.getLogger(__name__)


def table_exists(conn: Connection, table_name: str) -> bool:
    """Check if a table exists in the database."""
    result = conn.execute(
        text(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name = :table_name
            )
            """
        ).bindparams(table_name=table_name)
    )
    return bool(result.scalar())


def column_exists(conn: Connection, table_name: str, column_name: str) -> bool:
    """Check if a column exists in a table."""
    result = conn.execute(
        text(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.columns
                WHERE table_schema = 'public'
                AND table_name = :table_name
                AND column_name = :column_name
            )
            """
        ).bindparams(table_name=table_name, column_name=column_name)
    )
    return bool(result.scalar())


def index_exists(conn: Connection, index_name: str) -> bool:
    """Check if an index exists."""
    result = conn.execute(
        text(
            """
            SELECT EXISTS (
                SELECT FROM pg_indexes
                WHERE schemaname = 'public'
                AND indexname = :index_name
            )
            """
        ).bindparams(index_name=index_name)
    )
    return bool(result.scalar())


def view_exists(conn: Connection, view_name: str, is_materialized: bool = False) -> bool:
    """Check if a view exists."""
    if is_materialized:
        result = conn.execute(
            text(
                """
                SELECT EXISTS (
                    SELECT FROM pg_matviews
                    WHERE schemaname = 'public'
                    AND matviewname = :view_name
                )
                """
            ).bindparams(view_name=view_name)
        )
    else:
        result = conn.execute(
            text(
                """
                SELECT EXISTS (
                    SELECT FROM pg_views
                    WHERE schemaname = 'public'
                    AND viewname = :view_name
                )
                """
            ).bindparams(view_name=view_name)
        )
    return bool(result.scalar())


def function_exists(conn: Connection, function_name: str) -> bool:
    """Check if a function exists."""
    result = conn.execute(
        text(
            """
            SELECT EXISTS (
                SELECT FROM pg_proc
                WHERE proname = :function_name
            )
            """
        ).bindparams(function_name=function_name)
    )
    return bool(result.scalar())


def upgrade() -> None:
    """Add complete chunking infrastructure."""

    conn = op.get_bind()
    logger.info("Starting consolidated chunking infrastructure migration")

    # ========================================================================
    # STEP 1: Create chunking_strategies table
    # ========================================================================
    if not table_exists(conn, "chunking_strategies"):
        logger.info("Creating chunking_strategies table...")
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

        if not index_exists(conn, "ix_chunking_strategies_is_active"):
            op.create_index("ix_chunking_strategies_is_active", "chunking_strategies", ["is_active"])

    # ========================================================================
    # STEP 2: Create chunking_configs table
    # ========================================================================
    if not table_exists(conn, "chunking_configs"):
        logger.info("Creating chunking_configs table...")
        op.create_table(
            "chunking_configs",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("strategy_id", sa.Integer(), nullable=False),
            sa.Column("config_hash", sa.String(length=64), nullable=False),
            sa.Column("config_data", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
            sa.Column("use_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
            sa.ForeignKeyConstraint(["strategy_id"], ["chunking_strategies.id"]),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("config_hash"),
        )

        if not index_exists(conn, "ix_chunking_configs_strategy_id"):
            op.create_index("ix_chunking_configs_strategy_id", "chunking_configs", ["strategy_id"])
        if not index_exists(conn, "ix_chunking_configs_config_hash"):
            op.create_index("ix_chunking_configs_config_hash", "chunking_configs", ["config_hash"], unique=True)
        if not index_exists(conn, "ix_chunking_configs_use_count"):
            op.create_index("ix_chunking_configs_use_count", "chunking_configs", ["use_count"])

    # ========================================================================
    # STEP 3: Create chunks table with 100 LIST partitions
    # ========================================================================
    if not table_exists(conn, "chunks"):
        logger.info("Creating chunks table with 100 LIST partitions...")

        # Create the partitioned table
        conn.execute(
            text(
                """
                CREATE TABLE chunks (
                    id BIGSERIAL,
                    collection_id VARCHAR NOT NULL,
                    partition_key INTEGER NOT NULL,
                    document_id VARCHAR,
                    chunking_config_id INTEGER,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    start_offset INTEGER,
                    end_offset INTEGER,
                    token_count INTEGER,
                    embedding_vector_id VARCHAR,
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (id, collection_id, partition_key),
                    FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE,
                    FOREIGN KEY (chunking_config_id) REFERENCES chunking_configs(id)
                ) PARTITION BY LIST (partition_key)
                """
            )
        )

        # Create the partition key computation function
        conn.execute(
            text(
                """
                CREATE OR REPLACE FUNCTION compute_partition_key()
                RETURNS TRIGGER AS $$
                BEGIN
                    -- Use composite hash for better distribution
                    IF NEW.document_id IS NOT NULL THEN
                        NEW.partition_key := abs(hashtext(NEW.collection_id::text || ':' || NEW.document_id::text)) % 100;
                    ELSE
                        NEW.partition_key := abs(hashtext(
                            NEW.collection_id::text || ':' ||
                            COALESCE(NEW.chunk_index::text, NEW.id::text, 'null')
                        )) % 100;
                    END IF;
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql IMMUTABLE;
                """
            )
        )

        # Create the trigger
        conn.execute(
            text(
                """
                CREATE TRIGGER set_partition_key
                BEFORE INSERT ON chunks
                FOR EACH ROW
                EXECUTE FUNCTION compute_partition_key();
                """
            )
        )

        # Create 100 partitions with indexes
        logger.info("Creating 100 partitions...")
        for i in range(100):
            partition_name = f"chunks_part_{i:02d}"

            # Create partition
            conn.execute(
                text(
                    f"""
                    CREATE TABLE {partition_name} PARTITION OF chunks
                    FOR VALUES IN ({i})
                    """
                )
            )

            # Create indexes on each partition
            conn.execute(
                text(
                    f"""
                    CREATE INDEX idx_{partition_name}_collection
                    ON {partition_name}(collection_id);

                    CREATE INDEX idx_{partition_name}_document
                    ON {partition_name}(document_id)
                    WHERE document_id IS NOT NULL;

                    CREATE INDEX idx_{partition_name}_created
                    ON {partition_name}(created_at);

                    CREATE INDEX idx_{partition_name}_chunk_index
                    ON {partition_name}(collection_id, chunk_index);

                    -- Performance indexes
                    CREATE INDEX idx_{partition_name}_collection_document
                    ON {partition_name}(collection_id, document_id);

                    CREATE INDEX idx_{partition_name}_document_index
                    ON {partition_name}(document_id, chunk_index);

                    CREATE INDEX idx_{partition_name}_created_brin
                    ON {partition_name} USING brin(created_at);

                    CREATE INDEX idx_{partition_name}_no_embedding
                    ON {partition_name}(collection_id, id)
                    WHERE embedding_vector_id IS NULL;
                    """
                )
            )

    # ========================================================================
    # STEP 4: Add columns to collections table
    # ========================================================================
    logger.info("Adding columns to collections table...")

    if not column_exists(conn, "collections", "default_chunking_config_id"):
        op.add_column("collections", sa.Column("default_chunking_config_id", sa.Integer(), nullable=True))
        op.create_foreign_key(
            "fk_collections_default_chunking_config",
            "collections",
            "chunking_configs",
            ["default_chunking_config_id"],
            ["id"],
        )
        if not index_exists(conn, "ix_collections_default_chunking_config_id"):
            op.create_index("ix_collections_default_chunking_config_id", "collections", ["default_chunking_config_id"])

    if not column_exists(conn, "collections", "chunks_total_count"):
        op.add_column("collections", sa.Column("chunks_total_count", sa.Integer(), nullable=False, server_default="0"))

    if not column_exists(conn, "collections", "chunking_completed_at"):
        op.add_column("collections", sa.Column("chunking_completed_at", sa.DateTime(timezone=True), nullable=True))

    if not column_exists(conn, "collections", "chunking_strategy"):
        op.add_column("collections", sa.Column("chunking_strategy", sa.String(), nullable=True))

    if not column_exists(conn, "collections", "chunking_config"):
        op.add_column("collections", sa.Column("chunking_config", postgresql.JSON(astext_type=sa.Text()), nullable=True))

    # Add performance indexes for collections
    if not index_exists(conn, "idx_collections_owner_status"):
        op.create_index("idx_collections_owner_status", "collections", ["owner_id", "status"])

    # ========================================================================
    # STEP 5: Add columns to documents table
    # ========================================================================
    logger.info("Adding columns to documents table...")

    if not column_exists(conn, "documents", "chunking_config_id"):
        op.add_column("documents", sa.Column("chunking_config_id", sa.Integer(), nullable=True))
        op.create_foreign_key(
            "fk_documents_chunking_config",
            "documents",
            "chunking_configs",
            ["chunking_config_id"],
            ["id"]
        )
        if not index_exists(conn, "ix_documents_chunking_config_id"):
            op.create_index("ix_documents_chunking_config_id", "documents", ["chunking_config_id"])

    if not column_exists(conn, "documents", "chunks_count"):
        op.add_column("documents", sa.Column("chunks_count", sa.Integer(), nullable=False, server_default="0"))

    if not column_exists(conn, "documents", "chunking_started_at"):
        op.add_column("documents", sa.Column("chunking_started_at", sa.DateTime(timezone=True), nullable=True))

    if not column_exists(conn, "documents", "chunking_completed_at"):
        op.add_column("documents", sa.Column("chunking_completed_at", sa.DateTime(timezone=True), nullable=True))

    # Add performance indexes for documents
    if not index_exists(conn, "ix_documents_collection_id_chunking_completed_at"):
        op.create_index(
            "ix_documents_collection_id_chunking_completed_at",
            "documents",
            ["collection_id", "chunking_completed_at"]
        )

    if not index_exists(conn, "idx_documents_collection_status"):
        op.create_index("idx_documents_collection_status", "documents", ["collection_id", "status"])

    # ========================================================================
    # STEP 6: Add performance indexes to operations table
    # ========================================================================
    logger.info("Adding performance indexes to operations table...")

    if not index_exists(conn, "idx_operations_collection_type_status"):
        op.create_index(
            "idx_operations_collection_type_status",
            "operations",
            ["collection_id", "type", "status"]
        )

    if not index_exists(conn, "idx_operations_created_desc"):
        op.create_index(
            "idx_operations_created_desc",
            "operations",
            [sa.text("created_at DESC")]
        )

    if not index_exists(conn, "idx_operations_user_status"):
        op.create_index(
            "idx_operations_user_status",
            "operations",
            ["user_id", "status"],
            postgresql_where=sa.text("status IN ('processing', 'pending')")
        )

    if not index_exists(conn, "idx_operations_config_strategy"):
        op.create_index(
            "idx_operations_config_strategy",
            "operations",
            [sa.text("(config->>'strategy')")],
            postgresql_where=sa.text("config IS NOT NULL")
        )

    # ========================================================================
    # STEP 7: Create monitoring views and functions
    # ========================================================================
    logger.info("Creating monitoring views and functions...")

    # Helper functions
    if not function_exists(conn, "get_partition_for_collection"):
        conn.execute(
            text(
                """
                CREATE OR REPLACE FUNCTION get_partition_for_collection(collection_id VARCHAR)
                RETURNS TEXT AS $$
                BEGIN
                    RETURN 'chunks_part_' || LPAD((abs(hashtext(collection_id::text || ':' ||
                        COALESCE((SELECT document_id FROM chunks WHERE collection_id = $1 LIMIT 1), 'default'))) % 100)::text, 2, '0');
                END;
                $$ LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE;
                """
            )
        )

    if not function_exists(conn, "get_partition_key"):
        conn.execute(
            text(
                """
                CREATE OR REPLACE FUNCTION get_partition_key(collection_id VARCHAR)
                RETURNS INTEGER AS $$
                BEGIN
                    RETURN abs(hashtext(collection_id::text)) % 100;
                END;
                $$ LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE;
                """
            )
        )

    # Materialized view for collection statistics
    if not view_exists(conn, "collection_chunking_stats", is_materialized=True):
        conn.execute(
            text(
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
        )

        if not index_exists(conn, "ix_collection_chunking_stats_id"):
            op.create_index("ix_collection_chunking_stats_id", "collection_chunking_stats", ["id"], unique=True)

    # Refresh function for materialized view
    if not function_exists(conn, "refresh_collection_chunking_stats"):
        conn.execute(
            text(
                """
                CREATE OR REPLACE FUNCTION refresh_collection_chunking_stats()
                RETURNS void AS $$
                BEGIN
                    REFRESH MATERIALIZED VIEW CONCURRENTLY collection_chunking_stats;
                END;
                $$ LANGUAGE plpgsql;
                """
            )
        )

    # Partition health monitoring view
    if not view_exists(conn, "partition_health"):
        conn.execute(
            text(
                """
                CREATE OR REPLACE VIEW partition_health AS
                WITH partition_stats AS (
                    SELECT
                        schemaname,
                        relname as partition_name,
                        SUBSTRING(relname FROM 'chunks_part_([0-9]+)')::INT as partition_id,
                        pg_total_relation_size(schemaname||'.'||relname) as size_bytes,
                        n_live_tup as row_count,
                        n_dead_tup as dead_rows,
                        last_vacuum,
                        last_autovacuum
                    FROM pg_stat_user_tables
                    WHERE relname LIKE 'chunks_part_%'
                ),
                stats_summary AS (
                    SELECT
                        AVG(row_count) as avg_rows,
                        MAX(row_count) as max_rows,
                        MIN(row_count) as min_rows,
                        STDDEV(row_count) as stddev_rows
                    FROM partition_stats
                )
                SELECT
                    ps.*,
                    pg_size_pretty(ps.size_bytes) as size_pretty,
                    ROUND((ps.row_count::NUMERIC / NULLIF(ss.avg_rows, 0) - 1) * 100, 2) as pct_deviation_from_avg,
                    CASE
                        WHEN ss.avg_rows > 0 AND ps.row_count > ss.avg_rows * 1.2 THEN 'HOT'
                        WHEN ss.avg_rows > 0 AND ps.row_count < ss.avg_rows * 0.8 THEN 'COLD'
                        ELSE 'NORMAL'
                    END as partition_status,
                    ps.dead_rows > ps.row_count * 0.1 as needs_vacuum
                FROM partition_stats ps
                CROSS JOIN stats_summary ss
                ORDER BY partition_id;
                """
            )
        )

    # Partition distribution view
    if not view_exists(conn, "partition_distribution"):
        conn.execute(
            text(
                """
                CREATE OR REPLACE VIEW partition_distribution AS
                WITH partition_counts AS (
                    SELECT
                        partition_key,
                        COUNT(DISTINCT collection_id) as collection_count,
                        COUNT(*) as chunk_count
                    FROM chunks
                    GROUP BY partition_key
                ),
                distribution_stats AS (
                    SELECT
                        COUNT(*) as partitions_used,
                        AVG(chunk_count) as avg_chunks_per_partition,
                        STDDEV(chunk_count) as stddev_chunks,
                        MAX(chunk_count) as max_chunks,
                        MIN(chunk_count) as min_chunks,
                        MAX(chunk_count)::FLOAT / NULLIF(AVG(chunk_count), 0) as max_skew_ratio
                    FROM partition_counts
                )
                SELECT
                    ds.*,
                    CASE
                        WHEN max_skew_ratio > 1.2 THEN 'REBALANCE NEEDED'
                        WHEN max_skew_ratio > 1.1 THEN 'WARNING'
                        ELSE 'HEALTHY'
                    END as distribution_status,
                    100 - partitions_used as empty_partitions
                FROM distribution_stats ds;
                """
            )
        )

    # Active chunking configs view
    if not view_exists(conn, "active_chunking_configs"):
        conn.execute(
            text(
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
                    WHERE chunking_config_id IS NOT NULL
                    GROUP BY chunking_config_id
                ) sub ON cc.id = sub.chunking_config_id
                WHERE cc.use_count > 0;
                """
            )
        )

    # ========================================================================
    # STEP 8: Insert default chunking strategies
    # ========================================================================
    logger.info("Inserting default chunking strategies...")

    # Check if strategies already exist
    result = conn.execute(text("SELECT COUNT(*) FROM chunking_strategies"))
    if result.scalar() == 0:
        conn.execute(
            text(
                """
                INSERT INTO chunking_strategies (name, description, is_active, meta) VALUES
                ('character', 'Simple fixed-size character-based chunking using TokenTextSplitter', true, '{"supports_streaming": true}'),
                ('recursive', 'Smart sentence-aware splitting using SentenceSplitter', true, '{"supports_streaming": true, "recommended_default": true}'),
                ('markdown', 'Respects markdown structure using MarkdownNodeParser', true, '{"supports_streaming": true, "file_types": [".md", ".mdx"]}'),
                ('semantic', 'Uses AI embeddings to find natural boundaries using SemanticSplitterNodeParser', false, '{"supports_streaming": false, "requires_embeddings": true}'),
                ('hierarchical', 'Creates parent-child chunks using HierarchicalNodeParser', false, '{"supports_streaming": false}'),
                ('hybrid', 'Automatically selects strategy based on content', false, '{"supports_streaming": false}')
                ON CONFLICT (name) DO NOTHING;
                """
            )
        )

    # ========================================================================
    # STEP 9: Update existing collections with chunking strategies
    # ========================================================================
    logger.info("Updating existing collections with chunking strategies...")

    # Set strategy for collections with custom chunk settings
    conn.execute(
        text(
            """
            UPDATE collections
            SET chunking_strategy = 'character'
            WHERE chunking_strategy IS NULL
            AND chunk_size IS NOT NULL
            AND chunk_overlap IS NOT NULL
            AND (chunk_size != 1000 OR chunk_overlap != 200)
            """
        )
    )

    # Set default strategy for remaining collections
    conn.execute(
        text(
            """
            UPDATE collections
            SET chunking_strategy = 'recursive'
            WHERE chunking_strategy IS NULL
            """
        )
    )

    # Update chunking_config column
    conn.execute(
        text(
            """
            UPDATE collections
            SET chunking_config = jsonb_build_object(
                'chunk_size', COALESCE(chunk_size, 1000),
                'chunk_overlap', COALESCE(chunk_overlap, 200)
            )
            WHERE chunking_config IS NULL
            AND chunk_size IS NOT NULL
            AND chunk_overlap IS NOT NULL
            """
        )
    )

    logger.info("Consolidated chunking infrastructure migration completed successfully!")


def downgrade() -> None:
    """Remove all chunking infrastructure."""

    conn = op.get_bind()
    logger.info("Starting downgrade of chunking infrastructure")

    # Drop views
    conn.execute(text("DROP VIEW IF EXISTS active_chunking_configs CASCADE"))
    conn.execute(text("DROP VIEW IF EXISTS partition_distribution CASCADE"))
    conn.execute(text("DROP VIEW IF EXISTS partition_health CASCADE"))
    conn.execute(text("DROP MATERIALIZED VIEW IF EXISTS collection_chunking_stats CASCADE"))

    # Drop functions
    conn.execute(text("DROP FUNCTION IF EXISTS refresh_collection_chunking_stats() CASCADE"))
    conn.execute(text("DROP FUNCTION IF EXISTS get_partition_key(VARCHAR) CASCADE"))
    conn.execute(text("DROP FUNCTION IF EXISTS get_partition_for_collection(VARCHAR) CASCADE"))
    conn.execute(text("DROP FUNCTION IF EXISTS compute_partition_key() CASCADE"))

    # Drop indexes on operations
    try:
        op.drop_index("idx_operations_config_strategy", table_name="operations")
        op.drop_index("idx_operations_user_status", table_name="operations")
        op.drop_index("idx_operations_created_desc", table_name="operations")
        op.drop_index("idx_operations_collection_type_status", table_name="operations")
    except Exception:
        pass  # Indexes might not exist

    # Remove columns from documents
    try:
        op.drop_index("idx_documents_collection_status", table_name="documents")
        op.drop_index("ix_documents_collection_id_chunking_completed_at", table_name="documents")
        op.drop_index("ix_documents_chunking_config_id", table_name="documents")
        op.drop_constraint("fk_documents_chunking_config", "documents", type_="foreignkey")
        op.drop_column("documents", "chunking_completed_at")
        op.drop_column("documents", "chunking_started_at")
        op.drop_column("documents", "chunks_count")
        op.drop_column("documents", "chunking_config_id")
    except Exception:
        pass  # Columns might not exist

    # Remove columns from collections
    try:
        op.drop_index("idx_collections_owner_status", table_name="collections")
        op.drop_column("collections", "chunking_config")
        op.drop_column("collections", "chunking_strategy")
        op.drop_index("ix_collections_default_chunking_config_id", table_name="collections")
        op.drop_constraint("fk_collections_default_chunking_config", "collections", type_="foreignkey")
        op.drop_column("collections", "chunking_completed_at")
        op.drop_column("collections", "chunks_total_count")
        op.drop_column("collections", "default_chunking_config_id")
    except Exception:
        pass  # Columns might not exist

    # Drop chunks table and partitions
    conn.execute(text("DROP TABLE IF EXISTS chunks CASCADE"))

    # Drop chunking_configs table
    if table_exists(conn, "chunking_configs"):
        op.drop_table("chunking_configs")

    # Drop chunking_strategies table
    if table_exists(conn, "chunking_strategies"):
        op.drop_table("chunking_strategies")

    logger.info("Chunking infrastructure downgrade completed")
