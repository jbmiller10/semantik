"""Fix partition distribution strategy to spread chunks within collections

Revision ID: db005_partition_distribution
Revises: db004_add_chunking_indexes
Create Date: 2025-08-13 00:00:00.000000

CRITICAL FIX: The current partition strategy puts ALL chunks from a collection
into the SAME partition, defeating the purpose of partitioning. This migration
implements a composite hash strategy using both collection_id and document_id
to properly distribute chunks across partitions.

Before: partition_key = hash(collection_id) % 100
After:  partition_key = hash(collection_id || ':' || document_id) % 100

This ensures:
- Chunks from large collections are distributed across multiple partitions
- All chunks from the same document stay in the same partition (cache locality)
- Even distribution across all 100 partitions
- Improved query performance for large collections
"""

import contextlib
import logging
from collections.abc import Sequence

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from alembic import op

# revision identifiers, used by Alembic
revision: str = "db005_partition_distribution"
down_revision: str | None = "db004_add_chunking_indexes"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_chunks_table(conn) -> tuple[bool, int]:
    """Validate chunks table exists and get record count.

    Returns:
        Tuple of (table_exists, record_count)
    """
    try:
        # Check if chunks table exists
        result = conn.execute(
            text("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_name = 'chunks'
                    AND table_schema = 'public'
                )
            """)
        )
        table_exists = result.scalar()

        if not table_exists:
            logger.warning("Chunks table does not exist")
            return False, 0

        # Get record count
        result = conn.execute(text("SELECT COUNT(*) FROM chunks"))
        record_count = result.scalar() or 0

        logger.info(f"Chunks table exists with {record_count} records")
        return True, record_count

    except SQLAlchemyError as e:
        logger.error(f"Error validating chunks table: {e}")
        raise


def drop_monitoring_views(conn) -> None:
    """Drop monitoring views that depend on chunks table."""
    views_to_drop = [
        "partition_distribution",
        "partition_health",
        "partition_size_distribution",
        "partition_chunk_distribution",
        "partition_hot_spots",
        "partition_health_summary",
        "active_chunking_configs",
        "collection_chunking_stats"  # This is a materialized view
    ]

    for view in views_to_drop:
        with contextlib.suppress(SQLAlchemyError):
            # Try as regular view first
            conn.execute(text(f"DROP VIEW IF EXISTS {view} CASCADE"))

        with contextlib.suppress(SQLAlchemyError):
            # Try as materialized view
            conn.execute(text(f"DROP MATERIALIZED VIEW IF EXISTS {view} CASCADE"))

    logger.info("Dropped monitoring views")


def drop_partition_functions(conn) -> None:
    """Drop functions related to partition management."""
    functions_to_drop = [
        ("analyze_partition_skew", ""),
        ("get_partition_key", "VARCHAR"),
        ("get_partition_for_collection", "VARCHAR"),
        ("refresh_collection_chunking_stats", ""),
        ("validate_partition_distribution", "")  # New function we'll add
    ]

    for func_name, params in functions_to_drop:
        try:
            if params:
                conn.execute(text(f"DROP FUNCTION IF EXISTS {func_name}({params}) CASCADE"))
            else:
                conn.execute(text(f"DROP FUNCTION IF EXISTS {func_name}() CASCADE"))
        except SQLAlchemyError:
            pass

    logger.info("Dropped partition functions")


def create_new_partition_function(conn) -> None:
    """Create the new composite hash partition function."""
    # Drop old trigger and function
    conn.execute(text("DROP TRIGGER IF EXISTS set_partition_key ON chunks CASCADE"))
    conn.execute(text("DROP FUNCTION IF EXISTS compute_partition_key() CASCADE"))

    # Create new function with composite hash
    conn.execute(text("""
        CREATE OR REPLACE FUNCTION compute_partition_key()
        RETURNS TRIGGER AS $$
        BEGIN
            -- Use composite hash of collection_id and document_id for better distribution
            -- This ensures chunks from the same document stay together while
            -- distributing chunks from large collections across partitions

            IF NEW.document_id IS NOT NULL THEN
                -- Standard case: use collection_id + document_id composite hash
                NEW.partition_key := abs(hashtext(NEW.collection_id::text || ':' || NEW.document_id::text)) % 100;
            ELSE
                -- Fallback for chunks without document_id (should be rare)
                -- Use collection_id + chunk id/index for distribution
                NEW.partition_key := abs(hashtext(
                    NEW.collection_id::text || ':' ||
                    COALESCE(NEW.chunk_index::text, NEW.id::text, 'null')
                )) % 100;
            END IF;

            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql IMMUTABLE;
    """))

    # Create trigger to compute partition_key on insert
    conn.execute(text("""
        CREATE TRIGGER set_partition_key
        BEFORE INSERT ON chunks
        FOR EACH ROW
        EXECUTE FUNCTION compute_partition_key();
    """))

    logger.info("Created new composite hash partition function")


def migrate_existing_data(conn, source_count: int) -> bool:
    """Migrate existing chunks to new partition distribution.

    Args:
        conn: Database connection
        source_count: Number of records in source table

    Returns:
        True if migration successful, False otherwise
    """
    if source_count == 0:
        logger.info("No data to migrate")
        return True

    try:
        # Create new table with same structure
        conn.execute(text("""
            CREATE TABLE chunks_new (LIKE chunks INCLUDING ALL)
            PARTITION BY LIST (partition_key)
        """))

        # Create 100 partitions for new table
        for i in range(100):
            partition_name = f"chunks_new_part_{i:02d}"
            conn.execute(text(f"""
                CREATE TABLE {partition_name} PARTITION OF chunks_new
                FOR VALUES IN ({i})
            """))

        logger.info("Created new partitioned table structure")

        # Copy data with recalculated partition keys
        conn.execute(text("""
            INSERT INTO chunks_new (
                id, collection_id, chunk_index, content, metadata,
                document_id, chunking_config_id, start_offset, end_offset,
                token_count, embedding_vector_id, created_at, updated_at,
                partition_key
            )
            SELECT
                id, collection_id, chunk_index, content, metadata,
                document_id, chunking_config_id, start_offset, end_offset,
                token_count, embedding_vector_id, created_at, updated_at,
                CASE
                    WHEN document_id IS NOT NULL THEN
                        abs(hashtext(collection_id::text || ':' || document_id::text)) % 100
                    ELSE
                        abs(hashtext(
                            collection_id::text || ':' ||
                            COALESCE(chunk_index::text, id::text, 'null')
                        )) % 100
                END as partition_key
            FROM chunks
        """))

        # Verify record count
        result = conn.execute(text("SELECT COUNT(*) FROM chunks_new"))
        new_count = result.scalar() or 0

        if new_count != source_count:
            logger.error(f"Data migration failed: source={source_count}, new={new_count}")
            conn.execute(text("DROP TABLE chunks_new CASCADE"))
            return False

        logger.info(f"Successfully migrated {new_count} records")

        # Swap tables atomically
        conn.execute(text("ALTER TABLE chunks RENAME TO chunks_old"))
        conn.execute(text("ALTER TABLE chunks_new RENAME TO chunks"))

        # Rename partitions
        for i in range(100):
            old_name = f"chunks_new_part_{i:02d}"
            new_name = f"chunks_part_{i:02d}"
            conn.execute(text(f"ALTER TABLE {old_name} RENAME TO {new_name}"))

        # Drop old table
        conn.execute(text("DROP TABLE chunks_old CASCADE"))

        logger.info("Table swap completed successfully")
        return True

    except SQLAlchemyError as e:
        logger.error(f"Migration failed: {e}")
        # Cleanup on failure
        with contextlib.suppress(Exception):
            conn.execute(text("DROP TABLE IF EXISTS chunks_new CASCADE"))
        raise


def create_monitoring_views(conn) -> None:
    """Create updated monitoring views for new partition strategy."""

    # View to show partition distribution
    conn.execute(text("""
        CREATE OR REPLACE VIEW partition_distribution AS
        WITH partition_stats AS (
            SELECT
                partition_key,
                COUNT(*) as chunk_count,
                COUNT(DISTINCT collection_id) as collection_count,
                COUNT(DISTINCT document_id) as document_count,
                COUNT(DISTINCT collection_id || ':' || document_id) as collection_document_pairs
            FROM chunks
            GROUP BY partition_key
        ),
        overall_stats AS (
            SELECT
                AVG(chunk_count) as avg_chunks,
                STDDEV(chunk_count) as stddev_chunks,
                MAX(chunk_count) as max_chunks,
                MIN(chunk_count) as min_chunks,
                COUNT(*) as partitions_used
            FROM partition_stats
        )
        SELECT
            ps.partition_key,
            ps.chunk_count,
            ps.collection_count,
            ps.document_count,
            ps.collection_document_pairs,
            ROUND((ps.chunk_count::NUMERIC / NULLIF(SUM(ps.chunk_count) OVER (), 0)) * 100, 2) as chunk_percentage,
            ROUND((ps.chunk_count::NUMERIC / NULLIF(os.avg_chunks, 0) - 1) * 100, 2) as deviation_from_avg,
            CASE
                WHEN os.avg_chunks > 0 AND ps.chunk_count > os.avg_chunks * 1.5 THEN 'HOT'
                WHEN os.avg_chunks > 0 AND ps.chunk_count < os.avg_chunks * 0.5 THEN 'COLD'
                ELSE 'NORMAL'
            END as partition_status
        FROM partition_stats ps
        CROSS JOIN overall_stats os
        ORDER BY ps.partition_key;
    """))

    # View to analyze collection spread across partitions
    conn.execute(text("""
        CREATE OR REPLACE VIEW partition_collection_distribution AS
        WITH collection_spread AS (
            SELECT
                collection_id,
                COUNT(DISTINCT partition_key) as partitions_used,
                COUNT(*) as total_chunks,
                MIN(partition_key) as min_partition,
                MAX(partition_key) as max_partition,
                ARRAY_AGG(DISTINCT partition_key ORDER BY partition_key) as partition_list
            FROM chunks
            GROUP BY collection_id
        )
        SELECT
            collection_id,
            total_chunks,
            partitions_used,
            ROUND(partitions_used::NUMERIC / GREATEST(total_chunks / 1000, 1), 2) as partition_efficiency,
            min_partition,
            max_partition,
            CASE
                WHEN total_chunks > 10000 AND partitions_used < 10 THEN 'POOR DISTRIBUTION'
                WHEN total_chunks > 1000 AND partitions_used < 5 THEN 'SUBOPTIMAL'
                ELSE 'GOOD'
            END as distribution_quality,
            partition_list
        FROM collection_spread
        ORDER BY total_chunks DESC;
    """))

    # Recreate helper functions
    conn.execute(text("""
        CREATE OR REPLACE FUNCTION get_partition_key(p_collection_id VARCHAR, p_document_id VARCHAR)
        RETURNS INTEGER AS $$
        BEGIN
            IF p_document_id IS NOT NULL THEN
                RETURN abs(hashtext(p_collection_id::text || ':' || p_document_id::text)) % 100;
            ELSE
                RETURN abs(hashtext(p_collection_id::text || ':' || 'null')) % 100;
            END IF;
        END;
        $$ LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE;
    """))

    # Function to validate distribution
    conn.execute(text("""
        CREATE OR REPLACE FUNCTION validate_partition_distribution()
        RETURNS TABLE(
            metric TEXT,
            value NUMERIC,
            status TEXT,
            details TEXT
        ) AS $$
        DECLARE
            v_max_skew NUMERIC;
            v_empty_partitions INT;
            v_poor_collections INT;
            v_total_chunks BIGINT;
        BEGIN
            -- Calculate max skew
            SELECT
                MAX(chunk_percentage) / NULLIF(AVG(chunk_percentage), 0)
            INTO v_max_skew
            FROM partition_distribution;

            -- Count empty partitions
            SELECT 100 - COUNT(DISTINCT partition_key)
            INTO v_empty_partitions
            FROM chunks;

            -- Count collections with poor distribution
            SELECT COUNT(*)
            INTO v_poor_collections
            FROM partition_collection_distribution
            WHERE distribution_quality IN ('POOR DISTRIBUTION', 'SUBOPTIMAL');

            -- Total chunks
            SELECT COUNT(*) INTO v_total_chunks FROM chunks;

            -- Return metrics
            RETURN QUERY
            SELECT 'Max Skew Ratio', v_max_skew,
                   CASE WHEN v_max_skew > 2 THEN 'CRITICAL'
                        WHEN v_max_skew > 1.5 THEN 'WARNING'
                        ELSE 'HEALTHY' END,
                   'Maximum partition size relative to average'
            UNION ALL
            SELECT 'Empty Partitions', v_empty_partitions::NUMERIC,
                   CASE WHEN v_empty_partitions > 50 THEN 'WARNING'
                        ELSE 'HEALTHY' END,
                   'Number of unused partitions'
            UNION ALL
            SELECT 'Poor Distribution Collections', v_poor_collections::NUMERIC,
                   CASE WHEN v_poor_collections > 0 THEN 'WARNING'
                        ELSE 'HEALTHY' END,
                   'Collections not well distributed across partitions'
            UNION ALL
            SELECT 'Total Chunks', v_total_chunks::NUMERIC, 'INFO',
                   'Total number of chunks in system';
        END;
        $$ LANGUAGE plpgsql;
    """))

    logger.info("Created monitoring views and functions")


def upgrade() -> None:
    """Apply migration to fix partition distribution strategy."""
    conn = op.get_bind()

    logger.info("Starting partition distribution fix migration")

    # Step 1: Validate current state
    table_exists, record_count = validate_chunks_table(conn)
    if not table_exists:
        logger.error("Chunks table does not exist, cannot proceed")
        return

    # Step 2: Drop dependent objects
    drop_monitoring_views(conn)
    drop_partition_functions(conn)

    # Step 3: Create new partition function
    create_new_partition_function(conn)

    # Step 4: Migrate existing data if any
    if record_count > 0:
        logger.info(f"Migrating {record_count} existing chunks to new partition distribution")
        success = migrate_existing_data(conn, record_count)
        if not success:
            raise RuntimeError("Data migration failed")

    # Step 5: Create monitoring views
    create_monitoring_views(conn)

    # Step 6: Analyze distribution
    if record_count > 0:
        result = conn.execute(text("SELECT * FROM validate_partition_distribution()"))
        logger.info("Distribution validation results:")
        for row in result:
            logger.info(f"  {row[0]}: {row[1]} - {row[2]} ({row[3]})")

    logger.info("Partition distribution fix completed successfully")


def downgrade() -> None:
    """Revert to original partition strategy (NOT RECOMMENDED)."""
    conn = op.get_bind()

    logger.warning("Reverting partition distribution fix - this may cause performance issues!")

    # Drop new monitoring views
    drop_monitoring_views(conn)
    drop_partition_functions(conn)

    # Restore original partition function
    conn.execute(text("DROP TRIGGER IF EXISTS set_partition_key ON chunks CASCADE"))
    conn.execute(text("DROP FUNCTION IF EXISTS compute_partition_key() CASCADE"))

    conn.execute(text("""
        CREATE OR REPLACE FUNCTION compute_partition_key()
        RETURNS TRIGGER AS $$
        BEGIN
            -- Original (problematic) partition strategy
            NEW.partition_key := abs(hashtext(NEW.collection_id::text)) % 100;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """))

    conn.execute(text("""
        CREATE TRIGGER set_partition_key
        BEFORE INSERT ON chunks
        FOR EACH ROW
        EXECUTE FUNCTION compute_partition_key();
    """))

    logger.info("Reverted to original partition strategy")

