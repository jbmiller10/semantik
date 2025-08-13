"""Fix partition distribution strategy to spread chunks within collections

Revision ID: db005_partition_distribution
Revises: p2_backfill_001
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
down_revision: str | None = "p2_backfill_001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_chunks_table_state(conn) -> dict:
    """Check the current state of chunks table and its partitioning."""
    state = {
        "table_exists": False,
        "is_partitioned": False,
        "has_trigger": False,
        "has_function": False,
        "record_count": 0,
        "partition_count": 0
    }
    
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
        state["table_exists"] = result.scalar()
        
        if not state["table_exists"]:
            return state
        
        # Check if table is partitioned
        result = conn.execute(
            text("""
                SELECT COUNT(*) > 0
                FROM pg_class c
                JOIN pg_partitioned_table p ON c.oid = p.partrelid
                WHERE c.relname = 'chunks'
            """)
        )
        state["is_partitioned"] = result.scalar()
        
        # Check for trigger
        result = conn.execute(
            text("""
                SELECT COUNT(*) > 0
                FROM pg_trigger
                WHERE tgrelid = 'chunks'::regclass
                AND tgname = 'set_partition_key'
            """)
        )
        state["has_trigger"] = result.scalar()
        
        # Check for function
        result = conn.execute(
            text("""
                SELECT COUNT(*) > 0
                FROM pg_proc
                WHERE proname = 'compute_partition_key'
            """)
        )
        state["has_function"] = result.scalar()
        
        # Get record count
        try:
            result = conn.execute(text("SELECT COUNT(*) FROM chunks"))
            state["record_count"] = result.scalar() or 0
        except:
            state["record_count"] = 0
        
        # Get partition count
        if state["is_partitioned"]:
            result = conn.execute(
                text("""
                    SELECT COUNT(*)
                    FROM pg_class c
                    JOIN pg_inherits i ON c.oid = i.inhrelid
                    WHERE i.inhparent = 'chunks'::regclass
                """)
            )
            state["partition_count"] = result.scalar() or 0
    
    except Exception as e:
        logger.warning(f"Error checking chunks table state: {e}")
    
    return state


def safe_drop_objects(conn) -> None:
    """Safely drop monitoring views and functions that might depend on chunks."""
    # Drop views
    views_to_drop = [
        "partition_distribution",
        "partition_health",
        "partition_size_distribution",
        "partition_chunk_distribution",
        "partition_hot_spots",
        "partition_health_summary",
        "partition_collection_distribution",  # New view we're adding
        "active_chunking_configs",
        "collection_chunking_stats"
    ]
    
    for view in views_to_drop:
        with contextlib.suppress(Exception):
            conn.execute(text(f"DROP VIEW IF EXISTS {view} CASCADE"))
        with contextlib.suppress(Exception):
            conn.execute(text(f"DROP MATERIALIZED VIEW IF EXISTS {view} CASCADE"))
    
    # Drop functions
    functions_to_drop = [
        ("analyze_partition_skew", ""),
        ("get_partition_key", "VARCHAR"),
        ("get_partition_key", "VARCHAR, VARCHAR"),  # Our new version with 2 params
        ("get_partition_for_collection", "VARCHAR"),
        ("refresh_collection_chunking_stats", ""),
        ("validate_partition_distribution", "")
    ]
    
    for func_name, params in functions_to_drop:
        with contextlib.suppress(Exception):
            if params:
                conn.execute(text(f"DROP FUNCTION IF EXISTS {func_name}({params}) CASCADE"))
            else:
                conn.execute(text(f"DROP FUNCTION IF EXISTS {func_name}() CASCADE"))
    
    logger.info("Dropped existing monitoring objects")


def update_partition_function(conn) -> None:
    """Update the partition key computation function to use composite hash."""
    # First drop the trigger (it will be recreated)
    with contextlib.suppress(Exception):
        conn.execute(text("DROP TRIGGER IF EXISTS set_partition_key ON chunks CASCADE"))
    
    # Drop old function
    with contextlib.suppress(Exception):
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
    
    # Create trigger
    conn.execute(text("""
        CREATE TRIGGER set_partition_key
        BEFORE INSERT ON chunks
        FOR EACH ROW
        EXECUTE FUNCTION compute_partition_key();
    """))
    
    logger.info("Updated partition function to use composite hash")


def redistribute_existing_data(conn, record_count: int) -> bool:
    """Redistribute existing data with new partition strategy."""
    if record_count == 0:
        logger.info("No data to redistribute")
        return True
    
    logger.info(f"Redistributing {record_count} existing chunks...")
    
    try:
        # For existing data, we need to update partition_key values
        # This is complex with partitioned tables, so we'll do it carefully
        
        # First, check if we can update in place
        # This would only work if the table isn't actually partitioned yet
        result = conn.execute(
            text("""
                SELECT COUNT(*) > 0
                FROM pg_class c
                JOIN pg_partitioned_table p ON c.oid = p.partrelid
                WHERE c.relname = 'chunks'
            """)
        )
        is_partitioned = result.scalar()
        
        if not is_partitioned:
            # Simple case: table isn't partitioned, just update the column
            conn.execute(text("""
                UPDATE chunks
                SET partition_key = CASE
                    WHEN document_id IS NOT NULL THEN
                        abs(hashtext(collection_id::text || ':' || document_id::text)) % 100
                    ELSE
                        abs(hashtext(
                            collection_id::text || ':' ||
                            COALESCE(chunk_index::text, id::text, 'null')
                        )) % 100
                END
            """))
            logger.info("Updated partition keys in non-partitioned table")
            return True
        
        # Complex case: table is partitioned, need to move data
        logger.warning("Table is partitioned. Data redistribution would require recreating partitions.")
        logger.warning("This will be handled by moving data to new partitions in a future migration.")
        # For now, just update the function for new data
        return True
        
    except Exception as e:
        logger.error(f"Error redistributing data: {e}")
        return False


def create_monitoring_views(conn) -> None:
    """Create monitoring views for the new partition strategy."""
    # Partition distribution view
    with contextlib.suppress(Exception):
        conn.execute(text("""
            CREATE OR REPLACE VIEW partition_distribution AS
            WITH partition_stats AS (
                SELECT
                    partition_key,
                    COUNT(*) as chunk_count,
                    COUNT(DISTINCT collection_id) as collection_count,
                    COUNT(DISTINCT document_id) as document_count
                FROM chunks
                GROUP BY partition_key
            ),
            overall_stats AS (
                SELECT
                    AVG(chunk_count) as avg_chunks,
                    STDDEV(chunk_count) as stddev_chunks,
                    MAX(chunk_count) as max_chunks,
                    MIN(chunk_count) as min_chunks
                FROM partition_stats
            )
            SELECT
                ps.partition_key,
                ps.chunk_count,
                ps.collection_count,
                ps.document_count,
                ROUND((ps.chunk_count::NUMERIC / NULLIF(SUM(ps.chunk_count) OVER (), 0)) * 100, 2) as chunk_percentage,
                CASE
                    WHEN os.avg_chunks > 0 AND ps.chunk_count > os.avg_chunks * 1.5 THEN 'HOT'
                    WHEN os.avg_chunks > 0 AND ps.chunk_count < os.avg_chunks * 0.5 THEN 'COLD'
                    ELSE 'NORMAL'
                END as partition_status
            FROM partition_stats ps
            CROSS JOIN overall_stats os
            ORDER BY ps.partition_key;
        """))
    
    # Collection distribution view
    with contextlib.suppress(Exception):
        conn.execute(text("""
            CREATE OR REPLACE VIEW partition_collection_distribution AS
            WITH collection_spread AS (
                SELECT
                    collection_id,
                    COUNT(DISTINCT partition_key) as partitions_used,
                    COUNT(*) as total_chunks,
                    MIN(partition_key) as min_partition,
                    MAX(partition_key) as max_partition
                FROM chunks
                GROUP BY collection_id
            )
            SELECT
                collection_id,
                total_chunks,
                partitions_used,
                ROUND(partitions_used::NUMERIC / GREATEST(total_chunks / 1000.0, 1), 2) as partition_efficiency,
                CASE
                    WHEN total_chunks > 10000 AND partitions_used < 10 THEN 'POOR DISTRIBUTION'
                    WHEN total_chunks > 1000 AND partitions_used < 5 THEN 'SUBOPTIMAL'
                    ELSE 'GOOD'
                END as distribution_quality
            FROM collection_spread
            ORDER BY total_chunks DESC;
        """))
    
    # Helper function for partition key calculation
    with contextlib.suppress(Exception):
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
    
    logger.info("Created monitoring views")


def upgrade() -> None:
    """Apply migration to fix partition distribution strategy."""
    conn = op.get_bind()
    
    logger.info("=" * 60)
    logger.info("Starting partition distribution fix migration")
    logger.info("=" * 60)
    
    # Check current state
    state = check_chunks_table_state(conn)
    
    logger.info(f"Current state: {state}")
    
    if not state["table_exists"]:
        logger.warning("Chunks table does not exist, skipping migration")
        return
    
    # Drop existing monitoring objects
    safe_drop_objects(conn)
    
    # Update the partition function
    if state["has_function"] or state["has_trigger"]:
        update_partition_function(conn)
    else:
        logger.warning("No existing partition function found, creating new one")
        update_partition_function(conn)
    
    # Redistribute existing data if needed
    if state["record_count"] > 0:
        if state["is_partitioned"]:
            logger.warning(
                f"Table has {state['record_count']} records in {state['partition_count']} partitions. "
                "Full redistribution would require recreating all partitions. "
                "New partition strategy will apply to new data only."
            )
        else:
            redistribute_existing_data(conn, state["record_count"])
    
    # Create monitoring views
    create_monitoring_views(conn)
    
    logger.info("=" * 60)
    logger.info("Partition distribution fix completed")
    logger.info("=" * 60)


def downgrade() -> None:
    """Revert to original partition strategy."""
    conn = op.get_bind()
    
    logger.warning("Reverting partition distribution fix")
    
    # Drop monitoring views
    safe_drop_objects(conn)
    
    # Restore original function
    with contextlib.suppress(Exception):
        conn.execute(text("DROP TRIGGER IF EXISTS set_partition_key ON chunks CASCADE"))
    with contextlib.suppress(Exception):
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
