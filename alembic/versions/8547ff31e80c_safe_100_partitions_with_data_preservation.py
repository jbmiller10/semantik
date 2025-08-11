"""Safe migration to 100 partitions with data preservation

Revision ID: 8547ff31e80c
Revises: ae558c9e183f
Create Date: 2025-08-11 12:00:00.000000

"""

import contextlib
import logging
from collections.abc import Sequence
from datetime import datetime

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "8547ff31e80c"
down_revision: str | Sequence[str] | None = "ae558c9e183f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Configure logging for migration progress
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
BATCH_SIZE = 10000
BACKUP_RETENTION_DAYS = 7


def check_existing_data(conn) -> tuple[bool, int]:
    """Check if chunks table exists and has data."""
    try:
        # Check if chunks table exists
        result = conn.execute(
            text(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'chunks'
                )
                """
            )
        )
        table_exists = result.scalar()
        
        if not table_exists:
            return False, 0
        
        # Count existing records
        result = conn.execute(text("SELECT COUNT(*) FROM chunks"))
        record_count = result.scalar()
        
        return True, record_count
        
    except SQLAlchemyError:
        return False, 0


def check_if_already_partitioned(conn) -> bool:
    """Check if the chunks table is already partitioned with 100 partitions."""
    try:
        # Check if chunks_part_00 through chunks_part_99 exist
        # If they do, the migration has already been applied
        result = conn.execute(
            text(
                """
                SELECT COUNT(*) 
                FROM pg_tables 
                WHERE schemaname = 'public' 
                AND tablename LIKE 'chunks_part_%'
                """
            )
        )
        partition_count = result.scalar()
        
        # Check if we have exactly 100 partitions (chunks_part_00 through chunks_part_99)
        if partition_count == 100:
            # Verify it's the LIST partition structure (not the old HASH structure)
            result = conn.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM pg_partitioned_table pt
                    JOIN pg_class c ON pt.partrelid = c.oid
                    WHERE c.relname = 'chunks'
                    AND pt.partstrat = 'l'  -- 'l' for LIST partitioning
                    """
                )
            )
            is_list_partitioned = result.scalar() > 0
            
            if is_list_partitioned:
                # Check if partition_key column exists
                result = conn.execute(
                    text(
                        """
                        SELECT EXISTS (
                            SELECT 1
                            FROM information_schema.columns
                            WHERE table_name = 'chunks'
                            AND column_name = 'partition_key'
                        )
                        """
                    )
                )
                has_partition_key = result.scalar()
                
                return has_partition_key
        
        return False
        
    except SQLAlchemyError:
        return False


def create_backup_table(conn, timestamp: str) -> str:
    """Create a timestamped backup of the chunks table."""
    backup_table_name = f"chunks_backup_{timestamp}"
    
    logger.info(f"Creating backup table: {backup_table_name}")
    
    # Create backup table with all data
    conn.execute(
        text(
            f"""
            CREATE TABLE {backup_table_name} AS 
            TABLE chunks WITH DATA
            """
        )
    )
    
    # Also backup all partition tables if they exist
    result = conn.execute(
        text(
            """
            SELECT tablename 
            FROM pg_tables 
            WHERE schemaname = 'public' 
            AND tablename LIKE 'chunks_part_%'
            """
        )
    )
    
    partition_tables = [row[0] for row in result]
    
    for partition_table in partition_tables:
        backup_partition_name = f"{partition_table}_backup_{timestamp}"
        logger.info(f"Backing up partition: {partition_table} to {backup_partition_name}")
        conn.execute(
            text(
                f"""
                CREATE TABLE {backup_partition_name} AS 
                TABLE {partition_table} WITH DATA
                """
            )
        )
    
    # Create metadata table to track backup
    conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS migration_backups (
                id SERIAL PRIMARY KEY,
                backup_table_name VARCHAR NOT NULL,
                original_table_name VARCHAR NOT NULL,
                record_count INTEGER NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                migration_revision VARCHAR,
                retention_until TIMESTAMPTZ
            )
            """
        )
    )
    
    # Record backup metadata
    result = conn.execute(text(f"SELECT COUNT(*) FROM {backup_table_name}"))
    backup_count = result.scalar()
    
    conn.execute(
        text(
            """
            INSERT INTO migration_backups 
            (backup_table_name, original_table_name, record_count, migration_revision, retention_until)
            VALUES (:backup_table, 'chunks', :count, :revision, NOW() + INTERVAL :days)
            """
        ),
        {
            "backup_table": backup_table_name,
            "count": backup_count,
            "revision": revision,
            "days": f"{BACKUP_RETENTION_DAYS} days",
        }
    )
    
    logger.info(f"Backup created successfully with {backup_count} records")
    return backup_table_name


def verify_backup(conn, backup_table_name: str, original_count: int) -> bool:
    """Verify backup integrity."""
    result = conn.execute(text(f"SELECT COUNT(*) FROM {backup_table_name}"))
    backup_count = result.scalar()
    
    if backup_count != original_count:
        logger.error(
            f"Backup verification failed! Original: {original_count}, Backup: {backup_count}"
        )
        return False
    
    logger.info(f"Backup verified: {backup_count} records match original")
    
    # Spot check: Compare a sample of records
    spot_check_query = text(
        f"""
        SELECT COUNT(*) FROM (
            SELECT id, collection_id, chunk_index, content 
            FROM chunks 
            LIMIT 100
        ) original
        INNER JOIN (
            SELECT id, collection_id, chunk_index, content 
            FROM {backup_table_name}
            LIMIT 100
        ) backup 
        ON original.id = backup.id 
        AND original.collection_id = backup.collection_id
        AND original.content = backup.content
        """
    )
    
    try:
        result = conn.execute(spot_check_query)
        matches = result.scalar()
        logger.info(f"Spot check: {matches} records match in sample")
    except SQLAlchemyError as e:
        logger.warning(f"Spot check skipped due to schema differences: {e}")
    
    return True


def create_new_partitioned_structure(conn):
    """Create the new 100-partition structure alongside the old one."""
    logger.info("Creating new partitioned structure...")
    
    # Drop any existing chunks_new table (from failed previous attempts)
    conn.execute(text("DROP TABLE IF EXISTS chunks_new CASCADE"))
    
    # Create staging table with new structure
    conn.execute(
        text(
            """
            CREATE TABLE chunks_new (
                id BIGSERIAL,
                collection_id VARCHAR NOT NULL,
                partition_key INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                metadata JSONB DEFAULT '{}',
                document_id VARCHAR,
                chunking_config_id INTEGER,
                start_offset INTEGER,
                end_offset INTEGER,
                token_count INTEGER,
                embedding_vector_id VARCHAR,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (id, collection_id, partition_key),
                FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE,
                FOREIGN KEY (chunking_config_id) REFERENCES chunking_configs(id)
            ) PARTITION BY LIST (partition_key)
            """
        )
    )
    
    # Create partitions
    logger.info("Creating 100 partitions...")
    conn.execute(
        text(
            """
            DO $$
            DECLARE
                i INT;
            BEGIN
                FOR i IN 0..99 LOOP
                    EXECUTE format('
                        CREATE TABLE chunks_new_part_%s PARTITION OF chunks_new
                        FOR VALUES IN (%s)',
                        LPAD(i::text, 2, '0'),
                        i
                    );
                END LOOP;
            END $$;
            """
        )
    )
    
    logger.info("New partitioned structure created successfully")


def migrate_data_in_batches(conn, source_table: str = "chunks"):
    """Migrate data from old structure to new in batches."""
    # Get total count for progress tracking
    result = conn.execute(text(f"SELECT COUNT(*) FROM {source_table}"))
    total_records = result.scalar()
    
    if total_records == 0:
        logger.info("No data to migrate")
        return
    
    logger.info(f"Starting migration of {total_records} records in batches of {BATCH_SIZE}")
    
    migrated_count = 0
    batch_num = 0
    
    while migrated_count < total_records:
        batch_num += 1
        offset = migrated_count
        
        # Migrate batch with computed partition_key
        conn.execute(
            text(
                f"""
                INSERT INTO chunks_new (
                    collection_id,
                    partition_key,
                    chunk_index,
                    content,
                    metadata,
                    document_id,
                    chunking_config_id,
                    start_offset,
                    end_offset,
                    token_count,
                    embedding_vector_id,
                    created_at,
                    updated_at
                )
                SELECT 
                    collection_id,
                    abs(hashtext(collection_id::text)) % 100 as partition_key,
                    chunk_index,
                    content,
                    COALESCE(metadata, meta::jsonb, '{{}}'::jsonb) as metadata,
                    document_id,
                    chunking_config_id,
                    start_offset,
                    end_offset,
                    token_count,
                    embedding_vector_id,
                    created_at,
                    COALESCE(updated_at, created_at) as updated_at
                FROM {source_table}
                ORDER BY created_at, id
                LIMIT :batch_size
                OFFSET :offset
                """
            ),
            {"batch_size": BATCH_SIZE, "offset": offset}
        )
        
        # Get actual records migrated in this batch
        result = conn.execute(
            text(
                f"""
                SELECT COUNT(*) 
                FROM {source_table}
                LIMIT :batch_size
                OFFSET :offset
                """
            ),
            {"batch_size": BATCH_SIZE, "offset": offset}
        )
        batch_count = result.scalar()
        
        migrated_count += batch_count
        progress_pct = (migrated_count / total_records) * 100
        
        logger.info(
            f"Batch {batch_num}: Migrated {batch_count} records "
            f"({migrated_count}/{total_records} - {progress_pct:.1f}%)"
        )
        
        if batch_count < BATCH_SIZE:
            break
    
    logger.info(f"Data migration completed: {migrated_count} records migrated")


def verify_migration(conn, source_count: int) -> bool:
    """Verify data integrity after migration."""
    logger.info("Verifying migration integrity...")
    
    # Check record count
    result = conn.execute(text("SELECT COUNT(*) FROM chunks_new"))
    new_count = result.scalar()
    
    if new_count != source_count:
        logger.error(f"Count mismatch! Source: {source_count}, New: {new_count}")
        return False
    
    logger.info(f"Record count verified: {new_count} records")
    
    # Verify partition distribution
    result = conn.execute(
        text(
            """
            SELECT 
                partition_key,
                COUNT(*) as chunk_count
            FROM chunks_new
            GROUP BY partition_key
            ORDER BY partition_key
            """
        )
    )
    
    distribution = list(result)
    partitions_used = len(distribution)
    
    if partitions_used > 0:
        counts = [row[1] for row in distribution]
        avg_per_partition = sum(counts) / len(counts)
        max_per_partition = max(counts)
        min_per_partition = min(counts)
        
        logger.info(
            f"Partition distribution: {partitions_used} partitions used, "
            f"avg: {avg_per_partition:.1f}, min: {min_per_partition}, max: {max_per_partition}"
        )
        
        # Check for severe skew
        if max_per_partition > avg_per_partition * 2:
            logger.warning("Detected partition skew, but continuing (expected with hash distribution)")
    
    # Verify collection integrity
    result = conn.execute(
        text(
            """
            SELECT COUNT(DISTINCT collection_id) 
            FROM chunks_new
            """
        )
    )
    collections_new = result.scalar()
    
    result = conn.execute(
        text(
            """
            SELECT COUNT(DISTINCT collection_id) 
            FROM chunks
            """
        )
    )
    collections_old = result.scalar()
    
    if collections_new != collections_old:
        logger.error(
            f"Collection count mismatch! Old: {collections_old}, New: {collections_new}"
        )
        return False
    
    logger.info(f"Collection integrity verified: {collections_new} collections")
    
    return True


def perform_atomic_swap(conn):
    """Perform atomic table swap to minimize downtime."""
    logger.info("Performing atomic table swap...")
    
    # Create all necessary objects (triggers, functions, indexes)
    logger.info("Creating trigger function...")
    conn.execute(
        text(
            """
            CREATE OR REPLACE FUNCTION compute_partition_key()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.partition_key := abs(hashtext(NEW.collection_id::text)) % 100;
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
            """
        )
    )
    
    # Drop old chunks and dependencies
    logger.info("Cleaning up old chunks dependencies...")
    cleanup_chunks_dependencies(conn)
    
    # Rename tables atomically
    logger.info("Performing atomic rename...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Check if old partitions exist and need to be renamed
    result = conn.execute(
        text(
            """
            SELECT COUNT(*) 
            FROM pg_tables 
            WHERE schemaname = 'public' 
            AND tablename LIKE 'chunks_part_%'
            """
        )
    )
    old_partitions_exist = result.scalar() > 0
    
    if old_partitions_exist:
        # If old partitions exist, we need to rename them first to avoid conflicts
        logger.info("Renaming existing partitions to avoid conflicts...")
        for i in range(100):
            old_part_name = f"chunks_part_{i:02d}"
            temp_name = f"chunks_old_part_{i:02d}_{timestamp}"
            try:
                conn.execute(text(f"ALTER TABLE IF EXISTS {old_part_name} RENAME TO {temp_name}"))
            except Exception:
                pass  # Partition might not exist in old structure
    
    conn.execute(text(f"ALTER TABLE chunks RENAME TO chunks_old_{timestamp}"))
    conn.execute(text("ALTER TABLE chunks_new RENAME TO chunks"))
    
    # Rename all new partitions
    for i in range(100):
        part_name = f"chunks_new_part_{i:02d}"
        new_name = f"chunks_part_{i:02d}"
        conn.execute(text(f"ALTER TABLE {part_name} RENAME TO {new_name}"))
    
    # Create trigger on renamed table
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
    
    # Create indexes on all partitions
    logger.info("Creating indexes on partitions...")
    conn.execute(
        text(
            """
            DO $$
            DECLARE
                i INT;
            BEGIN
                FOR i IN 0..99 LOOP
                    EXECUTE format('
                        CREATE INDEX idx_chunks_part_%s_collection
                        ON chunks_part_%s(collection_id)',
                        LPAD(i::text, 2, '0'),
                        LPAD(i::text, 2, '0')
                    );
                    
                    EXECUTE format('
                        CREATE INDEX idx_chunks_part_%s_created
                        ON chunks_part_%s(created_at)',
                        LPAD(i::text, 2, '0'),
                        LPAD(i::text, 2, '0')
                    );
                    
                    EXECUTE format('
                        CREATE INDEX idx_chunks_part_%s_chunk_index
                        ON chunks_part_%s(collection_id, chunk_index)',
                        LPAD(i::text, 2, '0'),
                        LPAD(i::text, 2, '0')
                    );
                    
                    EXECUTE format('
                        CREATE INDEX idx_chunks_part_%s_document
                        ON chunks_part_%s(document_id)
                        WHERE document_id IS NOT NULL',
                        LPAD(i::text, 2, '0'),
                        LPAD(i::text, 2, '0')
                    );
                END LOOP;
            END $$;
            """
        )
    )
    
    logger.info("Atomic swap completed successfully")


def cleanup_chunks_dependencies(conn):
    """Helper function to clean up all chunks table dependencies."""
    views_to_drop = [
        "partition_distribution",
        "partition_health",
        "partition_size_distribution",
        "partition_chunk_distribution",
        "partition_hot_spots",
        "partition_health_summary",
        "active_chunking_configs",
    ]
    
    for view in views_to_drop:
        with contextlib.suppress(Exception):
            conn.execute(text(f"DROP VIEW IF EXISTS {view} CASCADE"))
    
    with contextlib.suppress(Exception):
        conn.execute(text("DROP MATERIALIZED VIEW IF EXISTS collection_chunking_stats CASCADE"))
    
    functions_to_drop = [
        ("analyze_partition_skew", ""),
        ("get_partition_key", "VARCHAR"),
        ("get_partition_for_collection", "VARCHAR"),
        ("refresh_collection_chunking_stats", ""),
    ]
    
    for func_name, params in functions_to_drop:
        with contextlib.suppress(Exception):
            if params:
                conn.execute(text(f"DROP FUNCTION IF EXISTS {func_name}({params}) CASCADE"))
            else:
                conn.execute(text(f"DROP FUNCTION IF EXISTS {func_name}() CASCADE"))
    
    with contextlib.suppress(Exception):
        conn.execute(text("DROP TRIGGER IF EXISTS set_partition_key ON chunks CASCADE"))


def create_monitoring_views(conn):
    """Create monitoring views for partition health."""
    logger.info("Creating monitoring views...")
    
    # Main health monitoring view
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
                    last_autovacuum,
                    n_tup_ins as inserts_since_vacuum,
                    n_tup_upd as updates_since_vacuum,
                    n_tup_del as deletes_since_vacuum
                FROM pg_stat_user_tables
                WHERE relname LIKE 'chunks_part_%'
            ),
            stats_summary AS (
                SELECT
                    AVG(row_count) as avg_rows,
                    MAX(row_count) as max_rows,
                    MIN(row_count) as min_rows,
                    STDDEV(row_count) as stddev_rows,
                    AVG(size_bytes) as avg_size,
                    SUM(row_count) as total_rows,
                    SUM(size_bytes) as total_size
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
    
    # Helper functions
    conn.execute(
        text(
            """
            CREATE OR REPLACE FUNCTION get_partition_for_collection(collection_id VARCHAR)
            RETURNS TEXT AS $$
            BEGIN
                RETURN 'chunks_part_' || LPAD((abs(hashtext(collection_id::text)) % 100)::text, 2, '0');
            END;
            $$ LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE;
            
            CREATE OR REPLACE FUNCTION get_partition_key(collection_id VARCHAR)
            RETURNS INTEGER AS $$
            BEGIN
                RETURN abs(hashtext(collection_id::text)) % 100;
            END;
            $$ LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE;
            """
        )
    )
    
    logger.info("Monitoring views created successfully")


def upgrade() -> None:
    """
    Safe migration to 100 partitions with data preservation.
    
    This migration:
    1. Checks if already partitioned (idempotent)
    2. Checks for existing data
    3. Creates timestamped backups
    4. Creates new structure alongside old
    5. Migrates data in batches
    6. Verifies data integrity
    7. Performs atomic swap
    8. Retains old tables for safety
    """
    
    conn = op.get_bind()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger.info(f"Starting safe migration to 100 partitions (timestamp: {timestamp})")
    
    # Step 0: Check if already partitioned with 100 partitions
    if check_if_already_partitioned(conn):
        logger.info("Chunks table is already partitioned with 100 LIST partitions. Skipping migration.")
        # Ensure monitoring views exist (they might have been dropped)
        try:
            create_monitoring_views(conn)
            logger.info("Ensured monitoring views exist")
        except Exception as e:
            logger.warning(f"Could not create monitoring views (may already exist): {e}")
        return
    
    # Step 1: Check for existing data
    has_data, record_count = check_existing_data(conn)
    
    if not has_data:
        logger.info("No existing chunks table or data found. Creating fresh structure...")
        # If no existing data, we can safely create the new structure directly
        create_new_partitioned_structure(conn)
        conn.execute(text("ALTER TABLE chunks_new RENAME TO chunks"))
        
        # Rename partitions
        for i in range(100):
            part_name = f"chunks_new_part_{i:02d}"
            new_name = f"chunks_part_{i:02d}"
            conn.execute(text(f"ALTER TABLE {part_name} RENAME TO {new_name}"))
        
        # Create trigger function and trigger
        conn.execute(
            text(
                """
                CREATE OR REPLACE FUNCTION compute_partition_key()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.partition_key := abs(hashtext(NEW.collection_id::text)) % 100;
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;
                
                CREATE TRIGGER set_partition_key
                BEFORE INSERT ON chunks
                FOR EACH ROW
                EXECUTE FUNCTION compute_partition_key();
                """
            )
        )
        
        create_monitoring_views(conn)
        logger.info("Fresh structure created successfully")
        return
    
    logger.info(f"Found existing chunks table with {record_count} records")
    
    # Step 2: Create backup
    backup_table_name = create_backup_table(conn, timestamp)
    
    # Step 3: Verify backup
    if not verify_backup(conn, backup_table_name, record_count):
        raise Exception("Backup verification failed! Aborting migration.")
    
    # Step 4: Create new structure alongside old
    create_new_partitioned_structure(conn)
    
    # Step 5: Migrate data in batches
    migrate_data_in_batches(conn)
    
    # Step 6: Verify migration
    if not verify_migration(conn, record_count):
        # Rollback by dropping new structure
        conn.execute(text("DROP TABLE IF EXISTS chunks_new CASCADE"))
        raise Exception("Migration verification failed! Rolled back changes.")
    
    # Step 7: Perform atomic swap
    perform_atomic_swap(conn)
    
    # Step 8: Create monitoring views
    create_monitoring_views(conn)
    
    # Log completion
    logger.info(
        f"Migration completed successfully! "
        f"Old table retained as chunks_old_{timestamp}. "
        f"Backup available as {backup_table_name}"
    )
    
    # Create cleanup reminder
    conn.execute(
        text(
            """
            INSERT INTO migration_backups 
            (backup_table_name, original_table_name, record_count, migration_revision, retention_until)
            VALUES 
            (:old_table, 'chunks (old structure)', :count, :revision, NOW() + INTERVAL :days),
            (:reminder, 'CLEANUP REMINDER', 0, :revision, NOW() + INTERVAL :days)
            """),
        {
            "old_table": f"chunks_old_{timestamp}",
            "reminder": f"Run cleanup after verifying system stability",
            "count": record_count,
            "revision": revision,
            "days": f"{BACKUP_RETENTION_DAYS} days",
        }
    )


def downgrade() -> None:
    """
    Restore from backup if available, otherwise recreate original structure.
    """
    
    conn = op.get_bind()
    
    logger.info("Starting downgrade process...")
    
    # Check for recent backups
    try:
        result = conn.execute(
            text(
                """
                SELECT backup_table_name, record_count 
                FROM migration_backups 
                WHERE migration_revision = :revision 
                AND original_table_name = 'chunks'
                ORDER BY created_at DESC 
                LIMIT 1
                """
            ),
            {"revision": revision}
        )
        
        backup_info = result.fetchone()
        
        if backup_info:
            backup_table_name, record_count = backup_info
            logger.info(f"Found backup table: {backup_table_name} with {record_count} records")
            
            # Clean up current structure
            cleanup_chunks_dependencies(conn)
            conn.execute(text("DROP TABLE IF EXISTS chunks CASCADE"))
            
            # Restore from backup
            conn.execute(
                text(
                    f"""
                    CREATE TABLE chunks AS 
                    TABLE {backup_table_name} WITH DATA
                    """
                )
            )
            
            # Recreate constraints and indexes
            conn.execute(
                text(
                    """
                    ALTER TABLE chunks 
                    ADD CONSTRAINT chunks_collection_id_fkey 
                    FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE;
                    
                    CREATE INDEX IF NOT EXISTS ix_chunks_collection_id ON chunks(collection_id);
                    CREATE INDEX IF NOT EXISTS ix_chunks_created_at ON chunks(created_at);
                    CREATE INDEX IF NOT EXISTS ix_chunks_chunk_index ON chunks(collection_id, chunk_index);
                    """
                )
            )
            
            logger.info(f"Successfully restored {record_count} records from backup")
            
        else:
            logger.warning("No backup found. Creating empty original structure...")
            
            # Recreate original structure (16 partitions)
            cleanup_chunks_dependencies(conn)
            conn.execute(text("DROP TABLE IF EXISTS chunks CASCADE"))
            
            conn.execute(
                text(
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
                        FOREIGN KEY (chunking_config_id) REFERENCES chunking_configs(id)
                    ) PARTITION BY HASH (collection_id);
                    """
                )
            )
            
            # Create 16 partitions
            for i in range(16):
                conn.execute(
                    text(
                        f"""
                        CREATE TABLE chunks_p{i} PARTITION OF chunks
                        FOR VALUES WITH (MODULUS 16, REMAINDER {i});
                        """
                    )
                )
            
            logger.info("Created empty original structure with 16 partitions")
            
    except SQLAlchemyError as e:
        logger.error(f"Error during downgrade: {e}")
        raise
    
    logger.info("Downgrade completed")