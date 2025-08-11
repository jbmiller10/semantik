"""Replace trigger with generated column for partition_key

Revision ID: db003_replace_trigger
Revises: 8547ff31e80c
Create Date: 2025-08-11 14:00:00.000000

This migration replaces the trigger-based partition_key computation with a 
GENERATED column for better performance. PostgreSQL 12+ supports GENERATED 
columns which are more efficient than triggers.

Benefits of GENERATED columns:
- Computed at storage level (more efficient)
- Query optimizer aware of generation expression
- No function call overhead
- Automatic dependency tracking
- Cleaner, more maintainable
"""

import logging
from collections.abc import Sequence
from typing import Any

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from alembic import op

# revision identifiers, used by Alembic
revision: str = "db003_replace_trigger"
down_revision: str | Sequence[str] | None = "8547ff31e80c"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_postgres_version(conn) -> tuple[int, int]:
    """Get PostgreSQL major and minor version.
    
    Returns:
        Tuple of (major, minor) version numbers
    """
    try:
        result = conn.execute(text("SELECT version()"))
        version_string = result.scalar()
        
        # Parse version from string like "PostgreSQL 16.1 on x86_64-pc-linux-gnu..."
        import re
        match = re.search(r'PostgreSQL (\d+)\.(\d+)', version_string)
        if match:
            major = int(match.group(1))
            minor = int(match.group(2))
            logger.info(f"Detected PostgreSQL version: {major}.{minor}")
            return major, minor
        
        # Try alternative format for major version only (PostgreSQL 12+)
        match = re.search(r'PostgreSQL (\d+)', version_string)
        if match:
            major = int(match.group(1))
            logger.info(f"Detected PostgreSQL version: {major}")
            return major, 0
            
    except Exception as e:
        logger.warning(f"Could not determine PostgreSQL version: {e}")
    
    # Default to assuming old version if detection fails
    return 11, 0


def check_current_implementation(conn) -> dict[str, bool]:
    """Check current partition_key implementation.
    
    Returns:
        Dictionary with implementation status
    """
    status = {
        "has_trigger": False,
        "has_generated_column": False,
        "is_partitioned": False,
        "trigger_name": None,
        "function_name": None,
    }
    
    try:
        # Check for trigger
        result = conn.execute(
            text("""
                SELECT tgname 
                FROM pg_trigger 
                WHERE tgrelid = 'chunks'::regclass 
                AND tgname = 'set_partition_key'
            """)
        )
        trigger = result.fetchone()
        if trigger:
            status["has_trigger"] = True
            status["trigger_name"] = trigger[0]
            logger.info(f"Found trigger: {trigger[0]}")
        
        # Check for function
        result = conn.execute(
            text("""
                SELECT proname 
                FROM pg_proc 
                WHERE proname = 'compute_partition_key'
            """)
        )
        function = result.fetchone()
        if function:
            status["function_name"] = function[0]
            logger.info(f"Found function: {function[0]}")
        
        # Check if partition_key is a generated column
        result = conn.execute(
            text("""
                SELECT attgenerated 
                FROM pg_attribute 
                WHERE attrelid = 'chunks'::regclass 
                AND attname = 'partition_key'
            """)
        )
        col_info = result.fetchone()
        if col_info and col_info[0] == 's':  # 's' means STORED generated column
            status["has_generated_column"] = True
            logger.info("partition_key is already a GENERATED column")
        
        # Check if table is partitioned
        result = conn.execute(
            text("""
                SELECT 
                    c.relkind,
                    p.partstrat,
                    p.partattrs
                FROM pg_class c
                LEFT JOIN pg_partitioned_table p ON c.oid = p.partrelid
                WHERE c.relname = 'chunks'
                AND c.relnamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'public')
            """)
        )
        partition_info = result.fetchone()
        if partition_info and partition_info[0] == 'p':  # 'p' means partitioned table
            status["is_partitioned"] = True
            logger.info(f"chunks table is partitioned (strategy: {partition_info[1]})")
            
            # Check if partition_key is part of the partition key
            result = conn.execute(
                text("""
                    SELECT a.attname
                    FROM pg_attribute a
                    JOIN pg_class c ON a.attrelid = c.oid
                    JOIN pg_partitioned_table p ON c.oid = p.partrelid
                    WHERE c.relname = 'chunks'
                    AND a.attnum = ANY(p.partattrs)
                """)
            )
            partition_cols = [row[0] for row in result]
            if 'partition_key' in partition_cols:
                logger.info(f"partition_key is part of the partition key columns: {partition_cols}")
        
    except SQLAlchemyError as e:
        logger.warning(f"Error checking current implementation: {e}")
    
    return status


def verify_partition_keys(conn, sample_size: int = 1000) -> bool:
    """Verify that partition keys are correctly computed.
    
    Args:
        conn: Database connection
        sample_size: Number of records to check
        
    Returns:
        True if all checked partition keys are correct
    """
    try:
        result = conn.execute(
            text(f"""
                SELECT 
                    collection_id,
                    partition_key,
                    abs(hashtext(collection_id::text)) % 100 as computed_key
                FROM chunks
                LIMIT {sample_size}
            """)
        )
        
        mismatches = 0
        total = 0
        for row in result:
            total += 1
            if row.partition_key != row.computed_key:
                mismatches += 1
                logger.warning(
                    f"Partition key mismatch for collection_id={row.collection_id}: "
                    f"stored={row.partition_key}, computed={row.computed_key}"
                )
        
        if mismatches > 0:
            logger.error(f"Found {mismatches} partition key mismatches out of {total} checked")
            return False
        
        logger.info(f"Verified {total} partition keys - all correct")
        return True
        
    except SQLAlchemyError as e:
        logger.error(f"Error verifying partition keys: {e}")
        return False


def convert_to_generated_column(conn):
    """Convert partition_key from trigger-computed to GENERATED column.
    
    This is the main conversion logic for PostgreSQL 12+.
    NOTE: This function should NOT be called for partitioned tables.
    """
    logger.info("Starting conversion to GENERATED column...")
    
    # Safety check: Verify table is not partitioned
    result = conn.execute(
        text("""
            SELECT c.relkind
            FROM pg_class c
            WHERE c.relname = 'chunks'
            AND c.relnamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'public')
        """)
    )
    table_info = result.fetchone()
    if table_info and table_info[0] == 'p':
        raise ValueError(
            "CRITICAL: convert_to_generated_column called on partitioned table! "
            "This should never happen. Aborting to prevent data loss."
        )
    
    # Step 1: Drop the trigger (but keep the function for now, in case we need to rollback)
    logger.info("Dropping trigger...")
    conn.execute(text("DROP TRIGGER IF EXISTS set_partition_key ON chunks"))
    
    # Step 2: PostgreSQL doesn't allow direct conversion to GENERATED, 
    # so we need to:
    # - Create a new column
    # - Copy data
    # - Drop old column
    # - Rename new column
    
    logger.info("Creating new GENERATED column...")
    
    # Check if we can add the column directly (in case of fresh install)
    result = conn.execute(
        text("""
            SELECT COUNT(*) FROM chunks
        """)
    )
    row_count = result.scalar()
    
    if row_count == 0:
        logger.info("No data in chunks table, performing direct conversion...")
        
        # Drop and recreate the column as GENERATED
        conn.execute(text("""
            ALTER TABLE chunks 
            DROP COLUMN partition_key
        """))
        
        # Add as GENERATED column
        conn.execute(text("""
            ALTER TABLE chunks 
            ADD COLUMN partition_key INTEGER 
            GENERATED ALWAYS AS (abs(hashtext(collection_id::text)) % 100) STORED NOT NULL
        """))
        
        # Re-add to primary key
        conn.execute(text("""
            ALTER TABLE chunks 
            DROP CONSTRAINT chunks_pkey,
            ADD PRIMARY KEY (id, collection_id, partition_key)
        """))
        
    else:
        logger.info(f"Table has {row_count} rows, performing careful migration...")
        
        # Create temporary column with GENERATED
        conn.execute(text("""
            ALTER TABLE chunks 
            ADD COLUMN partition_key_new INTEGER 
            GENERATED ALWAYS AS (abs(hashtext(collection_id::text)) % 100) STORED
        """))
        
        # Verify the generated values match existing ones
        result = conn.execute(text("""
            SELECT COUNT(*) 
            FROM chunks 
            WHERE partition_key != partition_key_new
        """))
        
        mismatches = result.scalar()
        if mismatches > 0:
            # Rollback if there are mismatches
            conn.execute(text("ALTER TABLE chunks DROP COLUMN partition_key_new"))
            raise ValueError(
                f"Found {mismatches} mismatches between existing partition_key "
                "and computed values. Aborting conversion."
            )
        
        logger.info("All partition keys match, proceeding with column swap...")
        
        # Drop the primary key constraint
        conn.execute(text("""
            ALTER TABLE chunks DROP CONSTRAINT chunks_pkey
        """))
        
        # Drop old column and rename new one
        conn.execute(text("""
            ALTER TABLE chunks DROP COLUMN partition_key
        """))
        
        conn.execute(text("""
            ALTER TABLE chunks RENAME COLUMN partition_key_new TO partition_key
        """))
        
        # Add NOT NULL constraint
        conn.execute(text("""
            ALTER TABLE chunks ALTER COLUMN partition_key SET NOT NULL
        """))
        
        # Recreate primary key
        conn.execute(text("""
            ALTER TABLE chunks ADD PRIMARY KEY (id, collection_id, partition_key)
        """))
    
    # Step 3: Clean up the trigger function (no longer needed)
    logger.info("Cleaning up trigger function...")
    conn.execute(text("DROP FUNCTION IF EXISTS compute_partition_key() CASCADE"))
    
    logger.info("Successfully converted to GENERATED column")


def create_performance_test(conn):
    """Create a simple performance test to measure improvement."""
    logger.info("Creating performance measurement...")
    
    try:
        # Create a test collection if it doesn't exist
        conn.execute(text("""
            INSERT INTO collections (id, name, description, created_at, updated_at)
            VALUES ('test-perf-00000000-0000-0000-0000-000000000000', 
                    'Performance Test Collection', 
                    'Used for migration performance testing',
                    NOW(), NOW())
            ON CONFLICT (id) DO NOTHING
        """))
        
        # Measure insert performance (small batch)
        import time
        
        start_time = time.time()
        
        conn.execute(text("""
            INSERT INTO chunks (
                collection_id, chunk_index, content, 
                metadata, created_at, updated_at
            )
            SELECT 
                'test-perf-00000000-0000-0000-0000-000000000000',
                generate_series,
                'Test content for performance measurement ' || generate_series,
                '{}',
                NOW(),
                NOW()
            FROM generate_series(1, 100)
        """))
        
        elapsed = (time.time() - start_time) * 1000  # Convert to milliseconds
        per_row = elapsed / 100
        
        logger.info(f"Insert performance: {elapsed:.2f}ms for 100 rows ({per_row:.3f}ms per row)")
        
        # Clean up test data
        conn.execute(text("""
            DELETE FROM chunks 
            WHERE collection_id = 'test-perf-00000000-0000-0000-0000-000000000000'
            AND content LIKE 'Test content for performance measurement%'
        """))
        
        return per_row
        
    except Exception as e:
        logger.warning(f"Could not run performance test: {e}")
        return None


def upgrade() -> None:
    """
    Replace trigger-based partition_key with GENERATED column.
    
    This migration:
    1. Checks PostgreSQL version (requires 12+ for GENERATED columns)
    2. Verifies current implementation
    3. Converts trigger to GENERATED column if possible
    4. Measures performance improvement
    5. Cleans up old trigger and function
    """
    
    conn = op.get_bind()
    
    logger.info("=" * 60)
    logger.info("Starting DB-003: Replace Trigger with Generated Column")
    logger.info("=" * 60)
    
    # Check PostgreSQL version
    major_version, minor_version = get_postgres_version(conn)
    
    if major_version < 12:
        logger.warning(
            f"PostgreSQL {major_version}.{minor_version} does not support GENERATED columns. "
            "Keeping trigger-based implementation. Consider upgrading to PostgreSQL 12+ "
            "for better performance."
        )
        
        # Ensure trigger exists for older PostgreSQL versions
        conn.execute(text("""
            CREATE OR REPLACE FUNCTION compute_partition_key()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.partition_key := abs(hashtext(NEW.collection_id::text)) % 100;
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql IMMUTABLE;
            
            DROP TRIGGER IF EXISTS set_partition_key ON chunks;
            
            CREATE TRIGGER set_partition_key
            BEFORE INSERT ON chunks
            FOR EACH ROW
            EXECUTE FUNCTION compute_partition_key();
        """))
        
        logger.info("Trigger-based implementation verified for PostgreSQL < 12")
        return
    
    # Check current implementation
    impl_status = check_current_implementation(conn)
    
    if impl_status["has_generated_column"]:
        logger.info("partition_key is already a GENERATED column. Nothing to do.")
        return
    
    # Special handling for partitioned tables
    if impl_status["is_partitioned"]:
        logger.info("=" * 60)
        logger.info("PARTITIONED TABLE DETECTED")
        logger.info("=" * 60)
        logger.info(
            "The chunks table is partitioned by partition_key. "
            "Cannot convert to GENERATED column because partition keys cannot be dropped. "
            "The trigger-based implementation will be kept for optimal compatibility."
        )
        
        # Ensure trigger is properly set up for partitioned table
        if not impl_status["has_trigger"]:
            logger.info("Setting up trigger for partitioned table...")
            conn.execute(text("""
                CREATE OR REPLACE FUNCTION compute_partition_key()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.partition_key := abs(hashtext(NEW.collection_id::text)) % 100;
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql IMMUTABLE;
                
                DROP TRIGGER IF EXISTS set_partition_key ON chunks;
                
                CREATE TRIGGER set_partition_key
                BEFORE INSERT ON chunks
                FOR EACH ROW
                EXECUTE FUNCTION compute_partition_key();
            """))
            logger.info("Trigger created for partitioned table")
        else:
            logger.info("Trigger already exists for partitioned table")
        
        # Verify partition keys are correct
        result = conn.execute(text("SELECT COUNT(*) FROM chunks"))
        chunk_count = result.scalar()
        if chunk_count > 0:
            if verify_partition_keys(conn):
                logger.info("✓ All partition keys verified as correct")
            else:
                logger.warning("⚠ Some partition keys may be incorrect")
        
        logger.info("=" * 60)
        logger.info("Migration completed for partitioned table")
        logger.info("Using trigger-based implementation (optimal for partitioned tables)")
        logger.info("=" * 60)
        return
    
    if not impl_status["has_trigger"]:
        logger.warning(
            "No trigger found. This might be a fresh installation. "
            "Setting up GENERATED column directly."
        )
    
    # Measure performance before conversion (if there's data)
    result = conn.execute(text("SELECT COUNT(*) FROM chunks"))
    chunk_count = result.scalar()
    
    before_perf = None
    if impl_status["has_trigger"] and chunk_count > 0:
        logger.info("Measuring current trigger performance...")
        before_perf = create_performance_test(conn)
    
    # Verify existing partition keys before conversion
    if chunk_count > 0:
        if not verify_partition_keys(conn):
            raise ValueError(
                "Existing partition keys have inconsistencies. "
                "Please fix these before converting to GENERATED column."
            )
    
    # Perform the conversion
    try:
        convert_to_generated_column(conn)
        
        # Measure performance after conversion
        if before_perf is not None:
            logger.info("Measuring GENERATED column performance...")
            after_perf = create_performance_test(conn)
            
            if after_perf is not None:
                improvement = ((before_perf - after_perf) / before_perf) * 100
                logger.info(
                    f"Performance improvement: {improvement:.1f}% "
                    f"(from {before_perf:.3f}ms to {after_perf:.3f}ms per row)"
                )
        
        # Final verification
        if chunk_count > 0:
            if not verify_partition_keys(conn):
                raise ValueError("Partition keys are incorrect after conversion!")
        
        logger.info("=" * 60)
        logger.info("Migration completed successfully!")
        logger.info(f"PostgreSQL {major_version} now uses GENERATED column for partition_key")
        logger.info("Benefits: Better performance, cleaner code, automatic optimization")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error during conversion: {e}")
        logger.info("Rolling back changes...")
        
        # Attempt to restore trigger-based implementation
        conn.execute(text("""
            CREATE OR REPLACE FUNCTION compute_partition_key()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.partition_key := abs(hashtext(NEW.collection_id::text)) % 100;
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
            
            DROP TRIGGER IF EXISTS set_partition_key ON chunks;
            
            CREATE TRIGGER set_partition_key
            BEFORE INSERT ON chunks
            FOR EACH ROW
            EXECUTE FUNCTION compute_partition_key();
        """))
        
        raise


def downgrade() -> None:
    """
    Revert from GENERATED column back to trigger-based implementation.
    
    This is needed for compatibility with PostgreSQL < 12 or if issues arise.
    """
    
    conn = op.get_bind()
    
    logger.info("=" * 60)
    logger.info("Reverting to trigger-based partition_key computation")
    logger.info("=" * 60)
    
    # Check current implementation
    impl_status = check_current_implementation(conn)
    
    if impl_status["has_trigger"] and not impl_status["has_generated_column"]:
        logger.info("Already using trigger-based implementation. Nothing to do.")
        return
    
    # Special handling for partitioned tables
    if impl_status["is_partitioned"]:
        logger.info("=" * 60)
        logger.info("PARTITIONED TABLE DETECTED (DOWNGRADE)")
        logger.info("=" * 60)
        logger.info(
            "The chunks table is partitioned. "
            "Ensuring trigger-based implementation is in place."
        )
        
        # Ensure trigger exists
        conn.execute(text("""
            CREATE OR REPLACE FUNCTION compute_partition_key()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.partition_key := abs(hashtext(NEW.collection_id::text)) % 100;
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
            
            DROP TRIGGER IF EXISTS set_partition_key ON chunks;
            
            CREATE TRIGGER set_partition_key
            BEFORE INSERT ON chunks
            FOR EACH ROW
            EXECUTE FUNCTION compute_partition_key();
        """))
        
        logger.info("Trigger-based implementation set up for partitioned table")
        return
    
    # Check if we have data
    result = conn.execute(text("SELECT COUNT(*) FROM chunks"))
    row_count = result.scalar()
    
    if row_count == 0:
        logger.info("No data in chunks table, performing direct conversion...")
        
        # Drop and recreate as regular column
        conn.execute(text("""
            ALTER TABLE chunks DROP CONSTRAINT chunks_pkey
        """))
        
        conn.execute(text("""
            ALTER TABLE chunks DROP COLUMN partition_key
        """))
        
        conn.execute(text("""
            ALTER TABLE chunks 
            ADD COLUMN partition_key INTEGER NOT NULL DEFAULT 0
        """))
        
        conn.execute(text("""
            ALTER TABLE chunks ALTER COLUMN partition_key DROP DEFAULT
        """))
        
        conn.execute(text("""
            ALTER TABLE chunks ADD PRIMARY KEY (id, collection_id, partition_key)
        """))
        
    else:
        logger.info(f"Table has {row_count} rows, performing careful migration...")
        
        # Create a regular column and copy values
        conn.execute(text("""
            ALTER TABLE chunks 
            ADD COLUMN partition_key_regular INTEGER
        """))
        
        # Copy values from generated column
        conn.execute(text("""
            UPDATE chunks 
            SET partition_key_regular = partition_key
        """))
        
        # Make it NOT NULL
        conn.execute(text("""
            ALTER TABLE chunks 
            ALTER COLUMN partition_key_regular SET NOT NULL
        """))
        
        # Drop primary key, swap columns, recreate primary key
        conn.execute(text("""
            ALTER TABLE chunks DROP CONSTRAINT chunks_pkey
        """))
        
        conn.execute(text("""
            ALTER TABLE chunks DROP COLUMN partition_key
        """))
        
        conn.execute(text("""
            ALTER TABLE chunks 
            RENAME COLUMN partition_key_regular TO partition_key
        """))
        
        conn.execute(text("""
            ALTER TABLE chunks ADD PRIMARY KEY (id, collection_id, partition_key)
        """))
    
    # Create trigger function and trigger
    logger.info("Creating trigger function and trigger...")
    
    conn.execute(text("""
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
    """))
    
    # Verify the trigger works
    if row_count > 0:
        verify_partition_keys(conn)
    
    logger.info("Successfully reverted to trigger-based implementation")