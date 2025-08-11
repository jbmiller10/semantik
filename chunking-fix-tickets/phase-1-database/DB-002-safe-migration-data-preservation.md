# DB-002: Create Safe Migration with Data Preservation

## Ticket Information
- **Priority**: BLOCKER
- **Estimated Time**: 3 hours
- **Dependencies**: DB-001 (Model fixes must be complete)
- **Risk Level**: CRITICAL - Current migration destroys data
- **Affected Files**:
  - `alembic/versions/ae558c9e183f_implement_100_direct_list_partitions.py`
  - Create new migration file

## Context

The current migration (`ae558c9e183f_implement_100_direct_list_partitions.py`) has a critical flaw: it uses `DROP TABLE IF EXISTS chunks CASCADE` which destroys all existing data without backup. This is unacceptable even in pre-production environments.

### Current Problems
1. Lines 85-86 drop tables without checking for data
2. No backup mechanism before destructive operations
3. CASCADE can affect dependent objects unexpectedly
4. Downgrade function recreates incompatible schema
5. No verification of data integrity after migration

## Requirements

1. Create a new migration that safely transitions to 100-partition structure
2. Implement automatic backup before any destructive operations
3. Provide data migration path from old to new structure
4. Ensure zero data loss during migration
5. Implement proper downgrade that restores original state
6. Add verification steps to confirm data integrity

## Technical Details

### New Migration Structure

```python
"""Safe implementation of 100-partition chunking tables

Revision ID: xxx_safe_100_partitions
Revises: ae558c9e183f
Create Date: 2024-xx-xx

This migration safely transitions from any previous chunk table structure
to the new 100-partition LIST partitioning scheme while preserving all data.
"""

def upgrade():
    """
    Safe upgrade process:
    1. Check if chunks table exists and has data
    2. Create timestamped backup if data exists
    3. Create new partitioned structure alongside old
    4. Migrate data in batches with progress tracking
    5. Verify data integrity
    6. Atomic table swap
    7. Clean up only after verification
    """
    
    # Step 1: Check current state
    # Query: SELECT COUNT(*) FROM chunks
    # If count > 0, proceed with backup
    
    # Step 2: Create backup with verification
    # backup_table = f"chunks_backup_{timestamp}"
    # CREATE TABLE {backup_table} AS SELECT * FROM chunks
    # Verify: SELECT COUNT(*) matches original
    
    # Step 3: Create new structure (non-destructive)
    # CREATE TABLE chunks_new (...) PARTITION BY LIST (partition_key)
    # Create all 100 partitions
    
    # Step 4: Migrate data in transaction batches
    # Use COPY or INSERT SELECT with batch size 10000
    # Track progress and handle errors
    
    # Step 5: Verify data integrity
    # Compare counts, spot-check records
    # Verify partition distribution
    
    # Step 6: Atomic swap
    # BEGIN;
    # ALTER TABLE chunks RENAME TO chunks_old;
    # ALTER TABLE chunks_new RENAME TO chunks;
    # COMMIT;
    
    # Step 7: Cleanup (only after verification period)
    # Keep chunks_old for configurable period (default 7 days)

def downgrade():
    """
    Safe downgrade process:
    1. Check for backup tables
    2. Restore from most recent backup
    3. Verify restoration
    4. Clean up partitioned tables
    """
```

### Detailed Implementation Steps

#### Step 1: Pre-Migration Check
```sql
-- Check if we have data to preserve
DO $$
DECLARE
    chunk_count INTEGER;
    backup_needed BOOLEAN;
BEGIN
    -- Check if chunks table exists
    IF EXISTS (SELECT 1 FROM information_schema.tables 
               WHERE table_name = 'chunks') THEN
        
        SELECT COUNT(*) INTO chunk_count FROM chunks;
        
        IF chunk_count > 0 THEN
            backup_needed := TRUE;
            RAISE NOTICE 'Found % chunks to migrate', chunk_count;
        END IF;
    END IF;
END $$;
```

#### Step 2: Backup Creation
```sql
-- Create backup with metadata
CREATE TABLE chunks_backup_YYYYMMDD_HHMMSS AS 
SELECT * FROM chunks;

-- Add metadata
COMMENT ON TABLE chunks_backup_YYYYMMDD_HHMMSS IS 
    'Backup created for 100-partition migration. Original count: X records';

-- Create backup verification record
INSERT INTO migration_backups (
    table_name, 
    backup_table, 
    record_count, 
    created_at
) VALUES (
    'chunks',
    'chunks_backup_YYYYMMDD_HHMMSS',
    (SELECT COUNT(*) FROM chunks),
    NOW()
);
```

#### Step 3: New Structure Creation
```sql
-- Create new partitioned table structure
CREATE TABLE chunks_new (
    id BIGSERIAL,
    collection_id VARCHAR(255) NOT NULL,
    partition_key INTEGER GENERATED ALWAYS AS 
        (mod(hashtext(collection_id::text), 100)) STORED,
    document_id VARCHAR(255) NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB,
    embedding_vector JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (id, collection_id, partition_key)
) PARTITION BY LIST (partition_key);

-- Create all 100 partitions
DO $$
BEGIN
    FOR i IN 0..99 LOOP
        EXECUTE format('
            CREATE TABLE chunks_new_part_%s 
            PARTITION OF chunks_new 
            FOR VALUES IN (%s)',
            i, i
        );
    END LOOP;
END $$;

-- Create indexes on each partition
DO $$
BEGIN
    FOR i IN 0..99 LOOP
        EXECUTE format('
            CREATE INDEX idx_chunks_new_part_%s_collection 
            ON chunks_new_part_%s (collection_id)',
            i, i
        );
        EXECUTE format('
            CREATE INDEX idx_chunks_new_part_%s_document 
            ON chunks_new_part_%s (document_id)',
            i, i
        );
    END LOOP;
END $$;
```

#### Step 4: Data Migration
```sql
-- Migrate data in batches with progress tracking
DO $$
DECLARE
    batch_size INTEGER := 10000;
    total_rows INTEGER;
    migrated_rows INTEGER := 0;
    batch_count INTEGER := 0;
BEGIN
    SELECT COUNT(*) INTO total_rows FROM chunks;
    
    -- Migrate in batches
    LOOP
        INSERT INTO chunks_new (
            collection_id, document_id, chunk_index,
            content, metadata, embedding_vector,
            created_at, updated_at
        )
        SELECT 
            collection_id, document_id, chunk_index,
            content, metadata, embedding_vector,
            created_at, updated_at
        FROM chunks
        ORDER BY id
        LIMIT batch_size
        OFFSET migrated_rows
        ON CONFLICT DO NOTHING;
        
        GET DIAGNOSTICS batch_count = ROW_COUNT;
        migrated_rows := migrated_rows + batch_count;
        
        -- Progress notification
        RAISE NOTICE 'Migrated % of % records (%%)',
            migrated_rows, total_rows, 
            (migrated_rows::FLOAT / total_rows * 100)::INT;
        
        EXIT WHEN batch_count < batch_size;
    END LOOP;
END $$;
```

#### Step 5: Verification
```sql
-- Verify migration completeness
DO $$
DECLARE
    old_count INTEGER;
    new_count INTEGER;
    sample_match BOOLEAN;
BEGIN
    SELECT COUNT(*) INTO old_count FROM chunks;
    SELECT COUNT(*) INTO new_count FROM chunks_new;
    
    IF old_count != new_count THEN
        RAISE EXCEPTION 'Count mismatch: old=%, new=%', 
            old_count, new_count;
    END IF;
    
    -- Spot check random records
    -- Verify partition distribution
    -- Check for data corruption
END $$;
```

## Acceptance Criteria

1. **Data Preservation**
   - [ ] All existing chunks preserved during migration
   - [ ] Backup created with timestamp before any changes
   - [ ] Backup verification confirms record count matches

2. **Safe Migration**
   - [ ] New structure created alongside old (non-destructive)
   - [ ] Data migrated in batches with progress tracking
   - [ ] Atomic swap ensures no downtime
   - [ ] Old table retained for safety period

3. **Verification Steps**
   - [ ] Record counts match between old and new
   - [ ] Partition distribution is relatively even
   - [ ] Spot checks confirm data integrity
   - [ ] No data corruption detected

4. **Rollback Capability**
   - [ ] Downgrade function can restore from backup
   - [ ] Backup tables are clearly labeled with timestamps
   - [ ] Migration metadata tracked for audit

## Testing Requirements

1. **Test with Empty Database**
   - Migration completes without errors
   - No backup created (no data to backup)

2. **Test with Existing Data**
   - Create 100,000 test chunks
   - Run migration
   - Verify all chunks preserved
   - Verify partition distribution

3. **Test Rollback**
   - Run upgrade
   - Run downgrade
   - Verify original state restored

4. **Test Error Scenarios**
   - Simulate failure during migration
   - Verify transaction rollback works
   - Verify no partial state

## Rollback Plan

1. If migration fails during execution:
   - Transaction automatically rolls back
   - Original chunks table unchanged
   - Retry after fixing issue

2. If issues found after migration:
   - Use downgrade() function
   - Restore from backup table
   - Investigate and fix issues

## Success Metrics

- Zero data loss during migration
- Migration completes in < 10 minutes for 1M records
- Even partition distribution (no partition > 2x average)
- All existing queries continue working
- Performance same or better after migration

## Notes for LLM Agent

- NEVER use DROP TABLE CASCADE without explicit user confirmation
- Always create timestamped backups before destructive operations
- Use transactions for atomic operations where possible
- Implement progress tracking for long-running operations
- Keep old tables for a safety period (don't immediately drop)
- Test migration on a copy of production data if possible
- Document all migration steps clearly in code comments