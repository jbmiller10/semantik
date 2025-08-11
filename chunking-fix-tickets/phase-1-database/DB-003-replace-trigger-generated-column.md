# DB-003: Replace Trigger with Generated Column

## Ticket Information
- **Priority**: HIGH
- **Estimated Time**: 1.5 hours
- **Dependencies**: DB-001, DB-002 (must complete model and migration fixes first)
- **Risk Level**: MEDIUM - Performance improvement but requires PostgreSQL 12+
- **Affected Files**:
  - New migration file to create
  - `packages/shared/database/partition_utils.py`

## Context

The current implementation uses a database trigger to compute `partition_key` on every INSERT:

```sql
CREATE TRIGGER compute_partition_key_trigger
BEFORE INSERT ON chunks
FOR EACH ROW
EXECUTE FUNCTION compute_partition_key();
```

This adds 2-3ms overhead per insert and can cause issues under high concurrency. PostgreSQL 12+ supports GENERATED columns which are more efficient.

### Current Problems
1. Trigger overhead on every insert (2-3ms per row)
2. Additional function call overhead
3. Potential race conditions under high concurrency
4. Harder to debug trigger-related issues
5. No query optimizer benefits

### Benefits of Generated Columns
1. Computed at storage level (more efficient)
2. Query optimizer aware of generation expression
3. No function call overhead
4. Automatic dependency tracking
5. Cleaner, more maintainable

## Requirements

1. Create migration to convert trigger-based partition_key to GENERATED column
2. Ensure backward compatibility for PostgreSQL 11 environments
3. Remove trigger and function after successful conversion
4. Verify partition key values remain consistent
5. Update any code that manually sets partition_key

## Technical Details

### Migration Implementation

```python
"""Convert partition_key from trigger to generated column

Revision ID: xxx_generated_partition_key
Revises: xxx_safe_100_partitions
Create Date: 2024-xx-xx

Requires PostgreSQL 12+ for GENERATED columns.
Falls back to trigger for older versions.
"""

def upgrade():
    # Step 1: Check PostgreSQL version
    # SELECT version()
    # Parse and check if >= 12
    
    # Step 2: If PG >= 12, convert to GENERATED
    # ALTER TABLE chunks 
    # ALTER COLUMN partition_key 
    # SET GENERATED ALWAYS AS (mod(hashtext(collection_id::text), 100)) STORED
    
    # Step 3: Drop trigger and function
    # DROP TRIGGER IF EXISTS compute_partition_key_trigger ON chunks
    # DROP FUNCTION IF EXISTS compute_partition_key()
    
    # Step 4: Verify existing values unchanged
    # SELECT COUNT(*) FROM chunks 
    # WHERE partition_key != mod(hashtext(collection_id::text), 100)

def downgrade():
    # Revert to trigger-based approach
    # Remove GENERATED constraint
    # Recreate trigger and function
```

### Detailed Steps

#### Step 1: PostgreSQL Version Check
```sql
DO $$
DECLARE
    pg_version INTEGER;
BEGIN
    -- Get major version number
    SELECT current_setting('server_version_num')::INTEGER / 10000
    INTO pg_version;
    
    IF pg_version < 12 THEN
        RAISE NOTICE 'PostgreSQL % detected. Keeping trigger-based partition_key', pg_version;
        -- Exit without changes
    ELSE
        RAISE NOTICE 'PostgreSQL % detected. Converting to GENERATED column', pg_version;
        -- Proceed with conversion
    END IF;
END $$;
```

#### Step 2: Convert to Generated Column
```sql
-- First, ensure all existing values are correct
UPDATE chunks
SET partition_key = mod(hashtext(collection_id::text), 100)
WHERE partition_key IS NULL 
   OR partition_key != mod(hashtext(collection_id::text), 100);

-- Convert column to GENERATED
-- Note: This requires dropping and recreating the column in some PG versions
BEGIN;
    -- Save constraints
    CREATE TEMP TABLE temp_constraints AS
    SELECT conname, pg_get_constraintdef(oid) as condef
    FROM pg_constraint
    WHERE conrelid = 'chunks'::regclass
    AND conname LIKE '%partition_key%';
    
    -- Drop constraints involving partition_key
    ALTER TABLE chunks DROP CONSTRAINT chunks_pkey CASCADE;
    
    -- Drop and recreate column as GENERATED
    ALTER TABLE chunks DROP COLUMN partition_key;
    ALTER TABLE chunks 
    ADD COLUMN partition_key INTEGER 
    GENERATED ALWAYS AS (mod(hashtext(collection_id::text), 100)) STORED;
    
    -- Recreate primary key
    ALTER TABLE chunks 
    ADD PRIMARY KEY (id, collection_id, partition_key);
    
    -- Recreate other constraints
    -- (execute saved constraints from temp table)
COMMIT;
```

#### Step 3: Remove Trigger
```sql
-- Drop trigger first (if exists)
DROP TRIGGER IF EXISTS compute_partition_key_trigger ON chunks;

-- Drop function (if no longer used)
DROP FUNCTION IF EXISTS compute_partition_key();

-- Clean up any related objects
DROP FUNCTION IF EXISTS compute_partition_key_v2();  -- Any variants
```

#### Step 4: Verification
```sql
-- Verify all partition keys are correct
DO $$
DECLARE
    incorrect_count INTEGER;
BEGIN
    SELECT COUNT(*)
    INTO incorrect_count
    FROM chunks
    WHERE partition_key != mod(hashtext(collection_id::text), 100);
    
    IF incorrect_count > 0 THEN
        RAISE EXCEPTION 'Found % incorrect partition_key values', incorrect_count;
    ELSE
        RAISE NOTICE 'All partition_key values verified correct';
    END IF;
END $$;

-- Verify INSERT still works
INSERT INTO chunks (collection_id, document_id, chunk_index, content)
VALUES ('test-collection', 'test-doc', 0, 'test content');

-- Check partition_key was auto-generated
SELECT partition_key, mod(hashtext('test-collection'::text), 100) as expected
FROM chunks
WHERE collection_id = 'test-collection'
ORDER BY id DESC
LIMIT 1;
```

### Backward Compatibility Function

```python
# packages/shared/database/partition_utils.py

async def ensure_partition_key_generation(session: AsyncSession) -> str:
    """
    Ensures partition_key generation is properly configured.
    Returns the method being used: 'generated' or 'trigger'
    """
    # Check PostgreSQL version
    result = await session.execute(text(
        "SELECT current_setting('server_version_num')::INTEGER / 10000 as version"
    ))
    pg_version = result.scalar()
    
    if pg_version >= 12:
        # Check if using GENERATED column
        result = await session.execute(text("""
            SELECT attgenerated 
            FROM pg_attribute 
            WHERE attrelid = 'chunks'::regclass 
            AND attname = 'partition_key'
        """))
        
        if result.scalar() == 's':  # 's' means STORED generated column
            return 'generated'
        else:
            # Should migrate to generated
            logger.warning("PostgreSQL 12+ but still using trigger for partition_key")
            return 'trigger'
    else:
        # Must use trigger for PG < 12
        return 'trigger'
```

## Acceptance Criteria

1. **For PostgreSQL 12+**
   - [ ] partition_key is a GENERATED column
   - [ ] No trigger exists for partition key computation
   - [ ] All INSERTs work without setting partition_key
   - [ ] Performance improved (measure insert time)

2. **For PostgreSQL 11**
   - [ ] Trigger remains in place
   - [ ] System continues working as before
   - [ ] Clear log message about version limitation

3. **Data Integrity**
   - [ ] All existing partition_key values unchanged
   - [ ] New inserts get correct partition_key
   - [ ] Partition distribution remains even

4. **Code Updates**
   - [ ] No application code manually sets partition_key
   - [ ] Repository methods don't reference partition_key in INSERTs
   - [ ] partition_key treated as read-only in application

## Testing Requirements

1. **Functional Tests**
   ```python
   async def test_generated_partition_key():
       # Insert without setting partition_key
       chunk = Chunk(
           collection_id="test-coll",
           document_id="test-doc",
           chunk_index=0,
           content="test"
       )
       session.add(chunk)
       await session.commit()
       
       # Verify partition_key was generated
       assert chunk.partition_key is not None
       expected = hash("test-coll") % 100
       assert chunk.partition_key == expected
   ```

2. **Performance Tests**
   ```python
   async def test_insert_performance():
       # Measure time for 10,000 inserts
       start = time.time()
       
       chunks = [
           Chunk(
               collection_id=f"coll-{i % 100}",
               document_id=f"doc-{i}",
               chunk_index=0,
               content=f"content-{i}"
           )
           for i in range(10000)
       ]
       
       session.add_all(chunks)
       await session.commit()
       
       duration = time.time() - start
       
       # Should be faster than trigger (< 30 seconds)
       assert duration < 30
       
       # Verify even distribution
       distribution = await check_partition_distribution()
       assert distribution.max_skew < 1.5
   ```

3. **Version Compatibility Tests**
   - Test on PostgreSQL 11 (trigger should remain)
   - Test on PostgreSQL 12+ (should use GENERATED)
   - Test on PostgreSQL 14+ (latest features)

## Rollback Plan

If issues occur after migration:

1. **Revert to Trigger**
   ```sql
   -- Remove GENERATED constraint
   ALTER TABLE chunks ALTER COLUMN partition_key DROP EXPRESSION;
   
   -- Recreate trigger
   CREATE OR REPLACE FUNCTION compute_partition_key()...
   CREATE TRIGGER compute_partition_key_trigger...
   ```

2. **Verify Functionality**
   - Test inserts work
   - Verify partition_key computation
   - Check application compatibility

## Success Metrics

- Insert performance improved by 10-20%
- Zero errors in partition_key computation
- Successful operation on both PG 11 and PG 12+
- Clean migration with no data changes
- Reduced complexity in codebase

## Notes for LLM Agent

- Check PostgreSQL version before attempting GENERATED column conversion
- PostgreSQL 11 does NOT support GENERATED columns - keep trigger
- Don't drop the trigger until you verify GENERATED column works
- Test thoroughly on a non-production database first
- The expression `mod(hashtext(collection_id::text), 100)` must remain exactly the same
- Some PostgreSQL versions may require dropping and recreating the column
- Be prepared to handle the primary key constraint during column modification