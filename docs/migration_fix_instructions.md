# Database Schema Fix Migration Guide

This guide documents the fixes applied to address database schema issues identified in the code review.

## Issues Fixed

### 1. UUID Type Inconsistency (HIGH PRIORITY)
- **Problem**: The chunks table used `UUID(as_uuid=True)` while other tables (collections, documents) used `String` for UUIDs
- **Solution**: Converted the chunks.id column from UUID to String type for consistency
- **Impact**: Ensures proper type conversion in joins between tables

### 2. Missing Indexes and Constraints (HIGH PRIORITY)
- **Problem**: Missing index on `chunks.embedding_vector_id` and unique constraint on `(collection_id, document_id, chunk_index)`
- **Solution**: 
  - Added index `ix_chunks_embedding_vector_id` for better Qdrant join performance
  - Added unique constraint `uq_chunks_collection_document_index` to prevent duplicate chunks

### 3. Configurable Partitions (MEDIUM PRIORITY)
- **Problem**: Hardcoded 16 partitions for the chunks table
- **Solution**: Made partition count configurable via `CHUNK_PARTITION_COUNT` environment variable (default: 16)

## Migration Files

1. **Modified**: `alembic/versions/52db15bd2686_add_chunking_tables_with_partitioning.py`
   - Updated to use `CHUNK_PARTITION_COUNT` environment variable

2. **Created**: `alembic/versions/6596eda04faa_fix_chunk_table_schema_issues.py`
   - Converts UUID to String type
   - Adds missing indexes
   - Adds unique constraint

## Configuration Changes

Updated `packages/shared/config/postgres.py` to include:
```python
CHUNK_PARTITION_COUNT: int = Field(default=16, description="Number of partitions for chunks table")
```

## Model Changes

Updated `packages/shared/database/models.py`:
- Changed `Chunk.id` from `UUID(as_uuid=True)` to `String`
- Added new indexes and unique constraint to `__table_args__`

## How to Apply the Migration

1. **Set Environment Variable** (optional):
   ```bash
   export CHUNK_PARTITION_COUNT=16  # Or your desired partition count
   ```

2. **Run the Migration**:
   ```bash
   poetry run alembic upgrade head
   ```

3. **Verify the Migration**:
   ```bash
   poetry run python test_migration_fix.py
   ```

## Important Notes

### Data Preservation
- The migration preserves all existing data
- UUID values are converted to strings during migration
- The downgrade path converts strings back to UUIDs

### Partition Count
- The partition count must be set before running the initial chunking migration
- Changing the partition count after data exists requires recreating the table
- Default is 16 partitions, which is suitable for most use cases

### Performance Impact
- The new indexes improve query performance for:
  - Qdrant vector lookups (embedding_vector_id index)
  - Chunk uniqueness checks (unique constraint)
- The String UUID type ensures consistent join performance

### Backward Compatibility
- The migration includes proper downgrade paths
- All changes are reversible if needed

## Testing

Run the provided test script to verify:
1. UUID type consistency across tables
2. Proper creation of indexes
3. Unique constraint enforcement
4. Correct partition count

```bash
poetry run python test_migration_fix.py
```

## Rollback Instructions

If you need to rollback the changes:

```bash
poetry run alembic downgrade 52db15bd2686
```

This will:
- Convert String UUIDs back to UUID type
- Remove the added indexes
- Remove the unique constraint
- Preserve all data