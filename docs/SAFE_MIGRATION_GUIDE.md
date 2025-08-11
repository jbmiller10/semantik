# Safe Migration Guide: 100-Partition Structure with Data Preservation

## Overview

This guide documents the safe migration from the existing chunks table structure to a 100-partition LIST-based structure, ensuring zero data loss and minimal downtime.

## Migration File

**File**: `alembic/versions/8547ff31e80c_safe_100_partitions_with_data_preservation.py`  
**Revision ID**: `8547ff31e80c`  
**Revises**: `ae558c9e183f`

## Key Safety Features Implemented

### 1. Pre-Migration Data Check
- Checks if chunks table exists
- Counts existing records before any changes
- Skips backup/migration if no data exists

### 2. Timestamped Backup Creation
- Creates backup tables with format: `chunks_backup_YYYYMMDD_HHMMSS`
- Backs up all partition tables if they exist
- Records backup metadata in `migration_backups` table
- Sets 7-day retention period by default

### 3. Backup Verification
- Verifies record count matches between original and backup
- Performs spot checks on sample records
- Aborts migration if backup verification fails

### 4. Non-Destructive Migration
- Creates new structure (`chunks_new`) alongside existing structure
- No data is deleted until migration is verified
- Original table renamed, not dropped

### 5. Batch Processing
- Migrates data in configurable batches (default: 10,000 records)
- Progress tracking with percentage completion
- Prevents memory overflow on large datasets

### 6. Data Integrity Verification
- Verifies record counts match after migration
- Checks partition distribution
- Validates collection integrity
- Ensures no data is lost

### 7. Atomic Table Swap
- Minimizes downtime with atomic rename operations
- Creates all indexes and constraints before swap
- Maintains referential integrity

### 8. Rollback Capability
- Downgrade function can restore from backup
- Original table retained with timestamp
- Full restoration possible within retention period

## How to Apply the Migration

### Prerequisites

1. **Backup your database** (additional safety measure):
   ```bash
   pg_dump -h localhost -U your_user -d your_db > backup_before_migration.sql
   ```

2. **Check current migration status**:
   ```bash
   poetry run alembic current
   ```

### Apply Migration

1. **Review the migration** (dry run):
   ```bash
   poetry run alembic show 8547ff31e80c
   ```

2. **Apply the migration**:
   ```bash
   poetry run alembic upgrade 8547ff31e80c
   ```

3. **Monitor the logs** - The migration provides detailed progress information:
   - Data detection and count
   - Backup creation confirmation
   - Migration progress (batch by batch)
   - Verification results
   - Completion status

### Post-Migration Verification

1. **Check migration status**:
   ```bash
   poetry run alembic current
   ```

2. **Verify data integrity**:
   ```sql
   -- Check total record count
   SELECT COUNT(*) FROM chunks;
   
   -- Check partition distribution
   SELECT partition_key, COUNT(*) as chunk_count
   FROM chunks
   GROUP BY partition_key
   ORDER BY chunk_count DESC
   LIMIT 10;
   
   -- Check collections are intact
   SELECT COUNT(DISTINCT collection_id) FROM chunks;
   ```

3. **Use the backup manager utility**:
   ```bash
   python alembic/migrations_utils/backup_manager.py \
     --database-url postgresql://user:pass@localhost/db \
     --action status
   ```

## Backup Management

### List All Backups
```bash
python alembic/migrations_utils/backup_manager.py \
  --database-url postgresql://user:pass@localhost/db \
  --action list
```

### Verify Backup Integrity
```bash
python alembic/migrations_utils/backup_manager.py \
  --database-url postgresql://user:pass@localhost/db \
  --action verify \
  --table-name chunks_backup_20250811_120000
```

### Extend Backup Retention
```bash
python alembic/migrations_utils/backup_manager.py \
  --database-url postgresql://user:pass@localhost/db \
  --action extend \
  --table-name chunks_backup_20250811_120000 \
  --days 14
```

### Clean Up Expired Backups
```bash
# Dry run first
python alembic/migrations_utils/backup_manager.py \
  --database-url postgresql://user:pass@localhost/db \
  --action cleanup \
  --dry-run

# Actual cleanup
python alembic/migrations_utils/backup_manager.py \
  --database-url postgresql://user:pass@localhost/db \
  --action cleanup
```

## Rollback Procedure

If you need to rollback the migration:

1. **Downgrade using Alembic**:
   ```bash
   poetry run alembic downgrade ae558c9e183f
   ```

2. **Manual restoration** (if needed):
   ```sql
   -- Find available backups
   SELECT * FROM migration_backups 
   WHERE migration_revision = '8547ff31e80c'
   ORDER BY created_at DESC;
   
   -- Restore from specific backup
   DROP TABLE IF EXISTS chunks CASCADE;
   CREATE TABLE chunks AS 
   TABLE chunks_backup_20250811_120000 WITH DATA;
   
   -- Recreate constraints
   ALTER TABLE chunks 
   ADD CONSTRAINT chunks_collection_id_fkey 
   FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE;
   ```

## Performance Considerations

### Expected Migration Times

Based on batch size of 10,000 records:
- 100K records: ~1-2 minutes
- 1M records: ~10-15 minutes  
- 10M records: ~100-150 minutes

### Resource Usage

- **CPU**: Moderate (hash computation for partition keys)
- **Memory**: Low (batch processing limits memory usage)
- **Disk I/O**: High during migration (reading and writing data)
- **Disk Space**: Requires 2x original data size temporarily

## Troubleshooting

### Migration Fails During Verification

**Symptom**: Migration aborts with "Migration verification failed!"

**Solution**:
1. Check logs for specific verification failure
2. Verify source data integrity
3. Check disk space availability
4. Retry migration after addressing issue

### Out of Disk Space

**Symptom**: Migration fails with disk space error

**Solution**:
1. Free up disk space (need ~2x current chunks table size)
2. Clean up old backups if safe to do so
3. Consider migrating in smaller batches

### Slow Migration Performance

**Symptom**: Migration taking excessive time

**Solution**:
1. Check for blocking queries: `SELECT * FROM pg_stat_activity WHERE state != 'idle';`
2. Increase `work_mem` temporarily: `SET work_mem = '256MB';`
3. Ensure adequate system resources

## Migration Testing

### Test in Development First

1. **Clone production data** (subset):
   ```sql
   -- Export sample data
   COPY (SELECT * FROM chunks LIMIT 100000) 
   TO '/tmp/chunks_sample.csv' CSV HEADER;
   ```

2. **Test migration**:
   ```bash
   # In development environment
   poetry run alembic upgrade 8547ff31e80c
   ```

3. **Verify results**:
   ```bash
   poetry run pytest tests/test_safe_migration.py -v
   ```

## Monitoring After Migration

### Check Partition Health
```sql
SELECT * FROM partition_health 
WHERE partition_status != 'NORMAL';
```

### Monitor Distribution
```sql
SELECT 
  COUNT(DISTINCT partition_key) as partitions_used,
  AVG(cnt) as avg_per_partition,
  MAX(cnt) as max_per_partition,
  MIN(cnt) as min_per_partition,
  STDDEV(cnt) as stddev
FROM (
  SELECT partition_key, COUNT(*) as cnt
  FROM chunks
  GROUP BY partition_key
) sub;
```

## Best Practices

1. **Always test in development first**
2. **Perform migrations during low-traffic periods**
3. **Monitor system resources during migration**
4. **Keep backups for at least 7 days after migration**
5. **Document any issues encountered for future reference**
6. **Verify application functionality after migration**

## Support

If you encounter issues:

1. Check migration logs for detailed error messages
2. Verify backup integrity before attempting rollback
3. Use backup manager utility for backup operations
4. Review this guide's troubleshooting section

## Summary

This migration implements comprehensive safety measures:
- ✅ Zero data loss guarantee
- ✅ Automatic backup with verification
- ✅ Batch processing for large datasets
- ✅ Progress tracking and logging
- ✅ Atomic operations for minimal downtime
- ✅ Complete rollback capability
- ✅ Retention period for backups
- ✅ Monitoring and verification tools

The migration prioritizes data safety over speed, ensuring your data remains intact throughout the process.