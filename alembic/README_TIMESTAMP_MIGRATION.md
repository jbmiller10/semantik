# Future Timestamp Migration Plan

## Current State

All timestamp columns in the database currently use `String` type to store ISO 8601 formatted timestamps (e.g., "2023-01-01T00:00:00+00:00"). This approach was inherited from the original SQLite implementation.

## Issues with String Timestamps

1. **Performance**: String comparisons for date queries are slower than native DateTime operations
2. **Storage**: Strings take more space than binary datetime representations
3. **Functionality**: Limited ability to use database-native date functions

## Migration Strategy

When ready to migrate to proper DateTime columns:

1. **Create Migration Script**:
   ```bash
   alembic revision -m "Convert string timestamps to datetime"
   ```

2. **Migration Steps**:
   - Add new DateTime columns with temporary names
   - Copy and parse existing string timestamps to new columns
   - Drop old string columns
   - Rename new columns to original names

3. **Example Migration Code**:
   ```python
   from alembic import op
   import sqlalchemy as sa
   from datetime import datetime
   
   def upgrade():
       # Add temporary DateTime columns
       op.add_column('jobs', sa.Column('created_at_new', sa.DateTime))
       op.add_column('jobs', sa.Column('updated_at_new', sa.DateTime))
       
       # Migrate data
       connection = op.get_bind()
       result = connection.execute('SELECT id, created_at, updated_at FROM jobs')
       for row in result:
           created = datetime.fromisoformat(row.created_at.replace('Z', '+00:00'))
           updated = datetime.fromisoformat(row.updated_at.replace('Z', '+00:00'))
           connection.execute(
               f"UPDATE jobs SET created_at_new = ?, updated_at_new = ? WHERE id = ?",
               (created, updated, row.id)
           )
       
       # Drop old columns and rename new ones
       op.drop_column('jobs', 'created_at')
       op.drop_column('jobs', 'updated_at')
       op.alter_column('jobs', 'created_at_new', new_column_name='created_at')
       op.alter_column('jobs', 'updated_at_new', new_column_name='updated_at')
   ```

4. **Update Application Code**:
   - Modify all timestamp handling to use datetime objects
   - Update SQLAlchemy models to use DateTime columns
   - Test thoroughly with existing data

## Considerations

- **Backup**: Always backup the database before migration
- **Testing**: Test migration on a copy of production data
- **Rollback Plan**: Prepare a downgrade migration
- **Coordination**: Ensure all services are updated to handle DateTime objects