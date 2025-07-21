# TICKET-001: Fix PostgreSQL Migration Compatibility

**Type:** Bug / Critical Infrastructure
**Priority:** Critical
**Blocks:** All testing and deployment
**Component:** Database / Migrations

## Problem Statement

The Alembic migration files were designed for SQLite and fail when running against PostgreSQL. The primary issue is that PostgreSQL requires explicit creation of ENUM types before they can be used in column definitions, while SQLite handles enums differently.

### Current Error
```
sqlalchemy.exc.ProgrammingError: (psycopg2.errors.UndefinedObject) type "collection_status" does not exist
```

## Root Cause Analysis

1. The migration file `91784cc819aa_add_operations_and_supporting_tables_.py` uses SQLAlchemy Enum columns without creating the PostgreSQL ENUM types first
2. SQLite doesn't support CREATE TYPE, so the migration was written to work with SQLite's CHECK constraints
3. When running on PostgreSQL, the migration fails because the enum types don't exist

## Technical Requirements

### Required ENUM Types
```sql
CREATE TYPE collection_status AS ENUM ('pending', 'ready', 'processing', 'error', 'degraded');
CREATE TYPE operation_type AS ENUM ('index', 'append', 'reindex', 'remove_source', 'delete');
CREATE TYPE operation_status AS ENUM ('pending', 'processing', 'completed', 'failed', 'cancelled');
```

## Implementation Steps

1. **Update Migration File** (`alembic/versions/91784cc819aa_add_operations_and_supporting_tables_.py`):
   - Add dialect detection to check if we're using PostgreSQL or SQLite
   - For PostgreSQL: Create ENUM types before using them
   - For SQLite: Keep existing behavior
   - Use `postgresql.ENUM` with `create_type=False` for PostgreSQL
   - Use generic `sa.Enum` for SQLite

2. **Example Implementation**:
```python
def upgrade() -> None:
    # Get the database dialect
    bind = op.get_bind()
    dialect_name = bind.dialect.name
    
    if dialect_name == 'postgresql':
        # Create enum types for PostgreSQL
        op.execute("""
            DO $$ BEGIN
                CREATE TYPE collection_status AS ENUM ('pending', 'ready', 'processing', 'error', 'degraded');
            EXCEPTION
                WHEN duplicate_object THEN null;
            END $$;
        """)
        # ... repeat for other enums
        
        # Use postgresql.ENUM with create_type=False
        status_enum = sa.Enum(..., name='collection_status', create_type=False)
    else:
        # Use generic Enum for SQLite
        status_enum = sa.Enum(..., name='collection_status')
```

3. **Update downgrade() function** to properly drop enum types in PostgreSQL

4. **Test Migration**:
   - Test on fresh PostgreSQL database
   - Test on fresh SQLite database
   - Test upgrade and downgrade paths

## Testing Requirements

1. Create automated tests that run migrations against both PostgreSQL and SQLite
2. Verify all enum columns work correctly in both databases
3. Test that existing SQLite deployments can still upgrade
4. Test that new PostgreSQL deployments work from scratch

## Acceptance Criteria

- [ ] Migrations run successfully on fresh PostgreSQL database
- [ ] Migrations run successfully on fresh SQLite database
- [ ] All enum columns function correctly in both databases
- [ ] No regression for existing SQLite deployments
- [ ] Downgrade migrations work properly

## Additional Notes

- Consider creating a migration utilities module for handling database-specific logic
- Document the PostgreSQL deployment requirements in the README
- Update docker-compose files to ensure PostgreSQL is properly initialized

## References

- Current migration file: `/alembic/versions/91784cc819aa_add_operations_and_supporting_tables_.py`
- PostgreSQL ENUM documentation: https://www.postgresql.org/docs/current/datatype-enum.html
- SQLAlchemy dialect-specific types: https://docs.sqlalchemy.org/en/20/dialects/