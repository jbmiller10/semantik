# PostgreSQL Enum Type Creation Fix

## Problem
When running tests in parallel with pytest-xdist, multiple test workers attempt to create the same PostgreSQL enum types concurrently. This causes duplicate key violations with the error:
```
duplicate key value violates unique constraint 'pg_type_typname_nsp_index'
```

## Solution
The `enum_helper.py` module provides thread-safe functions to create enum types by:

1. **Using PostgreSQL advisory locks** to serialize enum type creation across different database connections
2. **Checking for enum existence** before attempting to create them
3. **Gracefully handling race conditions** where another process may have created the enum between the check and creation

## Implementation Details

### Key Features:
- **Advisory Lock (12345)**: Ensures only one process creates enum types at a time
- **Idempotent Operations**: Safe to call multiple times without side effects
- **Error Handling**: Catches and ignores "already exists" errors from race conditions
- **Both Sync and Async**: Supports both synchronous and asynchronous database connections

### Enum Types Created:
- `document_status`: pending, processing, completed, failed, deleted
- `permission_type`: read, write, admin
- `collection_status`: pending, ready, processing, error, degraded
- `operation_type`: index, append, reindex, remove_source, delete
- `operation_status`: pending, processing, completed, failed, cancelled

## Usage

The helper functions are automatically called in test fixtures before creating tables:

```python
from tests.database.enum_helper import create_enum_types_if_not_exist

async with engine.begin() as conn:
    # Create enum types first (handles concurrent creation)
    await create_enum_types_if_not_exist(conn)
    # Now create tables
    await conn.run_sync(Base.metadata.create_all)
```

## Files Modified

- `/tests/conftest.py`: Main test configuration
- `/tests/integration/conftest.py`: Integration test configuration
- `/tests/unit/test_models.py`: Unit test configuration
- `/tests/database/enum_helper.py`: The helper module (new file)

## Testing

The solution has been tested with:
- Single worker execution
- Concurrent execution simulation
- Idempotency verification
- Race condition handling

This ensures reliable parallel test execution without enum type conflicts.