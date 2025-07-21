# TICKET-002: Fix SQLAlchemy Async Engine Initialization Error

**Type:** Bug
**Priority:** Critical
**Blocks:** Application startup with PostgreSQL
**Component:** Database Connection / PostgreSQL

## Problem Statement

The webui service fails to start with the following error:
```
TypeError: sqlalchemy.ext.asyncio.engine.create_async_engine() got multiple values for keyword argument 'echo'
```

This prevents the application from starting when using PostgreSQL as the database backend.

## Root Cause Analysis

The error occurs in `/packages/shared/database/postgres_database.py` at the `create_async_engine` call. The 'echo' parameter is being passed multiple times, likely through:
1. Explicit parameter: `echo=self.config.DB_ECHO`
2. Possibly included in `connect_args` or when using the spread operator with `get_pool_kwargs()`

## Current Code Location

File: `/packages/shared/database/postgres_database.py`
Function: `PostgresConnectionManager.initialize()`
Lines: ~40-65

## Implementation Steps

1. **Immediate Fix**:
   - Clean the `connect_args` dictionary to ensure no duplicate parameters
   - Remove any 'echo' or 'echo_pool' keys from dictionaries before passing to create_async_engine

```python
# Get connect_args and ensure no duplicate parameters
connect_args = self.config.get_connect_args()
connect_args.pop('echo', None)
connect_args.pop('echo_pool', None)

# For pool kwargs
pool_kwargs = self.config.get_pool_kwargs()
pool_kwargs.pop('echo', None)
pool_kwargs.pop('echo_pool', None)
```

2. **Verify Parameter Sources**:
   - Check `PostgresConfig.get_connect_args()` method
   - Check `PostgresConfig.get_pool_kwargs()` method
   - Ensure neither returns 'echo' or 'echo_pool' parameters

3. **Update Engine Creation**:
   - Instead of using spread operator, explicitly pass known parameters
   - This prevents unexpected parameter conflicts

```python
self._engine = create_async_engine(
    self.config.async_database_url,
    echo=self.config.DB_ECHO,
    echo_pool=self.config.DB_ECHO_POOL,
    connect_args=connect_args,
    pool_size=pool_kwargs.get("pool_size", 20),
    max_overflow=pool_kwargs.get("max_overflow", 40),
    pool_timeout=pool_kwargs.get("pool_timeout", 30),
    pool_recycle=pool_kwargs.get("pool_recycle", 3600),
    pool_pre_ping=pool_kwargs.get("pool_pre_ping", True),
)
```

## Testing Requirements

1. **Unit Tests**:
   - Test PostgresConnectionManager initialization with various configurations
   - Verify no duplicate parameters are passed to create_async_engine

2. **Integration Tests**:
   - Start the application with PostgreSQL configuration
   - Verify successful database connection
   - Test with different DB_ECHO settings

3. **Docker Testing**:
   - Rebuild Docker image without cache: `docker-compose build --no-cache webui`
   - Verify the fix works in containerized environment

## Acceptance Criteria

- [ ] Application starts successfully with PostgreSQL
- [ ] No TypeError on async engine initialization
- [ ] Database connection works properly
- [ ] Logging (echo) configuration works as expected
- [ ] No regression for other database configurations

## Additional Considerations

1. **Docker Caching Issue**: 
   - Current Docker builds are caching the old code
   - Document the need for `--no-cache` builds when fixing critical issues
   - Consider adding a make target: `make docker-rebuild-fresh`

2. **Configuration Validation**:
   - Add validation to ensure configuration methods don't return conflicting parameters
   - Consider using Pydantic models for stronger typing

3. **Error Handling**:
   - Improve error messages for configuration issues
   - Add specific exception handling for engine initialization

## References

- Error location: `/packages/shared/database/postgres_database.py:44`
- Configuration: `/packages/shared/config/postgres.py`
- SQLAlchemy async engine docs: https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html