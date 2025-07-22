# WebUI PostgreSQL Connection Fix

## Issue
The webui service was failing to start with the following error:
```
TypeError: 'AsyncAdapt_asyncpg_cursor' object does not support the context manager protocol
```

This occurred in `packages/shared/database/postgres_database.py` at line 154.

## Root Cause
The code was using a synchronous SQLAlchemy event listener (`@event.listens_for(Engine, "connect")`) with an async engine (asyncpg). The event handler was trying to use a synchronous cursor context manager which is incompatible with asyncpg's async connections.

## Solution
1. **Removed the synchronous event listener** from `postgres_database.py`
2. **Moved PostgreSQL settings to connection arguments** in `postgres.py`:
   - `statement_timeout`
   - `lock_timeout` 
   - `idle_in_transaction_session_timeout`

These settings are now applied via the `server_settings` in the connection arguments, which is the proper way to configure PostgreSQL connections with asyncpg.

## Files Modified

### 1. `/packages/shared/database/postgres_database.py`
- Removed the `set_postgresql_pragma` event listener
- Removed unused imports (`event` from sqlalchemy, `Engine`)

### 2. `/packages/shared/config/postgres.py`
- Updated `get_connect_args()` to include PostgreSQL settings in `server_settings`

## Result
The webui service should now start successfully with PostgreSQL. The database connection settings are properly applied when connections are created, compatible with the async engine architecture.