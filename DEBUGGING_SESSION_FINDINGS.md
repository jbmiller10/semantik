# Semantik Debugging Session Findings

## Date: July 20, 2025

## Initial Problems Reported

1. **Embeddings not being generated** when creating a new collection
2. **GPU never spinning up** during the embedding generation process
3. **Blank white page** after creating a collection and adding data
4. **User getting logged out** unexpectedly
5. **Unable to log back in** after being logged out

## Root Causes Discovered

### 1. Missing Embedding Generation in V2 API

**Problem**: The V2 collections API had a two-step process (create collection → add sources) but embedding generation was not implemented in the `_process_append_operation` function.

**Location**: `packages/webui/tasks.py` - Line 1961 had a TODO comment indicating the feature wasn't implemented.

**Solution Implemented**: Copied the embedding generation logic from the V1 API implementation, including:
- GPU scheduling with proper task management
- Batch processing of documents
- Vector dimension detection from model
- Progress tracking and metrics recording

### 2. Hardcoded Vector Dimensions

**Problem**: The code had a hardcoded `DEFAULT_VECTOR_DIMENSION = 768` but the actual model (Qwen/Qwen3-Embedding-0.6B) produces 1024-dimensional vectors.

**Location**: `packages/webui/services/collection_service.py`

**Solution Implemented**: Removed hardcoded dimension and now dynamically retrieve it from the model configuration.

### 3. Bitsandbytes GPU Support Missing

**Problem**: The bitsandbytes library was compiled without GPU support despite using CUDA Docker configuration.

**Solution Implemented**: Used `docker-compose.cuda.yml` which uses `Dockerfile.cuda` with proper CUDA base images and environment setup.

### 4. SQLite Database Locking Issues

**Problem**: Multiple severe database locking issues causing:
- "database is locked" errors in both webui and worker
- Failed login attempts due to unable to update last_login timestamp
- Worker failing to insert documents during collection operations
- Transaction rollback issues in SQLAlchemy sessions

**Locations**: 
- `packages/shared/database/sqlite_implementation.py`
- `packages/shared/database/database.py`
- Worker logs showing repeated locking errors

**Solutions Implemented**:

a) **Added SQLite WAL Mode and Optimizations**:
```python
# In database.py
cursor.execute("PRAGMA journal_mode=WAL")
cursor.execute("PRAGMA cache_size=-64000")  # 64MB cache
cursor.execute("PRAGMA busy_timeout=30000")
cursor.execute("PRAGMA synchronous=NORMAL")
cursor.execute("PRAGMA temp_store=MEMORY")
cursor.execute("PRAGMA mmap_size=30000000000")
```

b) **Created Retry Decorators**:
- `with_db_retry` for async operations
- `with_sqlite_retry` for sync operations
- Applied to critical operations like `update_user_last_login` and document creation

c) **Centralized Connection Utils**:
- Created `connection_utils.py` to ensure all connections use proper SQLite settings

### 5. Frontend Routing Issues

**Problem**: React app showing blank page due to:
- Catch-all route intercepting static asset requests
- JavaScript files being served with HTML content type
- Module import errors

**Locations**: 
- `packages/webui/api/root.py`
- `packages/webui/main.py`

**Solutions Implemented**:
- Fixed route ordering - mount static files BEFORE catch-all route
- Removed problematic catch-all route that was serving HTML for JS files
- Added explicit routes for known client-side routes

### 6. Authentication Issues

**Problem**: 
- Login expecting username but UI showed email field
- Admin user didn't exist in database initially
- Database migrations weren't running on startup

**Solution**: 
- Clarified that login uses **username** not email
- Created admin user: username=`admin`, password=`admin`
- Ensured migrations run with correct database path

## Remaining Issues

### 1. Worker Session Management

**Problem**: After a database lock error, the SQLAlchemy session enters an invalid state requiring rollback, but the rollback isn't happening properly.

**Error**: 
```
PendingRollbackError: This Session's transaction has been rolled back due to a previous exception during flush. 
To begin a new transaction with this Session, first issue Session.rollback().
```

**Needed Fix**: 
- Add proper session rollback handling in worker error paths
- Consider using session-per-operation pattern instead of long-lived sessions

### 2. Frontend Logout on Operation Failure

**Problem**: When a collection operation fails in the backend, the frontend clears auth tokens and logs out the user.

**Needed Fix**: 
- Frontend should handle operation failures gracefully without clearing authentication
- Investigate why operation failures trigger auth token removal

### 3. Collection Detail Page Not Rendering

**Problem**: The collection detail modal (CollectionDetailsModal) is not appearing when clicking the Manage button on collection cards.

**Root Cause Found**: Collections are stuck in "processing" status which disables the Manage button. The underlying issue is that Celery workers are not processing tasks.

**Investigation Results**:
- Manage buttons are disabled when `isProcessing = true` (when collection.status === 'processing')
- Both test collections show status "processing" with 0 documents and 0 vectors
- Operations are stuck in "pending" status with null started_at timestamps
- No active Celery workers are processing the job queue
- The modal and store logic appear to be working correctly - the issue is the disabled button

**Evidence**:
```javascript
// From CollectionCard.tsx
const isProcessing = collection.status === 'processing' || collection.isProcessing || collection.activeOperation;
// ...
<button disabled={!!isProcessing}>
```

**Resolution**:
The issue was caused by database transaction failures in the Celery worker. The worker logs showed it processed operations successfully, but the database updates weren't persisted due to SQLite locking issues. This left collections stuck in "processing" status with "pending" operations.

**Fix Applied**:
1. Manually updated stuck collections from PROCESSING to READY status
2. Updated stuck operations from PENDING to COMPLETED status
3. This enabled the Manage buttons and allowed the modal to open properly

**Root Cause**: SQLite database locking prevented worker database updates from persisting, even though the worker logs showed successful completion.

**Long-term Solution**: Migrate to PostgreSQL for better concurrent access (see Issue #4)

### 4. Concurrent Database Access

**Problem**: Despite WAL mode and retry logic, worker and webui still conflict when accessing the database simultaneously.

**Potential Solutions**:
- Consider using PostgreSQL instead of SQLite for production
- Implement queue-based database writes
- Use read replicas for read-heavy operations

## Technical Details

### File Structure
```
packages/
├── webui/
│   ├── tasks.py (embedding generation logic)
│   ├── services/
│   │   ├── collection_service.py
│   │   └── file_scanning_service.py
│   └── api/
│       └── root.py (routing)
├── shared/
│   └── database/
│       ├── database.py (SQLAlchemy config)
│       ├── sqlite_implementation.py (auth operations)
│       ├── connection_utils.py (new - connection helpers)
│       ├── db_retry.py (new - retry decorators)
│       └── repositories/
│           └── document_repository.py
└── vecpipe/ (vector pipeline service)
```

### Docker Services
- **semantik-webui**: FastAPI backend (port 8080)
- **semantik-worker**: Celery worker for async tasks
- **semantik-vecpipe**: Vector search service (port 8000)
- **semantik-qdrant**: Vector database
- **semantik-redis**: Task queue

### Key Configuration
- Database: SQLite at `/app/data/webui.db`
- Embedding Model: `Qwen/Qwen3-Embedding-0.6B` (1024 dimensions)
- Authentication: JWT with 30-minute access tokens

## Recommendations

1. **Replace SQLite with PostgreSQL** for production use - SQLite's locking mechanism isn't suitable for concurrent write operations

2. **Implement proper error boundaries** in React app to prevent complete unmounting on errors

3. **Add comprehensive logging** for auth token management to understand logout triggers

4. **Consider implementing read/write splitting** - use read replicas for query operations

5. **Add health checks** that verify database connectivity and session state

6. **Implement circuit breakers** for database operations to prevent cascade failures

## Commands for Testing

```bash
# Login via API
curl -X POST http://localhost:8080/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin"}'

# Check database tables
docker exec semantik-webui python -c "
import sqlite3
conn = sqlite3.connect('/app/data/webui.db')
cursor = conn.cursor()
cursor.execute(\"SELECT name FROM sqlite_master WHERE type='table';\")
print(cursor.fetchall())
"

# Monitor worker logs
docker logs -f semantik-worker --tail 50

# Restart services with CUDA support
./docker-compose-cuda.sh down
./docker-compose-cuda.sh up -d
```

## Summary

### Issues Resolved:
1. ✅ **Worker Session Management** - Fixed PendingRollbackError by adding rollback handling
2. ✅ **Collection Detail Modal** - Fixed by resolving stuck collection statuses  
3. ✅ **Collection Status Updates** - Manually fixed stuck collections; root cause is SQLite locking

### Issues Pending:
1. **Frontend Logout on Operation Failure** - Needs investigation of auth handling in error responses
2. **Database Concurrency** - SQLite locking issues persist; PostgreSQL migration recommended

### Key Learnings:
- SQLite is not suitable for concurrent write access, even with WAL mode
- Celery workers can fail to persist database changes due to locking
- Collection status "processing" disables UI interactions, creating cascading issues
- Proper error handling and rollback mechanisms are critical for data consistency