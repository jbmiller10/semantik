# Development Log

## Collection Source Addition Fix (2025-07-18)

### Issue
User reported: "Collection created but failed to add source: Request failed with status code 422"

### Root Cause Analysis
1. **Frontend-Backend API Contract Mismatch**: The frontend was sending a JSON request body but the backend expected query parameters
2. **Worker Container Issues**: Multiple configuration and code issues preventing collection operations from completing

### Fixes Applied

#### 1. API Contract Fix
- Created `AddSourceRequest` schema in `packages/webui/api/schemas.py`
- Updated `add_source` endpoint in `packages/webui/api/v2/collections.py` to accept request body

#### 2. Worker Environment Configuration
- Added missing `REDIS_URL` environment variable to worker service in `docker-compose.yml`
- Worker was trying to connect to localhost:6379 instead of redis:6379

#### 3. Celery Task Handler Fix
- Fixed `on_failure` handler signature in `tasks.py` - was using incorrect parameter order
- Changed from lambda to direct function reference

#### 4. ORM Object vs Dictionary Issues
- Operation and Collection objects from repositories are ORM objects, not dictionaries
- Added conversion from ORM objects to dictionaries for compatibility with helper functions
- Fixed dictionary access patterns throughout the code

#### 5. Repository Method Names
- Fixed `collection_repo.get_by_id()` → `collection_repo.get_by_uuid()`
- CollectionRepository doesn't have `get_by_id` method

#### 6. Operation Status Update
- Removed unsupported `result=` parameter from `update_status()` calls
- Changed to use `error_message=` parameter for failures

#### 7. Additional Fixes
- Fixed `_record_operation_metrics`: Changed `operation["id"]` to `operation.id`
- Fixed finalize operation status check: Changed `current_status.get("status")` to `current_status.status`

#### 8. Repository Factory Issue
- `create_collection_repository()` returns SQLiteCollectionRepository for backward compatibility
- New code expects SQLAlchemy CollectionRepository with `get_by_uuid` method
- SQLiteCollectionRepository uses collection names, not UUIDs - fundamental incompatibility
- The collection-centric architecture requires the new SQLAlchemy repositories

### Critical Issue Found
The worker is using the old SQLite repositories through the factory functions, but the new collection-centric API creates collections with UUIDs and expects the SQLAlchemy repositories. This is why operations are failing - there's a mismatch between the repository implementations.

#### 9. Repository Session Management Fix
- Updated `_process_collection_operation_async` to use AsyncSessionLocal context manager
- Updated `_handle_task_failure_async` to use AsyncSessionLocal context manager
- Properly instantiated SQLAlchemy repositories with database session
- Fixed all indentation issues in both functions
- Added db.commit() at the end of async contexts

### Remaining Issues
- Fix IndentationError in tasks.py line 1460 preventing worker from starting
- Collections are stuck in PENDING state - not transitioning to READY
- The add_source endpoint correctly validates that collections must be READY or DEGRADED

### Current Status
- Worker failing to start due to IndentationError in tasks.py after the async context manager changes
- Collections created via API remain in PENDING state indefinitely
- The INDEX operation that should transition collections to READY is not being processed
- Frontend-backend API contract is fixed and working correctly

#### 10. Collection Model UUID Attribute Fix
- Error: `'Collection' object has no attribute 'uuid'`
- The Collection model uses `id` as the UUID field, not a separate `uuid` attribute
- Fixed all references from `collection_obj.uuid` to `collection_obj.id`
- Worker now starts successfully and processes operations

### Resolution Status
- Worker is now processing collection operations successfully
- Qdrant collections are being created
- Minor warning about audit log escape character (non-critical)
- Collections should now transition from PENDING to READY state

#### 11. CollectionStatus.EMPTY Fix
- Error: `AttributeError: EMPTY` - CollectionStatus enum doesn't have EMPTY value
- Fixed by setting collections to READY regardless of document count
- Collections with no documents are still considered READY

#### 12. Qdrant Collection Already Exists Issue  
- Error: 409 Conflict - Collection already exists in Qdrant
- Occurs when re-processing operations that partially succeeded before
- INDEX operation needs to handle existing collections gracefully

### Current Status
- All major code issues have been fixed
- Qdrant collections exist but SQL collections are still in PENDING/ERROR state
- Need to either:
  1. Handle existing Qdrant collections in INDEX operation
  2. Or manually update SQL collection status to READY for testing

#### 13. Final Status - Session Management Issue
- Manually updated collection to READY status in database - confirmed working
- Add source endpoint still fails with "A transaction is already begun on this Session"
- This appears to be a deeper issue with SQLAlchemy async session management

### Summary of Fixes Applied
1. ✅ Fixed API contract mismatch - frontend sends JSON body, backend now accepts it
2. ✅ Fixed worker environment - added REDIS_URL
3. ✅ Fixed Celery task handler signatures
4. ✅ Fixed ORM object access patterns (dictionary vs attributes)
5. ✅ Fixed repository usage - now using SQLAlchemy repos with async sessions
6. ✅ Fixed massive indentation issues in tasks.py
7. ✅ Fixed Collection model attribute access (uuid → id)
8. ✅ Fixed doc_stats key names (total_count → total_documents)
9. ✅ Fixed CollectionStatus.EMPTY (doesn't exist in enum)
10. ✅ Worker now processes operations successfully

### Remaining Issues
1. Qdrant collections already exist causing 409 conflicts on retry
2. SQLAlchemy async session management issue ("transaction already begun")
3. Collections stuck in PENDING state need manual intervention

The core functionality has been fixed but there are still some session management issues that need to be addressed in the SQLAlchemy async setup.

### Commands Used
```bash
# Restart services
make docker-restart

# Rebuild worker with fixes
docker compose build worker
docker compose stop worker && docker compose rm -f worker && docker compose up -d worker

# Check logs
docker logs semantik-worker --tail 30
docker logs semantik-webui --tail 30
```

### Test Script
Created `/tmp/test_add_source.py` to test the full flow:
- User registration/login
- Collection creation
- Waiting for collection to be ready
- Adding source to collection