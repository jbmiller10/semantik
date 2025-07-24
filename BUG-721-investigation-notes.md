# BUG-721 Investigation Notes

## Bug Summary
Collection Status and Vector Count Not Updating in UI After Embedding Operation

## Root Cause Analysis

### 1. WebSocket Message Format Mismatch (FIXED)
The backend sends messages with this format:
```json
{
  "timestamp": "2025-07-23T23:00:00.000Z",
  "type": "operation_completed",
  "data": {
    "status": "completed",
    "result": {...}
  }
}
```

But the frontend expected:
```json
{
  "operation_id": "abc123",
  "status": "completed",
  "progress": 100,
  "message": "...",
  "error": null,
  "metadata": {}
}
```

**Fix Applied:** Updated `useOperationProgress.ts` to properly parse the backend's message format.

### 2. React Query Cache Update Issue (FIXED)
The `useUpdateOperationInCache` function only updated the `collectionKeys.detail(collectionId)` query, but not the `collectionKeys.lists()` query which the dashboard uses.

**Fix Applied:** Updated `useCollectionOperations.ts` to:
- Update both detail and list views
- Invalidate queries on completion to fetch fresh data

### 3. Method Name Mismatch (FIXED)
The collection service was calling `list_by_collection` but the repository method is `list_for_collection`.

**Fix Applied:** Updated `collection_service.py` to use the correct method name.

## Issues Fixed

### 1. Redis Connection in WebSocket Manager (FIXED)
The WebSocket manager was failing to connect to Redis during startup due to logging suppression in uvicorn.

**Fix Applied:** 
- Added reconnection logic to the WebSocket manager's `connect()` method
- Added proper logging to debug the connection process
- WebSocket manager now successfully connects to Redis

**Result:** Redis is now connected and attempting to consume streams. New error shows streams don't exist because worker isn't processing tasks.

### 2. Worker Database Connection Issues (FIXED)
The Celery worker had database connection errors due to password mismatch.

**Error:**
```
asyncpg.exceptions.InvalidPasswordError: password authentication failed for user "semantik"
```

**Root Cause:** 
- Worker container had stale password from previous runs
- WebUI has correct password: `bb26729ba8054eb9453...`
- Worker had old password: `3de6ce9874f8718281f...`

**Fix Applied:** 
- Recreated worker container with `docker compose down worker && docker compose up -d worker`
- This forced the container to pick up fresh environment variables from .env file

**Result:** Worker now connects successfully to both Redis and PostgreSQL

### 3. Task Processing Flow
The complete flow should be:
1. WebUI creates operation and dispatches Celery task
2. Worker picks up task and processes it
3. Worker sends updates to Redis Stream
4. WebSocket manager consumes from Redis Stream and broadcasts to clients

Currently failing at step 2 - worker can't process tasks due to database issues.

## Files Modified

1. `/home/dockertest/semantik/apps/webui-react/src/hooks/useOperationProgress.ts`
   - Fixed WebSocket message parsing
   - Removed unused import

2. `/home/dockertest/semantik/apps/webui-react/src/hooks/useCollectionOperations.ts`
   - Added collection list cache updates
   - Added query invalidation on operation completion

3. `/home/dockertest/semantik/packages/webui/services/collection_service.py`
   - Fixed method name from `list_by_collection` to `list_for_collection`

## Current Status

All major issues have been resolved:

1. **Frontend Fixes Applied**: ✅
   - WebSocket message parsing fixed to handle backend's nested format
   - React Query cache updates fixed to update both detail and list views
   - Method name mismatch fixed in collection service

2. **Backend Issues Resolved**: ✅
   - Redis WebSocket connection fixed with reconnection logic
   - Worker database connection fixed by recreating container with fresh credentials
   - Worker now successfully connects to both Redis and PostgreSQL

3. **Ready for Testing**: 
   - The complete flow should now work end-to-end
   - New collections should update in real-time
   - Existing stuck operations need fresh tasks to be created

## Current Status Update (2025-07-24)

### Progress Made

1. **All core infrastructure issues resolved**:
   - ✅ WebSocket now connects to Redis successfully
   - ✅ Worker connects to both Redis and PostgreSQL
   - ✅ Frontend WebSocket message parsing fixed
   - ✅ React Query cache updates implemented
   - ✅ Worker processes tasks successfully

2. **Cleanup performed**:
   - Marked 4 stuck operations as failed
   - Updated stuck collections to error status
   - Improved WebSocket error handling for missing streams

3. **Successful test**:
   - Created new collection "Test RT Updates v2"
   - Worker picked up and processed the task
   - Operation completed successfully (took ~90ms)
   - Collection status changed from "pending" to "ready"

### Remaining Issue - Documents Not Being Processed

After adding a source directory (`/mnt/docs`), the following issues were discovered:

1. **Documents are created but not processed**:
   - 5 documents were found and created in the database
   - BUT they have 0 chunks (not parsed/chunked)
   - No embeddings were generated
   - Documents show in the Files tab but with 0 chunks each

2. **Collection counts not updated**:
   - Collection shows 0 documents, 0 vectors in both UI views
   - Database query confirms: 5 actual documents exist but collection.document_count = 0
   - All documents have chunk_count = 0

3. **APPEND operation is incomplete**:
   - Worker log shows: "5 documents found, 0 new, 5 duplicates"
   - Documents were marked as duplicates (by content hash) but still created
   - The APPEND operation completed in only 63ms - too fast for actual processing
   - It only scanned files and created document records, but didn't:
     - Parse document content
     - Create chunks
     - Generate embeddings
     - Update collection statistics

### Root Cause

The worker's APPEND operation implementation appears to be incomplete. It only:
1. Scans the source directory
2. Creates document records in the database
3. Marks the operation as complete

But it's missing the actual document processing pipeline that should:
1. Parse each document
2. Split into chunks
3. Generate embeddings
4. Store vectors in Qdrant
5. Update collection counts

### Summary

While the real-time status updates are now working (the UI updates from "pending" → "ready"), the actual document processing isn't happening. This is a separate issue from the WebSocket updates - it's a problem with the worker's document processing implementation.

## Testing Commands Used

```bash
# Check running services
sudo docker ps

# Check logs
sudo docker logs semantik-webui --tail 50
sudo docker logs semantik-worker --tail 50

# Test Redis connectivity
sudo docker exec semantik-webui python -c "import redis; r = redis.from_url('redis://redis:6379/0'); print('Redis ping:', r.ping())"

# Check API endpoints
curl -s http://localhost:8080/api/v2/collections -H "Authorization: Bearer [TOKEN]"
curl -s http://localhost:8080/api/v2/collections/[COLLECTION_ID]/operations -H "Authorization: Bearer [TOKEN]"

# Rebuild and restart services
sudo docker compose build webui
sudo docker compose restart webui
sudo docker compose down && sudo docker compose up -d
```

## Final Status Update (2025-07-24) 

### Final Resolution Summary

BUG-721 ("Collection Status and Vector Count Not Updating in UI After Embedding Operation") has been fully resolved through a series of fixes addressing multiple interconnected issues:

#### 1. WebSocket Infrastructure (FIXED)
- **Issue**: WebSocket manager couldn't connect to Redis, preventing real-time updates
- **Root Cause**: The WebSocket manager's startup method wasn't connecting, and logging was suppressed
- **Fix**: Added reconnection logic in the `connect()` method and improved error handling
- **Result**: WebSocket now successfully connects to Redis and consumes update streams

#### 2. Worker Database Connection (FIXED)
- **Issue**: Worker had authentication errors and couldn't process tasks
- **Root Cause**: Stale database password in worker container environment
- **Fix**: Recreated worker container to pick up fresh credentials from .env
- **Result**: Worker now connects successfully to both Redis and PostgreSQL

#### 3. Frontend Message Parsing (FIXED)
- **Issue**: Frontend expected different message format than backend was sending
- **Root Cause**: Message format mismatch - backend sends nested structure, frontend expected flat
- **Fix**: Updated `useOperationProgress.ts` to parse `message.data.status` instead of `message.status`
- **Result**: Frontend correctly processes WebSocket messages

#### 4. React Query Cache Updates (FIXED)
- **Issue**: Dashboard collection list wasn't updating when operations completed
- **Root Cause**: Only updating detail view cache, not list view cache
- **Fix**: Updated `useCollectionOperations.ts` to update both `collectionKeys.detail()` and `collectionKeys.lists()`
- **Result**: Both detail and dashboard views update in real-time

#### 5. Document Processing in APPEND Operation (FIXED)
- **Issue**: Documents created but not chunked/embedded (0 chunks per document)
- **Root Cause**: APPEND operation only processed if `new_documents_registered > 0`, skipping existing unprocessed docs
- **Fix**: Changed logic to find and process any documents with `chunk_count == 0`
- **Result**: Documents now get properly chunked and embedded (e.g., 181 chunks created)

#### 6. Collection Statistics Updates (FIXED)
- **Issue**: UI showed 0 documents and 0 vectors even after processing
- **Root Cause**: Collection stats weren't updated after document processing
- **Fix**: Added proper stats update using `collection_repo.update_stats()` with accurate counts
- **Result**: UI now shows correct document and vector counts

#### 7. Vector Count Retrieval from Qdrant (FIXED)
- **Issue**: Vector count showed as 0 in database even though vectors existed in Qdrant
- **Root Cause**: Code was using `qdrant_info.vectors_count` which returns `None`, instead of `points_count`
- **Fix**: Changed to use `qdrant_info.points_count` in tasks.py:1603
- **Result**: Vector counts now correctly update in database and UI shows accurate counts

### Verification Steps Performed

1. Created new collection "Test RT Updates v2" - status updated in real-time
2. Added source directory `/mnt/docs` - 5 documents registered
3. Triggered reprocessing - documents chunked and embedded successfully
4. Confirmed UI shows correct counts without manual refresh

### Modified Files

1. `/home/dockertest/semantik/apps/webui-react/src/hooks/useOperationProgress.ts`
2. `/home/dockertest/semantik/apps/webui-react/src/hooks/useCollectionOperations.ts`
3. `/home/dockertest/semantik/packages/webui/services/collection_service.py`
4. `/home/dockertest/semantik/packages/webui/websocket_manager.py`
5. `/home/dockertest/semantik/packages/webui/tasks.py` (modified twice - document processing and vector count)
6. `/home/dockertest/semantik/docker-entrypoint.sh`

### Lessons Learned

1. **Interconnected Systems**: What appeared as a simple UI update issue involved WebSocket infrastructure, worker processing, and database operations
2. **Logging Importance**: Missing startup logs masked the Redis connection failure
3. **End-to-End Testing**: Need to verify the complete flow from document upload through processing to UI updates
4. **Defensive Programming**: The APPEND operation should handle both new and existing unprocessed documents

### Complete Resolution

BUG-721 is now fully resolved. All identified issues have been fixed:

1. ✅ Real-time status updates work via WebSocket
2. ✅ Documents are properly processed into chunks and embeddings  
3. ✅ Collection statistics (document_count and vector_count) update correctly in the database
4. ✅ UI reflects accurate counts without requiring manual page refresh
5. ✅ Operations show correct progress and completion status
6. ✅ Both dashboard and detail views update in real-time

The bug ticket requirements have been fully addressed - users now receive real-time feedback during long-running data ingestion operations without needing to refresh the page.