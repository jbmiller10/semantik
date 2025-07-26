# Embedding Generation Debug Notes

## Issue Summary
Embedding generation is failing with 404 errors when the worker tries to call the vecpipe `/embed` endpoint.

## Current Status
- Worker service is running (fixed by removing profile restriction from docker-compose.yml)
- Vecpipe service is running
- Worker attempts to call `http://vecpipe:8000/embed` but gets 404 Not Found
- Documents are being processed and chunked successfully, but fail at the embedding generation step

## Error Pattern
```
semantik-worker    | [2025-07-25 06:42:27,444: INFO/ForkPoolWorker-1] Calling vecpipe /embed for 60 texts
semantik-worker    | [2025-07-25 06:42:27,447: INFO/ForkPoolWorker-1] HTTP Request: POST http://vecpipe:8000/embed "HTTP/1.1 404 Not Found"
semantik-vecpipe   | INFO:     172.18.0.7:54660 - "POST /embed HTTP/1.1" 404 Not Found
semantik-worker    | [2025-07-25 06:42:27,448: ERROR/ForkPoolWorker-1] Failed to process document /mnt/docs/Outvote GOTV 15 min.pptx: Failed to generate embeddings via vecpipe: 404 - {"detail":"Not Found"}
```

## What We've Tried

### 1. Fixed Worker Service Not Running
- **Issue**: Worker service had `profiles: - backend` in docker-compose.yml
- **Solution**: Removed the profile restriction so worker starts by default
- **Result**: Worker now runs, but encounters 404 when calling vecpipe

### 2. Verified /embed Endpoint Exists in Code
- **Location**: `/home/dockertest/semantik/packages/vecpipe/search_api.py` line 1223
- **Code**: `@app.post("/embed")`
- **Status**: Endpoint is properly defined in the code

### 3. Fixed Frontend Search Type Mismatch
- **Issue**: Frontend was using 'vector' but API expected 'semantic'
- **Changes**:
  - Updated `searchStore.ts`: Changed `searchType: 'vector'` to `searchType: 'semantic'`
  - Updated `SearchInterface.tsx`: Changed toggle logic to use 'semantic' instead of 'vector'
  - Updated test files to match
- **Result**: Fixed TypeScript compilation errors, but unrelated to embedding issue

### 4. Discovered Uncommitted Code
- **Finding**: The `/embed` endpoint was added to search_api.py but not committed
- **Git Status**: We were in the middle of a rebase with conflicts
- **Resolution**: Aborted the rebase to get to a clean state

## Current Theory

The Docker container for vecpipe was built before the `/embed` endpoint was added to the code. The running container doesn't have this endpoint, which is why it returns 404.

### Evidence:
1. The `/embed` endpoint exists in the code at line 1280 of search_api.py (confirmed)
2. The vecpipe service logs show it's receiving the POST request but returning 404
3. The endpoint is properly defined with correct request/response models (EmbedRequest/EmbedResponse)
4. The worker correctly calls `http://vecpipe:8000/embed` with the right payload structure

## Next Steps

### Immediate Solution
Rebuild the Docker containers to include the latest code:

```bash
# Stop all services
docker compose down

# Rebuild and start the services
docker compose up --build -d

# OR using make commands:
make docker-down
make docker-build
make docker-up
```

### Verification Steps
1. After rebuild, test if `/embed` endpoint is accessible:
   ```bash
   curl -X POST http://localhost:8000/embed -H "Content-Type: application/json" -d '{"texts": ["test"]}'
   ```

2. Check vecpipe logs to ensure it's running the updated code:
   ```bash
   docker logs semantik-vecpipe
   ```

3. Monitor worker logs to see if embedding generation succeeds:
   ```bash
   docker logs -f semantik-worker
   ```

## Additional Notes

### Architecture Overview
- **Worker**: Processes documents, creates chunks, calls vecpipe for embeddings
- **Vecpipe**: Provides `/embed` endpoint for generating embeddings using GPU
- **Communication**: Worker makes HTTP POST requests to `http://vecpipe:8000/embed`

### Code Locations
- Worker embedding call: `/home/dockertest/semantik/packages/webui/tasks.py` lines 1496-1512
- Worker reindex embedding call: `/home/dockertest/semantik/packages/webui/tasks.py` lines 1862-1878  
- Vecpipe embed endpoint: `/home/dockertest/semantik/packages/vecpipe/search_api.py` lines 1280-1365

### Related Configuration
- Worker uses `httpx` to make async HTTP calls to vecpipe
- Embedding model: `Qwen/Qwen3-Embedding-0.6B` (default)
- Quantization: `float16` (default)

## Additional Issues

### WebSocket Connection Errors (FIXED)
After the initial embedding generation issue, we're also seeing WebSocket connection errors:

```
semantik-webui     | ERROR:    Exception in ASGI application
semantik-webui     | KeyError: '89e8d3fe-9ff3-4f7d-9455-ed5d998bbb71'
semantik-webui     | Failed to send current state for operation 89e8d3fe-9ff3-4f7d-9455-ed5d998bbb71:
```

**Issue**: The WebSocket manager was trying to delete a consumer task that doesn't exist in the dictionary.

**Location**: `/app/packages/webui/websocket_manager.py` line 200

**Cause**: Race condition where the consumer task could be deleted between the check and the deletion.

**Fix Applied**: Added defensive check before deletion in websocket_manager.py:
```python
if operation_id in self.consumer_tasks:
    del self.consumer_tasks[operation_id]
```

**Status**: FIXED - The defensive check has been added to prevent the KeyError.

## Root Cause Confirmed

The `/embed` endpoint exists in the codebase but the Docker container was built before this endpoint was added. The vecpipe container is running an older version of the code that doesn't include this endpoint.

## Solution Required

The Docker containers MUST be rebuilt to include the `/embed` endpoint. The user needs to run:

```bash
# Stop all services
docker compose down

# Rebuild and start the services
docker compose up --build -d

# OR using make commands:
make docker-down
make docker-build
make docker-up
```

## Summary - UPDATED

### Update: `/embed` Endpoint IS Working!

After further investigation, I discovered that the `/embed` endpoint is actually working correctly:

1. **Direct test succeeded**: `curl -X POST http://localhost:8000/embed` returns 200 OK with valid embeddings
2. **From worker network**: `docker exec semantik-worker curl -X POST http://vecpipe:8000/embed` also works correctly
3. **All endpoints present**: Checked `/openapi.json` and confirmed `/embed` is registered

### Current Status:
- ✅ Vecpipe service is running and healthy
- ✅ `/embed` endpoint exists and is functional
- ✅ Worker can reach vecpipe service in Docker network
- ❌ No collections exist in the database to test with
- ⚠️ Worker is trying to process old operations that no longer exist

### Real Issues Found:
1. **No collections in database**: Query shows 0 collections, so there's nothing to process
2. **Worker processing stale operations**: Worker is retrying operations that were deleted from DB
3. **Authentication required**: Cannot create test collections via API without auth token

### Issues Already Fixed:
1. **Worker service not starting** - Fixed by removing profile restriction from docker-compose.yml ✓
2. **WebSocket KeyError** - Fixed by adding defensive check in websocket_manager.py ✓
3. **Frontend API mismatches** - Fixed search type ('vector' → 'semantic') and parameter names ✓

### Next Steps:
1. Create a collection through the UI (with authentication)
2. Monitor worker logs to see if embedding generation works for new operations
3. Clear any stale Celery tasks that are retrying non-existent operations

### Verification Commands:
```bash
# Check if embed endpoint works
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["test"], "model_name": "Qwen/Qwen3-Embedding-0.6B", "quantization": "float16"}'

# Check collections in DB
sudo docker exec semantik-postgres psql -U semantik -d semantik \
  -c "SELECT id, name, status FROM collections;"

# Monitor worker logs for new operations
sudo docker logs -f semantik-worker
```