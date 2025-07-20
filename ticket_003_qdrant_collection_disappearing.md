# Ticket #003: Fix Qdrant Collection Disappearing Issue

**Priority**: MAJOR
**Type**: Bug Fix
**Component**: Backend (Vector Storage)
**Affects**: Collection persistence and data integrity

## Summary
Qdrant collections are being created successfully according to worker logs but then disappear when queried via the Qdrant API. This suggests either premature cleanup, failed creation with false success reporting, or issues with collection naming/persistence.

## Context
During the review, a collection was created for "Test Collection" with ID `b59d7da5-fa91-4742-b33e-62dbc1a29dbc`. The worker logs show:
- Collection `col_b59d7da5_fa91_4742_b33e_62dbc1a29dbc` was created
- Qdrant verified the collection exists
- Task completed successfully

However, when querying Qdrant directly, only the `_collection_metadata` collection exists.

## Current State
Evidence from logs:
```
[2025-07-20 01:50:06,213: INFO/ForkPoolWorker-1] HTTP Request: PUT http://qdrant:6333/collections/col_b59d7da5_fa91_4742_b33e_62dbc1a29dbc "HTTP/1.1 200 OK"
[2025-07-20 01:50:06,216: INFO/ForkPoolWorker-1] HTTP Request: GET http://qdrant:6333/collections/col_b59d7da5_fa91_4742_b33e_62dbc1a29dbc "HTTP/1.1 200 OK"
[2025-07-20 01:50:06,218: INFO/ForkPoolWorker-1] Verified Qdrant collection col_b59d7da5_fa91_4742_b33e_62dbc1a29dbc exists with None vectors
```

But API query shows:
```json
{
  "collections": [
    {
      "name": "_collection_metadata"
    }
  ]
}
```

## Potential Causes
1. **Premature Cleanup**: A cleanup task might be removing "empty" collections
2. **Transaction Rollback**: Qdrant might be rolling back the creation
3. **Name Mismatch**: Collection might be created with a different name
4. **Async Timing**: Collection might be deleted between creation and query
5. **Configuration Issue**: Qdrant might not be persisting collections properly

## Investigation Steps

### 1. Check Cleanup Tasks
Review the cleanup task implementation:
```python
# Check webui/tasks.py for cleanup_qdrant_collections
# Look for conditions that might delete empty collections
```

### 2. Add Detailed Logging
Add logging at each step:
- Before collection creation
- After creation with collection details
- Before any cleanup operations
- In any error handlers

### 3. Check Qdrant Configuration
Verify Qdrant persistence settings:
- Storage configuration
- Memory vs disk persistence
- Any automatic cleanup settings

### 4. Review Collection Creation Logic
Examine the full creation flow in:
- `process_collection_operation` task
- Any post-creation hooks
- Error handling that might silently delete

## Technical Requirements

### Immediate Fixes
1. **Disable Automatic Cleanup** (temporary):
   ```python
   # In cleanup task, add check for minimum age
   if collection.created_at > (datetime.now() - timedelta(hours=1)):
       logger.info(f"Skipping cleanup for new collection {collection.id}")
       continue
   ```

2. **Add Collection Verification**:
   ```python
   def verify_qdrant_collection(collection_name: str, max_retries: int = 3):
       """Verify collection exists in Qdrant with retries."""
       for i in range(max_retries):
           try:
               client.get_collection(collection_name)
               return True
           except Exception as e:
               if i < max_retries - 1:
                   time.sleep(2 ** i)  # Exponential backoff
               else:
                   logger.error(f"Collection {collection_name} not found after {max_retries} attempts")
                   return False
   ```

3. **Add Persistent Collection Check**:
   ```python
   # After creation, before marking as complete
   if not verify_qdrant_collection(qdrant_collection_name):
       raise Exception("Collection creation verified but collection not found")
   ```

### Long-term Solutions
1. **Implement Collection Locking**:
   - Add a lock mechanism to prevent cleanup during active operations
   - Use Redis or database flags to mark collections as "in use"

2. **Better Error Handling**:
   - Don't mark operations as successful unless fully verified
   - Add rollback mechanisms for partial failures

3. **Monitoring and Alerts**:
   - Add metrics for collection creation/deletion
   - Alert when collections disappear unexpectedly

## Testing Requirements
1. Create a collection and verify it persists for at least 24 hours
2. Test with multiple collections created simultaneously
3. Verify collections survive container restarts
4. Test cleanup task doesn't remove active collections
5. Load test with many create/delete operations

## Acceptance Criteria
- [ ] Collections persist after creation
- [ ] Collections visible in Qdrant API immediately after creation
- [ ] Collections survive container/service restarts
- [ ] No false positive success responses
- [ ] Cleanup tasks have proper safeguards
- [ ] Clear error messages when creation fails
- [ ] Monitoring in place for collection lifecycle

## Related Code Locations
- `packages/webui/tasks.py` - Collection operation tasks
- `packages/vecpipe/services/` - Qdrant interaction code
- `packages/webui/services/collection_service.py` - Collection management
- Docker compose Qdrant configuration

## Debugging Commands
```bash
# Check Qdrant collections
curl http://localhost:6333/collections | jq

# Check Qdrant storage
docker exec semantik-qdrant ls -la /qdrant/storage

# Check cleanup task logs
docker logs semantik-worker | grep -i cleanup

# Monitor collection creation in real-time
docker logs -f semantik-worker | grep -i "collection"
```

## Notes
This issue undermines user confidence as collections appear to be created successfully but then vanish. It's critical for data integrity and system reliability. The issue might be related to the empty collection (0 documents) state, suggesting cleanup logic might be too aggressive.