# TASK-010: Blue-Green Staging Creation Implementation Plan

## Task Overview
Implement the first part of the re-indexing flow: creating the "green" (staging) Qdrant collections.

## Current State
Upon review, TASK-010 has already been implemented in the codebase. The implementation includes:

### 1. reindex_handler Function (lines 1207-1263 in tasks.py)
- Creates staging collections for blue-green reindexing
- Uses QdrantManager to create uniquely named staging collections
- Returns staging info dictionary with collection details

### 2. Integration with _process_reindex_operation
- Called at line 1325 during the staging creation checkpoint
- Properly updates the database with staging info at lines 1329-1332
- Includes error handling and cleanup on failure

### 3. Key Components
- **QdrantManager Import**: Already imported from `shared.managers.qdrant_manager`
- **Field Names**: Correctly uses `vector_store_name` (not `qdrant_collection_name`)
- **Database Update**: Stores staging info in `collections.qdrant_staging` field as JSON

## Implementation Details

### reindex_handler Function
```python
async def reindex_handler(
    collection: dict,
    new_config: dict[str, Any],
    qdrant_manager_instance: QdrantManager,
) -> dict[str, Any]:
```

**Purpose**: Create a staging collection for zero-downtime reindexing

**Process**:
1. Validates collection has `vector_store_name` field
2. Determines vector dimension from new_config or existing config
3. Uses QdrantManager.create_staging_collection to create unique staging collection
4. Returns staging info with collection name, timestamp, vector dimension, and base collection

### Database Storage
The staging collection info is stored as:
```python
{
    "collection_name": "staging_collection_uuid_20250716_123456",
    "created_at": "2025-07-16T12:34:56Z",
    "vector_dim": 768,
    "base_collection": "collection_uuid"
}
```

### Error Handling
- Raises ValueError if collection missing required fields
- Logs errors if staging creation fails
- Cleanup of staging collection handled in _process_reindex_operation on failure

## Acceptance Criteria Verification
✅ **Staging collections are created in Qdrant**: Using QdrantManager.create_staging_collection
✅ **Names are persisted to database**: Stored in collections.qdrant_staging field

## Code Quality
- All black formatting applied
- All ruff linting checks pass
- All mypy type checking passes

## Conclusion
TASK-010 has been fully implemented and meets all requirements. The blue-green staging creation is ready for use in the reindexing workflow.