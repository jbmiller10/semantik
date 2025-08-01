# Migration Guide: Job-Centric to Collection-Centric Architecture

## Overview

This guide helps you migrate your Semantik deployment from the legacy job-centric architecture to the new collection-centric design. The new architecture provides better organization, multi-model support, and improved scalability.

## Table of Contents

1. [Understanding the Changes](#understanding-the-changes)
2. [API Migration](#api-migration)
3. [Database Migration](#database-migration)
4. [Configuration Changes](#configuration-changes)
5. [Frontend Updates](#frontend-updates)
6. [Common Pitfalls](#common-pitfalls)
7. [Rollback Strategy](#rollback-strategy)

## Understanding the Changes

### Terminology Mapping

| Old Term (Job-Centric) | New Term (Collection-Centric) | Description |
|------------------------|-------------------------------|-------------|
| Job | Operation | A background task for processing documents |
| Job Queue | Operation Queue | Queue for managing background tasks |
| Job Status | Operation Status | Status of a background task |
| Work Docs | Collection | A group of related documents |
| Index Job | Index Operation | Task to index documents into a collection |
| Reindex Job | Reindex Operation | Task to rebuild a collection's index |

### Conceptual Changes

#### Before: Job-Centric
```
User → Creates Job → Processes Documents → Creates Index
```

#### After: Collection-Centric
```
User → Creates Collection → Adds Documents → Triggers Operations
```

### Key Benefits

1. **Better Organization**: Documents are grouped into logical collections
2. **Multi-Model Support**: Each collection can use different embedding models
3. **Improved Performance**: Operations are scoped to collections
4. **Enhanced Security**: Collection-level permissions (future feature)

## API Migration

### Authentication Endpoints (No Changes)

Authentication endpoints remain the same:
- `POST /api/v1/auth/login`
- `POST /api/v1/auth/refresh`
- `POST /api/v1/auth/logout`

### Collection Management

#### Create Collection (New)

**Before (Job-Centric):**
```bash
# No direct collection creation - jobs created implicitly
POST /api/v1/jobs/index
{
  "document_paths": ["/path/to/docs"],
  "model_name": "sentence-transformers/all-MiniLM-L6-v2"
}
```

**After (Collection-Centric):**
```bash
# Step 1: Create collection
POST /api/v2/collections
{
  "name": "Engineering Docs",
  "description": "Technical documentation",
  "model_name": "sentence-transformers/all-MiniLM-L6-v2",
  "model_dimension": 384
}

# Step 2: Add documents
POST /api/v2/collections/{collection_id}/documents
{
  "paths": ["/path/to/docs"]
}

# Step 3: Trigger indexing
POST /api/v2/collections/{collection_id}/operations/index
```

#### List Collections

**Before:**
```bash
GET /api/v1/jobs  # Mixed jobs and implicit collections
```

**After:**
```bash
GET /api/v2/collections  # Clean collection list
```

#### Search

**Before:**
```bash
POST /api/v1/search
{
  "query": "semantic search",
  "collection_name": "work_docs",  # Fixed collection name
  "k": 10
}
```

**After:**
```bash
POST /api/v2/collections/{collection_id}/search
{
  "query": "semantic search",
  "limit": 10,
  "search_type": "hybrid",
  "rerank": true
}
```

### Operation Management

#### Get Operation Status

**Before:**
```bash
GET /api/v1/jobs/{job_id}
```

**After:**
```bash
GET /api/v2/operations/{operation_id}
```

#### List Operations

**Before:**
```bash
GET /api/v1/jobs?status=running
```

**After:**
```bash
GET /api/v2/collections/{collection_id}/operations
GET /api/v2/operations?status=running  # Global view
```

### WebSocket Updates

**Before:**
```javascript
ws.send(JSON.stringify({
  type: "subscribe",
  job_id: "job-123"
}));
```

**After:**
```javascript
ws.send(JSON.stringify({
  type: "subscribe_collection",
  collection_id: "coll-123"
}));

ws.send(JSON.stringify({
  type: "subscribe_operation",
  operation_id: "op-456"
}));
```

## Database Migration

### Automatic Migration

The system includes automatic migration support:

```bash
# Run the migration
poetry run python -m scripts.migrate_to_collections

# The migration will:
# 1. Create new collection records
# 2. Map existing jobs to operations
# 3. Update document associations
# 4. Preserve all data
```

### Manual Migration Steps

If you need to migrate manually:

```sql
-- 1. Create collections from existing data
INSERT INTO collections (id, name, model_name, created_at)
SELECT 
  gen_random_uuid(),
  'Migrated Documents',
  COALESCE(model_name, 'sentence-transformers/all-MiniLM-L6-v2'),
  NOW()
FROM jobs
WHERE status = 'completed'
GROUP BY model_name;

-- 2. Create operations from jobs
INSERT INTO operations (id, collection_id, type, status, created_at)
SELECT 
  j.id,
  c.id,
  CASE 
    WHEN j.job_type = 'index' THEN 'index'
    WHEN j.job_type = 'reindex' THEN 'reindex'
    ELSE 'index'
  END,
  j.status,
  j.created_at
FROM jobs j
JOIN collections c ON c.model_name = j.model_name;

-- 3. Update document associations
UPDATE documents d
SET collection_id = (
  SELECT c.id 
  FROM collections c
  JOIN operations o ON o.collection_id = c.id
  WHERE o.id = d.job_id
  LIMIT 1
);
```

## Configuration Changes

### Environment Variables

#### Deprecated Variables
```bash
# These are no longer used
JOB_QUEUE_NAME=default
MAX_JOBS_PER_WORKER=5
JOB_TIMEOUT_SECONDS=3600
```

#### New Variables
```bash
# Collection-specific settings
DEFAULT_COLLECTION_NAME="General Documents"
MAX_COLLECTIONS_PER_USER=10
COLLECTION_SIZE_LIMIT_GB=100

# Operation settings
OPERATION_QUEUE_NAME=operations
MAX_OPERATIONS_PER_WORKER=5
OPERATION_TIMEOUT_SECONDS=3600
```

### Docker Compose Updates

Update your `docker-compose.yml`:

```yaml
# Before
environment:
  - JOB_QUEUE_NAME=default
  - CELERY_TASK_ROUTES={"tasks.index_job": "default"}

# After
environment:
  - OPERATION_QUEUE_NAME=operations
  - CELERY_TASK_ROUTES={"tasks.index_operation": "operations"}
```

## Frontend Updates

### State Management

#### Before (Job Store)
```javascript
// stores/jobStore.js
const useJobStore = create((set) => ({
  jobs: [],
  createJob: async (data) => {
    const response = await api.post('/api/v1/jobs/index', data);
    // ...
  }
}));
```

#### After (Collection Store)
```javascript
// stores/collectionStore.js
const useCollectionStore = create((set) => ({
  collections: [],
  operations: [],
  
  createCollection: async (data) => {
    const response = await api.post('/api/v2/collections', data);
    // ...
  },
  
  indexCollection: async (collectionId) => {
    const response = await api.post(
      `/api/v2/collections/${collectionId}/operations/index`
    );
    // ...
  }
}));
```

### Component Updates

#### Search Component
```javascript
// Before
<Search 
  collectionName="work_docs"
  onSearch={handleSearch}
/>

// After
<Search 
  collectionId={selectedCollection.id}
  collectionName={selectedCollection.name}
  onSearch={handleSearch}
/>
```

#### Operation Status
```javascript
// Before
<JobStatus jobId={job.id} />

// After
<OperationStatus 
  operationId={operation.id}
  collectionId={operation.collection_id}
/>
```

## Common Pitfalls

### 1. Hardcoded Collection Names

**Problem:**
```javascript
// Hardcoded "work_docs" collection
const results = await api.post('/api/v1/search', {
  collection_name: 'work_docs',
  query: searchQuery
});
```

**Solution:**
```javascript
// Use dynamic collection ID
const results = await api.post(
  `/api/v2/collections/${collectionId}/search`,
  { query: searchQuery }
);
```

### 2. Missing Collection Creation

**Problem:**
```javascript
// Trying to index without creating collection first
await api.post('/api/v2/operations/index', { paths: [...] });
```

**Solution:**
```javascript
// Create collection first, then index
const collection = await api.post('/api/v2/collections', {...});
await api.post(`/api/v2/collections/${collection.id}/operations/index`);
```

### 3. WebSocket Subscription Syntax

**Problem:**
```javascript
// Old job subscription
ws.send(JSON.stringify({ type: 'subscribe', job_id: id }));
```

**Solution:**
```javascript
// New operation subscription
ws.send(JSON.stringify({ 
  type: 'subscribe_operation', 
  operation_id: id 
}));
```

### 4. Status Field Changes

**Problem:**
```javascript
// Checking old status values
if (job.status === 'indexing') { ... }
```

**Solution:**
```javascript
// Use new status values
if (operation.status === 'processing') { ... }
```

## Rollback Strategy

If you need to rollback to the job-centric architecture:

### 1. Database Rollback
```bash
# Restore from backup
pg_restore -d semantik semantik_backup_pre_migration.sql

# Or use Alembic
poetry run alembic downgrade -1
```

### 2. Code Rollback
```bash
# Checkout previous version
git checkout tags/v1.0.0-job-centric

# Rebuild and deploy
make docker-build
make docker-up
```

### 3. Configuration Rollback
```bash
# Restore old environment variables
cp .env.backup .env

# Restart services
docker-compose down
docker-compose up -d
```

## Validation Checklist

After migration, verify:

- [ ] All collections are visible in the UI
- [ ] Search functionality works for each collection
- [ ] New document indexing completes successfully
- [ ] WebSocket updates are received
- [ ] Operation status tracking works
- [ ] No errors in worker logs
- [ ] API endpoints return expected data

## Support Resources

- **Documentation**: `/docs/COLLECTIONS.md`
- **API Reference**: `/docs/API_REFERENCE.md`
- **Troubleshooting**: `/docs/TROUBLESHOOTING.md`
- **GitHub Issues**: Report migration problems

## Timeline Recommendations

### Phase 1: Preparation (1 week)
- Review documentation
- Test migration in staging
- Backup production data

### Phase 2: Migration (1 day)
- Run automated migration
- Verify data integrity
- Update configurations

### Phase 3: Validation (1 week)
- Monitor system performance
- Gather user feedback
- Fix any issues

### Phase 4: Cleanup (1 week)
- Remove deprecated code
- Update documentation
- Archive old configurations

## Conclusion

The migration from job-centric to collection-centric architecture is a significant improvement that provides better organization and scalability. While the migration requires careful planning, the automated tools and clear documentation make the process straightforward. The new architecture sets a solid foundation for future enhancements and features.