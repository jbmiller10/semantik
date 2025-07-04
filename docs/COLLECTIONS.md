# Collections Management System

## Overview

The Collections Management System in Semantik provides a user-friendly abstraction over the underlying job-based architecture. Collections allow users to organize, manage, and search across multiple data sources as a unified entity.

## Architecture

### Core Concepts

1. **Collection**: A logical grouping of data that can contain multiple jobs
2. **Job**: An individual data ingestion task (file upload, directory scan, etc.)
3. **Parent-Child Relationship**: Jobs can be linked to form a collection hierarchy

### Database Schema

Collections are implemented using the existing jobs table with additional fields:

```sql
-- Jobs table extensions for collections
parent_job_id INTEGER REFERENCES jobs(id),  -- Links to collection root job
mode TEXT CHECK(mode IN ('create', 'add')), -- Job creation mode
collection_name TEXT,                        -- User-friendly collection name
```

### Collection Lifecycle

1. **Creation**: Initial job with `mode='create'` establishes the collection
2. **Addition**: Subsequent jobs with `mode='add'` and `parent_job_id` reference
3. **Metadata Inheritance**: Child jobs inherit the collection name from parent
4. **Deletion**: Cascading deletion removes all associated jobs and data

## API Endpoints

### List Collections
```http
GET /api/collections/
```

Returns all collections with aggregated statistics:

```json
{
  "collections": [
    {
      "name": "Technical Documentation",
      "total_files": 156,
      "total_chunks": 3420,
      "created_at": "2024-01-15T10:30:00Z",
      "root_job_id": 1
    }
  ]
}
```

### Get Collection Details
```http
GET /api/collections/{collection_name}
```

Returns detailed information about a specific collection:

```json
{
  "name": "Technical Documentation",
  "jobs": [
    {
      "id": 1,
      "mode": "create",
      "source_path": "/docs/api",
      "files_found": 45,
      "status": "completed"
    },
    {
      "id": 7,
      "mode": "add",
      "source_path": "/docs/guides",
      "files_found": 111,
      "status": "completed"
    }
  ],
  "total_files": 156,
  "total_chunks": 3420,
  "duplicates_found": 3
}
```

### Get Collection Files
```http
GET /api/collections/{collection_name}/files?page={page}&per_page={per_page}
```

Returns paginated list of files in the collection:

```json
{
  "files": [
    {
      "id": 123,
      "path": "/docs/api/endpoints.md",
      "size": 15420,
      "chunks": 28,
      "content_hash": "sha256:abcd...",
      "job_id": 1
    }
  ],
  "total": 156,
  "page": 1,
  "per_page": 50
}
```

### Rename Collection
```http
PUT /api/collections/{collection_name}/rename
Content-Type: application/json

{
  "new_name": "API Documentation v2"
}
```

### Delete Collection
```http
DELETE /api/collections/{collection_name}
```

Permanently removes the collection and all associated data.

### Add to Collection
```http
POST /api/jobs/add-to-collection
Content-Type: application/json

{
  "collection_name": "Technical Documentation",
  "source_type": "file|directory|github|url",
  "source_path": "/path/to/new/docs",
  "filters": {
    "extensions": [".md", ".txt"],
    "ignore_patterns": ["**/node_modules/**"]
  }
}
```

## Features

### Duplicate Detection

The system automatically detects duplicate files using SHA-256 content hashing:

1. **Content Hash Calculation**: Each file's content is hashed during ingestion
2. **Duplicate Prevention**: Files with identical hashes are skipped
3. **Cross-Job Detection**: Duplicates are detected across all jobs in a collection
4. **Reporting**: Duplicate count is tracked and reported

### User Isolation

Collections are isolated per user for security:

- Each collection is associated with a user_id
- Users can only see and manage their own collections
- Legacy jobs (without user_id) are visible to all users

### Metadata Management

Collections maintain aggregated metadata:

- Total file count across all jobs
- Total chunk count for search sizing
- Creation and modification timestamps
- Source diversity tracking

## Frontend Integration

### Collections View

The main Collections tab displays:
- Collection cards with key statistics
- Quick actions (search, rename, delete)
- Visual indicators for collection size
- Recent activity timestamps

### Collection Details Modal

Provides detailed view including:
- All jobs within the collection
- File browser with pagination
- Duplicate file reporting
- Add data functionality

### Add to Collection Flow

1. User selects existing collection
2. Chooses data source type
3. Configures source-specific options
4. System prevents duplicate ingestion
5. Progress tracked via WebSocket

## Best Practices

### Collection Organization

1. **Logical Grouping**: Group related documents (e.g., "Product Documentation", "Research Papers")
2. **Version Management**: Use collection names with versions (e.g., "API Docs v2.1")
3. **Size Considerations**: Keep collections under 100k chunks for optimal performance

### Performance Optimization

1. **Batch Operations**: Add multiple sources in single operation when possible
2. **Duplicate Prevention**: System automatically handles duplicate detection
3. **Incremental Updates**: Add new data without re-processing existing files

### Data Management

1. **Regular Cleanup**: Remove outdated collections to free resources
2. **Naming Conventions**: Use descriptive, searchable collection names
3. **Access Control**: Ensure proper user authentication for sensitive data

## Implementation Details

### Collection Name Resolution

The system uses a hierarchical approach:
1. Check job's own collection_name field
2. If null, traverse to parent job
3. Use parent's collection_name
4. For root jobs, collection_name equals job description

### Query Optimization

Collections use efficient SQL queries:
- Indexed parent_job_id for fast hierarchy traversal
- Aggregated statistics cached where possible
- Pagination for large file lists

### Error Handling

- Collection name conflicts prevented at API level
- Orphaned jobs detected and cleaned up
- Graceful handling of missing parent jobs

## Future Enhancements

1. **Collection Sharing**: Allow controlled sharing between users
2. **Collection Templates**: Predefined configurations for common use cases
3. **Bulk Operations**: Multi-collection search and management
4. **Collection Versioning**: Track collection evolution over time
5. **Export/Import**: Collection backup and migration tools