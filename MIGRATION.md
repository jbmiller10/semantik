# Database Migration Guide

This guide helps you migrate from direct database function calls to the repository pattern introduced in PR #49.

## Overview

The database implementation has been moved from `webui.database` to `shared.database` to properly decouple the webui and vecpipe services. All direct database function calls are now deprecated and should be replaced with repository pattern usage.

## Migration Steps

### 1. Job Operations

#### Old Way (Deprecated)
```python
from shared.database import create_job, get_job, update_job, delete_job, list_jobs

# Create a job
job_id = create_job(job_data)

# Get a job
job = get_job(job_id)

# Update a job
update_job(job_id, {"status": "completed"})  # Returns None

# Delete a job
delete_job(job_id)  # Returns None

# List jobs
jobs = list_jobs(user_id=123)
```

#### New Way (Repository Pattern)
```python
from shared.database import create_job_repository

# Get repository instance
job_repo = create_job_repository()

# Create a job
job = await job_repo.create_job(job_data)  # Returns full job object

# Get a job
job = await job_repo.get_job(job_id)  # Returns job or None

# Update a job
job = await job_repo.update_job(job_id, {"status": "completed"})  # Returns updated job

# Delete a job
success = await job_repo.delete_job(job_id)  # Returns bool

# List jobs
jobs = await job_repo.list_jobs(user_id="123")  # Note: user_id is string
```

### 2. User Operations

#### Old Way (Deprecated)
```python
from shared.database import create_user, get_user, get_user_by_id

# Create a user
user = create_user(username="john", email="john@example.com", ...)

# Get user by username
user = get_user("john")

# Get user by ID
user = get_user_by_id(123)
```

#### New Way (Repository Pattern)
```python
from shared.database import create_user_repository

# Get repository instance
user_repo = create_user_repository()

# Create a user
user = await user_repo.create_user({
    "username": "john",
    "email": "john@example.com",
    ...
})

# Get user by username
user = await user_repo.get_user_by_username("john")

# Get user by ID
user = await user_repo.get_user("123")  # Note: ID is string
```

### 3. Collection Operations

Collection operations currently don't have repository implementations. Continue using the legacy functions but expect deprecation warnings:

```python
from shared.database import list_collections, get_collection_details

# These will show deprecation warnings
collections = list_collections(user_id=123)
details = get_collection_details("my_collection", user_id=123)
```

## Key Differences

### 1. Async/Await Pattern
All repository methods are async and must be called with `await`:
```python
# Old
job = get_job(job_id)

# New
job = await job_repo.get_job(job_id)
```

### 2. Return Values
- `create_job()` now returns the full job object, not just the ID
- `update_job()` now returns the updated job object, not None
- `delete_job()` now returns a boolean indicating success

### 3. String IDs
The repository pattern uses string IDs for consistency across different storage backends:
```python
# Old
jobs = list_jobs(user_id=123)  # int

# New
jobs = await job_repo.list_jobs(user_id="123")  # string
```

### 4. Error Handling
Repository methods provide consistent error handling:
- Methods that modify data (create, update, delete) raise exceptions on failure
- Methods that retrieve data (get, list) return None or empty list when not found
- All errors are logged with context

## Factory Pattern

Always use the factory functions to create repository instances:
```python
from shared.database import create_job_repository, create_user_repository

job_repo = create_job_repository()
user_repo = create_user_repository()
```

This allows for easy switching between different implementations (SQLite, PostgreSQL, etc.) in the future.

## Handling Deprecation Warnings

When you see deprecation warnings like:
```
DeprecationWarning: create_job is deprecated. Use create_job_repository().create_job() instead
```

1. Import the appropriate factory function
2. Create a repository instance
3. Replace the direct function call with the repository method
4. Add `await` if in an async context

## Future Plans

- Q1 2025: Implement user update and delete operations
- Q2 2025: Add repository implementations for collection operations
- Q3 2025: Remove deprecated legacy functions

## Transaction Support

The current implementation doesn't provide explicit transaction support. Each operation is atomic at the database level. Future versions may add transaction context managers:

```python
# Future API (not yet implemented)
async with job_repo.transaction():
    job = await job_repo.create_job(job_data)
    await job_repo.update_job(job["id"], {"status": "processing"})
```

## Getting Help

If you encounter issues during migration:
1. Check the deprecation warning message for the recommended replacement
2. Refer to the repository interface documentation in `shared/database/base.py`
3. Look at existing usage examples in the codebase