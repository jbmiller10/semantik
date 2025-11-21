# Data Access Catalog for Projection Pipeline

This document catalogs existing data access points for embeddings and operations that the projection pipeline should use or extend.

For an end‑to‑end view of how these data access points feed the projection
visualization (artifacts, metadata, sampling, render modes, selection, and
tooltips), see also `docs/EMBEDDING_VISUALIZATION.md`.

## Table of Contents
1. [Vector/Embedding Access (Qdrant)](#vectorembedding-access-qdrant)
2. [Database Repositories](#database-repositories)
3. [Service Layer](#service-layer)
4. [Authorization & Security](#authorization--security)
5. [Task Management](#task-management)
6. [Recommended Integration Points](#recommended-integration-points)

---

## Vector/Embedding Access (Qdrant)

### QdrantManager (`packages/shared/managers/qdrant_manager.py`)

The primary interface for Qdrant operations across services.

**Key Methods for Projection Pipeline:**

```python
# Collection management
async def get_collection_usage(collection_name: str) -> dict[str, int]
    """Returns: {"documents": int, "vectors": int, "storage_bytes": int}"""

def collection_exists(collection_name: str) -> bool
    """Verify collection exists before scrolling"""

def get_collection_info(collection_name: str) -> CollectionInfo
    """Get metadata including vector count, dimension, etc."""

# Scrolling vectors (for projection input)
def _copy_collection_points(source: str, destination: str, batch_size: int = 256) -> int
    """Example of scrolling all points with pagination
    Uses: client.scroll(collection_name, offset, limit, with_payload, with_vectors)
    Returns: (records, next_offset)
    """
```

**Direct Qdrant Client Access:**
```python
from shared.managers.qdrant_manager import QdrantManager

manager = resolve_qdrant_manager()  # From webui.tasks.utils
qdrant_client = manager.get_client()  # Returns QdrantClient instance

# Scroll all vectors for dimensionality reduction
offset = None
batch_size = 1000
while True:
    records, next_offset = qdrant_client.scroll(
        collection_name=vector_store_name,
        offset=offset,
        limit=batch_size,
        with_payload=True,  # Include metadata
        with_vectors=True,  # Include embedding vectors
    )
    if not records:
        break
    # Process records: record.id, record.vector, record.payload
    offset = next_offset
    if not next_offset:
        break
```

**Location:** `packages/shared/managers/qdrant_manager.py:442-477`

**Usage Pattern in Codebase:**
- **Imports:** `from webui.tasks.utils import resolve_qdrant_manager`
- **Pattern:** All ingestion tasks use `resolve_qdrant_manager()` to get QdrantManager instance
- **Example:** `packages/webui/tasks/ingestion.py:541, 761`

---

## Database Repositories

### ChunkRepository (`packages/shared/database/repositories/chunk_repository.py`)

**CRITICAL:** Chunks table is partitioned by `collection_id` (100 partitions). **Always** include `collection_id` in queries for partition pruning.

**Key Methods:**

```python
async def get_chunks_by_collection(
    collection_id: str,
    limit: int | None = None,
    offset: int = 0,
    created_after: datetime | None = None
) -> list[Chunk]
    """Fetch chunks for a collection with efficient partition access
    Location: chunk_repository.py:193-221
    """

async def get_chunk_statistics_optimized(collection_id: str) -> dict[str, Any]
    """Returns:
    {
        "total_chunks": int,
        "avg_chunk_size": float,
        "min_chunk_size": int,
        "max_chunk_size": int,
        "unique_documents": int,
        "first_chunk_created": str,
        "last_chunk_created": str
    }
    Location: chunk_repository.py:474-509
    """

async def get_chunks_paginated(
    collection_id: str,
    page: int = 1,
    page_size: int = 100
) -> tuple[list[Chunk], int]
    """Returns (chunks, total_count)
    Uses window function for efficient pagination
    Location: chunk_repository.py:432-472
    """

async def get_chunks_batch(
    collection_id: str,
    document_ids: list[str],
    limit: int = 1000
) -> list[Chunk]
    """Batch fetch chunks for multiple documents
    Location: chunk_repository.py:399-430
    """
```

**Important Notes:**
- All methods enforce partition key validation
- Uses `ChunkPartitionHelper.create_chunk_query_with_partition()` internally
- **Anti-pattern:** Never query chunks without `collection_id` filter

**Chunk Model Fields (`packages/shared/database/models.py`):**
```python
class Chunk:
    id: BigInteger  # Auto-generated sequence ID
    collection_id: str  # PARTITION KEY - REQUIRED
    document_id: str
    chunk_index: int
    content: Text  # The actual text chunk
    embedding_vector_id: str | None  # Qdrant point ID
    partition_key: int  # 0-99, auto-computed
    created_at: DateTime
    updated_at: DateTime
```

---

### ProjectionRunRepository (`packages/shared/database/repositories/projection_run_repository.py`)

**Purpose:** Manages projection run metadata and lifecycle.

**Key Methods:**

```python
async def create(
    *,
    collection_id: str,
    reducer: str,  # "umap", "tsne", "pca"
    dimensionality: int,  # Typically 2 or 3
    config: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None
) -> ProjectionRun
    """Create new projection run record
    Validates: dimensionality > 0, collection exists
    Location: projection_run_repository.py:26-63
    """

async def get_by_uuid(projection_uuid: str) -> ProjectionRun | None
    """Fetch by external UUID with eager loading of collection/operation
    Location: projection_run_repository.py:65-73
    """

async def update_status(
    projection_uuid: str,
    *,
    status: ProjectionRunStatus,
    error_message: str | None = None,
    started_at: datetime | None = None,
    completed_at: datetime | None = None
) -> ProjectionRun
    """Update lifecycle status (PENDING -> RUNNING -> COMPLETED/FAILED)
    Auto-sets started_at and completed_at based on status
    Location: projection_run_repository.py:102-131
    """

async def update_metadata(
    projection_uuid: str,
    *,
    storage_path: str | None = None,
    point_count: int | None = None,
    meta: dict[str, Any] | None = None
) -> ProjectionRun
    """Update storage attributes after computation
    storage_path: Path to serialized array (e.g., Parquet/NPY file)
    point_count: Number of projected points
    meta: Additional metadata (timing, parameters, etc.)
    Location: projection_run_repository.py:143-170
    """

async def list_for_collection(
    collection_id: str,
    *,
    limit: int = 50,
    offset: int = 0,
    statuses: Sequence[ProjectionRunStatus] | None = None
) -> tuple[list[ProjectionRun], int]
    """List runs with optional status filter
    Returns: (runs, total_count)
    Location: projection_run_repository.py:75-100
    """
```

**ProjectionRun Model (`packages/shared/database/models.py`):**
```python
class ProjectionRun:
    id: int  # Primary key
    uuid: str  # External identifier
    collection_id: str  # Foreign key
    operation_uuid: str | None  # Links to Operation
    reducer: str  # "umap", "tsne", "pca"
    dimensionality: int  # 2 or 3
    config: dict  # Reducer hyperparameters
    meta: dict  # Additional metadata
    status: ProjectionRunStatus  # Lifecycle state
    storage_path: str | None  # Path to saved projection
    point_count: int | None  # Number of points
    error_message: str | None
    started_at: DateTime | None
    completed_at: DateTime | None
    created_at: DateTime
    updated_at: DateTime
```

**ProjectionRunStatus Enum:**
```python
class ProjectionRunStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
```

---

### CollectionRepository (`packages/shared/database/repositories/collection_repository.py`)

**Key Methods for Authorization:**

```python
async def get_by_uuid_with_permission_check(
    collection_uuid: str,
    user_id: int
) -> Collection
    """Fetch collection and verify user ownership
    Raises: EntityNotFoundError if not found or unauthorized
    Location: collection_repository.py:157

    CRITICAL: Use this for all user-facing projection API calls
    """

async def get_by_uuid(collection_uuid: str) -> Collection | None
    """Fetch without permission check (for system tasks)"""
```

**Collection Model Fields (Relevant):**
```python
class Collection:
    id: str  # Internal ID
    uuid: str  # External ID (use this in APIs)
    name: str
    user_id: int  # Owner
    vector_store_name: str  # Qdrant collection name
    embedding_model: str  # Model used for embeddings
    vector_count: int
    status: CollectionStatus
    config: dict
```

---

### OperationRepository (`packages/shared/database/repositories/operation_repository.py`)

**Purpose:** Track background operations (INDEX, APPEND, REINDEX, PROJECTION_BUILD).

**Key Methods:**

```python
async def create(
    collection_id: str,
    user_id: int,
    operation_type: OperationType,
    config: dict[str, Any],
    meta: dict[str, Any] | None = None
) -> Operation
    """Create operation record before dispatching Celery task"""

async def update_status(
    operation_id: str,
    status: OperationStatus,
    error_message: str | None = None
) -> Operation
    """Update operation status (PENDING -> PROCESSING -> COMPLETED/FAILED)"""

async def get_by_uuid(operation_uuid: str) -> Operation | None
    """Fetch operation by UUID"""
```

**OperationType Enum:**
```python
class OperationType(str, enum.Enum):
    INDEX = "index"
    APPEND = "append"
    REINDEX = "reindex"
    REMOVE_SOURCE = "remove_source"
    PROJECTION_BUILD = "projection_build"  # Added for projections
```

---

## Service Layer

### ProjectionService (`packages/webui/services/projection_service.py`)

**Current Status:** Scaffolding only. Methods return placeholders.

**Methods to Implement:**

```python
async def start_projection_build(
    collection_id: str,
    user_id: int,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Initiate projection computation for a collection.

    Lifecycle semantics:
    1. Validate collection with get_by_uuid_with_permission_check()
    2. Create a new ProjectionRun (status=PENDING) with reducer/config/color_by and sampling config
    3. Create a new Operation (type=PROJECTION_BUILD, status=PENDING) linked via operation_uuid
    4. Commit transaction (commit-before-enqueue)
    5. Dispatch Celery task `webui.tasks.process_collection_operation`

    Recompute behaviour:
    - Every request creates a fresh ProjectionRun + Operation; previous runs are
      retained as history and may be marked `meta["degraded"] = True` when the
      underlying collection changes or the worker had to fall back to PCA.
    - API responses surface both `status` (ProjectionRunStatus) and
      `operation_status` (OperationStatus) alongside `operation_id` so callers
      can treat projections as long-running jobs.

    Location: packages/webui/services/projection_service.py::start_projection_build
    """

async def get_projection_array(
    collection_id: str,
    projection_id: str,
    user_id: int
) -> bytes
    """Stream binary projection coordinates

    Location: projection_service.py:183-195
    TODO: Load from storage_path and stream as binary
    """

async def select_projection_region(
    collection_id: str,
    projection_id: str,
    selection: dict[str, Any],
    user_id: int
) -> dict[str, Any]
    """Map screen coordinates to document/chunk IDs

    Semantics:
    - Resolves on-disk artifacts under data_dir/semantik/projections/<collection>/<projection>.
    - Returns chunk/document metadata and a degraded flag derived from ProjectionRun.meta
      and projection_artifacts.degraded.
    - When required artifacts are missing or meta.json is invalid, the run is marked
      degraded via ProjectionRunRepository.update_metadata and a 4xx error is raised so
      the UI can prompt for recompute.

    Location: packages/webui/services/projection_service.py::select_projection_region
    """

async def delete_projection(
    collection_id: str,
    projection_id: str,
    user_id: int,
) -> None:
    """Delete a projection run and its artifacts.

    Semantics:
    - Validates collection ownership.
    - Rejects deletion for in-progress runs (status pending/running) with HTTP 409 so
      workers are not racing with artifact removal.
    - For completed/failed/cancelled runs, removes only the run's artifacts directory and
      ProjectionRun row. Other runs and operations for the collection are unaffected.

    Location: packages/webui/services/projection_service.py::delete_projection
    """
```

**Initialization Pattern:**
```python
def create_projection_service(db: AsyncSession) -> ProjectionService:
    return ProjectionService(
        db_session=db,
        projection_repo=ProjectionRunRepository(db),
        operation_repo=OperationRepository(db),
        collection_repo=CollectionRepository(db),
    )
```

---

## Authorization & Security

### Permission Checking Pattern

**ALWAYS use repository methods with permission checks for user-facing APIs:**

```python
# ✅ CORRECT - For API endpoints
collection = await collection_repo.get_by_uuid_with_permission_check(
    collection_uuid, user_id
)

# ❌ WRONG - Only for system tasks
collection = await collection_repo.get_by_uuid(collection_uuid)
```

**Location:** `packages/shared/database/repositories/collection_repository.py:157`

**Raises:** `EntityNotFoundError` if collection not found or user doesn't own it.

**Example Usage:**
```python
# In projection_service.py:82
collection = await self.collection_repo.get_by_uuid_with_permission_check(
    collection_id, user_id
)
```

---

## Task Management

### Celery Task Pattern (Critical for Operations)

**RULE:** Always commit transaction BEFORE dispatching Celery task.

**Correct Pattern:**
```python
# 1. Create database records
operation = await operation_repo.create(...)
projection_run = await projection_repo.create(...)

# 2. Link records
projection_run.operation_uuid = operation.uuid
await db.flush()

# 3. COMMIT FIRST (prevents race conditions)
await db.commit()

# 4. THEN dispatch task
celery_app.send_task(
    "webui.tasks.process_collection_operation",
    args=[operation.uuid],
    task_id=str(uuid.uuid4()),
)
```

**Why:** Worker task queries operation by UUID immediately. If transaction not committed, worker sees empty database.

**Example:** `packages/webui/services/projection_service.py:112-119`

---

### Task Entry Point (`packages/webui/tasks/ingestion.py`)

**Main Dispatcher:**
```python
async def _process_collection_operation_async(
    operation_id: str,
    celery_task: Any
) -> dict[str, Any]:
    """Routes operations by type:
    - OperationType.INDEX -> _process_index_operation()
    - OperationType.APPEND -> _process_append_operation_impl()
    - OperationType.REINDEX -> _process_reindex_operation()
    - OperationType.REMOVE_SOURCE -> _process_remove_source_operation()
    - OperationType.PROJECTION_BUILD -> _process_projection_operation()

    Location: ingestion.py:105-369
    """
```

**Projection Handler (Stub):**
```python
# packages/webui/tasks/projection.py:19-42
async def _process_projection_operation(
    operation: dict[str, Any],
    collection: dict[str, Any],
    projection_repo: Any,
    updater: CeleryTaskWithOperationUpdates,
) -> dict[str, Any]:
    """TODO: Implement actual projection computation

    Expected steps:
    1. Fetch embeddings from Qdrant via scroll
    2. Apply dimensionality reduction (UMAP/t-SNE/PCA)
    3. Save coordinates to storage
    4. Update ProjectionRun metadata
    5. Send progress updates via updater
    """
```

**Progress Updates:**
```python
async with CeleryTaskWithOperationUpdates(operation_id) as updater:
    await updater.send_update(
        "projection_started",
        {"status": "fetching_embeddings", "collection_id": collection_id}
    )
    # ... compute projection ...
    await updater.send_update(
        "projection_progress",
        {"status": "reducing", "progress_percent": 50}
    )
```

**Location:** `packages/webui/tasks/utils.py` (CeleryTaskWithOperationUpdates)

---

## Recommended Integration Points

### 1. Fetching Embeddings for Projection

**Recommended Approach:** Use Qdrant scroll with QdrantManager

```python
from webui.tasks.utils import resolve_qdrant_manager

async def fetch_embeddings_for_projection(
    collection_id: str,
    collection_repo: CollectionRepository,
    batch_size: int = 1000
) -> tuple[list[list[float]], list[dict[str, Any]]]:
    """Fetch all embeddings and metadata for a collection

    Returns:
        (embeddings, metadata) where metadata includes doc_id, chunk_id, etc.
    """
    # Get collection to find Qdrant collection name
    collection = await collection_repo.get_by_uuid(collection_id)
    vector_store_name = collection.vector_store_name

    # Get Qdrant client
    manager = resolve_qdrant_manager()
    qdrant_client = manager.get_client()

    # Scroll all points
    embeddings = []
    metadata = []
    offset = None

    while True:
        records, next_offset = qdrant_client.scroll(
            collection_name=vector_store_name,
            offset=offset,
            limit=batch_size,
            with_payload=True,
            with_vectors=True,
        )

        if not records:
            break

        for record in records:
            embeddings.append(record.vector)
            metadata.append({
                "point_id": record.id,
                "doc_id": record.payload.get("doc_id"),
                "chunk_id": record.payload.get("chunk_id"),
                "path": record.payload.get("path"),
            })

        offset = next_offset
        if not next_offset:
            break

    return embeddings, metadata
```

**Alternative:** Query chunks table + match with Qdrant

```python
async def fetch_embeddings_with_chunk_validation(
    collection_id: str,
    chunk_repo: ChunkRepository,
    collection_repo: CollectionRepository,
) -> tuple[np.ndarray, list[Chunk]]:
    """Fetch embeddings ensuring alignment with chunks table"""
    # Get all chunks for collection (includes embedding_vector_id)
    chunks = await chunk_repo.get_chunks_by_collection(
        collection_id,
        limit=None  # Fetch all
    )

    # Get Qdrant point IDs
    point_ids = [chunk.embedding_vector_id for chunk in chunks if chunk.embedding_vector_id]

    # Retrieve vectors from Qdrant
    collection = await collection_repo.get_by_uuid(collection_id)
    manager = resolve_qdrant_manager()
    qdrant_client = manager.get_client()

    records = qdrant_client.retrieve(
        collection_name=collection.vector_store_name,
        ids=point_ids,
        with_vectors=True
    )

    # Align vectors with chunks
    id_to_vector = {r.id: r.vector for r in records}
    embeddings = [id_to_vector[chunk.embedding_vector_id] for chunk in chunks]

    return np.array(embeddings), chunks
```

---

### 2. Saving Projection Results

**Recommended Format:** Parquet or NumPy binary

```python
import pyarrow as pa
import pyarrow.parquet as pq

async def save_projection_results(
    projection_run: ProjectionRun,
    coordinates: np.ndarray,  # Shape: (N, dimensionality)
    metadata: list[dict[str, Any]],
    projection_repo: ProjectionRunRepository,
    storage_dir: Path = Path("/app/data/projections")
) -> str:
    """Save projection coordinates and metadata to disk

    Returns: storage_path
    """
    storage_dir.mkdir(parents=True, exist_ok=True)

    # Create filename
    filename = f"projection_{projection_run.uuid}.parquet"
    storage_path = storage_dir / filename

    # Create table with coordinates and metadata
    table = pa.table({
        "x": coordinates[:, 0],
        "y": coordinates[:, 1],
        "z": coordinates[:, 2] if coordinates.shape[1] > 2 else [0] * len(coordinates),
        "point_id": [m["point_id"] for m in metadata],
        "doc_id": [m["doc_id"] for m in metadata],
        "chunk_id": [m["chunk_id"] for m in metadata],
    })

    pq.write_table(table, storage_path)

    # Update projection run metadata
    await projection_repo.update_metadata(
        projection_run.uuid,
        storage_path=str(storage_path),
        point_count=len(coordinates),
        meta={
            "shape": coordinates.shape,
            "reducer_params": projection_run.config,
        }
    )

    return str(storage_path)
```

---

### 3. Progress Reporting Pattern

**Use WebSocket updates during long-running projection:**

```python
async def compute_projection_with_progress(
    operation: dict,
    collection: dict,
    projection_repo: ProjectionRunRepository,
    updater: CeleryTaskWithOperationUpdates,
):
    """Example implementation with progress updates"""

    # Step 1: Fetch embeddings
    await updater.send_update(
        "projection_started",
        {"status": "fetching_embeddings", "total_vectors": 0}
    )

    embeddings, metadata = await fetch_embeddings_for_projection(...)

    await updater.send_update(
        "embeddings_fetched",
        {"status": "reducing", "total_vectors": len(embeddings)}
    )

    # Step 2: Dimensionality reduction
    reducer = create_reducer(operation["config"])
    coordinates = reducer.fit_transform(embeddings)

    await updater.send_update(
        "reduction_complete",
        {"status": "saving", "points_generated": len(coordinates)}
    )

    # Step 3: Save results
    storage_path = await save_projection_results(...)

    await updater.send_update(
        "projection_complete",
        {"status": "completed", "storage_path": storage_path}
    )
```

---

### 4. Authorization Enforcement

**API Layer Pattern:**

```python
from fastapi import Depends, HTTPException
from webui.dependencies import get_current_user

@router.post("/collections/{collection_id}/projections")
async def create_projection(
    collection_id: str,
    parameters: ProjectionParameters,
    current_user: dict = Depends(get_current_user),
    projection_service: ProjectionService = Depends(get_projection_service),
):
    """API endpoint with automatic authorization"""
    result = await projection_service.start_projection_build(
        collection_id=collection_id,
        user_id=current_user["id"],  # Service enforces ownership
        parameters=parameters.dict(),
    )
    return result
```

**Service Layer (Already Implemented):**
```python
# projection_service.py:82
collection = await self.collection_repo.get_by_uuid_with_permission_check(
    collection_id, user_id
)
# Raises EntityNotFoundError if unauthorized
```

---

## Summary of Key Access Points

### For Projection Pipeline Implementation:

1. **Embedding Retrieval:**
   - Primary: `QdrantManager.client.scroll()` with `with_vectors=True`
   - Alternative: `ChunkRepository.get_chunks_by_collection()` + `qdrant_client.retrieve()`

2. **Metadata Storage:**
   - `ProjectionRunRepository.create()` → Create run
   - `ProjectionRunRepository.update_status()` → Track lifecycle
   - `ProjectionRunRepository.update_metadata()` → Save results

3. **Authorization:**
   - `CollectionRepository.get_by_uuid_with_permission_check()` → Validate access

4. **Task Orchestration:**
   - `OperationRepository.create()` → Track operation
   - `celery_app.send_task()` → Dispatch worker
   - `CeleryTaskWithOperationUpdates` → WebSocket progress

5. **Progress Updates:**
   - `CeleryTaskWithOperationUpdates.send_update()` → Real-time notifications

### Critical Patterns to Follow:

- ✅ **Always** include `collection_id` in chunk queries (partition pruning)
- ✅ **Always** use permission-checked repository methods for user APIs
- ✅ **Always** commit database transaction BEFORE dispatching Celery tasks
- ✅ **Use** `resolve_qdrant_manager()` for consistent Qdrant access
- ✅ **Send** progress updates via WebSocket for long operations

### Files to Extend:

1. `packages/webui/tasks/projection.py` → Implement `_process_projection_operation()`
2. `packages/webui/services/projection_service.py` → Implement placeholder methods
3. Create new: `packages/shared/dimensionality_reduction/` → UMAP/t-SNE/PCA logic
4. Create new: `packages/shared/storage/projection_storage.py` → Save/load utilities

---

**Document Version:** 1.0
**Last Updated:** 2025-10-21
**Related Documents:**
- `CLAUDE.md` - Project development guidelines
- `docs/API_REFERENCE.md` - REST API documentation
- `docs/ARCH.md` - System architecture overview
