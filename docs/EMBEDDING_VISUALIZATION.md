# Embedding Visualization

Projection pipeline: creation, computation, storage, visualization.

Covers sampling, render modes, WebGL fallback, tooltips, selection, labels.

See `DATA_ACCESS_CATALOG.md` for data access details.

## Pipeline

1. **Start projection (API / UI)**
   - The React UI calls the v2 projections API
     (`POST /api/v2/collections/{collection_id}/projections`) with a
     `ProjectionBuildRequest` / `StartProjectionRequest`.
   - The API delegates to `ProjectionService.start_projection_build(...)`.
   - A new `ProjectionRun` row and a backing `Operation` row are created;
     both start in `PENDING`.
   - A Celery task is dispatched via the generic
     `"webui.tasks.process_collection_operation"` route.

2. **Process operation (Celery dispatcher)**
   - `webui.tasks.projection._process_projection_operation(...)` is invoked
     from the collection‑operation dispatcher.
   - It links the `Operation` to the `ProjectionRun`, marks the run
     `RUNNING` and the operation `PROCESSING`, then enqueues the
     `compute_projection` task.

3. **Compute projection (Celery worker)**
   - `compute_projection` calls
     `packages/webui/tasks/projection._compute_projection_async(...)`.
   - The task:
     - Resolves the source Qdrant collection.
     - Applies sampling to scroll up to `sample_limit` vectors.
     - Computes a 2D projection via PCA / UMAP / t‑SNE (with fallbacks).
     - Assigns category indices and builds a legend.
     - Writes artifacts to disk and a `meta.json` payload describing them.
     - Updates `ProjectionRun.storage_path`, `ProjectionRun.point_count`,
       and `ProjectionRun.meta` via `ProjectionRunRepository.update_metadata(...)`.
     - Marks both the `ProjectionRun` and `Operation` `COMPLETED` (or
       `FAILED` on error) and emits WebSocket progress events.

4. **List / inspect projections (API / UI)**
   - The UI polls `GET /api/v2/collections/{collection_id}/projections`
     via `useCollectionProjections`.
   - Each item is encoded via `ProjectionService._encode_projection(...)`,
     which flattens important metadata (e.g. `color_by`, `degraded`) into
     the `meta` object returned to the client.

5. **View projection (UI)**
   - When a run is `completed`, the user clicks “View”; the UI:
     - Calls `GET /projections/{projection_id}` for metadata.
     - Streams `x`, `y`, `cat`, and `ids` artifacts via
       `GET /projections/{projection_id}/arrays/{artifact}`.
     - Validates array lengths and populates
       `EmbeddingVisualizationTab.activeProjection` and
       `activeProjectionMeta`.
   - `EmbeddingView` (from `embedding-atlas/react`) renders the projection,
     with render modes, labels, tooltips, and selection managed by the tab.

6. **Selection & tooltips**
   - Hover tooltips and selection are resolved via the selection API
     (`POST /projections/{projection_id}/select`) and surfaced in:
     - `useProjectionTooltip` (hover metadata).
     - The selection panel and degraded messaging.

Recompute creates new rows. Existing runs are immutable.

## Core Data Structures

### ProjectionRun and Operation

- `ProjectionRun` (database model):
  - `uuid`: projection run identifier.
  - `collection_id`: owning collection UUID.
  - `reducer`: requested reducer (`"umap"`, `"tsne"`, `"pca"`).
  - `dimensionality`: currently always `2`.
  - `status`: `PENDING`, `RUNNING`, `COMPLETED`, `FAILED`, `CANCELLED`.
  - `point_count`: number of projected points (mirrors `meta.point_count`).
  - `storage_path`: relative path to the artifacts directory.
  - `meta`: JSON dict; see below.

- `Operation`:
  - Tracks the long‑running projection build.
  - Lifecycle:
    - Created in `PENDING`.
    - Advanced to `PROCESSING` in `_process_projection_operation`.
    - Marked `COMPLETED` / `FAILED` by the compute task.
  - Exposed to clients via `operation_id` and `operation_status`.

**Lifecycle**: PENDING → RUNNING/PROCESSING → COMPLETED/FAILED

**Degraded**: Set `meta["degraded"] = True` when reducer fallback, missing artifacts, or invalidation. Frontend shows degraded flag.

## Artifacts

Path: `<data_dir>/semantik/projections/<collection_id>/<projection_uuid>/`

**Arrays** (all same length):
- `x.f32.bin`, `y.f32.bin` - Coordinates
- `ids.i32.bin` - Stable IDs
- `cat.u8.bin` - Category indices (0-254, 255=overflow)

**meta.json** fields:
- Identity: `projection_id`, `collection_id`, `created_at`, `source_vector_collection`
- Counts: `point_count`, `total_count`, `shown_count`, `sampled`, `sample_limit`
- Reducer: `reducer_requested`, `reducer_used`, `reducer_params`, diagnostics
- Color/legend: `color_by`, `legend`, `category_counts`, `original_ids`, `category_map`
- Status: `degraded`

Frontend uses top-level `meta.*` fields (canonical).

## Sampling

**Default**: 200k points (DEFAULT_SAMPLE_LIMIT)
**Config**: `sample_size`, `sample_limit`, `sample_n` (aliases)
**Metadata**: `sampled = (point_count < total_count)`

UI shows "Sampled" badge when true. Tooltip: "Showing X of Y points".

## Render Modes

**Modes**: `auto` | `points` | `density`
**Threshold**: 20k points (DENSITY_THRESHOLD)

Auto switches to density at 20k+ for better performance. Update threshold in `EmbeddingVisualizationTab.tsx` if needed.

## Labels & Legends

**Categories**: 0-254 + overflow (255="Other")
**Label limits**: MIN_POINTS=16, MAX_LABELS=120
**Algorithm**: Compute centroids, filter small clusters, sort by size, truncate

Auto-enabled when legend present.

## Tooltips & Selection

**Tooltips**: LRU cache, debounced, index→id→content via `projectionsV2Api.select`
**Selection**: Maps indices→ids, populates drawer, shows degraded flag when incomplete

Both use `getProjectionPointIndex` for normalization.

## WebGL Fallback

Forces WebGL via patched `navigator.gpu.requestAdapter` (returns null). Embedding Atlas WebGPU requires `shader-f16` not universally supported.

Patch in `embeddingAtlasWebgpuPatch.ts`. Remove when Atlas WebGPU stabilizes.

## Key Files

**Backend**: `projection.py`, `projection_service.py`, `projections.py` (API), `projection_run_repository.py`
**Frontend**: `EmbeddingVisualizationTab.tsx`, `embeddingAtlasWebgpuPatch.ts`, `clusterLabels.ts`, `projectionIndex.ts`, `useProjectionTooltip.ts`
**Tests**: See backend/frontend test dirs

**Invariants**:
- Recompute creates new rows
- `meta.sampled`, `meta.shown_count`, `meta.total_count` always populated
- `meta.degraded` is canonical
- Tests are executable docs
