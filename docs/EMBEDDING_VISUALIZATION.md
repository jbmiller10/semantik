# Embedding Projection & Visualization Pipeline

This document describes the end‑to‑end projection visualization stack used by
Semantik, with a focus on performance, sampling, and contracts that future
humans and LLM agents must preserve.

It is the canonical reference for:

- How a projection run is created, computed, stored, and visualized.
- Where artifacts and metadata live on disk and in the database.
- How sampling and render modes (`points` vs `density`) interact.
- How WebGPU/WebGL behaviour is forced for compatibility.
- How tooltips, selection, legends, and labels are wired.

For lower‑level data‑access details, see also
`docs/DATA_ACCESS_CATALOG.md` (projection sections).

---

## High‑Level Overview

At a high level, a projection run flows through the following stages:

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

Recompute always creates a **new** `ProjectionRun` and `Operation`. Any future
idempotent recompute based on a `metadata_hash` must short‑circuit
**before** creating new rows; existing runs are immutable.

---

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

**Lifecycle invariants**

- Creation:
  - `ProjectionRun(PENDING)`, `Operation(PENDING)`.
- Dispatcher:
  - `_process_projection_operation` links run ↔ operation, sets run
    `RUNNING`, operation `PROCESSING`, then enqueues `compute_projection`.
- Worker:
  - `_compute_projection_async` reasserts `RUNNING`/`PROCESSING` at start.
  - On success: marks both `COMPLETED`.
  - On failure: marks both `FAILED` and populates `error_message`.

Degraded semantics:

- When the reducer falls back, artifacts are missing/invalid, or ingestion
  changes invalidate a run, code sets:
  - `ProjectionRun.meta["degraded"] = True` and/or
  - `ProjectionRun.meta["projection_artifacts"]["degraded"] = True`.
- `_encode_projection` flattens this into a single `meta.degraded: bool`.
- Selection responses (`ProjectionSelectionResponse`) expose
  `degraded: boolean`, which the frontend treats as non‑optional and mirrors
  onto `activeProjectionMeta.degraded`.

See tests:

- `tests/webui/services/test_projection_service.py`
- `tests/webui/api/v2/test_projections.py`
- `tests/webui/test_celery_tasks.py`

These tests are authoritative examples of expected lifecycle and degraded
behaviour.

---

## Artifact Layout and meta.json

The compute task writes a canonical set of artifacts under:

```text
<settings.data_dir>/semantik/projections/<collection_id>/<projection_uuid>/
```

### Binary arrays

- `x.f32.bin` – `Float32Array` of length `point_count` with X coordinates.
- `y.f32.bin` – `Float32Array` of length `point_count` with Y coordinates.
- `ids.i32.bin` – `Int32Array` of length `point_count` with stable point IDs.
- `cat.u8.bin` – `Uint8Array` of length `point_count` with category indices.

All four arrays must share the same `point_count`. This invariant is
validated in `packages/webui/tasks/projection.py` before artifacts are
written and again in the frontend before rendering.

### meta.json schema

`meta.json` (and `ProjectionRun.meta["projection_artifacts"]`) have this
core shape:

- Identity and provenance:
  - `projection_id` (str)
  - `collection_id` (str)
  - `created_at` (ISO 8601 str)
  - `source_vector_collection` (str) – Qdrant collection name.
- Counts and sampling:
  - `point_count` (int) – number of projected points.
  - `total_count` (int) – total vectors available in the collection.
  - `shown_count` (int) – number of points shown in this run; normally
    equals `point_count`.
  - `sampled` (bool) – whether sampling was used
    (`point_count < total_count`).
  - `sample_limit` (int) – the sampling cap used when scrolling vectors.
- Reducer details:
  - `reducer_requested` / `reducer_used` (str).
  - `reducer_params` (dict) – effective reducer configuration.
  - Optional diagnostics (reducer‑specific):
    - `explained_variance_ratio`, `singular_values` (PCA).
    - `kl_divergence` (t‑SNE).
    - `fallback_reason` (when reducer falls back).
- Files and layout:
  - `files` (dict) – file names for `x`, `y`, `ids`, `categories`.
- Colouring and legend:
  - `color_by` (str) – active colour‑by mode.
  - `legend` (list[dict]) – entries `{ index, label, count }`.
  - `category_counts` (dict[str, int]) – counts per category index.
  - `original_ids` (list[str]) – original Qdrant point identifiers.
  - For `color_by == "document_id"`:
    - `category_map` maps document identifiers → legend indices.
- Degraded status:
  - `degraded` (bool) – marks fallback or partial results.

The task also updates top‑level `ProjectionRun.meta` via
`ProjectionRunRepository.update_metadata(...)`:

- `meta["projection_artifacts"]` – full `meta.json` payload.
- `meta["color_by"]` – active colour‑by.
- `meta["legend"]` – legend entries.
- `meta["sampled"]` – same as `meta["projection_artifacts"]["sampled"]`.
- `meta["shown_count"]` – same as `point_count`.
- `meta["total_count"]` – derived from the collection’s `vector_count`.
- `meta["degraded"]` – mirrors `degraded` from the artifacts payload.

`ProjectionService._encode_projection(...)`:

- Treats this top‑level `meta` as canonical.
- Ensures `meta.color_by` is present (preferring `projection_artifacts` and
  falling back to `run.config["color_by"]`).
- Flattens any `degraded` flags into `meta.degraded: bool`.

Frontend consumers (including future LLM agents) should rely on:

- `meta.sampled`
- `meta.shown_count`
- `meta.total_count`
- `meta.color_by`
- `meta.legend`
- `meta.degraded`

and avoid introducing parallel or conflicting fields.

---

## Sampling and Performance

Projections operate over collections that can contain hundreds of thousands
or millions of vectors. Sampling is mandatory for performance and browser
stability.

### Sampling configuration surface

- API schema: `packages/webui/api/v2/schemas.py::ProjectionBuildRequest`
  - `sample_size: int | None` (ge=1).
  - `sample_n: int | None` – alias for `sample_size`.
- Service: `ProjectionService.start_projection_build(...)`
  - Receives `parameters: dict[str, Any]` from the request model.
  - Normalises reducer‑specific config via `_normalise_reducer_config`.
  - Persists sampling controls into the immutable `ProjectionRun.config`:

    ```python
    sample_aliases = ("sample_size", "sample_limit", "sample_n")
    for alias in sample_aliases:
        if alias in parameters and parameters[alias] is not None:
            run_config[alias] = parameters[alias]
    ```

- Worker: `_compute_projection_async(...)`

  ```python
  config = run.config or {}
  configured_sample = config.get("sample_size")
  if configured_sample is None:
      configured_sample = config.get("sample_limit")
  if configured_sample is None:
      configured_sample = config.get("sample_n")
  try:
      sample_limit = int(configured_sample) if configured_sample is not None else DEFAULT_SAMPLE_LIMIT
  except (TypeError, ValueError):
      sample_limit = DEFAULT_SAMPLE_LIMIT
  sample_limit = max(sample_limit, 1)
  ```

  Where:

  - `DEFAULT_SAMPLE_LIMIT = 200_000` – default cap when no explicit
    sampling is requested.
  - Actual Qdrant scrolling stops once either:
    - `sample_limit` points have been collected, or
    - No more records are available.

### Semantics and metadata

- `point_count` is the number of sampled points used to compute and render
  the projection (values in `x`/`y`/`ids`/`cat` arrays).
- `total_vectors` (used to derive `total_count`) is:
  - `collection.vector_count` when available, coerced to be at least
    `point_count`, or
  - `point_count` when no reliable vector count is available.
- `sampled_flag = (point_count < total_vectors)`
  - Exposed as `meta.sampled` and
    `meta.projection_artifacts["sampled"]`.
- `meta.shown_count` is always set to `point_count`.
- `meta.total_count` is set to `total_vectors`.

### UI behaviour and messaging

In `EmbeddingVisualizationTab`:

- Sampling inputs:
  - The recompute dialog exposes a “Sample size” input.
  - When provided, it is encoded as `StartProjectionRequest.sample_size`
    (top‑level field).
  - The backend normalises this into `ProjectionRun.config` so workers can
    derive `sample_limit` deterministically.
- Sampling indicators:
  - `meta.sampled`, `meta.shown_count`, and `meta.total_count` are read
    from `ProjectionMetadata.meta`.
  - A “Sampled” badge is shown when `meta.sampled` is true.
  - The badge tooltip explains how many points are shown vs. the total,
    e.g.:

    > “Showing 50,000 of 2,000,000 points”

  - If metadata is missing or partially populated, the UI falls back to
    safe defaults while keeping the badge semantics stable.

### Performance guidance

- Default behaviour:
  - If no sampling is specified, the worker uses
    `DEFAULT_SAMPLE_LIMIT = 200,000`.
  - This is tuned for Qdrant scrolling and WebGL performance on typical
    hardware.
- For future changes:
  - Keep sampling controls as **caps**, not “exact sizes”.
  - Always populate `meta.sampled`, `meta.shown_count`,
    `meta.total_count`, and `projection_artifacts.sample_limit` so the UI
    can explain what users are seeing.
  - When adjusting defaults or caps, update:
    - `DEFAULT_SAMPLE_LIMIT` in `projection.py`.
    - `SAMPLE_LIMIT_CAP` and related copy in `EmbeddingVisualizationTab.tsx`.
    - This document.

---

## Render Modes and Density Threshold

Rendering hundreds of thousands of points as individual glyphs can
overwhelm the browser. The projection tab implements a simple but effective
render‑mode strategy:

- `RenderMode = 'auto' | 'points' | 'density'`.
- `DENSITY_THRESHOLD = 20_000`.

In `EmbeddingVisualizationTab`:

- `currentRenderMode`:
  - Per‑projection user preference, persisted in local state.
  - Default is `'auto'`.
- `effectiveRenderMode`:

  ```ts
  const effectiveRenderMode = useMemo<'points' | 'density'>(() => {
    if (currentRenderMode !== 'auto') {
      return currentRenderMode;
    }
    return activeProjection.pointCount >= DENSITY_THRESHOLD ? 'density' : 'points';
  }, [currentRenderMode, activeProjection.pointCount]);
  ```

  - This is the concrete mode passed to `EmbeddingView.config.mode`.

UX messaging:

- When `currentRenderMode === 'auto'`, the UI explains:

  > “Auto switches to density at 20,000+ points”

Rationale:

- Below ~20k points:
  - Per‑point rendering remains responsive for pan/zoom/hover on most
    hardware.
  - Using `'points'` preserves fine‑grained structure and is easier to
    interpret visually.
- At or above ~20k points:
  - Individual glyphs become visually noisy and more expensive to render.
  - Density rendering provides better UX and performance; users get an
    immediate sense of cluster structure without the cost of drawing every
    point.

Future tuning:

- If you adjust `DENSITY_THRESHOLD` or introduce more sophisticated
  heuristics (e.g. dynamic thresholds based on device performance), update:
  - `EmbeddingVisualizationTab.tsx` (constant + `effectiveRenderMode`).
  - This document with the new rationale.

---

## Labels, Legends, Tooltips, and Selection

### Legends and label generation

Category indices and legends are produced in the worker:

- `packages/webui/tasks/projection.py`:
  - Category indices for `cat.u8.bin` are assigned sequentially per distinct
    category label, starting at `0`.
  - Once indices `0–254` are exhausted, additional categories are mapped to
    an overflow bucket at `OVERFLOW_CATEGORY_INDEX = 255` with label
    `"Other"`.
  - Every index stored in `cat.u8.bin` either has a corresponding legend
    entry or is grouped into this overflow bucket.

Label heuristics are implemented in
`apps/webui-react/src/utils/clusterLabels.ts`:

- `DEFAULT_MIN_POINTS = 16`
- `DEFAULT_MAX_LABELS = 120`
- Algorithm:
  - Compute centroids per legend index (using `x`, `y`, `category` arrays).
  - Filter categories with fewer than `minPoints` points.
  - Sort labels by cluster size (`priority`).
  - Truncate to `maxLabels`.

In `EmbeddingVisualizationTab`:

- `hasLegend` is true when `meta.legend` is non‑empty.
- `clusterLabels`:
  - Derived from `activeProjection.arrays` and `activeProjectionMeta.legend`.
  - Only computed when `labelsEnabled` and `hasLegend` are true.
  - Memoised to avoid recomputation on every render.
  - Passed to `EmbeddingView.labels` as `EmbeddingLabel[]`.
- Labels are auto‑enabled when a non‑empty legend is present and at least
  one label is produced.
- Users can toggle labels via the “Show labels” checkbox.

Tests:

- `apps/webui-react/src/utils/__tests__/clusterLabels.test.ts`
- `apps/webui-react/src/components/__tests__/EmbeddingVisualizationTab.test.tsx`
  - Assert that labels are enabled by default when a legend is present and
    that toggling behaves as expected.

### Tooltip pipeline

Tooltips follow a shared, index‑based contract:

- `getProjectionPointIndex` (`projectionIndex.ts`):
  - Normalises various selection values into a `number | null`:
    - `DataPoint.identifier` (preferred when it is a non‑negative integer; Embedding Atlas uses this as the row id in plain `EmbeddingView` mode);
    - raw numeric indices;
    - `DataPoint` objects with an `index` field;
    - legacy shapes with `rowIndex`, `pointIndex`, or `i`;
    - nested `fields.index` / `fields.rowIndex` / `fields.pointIndex` / `fields.i` shapes that may appear in future Atlas versions.
  - Ensures the result is a non‑negative integer.

- `useProjectionTooltip`:
  - Accepts `collectionId`, `projectionId`, and the `ids` array
    (`Int32Array`).
  - Maintains an LRU cache keyed by selected id with TTL and a small cap
    on in‑flight requests.
  - On hover:
    - Debounces requests.
    - Resolves the hovered point index into an id using `ids`.
    - Calls `projectionsV2Api.select` with a single id.
    - Populates tooltip state with truncated content previews and document
      identifiers.

- `EmbeddingVisualizationTab`:
  - Wires `EmbeddingView.onTooltip` to `handleTooltipEvent`, which:
    - Stores the active `DataPoint` in `activeTooltip`.
    - Forwards it to `useProjectionTooltip.handleTooltip`.
  - Supplies `tooltipCustomProps` and `tooltipState` to
    `EmbeddingTooltipAdapter`, which renders the React tooltip UI.

### Selection pipeline

Selection is similarly index‑based:

- `EmbeddingView.onSelection` is wired to `handleSelectionChange`, which:
  - Normalises the incoming indices.
  - Maps indices through `ids` to stable integer ids.
  - Calls `projectionsV2Api.select` with the id list.
  - Populates the selection panel with:
    - `ProjectionSelectionItem[]` (document + chunk metadata).
    - `missing_ids` (ids that could not be resolved).
    - `degraded` flag, mirrored onto `activeProjectionMeta.degraded`.
  - Drives:
    - The selection drawer UI.
    - “Open” (document viewer) actions.
    - “Find Similar” semantic search via `searchV2Api.search`.
    - Degraded messaging when selection results may be incomplete.

Contracts:

- `ProjectionSelectionResponse` (frontend type) is treated as:

  ```ts
  export interface ProjectionSelectionResponse {
    projection_id: string;
    items: ProjectionSelectionItem[];
    missing_ids: number[];
    degraded: boolean;
  }
  ```

- `degraded` is always present and is the canonical flag for “selection
  results may be incomplete”.

Tests:

- `apps/webui-react/src/hooks/__tests__/useProjectionTooltip.test.ts`
- `apps/webui-react/src/components/__tests__/EmbeddingVisualizationTab.test.tsx`

These tests encode the expected UX around tooltips, selection, and degraded
behaviour; changes should be validated against them.

---

## WebGPU / WebGL Behaviour

The project currently forces Embedding Atlas to use WebGL for maximum
compatibility.

File: `apps/webui-react/src/utils/embeddingAtlasWebgpuPatch.ts`

Behaviour:

- Installs a patched `navigator.gpu.requestAdapter` that:
  - Calls the original `requestAdapter` once (for any side‑effects).
  - Returns `null` to indicate that no WebGPU adapter is available.
  - Sets `window.__embeddingAtlasWebgpuFallback = true`.
  - Logs a console warning:

    > “[EmbeddingAtlas] Forcing WebGL renderer (WebGPU disabled for compatibility).”

- Only runs:
  - In the browser (no‑ops on the server).
  - When `navigator.gpu` and `navigator.gpu.requestAdapter` exist.
- Is invoked at the top of `EmbeddingVisualizationTab.tsx` via
  `ensureEmbeddingAtlasWebgpuCompatibility()`.

Rationale:

- Embedding Atlas’ WebGPU path currently requires optional features such as
  `shader-f16` that are not universally supported.
- On unsupported adapters, `requestDevice` rejects and the projection view
  remains blank.
- Forcing the WebGL path keeps the visualization robust on a wide range of
  hardware and browsers.

Future work:

- Once Embedding Atlas’ WebGPU implementation stabilises, this shim may be
  relaxed or removed.
- Any change to WebGPU/WebGL behaviour should:
  - Preserve a clear detection mechanism (e.g.
    `window.__embeddingAtlasWebgpuFallback`).
  - Be documented here so future agents understand when and why WebGPU is
    enabled.

---

## LLM Onboarding Checklist

If you are a stateless LLM agent implementing changes in this area, start
by loading and understanding:

**Docs**

- `docs/EMBEDDING_VISUALIZATION.md` (this document) – overall pipeline,
  contracts, and invariants.
- `docs/DATA_ACCESS_CATALOG.md` – projection‑related data‑access points.
- `tickets/4-projections-backend-artifacts-and-metadata-contracts.md` –
  detailed contract notes.
- `tickets/Complete/4-projections-frontend-tab-state-and-contracts.md` –
  frontend state machine and EmbeddingView contracts.

**Backend**

- `packages/webui/tasks/projection.py`
  - Sampling behaviour and Qdrant scrolling.
  - Reducer implementations and fallbacks.
  - Artifact layout and `meta.json`.
- `packages/webui/services/projection_service.py`
  - `start_projection_build` sampling config surface.
  - `_encode_projection` meta flattening and degraded semantics.
- `packages/webui/api/v2/projections.py`
  - Projection endpoints (`/projections`, `/arrays/{artifact}`, `/select`).
- `packages/shared/database/repositories/projection_run_repository.py`
  - `update_metadata` and lifecycle updates.

**Frontend**

- `apps/webui-react/src/components/EmbeddingVisualizationTab.tsx`
  - Projection lifecycle, render modes, sampling UX, labels, selection.
- `apps/webui-react/src/utils/embeddingAtlasWebgpuPatch.ts`
  - WebGPU/WebGL shim.
- `apps/webui-react/src/utils/clusterLabels.ts`
  - Label heuristics and performance considerations.
- `apps/webui-react/src/utils/projectionIndex.ts`
  - Index normalisation for tooltips/selection.
- `apps/webui-react/src/hooks/useProjectionTooltip.ts`
  - Tooltip caching and selection calls.

**Tests**

- Backend:
  - `tests/webui/test_projection_compute_regression.py`
  - `tests/webui/api/v2/test_projections.py`
  - `tests/webui/services/test_projection_service.py`
  - `tests/webui/test_celery_tasks.py`
- Frontend:
  - `apps/webui-react/src/components/__tests__/EmbeddingVisualizationTab.test.tsx`
  - `apps/webui-react/src/hooks/__tests__/useProjectionTooltip.test.ts`
  - `apps/webui-react/src/utils/__tests__/clusterLabels.test.ts`

**Invariants to preserve**

- Artifact layout and `meta.json` schema as described above.
- Sampling metadata:
  - `meta.sampled`, `meta.shown_count`, `meta.total_count`.
  - `projection_artifacts.sample_limit`.
- Lifecycle:
  - Recompute always creates a new `ProjectionRun` and `Operation`.
  - Idempotent recompute (if introduced) must short‑circuit before
    creating rows.
- Degraded semantics:
  - `ProjectionRun.meta["degraded"]` and
    `projection_artifacts["degraded"]` flattened to `meta.degraded`.
  - `ProjectionSelectionResponse.degraded` treated as canonical in the UI.
- WebGPU/WebGL behaviour:
  - The compatibility shim remains in place or is consciously updated,
    with reasoning captured here.

When in doubt, use the tests listed above as executable documentation for
expected behaviour. Any change that affects sampling, degraded semantics,
artifact layout, or render thresholds should be accompanied by updated
tests and an update to this document.
