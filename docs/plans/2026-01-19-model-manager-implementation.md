# Model Manager Implementation Plan (Curated-Only v1)

**Date:** 2026-01-19

## Goals

Implement a **Models** tab in Settings that lets users:

- Browse a curated set of *downloadable* local models:
  - **Embedding**: from `packages/shared/embedding/models.py`
  - **Local LLMs**: from `packages/shared/llm/model_registry.yaml` (provider `local` only)
  - **Rerankers**: from built-in reranker plugin metadata (`packages/shared/plugins/builtins/qwen3_reranker.py`)
  - **SPLADE**: from existing SPLADE plugin defaults + known SPLADE IDs used elsewhere (e.g., `packages/vecpipe/memory_utils.py`)
- **Download** a curated model into the HuggingFace cache (background Celery task with progress).
- **Delete** a model from the HuggingFace cache (background Celery task, invoked via a slash-safe endpoint).
- Restrict all Model Manager UI + API access to **superusers only** (shared HF cache is global).
- Make disk usage understandable by reporting both:
  - total HF cache size (entire hub cache dir), and
  - “managed” cache size attributable to curated models.

## Non-Goals (Deferred to v2)

- No DB-backed “custom models” registry.
- No per-user custom model overrides (dimension/pooling/prefixes/etc.).
- No attempt to introspect “in use” from plugin configs or other config stores beyond what’s specified below.
- No UI for managing arbitrary (non-curated) Hub repos in the cache beyond size transparency.

## Key Decisions (From Review + Follow-up)

1. **No new unified YAML registry in v1.** Aggregate existing registries/plugin metadata into a unified API response.
2. **Slash-safe deletion endpoint:** use `DELETE /api/v2/models/cache?model_id=...` (not a path param).
3. **Download & delete run in Celery tasks** to avoid HF cache permission issues across container users.
4. **Superuser-only:** the Models tab and all `/api/v2/models*` endpoints require `is_superuser == true`.
5. **De-duplicate + cross-op exclusion:** store a single “active op” per `model_id` in Redis and return the existing task if the same op is already running; block starting a different op while one is active.
6. **In-use policy**
   - **Block** deletion if *any collection* references the model (global, not per-user): `collections.embedding_model == model_id`.
   - **Warn (do not block)** deletion if referenced by:
     - `user_preferences.default_embedding_model == model_id`
     - `llm_provider_configs.high_quality_model == model_id` or `low_quality_model == model_id`
   - **Do not inspect plugin configs** (reranker/sparse/other) for usage.
7. **Delete warnings require confirmation:** deletion is a two-step flow so warnings can be shown before enqueuing.
8. **Cache size semantics:** cache size numbers refer to the resolved HuggingFace Hub cache dir (entire cache). Also return a “managed” subset size for curated models so totals are explainable.
9. **Additional deletion warnings (confirmation required):**
   - Deleting the configured system default embedding model (`shared.config.settings.DEFAULT_EMBEDDING_MODEL`) should warn even if not referenced by collections.
   - Deleting a model that appears currently loaded in VecPipe should warn (best-effort; does not block).
10. **Operational isolation (recommended):** route model-manager Celery tasks to a dedicated queue (e.g., `model-manager`) to reduce contention with ingestion/reindex tasks.
11. **Idempotent operations:** repeated “start download/delete” calls for the same `model_id` should be safe. If the model is already in the desired state (already installed / already absent), return a terminal response without enqueuing unnecessary work.

## Architecture

- Backend:
  - Model Manager API under `/api/v2/models` (new router).
  - HuggingFace cache scanning/deletion via `huggingface_hub.scan_cache_dir()`.
  - Background operations via Celery tasks (`download`, `delete`).
  - Progress + de-dupe state stored in Redis.
  - Superuser-only access control for all endpoints in the router.
- Frontend:
  - New Settings tab: **Models**
  - Horizontal sub-tabs: Embedding / Local LLMs / Rerankers / SPLADE
  - TanStack Query for list + polling task progress
  - Remove/omit all “custom model” UI
  - Models tab visible to superusers only

---

## Phase 0: Contracts + Guardrails

### Task 0: Contracts + Guardrails

**Purpose**
Lock down API contracts, access control, and operational guardrails so the backend/frontend implementations can proceed in parallel with minimal churn.

**Items to finalize**
- API schemas + status codes:
  - Use `409 Conflict` for cross-op exclusion, in-use blocks, and “requires confirmation”.
  - Define idempotent responses for “already installed” and “not installed” cases (no enqueue).
- Cache size semantics and naming:
  - `total_cache_size_mb` (entire Hub cache dir) vs `managed_cache_size_mb` (curated-installed subset) vs `unmanaged_*`.
- Superuser-only gating:
  - Enforce `is_superuser == true` for all `/api/v2/models*` endpoints.
  - Hide the Models tab for non-superusers in the UI.
- Queue routing decision:
  - Decide whether to route model-manager tasks to a dedicated queue and how the worker consumes it.

**Verification**
- Add a minimal API contract test or schema test for the “requires confirmation” and idempotency response shapes (even if the endpoints are implemented in later phases).

---

## Phase 1A: Backend Read-Only (Curated + Cache Scan + List API)

### Task 1: Curated Model Aggregation (No New YAML)

**Files**
- Create: `packages/shared/model_manager/__init__.py`
- Create: `packages/shared/model_manager/curated_registry.py`

**Purpose**
Provide a single “curated models” view for the Model Manager API by aggregating existing sources:

- Embeddings: `shared.embedding.models.MODEL_CONFIGS`
- Local LLMs: `shared.llm.model_registry.get_models_by_provider("local")`
- Rerankers: `shared.plugins.builtins.qwen3_reranker.SUPPORTED_MODELS`
- SPLADE: `shared.plugins.builtins.splade_indexer.DEFAULT_MODEL` + known SPLADE IDs (e.g., `naver/splade-v3`)

**Normalization**
Define a normalized representation for the API layer (dataclass or TypedDict):

- `id` (HuggingFace repo id)
- `name` (human label)
- `description`
- `model_type`: `embedding | llm | reranker | splade`
- `memory_mb` (dict of quantization → MB; best-effort; empty if unknown)
- Optional type-specific fields:
  - Embedding: `dimension`, `max_sequence_length`, `pooling_method`, `is_asymmetric`, `query_prefix`, `document_prefix`, `default_query_instruction`
  - LLM: `context_window`

**De-duplication + precedence**
- Treat each curated entry as uniquely identified by `(id, model_type)` (not just `id`).
- When multiple sources emit the same `(id, model_type)`, merge deterministically:
  1. Prefer the entry with richer type-specific fields populated (e.g., embedding dimension/max length/pooling).
  2. On conflicts for the same field, use a fixed precedence order by source:
     - Embedding: `shared.embedding.models.MODEL_CONFIGS` > anything else
     - LLM: `shared.llm.model_registry` > anything else
     - Reranker: plugin metadata list > anything else
     - SPLADE: plugin defaults > known SPLADE IDs list
- Ensure stable ordering in responses after merge: sort by `(model_type, name or id)` so UI ordering is deterministic.

**Notes**
- v1 does not require `download_size_mb` for uninstalled models (avoid HuggingFace Hub API calls during list).
- Keep aggregation logic pure (no DB, no HF cache scanning).

**Verification**
- Add a unit test verifying the aggregator returns:
  - ≥1 embedding model
  - ≥1 local LLM model (if registry includes local)
  - ≥1 reranker model id
  - ≥1 SPLADE model id

---

### Task 2: HuggingFace Cache Utilities (Correct Cache Path)

**Files**
- Create: `packages/shared/model_manager/hf_cache.py`

**Purpose**
- Resolve the hub cache directory in Semantik’s container setup.
- Scan installed HF model repos and sizes.
- Delete a repo safely (all revisions).

**Important**
- `scan_cache_dir()` is synchronous and can be slow. FastAPI endpoints must call it off the event loop (e.g., `asyncio.to_thread`) and rely on TTL caching to avoid repeated scans.
- Deletion should use HuggingFace Hub’s cache manager strategy:
  - `scan_cache_dir(...)` → `HFCacheInfo`
  - Find matching `CachedRepoInfo` for `repo_id == model_id` (repo_type `"model"`)
  - Collect all `commit_hash` values from `repo.revisions`
  - `HFCacheInfo.delete_revisions(*commit_hashes).execute()`

**Scan performance**
- `huggingface_hub.scan_cache_dir()` can be expensive on large caches.
- Add a small TTL cache (e.g., 5–30 seconds; choose 15s default) for scan results keyed by resolved cache dir.
  - This reduces repeated full scans when the UI refetches or multiple requests hit the endpoint.
  - Provide a `force_refresh` option (API-level) to bypass TTL for debugging/admin use.
  - Note: TTL caching is per-process (each WebUI worker maintains its own TTL).

**Cache directory resolution**
Use precedence:

1. `HF_HUB_CACHE` if set
2. `HF_HOME + "/hub"` if `HF_HOME` is set
3. `~/.cache/huggingface/hub`

(Semantik Compose uses `HF_HOME=/app/.cache/huggingface` and mounts `${HF_CACHE_DIR:-./models}:/app/.cache/huggingface`.)

---

### Task 5A: Model Manager API (Read-only List)

**Files**
- Create: `packages/webui/api/v2/model_manager_schemas.py`
- Create: `packages/webui/api/v2/model_manager.py`
- Modify: `packages/webui/main.py` (include router)

**Authorization**
- All endpoints in this router are **superuser-only** (shared HF cache is global across services/users).

**Endpoints (Phase 1A scope)**
1. `GET /api/v2/models`
   - Query params: `model_type?`, `installed_only?`
   - Response:
     - `models`: curated models with:
       - `is_installed`, `size_on_disk_mb`
       - `used_by_collections` (for embedding models)
       - `active_download_task_id?`, `active_delete_task_id?` (optional; populated once Phase 1B task state exists)
     - Cache size fields (optional; only when explicitly requested):
       - `total_cache_size_mb` (entire HF hub cache dir)
       - `managed_cache_size_mb` (sum of sizes for installed curated model repos)
       - `unmanaged_cache_size_mb` (`total - managed`)
       - `unmanaged_repo_count` (installed repos not in curated list; best-effort)
   - Additional query params:
     - `include_cache_size?` (default false)
     - `force_refresh_cache?` (default false; bypass HF cache scan TTL)

---

## Phase 1B: Backend Task Infrastructure (Redis + Celery)

### Task 3: Redis Progress + Task De-dupe Keys

**Files**
- Create: `packages/webui/model_manager/task_state.py`

**Implementation note**
Reuse the existing Redis infrastructure:
- FastAPI endpoints use `RedisManager.async_client()` (via `webui.services.factory.get_redis_manager()`).
- Celery tasks use `RedisManager.sync_client`.

**Redis keys**

- Active op per model (cross-op exclusion):
  - `model-manager:active:{model_id}` → `download:{task_id}` | `delete:{task_id}`
- Task progress payload:
  - `model-manager:task:{task_id}` → hash

**Task progress fields (hash)**
- `task_id`, `model_id`
- `operation`: `download | delete`
- `status`: `pending | running | completed | failed`
- `bytes_downloaded`, `bytes_total` (download only; may be 0/unknown)
- `error` (optional)
- `updated_at` (epoch seconds)

**TTL**
- Progress keys: 24 hours
- Active key: 24 hours (cleared explicitly on completion/failure)
  - Refresh the active-key TTL on every progress update to prevent expiry mid-download on slow links.

**De-dupe behavior**
- Starting a download/delete:
  - Use an **atomic claim** to make de-dupe race-free:
    - Attempt `SET model-manager:active:{model_id} {op}:{task_id} NX EX {ttl}`.
    - If the SET fails:
      - Read the existing value.
      - If it is the **same op**, return the existing `task_id` and do **not** enqueue another task.
      - If it is a **different op**, return `409 Conflict` (do not enqueue).
  - If the claim succeeds, enqueue the Celery task with the claimed `task_id`, and initialize the progress hash immediately (avoid “first poll returns null” race).
  - If Celery enqueue fails, clear the active key and set the progress hash to `failed` with an error.

**Resumability**
- `GET /api/v2/models` should include `active_download_task_id` / `active_delete_task_id` per model (if present).
  - This allows the UI to resume polling after refresh/new session without relying on client-side state.

---

### Task 4: Celery Tasks (Download + Delete)

**Files**
- Create: `packages/webui/tasks/model_manager.py`
- Modify: `packages/webui/tasks/__init__.py` (import module so Celery registers tasks)

**Task registration**
`packages/webui/tasks/__init__.py` must:
- Import the new module so Celery registers tasks
- Add it to the internal proxy module list (`_PROXY_MODULES`) so tests/monkeypatching via `webui.tasks` can see it

**Tasks**

1. `model_manager.download_model(model_id: str)`
   - Uses `huggingface_hub.snapshot_download(repo_id=model_id, ...)` into the shared HF cache.
   - Auth: rely on standard Hub auth resolution (env var `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN`, or token file from `huggingface-cli login`).
   - Writes progress to Redis.
   - Clears active key on completion/failure.

2. `model_manager.delete_model(model_id: str)`
   - Uses `scan_cache_dir(...)` + delete strategy to remove all revisions of `model_id`.
   - Writes progress to Redis (bytes fields can remain 0; deletion progress is optional).
   - Clears active key on completion/failure.

**Progress tracking**
- Implement via a custom `tqdm_class` (preferred) or best-effort updates:
  - If bytes are not available, still update `status` transitions so UI remains usable.

**Retries + failure classes**
- Retry transient network and hub errors (bounded retries with backoff).
- Fail fast with actionable error messages for:
  - Gated/unauthorized models (e.g., missing token / 401 / 403)
  - Offline mode (`HF_HUB_OFFLINE=true`)
  - Disk full / no space left on device
  - Permission denied in cache dir

**Time limits**
- Set explicit per-task time limits (override global defaults) to accommodate large model downloads on slower connections.
  - Example: `soft_time_limit=4h`, `time_limit=6h` for download/delete tasks.

**Concurrency**
- v1 allows parallel downloads/deletes across different `model_id` values, bounded by the Celery worker concurrency.
  - De-dupe remains per-model via Redis active keys.

**Queue routing (recommended)**
- Route these tasks to a dedicated Celery queue (e.g., `model-manager`) so large downloads don’t starve ingestion/reindex tasks.
  - If adopted, ensure the worker process consumes this queue (e.g., update `docker-entrypoint.sh` worker command to include `-Q celery,model-manager`, or add a dedicated worker service for model-manager tasks).

**Important**
- Do **not** add task imports to `packages/webui/celery_app.py`. This repo registers tasks by importing modules from `packages/webui/tasks/__init__.py`.

---

### Task 5B: Task Progress API

**Files**
- Modify: `packages/webui/api/v2/model_manager_schemas.py`
- Modify: `packages/webui/api/v2/model_manager.py`

**Endpoints (Phase 1B scope)**
1. `GET /api/v2/models/tasks/{task_id}`
   - Returns the Redis progress payload for either a download or delete.

---

## Phase 1C: Backend Write APIs (Download + Usage + Delete)

### Task 5C: Model Manager Write APIs

**Files**
- Modify: `packages/webui/api/v2/model_manager_schemas.py`
- Modify: `packages/webui/api/v2/model_manager.py`

**Endpoints (Phase 1C scope)**
1. `POST /api/v2/models/download`
   - Body: `{ "model_id": "Qwen/Qwen3-Embedding-0.6B" }`
   - Curated-only validation: `model_id` must be present in aggregated curated list.
   - De-dupe: return existing task if already downloading.
   - Cross-op exclusion: if a delete is active for the model, return `409 Conflict`.
   - Idempotency: if the model is already installed, return `status=completed` (no Celery enqueue).
   - Response: `{ task_id, model_id, status }`

2. `GET /api/v2/models/usage?model_id=...`
   - Purpose: preflight usage checks so the UI can show blockers/warnings before enqueueing deletion.
   - Response:
     - `model_id: string`
     - `is_installed: bool`
     - `size_on_disk_mb: number` (0 if not installed/unknown)
     - `estimated_freed_size_mb: number` (best-effort; 0 if not installed/unknown)
     - `blocked_by_collections: string[]` (collection names)
     - `warnings: string[]` (aggregate counts; avoid user identity)
     - `is_default_embedding_model: bool`
     - `loaded_in_vecpipe: bool` (best-effort)
     - `loaded_vecpipe_model_types: string[]` (best-effort; e.g., `embedding`, `reranker`, `llm`)
     - `can_delete: bool` (false if blocked by collections)
     - `requires_confirmation: bool` (true if any warnings exist)

3. `DELETE /api/v2/models/cache?model_id=...&confirm=true|false`
   - De-dupe: return existing delete task if already deleting.
   - Cross-op exclusion: if a download is active for the model, return `409 Conflict`.
   - **Block** if any collection uses the model (global): include collection names in `409 Conflict` response.
   - **Warn (do not block)** if referenced by:
     - UserPreferences.default_embedding_model
     - LLMProviderConfig high/low model fields
   - Confirmation contract:
     - If warnings exist and `confirm != true`, return `409 Conflict` with `requires_confirmation=true` + warnings (do not enqueue).
     - If `confirm == true`, enqueue delete and return `{ task_id, model_id, status, warnings: string[] }`.
   - Idempotency: if the model is not installed, return `status=completed` (no Celery enqueue).

**Usage detection details**
- Blocking “in use” check:
  - Query all collections where `embedding_model == model_id`; return their names.
- Warning-only checks:
  - Count users whose `user_preferences.default_embedding_model == model_id`.
  - Count users whose `llm_provider_configs.high_quality_model == model_id OR low_quality_model == model_id`.
  - Return warnings as aggregate counts (avoid leaking user identity).
- Additional warnings (confirmation required):
  - If `model_id == settings.DEFAULT_EMBEDDING_MODEL`, include a warning and set `is_default_embedding_model=true`.
  - Best-effort: query VecPipe `/memory/models` and if any loaded model matches `model_id`, set `loaded_in_vecpipe=true` and include a warning (do not block).

---

## Phase 2A: Frontend Read-Only (List + Filters + Basic Cards)

### Task 6: Frontend Types (New File; No Conflicts)

**Files**
- Create: `apps/webui-react/src/types/model-manager.ts`

**Notes**
- Do not create `apps/webui-react/src/types/model.ts`.
- Include:
  - `ModelType = 'embedding' | 'llm' | 'reranker' | 'splade'`
  - `ModelManagerModel`, `ModelListResponse`
  - `TaskProgress`, `StartDownloadResponse`, `StartDeleteResponse`
  - `ModelUsageResponse` (for `GET /api/v2/models/usage`)
  - `warnings?: string[]` on delete start response
  - `active_download_task_id?: string` and `active_delete_task_id?: string` on `ModelManagerModel`
  - Cache size fields on `ModelListResponse` (optional; present only when requested):
    - `total_cache_size_mb?: number`
    - `managed_cache_size_mb?: number`
    - `unmanaged_cache_size_mb?: number`
    - `unmanaged_repo_count?: number`

---

### Task 7: Frontend API Client (Rename to Avoid Conflict)

**Files**
- Create: `apps/webui-react/src/services/api/v2/model-manager.ts`
- Modify: `apps/webui-react/src/services/api/v2/index.ts` (export it)

**Why**
`apps/webui-react/src/services/api/v2/models.ts` already exists for legacy `/api/models` embedding discovery; do not overwrite it.

**API surface**
- `listModels(params)` should support:
  - `model_type?`, `installed_only?`
  - `include_cache_size?` (optional)
  - `force_refresh_cache?` (optional; use sparingly)
- `startDownload(model_id)`
- `getTask(task_id)`
- `getUsage(model_id)`
- `deleteFromCache(model_id, confirm?: boolean)`

---

### Task 8A: Hooks (List Only)

**Files**
- Create: `apps/webui-react/src/hooks/useModelManager.ts`

**Behavior**
- `useModelManagerModels(type, installedOnly, includeCacheSize?)`
  - Avoid background polling of the model list; cache-scan is potentially expensive even with TTL caching.
  - Refetch the list on demand (e.g., after a task reaches terminal state).

---

### Task 9A: UI (Read-only Models Tab + Settings Integration)

**Files**
- Create: `apps/webui-react/src/components/settings/model-manager/ModelCard.tsx`
- Create: `apps/webui-react/src/components/settings/model-manager/ModelsSettings.tsx`
- Modify: `apps/webui-react/src/pages/SettingsPage.tsx`

**UI requirements**
- No Add Custom button / modal.
- Models tab visible to **superusers only**.
- Tabs: Embedding / Local LLMs / Rerankers / SPLADE
- Search + status filter (All / Installed / Available)
- Render model cards with installed status + disk size.
- Action buttons may be rendered but should be disabled or hidden until Phase 2B/2C are implemented.

**SettingsPage integration**
Current Settings page is driven by:
- a `SettingsTab` union type
- `tabs: TabConfig[]`
- conditional rendering per tab

Add:
- `models` to `SettingsTab`
- a new tab entry (icon e.g. `Box` or `Cpu`) with `requiresSuperuser: true`
- render `<ModelsSettings />` when active

---

## Phase 2B: Frontend Download Flow (Start + Poll + Resumability)

### Task 8B: Hooks (Download + Task Polling)

**Files**
- Modify: `apps/webui-react/src/hooks/useModelManager.ts`

**Behavior**
- `useStartModelDownload()`:
  - Starts download and stores returned `task_id` keyed by `model_id`.
  - Polls `/api/v2/models/tasks/{task_id}` until terminal status.
- Resumability:
  - When `useModelManagerModels()` returns `active_download_task_id`, automatically begin polling so progress survives refresh.
- Post-terminal refresh:
  - Refetch the model list on terminal status (optionally with `force_refresh_cache=true` if needed to bypass HF scan TTL).

**Polling edge cases**
- Don’t stop polling just because the first fetch returns 404/null; allow a short grace window (race with task initialization).

---

### Task 9B: UI (Download Action + Progress)

**Files**
- Modify: `apps/webui-react/src/components/settings/model-manager/ModelCard.tsx`
- Modify: `apps/webui-react/src/components/settings/model-manager/ModelsSettings.tsx`

**UI requirements**
- Download button:
  - Enabled for available (not installed) models.
  - Shows progress if `active_download_task_id` exists.
  - Handles terminal states (success/failure) and triggers a list refresh.

---

## Phase 2C: Frontend Delete Flow (Usage Preflight + Confirm + Task)

### Task 8C: Hooks (Delete Preflight + Confirmation + Task Polling)

**Files**
- Modify: `apps/webui-react/src/hooks/useModelManager.ts`

**Behavior**
- `useStartModelDelete()`:
  - Preflight via `GET /api/v2/models/usage?model_id=...`.
  - If blocked, show error + do not start delete.
  - If warnings exist, show confirmation dialog.
  - Start delete via `DELETE /api/v2/models/cache?model_id=...&confirm=true`.
- Resumability:
  - When `useModelManagerModels()` returns `active_delete_task_id`, automatically begin polling so progress survives refresh.
- Post-terminal refresh:
  - Refetch the model list on terminal status (optionally with `force_refresh_cache=true` if needed to bypass HF scan TTL).

---

### Task 9C: UI (Delete Confirm + Warnings)

**Files**
- Modify: `apps/webui-react/src/components/settings/model-manager/ModelCard.tsx`
- Modify: `apps/webui-react/src/components/settings/model-manager/ModelsSettings.tsx`

**UI requirements**
- Delete button:
  - Disabled if `used_by_collections.length > 0`.
  - Confirmation dialog should show:
    - estimated freed disk (`estimated_freed_size_mb`)
    - warnings (prefs/LLM config counts, default embedding model warning, vecpipe-loaded warning)
  - Requires explicit confirmation before starting deletion if usage preflight returns warnings.

---

## Phase 2D: Hardening + UX/Perf Polish ✅ COMPLETED (2026-01-20)

**Purpose**
Address Phase 0–2C follow-ups discovered during implementation review:
- curated registry de-dup/merge rules
- TTL + force-refresh correctness
- curated-only delete semantics
- more accurate task statuses + progress UX
- reduce unnecessary cache scans
- optional list endpoint batching (DB/Redis)

### Task 2D-1: Curated Registry De-dupe + Precedence Merge

**Files**
- Modify: `packages/shared/model_manager/curated_registry.py`
- Add/Modify tests: `tests/unit/test_model_manager_curated_registry.py`

**Requirements**
- Treat each curated entry as uniquely identified by `(id, model_type)`.
- If multiple sources emit the same `(id, model_type)`, merge deterministically:
  1. Prefer the entry with richer type-specific fields populated.
  2. On conflicts for the same field, use fixed precedence by source:
     - Embedding: `shared.embedding.models.MODEL_CONFIGS` > anything else
     - LLM: `shared.llm.model_registry` > anything else
     - Reranker: plugin metadata list > anything else
     - SPLADE: plugin defaults > known SPLADE IDs list
- Maintain deterministic ordering after merge: sort by `(model_type, name or id)`.

**Verification**
- Add unit coverage that asserts:
  - no duplicate `(id, model_type)` pairs
  - deterministic ordering
  - merge precedence behaves as expected (use a small mocked overlap scenario)

---

### Task 1D-2: Backend TTL Alignment + Force-Refresh Correctness

**Files**
- Modify: `packages/webui/model_manager/task_state.py`
- Modify: `packages/webui/model_manager/constants.py` (if needed)
- Modify: `packages/shared/model_manager/hf_cache.py`
- Modify: `packages/webui/api/v2/model_manager.py`
- Add/Modify tests: `tests/unit/test_model_manager_hf_cache.py`

**Requirements**
- Align active-op TTL to the plan’s 24h semantics (and refresh-on-progress behavior):
  - Active key TTL: 24h (cleared explicitly on completion/failure; refreshed during running)
  - Progress key TTL: 24h
- Apply `force_refresh_cache=true` consistently:
  - `GET /api/v2/models?include_cache_size=true&force_refresh_cache=true` must bypass the HF scan TTL for:
    - installed status, and
    - cache size breakdown.
- Prefer a single constants source of truth (avoid duplicated TTL constants across modules).

**Verification**
- Extend unit tests to assert:
  - force-refresh yields a new scan result (via mocking)
  - include_cache_size path respects force_refresh_cache

---

### Task 1D-3: Curated-Only Delete Guardrails

**Files**
- Modify: `packages/webui/api/v2/model_manager.py`
- Add/Modify tests: `tests/unit/test_model_manager_write_api.py`

**Requirements**
- Enforce curated-only validation for deletion, consistent with download:
  - `DELETE /api/v2/models/cache?model_id=...` returns `400` if `model_id` is not in curated list.
  - (Optional future escape hatch: explicit `allow_unmanaged=true` admin-only flag; deferred unless needed.)
- Keep slash-safe semantics (`model_id` remains a query param).

**Verification**
- Add test: delete of non-curated model id returns `400`.

---

### Task 1D-4: Task Status + Progress UX Improvements

**Files**
- Modify: `packages/webui/tasks/model_manager.py`
- Modify: `packages/webui/api/v2/model_manager.py` (status mapping if needed)
- Modify: `apps/webui-react/src/hooks/useModelManager.ts`
- Modify: `apps/webui-react/src/components/settings/model-manager/ModelCard.tsx`

**Requirements**
- Fix “not found in cache” semantics in delete task:
  - If repo is missing, treat as idempotent no-op:
    - progress `status=not_installed` (or equivalent terminal no-op), and
    - no “error” field on a successful no-op.
- Improve download progress reporting:
  - Prefer bytes-based progress updates from HuggingFace download where feasible.
  - If bytes can’t be determined, keep status transitions accurate and switch UI to an indeterminate progress indicator (not a 0% bar).

**Verification**
- Add/adjust unit tests to assert:
  - delete “repo missing” yields a terminal no-op status
  - UI renders indeterminate progress when `bytes_total==0` and status is running

---

### Task 1D-5 (Optional): Batch DB + Redis Lookups in `GET /api/v2/models`

**Files**
- Modify: `packages/webui/api/v2/model_manager.py`

**Requirements**
- Replace per-model collection usage queries with a single batched query keyed by `embedding_model`.
- Replace per-model Redis gets with a single batched fetch (e.g., `MGET`) for active keys.
- Preserve response shape and deterministic ordering.

**Verification**
- Keep existing API contract tests passing; add a small regression test if feasible.

---

### Task 2D-6: Reduce Unnecessary HF Cache Scans from UI

**Files**
- Modify: `apps/webui-react/src/hooks/useModelManager.ts`
- Modify: `apps/webui-react/src/components/settings/model-manager/ModelsSettings.tsx`

**Requirements**
- Do not request `include_cache_size=true` on every list fetch by default.
  - Options: lazy-load cache size behind a toggle, or fetch cache size only on explicit “Refresh” action.
- Ensure the model list query does not refetch aggressively (e.g., disable refetch-on-window-focus) to avoid repeated HF scans across sessions.

**Verification**
- Manual: open Settings → Models, switch tabs, focus/unfocus window; ensure the list doesn’t spam refreshes and cache size is still accessible on demand.

---

## Phase 3: Tests + Docs

### Task 10: Backend Tests

**Files**
- Create: `tests/unit/test_model_manager_curated_registry.py`
- Create: `tests/unit/test_model_manager_task_state.py`
- Add API tests in existing `tests/api/` structure if feasible

**Minimum coverage**
- Curated registry aggregation returns expected model types
- Curated registry de-duplication: no duplicate `(id, model_type)` entries and ordering is deterministic
- De-dupe key behavior: second “start download” returns the original task_id
- Atomic de-dupe: concurrent start requests do not enqueue multiple tasks for the same `(op, model_id)`
- Cross-op exclusion: starting delete while download active (and vice versa) returns `409 Conflict`
- In-use blocking check (collections) blocks delete
- Preferences/LLM config checks return warnings (usage preflight + delete confirmation required)
- Default embedding model warning path
- VecPipe-loaded warning path (mocked/best-effort)
- Idempotency: download returns `completed` when already installed; delete returns `completed` when not installed

---

### Task 11: Frontend Tests

**Files**
- Create: `apps/webui-react/src/components/settings/model-manager/__tests__/ModelCard.test.tsx`

**Minimum coverage**
- Download button calls handler for uninstalled models
- Delete disabled for `used_by_collections.length > 0`
- Warnings from usage preflight display in confirm UI

---

### Task 12: Documentation Updates

**Files**
- Modify: `docs/plans/2026-01-19-model-manager-design.md`

**Update items**
- Explicitly mark v1 as “curated-only”
- Note custom models deferred to v2
- Document superuser-only scope (shared HF cache)
- Document delete semantics (usage preflight + confirmation, block collections, warn prefs/LLM config, ignore plugin configs)
- Document cross-op exclusion (no delete while download active, and vice versa)
- Document cache size semantics (total vs managed vs unmanaged)
- Document additional delete warnings (default embedding model, vecpipe-loaded best-effort)

---

## Final Verification

1. Backend checks:
   - `make lint`
   - `make type-check`
   - `make test` (or `make check`)
2. Frontend checks:
   - `cd apps/webui-react && npm run build`
   - `cd apps/webui-react && npm test -- --run`
3. Manual flow:
   - Settings → Models
   - Download a small embedding model; watch progress; confirm installed status updates
   - Attempt deletion of a model used by a collection → blocked
   - Delete an unused model → allowed; confirm cache size decreases
