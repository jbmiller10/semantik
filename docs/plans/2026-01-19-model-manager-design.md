# Model Manager Design

**Date**: 2026-01-19
**Status**: Approved (v1 Implemented)
**Author**: John + Claude

## v1 Scope

This document describes the v1 implementation of the Model Manager. Custom model support is deferred to v2.

### v1 Features (Implemented)
- Browse curated registry of recommended models
- Download models proactively to avoid runtime delays
- Delete unused models to free disk space
- Block deletion of models in use by collections
- Warn (don't block) for user preferences and LLM configs
- Superuser-only access (shared HF cache is global to all users)

### Deferred to v2
- Custom HuggingFace model support
- Per-user custom model registry
- Advanced configuration overrides

## Overview

A new "Models" tab in Settings that lets users manage all model types (Embedding, LLM, Reranker, SPLADE) in one place. Users can browse a curated registry of recommended models, download them proactively to avoid runtime delays, and delete unused models to free disk space.

## Key Principles

- **Curated Registry**: Ship with a registry of tested/recommended models
- **Non-destructive**: Block deletion of models in use by collections
- **Background operations**: Downloads happen in background, users can continue working
- **Superuser-only**: HuggingFace cache is shared globally, so model management is restricted to superusers

## Model Types Supported

| Type | Purpose | Examples |
|------|---------|----------|
| Embedding | Document/query encoding for search | Qwen3-Embedding, BGE, all-MiniLM |
| LLM | HyDE, summarization, entity extraction | Qwen2.5-Instruct, Llama |
| Reranker | Search result reranking | Qwen3-Reranker, cross-encoders |
| SPLADE | Sparse vectors for hybrid search | splade-cocondenser |

## UI Structure

The Models tab uses horizontal sub-tabs to separate model types, with search and filtering within each.

```
Settings > Models
┌─────────────────────────────────────────────────────────────┐
│ [Embedding] [LLMs] [Rerankers] [SPLADE]                     │
├─────────────────────────────────────────────────────────────┤
│ [🔍 Search models...]    [Status: All ▾]    [+ Add Custom]  │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Qwen/Qwen3-Embedding-0.6B                    [Installed]│ │
│ │ Small model, instruction-aware (1024d)                  │ │
│ │ Size: 1.2 GB  •  Used by: 2 collections                 │ │
│ │ ┌───────────────────────────────────────────────────────┐ │
│ │ │ Quantization │ VRAM Estimate              │           │ │
│ │ │ float16      │ ~1.2 GB                    │           │ │
│ │ │ int8         │ ~600 MB                    │           │ │
│ │ └───────────────────────────────────────────────────────┘ │
│ │                                              [Delete]   │ │
│ └─────────────────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Qwen/Qwen3-Embedding-4B                  [Not Installed]│ │
│ │ Medium model, MTEB top performer (2560d)                │ │
│ │ Size: 8.0 GB                                            │ │
│ │ ...                                        [Download]   │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Status Filter Options

- **All**: Show all models (installed and available)
- **Installed**: Only show downloaded models
- **Available**: Only show models not yet installed

### Model Card Information

- Model name and description
- Size on disk (or download size if not installed)
- Usage indicator (which collections use it) - only for installed models
- Quantization table with VRAM estimates (informational only - quantization happens at runtime)
- Action button: Download / Delete (or "In Use" disabled state)

## Download Flow

1. User clicks **[Download]** on a model card
2. Button changes to a progress bar showing percentage and bytes (e.g., "1.2 GB / 8.0 GB")
3. A toast appears: "Downloading Qwen3-Embedding-4B..."
4. User can navigate away - download continues in background
5. On completion:
   - Progress bar becomes **[Installed ✓]** then transitions to **[Delete]**
   - Toast: "Qwen3-Embedding-4B downloaded successfully"
6. On failure:
   - Progress bar becomes **[Retry]** button
   - Toast with error: "Download failed: Network error. Click retry to try again."

## Delete Flow

### Overview

1. User clicks **[Delete]** on an installed model
2. Backend performs preflight check (`GET /models/{model_id}/usage`)
3. Deletion is **blocked** (409) if collections use the model
4. Deletion **requires confirmation** if warnings exist (user preferences, LLM configs, etc.)
5. On confirm: Remove from HuggingFace cache, update status to "Not Installed"

### Blocking vs Warning

| Condition | Behavior |
|-----------|----------|
| Collections use model | **BLOCK** - Cannot delete until collections are updated |
| User preferences reference model | **WARN** - Requires confirmation, preferences reset to defaults |
| LLM tier configs reference model | **WARN** - Requires confirmation, configs updated |
| Model is default embedding model | **WARN** - Requires confirmation (env var must be updated) |
| Model is loaded in VecPipe | **WARN** - Best-effort check, may require service restart |
| Plugin configs reference model | **IGNORE** - Plugins can handle missing models gracefully |

### Cross-Operation Exclusion

Only one operation (download or delete) can be active per model at a time:
- Download while delete active → 409 with existing delete task_id
- Delete while download active → 409 with existing download task_id
- Second download request → Returns existing download task_id (idempotent)
- Second delete request → Returns existing delete task_id (idempotent)

### Delete Confirmation Dialog

When warnings exist but deletion is allowed, the confirmation dialog shows:
- Estimated disk space to be freed
- List of warnings (preferences, configs, etc.)
- "This action cannot be undone"

### Usage Preflight Response

```typescript
interface ModelUsageResponse {
  model_id: string;
  is_installed: boolean;
  size_on_disk_mb: number | null;
  estimated_freed_size_mb: number | null;
  blocked_by_collections: string[];    // If non-empty, deletion is blocked
  user_preferences_count: number;      // Users referencing this model
  llm_config_count: number;            // LLM tier configs using this model
  is_default_embedding_model: boolean; // True if matches DEFAULT_EMBEDDING_MODEL env
  loaded_in_vecpipe: boolean;          // Best-effort check if loaded in GPU memory
  loaded_vecpipe_model_types: string[]; // Which model types are loaded
  warnings: string[];                  // Human-readable warning messages
  can_delete: boolean;                 // True if not blocked by collections
  requires_confirmation: boolean;      // True if warnings exist
}
```

### In-Use Detection

A model is "in use" and **blocks deletion** if any collection references it as its embedding model. The backend query joins collections table with model name to determine usage. This hard block applies only to collections - other references (preferences, LLM configs) generate warnings instead.

## Add Custom Model Flow (v2 - Deferred)

> **Note**: Custom model support is deferred to v2. The following describes the planned implementation.

User clicks **[+ Add Custom]** button, which opens a modal:

```
┌─────────────────────────────────────────────────────────────┐
│ Add Custom Model                                        [×] │
├─────────────────────────────────────────────────────────────┤
│ Model ID *                                                  │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ BAAI/bge-small-en-v1.5                                  │ │
│ └─────────────────────────────────────────────────────────┘ │
│ Enter a HuggingFace model ID                                │
│                                                             │
│ Model Type *                                                │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Embedding                                             ▾ │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ ▶ Advanced Options                                          │
├─────────────────────────────────────────────────────────────┤
│                                    [Cancel]  [Add Model]    │
└─────────────────────────────────────────────────────────────┘
```

### Advanced Options (collapsed by default)

| Field | Default | Notes |
|-------|---------|-------|
| Dimension | Auto-detect | Detected on first load |
| Pooling Method | mean | Options: mean, cls, last_token |
| Asymmetric Mode | Off | Toggle on to show prefix fields |
| Query Prefix | (empty) | For BGE/E5 style models |
| Document Prefix | (empty) | Usually empty |
| Default Instruction | (empty) | For Qwen-style models |

### On Submit

1. Validate model ID exists on HuggingFace (quick API check)
2. Add to user's custom model registry
3. Model appears in list with "Not Installed" status
4. User can then download it

## Backend Architecture

### Model Registry Storage

**v1 Implementation**: Curated registry only (shipped with app)

1. **Curated registry** (shipped with app): `packages/shared/models/model_registry.yaml`
   - Contains tested/recommended models with full metadata
   - Updated with app releases

2. **User custom models** (v2 - deferred): `custom_models` table
   - Stores user-added HuggingFace model IDs and their overrides
   - Per-user scoping (multi-tenant safe)

```sql
-- v2: Custom model support
CREATE TABLE custom_models (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    model_id VARCHAR(255) NOT NULL,      -- e.g., "BAAI/bge-small-en-v1.5"
    model_type VARCHAR(50) NOT NULL,      -- embedding, llm, reranker, splade
    config JSONB DEFAULT '{}',            -- overrides (dimension, pooling, prefixes)
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(user_id, model_id)
);
```

### Cache Size Semantics

The model list endpoint returns cache size information with three components:

```typescript
interface CacheSizeInfo {
  total_cache_size_mb: number;       // Entire HuggingFace hub cache
  managed_cache_size_mb: number;     // Only curated models (known to registry)
  unmanaged_cache_size_mb: number;   // total - managed (unknown repos)
  unmanaged_repo_count: number;      // Count of repos not in curated registry
}
```

- **Total**: Everything in `~/.cache/huggingface/hub/`
- **Managed**: Models matching entries in the curated registry
- **Unmanaged**: Models downloaded outside the Model Manager (e.g., by other apps)

The UI displays "X GB used by curated models, Y GB by other downloads" to help users understand disk usage.

### Download/Cache Management

Models are stored in HuggingFace's cache (`~/.cache/huggingface/hub/`). The backend:

- **Lists installed models**: Scans cache directory, matches against registry
- **Downloads models**: Uses `huggingface_hub.snapshot_download()` with progress callbacks
- **Deletes models**: Uses `huggingface_hub.scan_cache_dir()` to find and remove specific model revisions
- **Reports size**: Calculates disk usage from cached files

### API Endpoints

New endpoints under `/api/v2/models/`:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/models` | GET | List all models (curated + custom) with install status |
| `/models/download` | POST | Start background download |
| `/models/download/{task_id}` | GET | Check download progress |
| `/models/{model_id}` | DELETE | Delete model from cache |
| `/models/custom` | POST | Add custom model to registry |
| `/models/custom/{id}` | DELETE | Remove custom model from registry |

## Download Progress & Background Tasks

### Background Download Mechanism

Downloads run as Celery tasks (consistent with existing operation pattern):

1. Frontend calls `POST /models/download` with `model_id`
2. Backend creates a Celery task, returns `task_id`
3. Frontend polls `GET /models/download/{task_id}` for progress
4. Alternatively: WebSocket updates via existing `/ws/operations` channel

### Progress Reporting

HuggingFace's `snapshot_download` supports progress callbacks. The task:
- Tracks total bytes and downloaded bytes
- Updates Redis with progress every ~1 second
- Frontend polls or receives WebSocket updates

```typescript
// Progress response shape
interface DownloadProgress {
  task_id: string;
  model_id: string;
  status: 'pending' | 'downloading' | 'completed' | 'failed';
  bytes_downloaded: number;
  bytes_total: number;
  error?: string;
}
```

### Concurrent Downloads

- Allow multiple simultaneous downloads (different models)
- Single download per model (if already downloading, return existing task_id)
- Downloads panel in UI could show all active downloads (future enhancement)

## Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Navigation pattern | Horizontal sub-tabs + search/filter | Scales to many models without overwhelming UI |
| Download mechanism | Celery background tasks | Consistent with existing operation pattern |
| Cache management | HuggingFace hub library | Standard tooling, handles revisions properly |
| Model registry | YAML (curated only in v1) | Curated ships with app, custom deferred to v2 |
| Quantization | Informational only | Quantization happens at runtime, not download time |
| Deletion protection | Block collections, warn for preferences | Prevents breakage while allowing cleanup |
| Access control | Superuser-only | HF cache is global, multi-tenant safety |
| Cross-op exclusion | One operation per model | Prevents race conditions in cache operations |

## Out of Scope (v1)

The following are not included in v1 but could be future enhancements:

- Custom HuggingFace model support (v2)
- Batch download/delete operations
- Model version management (always use latest)
- Automatic cleanup of unused models
- Downloads panel showing all active downloads
- Model recommendations based on GPU VRAM

## Implementation Notes

### Frontend Components (apps/webui-react)

v1 implementation:
- `ModelsSettings.tsx` component in `src/components/settings/model-manager/`
- `ModelCard.tsx` sub-component with download/delete progress
- Hooks: `useModelManager.ts` (queries + mutations + progress tracking)
- API service: `src/services/api/v2/model-manager.ts`
- Types: `src/types/model-manager.ts`

v2 additions (deferred):
- `AddCustomModelModal.tsx` for custom model registration

### Backend Components (packages/webui, packages/shared)

v1 implementation:
- Router: `packages/webui/api/v2/model_manager.py`
- Celery tasks: Download and delete in `packages/webui/tasks/model_manager_tasks.py`
- Redis state: `packages/webui/api/v2/model_manager_task_state.py`
- Curated registry: `packages/shared/models/model_registry.py`
- Schemas: `packages/webui/api/v2/model_manager_schemas.py`

v2 additions (deferred):
- Repository: `packages/shared/database/repositories/custom_model.py`
- Model: `packages/shared/database/models/custom_model.py`
- Migration for `custom_models` table
