# Model Manager Design

**Date**: 2026-01-19
**Status**: Approved
**Author**: John + Claude

## Overview

A new "Models" tab in Settings that lets users manage all model types (Embedding, LLM, Reranker, SPLADE) in one place. Users can browse a curated registry of recommended models, download them proactively to avoid runtime delays, and delete unused models to free disk space. Power users can add custom HuggingFace models with optional configuration overrides.

## Key Principles

- **Curated + Custom**: Ship with a registry of tested models, allow adding any HuggingFace model ID
- **Non-destructive**: Block deletion of models in use by collections
- **Background operations**: Downloads happen in background, users can continue working
- **Hybrid metadata**: Auto-detect dimension and estimate memory from parameter count; use sensible defaults (symmetric mode, mean pooling); allow advanced overrides

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

1. User clicks **[Delete]** on an installed model
2. **If model is in use**: Show error toast "Cannot delete: used by Collection A, Collection B"
3. **If model is not in use**:
   - Confirmation dialog: "Delete Qwen3-Embedding-0.6B? This will free 1.2 GB of disk space."
   - On confirm: Remove from HuggingFace cache, update status to "Not Installed"
   - Toast: "Model deleted"

### In-Use Detection

A model is "in use" if any collection references it as its embedding model. The backend query joins collections table with model name to determine usage. This applies to all model types - embedding models via collection config, rerankers via search config, etc.

## Add Custom Model Flow

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

Two sources of model metadata:

1. **Curated registry** (shipped with app): `packages/shared/models/model_registry.yaml`
   - Contains tested/recommended models with full metadata
   - Updated with app releases

2. **User custom models** (database): `custom_models` table
   - Stores user-added HuggingFace model IDs and their overrides
   - Per-user scoping (multi-tenant safe)

```sql
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
| Model registry | YAML (curated) + DB (custom) | Curated ships with app, custom persists per-user |
| Quantization | Informational only | Quantization happens at runtime, not download time |
| Deletion protection | Block if in use | Prevents accidental breakage of collections |
| Custom model metadata | Auto-detect + defaults + optional overrides | Balance of UX and correctness |

## Out of Scope

The following are not included in this design but could be future enhancements:

- Batch download/delete operations
- Model version management (always use latest)
- Automatic cleanup of unused models
- Downloads panel showing all active downloads
- Model recommendations based on GPU VRAM

## Implementation Notes

### Frontend Components (apps/webui-react)

- New `ModelsTab.tsx` component in `src/components/settings/`
- Sub-components: `ModelCard.tsx`, `AddCustomModelModal.tsx`
- New hooks: `useModels.ts`, `useModelDownload.ts`
- New API service: `src/services/api/v2/models.ts`
- New types: `src/types/model.ts`

### Backend Components (packages/webui, packages/shared)

- New router: `packages/webui/api/v2/models.py`
- New Celery task: `packages/webui/tasks/model_download.py`
- New repository: `packages/shared/database/repositories/custom_model.py`
- New model: `packages/shared/database/models/custom_model.py`
- Migration for `custom_models` table
- Consolidate model registries into `packages/shared/models/model_registry.yaml`
