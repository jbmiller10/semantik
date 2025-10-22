# [3] Optional Fast Path for Color-By Legend Refresh

## Context
- Switching color-by currently triggers a full projection recompute even when only the category mapping changes.
- Backend stores category arrays (`cat.u8.bin`) and legend metadata; we could recompute legend client-side when switching between compatible schemes (document_id ⇄ source_dir, etc.) without rerunning UMAP.
- PRD lists this as optional; we should evaluate feasibility and provide a plan.

## Acceptance Criteria
- Document feasibility: identify which combinations can be satisfied via existing payload (`original_ids`, doc metadata).
- Implement a fast-path endpoint or client routine that regenerates legend + cat array without full run when possible.
- UI switches legends instantly for supported color-by choices; falls back to recompute dialog otherwise.
- Fast-path clearly surfaced (log/telemetry) and respects degraded/invalidation rules.

## Implementation Outline
1. **Analysis phase**
   - Audit selection payloads & `original_ids` mapping for required metadata.
   - Determine which color_by values can be derived locally (document_id, source_dir, filetype, age_bucket).
2. **Server option** (preferred)
   - Add lightweight endpoint `/projections/{id}/legend?color_by=…` that streams updated `cat` + legend using existing vectors from Qdrant.
   - Reuse existing sampling logic; mark run meta updated.
3. **Client option** (fallback)
   - For metadata available in meta payload (`category_map`, `original_ids`), compute legend locally and update `selectionState`.
4. **UI wiring**
   - When user changes color_by, attempt fast path; show inline spinner and revert if unsupported.
5. **Telemetry**
   - Track fast-path success/failure to inform future optimizations.

## Affected Files / Areas
- `packages/webui/tasks/projection.py` (optional helper to regenerate cat array)
- `packages/webui/api/v2/projections.py` (new route)
- `apps/webui-react/src/components/EmbeddingVisualizationTab.tsx`
- Qdrant client utility if server-side.

## API & Data Contracts
- Proposed GET/POST should accept `color_by`, `projection_id`, maybe `sample_limit`; returns `cat` buffer + legend JSON.
- Document in API docs once finalized.

## Test Notes
- Unit tests for legend recalculation (one per color scheme).
- Integration: switch color_by in UI and assert no recompute operation triggered.
- Performance benchmark to ensure endpoint responds within target latency.
