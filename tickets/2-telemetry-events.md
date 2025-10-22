# [2] Telemetry for Visualization Interactions

## Context
- Need visibility into usage of visualization features (tab engagement, recompute frequency, selection actions).
- Current implementation lacks analytics hooks beyond toast notifications.

## Acceptance Criteria
- Track the following events with consistent payload:
  - `visualize_tab_open` (collection_id, projection_id, color_by).
  - `visualize_recompute_start` (metadata_hash, reducer, sample_size).
  - `visualize_selection_find_similar` and `visualize_selection_open`.
  - `visualize_tt_first_point` (time from tab open to first successful render).
- Events dispatched to existing telemetry pipeline (segment/logging) with error handling.
- Backend logs recompute reuse vs new run.

## Implementation Outline
1. Audit telemetry helper (`apps/webui-react/src/lib/telemetry.ts`) and add new event constants.
2. Instrument EmbeddingVisualizationTab for tab mount/unmount, recompute submit, selection actions.
3. Compute TTFP using `performance.now()` deltas stored in refs.
4. Backend optional: log idempotent reuse via structured logging.

## Affected Files / Areas
- `apps/webui-react/src/components/EmbeddingVisualizationTab.tsx`
- Telemetry utility module & TypeScript definitions.
- Backend logger (optional).

## Test Notes
- Unit tests verifying telemetry helper invoked with correct payload (use spies).
- QA checklist to confirm events appear in analytics dashboard.
