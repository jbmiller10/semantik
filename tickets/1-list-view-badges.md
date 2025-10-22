# [1] Projection List Badges – Sampled / Degraded Indicators

## Context
- Projection list currently shows reducer and status but not sampling/degraded flags.
- Need quick glance indicators to prompt recompute.

## Acceptance Criteria
- List rows display badges when `meta.sampled` or `meta.degraded` true.
- Tooltips explain badge meaning.
- Badges update in real time when meta changes (query invalidated).

## Implementation Outline
1. Extend `ProjectionMetadata` TypeScript type to include `meta.sampled` & `meta.degraded`.
2. Update list component to render badges next to reducer/status.
3. Ensure backend list response includes meta fields (confirm already true).

## Affected Files / Areas
- `apps/webui-react/src/components/EmbeddingVisualizationTab.tsx` (projection table)
- CSS/utility classes for badges.

## Test Notes
- Unit test verifying badges render when meta flags present.
