# [2] Invalidation UX Consistency After Collection Updates

## Context
- Backend marks projections as `degraded` after ingestion operations, but UI messaging is inconsistent (badge vs banner vs recompute prompt).
- Need cohesive UX so users understand when to recompute and what changes triggered invalidation.

## Acceptance Criteria
- Single visual indicator (badge/breadcrumb/banner) appears across list view, drawer, preview when `meta.degraded=true`.
- Copy explains why recompute needed and links to dialog.
- Telemetry logs when degraded projection viewed without recompute.
- Optional: display timestamp of invalidation (if available).

## Implementation Outline
1. Audit existing placements (list badges, recompute button text, selection drawer warning).
2. Design consistent badge styling + tooltip.
3. Update components to share helper (e.g., `renderDegradedBadge(meta)`).
4. Ensure recompute dialog auto-selects recommended settings when degraded.

## Affected Files / Areas
- `apps/webui-react/src/components/EmbeddingVisualizationTab.tsx`
- Shared badge components.
- Telemetry hooks.

## Test Notes
- Visual regression to confirm badge appears in all relevant locations.
