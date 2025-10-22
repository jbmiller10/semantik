# [2] `visualize_enabled` Feature Flag

## Context
- Visualization endpoints and UI should be gated behind a feature flag until rollout is complete.
- Currently routes/components are always available when the user has access to the collection.
- Need a backend + frontend guard that can be toggled via config or user entitlement.

## Acceptance Criteria
- Backend: `/api/v2/collections/{id}/projections/*` enforce flag (return 404 or 403 when disabled).
- Frontend: Visualization tab hidden unless flag is true (either from `/me` profile, collection metadata, or global config).
- Flag controllable via environment/config (e.g., `VISUALIZE_ENABLED=true|false`) and optionally per account.
- Documentation updated to describe flag usage.

## Implementation Outline
1. **Config plumbing**
   - Add `visualize_enabled` setting to backend settings module; optionally allow per-tenant override via DB or feature service.
2. **API guard**
   - Wrap projection routes with dependency checking flag; return consistent error (403 recommended) when disabled.
3. **Frontend gating**
   - Fetch flag via existing `/api/v2/users/me` or config endpoint; store in UI store.
   - Hide Visualization tab and disable recompute/sampling features when flag off.
4. **Tests & telemetry**
   - Unit/integration tests verifying flag disables routes & UI.
   - Log when requests hit disabled flag (for monitoring).

## Affected Files / Areas
- Backend settings (`packages/webui/settings.py`)
- Projection API routers
- Frontend config store (`apps/webui-react/src/stores/uiStore.ts` or config context)
- Navigation components (modal/tab switching)

## API & Data Contracts
- Add `visualize_enabled` boolean to user/profile response consumed by UI.
- Document error responses when disabled.

## Test Notes
- Backend: unit test flagged-off route returns 403; flagged-on returns 202/200.
- Frontend: render tests verifying tab hidden; Cypress toggling flag via fixture.
