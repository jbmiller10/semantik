# [5] Frontend Test Coverage for Visualization UX

## Context
- New UI features (sampling badge, recompute dialog, selection drawer) need automated coverage.
- Ensure React Testing Library & Cypress cover regressions.

## Acceptance Criteria
- Unit/integration tests cover:
  1. Arrays fetch states (loading/error/success) with legend rendering.
  2. Sampling badge displays N of M and hides when not sampled.
  3. Recompute dialog validation and success path (idempotent reuse, error display, progress banner).
  4. Selection drawer handles multi-selection, stale response guard, action buttons disabled/enabled states.
- Cypress e2e scenario: load projection, switch color, open selection, start recompute.
- Snapshot/visual regression optional but nice-to-have.

## Implementation Outline
1. Add React Testing Library cases in `apps/webui-react/src/components/__tests__/EmbeddingVisualizationTab.test.tsx` with mocked API responses.
2. Mock websocket hook to simulate progress updates.
3. Cypress test path `cypress/e2e/projection_visualize.cy.ts` using stubbed APIs.
4. Ensure test suite mocks new API shape (operation_id/status).

## Affected Files / Areas
- `apps/webui-react/src/components/__tests__/EmbeddingVisualizationTab.test.tsx`
- `cypress/e2e` directory & fixtures.
- Test utilities (mock service workers or intercepts).

## Test Notes
- Use `msw` or custom fetch mocks to simulate selection + arrays endpoints.
- For Cypress, intercept fetches to return deterministic buffers (use fixture). Validate drawer contents.
