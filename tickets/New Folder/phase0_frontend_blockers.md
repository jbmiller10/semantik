# Phase 0 Ticket: Restore Critical Frontend UX Blocks (Target: October 24, 2025)

## Background
During the October 18, 2025 review (`CODEBASE_REVIEW_REPORT.md`), the React app (`apps/webui-react`) was flagged with two critical issues that break core navigation paths a hiring manager would exercise when demoing Semantik:
- The "Active Operations" tab logs navigation intent to the console instead of routing to the collection detail view.
- The document viewer falls back to an `<object>` tag, so PDFs render inconsistently and lack page controls.

Additionally, the audit highlighted global-window escape hatches and noisy console logging that hurt polish. These problems are visible the moment someone opens the UI, so we must address them before showcasing the product.

## Objectives
1. Reinstate intuitive navigation from the operations dashboard into collection detail pages.
2. Deliver a reliable PDF viewing experience using pdf.js (or, if time-boxed, degrade gracefully with documentation).
3. Remove global window state dependencies and stray console logging that break SSR/tests.

## Requirements & Implementation Details
### 1. Active Operations Navigation
- Update `apps/webui-react/src/components/ActiveOperationsTab.tsx` so clicking an operation navigates via `useNavigate` to `/collections/:id`.
- Ensure any relevant query params (e.g., preselecting operation state) are preserved.
- Back-fill a React Testing Library test asserting that clicking fires the navigation (mock router history).

### 2. PDF Viewer Upgrade
- Replace the `<object>` fallback in `DocumentViewer.tsx` with a pdf.js-driven canvas renderer.
  - Initialize pdf.js worker (respecting Vite bundling) and render the first page with sensible scaling.
  - Provide loading & error states (spinners/toasts) aligned with the design system.
  - Support multi-page navigation or at least document how to extend it.
- Add unit/integration tests for the viewer component (mock pdf.js) validating loading/error flows.
- Document any new npm dependencies or build steps in `apps/webui-react/README.md`.

### 3. Eliminate `window.*` Escape Hatches
- Identify current usages (`SearchInterface.tsx`, `services/api/v2/client.ts`) and migrate to Zustand store events or React context.
- Ensure the new approach is SSR-safe and test-friendly.
- Update associated tests to rely on the new store/context rather than patching `window`.

### 4. Console Noise Cleanup
- Configure `vite.config.ts` to strip `console` and `debugger` in production builds via the `esbuild.drop` option.
- Remove any lingering `console.log` statements in code paths covered by tests and provide structured logging alternatives when necessary.

## Acceptance Criteria
- From the operations tab, choosing an item routes to the corresponding collection detail page without page reload.
- PDF documents render using pdf.js with a visible canvas-based viewer, and the component shows a toast or inline message on error.
- No direct references to `window.__gpuMemoryError` or similar remain; state is managed via the app store/context pattern.
- `npm run build --prefix apps/webui-react` succeeds and production bundles no longer contain `console.log` statements.
- New/updated tests cover navigation, PDF viewer loading/error handling, and state store updates (`npm test --prefix apps/webui-react`).
- Documentation for the frontend includes any new configuration steps.

## Validation Steps
1. `npm test --prefix apps/webui-react -- ActiveOperationsTab` (or component test suite) passes with the new navigation test.
2. `npm test --prefix apps/webui-react -- DocumentViewer` passes covering pdf.js behavior.
3. Manual smoke test in the browser: start `make frontend-dev`, run an indexing operation, confirm navigation and PDF render.
4. Inspect the production bundle (e.g., `grep -R "console.log" apps/webui-react/dist`) to confirm logs are stripped.

## Notes
- Coordinate with the backend Phase 0 ticket so navigation hits fully secured endpoints.
- If pdf.js integration cannot be stabilized within the sprint, propose and document an interim fallback plus QA checklist, but production must no longer silently show a blank frame.

## Out of Scope
- Redesigning the overall operations dashboard or collection detail layout.
- Building advanced PDF features like annotations, text extraction, or offline caching beyond baseline viewing controls.
- Implementing new navigation routes unrelated to operations → collection flows.
- Addressing unrelated medium/low severity frontend findings unless they block the acceptance criteria above.
