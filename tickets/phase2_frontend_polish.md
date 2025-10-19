# Phase 2 Ticket: Frontend UX Polish & Error Handling (Target Window: November 10 – November 21, 2025)

## Background
After backend gates are restored, Phase 2 requires the React frontend (`apps/webui-react`) to reflect production-ready quality. The audit noted lingering `window` globals, console noise, pdf.js ergonomics, and `alert` usage. This ticket isolates UI polish tasks while CI hardening occurs in a sibling ticket.

## Objectives
1. Provide polished PDF viewing with accessible controls and consistent error handling.
2. Remove remaining debug patterns (`window.*`, `alert`, console logging) from production bundles.
3. Ensure key flows (operations → collection → document view) meet accessibility basics.

## Requirements
### 1. PDF Viewer Enhancements
- Extend the Phase 0 pdf.js integration to include:
  - Zoom controls, page navigation, or clearly documented rationale if limited.
  - Loading spinners and error toasts consistent with the design system.
  - Keyboard navigation for core actions; add `aria` labels.
- Update or add tests (`DocumentViewer.test.tsx`) covering new controls and error states.

### 2. Debug Pattern Cleanup
- Confirm the build config strips `console`/`debugger` but also remove leftover statements in source (`rg "console.log" apps/webui-react/src`).
- Replace any `window.*` communication added earlier (double-check search results) with store/context patterns.
- Replace `window.alert` with toast notifications (`useToast` or equivalent component).
- Verify production bundles (`npm run build --prefix apps/webui-react`) contain no unwanted globals/logs.

### 3. Accessibility & Error UX
- Perform an accessibility pass using tools such as `axe-core` or `eslint-plugin-jsx-a11y` to ensure operations and document flows meet standards (focus states, aria labeling, color contrast where applicable).
- Update tests to cover new accessibility hooks if feasible; otherwise, document manual checklist results.
- Ensure error toasts include actionable copy and do not rely solely on color cues.

### 4. Documentation
- Update frontend README or `docs/frontend.md` with instructions for pdf.js dependencies, recommended accessibility checks, and the absence of `window` globals.
- Note the existence of the CI hardening ticket to clarify responsibilities.

## Acceptance Criteria
- Document viewer exposes basic controls, handles errors gracefully, and is keyboard accessible.
- No `console.log`, `window.alert`, or custom `window.__*` dependencies remain in production code.
- Accessibility scan of key flows shows no critical issues; findings (if any) are documented with remediation plan.
- `npm run build --prefix apps/webui-react` succeeds; `npm test --prefix apps/webui-react -- DocumentViewer` covers new logic.

## Validation Steps
1. `npm test --prefix apps/webui-react -- DocumentViewer`
2. `npm test --prefix apps/webui-react -- SearchInterface` (ensures no regressions from earlier updates)
3. `npm run build --prefix apps/webui-react`
4. Run accessibility tool (e.g., `npx axe http://localhost:5173/collections/<id>`) and document results.

## Coordination Notes
- Coordinate with CI hardening ticket owner to align documentation updates.
- Engage design/UX for feedback on pdf.js controls and toast copy.
- Communicate any accessibility remediation requiring backend changes back to the relevant teams.

## Out of Scope
- Major frontend redesigns, theming overhauls, or component library swaps.
- Implementing new features beyond the specified polish tasks.
- Backend changes outside of minor adjustments needed for accessibility (e.g., error message copy).
- Performance profiling or bundle size optimization beyond removing debug statements.
