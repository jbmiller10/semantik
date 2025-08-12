# FE-A11Y-006: Add Accessibility Tests and Fix Basic Issues

## Context
ORCHESTRATOR Phase 3 suggests axe-based accessibility validation. Add tests and fix basic issues for key chunking UI components.

## Requirements
- Add `jest-axe` (compatible with vitest) and write a11y tests for:
  - `ChunkingPreviewPanel`, `ChunkingStrategySelector`, `ChunkingParameterTuner`, and any chunking-related modals.
- Fix issues surfaced by tests:
  - Ensure interactive elements have labels/roles.
  - Keyboard navigation: focus states and arrow key behavior where applicable (e.g., sliders).
  - Announce async progress via ARIA live regions.
- Integrate tests into CI and ensure headless run compatibility.

## Acceptance Criteria
- Axe tests report zero violations for tested components.
- Keyboard-only navigation is validated in tests where appropriate.
- Add an “Accessibility” section to `apps/webui-react/ARCHITECTURE.md` explaining approach and standards.

## Suggested Files
- New tests under `apps/webui-react/src/components/chunking/__tests__/*a11y*.test.tsx`.
- `apps/webui-react/package.json` devDependencies for `jest-axe` and any needed types.
- Update `apps/webui-react/ARCHITECTURE.md`.

## Test Commands
- `cd apps/webui-react && npm run test:ci`

## Dependencies
- Prefer after FE-STRAT-002 so tests target the real strategies path.
