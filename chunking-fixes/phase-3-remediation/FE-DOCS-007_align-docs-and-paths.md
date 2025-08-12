# FE-DOCS-007: Align Docs and Paths with Current Implementation

## Context
Phase 3 FE tickets refer to `apps/webui-react/src/api/chunking.ts` while the implemented client lives under `src/services/api/v2/chunking.ts`. Align documentation and optionally add a re-export to match expectations.

## Requirements
- Update Phase 3 documentation under `chunking-fix-tickets/phase-3-frontend` to reference the correct FE client path and store usage.
- Optionally add a re-export file at `apps/webui-react/src/api/chunking.ts` that exports from `../services/api/v2/chunking` to satisfy path references without moving files.
- Ensure no broken imports; run build to verify.

## Acceptance Criteria
- Docs reflect accurate file paths, modules, and functions.
- Optional re-export file exists and builds successfully (if used).
- No broken imports are introduced.

## Suggested Files
- Docs: `chunking-fix-tickets/phase-3-frontend/*.md`.
- Re-export (optional): `apps/webui-react/src/api/chunking.ts`.

## Test Commands
- `npm run build:frontend` (or `cd apps/webui-react && npm run build`)

## Dependencies
- After FE-STRAT-002 so the docs can point to the final method names.
