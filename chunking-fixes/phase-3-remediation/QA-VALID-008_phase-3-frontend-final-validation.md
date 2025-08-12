# QA-VALID-008: Phase 3 Frontend Final Validation

## Context
After implementing the strategies API, FE integration, cancel flows, a11y tests, and docs alignment, perform a final validation of Phase 3 success criteria.

## Requirements
- Run frontend unit tests and ensure all pass.
- Validate strategies fetch in tests and in browser (if applicable):
  - Confirm `fetchStrategies()` returns 6 non-mock strategies.
- Validate WebSocket cancel:
  - Simulate long-running preview, send cancel; UI reflects cancellation and ignores further messages for that `requestId`.
- Validate REST cancel:
  - Start REST preview, cancel mid-flight; UI stops loading and does not update further.
- Validate accessibility:
  - Run axe tests; fix any remaining violations.
- Produce a short validation report file summarizing results.

## Acceptance Criteria
- `apps/webui-react` tests pass via `npm run test:ci`.
- Strategies fetch validated; UI shows API-backed strategies.
- WS and REST cancel flows confirmed working via tests.
- Axe tests show zero violations for target components.
- A `PHASE3_VALIDATION_REPORT.md` exists at repo root or under `chunking-fixes/` summarizing outcomes.

## Suggested Files
- Report: `PHASE3_VALIDATION_REPORT.md` or `chunking-fixes/phase-3-remediation/PHASE3_VALIDATION_REPORT.md`.

## Test Commands
- `cd apps/webui-react && npm run test:ci`

## Dependencies
- Final ticket. Depends on: FE-STRAT-002, BE-WS-003, FE-WS-004, FE-REST-005, FE-A11Y-006, FE-DOCS-007.
