# [2] Documentation – Visualization Endpoints & UX

## Context
- Feature nearing GA; need docs for backend endpoints, configuration, and failure modes.
- Current README/API docs lack projection-specific sections.

## Acceptance Criteria
- API reference documents:
  - `/api/v2/collections/{id}/projections` (POST/GET/DELETE/arrays/select/legend if added).
  - Request/response samples including `operation_id`, sampling metadata, degraded flag.
- Admin guide covers feature flag (`visualize_enabled`), idempotency hash, cleanup semantics.
- UI docs or runbook for retrying recompute, handling degraded state, selection drawer.
- Update CHANGELOG.

## Implementation Outline
1. Update API Markdown in `docs/api/projections.md` (create if missing).
2. Add config section to `docs/ops/configuration.md` covering new env vars & defaults.
3. Create user-facing walkthrough in `docs/user-guide/visualization.md` with screenshots.
4. Update README/CHANGELOG with feature summary and migration steps.

## Affected Files / Areas
- `docs/api` directory
- `docs/user-guide`
- `CHANGELOG.md`

## Test Notes
- Peer doc review; ensure curl samples validated via Postman.
