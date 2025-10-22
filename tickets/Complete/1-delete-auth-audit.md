# [1] DELETE Projection Auth Guard Audit

## Context
- Recent work added ownership checks for `DELETE /projections/{id}` but we need to audit imports/guards to ensure consistent behavior.
- Goal: confirm `get_collection_ownership_guard` (or service-level owner check) is enforced across all entry points.

## Acceptance Criteria
- Single source of truth for ownership enforcement (router dependency or service guard) with tests.
- Remove unused imports/dependencies if redundant.
- Document expected error codes (403 vs 404) for unauthorized deletes.

## Implementation Outline
1. Inspect router to ensure guard dependency resolves correctly in all environments (no circular imports).
2. Verify service-level check uses persisted `owner_id` and handles public collections.
3. Add tests (unit or integration) covering owner vs non-owner vs anonymous deletion.
4. Update documentation/README with auth expectations.

## Affected Files / Areas
- `packages/webui/api/v2/projections.py`
- `packages/webui/services/projection_service.py`
- Auth dependency modules (`packages/webui/api/dependencies/collections.py`)

## Test Notes
- Backend test ensuring non-owner receives 403; owner succeeds; missing run returns 404.
