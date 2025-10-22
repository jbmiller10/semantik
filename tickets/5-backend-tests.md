# [5] Backend Test Coverage for Projection Stack

## Context
- Several new code paths (selection API, array streaming, sampling, recompute deferral, DELETE guard) lack automated tests.
- Need comprehensive suite to protect regressions before GA.

## Acceptance Criteria
- Tests cover the following scenarios:
  1. Selection endpoint returns chunk metadata and handles missing IDs (404, 409).
  2. `/arrays/{artifact}` responds with 200, 404, 403 (ownership check) and guards path traversal.
  3. PCA vs UMAP reducer flow (fallback, error handling, reducer params stored).
  4. Color-by overflow bucket + legend counts.
  5. Recompute defer-completion (`defer_completion` flag ensuring collection state not degraded into READY).
  6. DELETE auth pathways (owner success, viewer denied, artifact cleanup).
- Tests runnable via `make test` and documented.

## Implementation Outline
1. Add pytest module `tests/webui/projections/test_api.py` for route-level tests using async client.
2. Mock Qdrant client to simulate vectors/payloads for selection & array streaming.
3. Add unit tests in `tests/webui/tasks/test_projection.py` for legend overflow & reducer fallback.
4. Write deletion tests verifying storage path removal (use temporary directories).
5. Ensure fixtures cover degraded flag toggling after ingestion operations.

## Affected Files / Areas
- `tests/webui/projections/…`
- Fixtures for Qdrant manager stubs.
- Possibly update Makefile test target if new markers added.

## Test Notes
- Use golden sample data for vectors to keep deterministic.
- Validate binary outputs (cat/bin) lengths.
