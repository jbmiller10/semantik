# [2] Optional `arrays` ETag & Client Caching

## Context
- `/arrays/{artifact}` downloads large binary payloads. Users often re-open the same projection, causing repeated downloads.
- Introduce HTTP caching (ETag/Last-Modified) to reduce bandwidth.

## Acceptance Criteria
- Server returns `ETag` (hash of file) and `Last-Modified` headers.
- Handles `If-None-Match` / `If-Modified-Since` to respond 304 when unchanged.
- Frontend uses `fetch` cache (`If-None-Match`) or stores binary in IndexedDB for reuse.
- Optional feature (can be disabled via config).

## Implementation Outline
1. Compute file hash once per run (store in `ProjectionRun.meta`).
2. Modify streaming route to check conditional headers and short-circuit with 304.
3. Update front-end fetch helper to respect 304 (skip parsing, reuse cached arrays).
4. Document behavior and fallback for proxies/CDNs.

## Affected Files / Areas
- `packages/webui/services/projection_service.py` (metadata update)
- `packages/webui/api/v2/projections.py` (streaming route)
- `apps/webui-react/src/services/api/v2/projections.ts` (fetch logic)

## Test Notes
- Backend tests for 200 vs 304 path.
- Frontend test mocking 304 response ensures arrays reused.
