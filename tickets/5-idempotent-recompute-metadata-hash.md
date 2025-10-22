# [5] Idempotent Projection Recompute via Metadata Hash

## Context
- Recompute dialog allows users to launch multiple runs with identical parameters, which can flood the queue and overwrite artifacts without need.
- Backend currently tracks `ProjectionRun` config, but lacks a deterministic dedupe key.
- We need an idempotency mechanism keyed by reducer, config, color_by, sample_limit, and collection state so identical requests reuse existing completed runs when safe.

## Acceptance Criteria
- Each recompute request carries a deterministic `metadata_hash` (hash of sorted reducer/config/color_by/sample inputs plus collection vector version).
- Backend stores `metadata_hash` on `ProjectionRun` and rejects or short-circuits creation if a **completed** run with same hash exists and collection has not changed since (no degraded flag).
- API returns `status=completed` immediately with existing run metadata in the idempotent case.
- Telemetry logs when idempotent shortcut is used.

## Implementation Outline
1. **Hash generation utility** (frontend + backend)
   - Canonicalize reducer config (sorted JSON) + color_by + sample size + embedding model version + collection `vector_count` timestamp.
   - Use SHA256 or Murmur for stability.
2. **Schema updates**
   - Add `metadata_hash` column + index to `projection_runs` via Alembic.
3. **Service logic**
   - In `start_projection_build`, check for existing run with same hash and `status=COMPLETED` & `degraded!=True`.
   - Return existing metadata if valid; otherwise proceed with new run and store hash.
   - Mark runs degraded when underlying collection mutates (already partially implemented).
4. **Frontend updates**
   - Recompute dialog sends `metadata_hash` in payload.
   - Handle 202 vs 200 responses gracefully in UI (show toast “Reused latest projection”).
5. **Telemetry & logging**
   - Emit event on reuse vs new run; add warning when hash collisions occur.

## Affected Files / Areas
- Alembic migration files
- `packages/webui/services/projection_service.py`
- `packages/webui/api/v2/projections.py`
- `apps/webui-react/src/components/EmbeddingVisualizationTab.tsx`
- Shared hashing utility (new helper module)

## API & Data Contracts
- POST `/projections` payload gains optional `metadata_hash`.
- Response includes `operation_id` and `status`; consider returning `idempotent_reuse=true` flag.

## Test Notes
- Backend unit tests for idempotent path (new run vs reuse, degraded invalidation).
- Frontend integration test: recompute same params twice -> second call short-circuits.
- Load test to ensure hash lookup scales with indexed column.
