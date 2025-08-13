# BE-STRAT-001: Implement Strategies API (GET /api/v2/chunking/strategies)

## Context
The frontend currently uses static `CHUNKING_STRATEGIES` for strategy metadata. Phase 3 validation and prior tickets reference a dynamic `getStrategies()` call that should return 6 strategies. A real backend endpoint is now in place in Phase 2 (`GET /api/v2/chunking/strategies`), returning 6 strategies and default configs. This ticket ensures contract alignment with FE and documents any server-side response shaping needed.

## Requirements (updated)
- Confirm the existing endpoint `GET /api/v2/chunking/strategies` returns a stable list of 6 strategies: `character`, `recursive`, `markdown`, `semantic`, `hierarchical`, `hybrid` (public-facing IDs may alias, see below).
- Ensure the response contract aligns with FE expectations in `apps/webui-react/src/types/chunking.ts`.
  - The FE expects fields: `type`, `name`, `description`, `icon`, `performance`, `supportedFileTypes`, `parameters`, `isRecommended?`, `recommendedFor?`.
  - Backend currently exposes `StrategyInfo` with: `id`, `name`, `description`, `best_for`, `pros`, `cons`, `default_config`, `performance_characteristics`.
  - Either:
    1) Extend backend response to include FE-native fields (recommended), or
    2) Document a thin FE adapter mapping server fields to FE types (see Mapping below).
- Alias handling: map internal IDs to public FE types where applicable:
  - `character` → public `fixed_size` (if FE uses that label); otherwise keep `character` if FE expects it.
  - `markdown` → `markdown`, `hierarchical` → `hierarchical`, `recursive` → `recursive`, `semantic` → `semantic`, `hybrid` → `hybrid`.
- Add unit/contract tests verifying the endpoint shape (either BE-native or FE-adapted) matches the FE type expectations.

### Mapping (if keeping current backend fields)
- `type` (FE) ← `id` (BE) with aliasing for `character` → `fixed_size` if FE requires.
- `performance` (FE) ← summarize from `performance_characteristics` (BE); provide defaults when missing.
- `supportedFileTypes` (FE) ← from `best_for` (BE) or default `['*']`.
- `parameters` (FE) ← derive from `default_config` (BE) where possible (keys, defaults) and supplement with documented ranges.
- `isRecommended`/`recommendedFor` (FE) ← optional flags inferred from BE data or defaults.

## Acceptance Criteria (updated)
- `GET /api/v2/chunking/strategies` returns HTTP 200 and an array of exactly 6 strategies with public IDs matching FE expectations (see alias rules).
- Either the backend response directly matches FE `ChunkingStrategy` shape, or a documented FE adapter performs a one-to-one mapping with test coverage.
- Contract tests validate field presence and types against FE's `ChunkingStrategy`.
- Error case returns a 5xx with a useful `detail` message.

## Technical Notes (updated)
- A BE response-shaping layer may expose FE-oriented fields without leaking domain internals.
- Prefer exposing a shared canonical schema/definitions under `packages/shared/chunking` to keep FE/BE aligned and reduce drift.
- The current router `packages/webui/api/v2/chunking.py` can be extended to include FE-native fields if we choose server-side response shaping.

## Suggested Files
- Backend API router/handler (example paths):
  - `packages/.../api/v2/chunking.py` (or equivalent router file)
- Shared types/constants (optional but recommended):
  - `packages/shared/chunking/application/strategies.py` (or similar)
- Tests:
  - `tests/backend/test_chunking_strategies_endpoint.py`

## Test Commands
- Python: `pytest tests/backend/test_chunking_strategies_endpoint.py -v`

## Dependencies
- Phase 2 (done): base strategies endpoint implemented.
- FE-STRAT-002 will consume this; coordinate on the chosen mapping approach.
