# BE-STRAT-001: Implement Strategies API (GET /api/v2/chunking/strategies)

## Context
The frontend currently uses static `CHUNKING_STRATEGIES` for strategy metadata. Phase 3 validation and prior tickets reference a dynamic `getStrategies()` call that should return 6 strategies. Provide a real backend endpoint so the frontend can fetch strategies with types that match the FE data model.

## Requirements
- Add an endpoint `GET /api/v2/chunking/strategies` that returns all 6 strategies with full fields used by FE:
  - `type`, `name`, `description`, `performance`, `supportedFileTypes`, `parameters` (with `key`, `type`, `defaultValue`, ranges/options, etc.), `isRecommended`, `recommendedFor`.
- Ensure the response schema aligns exactly with `apps/webui-react/src/types/chunking.ts` (do not force FE to map/transform fields).
- Return a stable list of 6 strategies: `character`, `recursive`, `markdown`, `semantic`, `hierarchical`, `hybrid`.
- Add unit tests for the endpoint and a contract validation to ensure schema compatibility.

## Acceptance Criteria
- `GET /api/v2/chunking/strategies` returns HTTP 200 and an array of exactly 6 strategy objects.
- Each strategy object matches FE `ChunkingStrategy` fields and types.
- Contract test verifies response structure compatible with FE `ChunkingStrategy`.
- Error case returns a 5xx with a useful `detail` message.

## Technical Notes
- Prefer exposing a shared constant/schema to keep FE/BE aligned. Consider placing canonical definitions under `packages/shared/chunking`.
- Implement as a thin controller serving the canonical dataset (no DB dependency required).

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
- None. This is a backend-first enabler for FE-STRAT-002.
