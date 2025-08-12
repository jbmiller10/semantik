# FE-STRAT-002: Integrate Frontend with Strategies API

## Context
Phase 3 validation expects the frontend to fetch strategies from the backend (6 total) instead of relying on local constants. Implement `getStrategies()` on the FE API client and wire it into the store and components.

## Requirements
- Add `getStrategies()` to `apps/webui-react/src/services/api/v2/chunking.ts` using `GET /api/v2/chunking/strategies` and returning `ChunkingStrategy[]`.
- Update the chunking store to fetch and use strategies:
  - Add `fetchStrategies()` in `apps/webui-react/src/stores/chunkingStore.ts` which calls `chunkingApi.getStrategies()`.
  - Populate `strategies` state, set `selectedStrategy` to the first id if none.
  - Handle errors via `handleChunkingError`.
- Update strategy-related components to read strategies from the store, not static constants.
- Add unit tests for store fetching and a representative component rendering strategies.

## Acceptance Criteria
- Calling `fetchStrategies()` populates 6 strategies from backend and selects a default strategy.
- Components render non-mock strategies from the store.
- Unit tests for store and at least one component pass.

## Technical Notes
- Optional fallback: keep local constants only if endpoint fails; otherwise prefer failing fast (pre-release stance).

## Suggested Files
- `apps/webui-react/src/services/api/v2/chunking.ts` (add `getStrategies()`).
- `apps/webui-react/src/stores/chunkingStore.ts` (add `fetchStrategies()` + state).
- `apps/webui-react/src/components/chunking/*` where strategies are displayed.
- Tests: `apps/webui-react/src/stores/__tests__/chunkingStore.test.ts`.

## Test Commands
- `cd apps/webui-react && npm run test:ci`

## Dependencies
- Depends on BE-STRAT-001.
