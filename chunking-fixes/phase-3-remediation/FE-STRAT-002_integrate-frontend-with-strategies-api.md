# FE-STRAT-002: Integrate Frontend with Strategies API

## Context (updated)
Phase 3 validation expects the frontend to fetch strategies from the backend (6 total) instead of relying on local constants. The backend strategies endpoint is now available at `GET /api/v2/chunking/strategies` (Phase 2). FE needs to consume it and present strategy metadata in the existing UI.

## Requirements (updated)
- Add `getStrategies()` to `apps/webui-react/src/services/api/v2/chunking.ts` using `GET /api/v2/chunking/strategies` and returning `ChunkingStrategy[]`.
- Map backend fields to FE types if backend response isn't FE-native yet (see Mapping). The goal is zero change to core FE types.
- Update the chunking store to fetch and use strategies:
  - Add `fetchStrategies()` in `apps/webui-react/src/stores/chunkingStore.ts` which calls `chunkingApi.getStrategies()`.
  - Populate `strategies` state, set `selectedStrategy` to the first strategy `type` if none.
  - Handle errors via `handleChunkingError`.
- Update strategy-related components to read strategies from the store, not static constants.
- Add unit tests for store fetching and a representative component rendering strategies.

### Mapping
If the backend response is not yet FE-native, map as follows in the FE API client:
- `type` (FE) ← `id` (BE) with aliasing if needed (e.g., backend `character` may be exposed as FE `fixed_size`).
- `name` (FE) ← `name` (BE).
- `description` (FE) ← `description` (BE).
- `icon` (FE) ← provide defaults per strategy (e.g., `Type`, `GitBranch`, `FileText`, `Brain`, `Network`, `Sparkles`).
- `performance` (FE) ← derive from `performance_characteristics` (BE) or use reasonable defaults (`speed`, `quality`, `memoryUsage`).
- `supportedFileTypes` (FE) ← from `best_for` (BE) or default `['*']`.
- `parameters` (FE) ← derive from `default_config` (BE) to populate keys and `defaultValue`; supply ranges/options consistent with existing constants.
- `isRecommended` / `recommendedFor` (FE) ← optional fields; provide sensible defaults (e.g., `recursive`/`hybrid` recommended).

## Acceptance Criteria (updated)
- Calling `fetchStrategies()` populates 6 strategies from backend and selects a default strategy.
- Components render non-mock strategies from the store.
- If a mapping layer is used, unit tests validate mapping correctness for at least 3 representative strategies.
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
- Depends on BE-STRAT-001 (backend endpoint implemented in Phase 2).
- Coordinate on aliasing: if backend exposes `character`, FE may represent it as `fixed_size` (or vice versa). Ensure consistency across UI and API client.
