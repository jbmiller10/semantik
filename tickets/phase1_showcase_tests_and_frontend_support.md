# Phase 1 Ticket: Showcase Tests & Frontend Alignment (Target Window: October 27 – November 7, 2025)

## Background
Once Phase 0 security work lands, we must demonstrate Semantik's strengths through automated tests and UI updates. The review identified skipped WebSocket tests, missing end-to-end coverage for ingestion/search, and outdated frontend types that break hybrid search. This ticket equips stateless implementers to build portfolio-grade validation artifacts and finish frontend wiring.

## Objectives
1. Re-enable and expand WebSocket/unit tests in the React app.
2. Add high-impact Python tests (E2E ingestion flow, partition pruning, security regression) to spotlight engineering rigor.
3. Align frontend request/response types with the backend updates from the "Phase 1 Core Feature Completeness" ticket.

## Detailed Requirements
### 1. React WebSocket & Hybrid Frontend Tests
- Unskip the tests in `apps/webui-react/src/services/__tests__/websocket.test.ts` and update mocks to reflect the current WebSocket payloads.
- Add coverage for reconnect/backoff behavior and permission-denied messages.
- Update `apps/webui-react/src/services/api/v2/types.ts` to tie into the backend's new hybrid search enums (`HybridMode = 'filter' | 'rerank'`, etc.). Update any API client calls to use the new types and default mapping.
- Provide component-level tests ensuring the UI sends valid payloads (e.g., `SearchInterface.test.tsx` verifying form submission builds the new request body).

### 2. Portfolio-Grade Python Tests
- **E2E Ingestion & Search Flow**: Create `tests/e2e/test_collection_workflow.py` covering collection creation → document upload → indexing operation monitoring → hybrid search confirmation.
  - Use the dedicated testing Postgres profile (`docker compose --profile testing up -d postgres_test`) if required.
  - Mock external services (Qdrant) where appropriate, but ensure the test remains realistic.
- **Partition Pruning Validation**: Add `tests/database/test_partition_pruning.py` to assert that queries against the `chunks` table leverage partition pruning (verify `EXPLAIN` results omit full table scans).
- **Security Regression Suite**: Extend existing auth tests or add `tests/webui/security/test_auth_guards.py` to cover #403 responses, default secret enforcement side effects, and partition monitoring access.

### 3. Documentation & CI Integration
- Document how to run the new E2E suite in `tests/README.md` (including any Docker profile requirements).
- Ensure new tests run (or are optionally triggered) via existing `make test` flows; update `Makefile` targets if needed to include the e2e path.

## Acceptance Criteria
- React WebSocket tests run without skipped markers; coverage includes reconnect behavior and permission denied messaging.
- Frontend hybrid search forms submit payloads matching the backend contract and TypeScript compiles cleanly.
- Python test suite includes the new E2E, partition pruning, and auth regression tests, all passing under `uv run pytest`.
- Documentation describes how to execute the new tests locally.
- `npm test --prefix apps/webui-react` and `uv run pytest tests -k "workflow or pruning or auth"` both succeed.

## Validation Checklist
1. `npm test --prefix apps/webui-react -- websocket` (ensures updated WebSocket tests pass).
2. `npm test --prefix apps/webui-react -- SearchInterface` (hybrid request payload test).
3. `uv run pytest tests/e2e/test_collection_workflow.py -v`.
4. `uv run pytest tests/database/test_partition_pruning.py -v`.
5. `uv run pytest tests/webui/security/test_auth_guards.py -v`.
6. Optionally, run `make test` to confirm suites integrate cleanly.

## Dependencies & Coordination
- Backend payload/schema updates must merge first (see `phase1_core_feature_completeness.md`).
- Coordinate with DevOps for any modifications to CI runners if E2E tests require Docker services.
- If test runtimes become lengthy, tag heavier suites so CI can run them nightly while providing an opt-in target for local validation.

## Out of Scope
- Creating entirely new UI features beyond the test coverage and type alignment described above.
- Expanding test suites to unrelated domains (e.g., vecpipe ingestion internals) unless they directly support the new workflows.
- Performance/stress testing of WebSocket throughput or ingestion pipelines.
- CI pipeline changes beyond wiring the new tests into existing commands.
