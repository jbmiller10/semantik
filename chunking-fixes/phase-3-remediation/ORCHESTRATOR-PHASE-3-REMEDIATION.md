# ORCHESTRATOR: Phase 3 Remediation – Chunking Frontend Integration

## Phase Overview
- Priority: Critical – close gaps left in Phase 3
- Duration: 1 working day (can be parallelized)
- Success Gate: API-backed strategies in UI; cancel flows (WS/REST) working; a11y tests passing; docs aligned; validation report complete
- Backend status: Strategies API implemented in Phase 2 (GET /api/v2/chunking/strategies). FE should either consume FE-shaped fields directly or apply the documented mapping from backend StrategyInfo to FE ChunkingStrategy.

## Tickets In Scope
- BE-STRAT-001: Implement Strategies API (GET /api/v2/chunking/strategies)
- FE-STRAT-002: Integrate Frontend with Strategies API
- BE-WS-003: Add WebSocket Cancel Support for Preview/Comparison
- FE-WS-004: Wire WebSocket Cancel in Frontend + UI Controls
- FE-REST-005: Add REST Cancel UI Control (Fallback Mode)
- FE-A11Y-006: Add Accessibility Tests and Fix Basic Issues
- FE-DOCS-007: Align Docs and Paths with Current Implementation
- QA-VALID-008: Phase 3 Frontend Final Validation

## Execution Order and Parallel Plan
1) BE-STRAT-001 → 2) FE-STRAT-002
3) BE-WS-003 → 4) FE-WS-004
5) FE-REST-005 (independent)
6) FE-A11Y-006 (after 2)
7) FE-DOCS-007 (after 2)
8) QA-VALID-008 (last)

Parallelization:
- Run (1) and (3) in parallel (separate implementers).
- After (2), start (6) and (7) in parallel.
- (5) can start any time; finish before (8).

Notes:
- (1) backend is already in place; FE (2) should align on public IDs (`character` vs `fixed_size`) and field names as per updated tickets.

## Pre‑Flight Checklist
- Phase 2 backend stable and running locally.
- WebSocket server reachable in dev.
- Local Node and Python toolchains installed per repo.
- Test data available for preview/comparison flows.

## Branch & PR Workflow
- Base branch: `feature/improve-chunking`
- Working branch: `phase-3-remediation-<date>`

Commands:
```
# prepare
git checkout feature/improve-chunking
git pull origin feature/improve-chunking

# branch
git checkout -b phase-3-remediation-<YYYYMMDD>
```

Commit style:
- One commit per ticket or cohesive group.
- Message includes ticket id, summary, key changes, and test status.

PR:
- Base: `feature/improve-chunking`
- Title: "Phase 3 Remediation: Chunking Frontend Integration"
- Body: summary, completed tickets checklist, test/lint status, success criteria mapping.

## Assignment Templates
Implementation:
```
Please implement [TICKET-ID] from chunking-fixes/phase-3-remediation/[TICKET-ID]_*.

Reminders:
- Pre-release: delete old code if needed; no backwards-compat layers.
- Follow acceptance criteria exactly.
- Add/Update unit tests.
- Keep interfaces simple and typed.

Report back:
1) Files modified/created/deleted
2) Test results
3) Open concerns/blockers
```

Review:
```
Please review implementation for [TICKET-ID].

Checklist:
- [ ] Acceptance criteria met
- [ ] No backwards-compat code (pre-release)
- [ ] Security and performance sane
- [ ] Tests cover changes; pass locally
- [ ] Code matches project patterns
- [ ] No over-engineering
```

## Technical Decisions (Reaffirm)
- 100 direct LIST partitions (no virtual mapping) – already implemented in DB layer.
- AsyncIO streaming with 64KB buffers – ensure FE handles progress events; UTF-8 boundaries handled server-side.
- Redis Pub/Sub for WebSocket fanout – ensure cancel signals propagate to workers.
- Modular monolith – simple adapters over domain logic.

## Quality Gates
- Frontend: `cd apps/webui-react && npm run test:ci` passes.
- Backend: `pytest` for new endpoints/WS tests pass.
- Linting: existing repo standards (ruff/black for Python; eslint/vitest for FE) pass.
- Accessibility: axe tests show zero violations in target components.

## Commands Reference
Frontend:
```
cd apps/webui-react
npm run test:ci
npm run build
```
Backend (examples, adapt to repo):
```
pytest tests/backend/test_chunking_strategies_endpoint.py -v
pytest tests/backend/test_websocket_chunking_cancel.py -v
```

## Escalation Triggers
- Backend strategies endpoint contract drift vs FE types.
- WebSocket cancel not halting worker within 200ms (simulated workload).
- Non-deterministic tests or flakiness in WS progress handling.
- Persistent a11y violations after fixes.

## Phase Success Criteria
- FE reads 6 strategies from API; UI reflects them (no mock data).
- WS cancel and REST cancel both functional and tested; no lingering updates post-cancel.
- Axe tests: zero violations on target chunking components.
- PR raised with green CI and validation report attached.

## Final Validation
- Execute QA-VALID-008; attach `PHASE3_VALIDATION_REPORT.md` summarizing:
  - Tests passing counts
  - Accessibility results
  - Manual sanity checks (preview, cancel, fallback)
  - Any known follow-ups (if any)
