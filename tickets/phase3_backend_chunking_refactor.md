# Phase 3 Ticket: Modularize Chunking & Error Handling Services (Target Window: November 24 – December 6, 2025)

## Background
The audit and Gemini agent recommendations identified "god objects" within `packages/webui/services/chunking_service.py` and `packages/webui/services/chunking_error_handler.py`. Responsibilities like progress updates, error classification, and orchestration are tightly coupled, making the code hard to test or explain in portfolio materials. This ticket isolates backend refactoring work for Phase 3, while documentation is handled in a sibling ticket.

## Objectives
1. Decompose chunking and error-handling logic into focused components with clear responsibilities.
2. Eliminate duplicated utilities by centralizing progress reporting and strategy mapping.
3. Maintain backward-compatible interfaces to avoid regressions in vecpipe/webui callers.

## Requirements
### 1. Service Decomposition
- Extract classes/modules from `chunking_service.py` and `chunking_error_handler.py`:
  - `ProgressUpdateManager` (encapsulates async progress broadcasts).
  - `ChunkingOrchestrator` (or similar) to coordinate parsing, strategy selection, storage, and callback triggering.
  - `ErrorClassifier` and recovery helpers that determine retry/abort behavior.
- Place new modules under `packages/shared/chunking/services/` (or a logical location) with type-safe interfaces.
- Update existing services to delegate to the new components while preserving public method signatures.

### 2. Shared Utilities
- Create/extend shared modules for:
  - Strategy mapping constants (ensure a single source of truth used by both services and tests).
  - Cache key generation used across chunking flows.
  - Default configuration retrieval (`get_default_config`) to eliminate duplicate implementations.
- Remove obsolete TODO comments referencing unimplemented pieces.

### 3. Testing
- Add unit tests for each new component (`tests/chunking/test_progress_update_manager.py`, etc.) covering success and failure scenarios.
- Update integration tests to ensure workflows still succeed (e.g., `tests/webui/services/test_chunking_service.py`).
- Verify error recovery paths behave identically (or improved) versus pre-refactor.

### 4. Backward Compatibility & Performance
- Ensure public API of chunking services remains stable; update docstrings and type hints accordingly.
- Run profiling or smoke tests to confirm no significant performance regressions occur (document findings).

## Acceptance Criteria
- `chunking_service.py` and `chunking_error_handler.py` primarily coordinate the new helper classes rather than containing monolithic logic.
- Shared utilities exist in dedicated modules without duplication or TODO stubs.
- `uv run pytest tests -k "chunking"` passes with new unit coverage.
- No regressions detected in functional chunking workflows.
- Inline docstrings/comments explain the new architecture.

## Validation Steps
1. `uv run pytest tests -k "chunking"`
2. `make type-check` (ensures new modules are typed correctly)
3. Optional: `uv run pytest tests/e2e/test_collection_workflow.py` to confirm end-to-end chunking still succeeds.
4. Document quick profiling (if performed) in commit notes or summary.

## Coordination Notes
- Coordinate with the documentation ticket to keep architecture diagrams consistent.
- Communicate any interface changes to teams relying on chunking services (e.g., vecpipe workers).
- Stage refactor in incremental commits to simplify review.

## Out of Scope
- Introducing new chunking strategies or semantic embedding integrations.
- Large-scale performance tuning beyond ensuring no regressions.
- Frontend/documentation updates (covered in companion tickets).
- Rewriting services in another language or framework.
