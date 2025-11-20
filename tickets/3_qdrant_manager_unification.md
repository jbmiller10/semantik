Title: Unify Qdrant manager usage across WebUI and tasks

Background
- Two managers exist: shared `packages/shared/managers/qdrant_manager.py` (blue/green, cleanup) and legacy `packages/webui/utils/qdrant_manager.py`.
- `CollectionService` defaults to the legacy manager; Celery ingestion/reindex tasks reference it.
- Duplicate paths complicate staging/cleanup and testing.

Goal
Use a single Qdrant manager implementation everywhere, with clear DI for tests/mocks.

Scope
- Replace all imports of the legacy manager with the shared manager (or an injected interface) in `CollectionService`, Celery tasks, and utilities.
- Remove `_LightweightQdrantManager` unless tests need it; if needed, move a minimal mock into tests.
- Delete `packages/webui/utils/qdrant_manager.py`; ensure blue/green staging/cleanup flows use the shared manager.
- Update tests/mocks to import from unified location; add a small integration-style test for staging collection create/list/delete using a mocked client.

Out of Scope
- Qdrant schema or vector config changes.

Suggested Steps
1) Grep for legacy manager imports; swap to shared manager with DI hook for tests.
2) Wire `CollectionService` and Celery tasks to accept injected manager; default to shared instance.
3) Remove legacy module; adjust any factory/utility that referenced it.
4) Update tests to use shared manager mocks; add a test covering staging collection lifecycle via the unified API.

Acceptance Criteria
- Only `shared.managers.qdrant_manager` is used in runtime code; legacy module removed.
- Collection create/reindex and cleanup paths work in dev stack with unified manager.
- Tests pass with updated mocks; staging lifecycle test present.
