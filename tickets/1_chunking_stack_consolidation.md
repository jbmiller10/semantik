Title: Consolidate chunking stack (keep plugin hook, remove legacy/JSON paths)

Background
- Current code mixes orchestrator DTOs with legacy factories/adapters (`packages/webui/services/chunking_service.py`, `services/chunking/adapter.py`, `services/chunking/container.py`, API v2 chunking routes, Celery tasks in `packages/webui/chunking_tasks.py` and `packages/webui/tasks/ingestion.py`).
- Configs persist to `data/chunking_configs.json`, unsuitable for multi-replica and audit.
- Third-party chunking strategies already plug in via the strategy registry; we need to keep and clarify that extension point.

Goal
Make the orchestrator the sole chunking path, delete legacy shims and file-based config, and explicitly support third-party strategy plugins.

Scope
- Preserve and document the plugin contract: keep `STRATEGY_REGISTRY`/loader; orchestrator DTOs must accept `strategy_name` + `strategy_config` resolved through the registry. Add a short plugin author guide (interface: validate config, chunk()) and where to register.
- Remove legacy factories/adapters/shims (`ChunkingService` legacy paths, `chunking_exceptions`, `get_legacy_chunking_service` DI). Collapse API and Celery to orchestrator-only.
- Replace JSON config persistence with a DB-backed store (model + repository + Alembic migration) and wire orchestrator/config manager to it.
- Simplify Celery chunking tasks to call orchestrator directly; drop legacy kwargs normalization and dead code.
- Update docs (`packages/webui/services/README.md`, `docs/api/CHUNKING_API.md`) to reflect single-path flow and plugin hook; refresh tests accordingly.

Out of Scope
- Changing chunk quality or strategy algorithms.
- UI changes beyond fixing broken references/tests.

Suggested Steps
1) Inventory imports of legacy `ChunkingService`/adapters and swap call sites to orchestrator equivalents (API deps, Celery, services).
2) Implement DB-backed chunking config store; migrate orchestrator/config manager to it; remove `chunking_configs.json` reads/writes.
3) Delete legacy modules/adapters and related DI hooks; ensure tests/fixtures use orchestrator services.
4) Rework Celery chunking tasks to consume orchestrator DTOs directly; remove kwargs normalization paths.
5) Document plugin interface + registration location; add a minimal example plugin and a test ensuring registry-based strategy executes end-to-end.
6) Update docs/tests to match the new stack; remove legacy-only tests.

Acceptance Criteria
- No runtime imports of legacy chunking adapters/factories; DI returns orchestrator services only.
- Strategy/plugin registry remains functional and documented; a sample plugin test passes through orchestrator.
- Config persistence uses DB model; no references to `chunking_configs.json` remain.
- Celery chunking tasks invoke orchestrator directly; chunking-related tests pass.
- Documentation reflects single-path architecture and plugin hook; no mentions of “legacy chunking service.”
