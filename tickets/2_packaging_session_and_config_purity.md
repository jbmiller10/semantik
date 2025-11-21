Title: Remove path hacks, make configs pure, unify session creation for API and workers

Background
- Entry points (`packages/webui/main.py`, `packages/vecpipe/search_api.py`) mutate `sys.path` to import siblings.
- Config modules (`shared/config/base.py`, `shared/config/webui.py`) create dirs and JWT secrets on import, causing side effects and cross-pod drift.
- DB session access uses `AsyncSessionLocalWrapper` and ad-hoc event loop creation in Celery tasks (`packages/webui/tasks/ingestion.py`, `packages/webui/chunking_tasks.py`).

Goal
Provide clean packaging/imports, side-effect-free config, and a single async session factory injected everywhere (API + Celery).

Scope
- Package layout: install repo as editable/namespace package; remove all `sys.path.append` and wrapper files (`packages/webui/app.py`, `packages/webui/tasks.py`). Update Makefile/dev scripts accordingly.
- Config purity: move directory creation and JWT generation to explicit startup hooks (FastAPI lifespan, Celery boot). Require JWT via env even in dev; provide a helper script to generate once. Imports must be pure.
- Session factory: expose one async sessionmaker from `shared/database/postgres_database.py`; inject into FastAPI deps and Celery. Delete `AsyncSessionLocalWrapper`, loop-creation fallbacks, `_pg_connection_manager_override`.
- Tests/smoke: add fixtures ensuring session acquisition works in TESTING=true (mock DB) and normal dev.

Out of Scope
- Connection pool tuning or perf benchmarking.

Suggested Steps
1) Create/adjust pyproject/uv config for editable install; update run scripts to use module entrypoints without path hacks.
2) Strip `sys.path` mutations and delete wrapper modules; fix imports that relied on them.
3) Refactor config modules to pure dataclasses/settings; move fs/JWT side effects to startup hooks; add helper script for JWT generation.
4) Replace session access in FastAPI deps and Celery tasks with injected sessionmaker; remove custom loop logic; add small integration test for worker session use.
5) Update docs (README/AGENTS) to describe new startup hook requirements.

Acceptance Criteria
- Entry points contain no `sys.path` manipulation; services run via installed package.
- Config imports perform zero filesystem writes; JWT is provided via env or startup hook. Purity validated by import in a temp dir.
- API and Celery use the same sessionmaker; no custom event loop creation remains; related tests pass.
- Dev commands (`make dev`, `make run`, Celery worker) succeed using the new pattern and documented steps.
