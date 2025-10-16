# HANDOFF.md

## Context

Semantik is a monorepo that delivers a self-hosted semantic-search stack:

- **Backend services**
  - `packages/shared`: shared configs, embedding subsystem, chunking domain, database abstractions.
  - `packages/vecpipe`: FastAPI search API, model manager, hybrid retrieval, Qdrant integration.
  - `packages/webui`: FastAPI web app, Celery worker, Redis/WebSocket infrastructure, PostgreSQL data layer.
- **Frontend**: `apps/webui-react`, a Vite/React 19 SPA using Zustand + TanStack Query.
- **Tooling**: `uv` for dependency management, pytest/ruff/mypy, Docker Compose for Postgres + Redis + Qdrant.

CI (`.github/workflows/main.yml`) runs linting, type-checks, backend/frontend tests, build verification, and Trivy scans.

You have been asked to execute four immediate remediation tasks (numbers refer to the earlier audit list):

1. **Internal API key pipeline robustness.**
3. **Chunking v2 endpoints implementation parity.**
5. **Stabilise test harness to avoid mandatory Redis/Celery.**
6. **Increase test coverage for high-risk modules.**

This document captures the current state, concrete issues, and the implementation plan for each task. Treat the repo as production-adjacent: avoid breaking the Docker workflow or existing CI unless explicitly coordinated.

---

## Repository Notes

- Python requires 3.11 (see `pyproject.toml`).
- Dependencies managed via `uv.lock`; prefer `uv sync --frozen` / `uv run`.
- Celery + Redis heavily integrated; tests sometimes patch these.
- Configuration is Pydantic-based (`packages/shared/config`). Settings are cached module singletons (`settings`).
- React app lives in `apps/webui-react`. Tokens currently stored in `localStorage`.

Useful scripts:

- `make dev-install` / `uv sync` – install deps.
- `make lint`, `make type-check`, `make test` – backend automation.
- `npm ci`, `npm run test:ci` inside `apps/webui-react`.

---

## Findings Recap & File Map

| Area | File(s) | Notes |
|------|---------|-------|
| Internal API key handling | `packages/webui/main.py`, `packages/webui/tasks.py`, `packages/shared/config/base.py` | Key defaults to `None`; Celery sends `X-Internal-API-Key: None`, so `/api/internal/complete-reindex` rejects reindex swaps. Current workaround only triggers when env value is literal `"change-me-in-production"`. Key is logged at WARN when auto-generated. |
| Chunking v2 API | `packages/webui/api/v2/chunking.py` | Multiple endpoints return stub data with TODOs (lines ~600+, 708, 793, 920, 977). No orchestration to service layer. |
| Test harness failure | `tests/unit/test_collection_service.py` integration case, Celery backend initialization. Running `uv run pytest tests/unit --maxfail=1 -q` fails because Celery tries to open Redis. |
| Coverage gaps | `packages/webui/tasks.py`, `packages/webui/rate_limiter.py`, `packages/webui/middleware/*`, `packages/webui/services/resource_manager.py`, etc., have <20 % coverage. HTML report lives under `htmlcov/`. Need targeted tests aligned with features we touch. |

---

## Task Guides

### 1. Harden Internal API Key Handling

**Goal**: Ensure internal reindex handoffs succeed reliably without leaking secrets.

**Current Flow**:
- `packages/shared/config/base.py` defines `INTERNAL_API_KEY` default `None`.
- `packages/webui/main.py::_configure_internal_api_key` only generates a random key if value equals `"change-me-in-production"`. Otherwise, leaves `None`.
- `packages/webui/tasks.py::_process_collection_operation_async` constructs `headers = {"X-Internal-API-Key": settings.INTERNAL_API_KEY, ...}`; when `None`, `/api/internal/complete-reindex` throws 401.
- WARN log exposes generated key.

**Plan**:
1. Update config logic so that absence (`None` or empty) triggers generation in **non-production** but requires explicit value in production:
   - Modify `_configure_internal_api_key` to check `not shared_settings.INTERNAL_API_KEY`.
   - Move generation to shared utility to reuse in Celery; when a key is minted persist it to `.data/internal_api_key` (or similar bind-mounted secret path) so worker-only processes can read the same value at boot.
   - Remove logging of raw secret; emit a hash or reference to the persistence path instead.
2. Guarantee Celery side picks up the same value:
   - On web app startup call the shared helper before tasks fire, then have `packages/webui/celery_app.py` invoke the same helper when the worker boots so standalone Celery instances load the persisted key if present and only generate a new one in non-production test flows.
3. Add regression tests:
   - Unit test for `_configure_internal_api_key` verifying generation when empty and raising when production with default.
   - Integration-style test that exercises `complete_reindex` call path with patched HTTP client to ensure header is populated.

**Files to touch**: `packages/webui/main.py`, potentially `packages/webui/celery_app.py` or `packages/webui/tasks.py` for fallback, maybe `packages/shared/config/base.py` or new helper module for persistence.

### 3. Wire Up Chunking v2 Endpoints

**Goal**: Replace placeholder responses with real service-layer interactions (or explicitly 501) for public API reliability.

**Relevant Endpoints** (`packages/webui/api/v2/chunking.py`):
- `/collections/{collection_uuid}/chunks`
- `/collections/{collection_id}/chunking-stats`
- `/analytics/*`, `/quality-scores`, `/configurations` etc. (search for `TODO: Implement service layer method`).

**Existing Infrastructure**:
- Chunking services live in `packages/webui/services/chunking_service.py` and related DTOs (`packages/webui/services/dtos/`).
- There’s a README guiding service usage (`packages/webui/services/README.md`).
- Some service methods may already exist; confirm before implementing. If missing, define them alongside DTO outputs.

**Plan**:
1. Audit `ChunkingService` & `ChunkingStrategy*` classes to see which features are implemented. Fill in missing service methods aligning with API expectations (e.g., pagination, stats) and cross-check `packages/shared/chunking/application` (and related domain helpers) before adding new logic.
2. For each TODO endpoint:
   - Implement call to service; transform service DTOs → API schemas defined in `packages/webui/api/v2/chunking_schemas.py`.
   - Handle errors via consistent `HTTPException`.
   - Apply rate limiting/circuit breaker logic already present.
3. Ensure tests cover both success and failure (permission, validation). Extend `tests/webui/api/v2` suite or create new module(s). Use dependency overrides + fakeredis fixtures (see `tests/conftest.py` for patterns).
4. Update API docs/comments once functional, including FastAPI response models and router-level docstrings so OpenAPI stays accurate. Remove TODO blocks.

**Deliverables**: Fully wired endpoints, new/updated tests, green CI.

### 5. Stabilise Test Harness (Celery/Redis)

**Problem**: Unit suite fails without Redis because Celery attempts real connections (see failure at `TestCollectionServiceIntegration::test_collection_state_validation`).

**Plan**:
1. Identify entry point causing Redis connection during tests:
   - Celery app import (`packages/webui/services/collection_service.py` uses `from packages.webui.celery_app import celery_app`).
   - Celery backend likely initialises at import time.
2. Provide a test fixture or settings flag to disable Celery backend initialization:
   - Option A: In test mode (`os.environ["TESTING"]` is set in `tests/conftest.py`), configure Celery to use in-memory backend (e.g., `celery_app.conf.update(task_always_eager=True, broker_url="memory://", result_backend="cache+memory://")`).
   - Option B: Patch Celery `send_task` in tests to no-op.
   - Option C: Introduce lazy initialization when `settings.USE_MOCK_EMBEDDINGS` or `TESTING` set.
3. Codify the chosen approach as an autouse fixture in `tests/conftest.py` (e.g., `@pytest.fixture(autouse=True) def celery_memory_backend(...)`) so every unit suite inherits the in-memory configuration, and document the `TESTING` flag expectation for developers.
4. Ensure fakeredis patch in `tests/conftest.py` covers Celery if we keep Redis interface but want fake connections.
5. Re-run `uv run pytest tests/unit -q` locally to confirm no external dependency is needed.
6. Document how to run the suite in `TEST_REPORT.md` or new developer README snippet.

**Additional**: If we change Celery config, run integration tests referencing Celery to confirm behaviour unchanged in production.

### 6. Improve Coverage on High-Risk Modules

Focus on modules touched or related to tasks 1 & 3 and critical infrastructure:

1. **`packages/webui/tasks.py`** (Celery pipeline for reindex, operations):
   - Write tests to cover `process_collection_operation` happy path, error handling, and reindex API call (with HTTPX mocked).
   - Use `pytest` + `pytest-asyncio`. Provide event loop fixture.
2. **Middleware & Rate Limiter**:
   - Add tests for `RateLimitMiddleware`, `rate_limit_exceeded_handler`, and `CSPMiddleware` applying appropriate headers. Use FastAPI TestClient + dependency overrides.
3. **Resource Manager**:
   - Unit tests for `can_create_collection`, `can_allocate`, `estimate_resources`, using psutil mocks.
4. **Chunking endpoints** (from task 3) – tests will contribute to coverage.

Leverage `pytest --cov` to ensure improvements; update `TEST_REPORT.md` with new results. Keep tests deterministic and service-light (mock heavy IO).

---

## Suggested Implementation Order

1. **Test harness stabilisation (Task 5)**: ensures you have reliable feedback loop before deeper changes.
2. **Internal API key fix (Task 1)**: unblocks reindex behaviour; add tests.
3. **Chunking endpoint wiring (Task 3)**: more involved; build on stable testing.
4. **Coverage expansion (Task 6)**: once functionality solid, backfill tests.

---

## Testing Checklist

After each major change run:

```bash
uv run pytest tests/unit -q
uv run pytest tests/webui -q
make lint
make type-check
```

For chunking endpoints, add targeted API tests and, if possible, run a subset with mocked DB/Redis to confirm responses.

React changes (if any) → `cd apps/webui-react && npm run test:ci`.

Once ready, execute `uv run pytest --cov=packages.webui.tasks --cov=packages.webui.api.v2.chunking --cov=packages.webui.middleware --cov=packages.webui.services.resource_manager` to verify improved coverage.

---

## Open Questions / Assumptions

- We ignore tasks 2 & 4 per instruction (reset endpoint hardening, authentication redesign). Keep note but no action now.
- Celery worker lifecycle vs. internal API key generation: solution will persist the generated key to `.data/internal_api_key` and have both webui and worker call the same helper; verify that path is shared/mounted in container deployments and document rotation steps.
- Chunking service capabilities: double-check for overlapping implementations; avoid re-inventing if functions exist under `packages/shared/chunking/application`.

---

## Resources & References

- `TEST_REPORT.md` documents previous chunking test results—use as baseline for coverage updates.
- `apps/webui-react/ARCHITECTURE.md` for front-end context (useful if endpoints change payloads).
- `packages/webui/services/README.md` for service orchestration patterns.
- `scripts/` folder contains helpers (e.g., `verify_*`) that illustrate intended behaviours.

---

## Delivery Expectations

Produce PR(s) that:

1. Fix internal API key generation + add regression tests.
2. Implement chunking v2 endpoints using service layer, remove TODOs, add tests.
3. Ensure unit tests run without Redis/Celery infrastructure; document approach.
4. Raise coverage for the specified modules, with updated `TEST_REPORT.md` or new coverage summary.

Ensure all CI jobs pass and provide instructions for operators if new environment variables or secrets are required.

---

## Progress Update – October 16, 2025

### Completed Work

- **Task 1 (Internal API key handling)** is implemented end-to-end:
  - Added `packages/shared/config/internal_api_key.py` to manage key generation, persistence (`data/internal_api_key`), and reuse across processes.
  - `packages/webui/main.py`, `packages/webui/celery_app.py`, and `packages/webui/tasks.py` now consume the helper, removing inline key generation and ensuring Celery-only processes load the shared secret without logging it.
  - Coverage added via `tests/shared/test_internal_api_key.py`, updates to `tests/test_internal_api.py`, and `tests/webui/test_celery_tasks.py`.
- **Task 3 (Chunking v2 endpoints)** now calls real service-layer implementations:
  - `packages/webui/services/chunking_service.py` gained chunk pagination, global metrics, quality analysis, document analysis, and user configuration persistence (`data/chunking_configs.json`).
  - DTO/schema updates in `packages/webui/services/dtos/api_models.py` and `packages/webui/services/dtos/chunking_dtos.py` plus router wiring in `packages/webui/api/v2/chunking.py`.
  - Direct endpoint coverage expanded in `tests/webui/api/v2/test_chunking_direct.py`.
- **Task 5 (Test harness stabilization)** remains in place from earlier work:
  - `packages/webui/celery_app.py` now builds the Celery app through helper functions that detect the `TESTING` flag. In that mode it swaps to in-memory broker/result backends, disables retry/event wiring, and otherwise leaves production defaults intact.
  - `tests/conftest.py` imports the Celery module via `importlib` and installs an autouse fixture that replaces `celery_app.send_task` with a stubbed `MagicMock`, preventing accidental broker calls in unit tests.
  - `tests/unit/test_models.py` falls back to an in-memory SQLite engine when PostgreSQL isn’t reachable. The fallback creates only the tables these timezone-awareness tests depend on, avoiding previous foreign-key/table errors while preserving FK enforcement.
  - Verified locally with `uv run pytest tests/unit --maxfail=1 -q` (passes ~163 s).
- **Task 6 (Coverage improvements)** raised coverage on the flagged modules:
  - Added focused suites for rate limiting/middleware (`tests/unit/test_rate_limiter.py`, `tests/webui/test_rate_limit_middleware.py`, `tests/webui/test_csp_middleware.py`) and resource management (`tests/webui/services/test_resource_manager.py`).
  - Chunking endpoint tests (above) contribute to API coverage.

### Notes & Tips for Successor

- The Celery changes rely on `TESTING` being set before importing webui modules. `tests/conftest.py` already ensures this; keep that ordering if you add new fixtures.
- The SQLite fallback in `tests/unit/test_models.py` is intentionally scoped—if you add assertions that need additional tables, extend the `timezone_tables` list accordingly.
- Disk pressure can cause Coverage/pytest cache errors; if you encounter “No space left on device,” clear `.pytest_cache`, `.coverage`, `coverage.xml`, and `htmlcov/`.
- After making further changes, prefer targeted pytest invocations first (`uv run pytest tests/unit/some_module.py`) before running the full suite to keep iteration fast.
- Latest validation: `uv run pytest tests/webui/api/v2/test_chunking_direct.py tests/unit/test_rate_limiter.py tests/webui/test_rate_limit_middleware.py tests/webui/test_csp_middleware.py tests/webui/services/test_resource_manager.py -q`.
