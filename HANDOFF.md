# HANDOFF

This document captures the current state of the Semantik codebase as of **16 October 2025**, the issues discovered during audit, and the concrete next actions required. It is intended for the developer taking over implementation. You have no prior exposure to this repository.

---

## 1. Project Snapshot
- **Monorepo Layout**
  - `packages/vecpipe`: primary search API (FastAPI) plus Qdrant integration, model management, embeddings.
  - `packages/webui`: user-facing FastAPI app, Celery tasks, and service layer.
  - `packages/shared`: shared configuration, repositories, metrics, embedding infrastructure.
  - `apps/webui-react`: React 19 SPA (Vite), Zustand state, React Query.
  - Ops tooling: Docker (CUDA-ready) + docker-compose stacks, Alembic migrations (`alembic/`), Makefile task runners, docs in `docs/`.
- **Dependencies**
  - Python 3.11–3.12 (Poetry-managed). GPUs via PyTorch/transformers/bitsandbytes. Qdrant client, SQLAlchemy async, Redis, Celery.
  - Front-end: React 19, Vite 7, Tailwind 3, React Query 5, Zustand 5, Vitest/Playwright for testing.
- **Testing State**
  - Python dev dependencies (pytest, etc.) not installed when we attempted `poetry run pytest tests/test_maintenance_service.py`. Install with `poetry install --with dev` before running backend tests.

---

## 2. Critical Issues Requiring Fixes

### 2.1 `ResourceManager` Is Broken (High Severity)
- **Location:** `packages/webui/services/resource_manager.py`
- **Problem**
  - Calls repository methods that do not exist (`collection_repo.list_by_user`, `collection_repo.get_by_id`, `operation_repo.list_by_user`).
  - Assumes repository returns `dict`, but SQLAlchemy returns model instances (so `.get(...)` fails).
  - Any code path that instantiates `ResourceManager` will raise immediately.
- **Action Items**
  1. Decide whether we actually need `ResourceManager` in the new architecture. If yes, implement the required repository APIs and return serializable DTOs. If no, remove/replace this service before it is wired into endpoints or tasks.
  2. Add unit tests for `ResourceManager` (or the replacement) covering quota checks, resource reservations, and error paths.

### 2.2 Missing `CollectionRepository.list_all` Used by Maintenance (Critical)
- **Locations:**  
  - `packages/webui/api/internal.py:25-44` → `collection_repo.list_all()`  
  - `packages/webui/tasks.py:478-505` (Celery cleanup)
- **Problem:** `CollectionRepository` (`packages/shared/database/repositories/collection_repository.py`) lacks `list_all`, so every call to those endpoints/tasks will raise `AttributeError`. That blocks internal cleanup and auditing.
- **Action Items**
  1. Implement `list_all(self) -> list[dict] | list[Collection]` returning every collection plus relevant Qdrant metadata (vector store name, staging info, status). Make sure to return serializable dicts if consumers expect key access.
  2. Update `_get_active_collections` and internal API to work with the returned type (attribute vs dict). Prefer a consistent DTO.
  3. Cover the new repository method with unit tests and add integration tests for the internal endpoint & the Celery cleanup routine.

### 2.3 Internal API Logs Secrets (High Severity)
- **Location:** `_configure_internal_api_key` in `packages/webui/main.py`
- **Problem:** When the internal API key is auto-generated (default `"change-me-in-production"`), the generated key is logged (`logger.warning(...)`). Anyone with log access can hijack internal endpoints (collection cleanup, reindex completion).
- **Action Items**
  1. Stop logging the actual key. Log only that a random key was generated or require explicit configuration in production.
  2. In dev, persist the generated key securely (e.g., to `data/.internal_api_key`) so restarts remain stable.
  3. Document the required env var in deployment docs; add a startup assertion that rejects default key in production.

### 2.4 Collection Metadata API Cannot Retrieve Records (High Severity)
- **Location:** `packages/shared/database/collection_metadata.py`
- **Problem:** `store_collection_metadata` upserts using random UUIDs as point IDs, while `get_collection_metadata` retrieves by `collection_name`. Retrieval always returns `None`, so we lose metadata (embedding model, quantization, etc).
- **Action Items**
  1. Change storage to use deterministic IDs (e.g., `id=collection_name`). Or, adjust retrieval to query via payload filter instead of ID.
  2. Add tests covering both storage and retrieval.
  3. Run a migration/cleanup task to restamp existing metadata entries if we switch IDs.

### 2.5 Qdrant Client Leak in Search Utilities (High Severity)
- **Location:** `packages/vecpipe/search_utils.py:12-45`
- **Problem:** `search_qdrant` instantiates an `AsyncQdrantClient` per call and never closes it. Under load this will exhaust HTTP connections/file descriptors.
- **Action Items**
  1. Memoize a client (async context manager) or pass a reused client from the caller. At minimum, call `await client.aclose()` in `finally`.
  2. Audit all other Qdrant interactions (vecpipe, maintenance) for similar leaks.
  3. Add load/integration tests to detect connection exhaustion.

### 2.6 Front-End API Mismatch with Back-End (High Severity)
- **Locations:** `apps/webui-react/src/services/api/v2/*`
- **Problems:**
  - `searchV2Api.multiSearch` calls `/api/v2/search/multi`, but server exposes only `/api/v2/search` and `/api/v2/search/single`.
  - `collectionsV2Api.removeSource` sends JSON body; server expects `source_path` query parameter.
  - Type definitions (`SearchRequest` enum values) don't align with server validation (hybrid modes, keyword modes).
  - `documentsV2Api.get` references `/documents/{id}` endpoint that is not implemented (only `/content` exists).
- **Action Items**
  1. Align client routes/params with actual FastAPI endpoints. Update hooking components accordingly.
  2. Harmonize `SearchRequest` types with backend contract (`packages/shared/contracts/search.py`).
  3. Remove or implement missing endpoints; ensure DocumentViewer fetches metadata using supported APIs.
  4. Add integration tests in Vitest/Playwright to cover key flows (multi-collection search, add/remove source).

---

## 3. Additional Observations & Tech Debt
- **Permission model**: Several TODOs remain (`CollectionRepository` / `OperationRepository` mention future checks against `CollectionPermission`). Confirm product requirements and implement before multitenant launch.
- **Resource calculations**: `ResourceManager.can_allocate` uses `psutil.disk_usage("/")`, ignoring configured data roots or docker volumes; consider using `settings.data_dir`.
- **SearchService HTTP clients**: `SearchService.search_single_collection` spins up a new `httpx.AsyncClient` per request. Investigate shared clients or connection pooling if latency/throughput suffers.
- **Status Logging**: In `_configure_internal_api_key` we log success with sensitive values; ensure logging sanitization across other settings (JWT secrets, DB passwords).
- **Testing**: Back-end tests require `poetry install --with dev`; front-end tests rely on local Playwright browsers (ensure `npx playwright install` run in CI).

---

## 3.5 Onboarding Pointers for New Developer

Orientation:
- Read `README.md` and `docs/ARCH.md` first to understand how `packages/vecpipe`, `packages/webui`, and `packages/shared` layer responsibilities are split.
- Follow up with `docs/SEARCH_SYSTEM.md` to see how a search request flows from the React app through FastAPI and into Qdrant.
- Skim `docs/TESTING.md` to learn which pytest and Vitest suites target the areas you will modify.
- Check the `Makefile` descriptions so you know which shortcuts (`make dev`, `make test`, etc.) stand up the full stack.

Environment & Data:
- Use `make dev` (wrapping `docker-compose.dev.yml`) to launch Postgres, Qdrant, and Redis; confirm connection strings match your overrides in `.env.example` or `.env.docker.example`.
- Keep Poetry on Python >=3.11,<3.13 (3.12 preferred but not strictly required) and run `poetry install --with dev` before touching backend code.
- Review `test_data/` and `test_documents/` to see example collections and documents that illustrate repository fields referenced in the fixes.
- When `_configure_internal_api_key` stops logging secrets, surface the generated key in dev via `data/.internal_api_key` (or set `INTERNAL_API_KEY`) and share it with the React app through `apps/webui-react/.env.example`.

Backend & Frontend Touchpoints:
- When implementing `CollectionRepository.list_all`, return dicts containing at least `id`, `name`, `vector_store_name`, `qdrant_collections`, `qdrant_staging`, `document_count`, `vector_count`, and `total_size_bytes` because both `packages/webui/api/internal.py` and `packages/webui/tasks.py` access those keys.
- Inspect `packages/webui/services/factory.py` to see where `ResourceManager` is wired; this helps decide whether to refactor or retire it without breaking dependency injection.
- Note that `make dev` exposes Qdrant at the host/port defined in `docker-compose.dev.yml`; lean on the async context manager support in `qdrant_client.AsyncQdrantClient` so new pooling logic fits the library’s patterns.
- Trace who consumes the v2 REST clients by starting at `apps/webui-react/src/stores` (e.g., `collectionStore.ts`) so request/response shape changes stay in sync with UI state.
- For document APIs, remember only `/api/v2/collections/{uuid}/documents/{doc_uuid}/content` exists today; the TypeScript `documentsV2Api.get` helper will need either a new endpoint or removal.
- Keep an eye on shared contracts under `packages/shared/contracts/search.py` when adjusting the TS `SearchRequest` enums so both sides stay aligned.

Validation Workflow:
- After backend work, run `poetry run pytest` (or targeted modules) to ensure repository and service tests stay green.
- For the web UI, execute `npm run lint` and `npm run test:coverage`, then follow with `npm run test:e2e` once Playwright browsers are installed.
- Perform a manual smoke test: create a collection, add a source, run a search (single and multi), and trigger the cleanup path to ensure the Qdrant list stays accurate.
- Capture any required migrations or data backfills (e.g., collection metadata ID changes) in either Alembic scripts or documented operational steps before shipping.

## 4. Implementation Plan (Prioritized)

### Phase 0 – Environment Prep
1. Install Python dev deps: `poetry install --with dev`.
2. Ensure Poetry uses Python 3.12 (project does not support 3.13 yet).
3. For front-end work: `cd apps/webui-react && npm install`.

### Phase 1 – Repository & Service Fixes
1. ✅ Completed October 16, 2025 — `CollectionRepository.list_all()` now returns normalized dicts with vector metadata, status, and counts (`packages/shared/database/repositories/collection_repository.py`).
2. ✅ Completed October 16, 2025 — Internal API and `_get_active_collections()` consume normalized dicts; added/updated tests covering the vector-store endpoint (`packages/webui/api/internal.py`, `tests/test_internal_api.py`).
3. ✅ Completed October 16, 2025 — Added repository support for `list_by_user`/`get_by_id` and `OperationRepository.list_by_user` to keep `ResourceManager` functional; repository unit tests updated (`packages/shared/database/repositories/collection_repository.py`, `packages/shared/database/repositories/operation_repository.py`, `tests/unit/test_collection_repository.py`, `tests/unit/test_operation_repository.py`).

### Phase 2 – Security Hardening
1. ✅ Completed October 16, 2025 — `_configure_internal_api_key` now avoids logging secrets, persists a dev key at `data/.internal_api_key`, and enforces production configuration (`packages/webui/main.py`).
2. ✅ Completed October 16, 2025 — Documented production requirement for `INTERNAL_API_KEY` and surfaced dev key behaviour in README/docs (`README.md`, `docs/CONFIGURATION.md`).

### Phase 3 – Qdrant Metadata & Client Hygiene
1. ✅ Completed October 16, 2025 — `store_collection_metadata` now uses deterministic IDs and `get_collection_metadata` falls back to legacy entries; added unit coverage (`packages/shared/database/collection_metadata.py`, `tests/unit/test_collection_metadata.py`).
2. ✅ Completed October 16, 2025 — Closed Qdrant clients in search utilities and admin reset workflow with regression tests (`packages/vecpipe/search_utils.py`, `packages/webui/api/settings.py`, `tests/unit/test_search_utils.py`, `tests/webui/test_settings_api_reset.py`, integration test updates).
3. ✅ Completed October 16, 2025 — Added restamp helper `restamp_collection_metadata` and accompanying tests to support high-load migrations of metadata IDs.

### Phase 4 – Front-End / Back-End Contract Alignment
1. Update `apps/webui-react/src/services/api/v2/collections.ts` and `types.ts` to match FastAPI routes and payloads.
2. Decide whether to expose new endpoints (`/search/multi`, `/collections/{id}/documents`) or adjust the client to existing ones.
3. Build integration tests (Vitest/Playwright) covering multi-collection search, add/remove source, document viewing.

### Phase 5 – Regression Tests & Validation
1. Run backend tests: `poetry run pytest`.
2. Run web UI tests: `npm run lint`, `npm run test:coverage`, `npm run test:e2e` (requires Playwright).
3. If possible, run docker-compose stack to smoke test flows (collection creation, indexing, search, cleanup).

---

## 5. Reference Pointers
- **Key Files**
  - `packages/shared/database/repositories/collection_repository.py`
  - `packages/webui/services/resource_manager.py`
  - `packages/webui/api/internal.py`
  - `packages/webui/tasks.py` (cleanup + audit helpers)
  - `packages/shared/database/collection_metadata.py`
  - `packages/vecpipe/search_utils.py`
  - `apps/webui-react/src/services/api/v2/*.ts`
  - `packages/shared/contracts/search.py`
- **Docs to Review**
  - `docs/ARCH.md`, `docs/SEARCH_SYSTEM.md`, `docs/TESTING.md`, `docs/CONFIGURATION.md`, `docs/DOCKER.md`.
  - Makefile targets (`make wizard`, `make docker-up`, `make test`, etc.).

---

## 6. Closing Notes
- Treat GPU/quantization paths carefully (bitsandbytes, adaptive batching) to avoid regressions.
- Internal APIs (`/api/internal`) are used by maintenance scripts—ensure authentication changes are mirrored wherever the key is consumed.
- Keep logs free of secrets and ensure default credentials are replaced in environment configs (`docker-compose.yml` still exposes placeholders).

Reach out to the repo owner if product/business context is needed for permissions or quotas; otherwise the above plan should unblock critical fixes. Good luck!
