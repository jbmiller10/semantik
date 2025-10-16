# Semantik Production Readiness – Implementation Guide

> **Audience:** Engineers who have no prior exposure to the Semantik codebase.  
> **Goal:** Provide enough background, rationale, and step-by-step guidance to resolve the issues uncovered during the repository review and bring the product to production quality.

Each section below links to concrete file locations (relative to the repository root), describes the underlying problem, explains how to reproduce it, and outlines the remediation steps and validation strategy.

---

## 1. Hybrid Search Contract Breakage

**Summary**  
Hybrid search requests currently fail with HTTP 422 (Unprocessable Entity). The backend builds a payload using the wrong field names and enum values, and the frontend exports matching invalid values. The shared Pydantic contract rejects these requests before they reach vecpipe, so hybrid mode is effectively unusable.

**Primary locations**
- Backend contract: `packages/shared/contracts/search.py` – `SearchRequest.hybrid_mode` (`filter|rerank`) and `keyword_mode` (`any|all`)
- Backend service: `packages/webui/services/search_service.py` lines 200–376 (see `.update()` calls for hybrid parameters)
- Frontend store: `apps/webui-react/src/stores/searchStore.ts` lines 23–90 (default state & validation)
- Frontend API types: `apps/webui-react/src/services/api/v2/types.ts` lines 47–103
- Frontend request call: `apps/webui-react/src/components/SearchInterface.tsx` lines ~117–189

**Reproduction steps**
1. Bring up the stack (`make dev` or `docker compose up`).
2. Call `POST /api/v2/search` with `"search_type": "hybrid"` (can use the UI or `curl`):
   ```bash
   curl -X POST http://localhost:8080/api/v2/search \
        -H "Authorization: Bearer <token>" \
        -H "Content-Type: application/json" \
        -d '{"collection_uuids":["<uuid>"],"query":"test","search_type":"hybrid","hybrid_mode":"weighted"}'
   ```
3. FastAPI responds with 422 because `SearchRequest.hybrid_mode` only allows `"filter"` or `"rerank"`.

**Remediation plan**
1. **Backend**  
   - Update `search_service.py` so multi-collection and single-collection branches send `hybrid_mode` (rename the key) and pass only the supported enum values. If you need a “weighted” behavior, map it internally to `rerank` or extend the contract in `packages/shared/contracts/search.py` (but keep backend + frontend + vecpipe in sync).  
   - When submitting to vecpipe (`httpx.AsyncClient.post`), ensure `keyword_mode` matches the contract (`any|all`). Remove or translate references to `"bm25"` if that synonym is desired.
2. **Frontend**  
   - Align the `SearchParams` type and defaults with the contract (`hybridMode: 'filter' | 'rerank'`, `keywordMode: 'any' | 'all'`).  
   - Update validation utilities in `apps/webui-react/src/utils/searchValidation.ts` to enforce the corrected enums.  
   - Ensure UI controls (e.g., `RerankingConfiguration`, hybrid search toggles) reflect the available options.
3. **Tests**  
   - Add backend integration coverage that posts a hybrid request through `/api/v2/search` and asserts a 200 response (use the existing test harness in `tests/webui/services/test_search_service.py` or create a new test under `tests/webui/api/v2`).  
   - Add a Vitest test that fires the `searchV2Api.search` call with hybrid parameters and asserts the payload matches expectations.  
   - If vecpipe supports hybrid mode with additional options, ensure `packages/vecpipe/search_api.py` accepts the same schema before changing the shared contract.

**Validation**
- Re-run the reproduction curl; expect 200 OK and a result payload.
- Exercise the React UI: select “Hybrid” mode and submit a search—no validation errors should appear, and the response list should populate.
- Run the relevant suites: `uv run pytest tests/webui -k hybrid` and `npm run test -- --run TestsForSearch`.

---

## 2. Rerank Ordering Bug

**Summary**  
Even when vecpipe returns `reranked_score` values from the cross-encoder, the backend sorts results using the original vector `score`. Users cannot tell that reranking is active, and quality degrades.

**Primary locations**
- Sorting logic: `packages/webui/services/search_service.py` lines 252–308
- Reranking integration tests: `tests/webui/services/test_search_service_reranking.py`

**Reproduction steps**
1. Enable reranking in a request (`use_reranker: true`).
2. Inspect the response array—`reranked_score` differs from `score`, yet results remain in descending `score` order.

**Remediation plan**
1. Change the sort key to prefer `reranked_score` when present:
   ```python
   all_results.sort(
       key=lambda item: item.get("reranked_score", item.get("score", 0.0)),
       reverse=True,
   )
   ```
2. When constructing `SearchResult`, propagate both scores so clients can display them (`packages/webui/api/v2/search.py` already maps `reranked_score` in the response model).

**Validation**
- Extend `tests/webui/services/test_search_service_reranking.py` to assert the sorted order matches descending `reranked_score`.
- Manual UI check: run a search with reranking enabled and verify the top result changes when reranking is toggled off/on.

---

## 3. Secret Management Gaps

### 3.1 JWT Secret Enforcement Across Services
**Problem**  
`packages/webui/main.py` enforces non-default `JWT_SECRET_KEY`, but `vecpipe` and the Celery worker accept the placeholder value defined in Compose (`CHANGE_THIS_TO_A_STRONG_PASSWORD`). Attackers could forge tokens if any service is misconfigured.

**Files involved**
- Entry point script: `docker-entrypoint.sh`
- Compose files: `docker-compose.yml`, `docker-compose.prod.yml`, `docker-compose.cuda.yml`
- Config class: `packages/shared/config/webui.py` (`WebuiConfig.__init__`)

**Actions**
1. In `docker-entrypoint.sh`, extend `validate_env_vars` to check `JWT_SECRET_KEY` for all services (`vecpipe`, `worker`, `flower`). Exit with a clear error message if the default is detected.
2. Update Compose examples to encourage using Docker secrets or environment overrides. Add documentation in `README.md` (Deployment section) instructing operators to generate strong keys (e.g., `openssl rand -hex 32`) and mount them via secrets.

### 3.2 Persistent Secrets on Disk
**Problem**  
`WebuiConfig` persists generated secrets to `/app/data/.jwt_secret`. In production, we should rely on ephemeral secret stores instead of files on shared volumes.

**Action items**
- Document recommended secret providers (e.g., AWS Secrets Manager, Hashicorp Vault, Kubernetes Secrets) and update `README.md` / `docs/DEPLOYMENT.md`.  
- Optionally, allow disabling file persistence via an environment flag (`JWT_PERSIST_SECRET=false`) and adjust `WebuiConfig` accordingly.

**Validation**
- Start containers with missing `JWT_SECRET_KEY` → entrypoint should abort.
- Provide a proper key via secrets/env var → service should boot without creating a file unless explicitly allowed.

---

## 4. Backend Reliability Fixes

### 4.1 Batch Collection Access Checks
**Issue**  
`SearchService.validate_collection_access` performs sequential repository calls inside a loop, creating N round trips and potential inconsistent authorization errors.

**Files**
- `packages/webui/services/search_service.py` lines 210–233
- `packages/shared/database/repositories/collection_repository.py` (add helper method)

**Steps**
1. Add a repository method (e.g., `CollectionRepository.get_many_with_permission_check(user_id, collection_ids: list[str])`) that fetches all target collections in one query using `IN` and joins as needed.  
2. Replace the loop in `validate_collection_access` with a call to the new method and raise `AccessDeniedError` if the returned list length is shorter than requested.

**Testing**
- Unit test: ensure the repository method returns only allowed collections and raises for missing IDs.  
- Integration test: run `uv run pytest tests/webui/services -k multi_collection`.

### 4.2 Prometheus Metrics Server Re-entry
**Issue**  
`packages/vecpipe/search_api.py` calls `start_metrics_server` on each FastAPI lifespan event. During reloads (e.g., `uvicorn --reload`), Python raises `OSError: Address already in use`.

**Fix**
- Add module-level guard: define `metrics_server_started = False`. On startup, check the flag before calling `start_metrics_server`; set it to `True` after successful start.  
- Alternatively, catch `OSError` and log a warning that the server is already running.

**Validation**
- Run `uvicorn vecpipe.search_api:app --reload` locally; there should be no repeated stack traces after code changes.

### 4.3 Configuration Side Effects
**Problem**  
`BaseConfig.__init__` unconditionally creates directories, which fails under read-only deployments or AWS Lambda layers.

**Solution**
- Wrap `mkdir` calls in a `try/except PermissionError`, log a friendly message instructing the operator to pre-create directories, and re-raise with context.
- Consider adding a config flag (e.g., `AUTO_CREATE_DIRECTORIES`) defaulting to `True` for local dev but disableable in production.

**Validation**
- Deploy in an environment with read-only `/app`; import `shared.config` should now yield a descriptive error rather than a raw traceback.

---

## 5. Frontend Clean-up & Reliability

### 5.1 Global Window State
**Problem**  
`SearchInterface.tsx` stores GPU error information and reranking callbacks on `window.__gpuMemoryError` and `window.__handleSelectSmallerModel`. This breaks SSR, confuses tests, and risks naming collisions.

**Remediation**
- Store GPU error state in the existing Zustand store (`useSearchStore`). Add fields like `gpuMemoryError: { message: string; suggestion: string; currentModel: string } | null`.  
- Replace the global function with a store action (e.g., `useSearchStore.getState().handleSelectSmallerModel(...)`).  
- Update `SearchResults` (or any component reading from `window`) to consume the store via hooks.

### 5.2 Console Logging in Production
**Locations**
- `apps/webui-react/src/App.tsx` line 24 (`console.log('App component rendering')`)
- `apps/webui-react/src/stores/uiStore.ts` lines 46–48 (`console.log` in setter)

**Action**  
Remove these logs or gate them behind `if (import.meta.env.DEV)` checks.

### 5.3 Axios Navigation Hack
**Problem**  
`api/v2/client.ts` attempts to redirect users by writing `window.__navigate`. This tightly couples tests and runtime behavior.

**Solution**
- Use the established auth store to emit a logout event and handle navigation inside React components:
  - After detecting a 401, call `useAuthStore.getState().logout()` and publish an event through a shared observable or event emitter.
  - In `App.tsx`, subscribe to that event and call `navigate('/login')` using React Router hooks.

**Testing**
- Update MSW/Vitest tests to assert that the store clears tokens on 401 responses.
- Run `npm run test:ci` (Vitest) and UI smoke tests.

---

## 6. CI/CD & Testing Improvements

**Current behavior**  
`.github/workflows/main.yml` marks mypy as `continue-on-error: true` and the Safety scan as non-blocking. This allows type regressions and dependency vulnerabilities into main.

**Required changes**
1. Remove `continue-on-error` in the “Python Linting” job so mypy failures block the pipeline.
2. If Safety must remain non-blocking, capture its output and explicitly warn, but document a plan for remediation.
3. Add a new job (or extend existing backend matrix) that runs a hybrid search integration test to prevent regression.

**Validation**
- Push an intentionally failing mypy change to a feature branch; confirm GitHub Actions fails.
- Ensure the pipeline still completes within acceptable time by caching `uv` and Node dependencies (already configured).

---

## 7. Deployment & Operations

### 7.1 GPU vs. CPU Builds
**Problem**  
The primary Dockerfile inherits from `nvidia/cuda` and the Compose file reserves GPU resources. Users without NVIDIA GPUs cannot run the stack.

**Steps**
1. Add a CPU-only Docker stage (e.g., `FROM python:3.11-slim AS runtime-cpu`) that skips CUDA libraries and bitsandbytes.  
2. Provide `docker-compose.cpu.yml` that references `runtime-cpu`, omits GPU reservations, and sets `USE_MOCK_EMBEDDINGS=true` by default.
3. Document selection logic in `README.md` (e.g., “Use `docker compose -f docker-compose.yml -f docker-compose.cpu.yml up` on CPU-only hosts”).

### 7.2 Documentation Drift
**Problem**  
`README.md` references Search API port 8001, but `docker-compose.yml` exposes 8000 (`vecpipe` service). This confuses operators.

**Fix**  
- Update all docs (`README.md`, `docs/*`) to reflect the correct port and note that the web UI proxies to vecpipe internally.

### 7.3 `start_webui.sh` Process Kill
**Problem**  
The script runs `kill -9 $(lsof -ti:8080)`, potentially terminating unrelated services.

**Fix**  
- Replace with a targeted process lookup:
  ```bash
  pid=$(lsof -t -sTCP:LISTEN -i :8080 | head -n1)
  if [ -n "$pid" ]; then
      echo "Stopping process on 8080 (PID $pid)"
      kill "$pid"
      sleep 2
  fi
  ```
  Use `pkill -f "uvicorn webui.main:app"` if you’re certain of the process name.

### 7.4 Compose `deploy` Blocks
**Problem**  
`docker-compose.yml` uses the `deploy:` stanza (intended for Swarm). Docker Compose ignores it, so resource constraints are unenforced.

**Options**
1. Remove `deploy` sections and rely on documentation to instruct operators about resource limits; or

---

## 8. Additional Follow-ups
- **Internal API key management**  
  - Evaluate whether `ensure_internal_api_key` should run for every service. Document how to rotate the key without downtime.
- **Rate limiting readiness**  
  - Many tests set `DISABLE_RATE_LIMITING=true`. Provide production defaults in `.env.example` and ensure deployment docs emphasize enabling rate limits.
- **Observability**  
  - Confirm correlation IDs propagate to Celery workers and Redis logs.  
  - Verify that Grafana/Prometheus dashboards (if any) cover vecpipe, webui, and worker metrics.

---

## Validation Checklist After All Fixes
1. **Automated tests**  
   - `uv run pytest` (full suite)  
   - `npm run test:ci` and `npm run test:coverage -- --run`
2. **Manual smoke tests**  
   - Authentication flow (register → login → protected routes)  
   - Semantic search, hybrid search with reranking enabled/disabled  
   - Document ingestion (collection creation, background operations)  
   - WebSocket updates for operations  
   - Frontend logout on expired token
3. **Deployment checks**  
   - Run CUDA-enabled compose on a GPU host and ensure metrics server, Celery worker, and vecpipe boot.  
   - Run CPU-only compose on a machine without NVIDIA drivers.  
   - Verify secrets are injected correctly (no defaults, no plain-text exposures).
4. **Documentation**  
   - Ensure README quick start, deployment instructions, and troubleshooting sections reflect the new behaviors and scripts.

---

By following the instructions in this guide, a new contributor should be able to address the identified issues methodically and prepare Semantik for a production-grade release. Keep the document updated as fixes land to provide an accurate implementation log. 
