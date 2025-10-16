# Semantik Production Readiness – Review Findings & Next Steps

## Overview
This document captures the major issues identified during the full‑stack review and the concrete actions required to move Semantik from its current pre‑release state toward production readiness. The findings span backend/vecpipe services, the FastAPI web UI, the React frontend, infrastructure, and CI/CD.

## 1. Critical Functional Bugs
- **Hybrid search contract mismatch**  
  - *Issue*: `SearchService.multi_collection_search` sends `hybrid_search_mode="weighted"` (and the React store exports `hybridMode: 'reciprocal_rank'`, `keywordMode: 'bm25'`). The shared contract (`packages/shared/contracts/search.py`) only allows `hybrid_mode` values `filter|rerank` and `keyword_mode` `any|all`. Requests currently fail with HTTP 422.  
  - *Fix*: Align backend parameters and frontend enums with the shared contract; add integration tests for semantic and hybrid flows.

- **Reranking ineffectiveness**  
  - *Issue*: Results are sorted by the original `score`, ignoring any `reranked_score` returned from vecpipe (`packages/webui/services/search_service.py`).  
  - *Fix*: Prefer `reranked_score` when available so cross-encoder improvements surface; add test coverage for reranked ordering.

## 2. Security Hardening
- **JWT secret enforcement is inconsistent**  
  - WebUI refuses to boot with the default secret, but `vecpipe` and the `worker` service still start with `CHANGE_THIS_TO_A_STRONG_PASSWORD`. Harden `docker-entrypoint.sh` so every service validates secrets before binding, and highlight secret provisioning in ops docs.
- **Secret storage & rotation**  
  - Generated `.env` values are persisted to disk and checked into local clones. Recommend adopting Docker/Swarm/Kubernetes secrets or cloud secret managers for production, plus guidance on rotation and auditing.

## 3. Reliability & Performance
- **Sequential DB access**  
  - `SearchService.validate_collection_access` fetches each UUID individually. Batch the lookup to reduce latency for multi-collection search.
- **Prometheus lifecycle**  
  - `start_metrics_server` runs unconditionally on each ASGI lifespan event in vecpipe, triggering `OSError` on reload. Guard the call so the Prometheus HTTP server starts only once.
- **Config side-effects**  
  - Importing `shared.config` auto-creates `data/` and `logs/`. This breaks read-only deployments; wrap directory creation with permission checks and meaningful errors.

## 4. Frontend Improvements
- Replace `window.__gpuMemoryError` and `window.__handleSelectSmallerModel` hacks in `SearchInterface.tsx` with store-driven or context-based state.  
- Remove console logging that ships in production bundles (`App.tsx`, `uiStore.ts`).  
- Drop the implicit `window.__navigate` contract in `api/v2/client.ts` and rely on router navigation APIs.

## 5. Testing & CI
- CI presently allows mypy and Safety to fail without failing the pipeline. Update `.github/workflows/main.yml` so type checks and dependency scans are blocking.  
- Add hybrid search contract tests (backend + Playwright/Vitest) to prevent another silent regression.

## 6. Deployment & Operations
- **GPU vs. CPU deployments**: The default Dockerfile and Compose reserve an NVIDIA GPU. Provide an alternate CPU profile (Dockerfile stage or compose override) and verify both paths.  
- **Documentation drift**: README still references Search API port 8001 while compose exposes 8000—correct the docs.  
- **Process management**: `start_webui.sh` kills anything on port 8080 via `kill -9`; replace with a safer PID check.  
- **Compose `deploy` blocks**: Current config uses Swarm-only directives. Either document the requirement or supply non-Swarm resource controls.

## 7. Additional Follow-ups
- Extend story around internal API key management (defaults, rotation, shared between services).  
- Confirm rate-limiting defaults are production-safe once `DISABLE_RATE_LIMITING` is removed from `.env`/test fixtures.  
- Assess logging/metrics ingestion (correlation IDs, Celery metrics) so observability is production-ready.

---

**Next actions:** Prioritize fixing the hybrid search contract and reranking sorter, enforce secrets everywhere, and ship CI+deployment improvements. Track each item as work items/PRs until the production checklist is satisfied.
