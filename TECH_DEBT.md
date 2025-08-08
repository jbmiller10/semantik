# Technical Debt Audit

This document tracks potential technical debt identified in the repository. Items are organized by category with file references and suggested actions.

Last updated: $(date -u +"%Y-%m-%d %H:%M:%S UTC")

## Marker Scan (TODO/FIXME/HACK/XXX/BUG)

The following annotations were found in the codebase (excluding vendor directories like node_modules, .git, and build artifacts). Each entry is `file:line` with the first snippet of the note.

### Curated markers (source code)

- `packages/shared/database/repositories/collection_repository.py:172`:            # TODO: Check CollectionPermission table for shared access
- `packages/shared/database/repositories/operation_repository.py:68`:                # TODO: Check CollectionPermission table for write access
- `packages/shared/database/repositories/operation_repository.py:145`:            # TODO: Check CollectionPermission table for access
- `packages/shared/database/repositories/operation_repository.py:278`:                # TODO: Check CollectionPermission table
- `packages/webui/api/settings.py:130`:    db_size = 0  # TODO: Implement PostgreSQL database size query
- `packages/webui/api/v2/partition_monitoring.py:16`:# TODO: Implement proper admin/API key authentication for partition monitoring endpoints
- `packages/webui/api/v2/partition_monitoring.py:24`:    # TODO: Add authentication once require_api_key is implemented
- `packages/webui/chunking_tasks.py:179`:            # TODO: Fix Redis client type mismatch
- `packages/webui/chunking_tasks.py:684`:                    # TODO: Implement actual chunking logic
- `packages/webui/services/resource_manager.py:50`:            # TODO: Get user's collection limit from user settings/subscription
- `packages/webui/services/resource_manager.py:85`:            # TODO: Get user's resource limits from settings/subscription
- `packages/webui/services/resource_manager.py:203`:            # TODO: Query actual Qdrant usage
- `apps/webui-react/src/components/DocumentViewer.tsx:187`:              // TODO: Implement PDF.js rendering
- `apps/webui-react/src/components/ActiveOperationsTab.tsx:39`:    // TODO: Implement proper navigation using React Router
- `apps/webui-react/src/stores/chunkingStore.ts:183`:          // TODO: Replace with actual API call
- `apps/webui-react/src/stores/chunkingStore.ts:291`:          // TODO: Replace with actual API call
- `apps/webui-react/src/stores/chunkingStore.ts:358`:          // TODO: Replace with actual API call


## High-Priority Issues
- Unauthenticated admin endpoints: `packages/webui/api/v2/partition_monitoring.py` has TODOs to add authentication for partition monitoring endpoints. Action: enforce JWT/API key on all routes.
- Placeholder logic in chunking: `packages/webui/chunking_tasks.py` contains TODOs including “Implement actual chunking logic” and a Redis client type mismatch. Action: implement production-safe chunking and fix Redis client typing.

## Security & Compliance
- `.env` checked into workspace: A `.env` file exists at repo root (placeholders), while `.gitignore` excludes it. Action: ensure it’s not tracked in git; rely on `.env.example` and secrets in CI/CD.
- CORS warnings: `packages/webui/main.py` allows wildcard/null origins in non-prod and warns for prod. Action: validate prod config and consider rejecting insecure origins by default in prod.
- JWT key generation in dev: `_configure_internal_api_key()` generates a random key when default is present. Action: require explicit key in non-dev environments, add startup failure if unset.

## Config & Build Hygiene
- Pinned vs latest images: `docker-compose.yml` uses `qdrant/qdrant:latest` while `docker-compose.prod.yml` pins `v1.7.4`. Action: pin in all profiles for reproducibility.
- Multiple compose files: `docker-compose.yml`, `docker-compose.dev.yml`, `docker-compose.prod.yml`, `docker-compose.cuda.yml`. Action: refactor to a base file + minimal overrides; remove duplication of env vars.
- Heavy/unpinned Python deps: `pyproject.toml` has `torch = "*"` and `accelerate >=0.26.0`. Action: pin compatible versions and/or add a `constraints.txt` for CUDA variants.
- Frontend build artifacts: `packages/webui/static/*` and `.map` files exist locally and are ignored by git. Action: keep them untracked; ensure CI builds them; consider excluding `.map` if size is a concern.
- Node workspaces: Root `package.json` defines workspaces; `node_modules/` present locally. Action: verify `node_modules` is not committed and lockfiles are consistent; prefer `npm ci` in CI.
 - Note: `DATABASE_URL` env var lines in compose are valid nested parameter expansions (no extra brace). No change needed.

## Database Migrations
- Inconsistent Alembic version filenames: `alembic/versions/add_chunking_strategy_columns.py` uses a non-hash filename with revision id `add_chunking_strategy_cols`, whereas others use hash/timestamp prefixes. Action: normalize naming to avoid confusion; ensure a single linear head.
- Migration chain recency: Verify linear history: 005a8fe3aedc → 20250727151108 → 52db15bd2686 → 6596eda04faa → 8f67aa430c5d → a1b2c3d4e5f6 → add_chunking_strategy_cols. Action: run `alembic heads`/`history` and document; consider squashing pre-release migrations.

## Code Quality Hotspots
- Silent exception swallowing:
  - `packages/shared/metrics/collection_metrics.py` `record_metric_safe()` silently `pass`es on errors. Action: log at debug once per metric name to aid troubleshooting.
  - Several API health checks (`packages/webui/api/health.py`) use `pass` for embedding health. Action: clarify docs or log rationale.
- Stubbed/placeholder implementations:
  - `packages/webui/api/settings.py`: `db_size = 0  # TODO` — implement PostgreSQL size query.
  - `packages/webui/services/resource_manager.py`: multiple TODOs to fetch limits from user/subscription and to query actual Qdrant usage.
  - Frontend TODOs: PDF rendering (`DocumentViewer.tsx`), proper navigation (`ActiveOperationsTab.tsx`), and mocked API calls in `stores/chunkingStore.ts`.

## Repo Hygiene & Organization
- `archive/` directory: Contains dev logs and notes with many TODOs. Action: move relevant design notes to `docs/` or remove the folder from the main repo.
- Utility scripts sprawl: `scripts/` contains many one-off utilities (benchmarking, manual embed, maintenance). Action: move prod-irrelevant scripts to `tools/` or `examples/`, add READMEs, and mark unsupported scripts.
- Stray reports: `test_health_report.md` and `TEST_REPORT.md` look like generated artifacts. Action: either delete or add patterns to `.gitignore` (e.g., `test_*report.md`).

## Pre‑Release Simplifications (drop legacy shims)
- Rationale: The project is pre-release; maintaining backwards compatibility slows progress. Remove legacy pathways and compatibility layers.
- Examples to review and remove:
  - `packages/webui/api/v2/__init__.py`: mentions legacy operation-centric endpoints; ensure only v2 style remains.
  - Frontend legacy transforms:
    - `apps/webui-react/src/hooks/useDirectoryScan.ts`: cleaned to use v2 response shape directly.
    - `apps/webui-react/src/hooks/useDirectoryScanWebSocket.ts`: cleaned; warnings now strings; preserves files from preview.
    - `apps/webui-react/src/components/ReindexCollectionModal.tsx`: removed legacy parameter handling; strategy-first.
  - Temporary/compat code in `apps/webui-react/src/hooks/useCollections.ts` (temporary collection handoff) if only needed for old flows.
- Action: remove legacy conversions, align payloads to the v2 API, delete unused code paths, and update tests accordingly.

## Completed Cleanups (tracking)
- Secured admin-only partition monitoring endpoints with JWT checks.
- Centralized password hashing to `shared.database.pwd_context` and removed duplicate in `webui/auth.py`.
- Normalized Alembic migration filename: `alembic/versions/add_chunking_strategy_cols_add_chunking_strategy_columns.py` (content/revision unchanged).

## Dependency Signals (possible unused)
- Direct imports found:
  - PyTorch used in `packages/shared/embedding/*` and `packages/webui/api/v2/system.py`.
  - Celery used in `packages/webui/celery_app.py` and `packages/webui/chunking_tasks.py`.
  - `bitsandbytes` referenced in `dense.py` (optional).
- Suggested next: run static import scans vs declared deps to flag likely-unused packages (offline), or enable runtime usage telemetry in dev.

## CI/CD Observations
- CI uses `qdrant:latest` for tests. Action: pin to a stable version to reduce flakiness.
- Mypy marked `continue-on-error: true`. Action: create a small ignore baseline or fix high-value type errors, then fail CI to prevent regressions.

- P0 – protect partition monitoring endpoints.
- P1 – security hardening: require explicit JWT secret in non-dev; lock down CORS in prod; ensure `.env` is untracked.
- P1 – reproducibility: pin `torch`/`accelerate`; pin Qdrant version in CI and non-prod compose; consider `constraints.txt`.
- P2 – API completeness: implement DB size query; resource limit/usage reads; resolve Redis typing and finalize chunking logic.
- P2 – cleanup: archive/move `archive/`; relocate utility scripts; remove or ignore stray test reports.
- P3 – consistency: normalize Alembic version filenames; password hashing now centralized via shared `pwd_context`.
