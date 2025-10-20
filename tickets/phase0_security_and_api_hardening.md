# Phase 0 Ticket: Lock Down Security & Critical API Reliability (Target: October 24, 2025)

## Background
Semantik's audit on October 18, 2025 (see `CODEBASE_REVIEW_REPORT.md`) surfaced eight critical issues that block the project from being shared as a portfolio piece. The most severe items live in the backend stack (`packages/webui`, shared config, infra scripts) and revolve around missing admin protections, default credentials, and API endpoints returning incorrect status codes. These flaws are visible during even a casual demo, so we must close them before any further marketing or feature work.

The codebase uses FastAPI with SQLAlchemy, async services, and a Docker-driven deployment pipeline. Secret management and runtime validation live in `docker-entrypoint.sh` and the Makefile wizard. Tests run via `uv run pytest`, and we lean on Alembic-managed Postgres.

## Objectives
1. Enforce proper authorization for partition monitoring endpoints.
2. Block default secrets (JWT, Postgres, Flower) from ever starting services.
3. Replace placeholder database size reporting with a real metric.
4. Ensure access control errors surface as HTTP 403 instead of 500.
5. Prevent document path traversal on download endpoints.
6. Provide documentation/backfill for any new environment expectations.

## Requirements & Deliverables
### 1. Partition Monitoring Admin Auth
- Update `packages/webui/api/v2/partition_monitoring.py` to require admin privileges or an internal API key dependency (e.g., existing `require_admin` hook from the auth layer).
- Add automated coverage to ensure non-admin users receive 403 (`tests/webui/api/v2/test_partition_monitoring.py` or similar).

### 2. Default Secret Enforcement Across Services
- Extend `docker-entrypoint.sh` so **every** service (webui, vecpipe, worker, flower) exits early if `JWT_SECRET_KEY` or `INTERNAL_API_KEY` match known placeholders (see `.env.docker.example`).
- Guard against `POSTGRES_PASSWORD=CHANGE_THIS_TO_A_STRONG_PASSWORD` during startup. Provide a clear error message guiding users to rerun `make docker-up` or regenerate secrets via `wizard.sh`.
- Ensure Flower credentials are provisioned via `FLOWER_USERNAME`/`FLOWER_PASSWORD` and validation fails for placeholders (e.g., `admin`). Update the wizard/docker configs to generate rotate-able values and document the `make wizard` regeneration path.
- Add `.env.example` (root) and `packages/webui/.env.example` showing compliant values and linking to the Makefile wizard.

### 3. Accurate Database Size Metric
- Implement `SELECT pg_database_size(current_database())` in `packages/webui/api/settings.py` so `/api/v1/settings` returns the true database size in bytes.
- Provide an integration test hitting the settings endpoint and asserting the field is non-zero when Postgres is seeded.

### 4. AccessDeniedError → HTTP 403
- Add a FastAPI exception handler module (e.g., `packages/webui/middleware/exception_handler.py`) that maps `AccessDeniedError` to status 403 without exposing stack traces.
- Register the handler in the FastAPI app startup (`packages/webui/main.py`).
- Remove `pytest.mark.xfail` decorators from the affected tests (`tests/webui/api/v2/test_collections*.py`) and make them pass.

### 5. Document Path Traversal Safeguard
- In `packages/webui/api/v2/documents.py`, resolve requested file paths and ensure they remain under the configured document root. Reject suspicious paths with 403 and log the attempt.
- Add unit/integration tests covering `../../etc/passwd`–style inputs.

### 6. Operational Documentation
- Update relevant docs (`docs/security.md` or new section in `README.md`) describing the new secret validation flow and how developers should provision credentials locally.
- Mention the new admin requirement for partition monitoring in the API reference (`docs/api.md` if present).

## Acceptance Criteria
- All critical issues from CODEBASE_REVIEW_REPORT.md Section "Critical Issues" are resolved and marked verified.
- Running `JWT_SECRET_KEY=CHANGE_THIS_TO_A_STRONG_SECRET_KEY docker compose up webui` fails with an actionable error message.
- Both `/api/v2/admin/partitions` and `/api/v1/settings` endpoints pass new automated tests (403 for non-admin, and `db_size > 0`).
- `uv run pytest tests/webui/api/v2 -k "collections or partition"` succeeds without xfails.
- Static analysis and formatting remain clean (`make format`, `make lint`, `make type-check`).
- Security documentation references regenerated secrets and the partition admin guard.

## Testing & Validation
1. `uv run pytest tests/webui/api/v2 -k "collections or partition" -v`
2. `uv run pytest tests/webui/api/v1/test_settings.py::test_database_size_returns_positive`
3. Manual curl check for document traversal: attempt to download `../../etc/passwd` and confirm 403.
4. `docker compose up flower` with missing custom creds should exit with a descriptive error; with generated creds, dashboard should require login.

## Dependencies & Notes
- Ensure no existing deployment scripts rely on the old default credentials; coordinate with infra maintainers if CI pipelines need updated env vars.
- If large docs edits are required, coordinate with the documentation ticket in Phase 3 to avoid conflicts.
- Keep commits granular so security reviewers can audit each guard individually.

## Out of Scope
- Any new authentication flows (e.g., SSO, OAuth providers) beyond enforcing existing admin/API-key checks.
- Broader infrastructure hardening such as rate limiting, WAF configuration, or DDoS mitigation.
- Performance tuning of partition monitoring or settings endpoints beyond basic correctness.
- Cleanup of non-critical warnings flagged in the audit (e.g., medium/low items) unless they block the acceptance criteria above.
