# Repository Guidelines

## Project Structure & Module Organization
Semantik is a polyglot monorepo. Python services live in `packages/`: `vecpipe` powers ingestion/search workers, `webui` exposes the FastAPI backend, and `shared` holds contracts, config, and utilities shared across services. The React frontend sits in `apps/webui-react` (Vite project with assets in `public/`). Database migrations live under `alembic/`. Shell automation and orchestration scripts are in `scripts/`. All automated tests reside in `tests/` with domain-specific folders (`tests/domain`, `tests/performance`, `tests/e2e`), and sample documents live in `data/`.

## Build, Test, and Development Commands
Run `make dev-install` for a full `uv` sync of Python dependencies and tooling; use `npm install --prefix apps/webui-react` (or `make frontend-install`) for the UI. `make dev` starts the integrated stack via `scripts/dev.sh`. For API-only hot reload, use `make run`. Frontend iterations happen through `make frontend-dev`. Server-side builds rely on `make build`; React production bundles come from `npm run build --prefix apps/webui-react`. Docker workflows (`make docker-up`, `make docker-down`, `make docker-dev-up`) provision services and secrets automatically.

`make docker-down` now mirrors `docker compose down` and keeps named volumes; run `make docker-down-clean` when you explicitly want to blow away volumes (e.g., to reset Postgres data).

## Coding Style & Naming Conventions
Python code follows 4-space indentation, 120-character lines, and exhaustive typing. Before pushing, run `make format` (Black + Isort) and `make lint` (Ruff) followed by `make type-check` (Mypy). Modules stay snake_case (`search_api.py`), pytest fixtures snake_case, and React components PascalCase inside `apps/webui-react/src`. Prefer descriptive folder names aligned with existing domains (e.g., `ingest`, `metrics`). JSON/YAML config files should remain kebab-case.

## Testing Guidelines
Invoke `make test` or `uv run pytest tests -v` for the full Python suite. Generate coverage with `make test-coverage` (report in `htmlcov/`). E2E suites in `tests/e2e` require the Docker stack and run via `make test-e2e`. UI checks run with `npm test --prefix apps/webui-react`. Name new test files `test_<feature>.py` (Python) or `<Component>.test.tsx` (frontend) and keep fixtures within `tests/fixtures` or colocated `__fixtures__`.

The Docker setup wizard can optionally emit a host-side `.env.test` that hardcodes localhost Postgres credentials (`semantik_test` DB) for integration tests; if the file is absent, `tests/conftest.py` falls back to safe defaults but will never touch remote credentials.

## Commit & Pull Request Guidelines
Aim for Conventional Commit formatting: `feat(search): add reranker fallback`, `fix(webui): guard empty filter`, etc.; reference tracking numbers or PR IDs like `(#212)` when merging. Squash local WIP commits. PRs should outline scope, highlight touchpoints (`vecpipe`, `webui`, `frontend`), list validation commands executed, and include screenshots or API traces for user-visible changes. Confirm linting, typing, and relevant tests before requesting review.

## Security & Configuration Tips
Use `wizard.sh` or `make docker-up` to generate `.env` secrets, and keep secrets out of version control. Mount document corpora in `data/` and ensure GDPR/PII redaction before sharing datasets. Rotate embeddings or clear indexes with the helper scripts in `packages/vecpipe/maintenance.py` when working with sensitive data.

### Partition-Test Troubleshooting
When resetting the Postgres volume, run `uv run alembic upgrade head` so helper functions such as `get_partition_key` exist. The integration tests now inject `partition_key` values directly; `tests/conftest.py` installs the `compute_partition_key` helper but avoids re-creating the trigger so manual keys persist. All chunk inserts in tests should flow through `PartitionAwareMixin.bulk_insert_partitioned` or `ChunkRepository`, which call the helper to compute the same partition value the trigger would have provided. This resolved the “no partition of relation `chunks` found for row” failure that appeared once the generated column/trigger wasn’t available in the test DB.
