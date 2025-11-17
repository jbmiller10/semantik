# Repository Guidelines

## Project Structure & Module Organization
Semantik is a polyglot monorepo. Python services live in `packages/`: `vecpipe` powers ingestion/search workers, `webui` exposes the FastAPI backend, and `shared` holds contracts, config, and utilities shared across services. The React frontend sits in `apps/webui-react` (Vite project with assets in `public/`). Database migrations live under `alembic/`. Shell automation and orchestration scripts are in `scripts/`. All automated tests reside in `tests/` with domain-specific folders (`tests/domain`, `tests/performance`, `tests/e2e`), and sample documents live in `data/`.

## Build, Test, and Development Commands
Run `make dev-install` for a full `uv` sync of Python dependencies and tooling; use `npm install --prefix apps/webui-react` (or `make frontend-install`) for the UI. `make dev` starts the integrated stack via `scripts/dev.sh`. For API-only hot reload, use `make run`. Frontend iterations happen through `make frontend-dev`. Server-side builds rely on `make build`; React production bundles come from `npm run build --prefix apps/webui-react`. Docker workflows (`make docker-up`, `make docker-down`, `make docker-dev-up`) provision services and secrets automatically.

`make docker-down` now mirrors `docker compose down` and keeps named volumes; run `make docker-down-clean` when you explicitly want to blow away volumes (e.g., to reset Postgres data).

## Coding Style & Naming Conventions
Python code follows 4-space indentation, 120-character lines, and exhaustive typing. Before pushing, run `make format` (Black + Isort) and `make lint` (Ruff) followed by `make type-check` (Mypy). Modules stay snake_case (`search_api.py`), pytest fixtures snake_case, and React components PascalCase inside `apps/webui-react/src`. Prefer descriptive folder names aligned with existing domains (e.g., `ingest`, `metrics`). JSON/YAML config files should remain kebab-case.

Always ensure Black and Ruff pass before marking a ticket complete—rerun `make format` and `make lint` (or the equivalent `uv run` commands) after your final changes so reviewers never see style regressions.

## Testing Guidelines
Invoke `make test` or `uv run pytest tests -v` for the full Python suite. Generate coverage with `make test-coverage` (report in `htmlcov/`). E2E suites in `tests/e2e` require the Docker stack and run via `make test-e2e`. UI checks run with `npm test --prefix apps/webui-react`. Name new test files `test_<feature>.py` (Python) or `<Component>.test.tsx` (frontend) and keep fixtures within `tests/fixtures` or colocated `__fixtures__`.

### Frontend E2E (Cypress)
A minimal Cypress harness exists for projection visualization flows (`cypress/e2e/projection_visualize.cy.ts`) with config in `cypress.config.ts` (base URL `http://localhost:5173`). To run these tests:

- Start the backend as usual, e.g.:
  - `source .env && PYTHONPATH=/home/john/semantik uv run python -m uvicorn packages.webui.main:app --host 0.0.0.0 --port 8080 --reload`
- Start the frontend dev server:
  - `npm run dev:frontend`
- Install Cypress once at the repo root:
  - `npm install -D cypress`
- Then run Cypress against the projection spec:
  - Interactive: `npx cypress open` (select `cypress/e2e/projection_visualize.cy.ts`)
  - Headless: `npx cypress run --spec cypress/e2e/projection_visualize.cy.ts`

### Dedicated Test Database
The project includes a dedicated PostgreSQL instance for integration and API tests to ensure test isolation. Start it using the `testing` Docker Compose profile:

```bash
# Start the dedicated test database
docker compose --profile testing up -d postgres_test
```

The `postgres_test` service uses separate credentials defined in `.env` or `.env.test` (e.g., `POSTGRES_TEST_PORT=55432`, `POSTGRES_TEST_DB=semantik_test`, `POSTGRES_TEST_USER=semantik`, `POSTGRES_TEST_PASSWORD=semantik_test_password`). This prevents test data pollution and allows parallel test execution without interfering with the development database. The test database runs on port 55432 by default to avoid conflicts.

The Docker setup wizard can optionally emit a host-side `.env.test` that hardcodes localhost Postgres credentials for integration tests; if the file is absent, `tests/conftest.py` falls back to safe defaults but will never touch remote credentials.

## Commit & Pull Request Guidelines
Aim for Conventional Commit formatting: `feat(search): add reranker fallback`, `fix(webui): guard empty filter`, etc.; reference tracking numbers or PR IDs like `(#212)` when merging. Squash local WIP commits. PRs should outline scope, highlight touchpoints (`vecpipe`, `webui`, `frontend`), list validation commands executed, and include screenshots or API traces for user-visible changes. Confirm linting, typing, and relevant tests before requesting review.

## Security & Configuration Tips
Use `wizard.sh` or `make docker-up` to generate `.env` secrets, and keep secrets out of version control. Mount document corpora in `data/` and ensure GDPR/PII redaction before sharing datasets. Rotate embeddings or clear indexes with the helper scripts in `packages/vecpipe/maintenance.py` when working with sensitive data.

### Partition-Test Troubleshooting
When resetting the Postgres volume, run `uv run alembic upgrade head` so helper functions such as `get_partition_key` exist. The integration tests now inject `partition_key` values directly; `tests/conftest.py` installs the `compute_partition_key` helper but avoids re-creating the trigger so manual keys persist. All chunk inserts in tests should flow through `PartitionAwareMixin.bulk_insert_partitioned` or `ChunkRepository`, which call the helper to compute the same partition value the trigger would have provided. This resolved the “no partition of relation `chunks` found for row” failure that appeared once the generated column/trigger wasn’t available in the test DB.
