# Repository Guidelines

## Project Structure & Module Organization
- `packages/webui/`: FastAPI app (REST + WebSocket APIs) and static UI hosting (`packages/webui/static/`).
- `packages/vecpipe/`: embedding/search HTTP service that talks to Qdrant.
- `packages/shared/`: shared configs, DB models/repos, and chunking/embedding utilities.
- `apps/webui-react/`: React/Vite frontend (TypeScript + Tailwind); production build is bundled into `packages/webui/static/`.
- `alembic/` + `alembic.ini`: database migrations for Postgres.
- `tests/`: backend tests (pytest) plus `tests/e2e/`; non-`test_*.py` scripts are manual harnesses.
- `scripts/`: dev helpers (`dev.sh`, `dev-local.sh`).
- `docs/`: architecture/configuration/API references.

Note: top-level `webui/`, `shared/`, and `vecpipe/` are import shims pointing at `packages/*`.

## Build, Test, and Development Commands
Backend (Python 3.11+, dependencies via `uv`):
- `make dev-install`: install dev dependencies (`uv sync --frozen`).
- `make run`: run the API with hot reload on `http://localhost:8080` (requires services available).
- `make dev-local`: start infra via Docker profile `backend`, run migrations, then start `uvicorn`.

Frontend (Node >= 18):
- `make frontend-install`: install UI dependencies.
- `make frontend-dev`: start Vite dev server on `http://localhost:5173`.

Docker stack:
- `make wizard`: interactive setup (generates `.env` and starts the stack).
- `make docker-up` / `make docker-down`: start/stop Compose services.

## Coding Style & Naming Conventions
- Python: `black` + `isort` (line length 120), `ruff` for linting, `mypy` for type-checking (strict in `packages/*`).
  - Common workflow: `make format` → `make lint` → `make type-check` → `make test` (or `make check`).
- Frontend: ESLint via `npm run lint --prefix apps/webui-react`.
- Naming: Python modules `snake_case.py`, classes `PascalCase`; React components `PascalCase.tsx`.

## Testing Guidelines
- Backend: pytest discovers `tests/test_*.py`. Run `make test`, `make test-ci`, `make test-e2e`, or `make test-coverage`.
  - Markers: `-m e2e` (requires running services) or `-m "not e2e"`.
- Frontend: Vitest via `npm test --prefix apps/webui-react`; Playwright E2E via `npm run test:e2e --prefix apps/webui-react`.

## Commit & Pull Request Guidelines
- Commit messages are generally imperative; many use a loose Conventional Commits style (`fix:`, `feat:`, `test:`, `chore:`, `lint:`, `style:`) and sometimes include `(#123)`.
- PRs should include: short “what/why”, test plan (commands run), and screenshots for UI changes.
- If you add env vars or migrations, update the relevant templates/docs (e.g., `.env.docker.example`, wizard flow, `alembic/`).

## Security & Configuration Tips
- Never commit secrets; use `.env` / `.env.local` (generated/managed via `make wizard`).
- Avoid committing runtime/artifact directories like `models/`, `data/`, `logs/`, and built UI assets under `packages/webui/static/`.
