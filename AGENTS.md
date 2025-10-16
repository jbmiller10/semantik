# Repository Guidelines

## Project Structure & Module Organization
- `packages/vecpipe/`: FastAPI search service, model management, Qdrant integration.
- `packages/webui/`: Web UI backend (FastAPI), Celery tasks, service layer, shared dependencies.
- `packages/shared/`: Cross-cutting config, repositories, embedding utilities, metrics.
- `apps/webui-react/`: React 19 SPA with Vite build, Zustand stores, Playwright/Vitest tests.
- `tests/`: Python test suites; frontend tests reside under `apps/webui-react/tests`.
- Supporting assets: `docs/` (architecture and ops), `alembic/` (DB migrations), Docker files, Makefile.

## Build, Test, and Development Commands
- `poetry install --with dev`: install Python deps (run once per environment).
- `make dev`: start full local stack (honors docker-compose for services).
- `make test` / `poetry run pytest`: execute backend test suite with coverage.
- `npm install` (in `apps/webui-react`): install frontend deps.
- `npm run dev`: launch Vite dev server.
- `npm run test:coverage`: run Vitest with coverage; `npm run test:e2e` executes Playwright tests.

## Coding Style & Naming Conventions
- Python: Black (120-char line limit), Ruff, isort, mypy (targets Py3.11). Run via `make format` / `make lint`.
- React/TypeScript: ESLint (config in `apps/webui-react/eslint.config.js`), Prettier via Vite defaults.
- Use descriptive snake_case for Python modules/functions, PascalCase for React components.

## Testing Guidelines
- Backend tests in `tests/`; follow `test_*.py` naming. Use pytest fixtures, aim for meaningful coverage (Makefile enforces HTML/XML reports).
- Frontend unit tests under `apps/webui-react/src/**/__tests__/*`; integration/E2E in `apps/webui-react/tests`.
- Ensure Playwright browsers installed before `npm run test:e2e` (`npx playwright install`).

## Commit & Pull Request Guidelines
- Follow conventional, present-tense commit messages (e.g., `fix: handle missing qdrant metadata`). Reference work item IDs when available.
- PRs should include: summary, linked issue, testing notes (`make test`, `npm run test:coverage`), screenshots/GIFs for UI changes.
- Request review from domain owners (backend/frontend) and run CI before submit.

## Security & Configuration Tips
- Never commit real secrets. Update `.env.example` and `.env.docker.example` when config changes.
- In production ensure `INTERNAL_API_KEY`, `JWT_SECRET_KEY`, and DB credentials are strong and set explicitly; defaults are dev-only.
