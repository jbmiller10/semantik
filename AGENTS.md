# Repository Guidelines

## Project Structure & Module Organization
- `packages/webui/` – FastAPI REST/WebSocket API and Celery worker logic; database migrations live in `alembic/`.
- `packages/vecpipe/` – embedding/search HTTP service that talks to Qdrant.
- `packages/shared/` – common models, configs, and utilities shared across services.
- `apps/webui-react/` – React + Vite frontend.
- `tests/` – backend pytest suite (unit/integration/e2e); fixtures in `tests/fixtures/`.
- `cypress/` and Playwright tests in `apps/webui-react/` cover browser E2E flows.
- Runtime data/caches are written to `models/`, `data/`, and `logs/` (git‑ignored).

## Build, Test, and Development Commands
Backend and root workflows use `make`:
- `make dev-install` / `make install` – sync Python deps with `uv`.
- `make run` – start the WebUI FastAPI server with hot reload.
- `make dev` – run the integrated dev stack (API + worker + vecpipe).
- `make format`, `make lint`, `make type-check` – Black/Isort, Ruff, and Mypy.
- `make test`, `make test-coverage`, `make test-e2e` – run pytest suites.
- `make wizard` or `make docker-up` – guided or manual Docker stack.

Frontend:
Requires Node 18+ (see `package.json` engines).
- `make frontend-install`, `make frontend-dev`, `make frontend-build`.
- `npm run lint|test|test:e2e --prefix apps/webui-react` for ESLint, Vitest, and Playwright.

## Coding Style & Naming Conventions
- Python: 4‑space indent, 120‑char lines, and required type hints in `packages/*`. Format with `make format`; lint with `make lint`.
- TypeScript/React: 2‑space indent, ESLint enforced. Components `PascalCase`, hooks `useCamelCase`, tests `*.test.ts(x)`.

## Testing Guidelines
- Pytest tests live in `tests/`, named `test_*.py`. Use markers like `e2e`, `integration`, and `performance`.
- Frontend tests are under `apps/webui-react/src/**/__tests__/` and run with Vitest.

## Commit & Pull Request Guidelines
- Commits: short imperative subject, no trailing period; add issue/PR numbers when relevant (e.g., `rework ingestion (#257)`).
- PRs: include a clear description, link related issues, call out schema/migration changes, add UI screenshots, and ensure `make check` passes.

## Security & Configuration Tips
- Never commit secrets. Put local values in `.env`/`.env.local`.
- When adding env vars, update `.env.docker.example` and relevant docs.

## Agent‑Specific Instructions
Check for nested `AGENTS.md` files before editing scoped areas. Keep changes focused and avoid regenerating lockfiles unless required.
