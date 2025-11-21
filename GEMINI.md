# Semantik - Project Context

## Project Overview

**Semantik** is a self-hosted, high-performance document embedding and search system. It is designed for privacy and control, running entirely on local hardware (GPU recommended).

The system follows a **collection-centric architecture**, allowing documents to be grouped with specific embedding models and configurations. It features a clear separation between the control plane (WebUI) and the search engine (VecPipe).

### Key Technologies

*   **Backend:** Python 3.11+, FastAPI, Celery (Worker), SQLAlchemy (ORM), Alembic (Migrations), Pydantic.
*   **Frontend:** React 19, Vite, TypeScript, Tailwind CSS, React Query, Zustand.
*   **Vector Engine:** Qdrant, PyTorch, Sentence Transformers, Hugging Face Models (e.g., Qwen, BAAI).
*   **Infrastructure:** Docker Compose, PostgreSQL 16, Redis 7.
*   **Package Manager:** `uv` (Python), `npm` (Node.js).

### Architecture

*   **`packages/vecpipe` (Port 8001):** Headless vector processing and search API. Independent service. Handles extracting, chunking, embedding, and searching.
*   **`packages/webui` (Port 8080):** User-facing control plane. Handles auth, collection management, task orchestration (via Celery), and proxies search requests.
*   **`packages/shared`:** Common code, database models, and contracts shared between `webui` and `vecpipe`.
*   **`apps/webui-react`:** Modern Single Page Application (SPA) for managing the system.

## Building and Running

### Prerequisites
*   Docker & Docker Compose
*   Python 3.11+
*   Node.js 18+
*   `uv` (Python package manager)

### Quick Start (Docker)
The easiest way to run the full stack is via Docker.

```bash
# Setup wizard (generates .env)
make wizard

# Start all services
make docker-up

# Stop services
make docker-down
```

### Development Workflows

#### 1. Local Development (Hybrid)
Run supporting services in Docker, but run the WebUI and/or Frontend locally for rapid iteration.

```bash
# Start infra (Postgres, Redis, Qdrant, Vecpipe) in Docker
make docker-dev-up

# Run WebUI backend locally (hot reload)
# Requires .env configuration
make run

# Run React Frontend locally (hot reload)
make frontend-dev
```

#### 2. Full Local Development
If you need to modify `vecpipe` or shared packages frequently.

```bash
# Install Python dependencies
make dev-install

# Install Frontend dependencies
make frontend-install
```

### Key Commands (`Makefile`)

*   **Setup:** `make wizard`, `make install`
*   **Docker:** `make docker-up`, `make docker-down`, `make docker-logs`
*   **Backend Dev:** `make run` (starts WebUI), `make dev-local` (script)
*   **Frontend Dev:** `make frontend-dev`
*   **Testing:**
    *   Backend: `make test` (runs pytest)
    *   Frontend: `make frontend-test` (runs vitest)
    *   E2E: `make test-e2e` (requires running stack)
*   **Quality:** `make format`, `make lint`, `make type-check`

## Development Conventions

*   **Code Style:**
    *   **Python:** Enforced via `ruff` (linting) and `black` (formatting).
    *   **Frontend:** `eslint` and `prettier` (implied).
*   **Type Safety:**
    *   **Python:** `mypy` is used for static type checking. Strict typing is encouraged.
    *   **Frontend:** TypeScript is used throughout.
*   **Testing:**
    *   Write unit tests for new logic (`tests/`).
    *   Use `pytest` fixtures from `conftest.py`.
    *   Frontend tests use `vitest` and `testing-library`.
*   **Architecture:**
    *   **Collection-Centric:** Logic should revolve around Collections, not generic "jobs".
    *   **Separation of Concerns:** `vecpipe` should remain unaware of users/auth. `webui` handles permissions.
    *   **Shared Code:** Put common models/utils in `packages/shared` to avoid duplication.

## Directory Structure

*   `apps/` - Frontend applications.
    *   `webui-react/` - Main React app.
*   `packages/` - Python source code.
    *   `shared/` - Shared library.
    *   `vecpipe/` - Search engine service.
    *   `webui/` - Management API & Worker.
*   `scripts/` - Utility scripts for dev/ops.
*   `tests/` - Python backend tests.
*   `docs/` - Extensive project documentation.
