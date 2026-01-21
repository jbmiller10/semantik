# Tech Stack

## Backend
- **Language:** Python 3.11+
- **Frameworks:** FastAPI (Web API), Celery (Background Tasks)
- **Database (Relational):** Postgres (managed with SQLAlchemy and Alembic)
- **Database (Vector):** Qdrant
- **Caching & Message Broker:** Redis
- **Dependency Management:** `uv`

## Frontend
- **Language:** TypeScript
- **Framework:** React (built with Vite)
- **UI Framework:** Material Design (MUI)
- **Dependency Management:** `npm`

## Infrastructure & DevOps
- **Containerization:** Docker & Docker Compose
- **Orchestration:** Docker Compose (local dev and prod)
- **CI/CD:** GitHub Actions (for testing and linting)

## Tooling & Quality Assurance
- **Linting & Formatting:** Ruff, Black, isort
- **Type Checking:** MyPy
- **Testing:** Pytest (Backend), Vitest/Jest (Frontend)
