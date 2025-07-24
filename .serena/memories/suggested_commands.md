# Suggested Commands for Semantik Development

## Code Quality & Testing
```bash
make check       # Run format, lint, and test
make format      # Format code with black and isort  
make lint        # Run linting with ruff
make type-check  # Run type checking with mypy
make test        # Run all tests
make test-ci     # Run tests excluding E2E
make test-e2e    # Run only E2E tests
make test-coverage # Run tests with coverage report
```

## Frontend Development
```bash
make frontend-install  # Install frontend dependencies
make frontend-build    # Build frontend for production
make frontend-dev      # Start frontend dev server
make frontend-test     # Run frontend tests
npm run lint          # Lint frontend code (from apps/webui-react)
```

## Docker Commands
```bash
make wizard           # Interactive Docker setup wizard
make docker-up        # Start all services
make docker-down      # Stop and remove containers
make docker-logs      # View logs from all services
make docker-ps        # Show container status
make docker-restart   # Restart all services
make docker-build-fresh # Rebuild without cache

# Development mode (backend in Docker, frontend local)
make docker-dev-up    # Start backend services only
make docker-dev-down  # Stop backend services
make run              # Run webui locally with hot reload
```

## Database Management
```bash
poetry run alembic upgrade head        # Run migrations
make docker-postgres-backup            # Create database backup
make docker-shell-postgres             # Access PostgreSQL shell
BACKUP_FILE=path/to/backup.sql make docker-postgres-restore  # Restore backup
```

## Local Development
```bash
poetry install                         # Install dependencies
./start_all_services.sh               # Start all services (non-Docker)
./stop_all_services.sh                # Stop all services
./status_services.sh                  # Check service status
./restart_all_services_rebuild.sh     # Rebuild and restart
```

## System Utilities (Linux)
```bash
git status        # Check repository status
git diff          # View uncommitted changes
git log           # View commit history
ls -la            # List files with details
cd                # Change directory
grep -r "pattern" # Recursive text search
find . -name "*.py" # Find files by pattern
```