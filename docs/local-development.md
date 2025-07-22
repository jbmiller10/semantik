# Local Development Setup

This guide explains how to run the WebUI locally for faster development while keeping supporting services in Docker.

## Why Use This Setup?

- **No Docker rebuild delays** - Changes to WebUI code take effect immediately
- **Hot reload** - FastAPI automatically reloads when you save files
- **Direct logs** - See WebUI logs directly in your terminal
- **Faster debugging** - Use debuggers and development tools directly
- **Reduced resource usage** - Only rebuild Docker images for service changes

## Setup Instructions

### 1. Initial Configuration

```bash
# Copy the example environment file
cp .env.local.example .env.local

# Edit .env.local and update these values to match your .env file:
# - POSTGRES_PASSWORD
# - JWT_SECRET_KEY
```

### 2. Start Development Environment

**Option A: All-in-one command**
```bash
make dev-local
```

**Option B: Manual control**
```bash
# Start Docker services (PostgreSQL, Redis, Qdrant, VecPipe, Worker)
make docker-dev-up

# In another terminal, run the WebUI locally
make run

# View Docker service logs if needed
make docker-dev-logs

# When done, stop Docker services
make docker-dev-down
```

## What's Running Where?

**In Docker:**
- PostgreSQL (port 5432)
- Redis (port 6379)
- Qdrant (port 6333)
- VecPipe/Search API (port 8000)
- Celery Worker
- Flower (port 5555)

**Locally:**
- WebUI (port 8080) with hot reload

## Common Tasks

### Run database migrations
```bash
poetry run alembic upgrade head
```

### Create a new migration
```bash
poetry run alembic revision --autogenerate -m "description"
```

### Run tests
```bash
# All tests
make test

# Without E2E tests
make test-ci
```

### Format code
```bash
make format
```

## Troubleshooting

### Port already in use
If port 8080 is already in use, you can change it:
```bash
WEBUI_PORT=8081 make run
```

### Database connection errors
1. Check Docker services are running: `docker compose -f docker-compose.dev.yml ps`
2. Verify passwords in .env.local match your .env file
3. Check PostgreSQL logs: `docker compose -f docker-compose.dev.yml logs postgres`

### Missing dependencies
```bash
poetry install
make frontend-install
```

## Switching Between Setups

### From full Docker to local development
```bash
# Stop full Docker stack
make docker-down

# Start local development
make dev-local
```

### From local development to full Docker
```bash
# Stop local development (Ctrl+C in terminal running webui)
make docker-dev-down

# Start full Docker stack
make docker-up
```