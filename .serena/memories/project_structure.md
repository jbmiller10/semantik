# Semantik Project Structure

## Root Directory
```
semantik/
├── packages/               # Python packages
│   ├── webui/             # FastAPI backend service
│   │   ├── api/           # API routers (v2 for new endpoints)
│   │   ├── services/      # Business logic layer
│   │   ├── repositories/  # Database access layer
│   │   └── static/        # Static files served by FastAPI
│   ├── vecpipe/           # Document processing & search service
│   └── shared/            # Shared models and utilities
├── apps/
│   └── webui-react/       # React frontend application
│       ├── src/
│       │   ├── components/
│       │   ├── stores/    # Zustand state management
│       │   ├── services/  # API client services
│       │   └── hooks/     # Custom React hooks
├── alembic/               # Database migrations
├── tests/                 # Test suite
├── scripts/               # Utility scripts
├── docs/                  # Documentation
└── docker/                # Docker-related files
```

## Configuration Files
- `pyproject.toml` - Python dependencies and tool configs
- `package.json` - Frontend dependencies (in apps/webui-react)
- `docker-compose.yml` - Main Docker configuration
- `docker-compose.prod.yml` - Production overrides
- `docker-compose.cuda.yml` - GPU-specific configuration
- `alembic.ini` - Database migration configuration
- `Makefile` - Common development commands
- `.env` - Environment variables (create from .env.docker.example)

## Important Patterns
1. **API Versioning**: New endpoints go in `api/v2/`
2. **Service Layer**: All business logic in `services/`
3. **Repository Pattern**: Database access through repositories
4. **Async Throughout**: Use async/await consistently
5. **Type Safety**: Full type hints in Python, strict TypeScript

## Key Entry Points
- **WebUI Backend**: `packages/webui/main.py`
- **VecPipe Service**: `packages/vecpipe/main.py`
- **Worker**: `packages/webui/celery_app.py`
- **Frontend**: `apps/webui-react/src/main.tsx`