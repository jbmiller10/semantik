# PostgreSQL Migration - Ticket 1 Implementation

This document describes the implementation of Ticket 1: Infrastructure Setup and Configuration for the PostgreSQL migration.

## Overview

Ticket 1 establishes the foundational infrastructure for PostgreSQL support in Semantik, maintaining backward compatibility with SQLite while adding PostgreSQL as an option.

## Implemented Components

### 1. Dependencies Added

Updated `pyproject.toml` with:
- `psycopg2-binary = "^2.9.9"` - PostgreSQL adapter for Python
- `sqlalchemy = {extras = ["postgresql"], version = "^2.0.23"}` - Added PostgreSQL extras to existing SQLAlchemy

### 2. Docker Configuration

Created `docker-compose.postgres.yml`:
- PostgreSQL 16 Alpine image for minimal footprint
- Configured with health checks and resource limits
- Persistent volume for data storage
- Security hardening with minimal capabilities
- Environment-based configuration

### 3. Makefile Commands

Added PostgreSQL-specific commands:
- `make docker-postgres-up` - Start services with PostgreSQL
- `make docker-postgres-down` - Stop PostgreSQL services
- `make docker-postgres-logs` - View PostgreSQL logs
- `make docker-shell-postgres` - Access PostgreSQL shell
- `make docker-postgres-backup` - Create database backup
- `make docker-postgres-restore` - Restore from backup

### 4. Connection Configuration

Created `packages/shared/config/postgres.py`:
- PostgreSQL-specific configuration class
- Connection pooling settings
- Retry logic configuration
- Database URL construction
- Async/sync URL conversion

Created `packages/shared/database/postgres_database.py`:
- Connection manager with retry logic
- Session management
- PostgreSQL-specific optimizations
- Connection health checks

### 5. Environment Configuration

Updated `.env.example` and `.env.docker.example` with:
- PostgreSQL connection parameters
- Connection pool settings
- Database URL format examples
- Docker-specific configurations

### 6. Testing

Created `scripts/test_postgres_connection.py`:
- Verifies PostgreSQL connectivity
- Tests basic database operations
- Validates connection pooling
- Provides diagnostic information

## Usage

### Starting PostgreSQL Services

```bash
# Generate secure passwords and start services
make docker-postgres-up

# View logs
make docker-postgres-logs

# Access PostgreSQL shell
make docker-shell-postgres
```

### Testing Connection

```bash
# First, ensure PostgreSQL is running
docker compose -f docker-compose.yml -f docker-compose.postgres.yml up -d postgres

# Run the test script
poetry run python scripts/test_postgres_connection.py
```

### Environment Variables

Key PostgreSQL configuration variables:

```env
# Option 1: Full database URL
DATABASE_URL=postgresql://semantik:password@localhost:5432/semantik

# Option 2: Individual parameters
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=semantik
POSTGRES_USER=semantik
POSTGRES_PASSWORD=your-secure-password

# Connection pool settings
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=40
DB_POOL_TIMEOUT=30
DB_POOL_RECYCLE=3600
DB_POOL_PRE_PING=true
```

## Security Considerations

1. **Password Generation**: The Makefile automatically generates secure passwords using OpenSSL
2. **Container Security**: PostgreSQL container runs with minimal capabilities and no-new-privileges
3. **Connection Security**: Pre-ping enabled to verify connections before use
4. **Resource Limits**: Configured with CPU and memory limits to prevent resource exhaustion

## Next Steps

With Ticket 1 complete, the infrastructure is ready for:
- Ticket 2: Repository Pattern Implementation
- Ticket 3: Data Migration Implementation
- Ticket 4: Testing and Performance Validation
- Ticket 5: Production Deployment and Migration

## Troubleshooting

### Connection Failed

1. Ensure PostgreSQL container is running:
   ```bash
   docker compose -f docker-compose.yml -f docker-compose.postgres.yml ps
   ```

2. Check PostgreSQL logs:
   ```bash
   make docker-postgres-logs
   ```

3. Verify environment variables are set correctly

### Permission Errors

Ensure the data directories have correct permissions:
```bash
sudo chown -R 1000:1000 ./models ./data ./logs
```

### Port Conflicts

If port 5432 is already in use, modify the port in:
- `docker-compose.postgres.yml`
- Your `.env` file
- Any connection strings