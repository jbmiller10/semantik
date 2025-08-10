# Database Tests

These tests require a running PostgreSQL database with the proper schema setup.

## Prerequisites

1. **Start PostgreSQL Database**:
   ```bash
   # Using Docker Compose (recommended)
   make docker-postgres-up
   # OR
   docker compose up postgres -d
   ```

2. **Run Database Migrations**:
   ```bash
   poetry run alembic upgrade head
   ```

## Running Tests

### Option 1: Using the provided script (recommended)
```bash
bash tests/database/run_partition_tests.sh
```

### Option 2: Manual test execution
```bash
# Set environment variables
export POSTGRES_PASSWORD="postgres"
export DATABASE_URL="postgresql+asyncpg://postgres:postgres@localhost:5432/semantik_test"

# Run specific test
poetry run pytest tests/database/test_partitioning.py -v

# Run all database tests
poetry run pytest tests/database/ -v
```

## Test Database Configuration

The tests use the following default configuration:
- Host: localhost
- Port: 5432
- User: postgres
- Password: postgres
- Database: semantik_test

You can override these by setting environment variables:
- `DATABASE_URL`: Complete database URL
- `POSTGRES_USER`: Database user
- `POSTGRES_PASSWORD`: Database password
- `POSTGRES_DB`: Database name
- `POSTGRES_HOST`: Database host
- `POSTGRES_PORT`: Database port

## Troubleshooting

### "password authentication failed"
This means PostgreSQL is running but the credentials are wrong. Check:
1. Your `.env` file for the correct password
2. That you're using the test database credentials, not production

### "could not connect to server"
PostgreSQL is not running. Start it with:
```bash
make docker-postgres-up
```

### "relation does not exist" errors
The database schema is not set up. Run:
```bash
poetry run alembic upgrade head
```