# TICKET-004: PostgreSQL Deployment Testing and Documentation

**Type:** Infrastructure / Documentation
**Priority:** High
**Blocks:** Production deployment with PostgreSQL
**Component:** Deployment / Documentation

## Problem Statement

The PostgreSQL deployment path is broken and appears untested. Fresh setup instructions do not work out of the box with PostgreSQL, indicating this deployment option hasn't been properly validated. This is critical as PostgreSQL is likely the preferred production database.

## Current Issues

1. **Setup Wizard**: Doesn't properly configure for PostgreSQL deployment
2. **Docker Compose**: PostgreSQL overlay has issues:
   - Migration compatibility problems
   - Environment variable configuration issues
   - Service startup failures

3. **Documentation**: 
   - No clear PostgreSQL setup instructions
   - Missing troubleshooting guide
   - No mention of PostgreSQL-specific requirements

## Implementation Steps

### 1. Fix Setup Wizard
- Add PostgreSQL option to `docker_setup_tui.py`
- Generate appropriate `.env` configuration for PostgreSQL
- Set correct environment variables for all services

### 2. Create PostgreSQL Setup Documentation

Create `docs/POSTGRESQL_SETUP.md`:
```markdown
# PostgreSQL Setup Guide

## Quick Start
1. Copy `.env.postgres.example` to `.env`
2. Run `make wizard` and select PostgreSQL
3. Run `docker compose -f docker-compose.yml -f docker-compose.postgres.yml up -d`

## Configuration
- Database URL format
- Required environment variables
- Connection pool settings

## Troubleshooting
- Common startup errors
- Migration issues
- Connection problems
```

### 3. Add Integration Tests

Create `tests/deployment/test_postgres_deployment.py`:
```python
def test_fresh_postgres_deployment():
    """Test that fresh PostgreSQL deployment works."""
    # Start PostgreSQL container
    # Run migrations
    # Create test collection
    # Verify operations work
```

### 4. Update Makefile

Add PostgreSQL-specific targets:
```makefile
# PostgreSQL deployment
docker-postgres-up:
	docker compose -f docker-compose.yml -f docker-compose.postgres.yml up -d

docker-postgres-down:
	docker compose -f docker-compose.yml -f docker-compose.postgres.yml down

docker-postgres-fresh:
	docker compose -f docker-compose.yml -f docker-compose.postgres.yml down -v
	docker compose -f docker-compose.yml -f docker-compose.postgres.yml up -d --build

test-postgres-deployment:
	pytest tests/deployment/test_postgres_deployment.py
```

### 5. Create CI/CD Test

Add GitHub Action workflow:
```yaml
name: PostgreSQL Deployment Test
on: [push, pull_request]

jobs:
  test-postgres:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Test PostgreSQL Deployment
        run: |
          cp .env.postgres.example .env
          docker compose -f docker-compose.yml -f docker-compose.postgres.yml up -d
          # Wait for healthy
          # Run basic tests
```

### 6. Fix Environment Templates

Create proper `.env.postgres.example`:
```bash
# PostgreSQL Configuration
DATABASE_URL=postgresql://semantik:changeme@postgres:5432/semantik
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=semantik
POSTGRES_USER=semantik
POSTGRES_PASSWORD=changeme

# Must be same for all services
JWT_SECRET_KEY=generate-this
INTERNAL_API_KEY=generate-this

# Other settings...
```

## Testing Requirements

1. **Fresh Deployment Test**:
   - Start from clean state (no volumes)
   - Follow documented steps exactly
   - Verify all services start
   - Create and search collection

2. **Migration Test**:
   - Test migrations on empty PostgreSQL
   - Test migrations with existing data
   - Test rollback procedures

3. **Performance Test**:
   - Verify connection pooling works
   - Test under load
   - Monitor resource usage

4. **Compatibility Test**:
   - Ensure SQLite deployments still work
   - Test switching between databases

## Acceptance Criteria

- [ ] Fresh PostgreSQL deployment works with documented steps
- [ ] All services start without errors
- [ ] Migrations run successfully
- [ ] Collections can be created and searched
- [ ] CI/CD tests pass
- [ ] Documentation is clear and complete

## Documentation Requirements

1. **README.md** updates:
   - Add PostgreSQL option to deployment section
   - Link to detailed PostgreSQL guide

2. **Deployment Guide**:
   - Step-by-step PostgreSQL setup
   - Configuration options
   - Performance tuning

3. **Troubleshooting Guide**:
   - Common PostgreSQL errors
   - How to check logs
   - How to verify setup

## Additional Considerations

1. **Default Database**: Consider making PostgreSQL the default for production
2. **Migration Path**: Document how to migrate from SQLite to PostgreSQL
3. **Backup Procedures**: Document PostgreSQL backup/restore
4. **Monitoring**: Add PostgreSQL-specific health checks

## References

- Current docker-compose.postgres.yml
- PostgreSQL Docker documentation
- SQLAlchemy PostgreSQL dialect docs