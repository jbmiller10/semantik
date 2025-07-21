# TICKET-005: Add Database Compatibility Testing

**Type:** Testing / Quality Assurance
**Priority:** High
**Depends On:** TICKET-001 (Migration fixes)
**Component:** Testing Infrastructure

## Problem Statement

The current test suite doesn't validate that the application works correctly with both SQLite and PostgreSQL. This led to PostgreSQL-specific issues going undetected until deployment. We need automated tests that run against both database backends to prevent future compatibility issues.

## Requirements

### 1. Test Infrastructure
- Run same test suite against both SQLite and PostgreSQL
- Use test containers for PostgreSQL tests
- Parallel test execution for different databases
- Clear test output showing which database is being tested

### 2. Migration Testing
- Test all migrations run successfully on both databases
- Test upgrade and downgrade paths
- Test migration with existing data

### 3. Feature Testing
- All CRUD operations work identically
- Enum handling works correctly
- Transaction behavior is consistent
- Search functionality performs equally

## Implementation Steps

### 1. Create Database Test Fixtures

`tests/conftest.py`:
```python
import pytest
from testcontainers.postgres import PostgresContainer

@pytest.fixture(params=["sqlite", "postgresql"])
def database_url(request):
    """Provide database URL for both SQLite and PostgreSQL."""
    if request.param == "sqlite":
        return "sqlite+aiosqlite:///:memory:"
    else:
        with PostgresContainer("postgres:16-alpine") as postgres:
            yield postgres.get_connection_url()

@pytest.fixture
async def db_session(database_url):
    """Create database session for tests."""
    # Setup database with URL
    # Run migrations
    # Yield session
    # Cleanup
```

### 2. Create Compatibility Test Suite

`tests/compatibility/test_database_operations.py`:
```python
class TestDatabaseCompatibility:
    """Test that operations work identically on all databases."""
    
    async def test_collection_crud(self, db_session):
        """Test collection creation, read, update, delete."""
        # Create collection
        # Verify enum fields work
        # Update status
        # Delete collection
    
    async def test_enum_handling(self, db_session):
        """Test that enums work correctly."""
        # Test all enum types
        # Test invalid enum values
        # Test enum queries
    
    async def test_concurrent_operations(self, db_session):
        """Test concurrent access patterns."""
        # Multiple operations on same collection
        # Test locking behavior
        # Test transaction isolation
```

### 3. Migration Compatibility Tests

`tests/compatibility/test_migrations.py`:
```python
async def test_migrations_on_all_databases(database_url):
    """Test that all migrations run on both databases."""
    # Create fresh database
    # Run all migrations
    # Verify schema is correct
    # Test downgrade

async def test_migration_with_data(database_url):
    """Test migrations with existing data."""
    # Create old schema
    # Insert test data
    # Run migrations
    # Verify data integrity
```

### 4. Update CI/CD Pipeline

`.github/workflows/test-compatibility.yml`:
```yaml
name: Database Compatibility Tests

on: [push, pull_request]

jobs:
  test-sqlite:
    name: SQLite Tests
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run SQLite Tests
        run: |
          pytest tests/ -m "not postgresql_only"
  
  test-postgresql:
    name: PostgreSQL Tests
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16-alpine
        env:
          POSTGRES_PASSWORD: testpass
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    steps:
      - uses: actions/checkout@v3
      - name: Run PostgreSQL Tests
        env:
          DATABASE_URL: postgresql://postgres:testpass@localhost/test
        run: |
          pytest tests/ -m "not sqlite_only"
  
  test-compatibility:
    name: Cross-Database Compatibility
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Compatibility Tests
        run: |
          pytest tests/compatibility/ -v
```

### 5. Test Markers and Configuration

`pytest.ini`:
```ini
[pytest]
markers =
    sqlite_only: Tests that only run on SQLite
    postgresql_only: Tests that only run on PostgreSQL
    slow: Slow tests that can be skipped
    compatibility: Database compatibility tests
```

### 6. Performance Comparison Tests

`tests/compatibility/test_performance.py`:
```python
@pytest.mark.slow
async def test_bulk_insert_performance(database_url, benchmark):
    """Compare bulk insert performance."""
    # Insert 10k documents
    # Measure time
    # Assert reasonable performance

@pytest.mark.slow  
async def test_search_performance(database_url, benchmark):
    """Compare search performance."""
    # Create collection with 1k documents
    # Run various searches
    # Compare performance
```

## Testing Matrix

| Test Category | SQLite | PostgreSQL | Both |
|--------------|--------|------------|------|
| Unit Tests | ✓ | ✓ | ✓ |
| Integration Tests | ✓ | ✓ | ✓ |
| Migration Tests | ✓ | ✓ | ✓ |
| Performance Tests | ✓ | ✓ | Compare |
| Deployment Tests | ✓ | ✓ | - |

## Acceptance Criteria

- [ ] All tests pass on both SQLite and PostgreSQL
- [ ] CI/CD runs tests on both databases
- [ ] Migration tests catch database-specific issues
- [ ] Performance is comparable between databases
- [ ] Test output clearly shows which database is being used
- [ ] Documentation explains how to run compatibility tests

## Additional Considerations

1. **Test Data**: Create realistic test data that exercises edge cases
2. **Error Cases**: Test database-specific error handling
3. **Connection Pooling**: Test connection pool behavior
4. **Async Behavior**: Ensure async operations work correctly
5. **Resource Cleanup**: Ensure tests don't leak resources

## References

- pytest documentation
- testcontainers-python for PostgreSQL testing
- SQLAlchemy testing best practices
- Current test suite structure