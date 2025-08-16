# Integration Tests Documentation

## Overview

This directory contains comprehensive integration tests for the Semantik chunking feature. These tests validate the complete workflow from document upload through chunking, storage, and search functionality.

## Test Files

### Core Workflow Tests
- **`test_chunking_workflow.py`** - End-to-end chunking workflow tests
- **`test_partition_distribution.py`** - PostgreSQL partition distribution and performance tests
- **`test_chunking_error_recovery.py`** - Error handling and recovery mechanisms

### Performance Tests
- **`test_memory_stability.py`** - Memory leak detection and stability tests
- **`test_websocket_load.py`** - WebSocket load testing (tests/load/)

### Helper Files
- **`conftest.py`** - Shared fixtures and test configuration
- **`test_helpers.py`** - Common helper functions to avoid code duplication

## Prerequisites

### Required Services

1. **PostgreSQL Database**
   ```bash
   # Start test database
   docker run -d \
     --name semantik-test-db \
     -e POSTGRES_USER=semantik \
     -e POSTGRES_PASSWORD=semantik \
     -e POSTGRES_DB=semantik_test \
     -p 5432:5432 \
     postgres:14
   ```

2. **Redis**
   ```bash
   # Start Redis
   docker run -d \
     --name semantik-test-redis \
     -p 6379:6379 \
     redis:7-alpine
   ```

3. **Full Stack (Recommended)**
   ```bash
   # Use docker-compose for all services
   docker-compose -f docker-compose.test.yml up -d
   ```

### Environment Variables

Create a `.env.test` file:
```bash
TEST_DATABASE_URL=postgresql+asyncpg://semantik:semantik@localhost:5432/semantik_test
TEST_REDIS_URL=redis://localhost:6379/1
TEST_SERVER_URL=ws://localhost:8080
```

### Python Dependencies

```bash
# Install test dependencies
poetry install --with dev

# Additional dependencies for load testing
poetry add --group dev locust memory-profiler faker
```

## Running Tests

### All Integration Tests
```bash
# Run all integration tests
poetry run pytest tests/integration -v

# Run with coverage
poetry run pytest tests/integration --cov=packages --cov-report=html
```

### Specific Test Categories

```bash
# Fast tests only (excludes slow tests)
poetry run pytest tests/integration -m "not slow" -v

# Partition tests
poetry run pytest tests/integration/test_partition_distribution.py -v

# Error recovery tests
poetry run pytest tests/integration/test_chunking_error_recovery.py -v

# Memory tests (may take longer)
poetry run pytest tests/integration/test_memory_stability.py -v -m memory
```

### Load Tests

```bash
# WebSocket load tests
poetry run pytest tests/load/test_websocket_load.py -v -m load

# Run with Locust UI
locust -f tests/load/test_websocket_load.py --host http://localhost:8080
```

## Test Markers

Tests are marked with categories for selective execution:

- **`@pytest.mark.slow`** - Tests that take >5 seconds
- **`@pytest.mark.integration`** - Integration tests requiring external services
- **`@pytest.mark.load`** - Load/performance tests
- **`@pytest.mark.memory`** - Memory profiling tests

Skip slow tests in CI:
```bash
pytest tests/integration -m "not slow and not load"
```

## Performance Optimization

### Test Data Sizes

Tests use configurable data sizes to balance coverage and speed:

| Size     | Bytes   | Use Case                    |
|----------|---------|----------------------------|
| tiny     | 100     | Unit tests, quick checks    |
| small    | 500     | Default for most tests      |
| medium   | 5KB     | Typical document size       |
| large    | 50KB    | Large document handling     |
| huge     | 500KB   | Stress testing (marked slow)|

### Fixture Optimization

- Database sessions are function-scoped with automatic rollback
- Redis is cleared between tests
- Heavy fixtures are lazy-loaded only when needed

### Parallel Execution

```bash
# Run tests in parallel (requires pytest-xdist)
poetry run pytest tests/integration -n auto

# Run with 4 workers
poetry run pytest tests/integration -n 4
```

## Troubleshooting

### Common Issues

1. **Database Connection Errors**
   ```bash
   # Check PostgreSQL is running
   docker ps | grep postgres
   
   # Check connection
   psql postgresql://semantik:semantik@localhost:5432/semantik_test
   ```

2. **Redis Connection Errors**
   ```bash
   # Check Redis is running
   docker ps | grep redis
   
   # Test connection
   redis-cli ping
   ```

3. **Fixture Not Found**
   - Ensure `conftest.py` is in the integration directory
   - Check imports match the package structure

4. **Tests Too Slow**
   - Use smaller datasets: `TestDataGenerator.generate_documents(count=5, size="small")`
   - Mark slow tests: `@pytest.mark.slow`
   - Run fast tests only: `pytest -m "not slow"`

### Debug Mode

```bash
# Run with verbose output
poetry run pytest tests/integration -vvs

# Run with pdb on failure
poetry run pytest tests/integration --pdb

# Show local variables on failure
poetry run pytest tests/integration -l
```

## CI/CD Configuration

### GitHub Actions Example

```yaml
name: Integration Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:14
        env:
          POSTGRES_USER: semantik
          POSTGRES_PASSWORD: semantik
          POSTGRES_DB: semantik_test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
      
      redis:
        image: redis:7-alpine
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install poetry
          poetry install --with dev
      
      - name: Run integration tests (fast only)
        env:
          TEST_DATABASE_URL: postgresql+asyncpg://semantik:semantik@localhost:5432/semantik_test
          TEST_REDIS_URL: redis://localhost:6379/1
        run: |
          poetry run pytest tests/integration -m "not slow and not load" -v --cov
```

## Best Practices

1. **Test Isolation**
   - Each test should be independent
   - Use fixtures for setup/teardown
   - Clear state between tests

2. **Performance**
   - Keep test data small by default
   - Mark slow tests appropriately
   - Use test data factories

3. **Debugging**
   - Use descriptive test names
   - Add comments for complex assertions
   - Log important state changes

4. **Maintenance**
   - Keep helper functions in `test_helpers.py`
   - Update this README when adding new tests
   - Review and optimize slow tests regularly

## Coverage Goals

Target coverage for chunking feature:
- Overall: >80%
- Core logic: >90%
- Error handling: >75%
- Edge cases: 100%

Check current coverage:
```bash
poetry run pytest tests/integration --cov=packages.webui.services.chunking_service --cov-report=term-missing
```