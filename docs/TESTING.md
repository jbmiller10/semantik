# Testing Guide for Semantik

## Overview

This guide covers the testing philosophy, practices, and procedures for the Semantik codebase. We maintain a comprehensive test suite to ensure reliability, performance, and maintainability.

## Testing Philosophy

### Core Principles
1. **Test-Driven Development**: Write tests before or alongside feature implementation
2. **Comprehensive Coverage**: Aim for high code coverage but prioritize meaningful tests
3. **Fast Feedback**: Tests should run quickly to encourage frequent execution
4. **Isolation**: Tests should be independent and not rely on external services when possible
5. **Clarity**: Test names should clearly describe what they test

### Testing Pyramid
```
         /\
        /  \  E2E Tests (Few)
       /────\
      /      \  Integration Tests (Some)
     /────────\
    /          \  Unit Tests (Many)
   /────────────\
```

## Test Environment Setup

### Prerequisites
```bash
# Install development dependencies
poetry install --with dev

# Verify pytest is available
poetry run pytest --version
```

### Environment Variables
Create a `.env.test` file for test-specific configuration:
```bash
# Use mock embeddings to avoid GPU requirements
USE_MOCK_EMBEDDINGS=true

# Use test database
WEBUI_DB=data/test.db

# Disable authentication for API tests
DISABLE_AUTH=true

# Use local test Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

## Running Tests

### Basic Test Execution
```bash
# Run all tests
make test

# Run with verbose output
poetry run pytest -v

# Run specific test file
poetry run pytest tests/test_auth.py

# Run specific test function
poetry run pytest tests/test_auth.py::test_login_success

# Run tests matching pattern
poetry run pytest -k "rerank"
```

### Test Modes

#### CPU Mode (Default)
```bash
# Force CPU mode for testing
export CUDA_VISIBLE_DEVICES=""
poetry run pytest
```

#### GPU Mode
```bash
# Run GPU-specific tests
poetry run pytest -m gpu
```

#### Mock Mode
```bash
# Use mock embeddings (no GPU required)
export USE_MOCK_EMBEDDINGS=true
poetry run pytest
```

### Coverage Reports
```bash
# Generate coverage report
poetry run pytest --cov=packages --cov-report=html

# View coverage in terminal
poetry run pytest --cov=packages --cov-report=term-missing

# Open HTML report
open htmlcov/index.html
```

## Test Structure

### Directory Organization
```
tests/
├── conftest.py          # Shared fixtures and configuration
├── unit/                # Unit tests
│   ├── test_extract_chunks.py
│   ├── test_embedding_service.py
│   └── test_model_manager.py
├── integration/         # Integration tests
│   ├── test_search_api.py
│   ├── test_job_processing.py
│   └── test_qdrant_integration.py
├── e2e/                 # End-to-end tests
│   └── test_full_pipeline.py
└── fixtures/            # Test data and fixtures
    ├── documents/
    └── embeddings/
```

### Test File Naming
- Unit tests: `test_<module_name>.py`
- Integration tests: `test_<feature>_integration.py`
- E2E tests: `test_<workflow>_e2e.py`

## Writing Tests

### Unit Test Example
```python
"""Unit tests for document chunking"""
import pytest
from vecpipe.extract_chunks import chunk_text

class TestChunkText:
    """Test the chunk_text function"""
    
    def test_chunk_text_basic(self):
        """Test basic text chunking"""
        text = "This is a test. " * 100
        chunks = chunk_text(text, chunk_size=50, overlap=10)
        
        assert len(chunks) > 1
        assert all(len(chunk) <= 50 for chunk in chunks)
        
    def test_chunk_text_empty(self):
        """Test chunking empty text"""
        chunks = chunk_text("", chunk_size=50, overlap=10)
        assert chunks == []
        
    @pytest.mark.parametrize("chunk_size,overlap", [
        (100, 20),
        (500, 50),
        (1000, 100),
    ])
    def test_chunk_text_sizes(self, chunk_size, overlap):
        """Test various chunk sizes"""
        text = "word " * 1000
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        assert all(len(chunk) <= chunk_size for chunk in chunks)
```

### Integration Test Example
```python
"""Integration tests for search API"""
import pytest
from httpx import AsyncClient
from vecpipe.search_api import app

@pytest.mark.asyncio
async def test_search_integration():
    """Test search API with real Qdrant"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        # Perform search
        response = await client.get("/search", params={
            "q": "machine learning",
            "k": 5
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) <= 5
```

### Using Fixtures
```python
"""Fixtures for testing"""
import pytest
from pathlib import Path

@pytest.fixture
def sample_pdf():
    """Provide sample PDF for testing"""
    return Path("tests/fixtures/documents/sample.pdf")

@pytest.fixture
def mock_embedding_service(monkeypatch):
    """Mock embedding service for tests"""
    def mock_embed(text):
        return [0.1] * 384  # Mock 384-dim embedding
    
    monkeypatch.setattr(
        "webui.embedding_service.EmbeddingService.embed",
        mock_embed
    )

def test_with_fixtures(sample_pdf, mock_embedding_service):
    """Test using fixtures"""
    assert sample_pdf.exists()
    # Test continues...
```

## Testing Best Practices

### 1. Use Descriptive Names
```python
# Good
def test_search_returns_relevant_results_for_exact_match():
    pass

# Bad
def test_search_1():
    pass
```

### 2. Follow AAA Pattern
```python
def test_user_creation():
    # Arrange
    user_data = {"username": "test", "email": "test@example.com"}
    
    # Act
    user = create_user(user_data)
    
    # Assert
    assert user.username == "test"
    assert user.email == "test@example.com"
```

### 3. Test Edge Cases
```python
def test_chunk_text_edge_cases():
    """Test edge cases for text chunking"""
    # Empty text
    assert chunk_text("") == []
    
    # Single word
    assert len(chunk_text("word")) == 1
    
    # Very long word
    long_word = "a" * 1000
    chunks = chunk_text(long_word, chunk_size=100)
    assert len(chunks) > 1
```

### 4. Use Mocks Appropriately
```python
@patch('requests.get')
def test_external_api_call(mock_get):
    """Test with mocked external dependency"""
    mock_get.return_value.json.return_value = {"status": "ok"}
    
    result = check_external_service()
    assert result == "ok"
    mock_get.assert_called_once()
```

## Test Categories

### Unit Tests
Focus on individual functions and classes:
- Document parsing
- Text chunking
- Embedding generation (mocked)
- Database operations
- Authentication logic

### Integration Tests
Test component interactions:
- API endpoint functionality
- Database transactions
- Qdrant operations
- Authentication flow
- Job processing pipeline

### End-to-End Tests
Test complete workflows:
- Upload documents → Process → Search
- User registration → Login → Create job
- Collection management lifecycle

#### Refactoring Validation E2E Test
We maintain a special E2E test that captures the exact current behavior of the system. This test is critical for ensuring no regressions are introduced during the refactoring initiative.

**Location**: `tests/e2e/test_refactoring_validation.py`

**Purpose**: This test serves as a "golden master" that validates the entire document processing and search pipeline via the public API.

**Running the test with Docker Compose**:
```bash
# Start the application with docker compose
docker compose up -d

# Set the API endpoint (defaults to http://localhost:8080)
export API_BASE_URL=http://localhost:8080

# Run the E2E validation test
poetry run pytest tests/e2e/test_refactoring_validation.py -v

# Or run with a custom endpoint
API_BASE_URL=http://localhost:3000 poetry run pytest tests/e2e/test_refactoring_validation.py
```

**What the test validates**:
1. Creates a job with test documents from `test_data/` directory
2. Waits for the job to complete (with timeout)
3. Performs a search to verify embeddings were created correctly
4. Cleans up all created resources (job and Qdrant collection)

**Requirements**:
- Running instance of the application (via docker compose or locally)
- Test data files in `test_data/` directory
- Network access to the API endpoint

**Note**: This test makes real HTTP calls to a running instance and is intentionally not mocked. It should pass identically before and after any refactoring.

**Important**: When running with Docker Compose, the test uses `/mnt/docs` directory which contains production documents. This can make the test take 1-2 minutes to complete. For faster testing in CI/CD, consider:
- Creating a smaller test dataset mounted at a specific path
- Using the `test_data/` directory when running locally without Docker
- Setting a job size limit for test environments

**CI/CD Integration**:
The E2E test is marked with `@pytest.mark.e2e` and automatically skips if the service is not available. To exclude E2E tests in CI:
```bash
# Run all tests except E2E
make test-ci
# or
pytest tests -v -m "not e2e"
```

To run only E2E tests when services are available:
```bash
# Start services first
docker compose up -d
# Run E2E tests
make test-e2e
# or
pytest tests -v -m e2e
```

## Continuous Integration

### GitHub Actions Workflow
```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: |
          pip install poetry
          poetry install
      - name: Run tests (excluding E2E)
        run: |
          make test-ci
```

## Performance Testing

### Benchmark Tests
```python
import pytest
import time

@pytest.mark.benchmark
def test_embedding_performance(benchmark):
    """Benchmark embedding generation"""
    text = "Sample text" * 100
    
    def embed():
        return generate_embedding(text)
    
    result = benchmark(embed)
    assert len(result) == 384
```

### Load Testing
```python
@pytest.mark.load
async def test_concurrent_searches():
    """Test system under load"""
    async def search():
        async with AsyncClient() as client:
            return await client.get("/search?q=test")
    
    # Run 100 concurrent searches
    tasks = [search() for _ in range(100)]
    results = await asyncio.gather(*tasks)
    
    assert all(r.status_code == 200 for r in results)
```

## Debugging Tests

### Verbose Output
```bash
# Show print statements
poetry run pytest -s

# Show full assertion details
poetry run pytest -vv

# Stop on first failure
poetry run pytest -x

# Drop into debugger on failure
poetry run pytest --pdb
```

### Logging in Tests
```python
import logging

def test_with_logging(caplog):
    """Test with captured logs"""
    with caplog.at_level(logging.INFO):
        process_document("test.pdf")
    
    assert "Processing document" in caplog.text
```

## Test Data Management

### Fixtures Directory
```
tests/fixtures/
├── documents/
│   ├── sample.pdf
│   ├── sample.docx
│   └── sample.txt
├── embeddings/
│   └── mock_embeddings.json
└── responses/
    └── qdrant_responses.json
```

### Factory Pattern
```python
# tests/factories.py
import factory
from shared.database import User

class UserFactory(factory.Factory):
    class Meta:
        model = User
    
    username = factory.Sequence(lambda n: f"user{n}")
    email = factory.LazyAttribute(lambda obj: f"{obj.username}@example.com")
    
# Usage in tests
def test_user_creation():
    user = UserFactory()
    assert user.username.startswith("user")
```

## Coverage Goals

### Target Coverage
- Overall: 80%+
- Critical paths: 90%+
- New features: 85%+

### Coverage Exclusions
```python
# pragma: no cover - for untestable code
if __name__ == "__main__":  # pragma: no cover
    main()
```

### Monitoring Coverage
```bash
# Check coverage locally
make test-coverage

# Generate badge for README
coverage-badge -o coverage.svg
```

## Troubleshooting

### Common Issues

1. **GPU Tests Failing**
   ```bash
   # Run without GPU
   export CUDA_VISIBLE_DEVICES=""
   poetry run pytest
   ```

2. **Database Lock Errors**
   ```bash
   # Use separate test database
   export WEBUI_DB=data/test.db
   rm data/test.db
   poetry run pytest
   ```

3. **Import Errors**
   ```bash
   # Ensure proper Python path
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   poetry run pytest
   ```

4. **Async Test Issues**
   ```python
   # Use pytest-asyncio
   @pytest.mark.asyncio
   async def test_async_function():
       result = await async_function()
       assert result is not None
   ```

## Next Steps

1. **Expand Test Coverage**: Focus on untested modules
2. **Add Property-Based Tests**: Use hypothesis for edge cases
3. **Implement Mutation Testing**: Ensure test quality
4. **Create Test Templates**: Standardize test patterns
5. **Add Visual Regression Tests**: For frontend components

Remember: Good tests are an investment in code quality and developer confidence. Write tests that you'll thank yourself for later!