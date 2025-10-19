# Testing Guide for Semantik

## Overview

This guide covers the testing philosophy, practices, and procedures for the Semantik codebase. We maintain a comprehensive test suite to ensure reliability, performance, and maintainability across our collection-centric architecture.

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
uv sync --frozen

# Verify pytest is available
uv run pytest --version
```

### Environment Variables
The Docker setup wizard now offers to generate a `.env.test` file with host overrides; accept the prompt to reuse
those settings. If you skipped that step, create one manually for test-specific configuration:
```bash
# Use mock embeddings to avoid GPU requirements
USE_MOCK_EMBEDDINGS=true

# Flag test environment (disables Prometheus server, enables test hooks)
TESTING=true

# Use test database (PostgreSQL required)
DATABASE_URL=postgresql://test_user:test_pass@localhost:5432/semantik_test

# Disable authentication for API tests
DISABLE_AUTH=true

# Use local test Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Disable rate limiting for tests
DISABLE_RATE_LIMITING=true

# Redis configuration for WebSocket testing
REDIS_URL=redis://localhost:6379/0

# JWT secret for auth tests
JWT_SECRET_KEY=test-secret-key-for-testing-only
```

### PostgreSQL Test Database Setup

We now ship a dedicated Docker service for the test database. Bring it up with the
new `testing` profile before running suites that hit FastAPI endpoints:

```bash
# Start the disposable test database (uses credentials from .env.test)
docker compose --profile testing up -d postgres_test
```

If you prefer a local Postgres instance, you can still create the database by hand:

```bash
# Create test database
createdb semantik_test

# Run migrations on test database
DATABASE_URL=postgresql://test_user:test_pass@localhost:5432/semantik_test uv run alembic upgrade head

# Clean the database before running the test suite (destroys all data!)
TEST_DATABASE_URL=postgresql://test_user:test_pass@localhost:5432/semantik_test \
CONFIRM_TEST_DB_RESET=true \
uv run python scripts/reset_test_db.py
```

## Running Tests

### Basic Test Execution
```bash
# Run all tests
make test

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_auth.py

# Run specific test function
uv run pytest tests/test_auth.py::test_login_success

# Run tests matching pattern
uv run pytest -k "rerank"
```

### Test Modes

#### CPU Mode (Default)
```bash
# Force CPU mode for testing
export CUDA_VISIBLE_DEVICES=""
uv run pytest
```

#### GPU Mode
```bash
# Run GPU-specific tests
uv run pytest -m gpu
```

#### Mock Mode
```bash
# Use mock embeddings (no GPU required)
export USE_MOCK_EMBEDDINGS=true
uv run pytest
```

### Coverage Reports
```bash
# Generate coverage report
uv run pytest --cov=packages --cov-report=html

# View coverage in terminal
uv run pytest --cov=packages --cov-report=term-missing

# Open HTML report
open htmlcov/index.html
```

## Test Structure

### Backend Test Organization
```
tests/
├── conftest.py                    # Root test configuration
├── unit/                          # Unit tests
│   ├── test_collection_service.py
│   ├── test_collection_repository.py
│   ├── test_operation_service.py
│   ├── test_operation_repository.py
│   ├── test_document_repository.py
│   ├── test_extract_chunks.py
│   ├── test_model_manager.py
│   └── test_websocket_manager.py
├── integration/                   # Integration tests
│   ├── test_search_api_integration.py
│   ├── test_collection_persistence.py
│   ├── test_websocket_redis_integration.py
│   └── test_embedding_gpu_memory.py
├── e2e/                          # End-to-end tests
│   ├── test_collection_deletion_e2e.py
│   ├── test_websocket_integration.py
│   ├── test_websocket_reindex.py
│   └── test_refactoring_validation.py
├── webui/                        # WebUI specific tests
│   ├── conftest.py               # WebUI test configuration
│   ├── api/v2/                   # API endpoint tests
│   │   ├── test_collections.py
│   │   ├── test_operations.py
│   │   ├── test_documents.py
│   │   └── test_search.py
│   └── services/                 # Service layer tests
│       ├── test_collection_service.py
│       └── test_search_service.py
└── api/                          # API integration tests
    └── test_collection_deletion_endpoint.py
```

### Frontend Test Organization
```
apps/webui-react/
├── src/
│   ├── components/__tests__/     # Component tests
│   │   ├── CollectionCard.test.tsx
│   │   ├── CollectionOperations.test.tsx
│   │   ├── CreateCollectionModal.test.tsx
│   │   └── SearchInterface.test.tsx
│   ├── hooks/__tests__/          # Hook tests
│   │   ├── useCollections.test.tsx
│   │   ├── useCollectionOperations.test.tsx
│   │   └── useWebSocket.error.test.tsx
│   ├── stores/__tests__/         # Store tests
│   │   ├── authStore.test.ts
│   │   └── searchStore.test.ts
│   └── utils/__tests__/          # Utility tests
│       └── errorUtils.test.ts
└── tests/                        # Test utilities
    └── utils/
        └── test-utils.tsx
```

### Test File Naming
- Unit tests: `test_<module_name>.py`
- Integration tests: `test_<feature>_integration.py`
- E2E tests: `test_<workflow>_e2e.py`

## Writing Tests

### Backend Unit Test Examples

#### Testing Collection Service
```python
"""Unit tests for collection service"""
import pytest
from unittest.mock import AsyncMock, Mock
from datetime import datetime, UTC

from packages.shared.database.models import Collection, CollectionStatus, OperationType
from packages.shared.database.exceptions import EntityAlreadyExistsError, InvalidStateError
from packages.webui.services.collection_service import CollectionService

class TestCollectionService:
    """Test CollectionService implementation"""
    
    @pytest.fixture
    def mock_session(self):
        """Create a mock AsyncSession"""
        session = AsyncMock()
        session.commit = AsyncMock()
        return session
    
    @pytest.fixture
    def collection_service(self, mock_session, mock_collection_repo, mock_operation_repo):
        """Create CollectionService with mocked dependencies"""
        return CollectionService(
            db_session=mock_session,
            collection_repo=mock_collection_repo,
            operation_repo=mock_operation_repo,
            document_repo=mock_document_repo,
        )
    
    async def test_create_collection_success(self, collection_service, mock_collection_repo):
        """Test successful collection creation"""
        # Arrange
        collection_data = {
            "name": "Test Collection",
            "description": "A test collection",
            "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
        }
        mock_collection_repo.get_by_name.return_value = None
        
        # Act
        result = await collection_service.create_collection(
            user_id=1,
            **collection_data
        )
        
        # Assert
        assert result is not None
        mock_collection_repo.create.assert_called_once()
```

#### Testing Operations Repository
```python
"""Unit tests for operation repository"""
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from packages.shared.database.models import Operation, OperationType, OperationStatus
from packages.webui.repositories.operation_repository import OperationRepository

class TestOperationRepository:
    """Test OperationRepository implementation"""
    
    @pytest.fixture
    async def operation_repo(self, async_session: AsyncSession):
        """Create OperationRepository instance"""
        return OperationRepository(async_session)
    
    async def test_create_operation(self, operation_repo, test_collection):
        """Test creating an operation"""
        # Create operation
        operation = await operation_repo.create(
            collection_id=test_collection.id,
            type=OperationType.INDEX,
            parameters={"batch_size": 100}
        )
        
        assert operation.id is not None
        assert operation.type == OperationType.INDEX
        assert operation.status == OperationStatus.PENDING
        assert operation.collection_id == test_collection.id
```

### Integration Test Examples

#### API Endpoint Integration Test
```python
"""Integration tests for collections API v2"""
import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from packages.shared.database.models import User, Collection

class TestCollectionsAPIV2:
    """Test collections API v2 endpoints"""
    
    @pytest.mark.asyncio
    async def test_create_collection(self, async_client: AsyncClient, test_user: User):
        """Test creating a collection via API"""
        response = await async_client.post(
            "/api/v2/collections",
            json={
                "name": "Test Collection",
                "description": "Integration test collection",
                "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
                "chunk_size": 512,
                "chunk_overlap": 50,
            }
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Test Collection"
        assert data["status"] == "empty"
        assert data["owner_id"] == test_user.id
```

#### WebSocket Integration Test
```python
"""Integration tests for WebSocket operations"""
import pytest
import asyncio
from fastapi.testclient import TestClient
from packages.webui.main import app

class TestWebSocketIntegration:
    """Test WebSocket integration with operations"""
    
    @pytest.mark.asyncio
    async def test_operation_progress_updates(self, test_client: TestClient, test_collection):
        """Test receiving operation progress via WebSocket"""
        with test_client.websocket_connect("/ws") as websocket:
            # Start an operation
            response = test_client.post(
                f"/api/v2/collections/{test_collection.id}/reindex"
            )
            operation_id = response.json()["id"]
            
            # Listen for updates
            updates = []
            for _ in range(5):  # Collect up to 5 updates
                data = websocket.receive_json()
                if data["type"] == "operation_update":
                    updates.append(data)
                    if data["data"]["status"] in ["completed", "failed"]:
                        break
            
            assert len(updates) > 0
            assert updates[-1]["data"]["status"] in ["completed", "failed"]
```

### Common Test Fixtures

#### Backend Fixtures (conftest.py)
```python
"""Common fixtures for backend tests"""
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from fastapi.testclient import TestClient

from packages.shared.database import Base, User, Collection
from packages.webui.main import app

@pytest_asyncio.fixture
async def async_session():
    """Create async database session for tests"""
    engine = create_async_engine(
        "postgresql+asyncpg://test_user:test_pass@localhost:5432/semantik_test"
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    async with AsyncSession(engine) as session:
        yield session
        await session.rollback()

@pytest.fixture
def test_user(async_session: AsyncSession) -> User:
    """Create a test user"""
    user = User(
        username="testuser",
        email="test@example.com",
        hashed_password="hashed_password"
    )
    async_session.add(user)
    await async_session.commit()
    return user

@pytest.fixture
def test_collection(async_session: AsyncSession, test_user: User) -> Collection:
    """Create a test collection"""
    collection = Collection(
        name="Test Collection",
        owner_id=test_user.id,
        embedding_model="Qwen/Qwen3-Embedding-0.6B",
        status=CollectionStatus.READY,
        vector_store_name="test_collection_vectors"
    )
    async_session.add(collection)
    await async_session.commit()
    return collection

@pytest.fixture
def mock_celery_task(monkeypatch):
    """Mock Celery task execution"""
    from unittest.mock import MagicMock
    
    mock_task = MagicMock()
    mock_task.delay.return_value.id = "mock-task-id"
    
    monkeypatch.setattr(
        "packages.webui.tasks.index_collection",
        mock_task
    )
    return mock_task
```

#### Frontend Test Utilities
```typescript
// apps/webui-react/tests/utils/test-utils.tsx
import { render, RenderOptions } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { vi } from 'vitest'

// Create a custom render function that includes providers
export function renderWithProviders(
  ui: React.ReactElement,
  options?: RenderOptions
) {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  })
  
  return render(
    <QueryClientProvider client={queryClient}>
      {ui}
    </QueryClientProvider>,
    options
  )
}

// Mock collection factory
export function createMockCollection(overrides = {}): Collection {
  return {
    id: 'test-collection-id',
    name: 'Test Collection',
    status: 'ready',
    document_count: 10,
    vector_count: 100,
    created_at: new Date().toISOString(),
    ...overrides,
  }
}
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
- API endpoint functionality with collections
- Database transactions and rollbacks
- Qdrant vector operations
- Authentication and authorization flows
- Operation processing pipeline
- WebSocket message flow
- Redis pub/sub for real-time updates

### End-to-End Tests
Test complete workflows:
- Create collection → Add documents → Index → Search
- User registration → Login → Create collection → Manage operations
- Collection lifecycle: Create → Index → Search → Reindex → Delete
- Real-time operation tracking via WebSocket
- Multi-user collection sharing and permissions

#### Collection-Centric E2E Tests

##### WebSocket Operation Tracking
**Location**: `tests/e2e/test_websocket_integration.py`

```python
@pytest.mark.e2e
async def test_collection_indexing_with_websocket():
    """Test complete indexing flow with WebSocket updates"""
    async with AsyncClient(base_url=API_BASE_URL) as client:
        # Create collection
        collection_response = await client.post(
            "/api/v2/collections",
            json={"name": "E2E Test Collection", "embedding_model": "Qwen/Qwen3-Embedding-0.6B"}
        )
        collection_id = collection_response.json()["id"]
        
        # Connect WebSocket
        async with client.websocket_connect("/ws") as websocket:
            # Add documents
            add_response = await client.post(
                f"/api/v2/collections/{collection_id}/add",
                json={"path": "/test/data", "recursive": True}
            )
            operation_id = add_response.json()["id"]
            
            # Track operation progress
            completed = False
            while not completed:
                msg = await websocket.receive_json()
                if msg["type"] == "operation_update" and msg["data"]["id"] == operation_id:
                    if msg["data"]["status"] == "completed":
                        completed = True
```

##### Collection Deletion E2E Test
**Location**: `tests/e2e/test_collection_deletion_e2e.py`

**Purpose**: Validates complete collection deletion including:
1. Removing collection from PostgreSQL
2. Deleting vectors from Qdrant
3. Canceling any running operations
4. Cleaning up orphaned documents

**Running E2E tests**:
```bash
# Start all services
make docker-up

# Run all E2E tests
make test-e2e

# Run specific E2E test
uv run pytest tests/e2e/test_collection_deletion_e2e.py -v

# With custom endpoint
API_BASE_URL=http://localhost:8000 uv run pytest tests/e2e/ -v
```

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

### Test Commands

```bash
# Backend tests
make test              # Run all tests
make test-ci           # Run tests excluding E2E (for CI)
make test-e2e          # Run only E2E tests
make test-coverage     # Generate coverage report

# Frontend tests
make frontend-test     # Run all frontend tests
cd apps/webui-react && npm run test:coverage  # With coverage

# Full test suite
make check            # Format, lint, and test everything
```

### GitHub Actions Workflows

#### Backend Tests (`.github/workflows/test-backend.yml`)
```yaml
name: Backend Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: test_pass
          POSTGRES_DB: semantik_test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install uv
          uv sync
      - name: Run migrations
        run: |
          uv run alembic upgrade head
        env:
          DATABASE_URL: postgresql://postgres:test_pass@localhost:5432/semantik_test
      - name: Run tests
        run: |
          make test-ci
        env:
          DATABASE_URL: postgresql://postgres:test_pass@localhost:5432/semantik_test
          REDIS_URL: redis://localhost:6379/0
```

#### Frontend Tests (`.github/workflows/test-frontend.yml`)
```yaml
name: Frontend Tests
on: 
  push:
    paths:
      - 'apps/webui-react/**'
      - '.github/workflows/test-frontend.yml'
  pull_request:
    paths:
      - 'apps/webui-react/**'

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        node-version: [18.x, 20.x]
    
    steps:
      - uses: actions/checkout@v4
      - name: Use Node.js ${{ matrix.node-version }}
        uses: actions/setup-node@v4
        with:
          node-version: ${{ matrix.node-version }}
      - name: Install dependencies
        working-directory: apps/webui-react
        run: npm ci
      - name: Run linter
        working-directory: apps/webui-react
        run: npm run lint
      - name: Run tests
        working-directory: apps/webui-react
        run: npm test -- --coverage
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          directory: apps/webui-react/coverage
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: backend-tests
        name: Backend Tests
        entry: uv run pytest tests/unit -x
        language: system
        pass_filenames: false
        files: \.(py)$
      
      - id: frontend-tests
        name: Frontend Tests
        entry: bash -c "cd apps/webui-react && npm test -- --run --passWithNoTests"
        language: system
        pass_filenames: false
        files: \.(tsx?|jsx?)$
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
uv run pytest -s

# Show full assertion details
uv run pytest -vv

# Stop on first failure
uv run pytest -x

# Drop into debugger on failure
uv run pytest --pdb
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

### Test Data Factories

#### Backend Factory Pattern
```python
# tests/factories.py
import factory
from datetime import datetime, UTC
from packages.shared.database.models import User, Collection, Operation, Document

class UserFactory(factory.Factory):
    class Meta:
        model = User
    
    username = factory.Sequence(lambda n: f"user{n}")
    email = factory.LazyAttribute(lambda obj: f"{obj.username}@example.com")
    is_active = True

class CollectionFactory(factory.Factory):
    class Meta:
        model = Collection
    
    name = factory.Sequence(lambda n: f"Collection {n}")
    owner_id = factory.SubFactory(UserFactory)
    embedding_model = "Qwen/Qwen3-Embedding-0.6B"
    status = "ready"
    vector_store_name = factory.LazyAttribute(lambda obj: f"{obj.name.lower().replace(' ', '_')}_vectors")
    created_at = factory.LazyFunction(lambda: datetime.now(UTC))

class OperationFactory(factory.Factory):
    class Meta:
        model = Operation
    
    collection_id = factory.SubFactory(CollectionFactory)
    type = "index"
    status = "pending"
    parameters = {"batch_size": 100}
    created_at = factory.LazyFunction(lambda: datetime.now(UTC))

# Usage in tests
def test_collection_with_operations():
    collection = CollectionFactory()
    operations = OperationFactory.create_batch(3, collection_id=collection.id)
    assert len(operations) == 3
    assert all(op.collection_id == collection.id for op in operations)
```

#### Frontend Mock Data Builders
```typescript
// tests/builders/collection.builder.ts
export class CollectionBuilder {
  private collection: Partial<Collection> = {
    id: 'default-id',
    name: 'Default Collection',
    status: 'ready',
    document_count: 0,
    vector_count: 0,
    created_at: new Date().toISOString(),
  }
  
  withId(id: string): this {
    this.collection.id = id
    return this
  }
  
  withName(name: string): this {
    this.collection.name = name
    return this
  }
  
  withDocuments(count: number): this {
    this.collection.document_count = count
    this.collection.vector_count = count * 10 // Assume 10 chunks per doc
    return this
  }
  
  withStatus(status: CollectionStatus): this {
    this.collection.status = status
    return this
  }
  
  build(): Collection {
    return this.collection as Collection
  }
}

// Usage
const testCollection = new CollectionBuilder()
  .withName('Test Collection')
  .withDocuments(50)
  .withStatus('indexing')
  .build()
```

## Test Coverage

### Coverage Goals
- Overall: 80%+ (backend), 80%+ (frontend)
- Critical paths: 90%+ (auth, collections, operations)
- New features: 85%+ before merging
- API endpoints: 95%+ coverage required

### Backend Coverage
```bash
# Generate coverage report
uv run pytest --cov=packages --cov-report=html --cov-report=term-missing

# View coverage in browser
open htmlcov/index.html

# Coverage by package
uv run pytest --cov=packages.webui --cov=packages.vecpipe --cov=packages.shared
```

### Frontend Coverage
```bash
# Generate coverage report
cd apps/webui-react
npm run test:coverage

# View coverage report
open coverage/lcov-report/index.html
```

### Coverage Configuration

#### Backend (.coveragerc)
```ini
[run]
source = packages
omit = 
    */tests/*
    */migrations/*
    */__pycache__/*
    */venv/*
    */.venv/*

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
    if TYPE_CHECKING:
```

#### Frontend (vite.config.ts)
```typescript
export default defineConfig({
  test: {
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      exclude: [
        'node_modules/',
        'src/tests/',
        '**/*.d.ts',
        '**/*.config.*',
        '**/mockServiceWorker.js',
      ],
    },
  },
})
```

## Troubleshooting

### Common Backend Test Issues

1. **Async Test Failures**
   ```python
   # Ensure proper async test setup
   @pytest.mark.asyncio
   async def test_async_operation():
       # Use async fixtures
       async with AsyncSession() as session:
           result = await operation
           assert result is not None
   ```

2. **Database Migration Errors**
   ```bash
   # Reset test database
   dropdb semantik_test
   createdb semantik_test
   DATABASE_URL=postgresql://test_user:test_pass@localhost:5432/semantik_test \
     uv run alembic upgrade head
   ```

3. **WebSocket Test Timeouts**
   ```python
   # Increase timeout for WebSocket tests
   @pytest.mark.timeout(30)  # 30 second timeout
   async def test_websocket_operation():
       # Test code
   ```

4. **Mock Service Issues**
   ```python
   # Ensure mocks are properly reset
   @pytest.fixture(autouse=True)
   def reset_mocks():
       yield
       mock.reset_mock()
   ```

### Common Frontend Test Issues

1. **Act() Warnings**
   ```typescript
   // Wrap state updates in act()
   await act(async () => {
     await user.click(button)
   })
   ```

2. **Timer Issues**
   ```typescript
   // Use fake timers for time-dependent tests
   beforeEach(() => {
     vi.useFakeTimers()
   })
   
   afterEach(() => {
     vi.runOnlyPendingTimers()
     vi.useRealTimers()
   })
   ```

3. **Unhandled Promise Rejections**
   ```typescript
   // Always handle async errors in tests
   it('handles errors gracefully', async () => {
     const error = new Error('Test error')
     mockApi.create.mockRejectedValueOnce(error)
     
     render(<Component />)
     // Test error handling
   })
   ```

## Frontend Testing

### Frontend Test Structure

Frontend tests are organized by feature and test type, co-located with the code they test:

```
apps/webui-react/src/
├── components/__tests__/          # Component tests
│   ├── *.test.tsx                # Standard component tests
│   ├── *.network.test.tsx        # Network/API interaction tests
│   ├── *.websocket.test.tsx      # WebSocket specific tests
│   └── *.validation.test.tsx     # Input validation tests
├── hooks/__tests__/              # Custom hook tests
├── stores/__tests__/             # Zustand store tests
└── utils/__tests__/              # Utility function tests
```

### Component Test Coverage

Comprehensive test coverage for collection and operation components:

| Component | Test Files | Coverage Areas |
|-----------|------------|----------------|
| CollectionCard | CollectionCard.test.tsx | Display, interactions, status badges |
| CreateCollectionModal | CreateCollectionModal.test.tsx<br>CreateCollectionModal.network.test.tsx | Form validation, API calls, error handling |
| CollectionOperations | CollectionOperations.test.tsx<br>CollectionOperations.websocket.test.tsx | Operation tracking, WebSocket updates |
| CollectionsDashboard | CollectionsDashboard.test.tsx<br>CollectionsDashboard.network.test.tsx | Collection listing, filtering, pagination |
| SearchInterface | SearchInterface.test.tsx<br>SearchInterface.reranking.test.tsx | Search functionality, reranking options |

### Running Frontend Tests

```bash
# Change to frontend directory
cd apps/webui-react

# Run all tests
npm test

# Run tests in watch mode
npm run test:watch

# Run with coverage
npm run test:coverage

# Run specific test file
npm test -- CreateCollectionModal

# Using the test script
./scripts/test-frontend.sh --collections
./scripts/test-frontend.sh --coverage --collections
./scripts/test-frontend.sh --watch
./scripts/test-frontend.sh --file CreateCollectionModal
```

### Frontend Test Guidelines

#### Test Patterns

1. **Component Testing with Vitest**
   ```typescript
   import { describe, it, expect, vi, beforeEach } from 'vitest'
   import { render, screen, waitFor } from '@/tests/utils/test-utils'
   import userEvent from '@testing-library/user-event'
   ```

2. **Mock Management**
   ```typescript
   // Mock hooks
   vi.mock('@/hooks/useCollections', () => ({
     useCollections: vi.fn(() => ({
       data: mockCollections,
       isLoading: false,
       error: null,
       refetch: vi.fn(),
     })),
   }))
   
   // Mock API calls
   vi.mock('@/api/collectionsV2', () => ({
     collectionsV2Api: {
       create: vi.fn(),
       update: vi.fn(),
       delete: vi.fn(),
     },
   }))
   ```

3. **WebSocket Testing**
   ```typescript
   // Mock WebSocket connection
   const mockWebSocket = {
     send: vi.fn(),
     close: vi.fn(),
     addEventListener: vi.fn(),
     removeEventListener: vi.fn(),
   }
   
   beforeEach(() => {
     global.WebSocket = vi.fn(() => mockWebSocket)
   })
   ```

4. **Async Testing Patterns**
   ```typescript
   it('should handle async operations', async () => {
     const user = userEvent.setup()
     render(<CreateCollectionModal />)
     
     await user.type(screen.getByLabelText('Name'), 'Test Collection')
     await user.click(screen.getByRole('button', { name: 'Create' }))
     
     await waitFor(() => {
       expect(mockCreateCollection).toHaveBeenCalledWith({
         name: 'Test Collection',
       })
     })
   })
   ```

### CI/CD Integration

The project includes GitHub Actions workflows for automated testing:

1. **`frontend-tests.yml`**: Runs on every push/PR affecting frontend code
   - Linting
   - Unit tests with multiple Node.js versions
   - Coverage reports
   - Build verification

2. **`test-all.yml`**: Comprehensive test suite
   - Backend tests with PostgreSQL and Redis
   - Frontend tests in parallel groups
   - Integration tests
   - Linting and formatting checks

3. **`pr-checks.yml`**: Pull request specific checks
   - Verify test coverage for changed files
   - Run affected tests
   - Generate coverage reports

### Test Script Usage

A convenience script is provided for running frontend tests:

```bash
# Show help
./scripts/test-frontend.sh --help

# Run collection component tests (default)
./scripts/test-frontend.sh

# Run all frontend tests
./scripts/test-frontend.sh --all

# Run with coverage
./scripts/test-frontend.sh --coverage --collections

# Watch mode for development
./scripts/test-frontend.sh --watch

# Run specific test file
./scripts/test-frontend.sh --file CreateCollectionModal
```

## Best Practices

### Testing Collections and Operations

1. **Always Test State Transitions**
   ```python
   async def test_collection_state_transitions():
       # Test: empty -> indexing -> ready
       # Test: ready -> reindexing -> ready
       # Test: any -> deleting -> (deleted)
   ```

2. **Test Concurrent Operations**
   ```python
   async def test_concurrent_operations_prevented():
       # Ensure only one operation per collection at a time
       operation1 = await start_indexing(collection_id)
       operation2 = await start_reindexing(collection_id)
       assert operation2 is None  # Should be rejected
   ```

3. **Test WebSocket Updates**
   ```python
   async def test_operation_progress_broadcast():
       # Verify all connected clients receive updates
       # Test reconnection handling
       # Test message ordering
   ```

### Mocking External Services

1. **Qdrant Mocking**
   ```python
   @pytest.fixture
   def mock_qdrant_client():
       client = AsyncMock()
       client.create_collection.return_value = True
       client.search.return_value = SearchResult(...)
       return client
   ```

2. **Redis Mocking**
   ```python
   @pytest.fixture
   def mock_redis():
       redis = AsyncMock()
       redis.publish.return_value = 1  # Number of subscribers
       return redis
   ```

3. **Celery Task Mocking**
   ```python
   @pytest.fixture
   def mock_celery_app(monkeypatch):
       mock_task = MagicMock()
       mock_task.delay.return_value.id = "task-123"
       monkeypatch.setattr("celery.current_app.send_task", mock_task)
       return mock_task
   ```

## Next Steps

1. **Improve WebSocket Test Coverage**: Add tests for connection drops, reconnection, and message ordering
2. **Add Performance Benchmarks**: Track operation processing times and search latencies
3. **Implement Contract Testing**: Ensure API compatibility between services
4. **Add Load Testing**: Test system behavior under concurrent operations
5. **Create Integration Test Suites**: For complete user workflows
6. **Add Security Testing**: Test authorization, rate limiting, and input validation

Remember: Tests are living documentation. Keep them clean, focused, and meaningful. A well-tested codebase is a maintainable codebase!
