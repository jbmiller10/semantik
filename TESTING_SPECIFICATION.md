# Semantik Testing Architecture & Strategies Specification

## Table of Contents
1. [Overview](#overview)
2. [Backend Testing](#backend-testing)
3. [Frontend Testing](#frontend-testing)
4. [Test Coverage](#test-coverage)
5. [Test Data Management](#test-data-management)
6. [CI/CD Testing](#cicd-testing)
7. [Testing Standards & Best Practices](#testing-standards--best-practices)

## Overview

The Semantik codebase implements a comprehensive testing strategy across both backend (Python/FastAPI) and frontend (React/TypeScript) components. The testing architecture emphasizes:

- **Test Isolation**: Each test is completely independent with proper setup and teardown
- **Mock-First Approach**: External dependencies are mocked to ensure fast, reliable tests
- **Coverage Requirements**: Critical paths must have comprehensive test coverage
- **Collection-Centric Architecture**: All tests follow the new collection-centric patterns, avoiding legacy "job" terminology

### Testing Philosophy
- Tests serve as living documentation of system behavior
- Every bug fix must include a regression test
- Tests follow the AAA pattern (Arrange, Act, Assert)
- Integration tests validate API contracts; unit tests validate business logic

## Backend Testing

### Test Suite Structure

```
tests/
├── api/                    # API endpoint tests
├── application/           # Application use case tests
├── chunking/             # Chunking strategy tests
├── database/             # Database and migration tests
├── domain/               # Domain model tests
├── e2e/                  # End-to-end tests
├── fixtures/             # Shared test fixtures
├── integration/          # Integration tests
├── performance/          # Performance benchmarks
├── security/             # Security tests
├── streaming/            # Streaming functionality tests
├── unit/                 # Unit tests
├── websocket/            # WebSocket tests
└── webui/                # WebUI specific tests
    ├── api/v2/          # V2 API tests
    └── services/        # Service layer tests
```

### Testing Framework & Tools

#### Core Dependencies
- **pytest**: Primary testing framework (v8.0.0+)
- **pytest-asyncio**: Async test support (v0.23.0+)
- **pytest-cov**: Coverage reporting (v4.1.0+)
- **fakeredis**: Redis mocking (v2.31.0+)
- **pyfakefs**: Filesystem mocking (v5.3.0+)

#### Configuration (pyproject.toml)
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --cov=packages.vecpipe --cov=packages.webui --cov-report=html --cov-report=term --cov-report=xml"
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
pythonpath = ["."]
asyncio_mode = "auto"
markers = [
    "e2e: End-to-end tests that require a running instance",
]
```

### Test Patterns & Conventions

#### 1. Service Layer Testing
```python
class TestCollectionService:
    """Test CollectionService implementation"""
    
    @pytest.fixture
    def mock_session(self):
        """Create a mock AsyncSession"""
        session = AsyncMock()
        session.commit = AsyncMock()
        return session
    
    @pytest.fixture
    def collection_service(self, mock_session, mock_collection_repo, ...):
        """Create CollectionService with mocked dependencies"""
        return CollectionService(
            db_session=mock_session,
            collection_repo=mock_collection_repo,
            operation_repo=mock_operation_repo,
            document_repo=mock_document_repo,
        )
    
    @pytest.mark.asyncio
    async def test_create_collection_success(self, collection_service, ...):
        """Test successful collection creation"""
        # Arrange
        mock_collection = Mock(spec=Collection)
        mock_collection.id = 123
        # ... setup
        
        # Act
        result = await collection_service.create_collection(...)
        
        # Assert
        assert result is not None
        mock_collection_repo.create.assert_called_once()
```

#### 2. API Endpoint Testing
```python
@pytest.mark.asyncio
async def test_create_collection_endpoint(async_client, auth_headers):
    """Test POST /api/v2/collections endpoint"""
    # Arrange
    payload = {
        "name": "Test Collection",
        "description": "Test description",
        "chunk_size": 1000
    }
    
    # Act
    response = await async_client.post(
        "/api/v2/collections",
        json=payload,
        headers=auth_headers
    )
    
    # Assert
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == payload["name"]
    assert "id" in data
```

#### 3. Repository Testing
```python
@pytest.mark.asyncio
async def test_collection_repository_create(db_session, test_user_db):
    """Test collection creation in database"""
    # Arrange
    repo = CollectionRepository(db_session)
    
    # Act
    collection = await repo.create(
        name="Test Collection",
        owner_id=test_user_db.id,
        embedding_model="test-model"
    )
    
    # Assert
    assert collection.id is not None
    assert collection.status == CollectionStatus.PENDING
    
    # Verify in database
    result = await db_session.execute(
        select(Collection).where(Collection.id == collection.id)
    )
    assert result.scalar_one_or_none() is not None
```

### Key Fixtures (tests/conftest.py)

#### Database Fixtures
```python
@pytest_asyncio.fixture
async def db_session():
    """Create a new database session for testing"""
    engine = create_async_engine(async_database_url, echo=False)
    
    # Drop and recreate all tables for test isolation
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    
    async_session = async_sessionmaker(engine, ...)
    
    async with async_session() as session:
        yield session
        await session.rollback()
    
    await engine.dispose()
```

#### Factory Fixtures
```python
@pytest_asyncio.fixture
async def collection_factory(db_session):
    """Factory for creating test collections"""
    created_collections = []
    
    async def _create_collection(**kwargs):
        defaults = {
            "id": str(uuid4()),
            "name": f"Test Collection {len(created_collections)}",
            "status": CollectionStatus.READY,
            # ... other defaults
        }
        defaults.update(kwargs)
        
        collection = Collection(**defaults)
        db_session.add(collection)
        await db_session.commit()
        await db_session.refresh(collection)
        
        created_collections.append(collection)
        return collection
    
    yield _create_collection
```

#### Mock Fixtures
```python
@pytest.fixture
def mock_redis_client():
    """Create a mock Redis client for testing WebSocket functionality"""
    class MockRedisStreams:
        def __init__(self):
            self.streams = {}
            self.consumer_groups = {}
            self.message_counter = 0
    
    mock_streams = MockRedisStreams()
    
    async def mock_xadd(stream_key, data, maxlen=None):
        # Implementation...
        pass
    
    mock = AsyncMock(spec=redis.Redis)
    mock.xadd = AsyncMock(side_effect=mock_xadd)
    # ... other mocked methods
    
    return mock
```

### Testing WebSockets

```python
class TestOperationsWebSocket:
    """Integration tests for the operations WebSocket endpoint"""
    
    @pytest.fixture
    def mock_websocket_client(self):
        """Create a mock WebSocket client"""
        mock = AsyncMock(spec=WebSocket)
        mock.accept = AsyncMock()
        mock.send_json = AsyncMock()
        mock.close = AsyncMock()
        mock.query_params = {"token": "valid-test-token"}
        mock.received_messages = []
        
        async def track_send_json(data):
            mock.received_messages.append(data)
        
        mock.send_json.side_effect = track_send_json
        return mock
    
    @pytest.mark.asyncio
    async def test_websocket_authentication_success(self, mock_websocket_client):
        """Test successful WebSocket authentication and connection"""
        # Test implementation...
```

## Frontend Testing

### Test Suite Structure

```
apps/webui-react/
├── src/
│   ├── components/__tests__/     # Component tests
│   ├── hooks/__tests__/          # Hook tests
│   ├── stores/__tests__/         # Store tests
│   ├── utils/__tests__/          # Utility tests
│   └── tests/
│       ├── mocks/                # MSW mocks
│       └── utils/                # Test utilities
└── vitest.config.ts              # Vitest configuration
```

### Testing Framework & Tools

#### Core Dependencies
- **Vitest**: Test runner and assertion library
- **React Testing Library**: Component testing utilities
- **MSW (Mock Service Worker)**: API mocking
- **@testing-library/jest-dom**: DOM matchers
- **@testing-library/user-event**: User interaction simulation

#### Configuration (vitest.config.ts)
```typescript
export default defineConfig({
  plugins: [react()],
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './vitest.setup.ts',
    include: [
      'src/**/*.{test,spec}.{js,mjs,cjs,ts,mts,cts,jsx,tsx}',
    ],
    exclude: ['**/node_modules/**', '**/e2e/**'],
    coverage: {
      reporter: ['text', 'json', 'html'],
      exclude: ['**/node_modules/**', '**/tests/**'],
    },
  },
})
```

### Test Patterns & Conventions

#### 1. Component Testing
```typescript
describe('CollectionCard', () => {
  it('should render collection information', () => {
    const collection = {
      id: '123',
      name: 'Test Collection',
      document_count: 100,
      status: 'ready',
    }
    
    const { getByText, getByRole } = render(
      <CollectionCard collection={collection} />
    )
    
    expect(getByText('Test Collection')).toBeInTheDocument()
    expect(getByText('100 documents')).toBeInTheDocument()
    expect(getByRole('button', { name: /view details/i })).toBeInTheDocument()
  })
  
  it('should handle click events', async () => {
    const onView = vi.fn()
    const { getByRole } = render(
      <CollectionCard collection={...} onView={onView} />
    )
    
    await userEvent.click(getByRole('button', { name: /view details/i }))
    
    expect(onView).toHaveBeenCalledWith('123')
  })
})
```

#### 2. Store Testing (Zustand)
```typescript
describe('chunkingStore', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    
    // Mock API responses
    (chunkingApi.preview as ReturnType<typeof vi.fn>).mockResolvedValue({
      chunks: [...],
      statistics: {...},
      performance: {...},
    })
  })
  
  it('should fetch preview successfully', async () => {
    const { result } = renderHook(() => useChunkingStore())
    
    await act(async () => {
      await result.current.fetchPreview('doc-123', {
        strategy: 'recursive',
        parameters: { chunk_size: 1000 },
      })
    })
    
    expect(result.current.preview).toBeDefined()
    expect(result.current.loading).toBe(false)
    expect(result.current.error).toBeNull()
  })
  
  it('should handle errors gracefully', async () => {
    (chunkingApi.preview as ReturnType<typeof vi.fn>).mockRejectedValue(
      new Error('Network error')
    )
    
    const { result } = renderHook(() => useChunkingStore())
    
    await act(async () => {
      await result.current.fetchPreview('doc-123', {...})
    })
    
    expect(result.current.error).toBe('Network error')
    expect(result.current.loading).toBe(false)
  })
})
```

#### 3. Hook Testing
```typescript
describe('useCollections', () => {
  it('should fetch collections on mount', async () => {
    const { result } = renderHook(() => useCollections(), {
      wrapper: AllTheProviders,
    })
    
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false)
    })
    
    expect(result.current.collections).toHaveLength(2)
    expect(result.current.collections[0].name).toBe('Test Collection 1')
  })
  
  it('should handle pagination', async () => {
    const { result } = renderHook(() => useCollections({ page: 2 }), {
      wrapper: AllTheProviders,
    })
    
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false)
    })
    
    expect(collectionsV2Api.list).toHaveBeenCalledWith(
      expect.objectContaining({ page: 2 })
    )
  })
})
```

### MSW Request Handlers

```typescript
// src/tests/mocks/handlers.ts
export const handlers = [
  // Collections endpoints
  http.get('/api/v2/collections', () => {
    return HttpResponse.json({
      collections: [
        {
          id: '123e4567-e89b-12d3-a456-426614174000',
          name: 'Test Collection 1',
          status: 'ready',
          document_count: 10,
          // ...
        }
      ],
      total: 1,
      page: 1,
      page_size: 10,
    })
  }),
  
  http.post('/api/v2/collections/:uuid/reindex', ({ params }) => {
    const operation: Operation = {
      id: 'op-' + Date.now(),
      collection_id: params.uuid as string,
      operation_type: 'reindex',
      status: 'pending',
      // ...
    }
    return HttpResponse.json(operation)
  }),
  
  // Search with reranking
  http.post('/api/v2/search', async ({ request }) => {
    const body = await request.json() as { use_reranker?: boolean }
    
    return HttpResponse.json({
      results: [...],
      reranking_used: body.use_reranker || false,
      reranker_model: body.use_reranker ? 'Qwen/Qwen3-Reranker-0.6B' : null,
      // ...
    })
  }),
]
```

### Test Utilities

```typescript
// src/tests/utils/test-utils.tsx
const customRender = (
  ui: ReactElement,
  options?: CustomRenderOptions
) => {
  const { initialEntries, ...renderOptions } = options || {}
  
  return render(ui, { 
    wrapper: ({ children }) => (
      <AllTheProviders initialEntries={initialEntries}>
        {children}
      </AllTheProviders>
    ),
    ...renderOptions 
  })
}

export { customRender as render }
```

## Test Coverage

### Current Coverage Levels

#### Backend Coverage
- **Target**: >80% for critical paths
- **Current**: ~75% overall
- **Critical Areas**:
  - API endpoints: 85%
  - Service layer: 80%
  - Repository layer: 70%
  - WebSocket handlers: 65%

#### Frontend Coverage
- **Target**: >70% for components and stores
- **Current**: ~65% overall
- **Critical Areas**:
  - Components: 70%
  - Stores: 75%
  - Hooks: 60%
  - Utils: 80%

### Coverage Reporting

#### Backend Coverage Command
```bash
poetry run pytest tests/ -v \
  --cov=packages \
  --cov-report=html \
  --cov-report=term-missing \
  --cov-report=xml
```

#### Frontend Coverage Command
```bash
npm run test:coverage
```

### Critical Paths Requiring Coverage

1. **Authentication Flow**: Login, token refresh, logout
2. **Collection Management**: CRUD operations, status transitions
3. **Operation Processing**: WebSocket updates, progress tracking
4. **Search & Reranking**: Query processing, result formatting
5. **Error Handling**: All error boundaries and recovery paths

## Test Data Management

### Fixture Patterns

#### 1. Factory Pattern
```python
@pytest_asyncio.fixture
async def document_factory(db_session):
    """Factory for creating test documents"""
    async def _create_document(**kwargs):
        defaults = {
            "id": str(uuid4()),
            "file_name": f"test_doc_{uuid4().hex[:8]}.txt",
            "status": DocumentStatus.COMPLETED,
            # ...
        }
        defaults.update(kwargs)
        
        document = Document(**defaults)
        db_session.add(document)
        await db_session.commit()
        return document
    
    yield _create_document
```

#### 2. Mock Data Builders
```python
class CollectionBuilder:
    def __init__(self):
        self.collection = {
            "id": str(uuid4()),
            "name": "Test Collection",
            "status": "ready",
        }
    
    def with_name(self, name):
        self.collection["name"] = name
        return self
    
    def with_documents(self, count):
        self.collection["document_count"] = count
        return self
    
    def build(self):
        return Collection(**self.collection)
```

#### 3. Database Seeding
```python
async def seed_test_data(db_session):
    """Seed database with test data"""
    # Create users
    users = [
        User(username="admin", email="admin@test.com"),
        User(username="user1", email="user1@test.com"),
    ]
    
    # Create collections
    collections = []
    for user in users:
        for i in range(3):
            collections.append(
                Collection(
                    name=f"{user.username}_collection_{i}",
                    owner_id=user.id,
                )
            )
    
    db_session.add_all(users + collections)
    await db_session.commit()
```

### Mock Patterns

#### 1. External Service Mocking
```python
@pytest.fixture
def mock_embedding_service():
    """Mock embedding service"""
    mock = MagicMock()
    mock.embed_texts.return_value = [[0.1] * 384]  # Mock embedding vector
    mock.embed_documents.return_value = [[0.1] * 384]
    return mock
```

#### 2. Redis Mocking with FakeRedis
```python
@pytest.fixture
def use_fakeredis():
    """Opt-in fixture to use fakeredis"""
    fake_sync_redis = fakeredis.FakeRedis(decode_responses=True)
    fake_async_redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    
    with (
        patch("redis.from_url", return_value=fake_sync_redis),
        patch("redis.asyncio.from_url", return_value=fake_async_redis),
    ):
        yield fake_sync_redis, fake_async_redis
```

## CI/CD Testing

### GitHub Actions Pipeline

#### Test Execution Strategy
1. **Parallel Execution**: Backend and frontend tests run in parallel
2. **Service Dependencies**: PostgreSQL, Redis, and Qdrant containers for integration tests
3. **Coverage Upload**: Automatic upload to Codecov
4. **Branch Protection**: All tests must pass before merging

#### CI Configuration (.github/workflows/main.yml)
```yaml
backend-tests:
  name: Backend Tests
  runs-on: ubuntu-latest
  services:
    postgres:
      image: postgres:16-alpine
      env:
        POSTGRES_USER: semantik_test
        POSTGRES_PASSWORD: test_password
        POSTGRES_DB: semantik_test
    redis:
      image: redis:7-alpine
    qdrant:
      image: qdrant/qdrant:latest
  steps:
    - name: Run backend tests
      run: |
        poetry run pytest tests/ -v \
          --ignore=tests/e2e \
          --cov=packages \
          --cov-report=xml \
          -m "not e2e"
```

### Test Environment Setup

#### Environment Variables
```bash
# Test environment
TESTING=true
ENV=test
DISABLE_RATE_LIMIT=true

# Database
DATABASE_URL=postgresql://semantik_test:test_password@localhost:5432/semantik_test

# Services
REDIS_URL=redis://localhost:6379
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Security
JWT_SECRET_KEY=test-secret-key-for-ci
USE_MOCK_EMBEDDINGS=true
```

### Test Parallelization

#### Backend Parallelization
```bash
# Run tests in parallel with pytest-xdist
poetry run pytest tests/ -n auto
```

#### Frontend Parallelization
```json
// vitest.config.ts
{
  "test": {
    "maxConcurrency": 5,
    "minThreads": 1,
    "maxThreads": 10
  }
}
```

## Testing Standards & Best Practices

### 1. Test Naming Conventions

#### Python Tests
- Test files: `test_*.py` or `*_test.py`
- Test classes: `Test*`
- Test functions: `test_*`
- Descriptive names: `test_create_collection_with_invalid_name_raises_validation_error`

#### TypeScript Tests
- Test files: `*.test.ts` or `*.spec.ts`
- Test suites: `describe('ComponentName', ...)`
- Test cases: `it('should do something when condition', ...)`

### 2. Test Organization

#### AAA Pattern
```python
@pytest.mark.asyncio
async def test_example():
    # Arrange - Set up test data and mocks
    mock_data = create_test_data()
    
    # Act - Execute the code under test
    result = await function_under_test(mock_data)
    
    # Assert - Verify the results
    assert result.status == "success"
    assert mock.called_with(expected_args)
```

### 3. Mock Usage Guidelines

- **Mock at boundaries**: Mock external dependencies, not internal components
- **Use real implementations when possible**: Prefer integration tests over heavily mocked unit tests
- **Verify mock interactions**: Use `assert_called_with` to verify correct usage
- **Reset mocks between tests**: Use `beforeEach`/`afterEach` or fixtures

### 4. Async Testing

#### Python Async Tests
```python
@pytest.mark.asyncio
async def test_async_operation():
    result = await async_function()
    assert result is not None
```

#### TypeScript Async Tests
```typescript
it('should handle async operations', async () => {
  const result = await asyncFunction()
  expect(result).toBeDefined()
})
```

### 5. Error Testing

#### Expected Exceptions
```python
@pytest.mark.asyncio
async def test_validation_error():
    with pytest.raises(ValidationError) as exc_info:
        await create_collection(name="")
    
    assert "Name cannot be empty" in str(exc_info.value)
```

#### Error Boundaries
```typescript
it('should display error message on failure', async () => {
  server.use(
    http.get('/api/collections', () => {
      return HttpResponse.error()
    })
  )
  
  render(<Collections />)
  
  await waitFor(() => {
    expect(screen.getByText(/error loading collections/i)).toBeInTheDocument()
  })
})
```

### 6. Performance Testing

#### Backend Performance Tests
```python
@pytest.mark.performance
async def test_bulk_operation_performance():
    start_time = time.time()
    
    # Create 1000 documents
    await bulk_create_documents(count=1000)
    
    elapsed = time.time() - start_time
    assert elapsed < 10  # Should complete within 10 seconds
```

### 7. Security Testing

#### Path Traversal Testing
```python
@pytest.mark.security
async def test_path_traversal_prevention():
    malicious_path = "../../etc/passwd"
    
    response = await client.post(
        "/api/collections",
        json={"name": "test", "path": malicious_path}
    )
    
    assert response.status_code == 400
    assert "Invalid path" in response.json()["detail"]
```

### 8. Test Documentation

#### Docstrings for Complex Tests
```python
@pytest.mark.asyncio
async def test_concurrent_operations_handling():
    """
    Test that concurrent operations on the same collection are properly queued.
    
    This test verifies that:
    1. Only one operation can be active per collection
    2. Subsequent operations wait for the current one to complete
    3. Operations are processed in FIFO order
    """
    # Test implementation...
```

### 9. Test Maintenance

#### Regular Review Checklist
- [ ] Remove obsolete tests
- [ ] Update tests for API changes
- [ ] Refactor duplicate test code into fixtures
- [ ] Review and update mock data
- [ ] Ensure tests match current business requirements

### 10. Continuous Improvement

#### Metrics to Track
- Test execution time
- Test flakiness rate
- Coverage trends
- Test maintenance burden
- Bug escape rate

#### Review Process
1. Monthly test suite review
2. Quarterly coverage analysis
3. Post-incident test gap analysis
4. Regular fixture and utility refactoring

## Test Execution Commands

### Backend Testing
```bash
# Run all tests
make test

# Run with coverage
make test-coverage

# Run specific test file
poetry run pytest tests/unit/test_collection_service.py -v

# Run tests matching pattern
poetry run pytest tests/ -k "collection" -v

# Run excluding E2E tests
poetry run pytest tests/ -m "not e2e"

# Run with parallel execution
poetry run pytest tests/ -n auto
```

### Frontend Testing
```bash
# Run all tests
npm test

# Run with coverage
npm run test:coverage

# Run in watch mode
npm run test:watch

# Run specific test file
npm test CollectionCard

# Run with UI
npm run test:ui

# Run for CI
npm run test:ci
```

### E2E Testing
```bash
# Backend E2E tests
poetry run pytest tests/e2e/ -v

# Frontend E2E tests (Playwright)
npm run test:e2e
```

## Troubleshooting Common Issues

### 1. Database Connection Issues
```bash
# Ensure PostgreSQL is running
docker-compose up -d postgres

# Check connection
poetry run alembic current
```

### 2. Redis Connection Issues
```bash
# Use fakeredis for tests that don't need real Redis
@pytest.mark.usefixtures("use_fakeredis")
async def test_with_fake_redis():
    # Test code
```

### 3. Flaky Tests
- Add retry logic for network-dependent tests
- Increase timeouts for async operations
- Use deterministic test data
- Mock time-dependent operations

### 4. Coverage Gaps
- Review untested error paths
- Add edge case tests
- Test all public API methods
- Verify exception handling

## Future Improvements

### Planned Enhancements
1. **Contract Testing**: Implement Pact for API contract validation
2. **Property-Based Testing**: Add Hypothesis for generative testing
3. **Load Testing**: Integrate Locust for performance testing
4. **Mutation Testing**: Add mutation testing to verify test effectiveness
5. **Visual Regression Testing**: Implement Percy or similar for UI testing
6. **Test Data Management**: Centralized test data factory system
7. **Test Parallelization**: Optimize test suite execution time
8. **Coverage Enforcement**: Fail builds on coverage regression

### Technical Debt
1. **WebSocket Test Coverage**: Increase coverage of real-time features
2. **E2E Test Stability**: Reduce flakiness in end-to-end tests
3. **Mock Consistency**: Standardize mocking patterns across tests
4. **Test Performance**: Optimize slow-running test suites
5. **Documentation**: Improve test documentation and examples

## Conclusion

The Semantik testing architecture provides a robust foundation for maintaining code quality and preventing regressions. By following these patterns and standards, the team ensures that:

1. All critical paths have comprehensive test coverage
2. Tests are maintainable and serve as documentation
3. The collection-centric architecture is properly validated
4. Performance and security requirements are verified
5. The CI/CD pipeline provides fast feedback on changes

Regular review and improvement of the testing strategy ensures it continues to meet the evolving needs of the project while maintaining high standards of quality and reliability.