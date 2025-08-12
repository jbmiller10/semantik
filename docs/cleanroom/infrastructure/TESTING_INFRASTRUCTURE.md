# TESTING_INFRASTRUCTURE Component Documentation

## 1. Component Overview

The TESTING_INFRASTRUCTURE component of Semantik provides a comprehensive, multi-layered testing framework that ensures code quality, reliability, and security across both backend Python services and frontend React applications. The infrastructure supports unit tests, integration tests, end-to-end tests, performance benchmarks, and security testing, all integrated into a CI/CD pipeline via GitHub Actions.

### Core Testing Technologies

**Backend Testing Stack:**
- **pytest** (v8.0.0+): Primary test framework with async support
- **pytest-asyncio** (v0.23.0+): Asynchronous test support
- **pytest-cov** (v4.1.0+): Coverage reporting
- **fakeredis** (v2.31.0+): Redis mocking for isolated tests
- **SQLAlchemy + asyncpg**: Database testing with transactions

**Frontend Testing Stack:**
- **Vitest** (v2.1.9+): Fast unit test runner for Vite-based React apps
- **React Testing Library** (v16.3.0+): Component testing utilities
- **MSW** (v2.10.4+): API mocking via service workers
- **Playwright** (v1.54.1+): E2E browser automation

**CI/CD Integration:**
- **GitHub Actions**: Automated test execution on push/PR
- **Codecov**: Coverage tracking and reporting
- **Trivy**: Security vulnerability scanning

### Testing Philosophy

1. **Test Isolation**: Each test runs in complete isolation with rollback transactions
2. **Mock External Dependencies**: Use fakeredis, MSW, and mock Qdrant clients
3. **Real Database Testing**: Integration tests use actual PostgreSQL instances
4. **Performance Baselines**: Track chunking strategy performance metrics
5. **Security-First**: Dedicated security test suites for path traversal, injection

## 2. Architecture & Design Patterns

### Backend Test Architecture

```
tests/
├── conftest.py                 # Shared fixtures and configuration
├── unit/                       # Isolated unit tests
│   ├── test_*_repository.py   # Repository layer tests
│   ├── test_*_service.py      # Service layer tests
│   └── test_*_chunker.py      # Chunking strategy tests
├── integration/                # Cross-component tests
│   ├── test_*_api.py          # API endpoint integration
│   └── test_*_flow.py         # End-to-end workflows
├── e2e/                        # Full system tests
│   └── test_websocket_*.py    # WebSocket real-time tests
├── performance/                # Performance benchmarks
│   └── chunking_benchmarks.py # Strategy performance tests
└── security/                   # Security-focused tests
    └── test_path_traversal.py # Path traversal prevention
```

### Frontend Test Architecture

```
apps/webui-react/
├── vitest.config.ts           # Vitest configuration
├── vitest.setup.ts            # Global test setup
└── src/
    ├── tests/
    │   └── mocks/
    │       ├── server.ts      # MSW server setup
    │       ├── handlers.ts    # API mock handlers
    │       └── test-utils.ts  # Testing utilities
    ├── components/__tests__/   # Component tests
    ├── stores/__tests__/       # State management tests
    └── utils/__tests__/        # Utility function tests
```

### Key Design Patterns

**1. Fixture Factory Pattern**
```python
@pytest_asyncio.fixture
async def collection_factory(db_session):
    """Factory for creating test collections."""
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
        created_collections.append(collection)
        return collection
    
    yield _create_collection
```

**2. Mock Service Pattern**
```typescript
// MSW handler for mocking API responses
http.post('/api/v2/collections/:uuid/reindex', ({ params }) => {
  const operation: Operation = {
    id: 'op-' + Date.now(),
    collection_id: params.uuid as string,
    operation_type: 'reindex',
    status: 'pending',
    // ... other fields
  }
  return HttpResponse.json(operation)
})
```

**3. Transactional Test Pattern**
```python
@pytest_asyncio.fixture
async def db_session():
    """Create a transactional database session."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    async_session = async_sessionmaker(engine, expire_on_commit=False)
    async with async_session() as session:
        yield session
        await session.rollback()  # Rollback after each test
```

## 3. Key Interfaces & Contracts

### Backend Test Fixtures

**Core Database Fixtures:**
```python
@pytest_asyncio.fixture
async def db_session() -> AsyncSession:
    """Provides isolated database session with automatic rollback."""
    
@pytest_asyncio.fixture
async def test_user_db(db_session) -> User:
    """Creates authenticated test user in database."""
    
@pytest.fixture
def auth_headers(test_user) -> dict[str, str]:
    """Generates JWT authorization headers."""
```

**Mock Service Fixtures:**
```python
@pytest.fixture
def mock_redis_client() -> AsyncMock:
    """Provides fake Redis client for WebSocket testing."""
    
@pytest.fixture
def mock_qdrant_client() -> MagicMock:
    """Mocks Qdrant vector database operations."""
    
@pytest.fixture
def mock_embedding_service() -> MagicMock:
    """Mocks embedding generation service."""
```

### Frontend Test Utilities

**Component Testing Helper:**
```typescript
const renderWithQueryClient = (component: React.ReactElement) => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      {component}
    </QueryClientProvider>
  );
};
```

**MSW Test Utilities:**
```typescript
export function mockReindexError(
  collectionId: string, 
  status: number, 
  detail?: string
) {
  server.use(
    http.post(`/api/v2/collections/${collectionId}/reindex`, () => {
      return HttpResponse.json({ detail }, { status })
    })
  )
}
```

## 4. Data Flow & Dependencies

### Test Execution Pipeline

```mermaid
graph TD
    A[Test Trigger] --> B{Test Type}
    B -->|Unit| C[Isolated Mocks]
    B -->|Integration| D[Service Dependencies]
    B -->|E2E| E[Full Stack]
    
    C --> F[Mock Redis/DB/Qdrant]
    D --> G[Real PostgreSQL]
    D --> H[Fake Redis]
    E --> I[Docker Services]
    
    F --> J[Test Execution]
    G --> J
    H --> J
    I --> J
    
    J --> K[Coverage Report]
    K --> L[CI/CD Pipeline]
```

### Dependency Injection for Testing

**Backend Dependency Override:**
```python
# In test client setup
app.dependency_overrides[get_current_user] = override_get_current_user
app.dependency_overrides[get_db] = override_get_db
app.dependency_overrides[create_collection_repository] = lambda: mock_repository
```

**Frontend Mock Provider:**
```typescript
// MSW intercepts all API calls
beforeAll(() => {
  server.listen({ onUnhandledRequest: 'error' });
});

afterEach(() => {
  server.resetHandlers();
});
```

## 5. Critical Implementation Details

### Mock Redis Implementation

```python
class MockRedisStreams:
    def __init__(self):
        self.streams = {}
        self.consumer_groups = {}
        self.message_counter = 0
    
    async def mock_xadd(self, stream_key, data, maxlen=None):
        if stream_key not in self.streams:
            self.streams[stream_key] = []
        
        self.message_counter += 1
        msg_id = f"{self.message_counter}-0"
        self.streams[stream_key].append((msg_id, data))
        
        if maxlen and len(self.streams[stream_key]) > maxlen:
            self.streams[stream_key] = self.streams[stream_key][-maxlen:]
        
        return msg_id
```

### Database Migration Handling in Tests

```python
async def drop_views_and_tables(conn):
    """Drop views before tables to handle dependencies."""
    views_to_drop = [
        "DROP VIEW IF EXISTS partition_hot_spots CASCADE",
        "DROP VIEW IF EXISTS partition_health_summary CASCADE",
        # ... other views
    ]
    
    for view_sql in views_to_drop:
        await conn.execute(text(view_sql))
    
    await conn.run_sync(Base.metadata.drop_all)
```

### WebSocket Testing Pattern

```python
def test_collection_creation_with_websocket_progress(self):
    """Test real-time updates via WebSocket during collection creation."""
    headers = self._get_auth_headers()
    token = self._get_auth_token()
    
    # Create collection
    response = requests.post(
        f"{self.API_BASE_URL}/api/v2/collections",
        json={...},
        headers=headers
    )
    
    # Connect WebSocket and monitor progress
    ws = websocket.WebSocket()
    ws.connect(f"{self.WS_BASE_URL}/ws", header=[f"Authorization: Bearer {token}"])
    
    # Verify progress messages
    message = json.loads(ws.recv())
    assert message["type"] == "operation_progress"
```

## 6. Security Considerations

### Path Traversal Testing

```python
class TestPathTraversalSecurity:
    def test_url_encoded_traversal_blocked(self):
        """Test URL-encoded directory traversal prevention."""
        dangerous_paths = [
            ["%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd"],  # Single encoding
            ["%252e%252e%252f"],                           # Double encoding
            ["%25252e%25252e%25252f"],                     # Triple encoding
            ["..%2f..%2f..%2fetc%2fpasswd"],              # Mixed encoding
        ]
        
        for paths in dangerous_paths:
            with pytest.raises(ValidationError, match="Invalid file path"):
                ChunkingSecurityValidator.validate_file_paths(paths)
```

### Authentication Testing

```python
@pytest.fixture
def test_user_headers(auth_headers):
    """Secure JWT token generation for tests."""
    token = create_access_token(data={"sub": test_user["username"]})
    return {"Authorization": f"Bearer {token}"}
```

### Security Test Patterns

1. **Input Validation**: Test all OWASP injection patterns
2. **Authentication**: Verify JWT token validation
3. **Authorization**: Test resource access controls
4. **Rate Limiting**: Verify rate limit enforcement
5. **Path Traversal**: Block all traversal attempts

## 7. Testing Requirements

### Coverage Targets

**Backend Coverage Requirements:**
- Overall: ≥80% line coverage
- Critical paths: ≥90% coverage
- Security modules: 100% coverage
- New code: ≥85% coverage

**Frontend Coverage Requirements:**
- Components: ≥75% coverage
- Stores: ≥80% coverage
- Utils: ≥90% coverage
- Critical UI flows: ≥85% coverage

### Test Categories

**Unit Tests (70% of tests):**
- Fast execution (<100ms per test)
- No external dependencies
- Mock all I/O operations
- Test single units in isolation

**Integration Tests (20% of tests):**
- Test component interactions
- Use real database with transactions
- Mock external services (Qdrant, Redis)
- Verify API contracts

**E2E Tests (10% of tests):**
- Full system testing
- Real services via Docker
- WebSocket real-time testing
- User workflow validation

### Performance Benchmarks

```python
PERFORMANCE_TARGETS = {
    "character": {
        "single_thread": 1000,      # chunks/sec
        "parallel_4": 3500,          # chunks/sec with 4 workers
        "memory_per_mb": 50,         # MB memory per MB document
    },
    "semantic": {
        "single_thread": 150,        # Lower due to embeddings
        "parallel_4": 400,           # Limited by embedding model
        "memory_per_mb": 200,
    },
}
```

## 8. Common Pitfalls & Best Practices

### Common Pitfalls

**1. Test Pollution**
```python
# BAD: Modifying global state
def test_collection_create():
    global test_collection
    test_collection = Collection(...)  # Pollutes other tests

# GOOD: Use fixtures with cleanup
@pytest.fixture
def collection(db_session):
    coll = Collection(...)
    db_session.add(coll)
    yield coll
    db_session.rollback()  # Automatic cleanup
```

**2. Flaky Async Tests**
```python
# BAD: Race conditions in async tests
async def test_websocket():
    await send_message()
    result = await receive_message()  # May timeout

# GOOD: Use proper waiting
async def test_websocket():
    await send_message()
    result = await asyncio.wait_for(receive_message(), timeout=5.0)
```

**3. Incomplete Mocking**
```typescript
// BAD: Partial mocking leads to real API calls
vi.mock('../api', () => ({
  getCollections: vi.fn()
  // Missing other methods
}))

// GOOD: Complete API mocking with MSW
server.use(
  http.get('/api/collections', () => HttpResponse.json([])),
  http.post('/api/collections', () => HttpResponse.json({...}))
)
```

### Best Practices

**1. Test Data Builders**
```python
def create_test_collection(**overrides):
    """Builder pattern for test data."""
    defaults = {
        "id": str(uuid4()),
        "name": "Test Collection",
        "status": CollectionStatus.READY,
    }
    return Collection(**{**defaults, **overrides})
```

**2. Descriptive Test Names**
```python
def test_collection_deletion_cascades_to_documents():
    """Test names should describe behavior, not implementation."""
    pass

def test_reindex_operation_sends_websocket_updates():
    """Clear description of what is being tested."""
    pass
```

**3. Arrange-Act-Assert Pattern**
```typescript
it('should handle search errors gracefully', async () => {
  // Arrange
  mockSearchError(500, 'Internal Server Error')
  
  // Act
  const { result } = renderHook(() => useSearchStore())
  await act(() => result.current.performSearch('test query'))
  
  // Assert
  expect(result.current.error).toBe('Search failed: Internal Server Error')
  expect(result.current.results).toEqual([])
})
```

## 9. Configuration & Environment

### Test Environment Configuration

**Backend Test Environment (.env.test):**
```bash
# Test Database
DATABASE_URL=postgresql://semantik_test:test_password@localhost:5432/semantik_test
POSTGRES_USER=semantik_test
POSTGRES_PASSWORD=test_password
POSTGRES_DB=semantik_test

# Services
QDRANT_HOST=localhost
QDRANT_PORT=6333
REDIS_URL=redis://localhost:6379/0

# Security
JWT_SECRET_KEY=test-secret-key-for-testing-only

# Test Flags
TESTING=true
ENV=test
USE_MOCK_EMBEDDINGS=true
DISABLE_AUTH=true
DISABLE_RATE_LIMITING=true
```

**Frontend Test Configuration (vitest.config.ts):**
```typescript
export default defineConfig({
  plugins: [react()],
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './vitest.setup.ts',
    include: [
      'src/**/*.{test,spec}.{js,ts,jsx,tsx}',
      '**/*.{test,spec}.{js,ts,jsx,tsx}'
    ],
    exclude: ['**/node_modules/**', '**/e2e/**'],
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
})
```

### CI/CD Test Configuration

**GitHub Actions Workflow:**
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

### Test Command Reference

**Backend Testing:**
```bash
# Run all tests
make test

# Run tests excluding E2E
make test-ci

# Run only E2E tests
make test-e2e

# Run with coverage
make test-coverage

# Run specific test file
poetry run pytest tests/unit/test_collection_service.py -v

# Run tests matching pattern
poetry run pytest -k "websocket" -v
```

**Frontend Testing:**
```bash
# Run all frontend tests
npm run test

# Run tests in watch mode
npm run test:watch

# Run with coverage
npm run test:coverage

# Run specific component tests
npm run test:collections

# Run tests in CI mode
npm run test:ci
```

## 10. Integration Points

### CI/CD Pipeline Integration

**Test Stages:**
1. **Quality Checks**: Linting, formatting, type checking
2. **Security Scan**: Trivy vulnerability scanning
3. **Backend Tests**: Unit + Integration tests with services
4. **Frontend Tests**: Component + Store tests
5. **Build Validation**: Production build verification
6. **Coverage Upload**: Codecov integration

### Coverage Reporting

**Backend Coverage:**
```python
# pytest.ini configuration
addopts = "-v --cov=packages.vecpipe --cov=packages.webui --cov-report=html --cov-report=term --cov-report=xml"
```

**Frontend Coverage:**
```json
{
  "scripts": {
    "test:coverage": "vitest --coverage"
  }
}
```

### Test Database Management

**Automatic Migration:**
```python
async with engine.begin() as conn:
    # Drop all views and tables
    await drop_views_and_tables(conn)
    # Recreate schema
    await conn.run_sync(Base.metadata.create_all)
```

**Partition Testing:**
```python
def test_partition_distribution():
    """Verify partition key calculation and distribution."""
    with engine.connect() as conn:
        result = conn.execute(text("SELECT * FROM partition_distribution"))
        assert result.rowcount > 0
```

### WebSocket Testing Infrastructure

**Real-time Update Testing:**
```python
class TestWebSocketIntegration:
    API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8080")
    WS_BASE_URL = API_BASE_URL.replace("http://", "ws://")
    
    def test_operation_progress_updates(self):
        ws = websocket.WebSocket()
        ws.connect(f"{self.WS_BASE_URL}/ws")
        # Monitor real-time updates
```

### Performance Monitoring

**Benchmark Execution:**
```python
class ChunkingBenchmarks:
    @staticmethod
    def run_benchmark(strategy: str, document_size: int):
        start_time = time.time()
        memory_before = psutil.Process().memory_info().rss
        
        # Execute chunking
        chunks = chunker.chunk(document)
        
        duration = time.time() - start_time
        memory_used = psutil.Process().memory_info().rss - memory_before
        
        return BenchmarkResult(
            strategy=strategy,
            duration_seconds=duration,
            memory_used_mb=memory_used / (1024 * 1024),
            chunks_per_second=len(chunks) / duration
        )
```

## Summary

The TESTING_INFRASTRUCTURE component provides a robust, multi-layered testing framework that ensures code quality across the entire Semantik application. Key strengths include:

1. **Comprehensive Coverage**: Unit, integration, E2E, performance, and security tests
2. **Isolation**: Transactional tests with automatic rollback
3. **Mocking Strategy**: FakeRedis, MSW, and mock Qdrant for fast, reliable tests
4. **CI/CD Integration**: Automated testing on every push/PR via GitHub Actions
5. **Performance Baselines**: Track and enforce performance requirements
6. **Security Focus**: Dedicated security test suites for common vulnerabilities

The infrastructure supports both TDD and BDD approaches, with clear patterns for writing maintainable, reliable tests that provide confidence in code changes while maintaining fast feedback loops during development.