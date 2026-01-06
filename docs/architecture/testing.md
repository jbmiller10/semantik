# Testing Architecture

> **Location:** `tests/` (backend), `apps/webui-react/src/` (frontend, with `__tests__/` subdirectories)
>
> See also: `tests/CLAUDE.md` for test organization guidelines, `apps/webui-react/src/CLAUDE.md` for frontend testing patterns

## Overview

Comprehensive testing infrastructure with:
- **pytest** for Python backend
- **Vitest + React Testing Library** for frontend
- **MSW** for API mocking
- **fakeredis** for Redis isolation

## Test Stack

| Layer | Framework | Coverage Target |
|-------|-----------|-----------------|
| Python Backend | pytest 8.0+ | ≥80% |
| React Frontend | Vitest 2.1+ | ≥75% |
| E2E | pytest-playwright | Critical paths |

## Directory Structure

### Backend Tests
```
tests/
├── conftest.py           # Root fixtures (~1000 lines)
├── unit/                 # Isolated unit tests
├── integration/          # Real infrastructure tests
│   ├── repositories/     # Repository integration tests
│   ├── services/         # Service integration tests
│   └── strategies/       # Chunking strategy tests
├── e2e/                  # End-to-end workflows
├── webui/                # WebUI package tests
│   ├── api/v1/           # Legacy API endpoint tests
│   ├── api/v2/           # V2 API endpoint tests (includes conftest.py)
│   ├── services/         # Service layer tests
│   └── tasks/            # Celery task tests
├── application/          # Application layer use case tests
├── domain/               # Domain model tests
├── shared/               # Shared package tests
│   ├── database/         # Database utility tests
│   ├── embedding/        # Embedding provider tests
│   └── repositories/     # Repository tests
├── database/             # Database migration tests
├── security/             # Security vulnerability tests
├── streaming/            # Streaming chunker tests
├── performance/          # Performance-focused tests
├── services/             # Service tests
│   └── chunking/         # Chunking service tests
└── test_data/            # Static test files
```

### Frontend Tests
```
apps/webui-react/
├── vitest.config.ts        # Vitest configuration
├── vitest.setup.ts         # Global test setup (MSW, mocks)
└── src/
    ├── components/__tests__/  # Component tests
    ├── hooks/__tests__/       # Hook tests
    ├── stores/__tests__/      # Store tests
    ├── pages/__tests__/       # Page tests
    ├── services/api/v2/__tests__/  # API client tests
    ├── utils/__tests__/       # Utility tests
    └── tests/
        ├── setup.ts           # Additional setup
        ├── mocks/
        │   ├── server.ts      # MSW server setup
        │   ├── handlers.ts    # API mock handlers
        │   └── errorHandlers.ts  # Error scenario handlers
        ├── types/             # Test type definitions
        └── utils/
            ├── test-utils.tsx   # Custom render utilities
            ├── providers.tsx    # Test provider wrappers
            └── queryClient.ts   # Test query client
```

## Key Fixtures

### Database Fixtures

**db_session** - Real PostgreSQL async session (from `tests/conftest.py`):
```python
@pytest_asyncio.fixture
async def db_session():
    """Create a new database session for testing.

    - Connects to test database (semantik_test by default)
    - Creates tables if they don't exist
    - Rolls back uncommitted changes after each test
    - Skips tests if database is unavailable
    """
    # Connects using DATABASE_URL or constructs from POSTGRES_* env vars
    engine = create_async_engine(async_database_url, pool_size=1)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session = async_sessionmaker(engine, expire_on_commit=False)
    async with async_session() as session:
        yield session
        if session.in_transaction():
            await session.rollback()
    await engine.dispose()
```

**collection_factory** - Create test collections (from `tests/conftest.py`):
```python
@pytest_asyncio.fixture
async def collection_factory(db_session):
    """Factory for creating test collections. Requires owner_id."""
    async def _create_collection(**kwargs):
        if "owner_id" not in kwargs:
            raise ValueError("owner_id must be provided")

        defaults = {
            "id": str(uuid4()),
            "name": f"Test Collection {uuid4().hex[:8]}",
            "status": CollectionStatus.READY,
            "embedding_model": "test-model",
            # ... other defaults
        }
        defaults.update(kwargs)
        collection = Collection(**defaults)
        db_session.add(collection)
        await db_session.commit()
        return collection
    return _create_collection
```

### Authentication Fixtures

> **Note:** These fixtures are defined in `tests/webui/api/v2/conftest.py` for V2 API tests.

**api_client** - Authenticated AsyncClient with database session:
```python
@pytest_asyncio.fixture()
async def api_client(db_session, test_user_db, use_fakeredis, reset_redis_manager):
    """Provide an AsyncClient with real DB session and fakeredis overrides."""

    async def override_get_db():
        yield db_session

    async def override_get_current_user():
        return {"id": test_user_db.id, "username": test_user_db.username, ...}

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = override_get_current_user

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    app.dependency_overrides.clear()
```

**api_auth_headers** - JWT bearer headers for the test user:
```python
@pytest.fixture()
def api_auth_headers(test_user_db) -> dict[str, str]:
    """Issue a valid bearer token for the persisted test user."""
    token = create_access_token(data={"sub": test_user_db.username})
    return {"Authorization": f"Bearer {token}"}
```

### Mocking Fixtures

**stub_celery_send_task** (autouse) - Prevents real Celery calls (from `tests/conftest.py`):
```python
@pytest.fixture(autouse=True)
def stub_celery_send_task(monkeypatch):
    """Ensure Celery does not require a live broker during tests."""
    send_task_mock = MagicMock(name="celery_send_task_stub")
    send_task_mock.return_value = MagicMock(name="celery_async_result_stub")
    monkeypatch.setattr(celery_module.celery_app, "send_task", send_task_mock, raising=False)
    return send_task_mock
```

**use_fakeredis** - Opt-in Redis mock (from `tests/conftest.py`):
```python
@pytest.fixture()
def use_fakeredis():
    """Opt-in fixture to use fakeredis for a specific test."""
    fake_sync_redis = fakeredis.FakeRedis(decode_responses=True)
    fake_async_redis = fakeredis.aioredis.FakeRedis(decode_responses=True)

    with (
        patch("redis.from_url", return_value=fake_sync_redis),
        patch("redis.asyncio.from_url", return_value=fake_async_redis),
        # Additional patches for WebSocket and service managers...
    ):
        yield fake_sync_redis, fake_async_redis
```

**mock_qdrant_client** - Qdrant mock (from `tests/conftest.py`):
```python
@pytest.fixture()
def mock_qdrant_client():
    """Mock Qdrant client for testing."""
    mock = MagicMock()
    mock.get_collections.return_value = MagicMock(collections=[])
    mock.search.return_value = []
    return mock
```

## Test Patterns

### API Endpoint Test
```python
# From tests/webui/api/v2/test_directory_scan.py
@pytest.mark.asyncio()
async def test_scan_directory_preview_returns_supported_files(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    tmp_path: Path,
) -> None:
    """Scanning a directory should return the supported files discovered."""
    (tmp_path / "doc1.txt").write_text("hello world", encoding="utf-8")

    payload = {
        "scan_id": str(uuid4()),
        "path": str(tmp_path),
        "recursive": True,
    }

    response = await api_client.post(
        "/api/v2/directory-scan/preview",
        json=payload,
        headers=api_auth_headers,
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total_files"] >= 1
```

### Repository Test
```python
@pytest.mark.asyncio()
async def test_get_collection(db_session, collection_factory, test_user_db):
    collection = await collection_factory(owner_id=test_user_db.id)
    repo = CollectionRepository(db_session)

    found = await repo.get_by_id(collection.id, user_id=test_user_db.id)

    assert found is not None
    assert found.id == collection.id
```

### Service Test
```python
@pytest.mark.asyncio()
async def test_search_service(
    db_session,
    mock_qdrant_client,
    collection_factory
):
    collection = await collection_factory(owner_id=1)
    mock_qdrant_client.search.return_value = [
        {"id": "chunk1", "score": 0.9}
    ]

    service = SearchService(db_session, mock_qdrant_client)
    results = await service.search("query", collection_ids=[collection.id])

    assert len(results) == 1
```

## Frontend Testing

### MSW Setup
```typescript
// src/tests/mocks/server.ts
import { setupServer } from 'msw/node'
import { handlers } from './handlers'

export const server = setupServer(...handlers)
```

```typescript
// vitest.setup.ts (at apps/webui-react root)
import { server } from './src/tests/mocks/server';
import { useAuthStore } from './src/stores/authStore';

beforeAll(() => server.listen({ onUnhandledRequest: 'error' }));

afterEach(() => {
  server.resetHandlers();
  // Reset auth state between tests
  useAuthStore.setState({ token: null, refreshToken: null, user: null });
  localStorage.removeItem('auth-storage');
});

afterAll(async () => await server.close());
```

### MSW Handlers
```typescript
// src/tests/mocks/handlers.ts
import { http, HttpResponse } from 'msw'
import type { Collection } from '../../types/collection'

export const handlers = [
  // Auth endpoints
  http.post('/api/auth/login', async ({ request }) => {
    const { username, password } = await request.json() as { username: string; password: string }
    if (username === 'testuser' && password === 'testpass') {
      return HttpResponse.json({ access_token: 'mock-jwt-token', ... })
    }
    return HttpResponse.json({ detail: 'Invalid credentials' }, { status: 401 })
  }),

  // V2 API endpoints
  http.get('/api/v2/collections', () => {
    return HttpResponse.json({
      collections: [{ id: '123e4567-...', name: 'Test Collection 1', status: 'ready', ... }],
      total: 1,
    });
  }),

  http.post('/api/v2/search', async ({ request }) => {
    const body = await request.json() as { use_reranker?: boolean }
    return HttpResponse.json({
      results: [{ document_id: 'doc_1', score: 0.85, text: 'Test result', ... }],
      total_results: 1,
      reranking_used: body.use_reranker || false,
    });
  }),
];
```

### Component Test
```typescript
// src/components/__tests__/SearchInterface.test.tsx
import { render, screen, waitFor } from '@/tests/utils/test-utils';
import userEvent from '@testing-library/user-event';
import { SearchInterface } from '../SearchInterface';
import { server } from '@/tests/mocks/server';
import { searchErrorHandlers } from '@/tests/mocks/errorHandlers';

describe('SearchInterface', () => {
  it('should perform search on submit', async () => {
    render(<SearchInterface />);  // Uses custom render with providers

    await userEvent.type(screen.getByPlaceholderText(/search/i), 'query');
    await userEvent.click(screen.getByRole('button', { name: /search/i }));

    await waitFor(() => {
      expect(screen.getByText(/result/i)).toBeInTheDocument();
    });
  });

  it('should handle API errors', async () => {
    // Override default handlers with error scenario
    server.use(...searchErrorHandlers.serverError());

    render(<SearchInterface />);
    await userEvent.click(screen.getByRole('button', { name: /search/i }));

    await waitFor(() => {
      expect(screen.getByText(/error/i)).toBeInTheDocument();
    });
  });
});
```

### Store Test
```typescript
import { useAuthStore } from '@/stores/authStore';

describe('authStore', () => {
  beforeEach(() => {
    useAuthStore.setState({ token: null, user: null });
  });

  it('should store auth token', () => {
    const { setAuth } = useAuthStore.getState();

    setAuth('token', { id: 1, username: 'test' });

    expect(useAuthStore.getState().token).toBe('token');
  });
});
```

### Hook Test
```typescript
import { renderHook, waitFor } from '@testing-library/react';
import { useCollections } from '@/hooks/useCollections';
import { AllTheProviders } from '@/tests/utils/providers';

describe('useCollections', () => {
  it('should fetch collections', async () => {
    const { result } = renderHook(() => useCollections(), {
      wrapper: AllTheProviders
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.data).toHaveLength(1);
  });
});
```

## Running Tests

### Backend
```bash
# All tests (uses uv)
make test
# Or directly:
uv run pytest tests -v

# Exclude E2E (for CI)
make test-ci
# Or directly:
uv run pytest tests -v --ignore=tests/e2e -m "not e2e"

# E2E tests only (requires running services)
make test-e2e

# With coverage (HTML, terminal, and XML reports)
make test-coverage
# Or directly:
uv run pytest tests -v --cov=vecpipe --cov=webui --cov=shared --cov-report=html --cov-report=term

# Specific file
uv run pytest tests/webui/api/v2/test_collections.py -v

# Pattern match
uv run pytest -k "test_search" -v

# Parallel (requires pytest-xdist)
uv run pytest -n auto
```

### Frontend
```bash
# All tests (runs vitest)
cd apps/webui-react && npm test
# Or:
make frontend-test

# Watch mode
npm run test:watch

# Coverage
npm run test:coverage

# UI debugger
npm run test:ui

# Specific file
npm test -- src/components/__tests__/SearchInterface.test.tsx

# CI mode (verbose, no watch)
npm run test:ci
```

## Coverage Configuration

### Python (pyproject.toml)
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --cov=vecpipe --cov=webui --cov=shared --cov-report=html --cov-report=term --cov-report=xml"
python_files = ["test_*.py"]
asyncio_mode = "auto"
markers = [
    "e2e: End-to-end tests that require a running instance",
    "integration: Integration tests that rely on real infrastructure",
    "performance: Performance-focused tests that may run slowly",
]
```

### TypeScript (vitest.config.ts)
```typescript
// apps/webui-react/vitest.config.ts
export default defineConfig({
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './vitest.setup.ts',
    testTimeout: 10000,
    exclude: ['**/e2e/**', '**/node_modules/**'],
  },
});
// Coverage uses @vitest/coverage-v8 package
```

## Extension Patterns

### Adding Tests for New Repository
1. Create `tests/unit/repositories/test_<name>_repository.py` (see `test_collection_sync_run_repository.py` for example)
2. Use `db_session` fixture for database access
3. Create factory fixture if needed
4. Test CRUD operations and edge cases

### Adding Tests for New API Endpoint
1. Create `tests/webui/api/v2/test_<name>.py` (see `test_collections.py` for example)
2. Use `api_client` and `api_auth_headers` fixtures
3. Test success, validation, and error cases
4. Verify Celery task dispatch if applicable

### Adding Tests for New React Component
1. Create `src/components/__tests__/<ComponentName>.test.tsx` (see `CollectionCard.test.tsx` for example)
2. Add MSW handlers for API calls
3. Use `renderWithProviders` for proper context
4. Test user interactions with `userEvent`
5. Test loading, success, and error states

### Adding MSW Error Handler
```typescript
// src/tests/mocks/errorHandlers.ts
import { http, HttpResponse, delay } from 'msw'

// Utility for common error responses
export const createErrorHandler = (
  method: 'get' | 'post' | 'put' | 'delete',
  path: string,
  status: number,
  response?: { detail?: string }
) => {
  return http[method](path, () => {
    return HttpResponse.json(response || { detail: `Error ${status}` }, { status })
  })
}

// Pre-built error scenarios
export const collectionErrorHandlers = {
  networkError: () => [
    http.get('/api/v2/collections', () => HttpResponse.error()),
  ],
  serverError: () => [
    createErrorHandler('get', '/api/v2/collections', 500),
  ],
  notFound: () => [
    createErrorHandler('get', '/api/v2/collections/:uuid', 404, { detail: 'Collection not found' }),
  ],
};

// In test file
import { collectionErrorHandlers } from '@/tests/mocks/errorHandlers';
server.use(...collectionErrorHandlers.serverError());
```

## Best Practices

1. **Use factories** for test data, not hard-coded values
2. **Isolate tests** with proper fixtures and cleanup
3. **Test behavior**, not implementation
4. **Use MSW** for frontend API mocking (never direct mocks)
5. **Clean up** in fixture teardown
6. **Mock external services** (Qdrant, Redis, Celery)
7. **Test error paths**, not just happy paths
8. **Use `waitFor()`** for async operations
9. **Minimize dependencies** between tests
