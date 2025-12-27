# Testing Architecture

> **Location:** `tests/`, `apps/webui-react/src/**/__tests__/`

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
├── conftest.py           # Root fixtures (895 lines)
├── unit/                 # Isolated unit tests
├── integration/          # Real infrastructure tests
├── e2e/                  # End-to-end workflows
├── webui/                # WebUI package tests
│   ├── api/v2/           # API endpoint tests
│   └── services/         # Service layer tests
├── shared/               # Shared package tests
├── domain/               # Domain model tests
└── test_data/            # Static test files
```

### Frontend Tests
```
src/
├── components/__tests__/
├── hooks/__tests__/
├── stores/__tests__/
├── utils/__tests__/
└── tests/
    ├── setup.ts
    ├── mocks/
    │   ├── server.ts
    │   ├── handlers.ts
    │   └── errorHandlers.ts
    └── utils/
        └── test-utils.tsx
```

## Key Fixtures

### Database Fixtures

**db_session** - Real PostgreSQL async session:
```python
@pytest_asyncio.fixture
async def db_session():
    engine = create_async_engine(test_database_url)
    async with async_sessionmaker(engine)() as session:
        yield session
        await session.rollback()
```

**collection_factory** - Create test collections:
```python
@pytest_asyncio.fixture
async def collection_factory(db_session):
    async def _create(owner_id: int, **kwargs):
        collection = Collection(
            id=str(uuid4()),
            name=f"Test Collection {uuid4().hex[:8]}",
            owner_id=owner_id,
            **kwargs
        )
        db_session.add(collection)
        await db_session.commit()
        return collection
    return _create
```

### Authentication Fixtures

**api_client** - Authenticated AsyncClient:
```python
@pytest_asyncio.fixture
async def api_client(db_session, test_user_db):
    async with AsyncClient(app=app, base_url="http://test") as client:
        # Override auth dependency
        app.dependency_overrides[get_current_user] = lambda: test_user_db
        yield client
```

**api_auth_headers** - JWT bearer headers:
```python
@pytest.fixture
def api_auth_headers(test_user):
    token = create_access_token(data={"sub": test_user["username"]})
    return {"Authorization": f"Bearer {token}"}
```

### Mocking Fixtures

**stub_celery_send_task** (autouse) - Prevents real Celery calls:
```python
@pytest.fixture(autouse=True)
def stub_celery_send_task(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr(celery_app, "send_task", mock)
    return mock
```

**use_fakeredis** - Opt-in Redis mock:
```python
@pytest.fixture
def use_fakeredis():
    fake_redis = fakeredis.aioredis.FakeRedis()
    with patch("redis.asyncio.from_url", return_value=fake_redis):
        yield fake_redis
```

**mock_qdrant_client** - Qdrant mock:
```python
@pytest.fixture
def mock_qdrant_client():
    client = MagicMock()
    client.search.return_value = []
    return client
```

## Test Patterns

### API Endpoint Test
```python
@pytest.mark.asyncio()
async def test_create_collection(
    api_client: AsyncClient,
    api_auth_headers: dict,
    stub_celery_send_task
):
    response = await api_client.post(
        "/api/v2/collections",
        json={"name": "Test", "embedding_model": "Qwen/Qwen3-Embedding-0.6B"},
        headers=api_auth_headers
    )

    assert response.status_code == 201
    assert response.json()["name"] == "Test"
    stub_celery_send_task.assert_called_once()
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
// tests/mocks/server.ts
import { setupServer } from 'msw/node';
import { handlers } from './handlers';

export const server = setupServer(...handlers);

// vitest.setup.ts
beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());
```

### MSW Handlers
```typescript
// tests/mocks/handlers.ts
import { http, HttpResponse } from 'msw';

export const handlers = [
  http.get('/api/v2/collections', () => {
    return HttpResponse.json({
      collections: [{ id: '1', name: 'Test', status: 'ready' }]
    });
  }),

  http.post('/api/v2/search', async ({ request }) => {
    const body = await request.json();
    return HttpResponse.json({
      results: [{ id: 'r1', content: 'Result', score: 0.9 }],
      query: body.query
    });
  }),
];
```

### Component Test
```typescript
import { render, screen, waitFor } from '@/tests/utils/test-utils';
import userEvent from '@testing-library/user-event';
import { SearchInterface } from '../SearchInterface';

describe('SearchInterface', () => {
  it('should perform search on submit', async () => {
    const { user } = renderWithProviders(<SearchInterface />);

    await user.type(screen.getByPlaceholderText('Search'), 'query');
    await user.click(screen.getByRole('button', { name: /search/i }));

    await waitFor(() => {
      expect(screen.getByText('Result')).toBeInTheDocument();
    });
  });

  it('should handle API errors', async () => {
    server.use(
      http.post('/api/v2/search', () => {
        return HttpResponse.json({ detail: 'Error' }, { status: 500 });
      })
    );

    const { user } = renderWithProviders(<SearchInterface />);
    await user.click(screen.getByRole('button', { name: /search/i }));

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
# All tests
make test

# Exclude E2E
make test-ci

# With coverage
make test-coverage

# Specific file
pytest tests/webui/api/v2/test_collections.py -v

# Pattern match
pytest -k "test_search" -v

# Parallel
pytest -n auto
```

### Frontend
```bash
# All tests
npm run test

# Watch mode
npm run test:watch

# Coverage
npm run test:coverage

# UI debugger
npm run test:ui

# Specific file
npm run test -- src/components/__tests__/SearchInterface.test.tsx
```

## Coverage Configuration

### Python (pyproject.toml)
```toml
[tool.pytest.ini_options]
addopts = "-v --cov=vecpipe --cov=webui --cov=shared --cov-report=html"
```

### TypeScript (vitest.config.ts)
```typescript
test: {
  coverage: {
    provider: 'v8',
    reporter: ['text', 'html', 'json']
  }
}
```

## Extension Patterns

### Adding Tests for New Repository
1. Create `tests/unit/repositories/test_my_repo.py`
2. Use `db_session` fixture for database access
3. Create factory fixture if needed
4. Test CRUD operations and edge cases

### Adding Tests for New API Endpoint
1. Create `tests/webui/api/v2/test_my_endpoint.py`
2. Use `api_client` and `api_auth_headers` fixtures
3. Test success, validation, and error cases
4. Verify Celery task dispatch if applicable

### Adding Tests for New React Component
1. Create `components/__tests__/MyComponent.test.tsx`
2. Add MSW handlers for API calls
3. Use `renderWithProviders` for proper context
4. Test user interactions with `userEvent`
5. Test loading, success, and error states

### Adding MSW Error Handler
```typescript
// tests/mocks/errorHandlers.ts
export const myEndpointErrors = {
  serverError: () => [
    http.get('/api/v2/my-endpoint', () => {
      return HttpResponse.json({ detail: 'Server error' }, { status: 500 });
    })
  ],
};

// In test
server.use(...myEndpointErrors.serverError());
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
