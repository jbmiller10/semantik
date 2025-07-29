# Testing Suite Improvement Tickets

## Overview
This document contains detailed tickets for improving the Semantik testing suite based on the comprehensive review conducted. Tickets are organized by priority and timeline (short-term: 2-3 weeks, medium-term: 1-2 months).

---

## Short-Term Tickets (Weeks 1-3)

### TICKET-001: Delete Outdated Frontend Store Tests and Create React Query Hook Tests

<priority>Critical</priority>
<effort>8-10 hours</effort>
<labels>testing, frontend, tech-debt, critical</labels>

<problem-statement>
The `collectionStore.test.ts` file is testing methods that no longer exist in the refactored code (`fetchCollections`, `createCollection`, etc.). The actual data fetching logic has moved to React Query hooks, but these hooks have zero test coverage.
</problem-statement>

<scope>
- All React Query hooks in `/apps/webui-react/src/hooks/` directory
- Delete outdated store tests in `/apps/webui-react/src/stores/__tests__/collectionStore.test.ts`
- Focus on collection-related hooks first, then expand to other data fetching hooks
- Does NOT include component tests that use these hooks (separate ticket)
</scope>

<acceptance-criteria>
- [ ] Delete the outdated `apps/webui-react/src/stores/__tests__/collectionStore.test.ts`
- [ ] Create comprehensive tests for all React Query hooks in `/hooks/`:
  - `useCollections` (list fetching, auto-refetch)
  - `useCreateCollection` (optimistic updates, rollback)
  - `useUpdateCollection` (partial updates, cache management)
  - `useDeleteCollection` (cascade invalidation)
  - `useCollectionOperations`
  - `useCollectionDocuments`
  - `useOperationProgress`
- [ ] Test coverage should include:
  - Loading states
  - Error handling
  - Optimistic updates and rollbacks
  - Cache invalidation patterns
  - Retry logic
</acceptance-criteria>

<technical-details>
```typescript
// Example test structure for useCreateCollection
import { renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { useCreateCollection } from '../../hooks/useCollections'

describe('useCreateCollection', () => {
  it('should handle optimistic updates correctly', async () => {
    const { result } = renderHook(() => useCreateCollection(), {
      wrapper: ({ children }) => (
        <QueryClientProvider client={queryClient}>
          {children}
        </QueryClientProvider>
      ),
    })
    
    // Test optimistic update
    act(() => {
      result.current.mutate({ name: 'New Collection' })
    })
    
    // Verify cache is updated optimistically
    const cachedData = queryClient.getQueryData(['collections'])
    expect(cachedData).toContainEqual(
      expect.objectContaining({ name: 'New Collection', id: expect.stringContaining('temp-') })
    )
  })
})
```
</technical-details>

<resources>
- [React Query Testing Guide](https://tanstack.com/query/latest/docs/react/guides/testing)
- Current hook implementations in `/apps/webui-react/src/hooks/`
- MSW for mocking API responses
</resources>

---

### TICKET-002: Add Unit Tests for Core Backend Services

<priority>Critical</priority>
<effort>10-12 hours</effort>
<labels>testing, backend, critical</labels>

<problem-statement>
Critical backend services have no unit tests despite being central to the architecture:
- `collection_service.py` - Heart of the new collection-centric architecture
- `directory_scan_service.py` - Handles file system operations
- `search_service.py` - Core search functionality
- `websocket_manager.py` - Real-time communication
</problem-statement>

<scope>
- Unit tests for services in `/packages/webui/services/` directory
- Focus on testing business logic, not infrastructure
- Mock all external dependencies (database, Qdrant, Redis, filesystem)
- Does NOT include integration tests with real services (separate ticket)
- Does NOT include API endpoint tests (those belong with route handlers)
</scope>

<acceptance-criteria>
- [ ] Create comprehensive unit tests for `collection_service.py`:
  - Collection CRUD operations
  - Permission checking
  - State transitions
  - Error handling
- [ ] Create tests for `directory_scan_service.py`:
  - Path validation
  - Permission checking
  - File discovery logic
  - Error scenarios
- [ ] Create tests for `search_service.py`:
  - Query building
  - Result processing
  - Multi-collection search
  - Reranking integration
- [ ] Create tests for `websocket_manager.py`:
  - Connection management
  - Message routing
  - Error handling
  - Reconnection logic
</acceptance-criteria>

<technical-details>
```python
# Example test structure for collection_service
import pytest
from unittest.mock import Mock, AsyncMock
from packages.webui.services.collection_service import CollectionService

class TestCollectionService:
    @pytest.fixture
    def service(self):
        return CollectionService(
            collection_repo=Mock(),
            operation_service=Mock(),
            qdrant_manager=Mock()
        )
    
    async def test_create_collection_with_valid_data(self, service):
        # Arrange
        collection_data = {
            "name": "Test Collection",
            "chunk_size": 512,
            "chunk_overlap": 50
        }
        service.collection_repo.create = AsyncMock(return_value=Mock(id="123"))
        
        # Act
        result = await service.create_collection(collection_data, user_id=1)
        
        # Assert
        assert result.id == "123"
        service.collection_repo.create.assert_called_once()
```
</technical-details>

<resources>
- Service implementations in `/packages/webui/services/`
- Repository patterns in `/shared/database/repositories/`
- Existing test patterns in `/tests/unit/`
</resources>

---

### TICKET-003: Implement Proper Integration Tests with Real Database

<priority>High</priority>
<effort>8-10 hours</effort>
<labels>testing, integration, backend</labels>

<problem-statement>
Current "integration" tests mock the database layer heavily, testing mock configuration rather than actual integration between components. This misses critical issues like transaction handling, race conditions, and data consistency.
</problem-statement>

<scope>
- Integration tests using real PostgreSQL test database
- Tests for service → repository → database flow
- Focus on collection and operation workflows
- Does NOT include E2E tests (separate ticket)
- Does NOT include performance tests (separate ticket)
</scope>

<acceptance-criteria>
- [ ] Create integration test framework using real PostgreSQL test database
- [ ] Implement transaction rollback between tests for isolation
- [ ] Create integration tests for:
  - Collection lifecycle (create → update → delete)
  - Concurrent operations on same collection
  - Transaction rollback scenarios
  - Permission cascade effects
  - Search with actual data
- [ ] Add proper cleanup mechanisms
- [ ] Document integration test setup process
</acceptance-criteria>

<technical-details>
```python
# Example integration test with real database
import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from tests.integration.fixtures import test_db, test_user

@pytest.mark.integration
class TestCollectionIntegration:
    async def test_concurrent_collection_updates(self, test_db: AsyncSession):
        # Create collection
        collection = await create_test_collection(test_db, "Test Collection")
        
        # Simulate concurrent updates
        async with test_db.begin() as tx1, test_db.begin() as tx2:
            # Both transactions try to update
            await update_collection_name(tx1, collection.id, "Name 1")
            await update_collection_name(tx2, collection.id, "Name 2")
            
            # Verify proper locking/conflict resolution
            with pytest.raises(OptimisticLockError):
                await tx2.commit()
```
</technical-details>

<resources>
- PostgreSQL test container setup
- Alembic migrations for test database
- Transaction isolation documentation
</resources>

---

### TICKET-004: Create Centralized Test Utilities and Mock Factories

<priority>High</priority>
<effort>6-8 hours</effort>
<labels>testing, tech-debt, frontend, backend</labels>

<problem-statement>
Test utilities and mock data are duplicated across test files, leading to maintenance burden and inconsistencies. Common patterns like `renderWithQueryClient`, mock collections, and error handlers are defined multiple times.
</problem-statement>

<scope>
- Create shared test utilities for both frontend and backend tests
- Consolidate mock data factories
- Standardize test rendering utilities
- Does NOT include updating all existing tests (follow-up tickets)
- Focus on most commonly used utilities first
</scope>

<acceptance-criteria>
- [ ] Create `/tests/utils/` directory structure:
  - `test-factories.ts` - Data factories for consistent test data
  - `render-utils.tsx` - Rendering utilities with providers
  - `mock-factories.ts` - Mock function factories
  - `assertions.ts` - Custom Jest matchers
- [ ] Migrate all tests to use centralized utilities
- [ ] Remove duplicate definitions
- [ ] Add TypeScript types for all utilities
- [ ] Document usage patterns
</acceptance-criteria>

<technical-details>
```typescript
// test-factories.ts
export const createMockCollection = (overrides?: Partial<Collection>): Collection => ({
  id: 'test-collection-id',
  uuid: 'test-collection-uuid',
  name: 'Test Collection',
  status: 'ready',
  created_at: new Date().toISOString(),
  updated_at: new Date().toISOString(),
  ...overrides
})

// render-utils.tsx
export const renderWithProviders = (
  ui: React.ReactElement,
  options?: RenderOptions
) => {
  const queryClient = createTestQueryClient()
  
  return render(
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <ErrorBoundary>
          {ui}
        </ErrorBoundary>
      </BrowserRouter>
    </QueryClientProvider>,
    options
  )
}

// mock-factories.ts
export const createMockMutation = <TData = any, TVariables = any>(
  overrides?: Partial<UseMutationResult<TData, Error, TVariables>>
) => ({
  mutate: vi.fn(),
  mutateAsync: vi.fn(),
  isPending: false,
  isError: false,
  isSuccess: false,
  ...overrides
})
```
</technical-details>

<resources>
- Current test utilities scattered in test files
- React Testing Library best practices
- Jest custom matchers documentation
</resources>

---

### TICKET-005: Add WebSocket Comprehensive Testing Suite

<priority>High</priority>
<effort>8-10 hours</effort>
<labels>testing, websocket, real-time</labels>

<problem-statement>
WebSocket functionality has minimal test coverage. Only error handling is tested, missing critical scenarios like connection lifecycle, message routing, reconnection logic, and concurrent connections.
</problem-statement>

<scope>
- WebSocket connection lifecycle tests
- Message handling and routing tests
- Reconnection and error recovery tests
- Does NOT include load testing (performance ticket)
- Focus on both client-side (React hook) and server-side (WebSocket manager)
</scope>

<acceptance-criteria>
- [ ] Create WebSocket test utilities for mocking WebSocket connections
- [ ] Add tests for connection lifecycle:
  - Initial connection
  - Authentication
  - Heartbeat/ping-pong
  - Graceful disconnection
- [ ] Add tests for message handling:
  - Operation progress updates
  - Error messages
  - Invalid message formats
- [ ] Add tests for reconnection:
  - Exponential backoff
  - Max retry limits
  - State recovery after reconnection
- [ ] Add tests for concurrent connections:
  - Multiple tabs/windows
  - Connection limits
  - Message synchronization
</acceptance-criteria>

<technical-details>
```typescript
// Mock WebSocket implementation for testing
class MockWebSocket {
  readyState = WebSocket.CONNECTING
  listeners: Record<string, Function[]> = {}
  
  constructor(public url: string) {
    setTimeout(() => {
      this.readyState = WebSocket.OPEN
      this.emit('open')
    }, 10)
  }
  
  send(data: string) {
    const message = JSON.parse(data)
    // Process message and emit response
  }
  
  addEventListener(event: string, handler: Function) {
    this.listeners[event] = [...(this.listeners[event] || []), handler]
  }
  
  emit(event: string, data?: any) {
    this.listeners[event]?.forEach(handler => handler(data))
  }
}

// Example test
it('should handle reconnection with exponential backoff', async () => {
  let attemptCount = 0
  const mockConnect = vi.fn(() => {
    attemptCount++
    if (attemptCount < 3) throw new Error('Connection failed')
    return new MockWebSocket('ws://localhost')
  })
  
  const { result } = renderHook(() => useWebSocket('/operations/123'))
  
  await waitFor(() => {
    expect(mockConnect).toHaveBeenCalledTimes(3)
    expect(result.current.isConnected).toBe(true)
  })
})
```
</technical-details>

<resources>
- WebSocket manager implementation
- Mock WebSocket libraries
- Reconnection pattern best practices
</resources>

---

### TICKET-006: Fix Test Cleanup Mechanisms

<priority>High</priority>
<effort>4-6 hours</effort>
<labels>testing, tech-debt, stability</labels>

<problem-statement>
Tests don't properly clean up after themselves, leaving behind:
- Redis streams from WebSocket tests
- Qdrant collections on test failure
- Temporary files from document processing
- Database records from failed transactions
</problem-statement>

<scope>
- Global test cleanup configuration
- Resource tracking during tests
- Cleanup verification
- Does NOT include rewriting existing tests
- Focus on preventing future test pollution
</scope>

<acceptance-criteria>
- [ ] Implement global test cleanup hooks
- [ ] Add cleanup for Redis:
  - Clear test-specific keys
  - Remove test streams
- [ ] Add cleanup for Qdrant:
  - Track created collections
  - Delete on test completion
- [ ] Add cleanup for filesystem:
  - Track temp files
  - Remove after tests
- [ ] Add cleanup for database:
  - Ensure transaction rollback
  - Clear test data
- [ ] Add cleanup verification tests
</acceptance-criteria>

<technical-details>
```typescript
// Global cleanup configuration
// jest.setup.ts
import { cleanup } from '@testing-library/react'

// Track resources for cleanup
const testResources = {
  qdrantCollections: new Set<string>(),
  tempFiles: new Set<string>(),
  redisKeys: new Set<string>()
}

// Cleanup after each test
afterEach(async () => {
  cleanup() // React Testing Library cleanup
  
  // Clean Qdrant collections
  for (const collection of testResources.qdrantCollections) {
    await qdrantClient.deleteCollection(collection).catch(() => {})
  }
  
  // Clean Redis
  for (const key of testResources.redisKeys) {
    await redisClient.del(key).catch(() => {})
  }
  
  // Clean temp files
  for (const file of testResources.tempFiles) {
    await fs.unlink(file).catch(() => {})
  }
  
  // Clear sets
  testResources.qdrantCollections.clear()
  testResources.tempFiles.clear()
  testResources.redisKeys.clear()
})

// Provide helpers for tests
export const trackQdrantCollection = (name: string) => {
  testResources.qdrantCollections.add(name)
}
```
</technical-details>

<resources>
- Jest setup and teardown documentation
- Current cleanup attempts in tests
- Resource tracking patterns
</resources>

---

## Medium-Term Tickets (Weeks 4-8)

### TICKET-007: Implement Performance Testing Framework

<priority>Medium</priority>
<effort>10-12 hours</effort>
<labels>testing, performance, infrastructure</labels>

<problem-statement>
No performance tests exist to verify system behavior under load. Critical operations like concurrent collection creation, large file processing, and vector search performance are untested.
</problem-statement>

<scope>
- Performance testing framework setup (k6 or Artillery)
- API endpoint performance tests
- Vector search performance tests
- WebSocket scalability tests
- Does NOT include infrastructure changes
- Focus on establishing baselines first
</scope>

<acceptance-criteria>
- [ ] Set up performance testing framework (e.g., k6, Artillery)
- [ ] Create performance tests for:
  - Concurrent collection operations (100+ simultaneous)
  - Large document processing (1GB+ files)
  - Vector search with large result sets (10k+ results)
  - WebSocket connection scaling (1000+ connections)
  - API endpoint response times under load
- [ ] Establish performance baselines
- [ ] Create performance regression detection
- [ ] Add performance metrics to CI/CD pipeline
- [ ] Document performance testing procedures
</acceptance-criteria>

<technical-details>
```javascript
// k6 performance test example
import http from 'k6/http'
import { check, sleep } from 'k6'
import { Rate } from 'k6/metrics'

const errorRate = new Rate('errors')

export const options = {
  stages: [
    { duration: '2m', target: 100 }, // Ramp up
    { duration: '5m', target: 100 }, // Stay at 100 users
    { duration: '2m', target: 0 },   // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<5000'], // 95% of requests under 5s
    errors: ['rate<0.1'],              // Error rate under 10%
  },
}

export default function () {
  // Create collection
  const createRes = http.post(
    `${__ENV.API_URL}/api/v2/collections`,
    JSON.stringify({
      name: `Load Test Collection ${__VU}-${__ITER}`,
      chunk_size: 512,
    }),
    {
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${__ENV.API_TOKEN}`,
      },
    }
  )
  
  check(createRes, {
    'collection created': (r) => r.status === 201,
  })
  
  errorRate.add(createRes.status !== 201)
  
  sleep(1)
}
```
</technical-details>

<resources>
- k6 documentation
- Performance testing best practices
- Current API endpoints documentation
- Infrastructure capacity planning
</resources>

---

### TICKET-008: Add Contract Testing for API Validation

<priority>Medium</priority>
<effort>8-10 hours</effort>
<labels>testing, api, contracts</labels>

<problem-statement>
No tests verify that API responses match documented schemas. This can lead to frontend-backend mismatches and breaking changes going unnoticed.
</problem-statement>

<scope>
- OpenAPI schema generation for all endpoints
- Contract validation tests
- Request/response validation
- Does NOT include API redesign
- Focus on v2 API endpoints first
</scope>

<acceptance-criteria>
- [ ] Set up contract testing framework (e.g., Pact, OpenAPI validation)
- [ ] Generate OpenAPI schemas for all API endpoints
- [ ] Create contract tests for:
  - All collection endpoints
  - Search API responses
  - WebSocket message formats
  - Authentication endpoints
- [ ] Add request validation tests
- [ ] Add response validation tests
- [ ] Integrate with CI/CD pipeline
- [ ] Document contract testing process
</acceptance-criteria>

<technical-details>
```python
# OpenAPI validation example
import pytest
from openapi_core import Spec
from openapi_core.validation.request import openapi_request_validator
from openapi_core.validation.response import openapi_response_validator

class TestAPIContracts:
    @pytest.fixture
    def openapi_spec(self):
        with open('openapi.yaml') as f:
            return Spec.create(yaml.safe_load(f))
    
    async def test_create_collection_contract(self, client, openapi_spec):
        # Make request
        response = await client.post(
            "/api/v2/collections",
            json={"name": "Test Collection", "chunk_size": 512}
        )
        
        # Validate request
        request = openapi_request_validator(openapi_spec)
        request.validate(response.request)
        
        # Validate response
        validator = openapi_response_validator(openapi_spec)
        result = validator.validate(response)
        
        assert result.errors == []
        assert response.status_code == 201
```
</technical-details>

<resources>
- OpenAPI specification
- Contract testing tools comparison
- API documentation
- Schema generation tools
</resources>

---

### TICKET-009: Create Comprehensive E2E Test Suite with Playwright

<priority>Medium</priority>
<effort>12-15 hours</effort>
<labels>testing, e2e, frontend</labels>

<problem-statement>
Only one E2E test file exists (`errorFlows.e2e.test.tsx`). Critical user journeys, cross-browser compatibility, and mobile responsiveness are untested.
</problem-statement>

<scope>
- Playwright test infrastructure setup
- Critical user journey tests
- Cross-browser testing setup
- Visual regression testing
- Does NOT include fixing found issues
- Focus on happy paths first, then error scenarios
</scope>

<acceptance-criteria>
- [ ] Set up Playwright test infrastructure
- [ ] Create E2E tests for critical user journeys:
  - Complete onboarding flow
  - Create collection → Add data → Search → View results
  - Multi-collection search workflow
  - Settings management
  - Error recovery flows
- [ ] Add cross-browser tests (Chrome, Firefox, Safari)
- [ ] Add mobile viewport tests
- [ ] Add visual regression tests
- [ ] Create page object models for maintainability
- [ ] Integrate with CI/CD pipeline
- [ ] Document E2E testing procedures
</acceptance-criteria>

<technical-details>
```typescript
// Page object model example
export class CollectionsPage {
  constructor(private page: Page) {}
  
  async createCollection(name: string) {
    await this.page.click('button:has-text("Create Collection")')
    await this.page.fill('[aria-label="Collection name"]', name)
    await this.page.click('button:has-text("Create")')
    await this.page.waitForResponse('/api/v2/collections')
  }
  
  async searchCollections(query: string) {
    await this.page.fill('[placeholder="Search collections"]', query)
    await this.page.waitForTimeout(300) // Debounce
  }
  
  async openCollection(name: string) {
    await this.page.click(`[data-testid="collection-card"]:has-text("${name}")`)
  }
}

// E2E test example
test.describe('Collection Management Flow', () => {
  test('should create collection and add data', async ({ page }) => {
    const collections = new CollectionsPage(page)
    
    // Create collection
    await collections.createCollection('My Documents')
    
    // Verify creation
    await expect(page.locator('text=My Documents')).toBeVisible()
    
    // Add data
    await page.click('button:has-text("Add Data")')
    await page.fill('[aria-label="Source path"]', '/home/user/documents')
    await page.click('button:has-text("Start Indexing")')
    
    // Wait for indexing
    await page.waitForSelector('text=Indexing complete', { timeout: 60000 })
  })
})
```
</technical-details>

<resources>
- Playwright documentation
- Page object model patterns
- Current E2E test examples
- CI/CD integration guides
</resources>

---

### TICKET-010: Add Accessibility Testing Suite

<priority>Medium</priority>
<effort>6-8 hours</effort>
<labels>testing, accessibility, frontend</labels>

<problem-statement>
Current accessibility testing is limited to basic aria-label checks. No comprehensive testing for keyboard navigation, screen reader compatibility, or WCAG compliance.
</problem-statement>

<scope>
- Automated accessibility testing with axe-core
- Keyboard navigation tests
- Screen reader compatibility tests
- WCAG 2.1 AA compliance
- Does NOT include fixing found issues
- Focus on critical user paths first
</scope>

<acceptance-criteria>
- [ ] Integrate axe-core for automated accessibility testing
- [ ] Add accessibility tests for all components:
  - Keyboard navigation
  - Screen reader announcements
  - Focus management
  - Color contrast
  - ARIA attributes
- [ ] Create manual accessibility test checklist
- [ ] Add accessibility tests to component test suite
- [ ] Create accessibility regression tests
- [ ] Document accessibility testing standards
</acceptance-criteria>

<technical-details>
```typescript
// Accessibility test utilities
import { axe, toHaveNoViolations } from 'jest-axe'

expect.extend(toHaveNoViolations)

// Component accessibility test
describe('CollectionCard Accessibility', () => {
  it('should have no accessibility violations', async () => {
    const { container } = render(
      <CollectionCard collection={mockCollection} />
    )
    
    const results = await axe(container)
    expect(results).toHaveNoViolations()
  })
  
  it('should be keyboard navigable', async () => {
    render(<CollectionCard collection={mockCollection} />)
    
    const card = screen.getByRole('article')
    const editButton = screen.getByRole('button', { name: /edit/i })
    
    // Tab to card
    await userEvent.tab()
    expect(card).toHaveFocus()
    
    // Tab to edit button
    await userEvent.tab()
    expect(editButton).toHaveFocus()
    
    // Activate with Enter
    await userEvent.keyboard('{Enter}')
    expect(mockOnEdit).toHaveBeenCalled()
  })
  
  it('should announce status changes to screen readers', async () => {
    const { rerender } = render(
      <CollectionCard collection={{ ...mockCollection, status: 'indexing' }} />
    )
    
    // Status change should be announced
    rerender(
      <CollectionCard collection={{ ...mockCollection, status: 'ready' }} />
    )
    
    const liveRegion = screen.getByRole('status')
    expect(liveRegion).toHaveTextContent('Collection is now ready')
  })
})
```
</technical-details>

<resources>
- WCAG 2.1 guidelines
- axe-core documentation
- React accessibility best practices
- Screen reader testing guides
</resources>

---

### TICKET-011: Establish Testing Standards and Documentation

<priority>Medium</priority>
<effort>6-8 hours</effort>
<labels>testing, documentation, standards</labels>

<problem-statement>
No standardized testing patterns or documentation exists, leading to inconsistent test quality and duplicated effort. New developers lack guidance on testing practices.
</problem-statement>

<scope>
- Testing standards documentation
- Test naming conventions
- Testing style guide
- PR checklist for tests
- Does NOT include enforcing standards retroactively
- Focus on guiding future development
</scope>

<acceptance-criteria>
- [ ] Create comprehensive testing documentation:
  - Testing philosophy and principles
  - Test organization structure
  - Naming conventions
  - When to use unit vs integration vs E2E tests
  - Mock vs real dependencies guidelines
- [ ] Create testing style guide with examples
- [ ] Set up ESLint rules for test files
- [ ] Create test templates for common scenarios
- [ ] Add testing checklist for PRs
- [ ] Document CI/CD test pipeline
</acceptance-criteria>

<technical-details>
```markdown
# Testing Standards Documentation

## Test Organization
```
tests/
├── unit/           # Isolated unit tests
├── integration/    # Integration tests with real dependencies
├── e2e/           # End-to-end user journey tests
└── utils/         # Shared test utilities

apps/webui-react/
└── src/
    └── components/
        └── __tests__/  # Component tests co-located
```

## Naming Conventions
- Test files: `ComponentName.test.tsx` or `module_name_test.py`
- Test descriptions: Use "should" format
- Group related tests with `describe` blocks

## Example Test Template
```typescript
describe('ComponentName', () => {
  // Setup shared across tests
  beforeEach(() => {
    // Reset mocks, clear stores, etc.
  })
  
  describe('when condition X', () => {
    it('should behavior Y', () => {
      // Arrange
      // Act
      // Assert
    })
  })
  
  describe('error handling', () => {
    it('should show error message when API fails', () => {
      // Test error scenarios
    })
  })
})
```

## Mock Guidelines
- Mock external dependencies (APIs, file system)
- Use real implementations for internal modules when possible
- Document why certain mocks are necessary
```
</technical-details>

<resources>
- Testing best practices documentation
- ESLint testing plugins
- Team testing preferences
- Industry testing standards
</resources>

---

## Implementation Notes

1. **Dependencies Between Tickets**: 
   - TICKET-004 (test utilities) should be completed before migrating other tests
   - TICKET-001 (React Query tests) depends on having proper test utilities
   - TICKET-011 (standards) should be established early to guide other work

2. **Resource Requirements**:
   - Docker for test databases and services
   - CI/CD pipeline updates for new test types
   - Additional npm packages for testing tools
   - Team training on new testing practices

3. **Success Metrics**:
   - Code coverage increase from current to 80%+
   - Reduction in production bugs by 50%
   - Faster development velocity due to confidence in tests
   - Reduced time spent on manual testing

4. **Risk Mitigation**:
   - Start with highest-risk areas (core services, data mutations)
   - Implement incrementally to avoid disrupting development
   - Run new tests in parallel with existing suite initially
   - Monitor test execution time to prevent slow CI/CD

Each ticket is designed to be self-contained with clear acceptance criteria and technical guidance, allowing any developer to pick it up and start implementation immediately.