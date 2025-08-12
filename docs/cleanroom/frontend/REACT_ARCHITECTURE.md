# Semantik React Frontend Architecture - Cleanroom Documentation

## 1. Component Overview

### System Description
The Semantik webui-react application is a React 19.1.0 single-page application (SPA) that provides a secure, type-safe interface for managing document embeddings, collections, and semantic search operations. Built with TypeScript 5.8.3 in strict mode, it implements a collection-centric architecture where all operations are scoped to collections rather than standalone jobs.

### Technology Foundation
```json
{
  "framework": "React 19.1.0",
  "language": "TypeScript 5.8.3",
  "build": "Vite 7.0.0",
  "state": "Zustand 5.0.6 + React Query 5.81.5",
  "routing": "React Router DOM 7.6.3",
  "styling": "TailwindCSS 3.4.17",
  "testing": "Vitest 2.1.9 + MSW 2.10.4"
}
```

### Project Structure
```
apps/webui-react/
├── src/
│   ├── main.tsx                 # Entry point with StrictMode
│   ├── App.tsx                  # Router & providers setup
│   ├── components/              # UI components (45+ files)
│   │   ├── chunking/           # Chunking strategy components
│   │   └── __tests__/          # Component test files
│   ├── hooks/                   # Custom React hooks
│   ├── pages/                   # Route page components
│   ├── services/                # API integration layer
│   │   └── api/v2/             # V2 API clients
│   ├── stores/                  # Zustand state stores
│   ├── tests/                   # Test utilities & mocks
│   │   └── mocks/              # MSW handlers
│   ├── types/                   # TypeScript definitions
│   └── utils/                   # Utility functions
├── vite.config.ts              # Vite build configuration
├── vitest.config.ts            # Test runner configuration
├── tsconfig.json               # TypeScript configuration
├── tailwind.config.js          # TailwindCSS configuration
└── package.json                # Dependencies & scripts
```

## 2. Architecture & Design Patterns

### Component Composition Hierarchy
```typescript
// Root application structure
<StrictMode>
  <App>
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <BrowserRouter>
          <Routes>
            <Route path="/login" element={<LoginPage />} />
            <Route path="/verification" element={<VerificationPage />} />
            <Route path="/" element={<ProtectedRoute><Layout /></ProtectedRoute>}>
              <Route index element={<HomePage />} />
              <Route path="settings" element={<SettingsPage />} />
            </Route>
          </Routes>
        </BrowserRouter>
      </QueryClientProvider>
    </ErrorBoundary>
  </App>
</StrictMode>
```

### State Management Architecture
The application uses a hybrid state management approach:

1. **Server State**: React Query for API data caching and synchronization
2. **Client State**: Zustand stores for UI and application state
3. **Component State**: useState/useReducer for local component state

```typescript
// Zustand store pattern with TypeScript
interface AuthState {
  token: string | null;
  refreshToken: string | null;
  user: User | null;
  setAuth: (token: string, user: User, refreshToken?: string) => void;
  logout: () => Promise<void>;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      token: null,
      refreshToken: null,
      user: null,
      setAuth: (token, user, refreshToken) => 
        set({ token, user, refreshToken: refreshToken || null }),
      logout: async () => {
        // API call then clear state
        set({ token: null, refreshToken: null, user: null });
        localStorage.removeItem('auth-storage');
      }
    }),
    { name: 'auth-storage' }
  )
);
```

### Hook Patterns
Custom hooks encapsulate complex logic and provide clean APIs:

```typescript
// WebSocket hook with auto-reconnect
export function useWebSocket(
  url: string | null,
  options: UseWebSocketOptions = {}
) {
  const ws = useRef<WebSocket | null>(null);
  const reconnectCount = useRef(0);
  const [readyState, setReadyState] = useState<number>(WebSocket.CONNECTING);
  
  // Connection logic with timeout handling
  const connect = useCallback(() => {
    if (!url) return;
    
    ws.current = new WebSocket(url);
    const timeoutId = setTimeout(() => {
      if (ws.current?.readyState === WebSocket.CONNECTING) {
        ws.current.close();
        onError?.(new Event('timeout'));
      }
    }, 5000);
    
    // Event handlers with auto-reconnect logic
    ws.current.onclose = (event) => {
      if (autoReconnect && reconnectCount.current < reconnectAttempts && !event.wasClean) {
        reconnectCount.current++;
        setTimeout(connect, reconnectInterval);
      }
    };
  }, [url, options]);
  
  return { sendMessage, readyState, disconnect, reconnect: connect };
}
```

### Component Design Patterns

#### Container/Presentational Pattern
```typescript
// Container component handles data and logic
function CollectionsDashboard() {
  const { data: collections, isLoading } = useCollections();
  const { createCollection } = useCreateCollection();
  
  if (isLoading) return <CollectionsSkeleton />;
  
  return (
    <div>
      {collections.map(c => (
        <CollectionCard key={c.id} collection={c} />
      ))}
    </div>
  );
}

// Presentational component focuses on UI
function CollectionCard({ collection }: { collection: Collection }) {
  return <div className="p-4 border rounded">{collection.name}</div>;
}
```

## 3. Key Interfaces & Contracts

### Core Type Definitions

```typescript
// Collection entity type
export interface Collection {
  id: string;                      // UUID
  name: string;
  description?: string;
  owner_id: number;
  vector_store_name: string;       // Qdrant collection name
  embedding_model: string;
  quantization: string;            // float32, float16, or int8
  chunk_size: number;              // Deprecated
  chunk_overlap: number;           // Deprecated
  chunking_strategy?: string;      // New field
  chunking_config?: Record<string, number | boolean | string>;
  is_public: boolean;
  status: CollectionStatus;
  status_message?: string;
  metadata?: Record<string, unknown>;
  document_count: number;
  vector_count: number;
  total_size_bytes?: number;
  created_at: string;              // ISO 8601
  updated_at: string;              // ISO 8601
  
  // Frontend-specific fields
  isProcessing?: boolean;
  activeOperation?: Operation;
  initial_operation_id?: string;
}

// Operation tracking type
export interface Operation {
  id: string;
  collection_id: string;
  type: OperationType;
  status: OperationStatus;
  config: Record<string, unknown>;
  error_message?: string;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  progress?: number;               // 0-100
  eta?: number;                    // seconds
}

// Type unions for status fields
export type CollectionStatus = 'pending' | 'ready' | 'processing' | 'error' | 'degraded';
export type OperationType = 'index' | 'append' | 'reindex' | 'remove_source' | 'delete';
export type OperationStatus = 'pending' | 'processing' | 'completed' | 'failed' | 'cancelled';
```

### Component Props Contracts

```typescript
// Modal component props pattern
interface CreateCollectionModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess?: (collection: Collection) => void;
  initialValues?: Partial<CreateCollectionRequest>;
}

// List component props with render props pattern
interface CollectionListProps<T extends Collection> {
  collections: T[];
  onSelect?: (collection: T) => void;
  renderItem?: (collection: T) => React.ReactNode;
  emptyState?: React.ReactNode;
  loading?: boolean;
}

// Hook return type pattern
interface UseCollectionsReturn {
  collections: Collection[];
  isLoading: boolean;
  error: Error | null;
  refetch: () => Promise<void>;
  invalidate: () => void;
}
```

## 4. Data Flow & Dependencies

### API Data Flow
```
User Action → Component → Custom Hook → React Query → API Client → Backend
                                                ↓
                                          Axios Interceptors
                                                ↓
                                          Response/Error
                                                ↓
Component Update ← State Update ← Cache Update ← Transform
```

### State Dependencies

```typescript
// Store composition and dependencies
AuthStore (persisted)
  ↓ provides token
API Client
  ↓ fetches data
React Query Cache
  ↓ provides data
Components
  ↓ updates
UIStore (toasts, modals)

// Query invalidation cascade
Collection Delete → Invalidate:
  - collectionKeys.all
  - collectionKeys.lists()
  - operationKeys.forCollection(id)
  - documentKeys.forCollection(id)
```

### WebSocket Event Flow

```typescript
// Real-time update flow
WebSocket Server → Message Event → Hook Handler → State Update → UI Update

// Example: Operation progress tracking
const { progress, status } = useOperationProgress(operationId, {
  onProgress: (data) => {
    // Update local state
    setProgress(data.progress);
  },
  onComplete: () => {
    // Invalidate queries to refresh data
    queryClient.invalidateQueries(collectionKeys.detail(collectionId));
  },
  onError: (error) => {
    // Show error toast
    addToast({ type: 'error', message: error.message });
  }
});
```

## 5. Critical Implementation Details

### Routing Configuration

```typescript
// Protected route implementation
function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const token = useAuthStore((state) => state.token);
  
  if (!token) {
    return <Navigate to="/login" replace />;
  }
  
  return <>{children}</>;
}

// Nested routing with Layout
<Route path="/" element={<ProtectedRoute><Layout /></ProtectedRoute>}>
  <Route index element={<HomePage />} />
  <Route path="settings" element={<SettingsPage />} />
</Route>
```

### Error Boundary Implementation

```typescript
class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Uncaught error:', error, errorInfo);
  }

  public render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen flex items-center justify-center bg-red-50">
          <div className="max-w-md w-full space-y-4 p-8">
            <h1 className="text-2xl font-bold text-red-600">Something went wrong</h1>
            <p className="text-gray-600">{this.state.error?.message}</p>
            <button onClick={() => window.location.reload()}>Reload page</button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
```

### Lazy Loading Pattern

```typescript
// Route-level code splitting
const HomePage = lazy(() => import('./pages/HomePage'));
const SettingsPage = lazy(() => import('./pages/SettingsPage'));

// Component-level lazy loading for modals
const DocumentViewerModal = lazy(() => import('./components/DocumentViewerModal'));

// Usage with Suspense
<Suspense fallback={<LoadingSpinner />}>
  <Routes>
    <Route path="/" element={<HomePage />} />
  </Routes>
</Suspense>
```

### Local Storage Migration

```typescript
// Version migration on app startup (main.tsx)
export function checkAndMigrateLocalStorage() {
  const CURRENT_VERSION = '2.0.0';
  const storedVersion = localStorage.getItem('app_version');
  
  if (storedVersion !== CURRENT_VERSION) {
    // Perform migrations
    if (!storedVersion || storedVersion < '2.0.0') {
      // Migrate from job-centric to collection-centric
      const oldData = localStorage.getItem('job-storage');
      if (oldData) {
        localStorage.removeItem('job-storage');
        // Transform and store as collection data
      }
    }
    
    localStorage.setItem('app_version', CURRENT_VERSION);
  }
}
```

## 6. Security Considerations

### XSS Prevention
- React automatically escapes all rendered content
- No use of dangerouslySetInnerHTML except in controlled scenarios
- Content Security Policy headers configured in production

### Authentication Security

```typescript
// Token injection in API client
apiClient.interceptors.request.use(
  (config) => {
    const state = useAuthStore.getState();
    const token = state.token;
    
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    
    return config;
  }
);

// Auto-logout on 401
apiClient.interceptors.response.use(
  (response) => response,
  async (error) => {
    if (error.response?.status === 401) {
      await useAuthStore.getState().logout();
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);
```

### Input Validation

```typescript
// Search validation example
export function validateSearchParams(params: SearchParams): ValidationResult {
  const errors: ValidationError[] = [];
  
  if (!params.query || params.query.trim().length === 0) {
    errors.push({
      field: 'query',
      message: 'Search query is required'
    });
  }
  
  if (params.limit && (params.limit < 1 || params.limit > 100)) {
    errors.push({
      field: 'limit',
      message: 'Limit must be between 1 and 100'
    });
  }
  
  if (params.similarity_threshold && 
      (params.similarity_threshold < 0 || params.similarity_threshold > 1)) {
    errors.push({
      field: 'similarity_threshold',
      message: 'Similarity threshold must be between 0 and 1'
    });
  }
  
  return { valid: errors.length === 0, errors };
}
```

### Secure WebSocket Connections

```typescript
// WebSocket URL construction with auth token
const getWebSocketUrl = (path: string): string => {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const host = window.location.host;
  const token = useAuthStore.getState().token;
  
  return `${protocol}//${host}${path}?token=${encodeURIComponent(token || '')}`;
};
```

## 7. Testing Requirements

### Unit Test Patterns

```typescript
// Component test with mocked hooks
describe('CollectionCard', () => {
  it('renders collection information', () => {
    const mockCollection: Collection = {
      id: '123',
      name: 'Test Collection',
      document_count: 10,
      vector_count: 100,
      status: 'ready'
    };
    
    render(
      <TestWrapper>
        <CollectionCard collection={mockCollection} />
      </TestWrapper>
    );
    
    expect(screen.getByText('Test Collection')).toBeInTheDocument();
    expect(screen.getByText('10 documents')).toBeInTheDocument();
  });
});

// Hook test pattern
describe('useCollections', () => {
  it('fetches and returns collections', async () => {
    const { result } = renderHook(() => useCollections(), {
      wrapper: TestWrapper
    });
    
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });
    
    expect(result.current.collections).toHaveLength(2);
    expect(result.current.collections[0].name).toBe('Test Collection 1');
  });
});
```

### Integration Test Patterns

```typescript
// API integration test with MSW
describe('Collection Operations', () => {
  it('creates collection and shows in list', async () => {
    const user = userEvent.setup();
    
    render(
      <TestWrapper>
        <CollectionsDashboard />
      </TestWrapper>
    );
    
    // Open create modal
    await user.click(screen.getByText('Create Collection'));
    
    // Fill form
    await user.type(screen.getByLabelText('Name'), 'New Collection');
    await user.selectOptions(screen.getByLabelText('Model'), 'Qwen/Qwen3-Embedding-0.6B');
    
    // Submit
    await user.click(screen.getByText('Create'));
    
    // Verify collection appears
    await waitFor(() => {
      expect(screen.getByText('New Collection')).toBeInTheDocument();
    });
  });
});
```

### MSW Mock Configuration

```typescript
// Mock handlers for testing
export const handlers = [
  http.post('/api/auth/login', async ({ request }) => {
    const { username, password } = await request.json();
    
    if (username === 'testuser' && password === 'testpass') {
      return HttpResponse.json({
        access_token: 'mock-jwt-token',
        refresh_token: 'mock-refresh-token',
        user: {
          id: 1,
          username: 'testuser',
          email: 'test@example.com',
          is_active: true
        }
      });
    }
    
    return HttpResponse.json(
      { detail: 'Invalid credentials' },
      { status: 401 }
    );
  }),
  
  http.get('/api/v2/collections', () => {
    return HttpResponse.json({
      collections: mockCollections,
      total: 2,
      page: 1,
      page_size: 10
    });
  })
];
```

### Test Setup Configuration

```typescript
// vitest.setup.ts
import '@testing-library/jest-dom';
import { server } from './src/tests/mocks/server';

beforeAll(() => {
  server.listen({ onUnhandledRequest: 'error' });
});

afterEach(() => {
  server.resetHandlers();
});

afterAll(() => {
  server.close();
});

// Mock browser APIs
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
});
```

## 8. Common Pitfalls & Best Practices

### React 19 Patterns

```typescript
// Use the new React 19 use() hook for async data
import { use } from 'react';

function CollectionDetails({ collectionPromise }) {
  const collection = use(collectionPromise);
  return <div>{collection.name}</div>;
}

// Proper StrictMode usage
createRoot(container).render(
  <StrictMode>
    <App />
  </StrictMode>
);
```

### Performance Optimization

```typescript
// Memo expensive components
const ExpensiveList = memo(({ items }: { items: Item[] }) => {
  return items.map(item => <ItemCard key={item.id} item={item} />);
});

// Use callback for stable references
const handleSearch = useCallback((query: string) => {
  setSearchQuery(query);
  debouncedSearch(query);
}, [debouncedSearch]);

// Optimize re-renders with proper dependencies
useEffect(() => {
  fetchData();
}, [collectionId]); // Only refetch when collectionId changes
```

### Common Mistakes to Avoid

```typescript
// ❌ BAD: Direct state mutation
const handleAdd = (item) => {
  state.items.push(item); // Mutates state
  setState(state);
};

// ✅ GOOD: Create new state
const handleAdd = (item) => {
  setState(prev => ({
    ...prev,
    items: [...prev.items, item]
  }));
};

// ❌ BAD: Async without cleanup
useEffect(() => {
  fetchData().then(setData);
}, []);

// ✅ GOOD: Proper cleanup
useEffect(() => {
  let cancelled = false;
  
  fetchData().then(data => {
    if (!cancelled) setData(data);
  });
  
  return () => { cancelled = true; };
}, []);
```

### Zustand Best Practices

```typescript
// Use selectors to prevent unnecessary re-renders
const token = useAuthStore(state => state.token);
// Instead of: const { token } = useAuthStore();

// Use shallow equality for multiple values
const { user, token } = useAuthStore(
  state => ({ user: state.user, token: state.token }),
  shallow
);

// Clear pattern for async actions
const fetchUser = async () => {
  set({ loading: true, error: null });
  try {
    const user = await api.getUser();
    set({ user, loading: false });
  } catch (error) {
    set({ error: error.message, loading: false });
  }
};
```

## 9. Configuration & Environment

### Vite Configuration

```typescript
// vite.config.ts
export default defineConfig({
  plugins: [react()],
  base: '/',
  build: {
    outDir: '../../packages/webui/static',
    assetsDir: 'assets',
    sourcemap: true,
    emptyOutDir: true,
    rollupOptions: {
      output: {
        manualChunks: undefined, // Let Vite optimize chunks
      },
    },
  },
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8080',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://localhost:8080',
        ws: true,
        changeOrigin: true,
      },
    },
  },
});
```

### TypeScript Configuration

```typescript
// tsconfig.app.json
{
  "compilerOptions": {
    "target": "ES2022",
    "useDefineForClassFields": true,
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    
    /* Bundler mode */
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "verbatimModuleSyntax": true,
    "moduleDetection": "force",
    "noEmit": true,
    "jsx": "react-jsx",
    
    /* Linting */
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "erasableSyntaxOnly": true,
    "noFallthroughCasesInSwitch": true,
    "noUncheckedSideEffectImports": true,
    
    /* Path Aliases */
    "baseUrl": ".",
    "paths": {
      "@/*": ["./src/*"]
    }
  }
}
```

### Environment Variables

```typescript
// Environment variable usage
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';
const WS_BASE_URL = import.meta.env.VITE_WS_BASE_URL || '';
const IS_DEV = import.meta.env.DEV;
const IS_PROD = import.meta.env.PROD;

// Type definitions for env vars
/// <reference types="vite/client" />
interface ImportMetaEnv {
  readonly VITE_API_BASE_URL: string;
  readonly VITE_WS_BASE_URL: string;
}
```

### Build Scripts

```json
{
  "scripts": {
    "dev": "vite",
    "build": "tsc -b && vite build",
    "preview": "vite preview",
    "test": "vitest",
    "test:ui": "vitest --ui",
    "test:coverage": "vitest --coverage",
    "test:ci": "vitest run --reporter=verbose",
    "lint": "eslint .",
    "type-check": "tsc --noEmit"
  }
}
```

## 10. Integration Points

### API Client Integration

```typescript
// V2 API structure
services/api/v2/
├── client.ts       // Axios instance with interceptors
├── types.ts        // TypeScript interfaces
├── auth.ts         // Authentication endpoints
├── collections.ts  // Collection CRUD
├── operations.ts   // Operation management
├── documents.ts    // Document operations
├── search.ts       // Search functionality
├── chunking.ts     // Chunking strategies
├── settings.ts     // User settings
├── system.ts       // System information
└── directoryScan.ts // Directory scanning

// Usage example
import { collectionsV2Api } from '@/services/api/v2/collections';

const { data } = await collectionsV2Api.list({
  page: 1,
  limit: 20,
  sort_by: 'created_at',
  sort_order: 'desc'
});
```

### WebSocket Integration

```typescript
// WebSocket connection for real-time updates
const ws = new WebSocket(`ws://localhost:8080/ws/operations/${operationId}`);

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  switch (data.type) {
    case 'progress':
      updateProgress(data.progress);
      break;
    case 'complete':
      onComplete(data.result);
      break;
    case 'error':
      onError(data.error);
      break;
  }
};
```

### Zustand Store Integration

```typescript
// Store composition pattern
const useAppState = () => {
  const auth = useAuthStore();
  const ui = useUIStore();
  const search = useSearchStore();
  
  return {
    isAuthenticated: !!auth.token,
    user: auth.user,
    activeTab: ui.activeTab,
    searchResults: search.results,
    // Computed values
    canSearch: !!auth.token && search.validationErrors.length === 0
  };
};
```

### React Query Integration

```typescript
// Query key factory pattern
export const queryKeys = {
  collections: {
    all: ['collections'] as const,
    lists: () => [...queryKeys.collections.all, 'list'] as const,
    list: (filters: string) => [...queryKeys.collections.lists(), { filters }] as const,
    details: () => [...queryKeys.collections.all, 'detail'] as const,
    detail: (id: string) => [...queryKeys.collections.details(), id] as const,
  },
  operations: {
    all: ['operations'] as const,
    byCollection: (collectionId: string) => 
      [...queryKeys.operations.all, 'collection', collectionId] as const,
  }
};

// Usage in hooks
export function useCollection(id: string) {
  return useQuery({
    queryKey: queryKeys.collections.detail(id),
    queryFn: () => collectionsV2Api.get(id),
    staleTime: 30000, // 30 seconds
  });
}
```

## Critical Files Reference

### Entry Points
- `/home/john/semantik/apps/webui-react/src/main.tsx` - Application entry with StrictMode
- `/home/john/semantik/apps/webui-react/src/App.tsx` - Router and provider setup
- `/home/john/semantik/apps/webui-react/index.html` - HTML template

### Core Components
- `/home/john/semantik/apps/webui-react/src/components/Layout.tsx` - Main layout with navigation
- `/home/john/semantik/apps/webui-react/src/components/ErrorBoundary.tsx` - Error handling
- `/home/john/semantik/apps/webui-react/src/components/CollectionsDashboard.tsx` - Collections UI
- `/home/john/semantik/apps/webui-react/src/components/SearchInterface.tsx` - Search functionality

### State Management
- `/home/john/semantik/apps/webui-react/src/stores/authStore.ts` - Authentication state
- `/home/john/semantik/apps/webui-react/src/stores/uiStore.ts` - UI state management
- `/home/john/semantik/apps/webui-react/src/stores/searchStore.ts` - Search state
- `/home/john/semantik/apps/webui-react/src/stores/chunkingStore.ts` - Chunking configuration

### API Integration
- `/home/john/semantik/apps/webui-react/src/services/api/v2/client.ts` - Axios configuration
- `/home/john/semantik/apps/webui-react/src/services/api/v2/collections.ts` - Collections API
- `/home/john/semantik/apps/webui-react/src/services/websocket.ts` - WebSocket service

### Configuration
- `/home/john/semantik/apps/webui-react/vite.config.ts` - Build configuration
- `/home/john/semantik/apps/webui-react/vitest.config.ts` - Test configuration
- `/home/john/semantik/apps/webui-react/tsconfig.app.json` - TypeScript configuration
- `/home/john/semantik/apps/webui-react/tailwind.config.js` - Styling configuration

### Testing
- `/home/john/semantik/apps/webui-react/vitest.setup.ts` - Test environment setup
- `/home/john/semantik/apps/webui-react/src/tests/mocks/handlers.ts` - MSW mock handlers
- `/home/john/semantik/apps/webui-react/src/tests/utils/TestWrapper.tsx` - Test utilities

## Implementation Notes

1. **React 19 Features**: The application uses React 19.1.0 with StrictMode enabled for better development warnings
2. **TypeScript Strict Mode**: All code must pass strict TypeScript checks with no implicit any
3. **Collection-Centric**: All operations are scoped to collections, not standalone jobs
4. **Real-time Updates**: WebSocket connections provide live operation progress
5. **Optimistic UI**: Updates show immediately while API calls complete in background
6. **Error Recovery**: Comprehensive error handling with boundaries and API interceptors
7. **Test Coverage**: Components have corresponding test files in __tests__ directories
8. **Performance**: Code splitting, lazy loading, and React Query caching for optimal performance
9. **Security**: XSS protection, secure token handling, input validation
10. **Accessibility**: Semantic HTML, ARIA labels where needed (ongoing improvement)

This document serves as the authoritative technical reference for the Semantik React frontend architecture.