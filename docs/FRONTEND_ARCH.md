# Frontend Architecture

React app for managing document collections and semantic search. Built with TypeScript in `apps/webui-react/`.

### Technology Stack

- **React 19.1.0** - UI framework
- **TypeScript 5.8.3** - Type safety and developer experience
- **Vite 7.0.0** - Build tool and development server
- **Tailwind CSS 3.4.17** - Utility-first CSS framework
- **Zustand 5.0.6** - State management
- **React Router DOM 7.6.3** - Client-side routing
- **React Query 5.81.5** - Server state management
- **Axios 1.10.0** - HTTP client

## Build Configuration

### Vite Configuration (`vite.config.ts`)

```typescript
export default defineConfig(({ mode }) => ({
  plugins: [react()],
  base: '/',
  esbuild: {
    drop: mode === 'production' ? ['console', 'debugger'] : [],
  },
  build: {
    outDir: '../../packages/webui/static',  // Builds to backend static directory
    assetsDir: 'assets',
    sourcemap: true,
    emptyOutDir: true,
    rollupOptions: {
      output: {
        manualChunks: undefined,
      },
    },
  },
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8080',    // Proxies API calls to backend
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://localhost:8080',      // WebSocket proxy
        ws: true,
        changeOrigin: true,
      },
    },
  },
}))
```

### TypeScript Configuration

The project uses a composite TypeScript configuration:
- `tsconfig.json` - Root config with project references
- `tsconfig.app.json` - Application-specific config with strict mode enabled
- `tsconfig.node.json` - Node.js specific config for build tools

Key compiler options:
- Target: ES2022
- Module: ESNext with bundler resolution
- JSX: react-jsx
- Strict mode enabled
- No unused locals/parameters

## Component Architecture

### Directory Structure

```
src/
├── components/          # Reusable UI components
│   ├── chunking/       # Chunking-related components
│   ├── common/         # Shared utility components
│   ├── search/         # Search form components
│   └── __tests__/      # Component tests
├── hooks/              # Custom React hooks
│   └── __tests__/      # Hook tests
├── pages/              # Route-level components
├── services/           # API client and external services
│   └── api/v2/         # Versioned API clients
│       └── __tests__/  # API client tests
├── stores/             # Zustand state stores
│   └── __tests__/      # Store tests
├── tests/              # Test utilities and mocks
│   └── mocks/          # MSW handlers
├── types/              # TypeScript type definitions
├── utils/              # Utility functions
├── App.tsx             # Root component with routing
├── main.tsx            # Application entry point
└── index.css           # Global styles with Tailwind
```

### Component Hierarchy

```
App
├── ErrorBoundary
├── QueryClientProvider
└── Router
    ├── LoginPage (unprotected)
    ├── VerificationPage (unprotected)
    └── Layout (protected)
        ├── HomePage
        │   ├── CollectionsDashboard (collections tab)
        │   │   ├── CollectionCard
        │   │   │   └── CollectionOperations
        │   │   │       └── OperationProgress
        │   │   └── CreateCollectionModal
        │   ├── ActiveOperationsTab (operations tab)
        │   │   └── OperationListItem
        │   └── SearchInterface (search tab)
        │       └── SearchResults
        ├── SettingsPage
        ├── Toast
        ├── DocumentViewerModal
        └── CollectionDetailsModal
```

### Key Components

#### Layout Components

**Layout.tsx**
- Main application shell with header navigation
- Tab-based navigation between Collections, Active Operations, and Search
- Integrates global modals (Toast, DocumentViewer, CollectionDetailsModal)
- Handles authentication state and logout

**ErrorBoundary.tsx**
- Catches React errors and displays fallback UI
- Prevents entire app crashes
- Logs errors to console

#### Collection Management Components

**CollectionsDashboard.tsx**
- Main dashboard for managing collections
- Features:
  - Grid view of all collections
  - Create collection action
  - Search and filter capabilities
  - Real-time status updates via React Query

**CollectionCard.tsx**
- Individual collection display with status indicators
- Shows document count, vector count, and current status
- Embedded CollectionOperations component for recent operations
- Actions: add data, re-index, view details, delete

**CollectionOperations.tsx**
- Displays operations history for a collection
- Real-time progress updates via WebSocket
- Shows active and recent completed operations
- Uses `useOperationProgress` hook for WebSocket connections

**CreateCollectionModal.tsx**
- Form for creating new collections
- Features:
  - Collection name and description
  - Model selection from available embedding models
  - Advanced parameter configuration (chunk size, overlap, quantization)
  - Privacy settings (public/private)

**AddDataToCollectionModal.tsx**
- Interface for adding documents to existing collections
- Directory scanning with WebSocket progress
- File filtering and exclusion patterns
- Real-time scan progress visualization

**ReindexCollectionModal.tsx**
- Allows re-indexing of collection documents
- Options for full or incremental re-indexing
- Parameter adjustments without data loss

#### Search Components

**SearchInterface.tsx**
- Comprehensive search form with:
  - Query input with search tips
  - Collection selection (only shows ready collections)
  - Hybrid search toggle with mode options
  - Result count configuration
- Auto-refreshes collection statuses for processing operations

**SearchResults.tsx**
- Displays search results with score indicators
- Chunk navigation (previous/next)
- Click to view full document functionality
- Empty state handling

#### Modal Components

**DocumentViewerModal.tsx**
- Full document viewer with authentication
- Supports multiple file types:
  - PDF rendering with pdf.js
  - DOCX with docx-preview library
  - Markdown with marked.js
  - Email (.eml) parsing
  - Plain text and code files
- Integrated text search and highlighting
- Authentication token handling for secure document access

**CollectionDetailsModal.tsx**
- Detailed collection information and metrics
- Shows embedding model, parameters, and settings
- Vector database statistics
- Operation history with filtering
- Document management interface

**DocumentViewer.tsx**
- Core document rendering logic
- File type detection and appropriate rendering
- PDF rendering with pdf.js integration
- Email parsing and display

#### Utility Components

**Toast.tsx**
- Global notification system
- Auto-dismiss with configurable duration
- Multiple types: success, error, info, warning
- Stacking for multiple notifications

**FeatureVerification.tsx**
- Development tool for feature testing
- Compares legacy vs React implementations
- API endpoint testing

## State Management

Zustand handles UI state (modals, toasts, tabs). React Query handles server data (collections, operations, search). Clean separation - no duplicated state.

#### authStore.ts
```typescript
interface AuthState {
  token: string | null;
  refreshToken: string | null;
  user: User | null;
  setAuth: (token: string, user: User, refreshToken?: string) => void;
  logout: () => Promise<void>;
}
```
- Persisted to localStorage
- Handles authentication tokens and user data
- Manages logout with API call
- Integrated with API interceptors for automatic token injection

#### collectionStore.ts
```typescript
interface CollectionUIStore {
  // UI State only - server state managed by React Query
  selectedCollectionId: string | null;
  setSelectedCollection: (id: string | null) => void;
  clearStore: () => void;
}
```
- Manages UI-specific collection state only
- Server data handled by `useCollections`, `useCollectionOperations` hooks
- Minimal state to avoid duplication with React Query cache

#### searchStore.ts
```typescript
interface SearchState {
  results: SearchResult[];
  loading: boolean;
  error: string | null;
  searchParams: SearchParams;
  collections: string[];  // Collection IDs for filtering
  failedCollections: FailedCollection[];
  partialFailure: boolean;
  rerankingMetrics: RerankingMetrics | null;
  gpuMemoryError: GPUMemoryError | null;
  validationErrors: ValidationError[];
  touched: Record<string, boolean>;
  abortController: AbortController | null;
  rerankingAvailable: boolean;
  rerankingModelsLoading: boolean;
  setResults: (results: SearchResult[]) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  updateSearchParams: (params: Partial<SearchParams>) => void;
  setCollections: (collections: string[]) => void;
  validateAndUpdateSearchParams: (params: Partial<SearchParams>) => void;
  setFieldTouched: (field: string, isTouched?: boolean) => void;
  clearValidationErrors: () => void;
  hasValidationErrors: () => boolean;
  getValidationError: (field: string) => string | undefined;
  // ... additional methods
}
```
- Manages search state and results
- Stores search parameters and configurations
- Handles hybrid search modes and reranking
- Includes validation state for form fields
- Supports request cancellation via AbortController

#### uiStore.ts
```typescript
interface UIState {
  toasts: Toast[];
  activeTab: 'search' | 'collections' | 'operations';
  showDocumentViewer: { collectionId: string; docId: string; chunkId?: string } | null;
  showCollectionDetailsModal: string | null;
  addToast: (toast: Omit<Toast, 'id'>) => void;
  removeToast: (id: string) => void;
  setActiveTab: (tab: 'search' | 'collections' | 'operations') => void;
  setShowDocumentViewer: (viewer: { collectionId: string; docId: string; chunkId?: string } | null) => void;
  setShowCollectionDetailsModal: (collectionId: string | null) => void;
}
```
- UI-specific state (modals, toasts, tabs)
- Auto-dismiss logic for toasts with configurable duration
- Modal visibility management
- Tab navigation state (Collections, Active Operations, Search)

#### chunkingStore.ts
- Manages chunking configuration state
- Stores preset selections and custom configurations
- Handles chunking preview state

### State Flow

```
User Action → Component → React Query → API → Cache → Re-render
                  ↓                      ↓
             Zustand (UI)          WebSocket (progress)
```

Design rules:
- Each store manages one domain
- Compute in selectors, not stores
- Immutable updates only
- Persist auth/preferences only

### Advanced Zustand Patterns

#### Selector Optimization
```typescript
// Avoid re-renders with specific selectors
const collectionCount = useCollections().data?.length ?? 0;
const activeOperations = useQuery(['active-operations']).data ?? [];

// Shallow equality check for arrays/objects
const searchParams = useSearchStore(
  (state) => state.searchParams,
  shallow // from zustand/shallow
);
```

#### Store Composition
```typescript
// Combine multiple stores
const useAppState = () => {
  const auth = useAuthStore();
  const ui = useUIStore();
  const { data: collections } = useCollections();

  return {
    isAuthenticated: !!auth.token,
    hasActiveOperations: collections?.some(c => c.status === 'processing') ?? false,
    currentView: ui.activeTab,
  };
};
```

#### Async Actions Pattern
```typescript
// Search store example with async actions
interface SearchState {
  results: SearchResult[];
  loading: boolean;
  error: string | null;

  setResults: (results: SearchResult[]) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  clearResults: () => void;
}

const useSearchStore = create<SearchState>((set) => ({
  results: [],
  loading: false,
  error: null,

  setResults: (results) => set({ results }),
  setLoading: (loading) => set({ loading }),
  setError: (error) => set({ error }),
  clearResults: () => set({ results: [], error: null }),
}));
```

#### Middleware Usage
```typescript
// DevTools integration
import { devtools } from 'zustand/middleware';

const useSearchStore = create<SearchState>()(
  devtools(
    (set) => ({
      // ... store implementation
    }),
    { name: 'search-store' }
  )
);

// Persist middleware with custom storage
import { persist } from 'zustand/middleware';

const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      // ... store implementation
    }),
    {
      name: 'auth-storage',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        token: state.token,
        refreshToken: state.refreshToken,
      }),
    }
  )
);
```

### React Query

Manages all server state with caching, auto-refetch, and optimistic updates.

#### Query Key Factory Pattern
```typescript
// Collection query keys
export const collectionKeys = {
  all: ['collections'] as const,
  lists: () => [...collectionKeys.all, 'list'] as const,
  list: (filters?: unknown) => [...collectionKeys.lists(), filters] as const,
  details: () => [...collectionKeys.all, 'detail'] as const,
  detail: (id: string) => [...collectionKeys.details(), id] as const,
};

// Operation query keys
export const operationKeys = {
  all: ['operations'] as const,
  lists: () => [...operationKeys.all, 'list'] as const,
  byCollection: (collectionId: string) => [...operationKeys.lists(), 'collection', collectionId] as const,
  detail: (id: string) => [...operationKeys.all, 'detail', id] as const,
};
```

#### Custom Hooks Architecture

**useCollections.ts**
- `useCollections()` - Fetch all collections with auto-refetch for active operations
- `useCollection(id)` - Fetch single collection details
- `useCreateCollection()` - Create with optimistic updates
- `useUpdateCollection()` - Update with cache synchronization
- `useDeleteCollection()` - Delete with cascade cache cleanup

**useCollectionOperations.ts**
- `useCollectionOperations(collectionId)` - Fetch operations for a collection
- `useOperation(id)` - Fetch single operation details
- `useCancelOperation()` - Cancel running operations
- `useUpdateOperationInCache()` - Utility for WebSocket updates

**useCollectionDocuments.ts**
- `useCollectionDocuments(collectionId, params)` - Paginated document fetching
- `useDeleteDocument()` - Remove documents from collection
- Supports filtering, sorting, and pagination

#### Optimistic Updates Pattern
```typescript
export function useCreateCollection() {
  const queryClient = useQueryClient();
  const { addToast } = useUIStore();

  return useMutation({
    mutationFn: collectionsV2Api.create,
    onMutate: async (newCollection) => {
      // Cancel in-flight queries
      await queryClient.cancelQueries({ queryKey: collectionKeys.lists() });

      // Snapshot for rollback
      const previousCollections = queryClient.getQueryData<Collection[]>(
        collectionKeys.lists()
      );

      // Optimistic update with temporary data
      const tempCollection: Collection = {
        id: `temp-${Date.now()}`,
        ...newCollection,
        status: 'pending',
        created_at: new Date().toISOString(),
      };

      queryClient.setQueryData<Collection[]>(
        collectionKeys.lists(),
        old => [...(old || []), tempCollection]
      );

      return { previousCollections, tempId: tempCollection.id };
    },
    onError: (error, _, context) => {
      // Rollback on failure
      queryClient.setQueryData(
        collectionKeys.lists(),
        context?.previousCollections
      );
      addToast({ type: 'error', message: handleApiError(error) });
    },
    onSuccess: (data, _, context) => {
      // Replace temp with real data
      queryClient.setQueryData<Collection[]>(
        collectionKeys.lists(),
        old => old?.map(c => c.id === context?.tempId ? data : c)
      );
    },
    onSettled: () => {
      // Ensure fresh data
      queryClient.invalidateQueries({ queryKey: collectionKeys.lists() });
    },
  });
}
```

#### Cache Invalidation Strategy
```typescript
// Cascade invalidation for related data
export function useDeleteCollection() {
  return useMutation({
    mutationFn: collectionsV2Api.delete,
    onSuccess: (_, collectionId) => {
      // Remove collection from cache
      queryClient.removeQueries({ queryKey: collectionKeys.detail(collectionId) });

      // Remove related data
      queryClient.removeQueries({ queryKey: operationKeys.byCollection(collectionId) });
      queryClient.removeQueries({ queryKey: ['collection-documents', collectionId] });

      // Refresh lists
      queryClient.invalidateQueries({ queryKey: collectionKeys.lists() });
    },
  });
}
```

#### Auto-refetch Configuration
```typescript
// Smart refetching based on collection status
export function useCollections() {
  return useQuery({
    queryKey: collectionKeys.lists(),
    queryFn: collectionsV2Api.list,
    // Refetch if any collection is processing
    refetchInterval: (query) => {
      const hasActiveOperations = query.state.data?.some(
        (c: Collection) => c.status === 'processing' || c.activeOperation
      );
      return hasActiveOperations ? 30000 : false;
    },
    staleTime: 5000, // Consider stale after 5s
  });
}
```

### State Synchronization

#### WebSocket State Updates
```typescript
// Operation progress hook with WebSocket integration
export function useOperationProgress(
  operationId: string | null,
  options: UseOperationProgressOptions = {}
) {
  const updateOperationInCache = useUpdateOperationInCache();
  const { addToast } = useUIStore();

  const wsUrl = operationId ? operationsV2Api.getWebSocketUrl(operationId) : null;

  const { sendMessage, readyState } = useWebSocket(wsUrl, {
    onMessage: (event) => {
      const rawMessage = JSON.parse(event.data);

      // Handle different message types
      switch (rawMessage.type) {
        case 'operation_completed':
          updateOperationInCache(operationId, { status: 'completed' });
          options.onComplete?.();
          break;

        case 'operation_failed':
          updateOperationInCache(operationId, {
            status: 'failed',
            error_message: rawMessage.data?.error_message
          });
          options.onError?.(rawMessage.data?.error_message);
          break;

        case 'progress_update':
          updateOperationInCache(operationId, {
            progress: rawMessage.data?.progress,
            progress_message: rawMessage.data?.message,
          });
          break;
      }
    },
  });

  return { sendMessage, readyState };
}
```

#### Cross-Tab Synchronization
```typescript
// Broadcast state changes across tabs
const useBroadcastChannel = (channel: string) => {
  const bc = new BroadcastChannel(channel);

  const broadcast = (data: any) => {
    bc.postMessage(data);
  };

  const subscribe = (handler: (data: any) => void) => {
    bc.onmessage = (event) => handler(event.data);
    return () => bc.close();
  };

  return { broadcast, subscribe };
};

// In auth store
const { broadcast } = useBroadcastChannel('auth');

const logout = async () => {
  await authApi.logout();
  set({ token: null, user: null });
  broadcast({ type: 'logout' });
};
```

## API Integration

### API Client (`services/api/v2/client.ts`)

The API client uses Axios with interceptors for:
- Automatic token injection
- 401 response handling with logout
- Base configuration

```typescript
// Request interceptor adds auth token
api.interceptors.request.use((config) => {
  const token = useAuthStore.getState().token;
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});
```

### API Endpoints (v2)

The application uses a versioned API structure under `/api/v2/` with TypeScript-first client implementations:

**collectionsV2Api** (`collections.ts`)
- `list()` - Get all collections with metadata
- `get(id)` - Get single collection details
- `create(data)` - Create new collection
- `update(id, data)` - Update collection metadata
- `delete(id)` - Delete collection and all data
- `getStats(id)` - Get collection statistics

**operationsV2Api** (`operations.ts`)
- `list(filters)` - Get operations with optional filtering
- `get(id)` - Get operation details
- `cancel(id)` - Cancel running operation
- `getWebSocketUrl(id)` - Get WebSocket URL for real-time updates
- `getGlobalWebSocketUrl()` - Get global operations WebSocket URL

**documentsV2Api** (`documents.ts`)
- `list(collectionId, params)` - Paginated document listing
- `get(collectionId, docId)` - Get document with authentication
- `delete(collectionId, docId)` - Remove document from collection
- `search(params)` - Semantic search across collections
- `getDownloadUrl(collectionId, docId)` - Get authenticated download URL

**authApi** (`auth.ts`)
- `login(credentials)` - User authentication
- `register(data)` - New user registration
- `me()` - Get current user profile
- `logout()` - End session

**systemApi** (`system.ts`)
- `getModels()` - Available embedding models
- `getStats()` - System metrics and health
- `checkRerankingSupport()` - Check if reranking is available

**modelsApi** (`models.ts`)
- `getModels()` - Fetch all available embedding models including plugin models

**projectionsV2Api** (`projections.ts`)
- `list(collectionId)` - Get projections for a collection
- `getMetadata(collectionId, projectionId)` - Get projection metadata
- `getArtifact(collectionId, projectionId, artifactName)` - Get projection artifact data
- `start(collectionId, payload)` - Start a new projection computation
- `delete(collectionId, projectionId)` - Delete a projection
- `select(collectionId, projectionId, ids)` - Select points in a projection

**chunkingApi** (`chunking.ts`)
- `preview(request, options)` - Preview chunking for a document with progress tracking
- `compare(request, options)` - Compare multiple chunking strategies
- `getAnalytics(params, options)` - Get chunking analytics
- `getPresets(options)` - Get all presets (system and custom)
- `savePreset(preset, options)` - Save a custom preset
- `deletePreset(presetId, options)` - Delete a custom preset
- `process(request, options)` - Process a document with specific chunking strategy
- `getRecommendation(fileType, options)` - Get recommended strategy for a file type
- `cancelRequest(requestId, reason)` - Cancel a specific request
- `cancelAllRequests(reason)` - Cancel all active requests

**settingsApi** (`settings.ts`)
- `getStats()` - Get system statistics
- `resetDatabase()` - Reset the database

**directoryScanApi** (`directoryScan.ts`)
- `scan(path, options)` - Scan directory for files
- `getWebSocketUrl(sessionId)` - WebSocket for scan progress

## Custom Hooks

### WebSocket Hooks

**useWebSocket.ts**
- Generic WebSocket connection management
- Auto-reconnection with exponential backoff
- Connection state tracking
- Message sending utilities

**useOperationProgress.ts**
- Operation-specific WebSocket for progress updates
- Automatically connects/disconnects based on operation status
- Updates React Query cache with progress data

**useOperationsSocket.ts**
- Global operations WebSocket for all operation updates
- Shared connection to avoid exceeding WebSocket limits
- Broadcasts updates to all listening components

**useDirectoryScanWebSocket.ts**
- Directory scanning via WebSocket
- Real-time scan progress updates
- Error handling and state management

**useChunkingWebSocket.ts**
- Chunking preview WebSocket connection
- Real-time chunking progress updates

### Other Hooks

**useDirectoryScan.ts**
- HTTP-based directory scanning (fallback)
- Returns scan results and loading state

**useProjections.ts**
- Projection data fetching and management
- Handles projection visualization state

**useProjectionTooltip.ts**
- Tooltip state for projection visualizations

**useRerankingAvailability.ts**
- Check if reranking is available on the backend

**useModels.ts**
- Fetch available embedding models

## Routing

### Route Structure

```typescript
<Routes>
  <Route path="/login" element={<LoginPage />} />
  <Route path="/verification" element={<VerificationPage />} />
  <Route path="/" element={<ProtectedRoute><Layout /></ProtectedRoute>}>
    <Route index element={<HomePage />} />
    <Route path="collections/:collectionId" element={<HomePage />} />
    <Route path="settings" element={<SettingsPage />} />
  </Route>
</Routes>
```

### Route Protection

The `ProtectedRoute` component checks for authentication token:
```typescript
function ProtectedRoute({ children }) {
  const token = useAuthStore((state) => state.token);
  if (!token) {
    return <Navigate to="/login" replace />;
  }
  return <>{children}</>;
}
```

## WebSocket Integration

### Connection Management

WebSocket connections are established for:
1. **Operation Progress** - Real-time updates for indexing operations
2. **Directory Scanning** - Live file discovery during data addition
3. **Collection Status** - Real-time collection state changes
4. **Global Operations** - Broadcast updates for all active operations

### WebSocket Architecture

**useWebSocket.ts** - Generic WebSocket hook
```typescript
interface WebSocketOptions {
  onOpen?: (event: Event) => void;
  onMessage?: (event: MessageEvent) => void;
  onError?: (event: Event) => void;
  onClose?: (event: CloseEvent) => void;
  shouldReconnect?: boolean;
  reconnectInterval?: number;
  reconnectAttempts?: number;
}
```

**useOperationProgress.ts** - Operation-specific WebSocket
- Automatically connects when operation is active
- Handles message parsing and cache updates
- Integrates with React Query for state synchronization

**useOperationsSocket.ts** - Global operations WebSocket
- Single shared connection for all operation updates
- Prevents exceeding browser WebSocket limits
- Updates React Query cache for all active operations

**useDirectoryScanWebSocket.ts** - Directory scanning
- Real-time file discovery progress
- Filters and exclusion pattern support
- Progress visualization with file counts

### Message Protocol

```typescript
// Operation progress messages
interface OperationMessage {
  type: 'operation_started' | 'operation_completed' | 'operation_failed' |
        'progress_update' | 'current_state';
  data: {
    operation_id: string;
    status?: string;
    progress?: number;
    message?: string;
    error_message?: string;
    processed_files?: number;
    total_files?: number;
  };
}

// Directory scan messages
interface ScanMessage {
  type: 'scan_progress' | 'scan_complete' | 'scan_error';
  data: {
    files_found: number;
    directories_scanned: number;
    current_directory: string;
    files?: FileInfo[];
  };
}
```

### Error Handling & Resilience

- Automatic reconnection with exponential backoff
- Maximum retry attempts to prevent infinite loops
- Graceful fallback to polling for critical data
- Token refresh on authentication errors
- Connection state tracking for UI feedback

## Styling and Design System

### Tailwind CSS Configuration

- Default Tailwind setup with no custom theme extensions
- Utility classes used throughout components
- Custom CSS minimal, mainly for:
  - Animations (breathing collection cards, progress shimmer)
  - PDF.js integration
  - Email viewer styles

### Design Patterns

1. **Cards** - White background with shadow for content sections
2. **Forms** - Consistent input styling with focus states
3. **Buttons** - Primary (blue), secondary (gray), danger (red)
4. **Status Indicators** - Color-coded badges with optional animations
5. **Modals** - Full-screen overlay with centered content

### Responsive Design

- Mobile-first approach with Tailwind breakpoints
- Responsive grid layouts
- Scrollable content areas for long lists

## Error Boundaries and Loading States

### Error Boundary Implementation

The application uses React Error Boundaries at multiple levels:

```typescript
// Global error boundary in App.tsx
<ErrorBoundary>
  <QueryClientProvider client={queryClient}>
    <Router>
      {/* Application routes */}
    </Router>
  </QueryClientProvider>
</ErrorBoundary>

// Feature-level boundaries in HomePage.tsx
{activeTab === 'collections' && (
  <ErrorBoundary>
    <CollectionsDashboard />
  </ErrorBoundary>
)}
```

### Loading State Patterns

**Skeleton Loading**
```typescript
// CollectionList loading skeleton
if (isLoading) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {[...Array(6)].map((_, i) => (
        <div key={i} className="animate-pulse">
          <div className="bg-gray-200 h-48 rounded-lg" />
        </div>
      ))}
    </div>
  );
}
```

**Suspense Integration**
```typescript
// Lazy-loaded components with Suspense
const DocumentViewer = lazy(() => import('./components/DocumentViewer'));

<Suspense fallback={<DocumentViewerSkeleton />}>
  <DocumentViewer {...props} />
</Suspense>
```

### Error State Management

**API Error Handling**
```typescript
// Centralized error handler
export function handleApiError(error: unknown): string {
  if (axios.isAxiosError(error)) {
    return error.response?.data?.detail ||
           error.response?.data?.message ||
           error.message;
  }
  return 'An unexpected error occurred';
}
```

**User-Friendly Error Display**
```typescript
// Collection error state
if (error) {
  return (
    <div className="text-center py-12">
      <AlertCircle className="h-12 w-12 text-red-500 mx-auto mb-4" />
      <h3 className="text-lg font-medium text-gray-900 mb-2">
        Failed to load collections
      </h3>
      <p className="text-gray-600 mb-4">{handleApiError(error)}</p>
      <button onClick={refetch} className="btn-primary">
        Try Again
      </button>
    </div>
  );
}
```

## Performance Optimizations

### Code Splitting

- Route-based splitting via React Router
- Dynamic imports for heavy components (DocumentViewer)
- Lazy loading of modals and complex forms

### Data Fetching Optimizations

- React Query for intelligent caching and background refetching
- Optimistic updates for instant UI feedback
- Query deduplication to prevent duplicate requests
- Stale-while-revalidate pattern for fast initial loads
- Selective invalidation to minimize unnecessary fetches

### Rendering Optimizations

- React.memo for expensive components (CollectionCard, OperationProgress)
- useMemo for complex computations (search results processing)
- useCallback for stable function references in dependency arrays
- Virtual scrolling consideration for large document lists
- Proper key usage in lists to minimize reconciliation

## Form Handling Patterns

### Form State Management

The application uses controlled components with local state for forms:

**Collection Creation Form**
```typescript
function CreateCollectionModal() {
  const [formData, setFormData] = useState<CreateCollectionRequest>({
    name: '',
    description: '',
    embedding_model: 'Qwen/Qwen3-Embedding-0.6B',
    chunk_size: 1000,
    chunk_overlap: 200,
    quantization: 'float16',
    is_public: false,
  });

  const [errors, setErrors] = useState<Record<string, string>>({});

  const validateForm = (): boolean => {
    const newErrors: Record<string, string> = {};

    if (!formData.name.trim()) {
      newErrors.name = 'Collection name is required';
    }
    if (formData.name.length > 100) {
      newErrors.name = 'Collection name must be less than 100 characters';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };
}
```

### Field Validation Patterns

**Real-time Validation**
```typescript
const handleNameChange = (value: string) => {
  setFormData(prev => ({ ...prev, name: value }));

  // Clear error when user starts typing
  if (errors.name) {
    setErrors(prev => ({ ...prev, name: '' }));
  }

  // Validate on blur or after delay
  if (value && !isValidCollectionName(value)) {
    setErrors(prev => ({
      ...prev,
      name: 'Only letters, numbers, and hyphens allowed'
    }));
  }
};
```

### Form Submission Pattern

```typescript
const { mutate: createCollection, isLoading } = useCreateCollection();

const handleSubmit = async (e: React.FormEvent) => {
  e.preventDefault();

  if (!validateForm()) return;

  createCollection(formData, {
    onSuccess: (data) => {
      onClose(); // Close modal
      navigate(`/collections/${data.id}`); // Navigate to new collection
    },
    onError: (error) => {
      // API validation errors
      if (error.response?.data?.errors) {
        setErrors(error.response.data.errors);
      }
    },
  });
};
```

### Directory Scan Form Pattern

```typescript
// Complex form with WebSocket integration
function AddDataToCollectionModal({ collectionId }: Props) {
  const [scanOptions, setScanOptions] = useState({
    directory_path: '',
    include_patterns: ['*'],
    exclude_patterns: ['.git', '__pycache__', 'node_modules'],
  });

  const { startScan, scanResults, isScanning, error } = useDirectoryScanWebSocket();

  const handleScan = () => {
    startScan({
      path: scanOptions.directory_path,
      filters: {
        include: scanOptions.include_patterns,
        exclude: scanOptions.exclude_patterns,
      },
    });
  };
}
```

## Testing

Tests cover critical user flows and component behavior:

### Testing Stack

```json
{
  "devDependencies": {
    "@testing-library/react": "^16.1.0",
    "@testing-library/jest-dom": "^6.6.3",
    "@testing-library/user-event": "^14.5.2",
    "vitest": "^2.1.8",
    "jsdom": "^25.0.1",
    "msw": "^2.7.0",
    "@vitest/coverage-v8": "^2.1.8"
  }
}
```

### Test Coverage Areas

**Component Tests** (in `src/components/__tests__/`)
- `CollectionOperations.test.tsx` - Operation display and progress updates
- `CollectionOperations.websocket.test.tsx` - WebSocket integration
- `CollectionOperations.network.test.tsx` - Network error handling
- `CreateCollectionModal.test.tsx` - Form validation and submission
- `CreateCollectionModal.network.test.tsx` - API integration
- `CreateCollectionModal.models.test.tsx` - Model selection
- `AddDataToCollectionModal.test.tsx` - Directory scanning integration
- `CollectionsDashboard.test.tsx` - Dashboard rendering and filtering
- `CollectionsDashboard.network.test.tsx` - Network scenarios
- `Collections.permission.test.tsx` - Access control
- `ActiveOperationsTab.websocket.test.tsx` - Real-time updates
- `ActiveOperationsTab.navigation.test.tsx` - Navigation behavior
- `SearchInterface.test.tsx` - Search functionality
- `SearchInterface.reranking.test.tsx` - Reranking features

**Hook Tests** (in `src/hooks/__tests__/`)
- `useCollections.test.tsx` - React Query integration and optimistic updates
- `useCollectionOperations.test.tsx` - Operation management
- `useOperationProgress.test.tsx` - WebSocket message handling
- `useWebSocket.error.test.tsx` - WebSocket error scenarios
- `useCollectionDocuments.test.tsx` - Document fetching

**Store Tests** (in `src/stores/__tests__/`)
- `authStore.test.ts` - Authentication state management
- `searchStore.test.ts` - Search functionality
- `searchStore.reranking.test.ts` - Reranking state
- `uiStore.test.ts` - UI state and toast notifications
- `chunkingStore.test.ts` - Chunking configuration

**API Tests** (in `src/services/api/v2/__tests__/`)
- `chunking.test.ts` - Chunking API client

### Unit Tests

#### Component Testing Example
```typescript
// CollectionCard.test.tsx
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { CollectionCard } from './CollectionCard';
import { vi } from 'vitest';

describe('CollectionCard', () => {
  const mockCollection = {
    id: '123',
    name: 'Test Collection',
    status: 'ready',
    document_count: 100,
    vector_count: 500,
  };

  it('displays collection information correctly', () => {
    render(<CollectionCard collection={mockCollection} />);

    expect(screen.getByText('Test Collection')).toBeInTheDocument();
    expect(screen.getByText('100')).toBeInTheDocument();
  });

  it('shows processing animation for active collections', () => {
    const processingCollection = { ...mockCollection, status: 'processing' };
    const { container } = render(<CollectionCard collection={processingCollection} />);
    const card = container.querySelector('.animate-breathing');
    expect(card).toBeInTheDocument();
  });

  it('handles view details action', async () => {
    const onViewDetails = vi.fn();
    render(<CollectionCard collection={mockCollection} onViewDetails={onViewDetails} />);

    const detailsButton = screen.getByRole('button', { name: /details/i });
    await userEvent.click(detailsButton);

    expect(onViewDetails).toHaveBeenCalledWith('123');
  });
});
```

#### Store Testing Example
```typescript
// authStore.test.ts
import { renderHook, act } from '@testing-library/react';
import { useAuthStore } from './authStore';

describe('authStore', () => {
  beforeEach(() => {
    // Reset store before each test
    useAuthStore.setState({ token: null, user: null });
  });

  it('sets auth data correctly', () => {
    const { result } = renderHook(() => useAuthStore());

    act(() => {
      result.current.setAuth('token123', {
        id: 1,
        username: 'testuser',
        email: 'test@example.com',
      });
    });

    expect(result.current.token).toBe('token123');
    expect(result.current.user?.username).toBe('testuser');
  });

  it('clears auth data on logout', async () => {
    const { result } = renderHook(() => useAuthStore());

    act(() => {
      result.current.setAuth('token123', { id: 1, username: 'test' });
    });

    await act(async () => {
      await result.current.logout();
    });

    expect(result.current.token).toBeNull();
    expect(result.current.user).toBeNull();
  });
});
```

#### Hook Testing Example
```typescript
// useOperationProgress.test.ts
import { renderHook, waitFor } from '@testing-library/react';
import { useOperationProgress } from './useOperationProgress';
import WS from 'jest-websocket-mock';

describe('useOperationProgress', () => {
  let server: WS;

  beforeEach(() => {
    server = new WS('ws://localhost:8080/ws/operations/op123');
  });

  afterEach(() => {
    WS.clean();
  });

  it('connects to WebSocket and receives updates', async () => {
    const { result } = renderHook(() => useOperationProgress('op123'));

    await server.connected;

    server.send(JSON.stringify({
      type: 'progress_update',
      data: {
        progress: 50,
        message: 'Processing files...',
      },
    }));

    await waitFor(() => {
      expect(result.current.progress).toBe(50);
    });
  });
});
```

### Integration Tests

#### API Mocking with MSW
```typescript
// handlers.ts
import { http, HttpResponse } from 'msw';

export const handlers = [
  http.get('/api/v2/collections', () => {
    return HttpResponse.json({
      collections: [
        { id: '1', name: 'Collection 1', status: 'ready' },
        { id: '2', name: 'Collection 2', status: 'processing' },
      ],
    });
  }),

  http.post('/api/v2/collections', async ({ request }) => {
    const body = await request.json();
    return HttpResponse.json({
      id: '3',
      name: body.name,
      status: 'pending',
    });
  }),
];

// server.ts (in src/tests/mocks/)
import { setupServer } from 'msw/node';
import { handlers } from './handlers';

export const server = setupServer(...handlers);
```

#### User Flow Testing
```typescript
// CreateCollectionFlow.test.tsx
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { App } from './App';

describe('Create Collection User Flow', () => {
  it('completes collection creation flow', async () => {
    render(<App />);
    const user = userEvent.setup();

    // Navigate to collections
    await user.click(screen.getByText('Collections'));

    // Open create modal
    await user.click(screen.getByText('Create Collection'));

    // Fill form
    await user.type(screen.getByLabelText('Name'), 'My Documents');
    await user.type(screen.getByLabelText('Description'), 'Test collection');

    // Select model
    await user.selectOptions(screen.getByLabelText('Model'), 'Qwen/Qwen3-Embedding-0.6B');

    // Submit
    await user.click(screen.getByText('Create'));

    // Verify success
    await waitFor(() => {
      expect(screen.getByText('Collection created successfully')).toBeInTheDocument();
    });
  });
});
```

### E2E Tests with Playwright

```typescript
// e2e/search.spec.ts
import { test, expect } from '@playwright/test';

test.describe('Search Functionality', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('http://localhost:5173');
    await page.fill('input[name="username"]', 'testuser');
    await page.fill('input[name="password"]', 'testpass');
    await page.click('button:has-text("Sign in")');
  });

  test('performs semantic search', async ({ page }) => {
    // Navigate to search
    await page.click('text=Search');

    // Enter query
    await page.fill('input[placeholder*="search query"]', 'machine learning');

    // Select collection
    await page.selectOption('select[name="collection"]', 'technical-docs');

    // Enable hybrid search
    await page.check('input[name="hybrid"]');

    // Search
    await page.click('button:has-text("Search")');

    // Verify results
    await expect(page.locator('.search-result')).toHaveCount(10);
    await expect(page.locator('.search-result').first()).toContainText('Score:');
  });
});
```

### Testing Configuration

#### Vitest Configuration
```typescript
// vitest.config.ts
import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './vitest.setup.ts',
    css: true,
    testTimeout: 10000,
    hookTimeout: 10000,
    include: [
      'src/**/*.{test,spec}.{js,mjs,cjs,ts,mts,cts,jsx,tsx}',
      '**/*.{test,spec}.{js,mjs,cjs,ts,mts,cts,jsx,tsx}'
    ],
    exclude: ['**/node_modules/**', '**/dist/**', '**/e2e/**'],
    root: './',
    allowOnly: true,
    passWithNoTests: false,
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
});
```

#### Test Setup File
```typescript
// vitest.setup.ts
import '@testing-library/jest-dom';
import { vi, beforeAll, afterEach, afterAll } from 'vitest';
import { server } from './src/tests/mocks/server';

// MSW setup
beforeAll(() => server.listen({ onUnhandledRequest: 'error' }));
afterAll(() => server.close());
afterEach(() => {
  server.resetHandlers();
});

// Mock WebSocket
class MockWebSocket extends EventTarget {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  readyState = MockWebSocket.CONNECTING;
  url: string;

  constructor(url: string) {
    super();
    this.url = url;
  }

  send = vi.fn();
  close = vi.fn();
}

global.WebSocket = MockWebSocket as unknown as typeof WebSocket;
```

### Testing Best Practices

1. **Test Organization**
   ```
   src/
   ├── components/
   │   ├── CollectionCard.tsx
   │   └── __tests__/
   │       └── CollectionCard.test.tsx
   ├── hooks/
   │   ├── useOperationProgress.ts
   │   └── __tests__/
   │       └── useOperationProgress.test.tsx
   ├── stores/
   │   ├── authStore.ts
   │   └── __tests__/
   │       └── authStore.test.ts
   └── tests/
       ├── mocks/
       │   ├── handlers.ts
       │   └── server.ts
       └── utils.tsx
   ```

2. **Test Utilities**
   ```typescript
   // tests/utils.tsx
   export function renderWithProviders(
     ui: React.ReactElement,
     options?: RenderOptions
   ) {
     function Wrapper({ children }: { children: React.ReactNode }) {
       return (
         <QueryClientProvider client={queryClient}>
           <BrowserRouter>
             {children}
           </BrowserRouter>
         </QueryClientProvider>
       );
     }
     return render(ui, { wrapper: Wrapper, ...options });
   }
   ```

3. **Coverage Goals**
   - Unit tests: 80% coverage
   - Critical paths: 100% coverage
   - UI components: Focus on behavior, not implementation

4. **Performance Testing**
   ```typescript
   test('renders large collection list efficiently', async () => {
     const collections = Array.from({ length: 1000 }, (_, i) => ({
       id: `collection-${i}`,
       name: `Collection ${i}`,
       status: 'ready',
     }));

     const start = performance.now();
     render(<CollectionList collections={collections} />);
     const end = performance.now();

     expect(end - start).toBeLessThan(100); // Should render in < 100ms
   });
   ```

## Build and Deployment

### Development
```bash
npm run dev
```
- Starts Vite dev server on port 5173
- Proxies API calls to backend on port 8080
- Hot module replacement enabled

### Production Build
```bash
npm run build
```
- TypeScript compilation
- Vite production build
- Output to `packages/webui/static`
- Source maps enabled for debugging

### Deployment Flow

1. Frontend builds to backend static directory
2. Backend serves the built files
3. API calls go directly to backend (no proxy)
4. WebSocket connections established directly

## Component Development Guidelines

### Component Structure

#### File Organization
```typescript
// components/
// ├── CollectionCard.tsx        # Main component
// └── __tests__/
//     └── CollectionCard.test.tsx   # Tests

// CollectionCard.tsx
import React, { memo } from 'react';
import type { Collection } from '../types/collection';
import { useCollectionOperations } from '../hooks/useCollectionOperations';

interface CollectionCardProps {
  collection: Collection;
  onViewDetails?: (id: string) => void;
  onDelete?: (id: string) => void;
}

export const CollectionCard = memo<CollectionCardProps>(({ collection, onViewDetails, onDelete }) => {
  // Hooks at the top
  const { data: operations } = useCollectionOperations(collection.id);

  // Derived state
  const isActive = collection.status === 'processing';

  // Event handlers
  const handleViewDetails = () => {
    onViewDetails?.(collection.id);
  };

  // Render
  return (
    <article className="collection-card" aria-label={`Collection: ${collection.name}`}>
      {/* Component JSX */}
    </article>
  );
});

CollectionCard.displayName = 'CollectionCard';
```

### Component Patterns

#### Compound Components
```typescript
// DocumentViewer compound component
export const DocumentViewer = {
  Root: DocumentViewerRoot,
  Header: DocumentViewerHeader,
  Content: DocumentViewerContent,
  Navigation: DocumentViewerNavigation,
};

// Usage
<DocumentViewer.Root document={doc}>
  <DocumentViewer.Header />
  <DocumentViewer.Content />
  <DocumentViewer.Navigation />
</DocumentViewer.Root>
```

#### Render Props Pattern
```typescript
interface DataFetcherProps<T> {
  url: string;
  children: (data: T | null, loading: boolean, error: Error | null) => React.ReactNode;
}

function DataFetcher<T>({ url, children }: DataFetcherProps<T>) {
  const { data, isLoading, error } = useQuery<T>({ queryKey: [url], queryFn: () => fetch(url) });
  return <>{children(data ?? null, isLoading, error)}</>;
}

// Usage
<DataFetcher url="/api/v2/collections">
  {(collections, loading, error) => (
    loading ? <Spinner /> : <CollectionList collections={collections} />
  )}
</DataFetcher>
```

#### Custom Hooks Pattern
```typescript
// Encapsulate complex logic in custom hooks
function useCollectionManagement(collectionId: string) {
  const queryClient = useQueryClient();
  const { addToast } = useUIStore();

  const cancelMutation = useMutation({
    mutationFn: () => collectionsV2Api.delete(collectionId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['collections'] });
      addToast({ message: 'Collection deleted successfully', type: 'success' });
    },
  });

  return {
    deleteCollection: cancelMutation.mutate,
    isLoading: cancelMutation.isPending,
  };
}
```

### TypeScript Best Practices

#### Strict Type Definitions
```typescript
// Prefer interfaces for public APIs
export interface CollectionCardProps {
  collection: Collection;
  onViewDetails?: (collectionId: string) => void;
  onDelete?: (collectionId: string) => void;
  className?: string;
}

// Use type for unions and intersections
export type CollectionStatus = 'pending' | 'processing' | 'ready' | 'error';

// Const assertions for literals
export const COLLECTION_STATUSES = ['pending', 'processing', 'ready', 'error'] as const;
export type CollectionStatus = typeof COLLECTION_STATUSES[number];

// Generic components
interface ListProps<T> {
  items: T[];
  renderItem: (item: T, index: number) => React.ReactNode;
  keyExtractor: (item: T) => string;
}

function List<T>({ items, renderItem, keyExtractor }: ListProps<T>) {
  return (
    <ul>
      {items.map((item, index) => (
        <li key={keyExtractor(item)}>{renderItem(item, index)}</li>
      ))}
    </ul>
  );
}
```

### Performance Guidelines

#### Memoization Strategy
```typescript
// Memoize expensive computations
const expensiveValue = useMemo(() => {
  return computeExpensiveValue(data);
}, [data]);

// Memoize callbacks passed to children
const handleClick = useCallback((id: string) => {
  doSomething(id);
}, [doSomething]);

// Memoize components with complex props
export const CollectionCard = memo(CollectionCardComponent, (prevProps, nextProps) => {
  // Custom comparison
  return prevProps.collection.id === nextProps.collection.id &&
         prevProps.collection.status === nextProps.collection.status;
});
```

#### Virtual Scrolling
```typescript
// For large lists, use react-window
import { FixedSizeList } from 'react-window';

function VirtualCollectionList({ collections }: { collections: Collection[] }) {
  const Row = ({ index, style }) => (
    <div style={style}>
      <CollectionCard collection={collections[index]} />
    </div>
  );

  return (
    <FixedSizeList
      height={600}
      itemCount={collections.length}
      itemSize={120}
      width="100%"
    >
      {Row}
    </FixedSizeList>
  );
}
```

### Error Handling

#### Component Error Boundaries
```typescript
interface ErrorFallbackProps {
  error: Error;
  resetErrorBoundary: () => void;
}

function ErrorFallback({ error, resetErrorBoundary }: ErrorFallbackProps) {
  return (
    <div role="alert">
      <h2>Something went wrong:</h2>
      <pre>{error.message}</pre>
      <button onClick={resetErrorBoundary}>Try again</button>
    </div>
  );
}

// Usage with react-error-boundary
<ErrorBoundary FallbackComponent={ErrorFallback}>
  <CollectionList />
</ErrorBoundary>
```

## Accessibility

### WCAG 2.1 Compliance

#### Semantic HTML
```tsx
// Use semantic elements
<nav aria-label="Main navigation">
  <ul>
    <li><a href="/">Home</a></li>
    <li><a href="/collections">Collections</a></li>
  </ul>
</nav>

<main>
  <h1>Document Search</h1>
  <section aria-labelledby="search-heading">
    <h2 id="search-heading">Search Documents</h2>
    {/* Search form */}
  </section>
</main>
```

#### ARIA Attributes
```tsx
// Loading states
<div role="status" aria-live="polite" aria-busy={isLoading}>
  {isLoading ? <Spinner /> : <SearchResults />}
</div>

// Progress indicators
<div
  role="progressbar"
  aria-valuenow={progress}
  aria-valuemin={0}
  aria-valuemax={100}
  aria-label="Operation progress"
>
  <div style={{ width: `${progress}%` }} />
</div>

// Modal dialogs
<div
  role="dialog"
  aria-modal="true"
  aria-labelledby="dialog-title"
  aria-describedby="dialog-description"
>
  <h2 id="dialog-title">Confirm Delete</h2>
  <p id="dialog-description">Are you sure you want to delete this collection?</p>
</div>
```

#### Keyboard Navigation
```tsx
// Ensure all interactive elements are keyboard accessible
function SearchResults({ results }: { results: SearchResult[] }) {
  const [selectedIndex, setSelectedIndex] = useState(0);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        setSelectedIndex((prev) => Math.min(prev + 1, results.length - 1));
        break;
      case 'ArrowUp':
        e.preventDefault();
        setSelectedIndex((prev) => Math.max(prev - 1, 0));
        break;
      case 'Enter':
        e.preventDefault();
        openDocument(results[selectedIndex]);
        break;
    }
  };

  return (
    <ul role="listbox" onKeyDown={handleKeyDown}>
      {results.map((result, index) => (
        <li
          key={result.id}
          role="option"
          aria-selected={index === selectedIndex}
          tabIndex={index === selectedIndex ? 0 : -1}
        >
          {result.title}
        </li>
      ))}
    </ul>
  );
}
```

#### Focus Management
```tsx
// Trap focus in modals
function Modal({ isOpen, onClose, children }: ModalProps) {
  const modalRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (isOpen && modalRef.current) {
      const focusableElements = modalRef.current.querySelectorAll(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
      );
      const firstElement = focusableElements[0] as HTMLElement;
      const lastElement = focusableElements[focusableElements.length - 1] as HTMLElement;

      firstElement?.focus();

      const handleTab = (e: KeyboardEvent) => {
        if (e.key === 'Tab') {
          if (e.shiftKey && document.activeElement === firstElement) {
            e.preventDefault();
            lastElement?.focus();
          } else if (!e.shiftKey && document.activeElement === lastElement) {
            e.preventDefault();
            firstElement?.focus();
          }
        }
      };

      document.addEventListener('keydown', handleTab);
      return () => document.removeEventListener('keydown', handleTab);
    }
  }, [isOpen]);

  return isOpen ? (
    <div ref={modalRef} role="dialog" aria-modal="true">
      {children}
    </div>
  ) : null;
}
```

### Screen Reader Support

#### Live Regions
```tsx
// Announce dynamic updates
function OperationProgress({ operation }: { operation: Operation }) {
  return (
    <>
      <div aria-live="polite" aria-atomic="true" className="sr-only">
        Operation {operation.type} is {operation.progress}% complete
      </div>
      <ProgressBar value={operation.progress} />
    </>
  );
}
```

#### Form Labels and Errors
```tsx
function SearchForm() {
  const [errors, setErrors] = useState<Record<string, string>>({});

  return (
    <form aria-label="Document search">
      <div>
        <label htmlFor="query">Search query</label>
        <input
          id="query"
          type="text"
          aria-describedby="query-error query-hint"
          aria-invalid={!!errors.query}
        />
        <span id="query-hint" className="hint">
          Enter keywords or phrases
        </span>
        {errors.query && (
          <span id="query-error" role="alert" className="error">
            {errors.query}
          </span>
        )}
      </div>
    </form>
  );
}
```

### Color and Contrast

#### High Contrast Support
```css
/* Support Windows High Contrast Mode */
@media (prefers-contrast: high) {
  .button {
    border: 2px solid;
  }

  .status-indicator {
    outline: 2px solid;
  }
}

/* Ensure sufficient color contrast */
:root {
  --text-primary: #1a1a1a;      /* 18:1 contrast ratio */
  --text-secondary: #666666;    /* 7:1 contrast ratio */
  --bg-primary: #ffffff;
  --bg-secondary: #f5f5f5;
}
```

### Testing Accessibility

#### Automated Testing
```typescript
// Using jest-axe
import { axe, toHaveNoViolations } from 'jest-axe';

expect.extend(toHaveNoViolations);

test('CollectionCard is accessible', async () => {
  const { container } = render(<CollectionCard collection={mockCollection} />);
  const results = await axe(container);
  expect(results).toHaveNoViolations();
});
```

#### Manual Testing Checklist
1. Navigate using only keyboard (Tab, Shift+Tab, Arrow keys)
2. Test with screen readers (NVDA, JAWS, VoiceOver)
3. Verify color contrast ratios (4.5:1 for normal text, 3:1 for large text)
4. Test with browser zoom at 200%
5. Disable CSS and verify content structure
6. Test with reduced motion preferences

## Future Work

**Scalability**: Virtual scrolling, server-side pagination, request batching
**Features**: Collection sharing, advanced search, bulk ops, i18n
**Tech**: React 19 features, performance monitoring, offline support
**DX**: Storybook, visual regression tests, API type generation
**A11y**: WCAG 2.1 AA audit, keyboard nav, screen reader optimization

## Architecture Principles

**State**: React Query for server data, Zustand for UI-only state. Never duplicate.
**Components**: Composition over inheritance. Use hooks for shared logic.
**Performance**: Memoize expensive ops. React.memo for pure components.
**Types**: No `any` types. Strict null checks. Proper error types.
