# Semantik Frontend Architecture Specification

## Overview
The Semantik webui-react application is a modern React 19 single-page application built with TypeScript, providing a secure, performant interface for managing document collections and semantic search operations. The architecture follows Domain-Driven Design principles with a clear separation of concerns between UI components, state management, and API integration layers.

## Technology Stack

### Core Technologies
- **React 19.1.0**: Latest React with concurrent features and improved performance
- **TypeScript 5.8.3**: Type-safe development with strict mode enabled
- **Vite 7.0.0**: Fast build tool with HMR and optimized production builds
- **React Router DOM 7.6.3**: Client-side routing with nested layouts

### State Management
- **Zustand 5.0.6**: Lightweight state management with TypeScript support
- **React Query (TanStack Query) 5.81.5**: Server state management with caching

### Styling & UI
- **TailwindCSS 3.4.17**: Utility-first CSS framework
- **Lucide React 0.525.0**: Icon library
- **PostCSS/Autoprefixer**: CSS processing

### Testing Infrastructure
- **Vitest 2.1.9**: Unit and integration testing
- **Testing Library**: React component testing
- **MSW 2.10.4**: API mocking for tests
- **Playwright**: E2E testing

## 1. Component Architecture

### Component Hierarchy
```
App.tsx
├── ErrorBoundary
├── QueryClientProvider
└── Router
    ├── LoginPage (public)
    ├── VerificationPage (public)
    └── Layout (protected)
        ├── Header
        │   ├── User Info
        │   ├── Navigation
        │   └── Logout
        ├── Tab Navigation
        │   ├── Collections Tab
        │   ├── Operations Tab
        │   └── Search Tab
        ├── Main Content (Outlet)
        │   ├── HomePage
        │   │   ├── CollectionsDashboard
        │   │   ├── ActiveOperationsTab
        │   │   └── SearchInterface
        │   └── SettingsPage
        └── Global Components
            ├── Toast Container
            ├── DocumentViewerModal
            ├── CollectionDetailsModal
            ├── CreateCollectionModal
            │   ├── ConnectorTypeSelector
            │   └── ConnectorForm
            └── AddDataToCollectionModal
                ├── ConnectorTypeSelector
                └── ConnectorForm
```

### Component Design Patterns

#### Composition Pattern
Components are built using composition for flexibility:
```typescript
// Parent provides structure
<CollectionsDashboard>
  <CollectionCard />
  <CreateCollectionModal />
</CollectionsDashboard>
```

#### Container/Presentational Pattern
- **Container Components**: Handle data fetching and business logic (e.g., `CollectionsDashboard`)
- **Presentational Components**: Focus on UI rendering (e.g., `CollectionCard`)

#### Modal Management
Modals are controlled through UIStore for global accessibility:
```typescript
// Global modal state in UIStore
showDocumentViewer: { collectionId, docId, chunkId } | null
showCollectionDetailsModal: string | null
```

### Custom Hooks Architecture

#### Data Fetching Hooks
- `useCollections()`: Fetch all collections with auto-refresh
- `useCollection(id)`: Fetch single collection details
- `useCollectionOperations(id)`: Fetch operations for a collection
- `useCollectionDocuments(id)`: Fetch documents in a collection

#### WebSocket Hooks
- `useWebSocket()`: Base WebSocket connection management
- `useOperationProgress()`: Real-time operation progress tracking
- `useDirectoryScanWebSocket()`: Directory scanning progress
- `useChunkingWebSocket()`: Chunking operation updates

#### Feature Hooks
- `useRerankingAvailability()`: Check reranking model availability
- `useDirectoryScan()`: Directory scanning functionality

#### Connector Hooks
- `useConnectorCatalog()`: Fetch available connector types from backend
- `useGitPreview()`: Test Git repository connections before adding
- `useImapPreview()`: Test IMAP server connections before adding

### Connector System

#### Overview
The connector system provides a dynamic UI for adding data sources (Directory, Git, IMAP) to collections. Connector definitions are fetched from the backend API (`/api/v2/connectors`), allowing new connector types to be added with minimal frontend changes.

#### Components
```
components/connectors/
├── index.ts                  # Barrel exports
├── ConnectorTypeSelector.tsx # Card-based connector picker
└── ConnectorForm.tsx         # Dynamic form field rendering
```

#### ConnectorTypeSelector
Card-based picker that displays available connector types with:
- Icons for each connector type
- Short descriptions
- Selection state highlighting
- Optional "None" option for skipping source addition

#### ConnectorForm
Dynamically renders form fields based on connector definition:
- Text inputs, numbers, checkboxes, selects
- Conditional field visibility (`show_when` conditions)
- Secret fields (passwords, tokens)
- Preview/test connection functionality
- Validation error display

#### Adding New Connectors
When a new connector is added to the backend, the frontend needs:
1. **Icon mapping**: Add to `connectorIcons` in `ConnectorTypeSelector.tsx`
2. **Display order**: Add to `displayOrder` array
3. **Short description**: Add case to `getShortDescription()`
4. **Preview handler**: Add logic to `handlePreview()` if connector supports preview
5. **Source path**: Add case to `getSourcePath()` for display purposes

## 2. State Management with Zustand

### Store Architecture

#### AuthStore (Persisted)
```typescript
interface AuthState {
  token: string | null
  refreshToken: string | null
  user: User | null
  setAuth: (token, user, refreshToken?) => void
  logout: () => Promise<void>
}
```
- Persisted to localStorage as 'auth-storage'
- Handles authentication state and logout flow
- Token automatically injected into API requests

#### CollectionStore (UI State Only)
```typescript
interface CollectionUIStore {
  selectedCollectionId: string | null
  setSelectedCollection: (id) => void
  clearStore: () => void
}
```
- Manages UI-only state for collections
- Server state handled by React Query

#### SearchStore (Complex State)
```typescript
interface SearchState {
  results: SearchResult[]
  searchParams: SearchParams
  validationErrors: ValidationError[]
  rerankingMetrics: RerankingMetrics | null
  // ... validation and update methods
}
```
- Manages search parameters and results
- Built-in validation with error handling
- Supports hybrid and semantic search modes

#### ChunkingStore (Feature-Rich)
```typescript
interface ChunkingStore {
  selectedStrategy: ChunkingStrategyType
  previewChunks: ChunkPreview[]
  comparisonResults: ComparisonResults
  analyticsData: ChunkingAnalytics | null
  // ... comprehensive chunking methods
}
```
- Complex state for chunking strategies
- Preview, comparison, and analytics features
- Preset management with custom configurations

#### UIStore (Global UI State)
```typescript
interface UIState {
  toasts: Toast[]
  activeTab: 'search' | 'collections' | 'operations'
  showDocumentViewer: ViewerState | null
  showCollectionDetailsModal: string | null
}
```
- Toast notification system with auto-dismiss
- Tab navigation state
- Modal visibility control

### State Update Patterns

#### Optimistic Updates
```typescript
// In useCreateCollection hook
onMutate: async (newCollection) => {
  // Create temporary optimistic collection
  const tempCollection = { id: `temp-${Date.now()}`, ...newCollection }
  // Update cache optimistically
  queryClient.setQueryData(key, old => [...old, tempCollection])
  // Return context for rollback
  return { previousCollections, tempId }
}
```

#### Error Recovery
```typescript
onError: (error, variables, context) => {
  // Rollback to previous state
  queryClient.setQueryData(key, context.previousCollections)
  // Show error toast
  addToast({ type: 'error', message: handleApiError(error) })
}
```

## 3. API Integration Layer

### API Client Architecture

#### Axios Configuration
```typescript
// Base client with interceptors
const apiClient = axios.create({
  baseURL: '',
  headers: { 'Content-Type': 'application/json' }
})

// Auth token injection
apiClient.interceptors.request.use(config => {
  const token = useAuthStore.getState().token
  if (token) config.headers.Authorization = `Bearer ${token}`
})

// 401 handling with auto-logout
apiClient.interceptors.response.use(
  response => response,
  error => {
    if (error.response?.status === 401) {
      useAuthStore.getState().logout()
      window.location.href = '/login'
    }
  }
)
```

### API Module Structure
```
services/api/v2/
├── client.ts       # Axios instance
├── types.ts        # TypeScript interfaces
├── auth.ts         # Authentication endpoints
├── collections.ts  # Collection CRUD operations
├── operations.ts   # Operation management
├── documents.ts    # Document operations
├── search.ts       # Search functionality
├── chunking.ts     # Chunking strategies
├── connectors.ts   # Connector catalog and preview APIs
├── settings.ts     # User settings
├── system.ts       # System information
└── directoryScan.ts # Directory scanning
```

### React Query Integration

#### Query Key Factory Pattern
```typescript
export const collectionKeys = {
  all: ['collections'] as const,
  lists: () => [...collectionKeys.all, 'list'],
  detail: (id) => [...collectionKeys.details(), id],
}
```

#### Automatic Refetching
```typescript
useQuery({
  queryKey: collectionKeys.lists(),
  queryFn: collectionsV2Api.list,
  refetchInterval: (query) => {
    // Refetch every 30s if operations active
    const hasActive = query.state.data?.some(c => c.status === 'processing')
    return hasActive ? 30000 : false
  }
})
```

### WebSocket Integration

#### Connection Management
```typescript
function useWebSocket(url: string | null, options: UseWebSocketOptions) {
  // Auto-reconnect logic with exponential backoff
  // Connection timeout handling
  // Message parsing and error recovery
  return { sendMessage, readyState, isConnected }
}
```

#### Real-time Updates Pattern
```typescript
// Operation progress tracking
useOperationProgress(operationId, {
  onComplete: () => invalidateQueries(),
  onError: (error) => showErrorToast(error),
  showToasts: true
})
```

## 4. Routing and Navigation

### Route Structure
```typescript
<Routes>
  <Route path="/login" element={<LoginPage />} />
  <Route path="/verification" element={<VerificationPage />} />
  <Route path="/" element={<ProtectedRoute><Layout /></ProtectedRoute>}>
    <Route index element={<HomePage />} />
    <Route path="settings" element={<SettingsPage />} />
  </Route>
</Routes>
```

### Authentication Flow
1. **Protected Routes**: Check for valid token in AuthStore
2. **Auto-redirect**: Navigate to /login if no token
3. **Token Refresh**: Handle refresh token rotation (if implemented)
4. **Logout Flow**: Clear store, call API, redirect to login

### Navigation Guards
```typescript
function ProtectedRoute({ children }) {
  const token = useAuthStore(state => state.token)
  if (!token) return <Navigate to="/login" replace />
  return children
}
```

## 5. UI/UX Patterns

### Design System

#### Form Styling Utilities
```typescript
// Consistent form input styling
export const getInputClassName = (
  hasError: boolean,
  isDisabled: boolean,
  additionalClasses = ''
): string => {
  // Returns Tailwind classes based on state
}
```

#### Component Styling Pattern
- Utility-first with TailwindCSS
- Conditional styling with template literals
- Responsive design with Tailwind breakpoints
- Dark mode support (if implemented)

### Toast Notification System
```typescript
interface Toast {
  id: string
  message: string
  type: 'success' | 'error' | 'info' | 'warning'
  duration?: number // Auto-dismiss timing
}
```
- Stacked notifications in bottom-right
- Auto-dismiss with configurable duration
- Color-coded by type with border accent

### Modal Patterns
- Global modal state in UIStore
- Lazy-loaded modal components
- Backdrop click to close
- ESC key handling
- Focus trap implementation

### Loading States
- Skeleton loaders for collections
- Spinner indicators for operations
- Progress bars for long-running tasks
- Optimistic UI updates for better UX

### Error Handling

#### Error Boundary
```typescript
class ErrorBoundary extends Component {
  // Catches React rendering errors
  // Shows fallback UI with error details
  // Provides reload button for recovery
}
```

#### API Error Handling
```typescript
export const handleApiError = (error: unknown): string => {
  // Standardized error message extraction
  // User-friendly error messages
  // Fallback for unknown errors
}
```

## 6. Build and Development Setup

### Vite Configuration
```typescript
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: '../../packages/webui/static',
    sourcemap: true,
    emptyOutDir: true
  },
  server: {
    proxy: {
      '/api': 'http://localhost:8080',
      '/ws': 'ws://localhost:8080'
    }
  }
})
```

### TypeScript Configuration
- **Strict Mode**: Full type safety enabled
- **Path Aliases**: `@/` maps to `./src/`
- **Module Resolution**: Bundler mode for Vite
- **JSX**: React 18+ transform

### Development Workflow
1. **Hot Module Replacement**: Instant updates in dev
2. **Proxy Configuration**: API requests proxied to backend
3. **Source Maps**: Full debugging support
4. **Type Checking**: Compile-time type safety

### Production Optimization
- **Code Splitting**: Dynamic imports for routes
- **Tree Shaking**: Remove unused code
- **Asset Optimization**: Image/font compression
- **Minification**: JavaScript and CSS minification
- **Source Maps**: Available for debugging

## 7. Testing Strategy

### Unit Testing (Vitest)
- Component testing with Testing Library
- Store testing for Zustand stores
- Hook testing with renderHook
- Utility function testing

### Integration Testing
- API integration with MSW mocks
- WebSocket testing with mock servers
- User flow testing

### E2E Testing (Playwright)
- Critical user journeys
- Cross-browser testing
- Visual regression testing

### Test Utilities
```typescript
// Test wrapper with providers
export function TestWrapper({ children }) {
  return (
    <QueryClientProvider client={testQueryClient}>
      <MemoryRouter>
        {children}
      </MemoryRouter>
    </QueryClientProvider>
  )
}
```

## 8. Security Considerations

### Authentication
- JWT tokens stored in Zustand with persistence
- Automatic token injection in API requests
- 401 response handling with auto-logout
- Secure token refresh flow

### Input Validation
- Client-side validation before API calls
- Sanitization of user inputs
- XSS prevention through React's built-in escaping
- CSRF protection (if implemented)

### Data Security
- No sensitive data in localStorage
- Secure WebSocket connections
- API error message sanitization
- Permission-based UI rendering

## 9. Performance Optimizations

### React Query Caching
- Stale-while-revalidate strategy
- Intelligent cache invalidation
- Optimistic updates for instant feedback
- Background refetching for fresh data

### Code Splitting
- Route-based splitting with React.lazy
- Component-level splitting for modals
- Vendor chunk optimization

### Rendering Optimizations
- React.memo for expensive components
- useMemo/useCallback for referential stability
- Virtual scrolling for large lists (if needed)
- Debounced search inputs

### Bundle Size Management
- Tree shaking with Vite
- Dynamic imports for heavy components
- Lazy loading of non-critical features
- Image optimization with responsive loading

## 10. Migration and Refactoring Notes

### Collection-Centric Architecture
- Migrated from job-centric to collection-centric model
- Operations tied to collections, not standalone jobs
- Consistent terminology throughout codebase

### Backward Compatibility
- Legacy exports maintained during migration
- Gradual deprecation of old patterns
- Clear migration path for components

### Future Improvements
- Implement comprehensive error recovery
- Add offline support with service workers
- Enhance accessibility with ARIA labels
- Implement internationalization (i18n)
- Add comprehensive analytics tracking
- Implement real-time collaboration features

## 11. Development Guidelines

### Code Organization
- Feature-based folder structure
- Colocation of related files
- Clear separation of concerns
- Consistent naming conventions

### Best Practices
- Prefer composition over inheritance
- Use TypeScript strictly (no `any`)
- Follow React hooks rules
- Implement proper error boundaries
- Write testable, modular code
- Document complex business logic

### Performance Guidelines
- Lazy load heavy components
- Optimize re-renders with memo
- Use proper React Query cache keys
- Debounce expensive operations
- Profile and optimize critical paths

This architecture provides a solid foundation for building scalable, maintainable, and performant React applications while ensuring type safety, good developer experience, and excellent user experience.