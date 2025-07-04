# Frontend Architecture Documentation

## Overview

The frontend for Semantik is a modern React application built with TypeScript, located in `apps/webui-react/`. It provides a sophisticated user interface for managing document embedding jobs, searching through embedded documents, and monitoring system performance.

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
export default defineConfig({
  plugins: [react()],
  base: '/',
  build: {
    outDir: '../../packages/webui/static',  // Builds to backend static directory
    assetsDir: 'assets',
    sourcemap: false,
    minify: 'esbuild',
    emptyOutDir: true,
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
})
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
├── hooks/              # Custom React hooks
├── pages/              # Route-level components
├── services/           # API client and external services
├── stores/             # Zustand state stores
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
        │   ├── CreateJobForm
        │   ├── JobList
        │   │   └── JobCard
        │   └── SearchInterface
        │       └── SearchResults
        ├── SettingsPage
        ├── Toast
        ├── DocumentViewerModal
        └── JobMetricsModal
```

### Key Components

#### Layout Components

**Layout.tsx**
- Main application shell with header navigation
- Tab-based navigation between Create Job, Jobs, and Search
- Integrates global modals (Toast, DocumentViewer, JobMetrics)
- Handles authentication state and logout

**ErrorBoundary.tsx**
- Catches React errors and displays fallback UI
- Prevents entire app crashes
- Logs errors to console

#### Job Management Components

**CreateJobForm.tsx**
- Complex form for creating embedding jobs
- Features:
  - Directory scanning with WebSocket progress
  - Model selection from available embedding models
  - Advanced parameter configuration (chunk size, overlap, etc.)
  - Real-time scan progress visualization
- Uses `useDirectoryScanWebSocket` hook for async scanning

**JobList.tsx**
- Displays all jobs with automatic refresh (5-second interval)
- Groups jobs by status (active, completed, failed)
- Integrates with React Query for data fetching

**JobCard.tsx**
- Individual job display with status indicators
- Real-time progress updates via WebSocket
- Actions: view metrics, cancel, delete
- Animated "breathing" effect for running jobs

#### Search Components

**SearchInterface.tsx**
- Comprehensive search form with:
  - Query input with search tips
  - Collection selection (only shows ready collections)
  - Hybrid search toggle with mode options
  - Result count configuration
- Auto-refreshes collection statuses for processing jobs

**SearchResults.tsx**
- Displays search results with score indicators
- Chunk navigation (previous/next)
- Click to view full document functionality
- Empty state handling

#### Modal Components

**DocumentViewerModal.tsx**
- Full document viewer with chunk highlighting
- Supports multiple file types (PDF, text, email)
- Navigation between chunks
- Integrated text search

**JobMetricsModal.tsx**
- Detailed job performance metrics
- Processing statistics
- Error logs display
- Real-time updates for running jobs

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

### Zustand Stores

The application uses Zustand for global state management with four main stores:

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

#### jobsStore.ts
```typescript
interface JobsState {
  jobs: Job[];
  activeJobs: Set<string>;
  setJobs: (jobs: Job[]) => void;
  updateJob: (jobId: string, updates: Partial<Job>) => void;
  addJob: (job: Job) => void;
  removeJob: (jobId: string) => void;
  setActiveJob: (jobId: string, active: boolean) => void;
}
```
- Manages job list and individual job updates
- Tracks active jobs for WebSocket connections
- No persistence (fetched from server)

#### searchStore.ts
```typescript
interface SearchState {
  results: SearchResult[];
  loading: boolean;
  error: string | null;
  searchParams: SearchParams;
  collections: string[];
  // ... methods
}
```
- Manages search state and results
- Stores search parameters
- Handles collection list

#### uiStore.ts
```typescript
interface UIState {
  toasts: Toast[];
  activeTab: 'create' | 'jobs' | 'search';
  showJobMetricsModal: string | null;
  showDocumentViewer: { jobId: string; docId: string; chunkId?: string } | null;
  // ... methods
}
```
- UI-specific state (modals, toasts, tabs)
- Auto-dismiss logic for toasts
- Modal visibility management

### State Flow Diagram

```
User Action → Component → Store Action → State Update → Re-render
     ↓                         ↓
     └→ API Call → Response → Store Update
```

## API Integration

### API Client (`services/api.ts`)

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

### API Endpoints

**jobsApi**
- `list()` - Get all jobs
- `create(data)` - Create new job
- `delete(jobId)` - Delete job
- `cancel(jobId)` - Cancel running job
- `getMetrics()` - Get system metrics
- `getCollectionsStatus()` - Get vector collection statuses

**searchApi**
- `search(params)` - Perform semantic search

**documentsApi**
- `getDocument(docId)` - Get full document
- `getChunk(docId, chunkIndex)` - Get specific chunk

**authApi**
- `login(credentials)` - User login
- `register(credentials)` - User registration
- `me()` - Get current user
- `logout()` - Logout user

**modelsApi**
- `list()` - Get available embedding models

**settingsApi**
- `getStats()` - Get system statistics
- `resetDatabase()` - Reset database (admin)

## Custom Hooks

### WebSocket Hooks

**useWebSocket.ts**
- Generic WebSocket connection management
- Auto-reconnection with exponential backoff
- Connection state tracking
- Message sending utilities

**useJobProgress.ts**
- Job-specific WebSocket for progress updates
- Automatically connects/disconnects based on job status
- Updates job store with progress data

**useDirectoryScanWebSocket.ts**
- Directory scanning via WebSocket
- Real-time scan progress updates
- Error handling and state management

### Other Hooks

**useDirectoryScan.ts**
- HTTP-based directory scanning (fallback)
- Returns scan results and loading state

## Routing

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
1. **Job Progress** - Real-time job status updates
2. **Directory Scanning** - Live scan progress

### Message Flow

```
Client                    Server
  |------ Connect -------->|
  |<----- Progress --------|
  |<----- Progress --------|
  |<----- Complete --------|
  |------ Close ---------->|
```

### Error Handling

- Automatic reconnection on disconnect
- Timeout handling for connection attempts
- Graceful degradation to polling if WebSocket fails

## Styling and Design System

### Tailwind CSS Configuration

- Default Tailwind setup with no custom theme extensions
- Utility classes used throughout components
- Custom CSS minimal, mainly for:
  - Animations (breathing job cards, progress shimmer)
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

## Performance Optimizations

### Code Splitting

- Route-based splitting via React Router
- Dynamic imports for heavy components (PDF viewer)

### Data Fetching

- React Query for caching and background refetching
- Optimistic updates for better UX
- Pagination ready (though not implemented)

### Rendering

- React.memo for expensive components
- Proper key usage in lists
- Minimal re-renders through Zustand selectors

## Testing Strategy

Currently, no test files are present in the codebase. Recommended testing approach:

### Unit Tests
- Component testing with React Testing Library
- Store testing for Zustand actions
- Hook testing with renderHook

### Integration Tests
- API mocking with MSW
- User flow testing
- WebSocket testing

### E2E Tests
- Critical user journeys
- Cross-browser testing
- Performance testing

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
- Minified with esbuild

### Deployment Flow

1. Frontend builds to backend static directory
2. Backend serves the built files
3. API calls go directly to backend (no proxy)
4. WebSocket connections established directly

## Future Considerations

### Scalability
- Consider server-side rendering for initial load
- Implement virtual scrolling for large lists
- Add pagination to search results

### Features
- Real-time collaboration features
- Advanced search filters
- Batch operations on jobs
- User preferences persistence

### Technical Debt
- Add comprehensive test coverage
- Implement error boundaries per feature
- Add performance monitoring
- Consider internationalization

## Conclusion

The Semantik frontend is a well-structured React application that effectively manages complex state, real-time updates, and user interactions. The architecture supports future growth while maintaining clean separation of concerns and type safety throughout.