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

## State Management Patterns

### Store Design Principles

1. **Single Responsibility**: Each store manages one domain
2. **Derived State**: Compute values in selectors, not in store
3. **Immutability**: Always create new objects/arrays
4. **Persistence**: Only persist necessary data (auth, preferences)

### Advanced Zustand Patterns

#### Selector Optimization
```typescript
// Avoid re-renders with specific selectors
const jobCount = useJobsStore((state) => state.jobs.length);
const activeJobIds = useJobsStore((state) => 
  state.jobs.filter(j => j.status === 'running').map(j => j.id)
);

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
  const jobs = useJobsStore();
  const ui = useUIStore();
  
  return {
    isAuthenticated: !!auth.token,
    hasActiveJobs: jobs.activeJobs.size > 0,
    currentView: ui.activeTab,
  };
};
```

#### Async Actions Pattern
```typescript
interface JobsState {
  jobs: Job[];
  loading: boolean;
  error: string | null;
  
  fetchJobs: () => Promise<void>;
  createJob: (data: JobData) => Promise<Job>;
}

const useJobsStore = create<JobsState>((set, get) => ({
  jobs: [],
  loading: false,
  error: null,
  
  fetchJobs: async () => {
    set({ loading: true, error: null });
    try {
      const response = await jobsApi.list();
      set({ jobs: response.data.jobs, loading: false });
    } catch (error) {
      set({ error: error.message, loading: false });
    }
  },
  
  createJob: async (data) => {
    set({ loading: true, error: null });
    try {
      const response = await jobsApi.create(data);
      const newJob = response.data.job;
      set((state) => ({
        jobs: [...state.jobs, newJob],
        loading: false,
      }));
      return newJob;
    } catch (error) {
      set({ error: error.message, loading: false });
      throw error;
    }
  },
}));
```

#### Middleware Usage
```typescript
// DevTools integration
import { devtools } from 'zustand/middleware';

const useJobsStore = create<JobsState>()(
  devtools(
    (set) => ({
      // ... store implementation
    }),
    { name: 'jobs-store' }
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

### React Query Integration

#### Query Key Patterns
```typescript
// Consistent query key structure
export const queryKeys = {
  jobs: {
    all: ['jobs'] as const,
    lists: () => [...queryKeys.jobs.all, 'list'] as const,
    list: (filters: string) => [...queryKeys.jobs.lists(), { filters }] as const,
    details: () => [...queryKeys.jobs.all, 'detail'] as const,
    detail: (id: string) => [...queryKeys.jobs.details(), id] as const,
  },
  search: {
    all: ['search'] as const,
    results: (params: SearchParams) => [...queryKeys.search.all, params] as const,
  },
};
```

#### Optimistic Updates
```typescript
const createJobMutation = useMutation({
  mutationFn: jobsApi.create,
  onMutate: async (newJob) => {
    // Cancel outgoing refetches
    await queryClient.cancelQueries({ queryKey: queryKeys.jobs.all });
    
    // Snapshot previous value
    const previousJobs = queryClient.getQueryData(queryKeys.jobs.lists());
    
    // Optimistically update
    queryClient.setQueryData(queryKeys.jobs.lists(), (old) => {
      return [...(old || []), { ...newJob, id: 'temp-id', status: 'created' }];
    });
    
    return { previousJobs };
  },
  onError: (err, newJob, context) => {
    // Rollback on error
    queryClient.setQueryData(queryKeys.jobs.lists(), context.previousJobs);
  },
  onSettled: () => {
    // Always refetch after error or success
    queryClient.invalidateQueries({ queryKey: queryKeys.jobs.lists() });
  },
});
```

### State Synchronization

#### WebSocket State Updates
```typescript
// Sync WebSocket updates with React Query
useEffect(() => {
  const ws = new WebSocket(`ws://localhost:8080/ws/${jobId}`);
  
  ws.onmessage = (event) => {
    const update = JSON.parse(event.data);
    
    // Update React Query cache
    queryClient.setQueryData(
      queryKeys.jobs.detail(jobId),
      (oldData) => ({
        ...oldData,
        ...update,
      })
    );
    
    // Update Zustand store
    useJobsStore.getState().updateJob(jobId, update);
  };
  
  return () => ws.close();
}, [jobId]);
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

While the codebase currently lacks test files, here's a comprehensive testing strategy for implementing tests:

### Testing Stack Recommendations

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

### Unit Tests

#### Component Testing Example
```typescript
// JobCard.test.tsx
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { JobCard } from './JobCard';
import { vi } from 'vitest';

describe('JobCard', () => {
  const mockJob = {
    id: '123',
    name: 'Test Job',
    status: 'running',
    progress: 50,
    total_files: 100,
    processed_files: 50,
  };

  it('displays job progress correctly', () => {
    render(<JobCard job={mockJob} />);
    
    expect(screen.getByText('Test Job')).toBeInTheDocument();
    expect(screen.getByText('50%')).toBeInTheDocument();
    expect(screen.getByText('50 / 100 files')).toBeInTheDocument();
  });

  it('shows breathing animation for running jobs', () => {
    const { container } = render(<JobCard job={mockJob} />);
    const card = container.querySelector('.animate-breathing');
    expect(card).toBeInTheDocument();
  });

  it('handles cancel action', async () => {
    const onCancel = vi.fn();
    render(<JobCard job={mockJob} onCancel={onCancel} />);
    
    const cancelButton = screen.getByRole('button', { name: /cancel/i });
    await userEvent.click(cancelButton);
    
    expect(onCancel).toHaveBeenCalledWith('123');
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
// useJobProgress.test.ts
import { renderHook, waitFor } from '@testing-library/react';
import { useJobProgress } from './useJobProgress';
import WS from 'jest-websocket-mock';

describe('useJobProgress', () => {
  let server: WS;

  beforeEach(() => {
    server = new WS('ws://localhost:8080/ws/job123');
  });

  afterEach(() => {
    WS.clean();
  });

  it('connects to WebSocket and receives updates', async () => {
    const { result } = renderHook(() => useJobProgress('job123'));
    
    await server.connected;
    
    server.send(JSON.stringify({
      type: 'file_processing',
      processed_files: 10,
      total_files: 20,
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
  http.get('/api/jobs', () => {
    return HttpResponse.json({
      jobs: [
        { id: '1', name: 'Job 1', status: 'completed' },
        { id: '2', name: 'Job 2', status: 'running' },
      ],
    });
  }),

  http.post('/api/jobs', async ({ request }) => {
    const body = await request.json();
    return HttpResponse.json({
      job: {
        id: '3',
        name: body.name,
        status: 'created',
      },
    });
  }),
];

// setup.ts
import { setupServer } from 'msw/node';
import { handlers } from './handlers';

export const server = setupServer(...handlers);
```

#### User Flow Testing
```typescript
// CreateJobFlow.test.tsx
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { App } from './App';

describe('Create Job User Flow', () => {
  it('completes job creation flow', async () => {
    render(<App />);
    const user = userEvent.setup();
    
    // Navigate to create job
    await user.click(screen.getByText('Create Job'));
    
    // Fill form
    await user.type(screen.getByLabelText('Job Name'), 'Test Documents');
    await user.type(screen.getByLabelText('Directory Path'), '/documents');
    await user.click(screen.getByText('Scan Directory'));
    
    // Wait for scan
    await waitFor(() => {
      expect(screen.getByText('42 files found')).toBeInTheDocument();
    });
    
    // Select model and create
    await user.selectOptions(screen.getByLabelText('Model'), 'Qwen/Qwen3-Embedding-0.6B');
    await user.click(screen.getByText('Create Job'));
    
    // Verify success
    await waitFor(() => {
      expect(screen.getByText('Job created successfully')).toBeInTheDocument();
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

export default defineConfig({
  plugins: [react()],
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './src/test/setup.ts',
    coverage: {
      reporter: ['text', 'json', 'html'],
      exclude: [
        'node_modules/',
        'src/test/',
      ],
    },
  },
});
```

#### Test Setup File
```typescript
// src/test/setup.ts
import '@testing-library/jest-dom';
import { cleanup } from '@testing-library/react';
import { afterEach, beforeAll, afterAll } from 'vitest';
import { server } from './mocks/server';

// MSW setup
beforeAll(() => server.listen({ onUnhandledRequest: 'error' }));
afterAll(() => server.close());
afterEach(() => {
  server.resetHandlers();
  cleanup();
});

// Mock WebSocket
global.WebSocket = vi.fn().mockImplementation(() => ({
  send: vi.fn(),
  close: vi.fn(),
  addEventListener: vi.fn(),
  removeEventListener: vi.fn(),
}));
```

### Testing Best Practices

1. **Test Organization**
   ```
   src/
   ├── components/
   │   ├── JobCard.tsx
   │   └── JobCard.test.tsx
   ├── hooks/
   │   ├── useJobProgress.ts
   │   └── useJobProgress.test.ts
   └── test/
       ├── setup.ts
       ├── utils.tsx
       └── mocks/
   ```

2. **Test Utilities**
   ```typescript
   // test/utils.tsx
   export function renderWithProviders(
     ui: React.ReactElement,
     options?: RenderOptions
   ) {
     function Wrapper({ children }: { children: React.ReactNode }) {
       return (
         <QueryClient value={queryClient}>
           <BrowserRouter>
             {children}
           </BrowserRouter>
         </QueryClient>
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
   test('renders large job list efficiently', async () => {
     const jobs = Array.from({ length: 1000 }, (_, i) => ({
       id: `job-${i}`,
       name: `Job ${i}`,
       status: 'completed',
     }));
     
     const start = performance.now();
     render(<JobList jobs={jobs} />);
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
- Minified with esbuild

### Deployment Flow

1. Frontend builds to backend static directory
2. Backend serves the built files
3. API calls go directly to backend (no proxy)
4. WebSocket connections established directly

## Component Development Guidelines

### Component Structure

#### File Organization
```typescript
// JobCard/
// ├── index.ts           // Re-export
// ├── JobCard.tsx        // Main component
// ├── JobCard.types.ts   // TypeScript interfaces
// ├── JobCard.styles.ts  // Styled components or CSS modules
// ├── JobCard.test.tsx   // Tests
// └── JobCard.stories.tsx // Storybook stories

// JobCard.tsx
import React, { memo } from 'react';
import { JobCardProps } from './JobCard.types';
import { useJobProgress } from '@/hooks/useJobProgress';

export const JobCard = memo<JobCardProps>(({ job, onCancel, onDelete }) => {
  // Hooks at the top
  const progress = useJobProgress(job.id);
  
  // Derived state
  const isActive = job.status === 'running' || job.status === 'processing';
  
  // Event handlers
  const handleCancel = () => {
    onCancel?.(job.id);
  };
  
  // Render
  return (
    <article className="job-card" aria-label={`Job: ${job.name}`}>
      {/* Component JSX */}
    </article>
  );
});

JobCard.displayName = 'JobCard';
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
  const { data, loading, error } = useQuery<T>(url);
  return <>{children(data, loading, error)}</>;
}

// Usage
<DataFetcher url="/api/jobs">
  {(jobs, loading, error) => (
    loading ? <Spinner /> : <JobList jobs={jobs} />
  )}
</DataFetcher>
```

#### Custom Hooks Pattern
```typescript
// Encapsulate complex logic in custom hooks
function useJobManagement(jobId: string) {
  const queryClient = useQueryClient();
  const { showToast } = useUIStore();
  
  const cancelMutation = useMutation({
    mutationFn: () => jobsApi.cancel(jobId),
    onSuccess: () => {
      queryClient.invalidateQueries(['jobs']);
      showToast('Job cancelled successfully', 'success');
    },
  });
  
  return {
    cancel: cancelMutation.mutate,
    isLoading: cancelMutation.isLoading,
  };
}
```

### TypeScript Best Practices

#### Strict Type Definitions
```typescript
// Prefer interfaces for public APIs
export interface JobCardProps {
  job: Job;
  onCancel?: (jobId: string) => void;
  onDelete?: (jobId: string) => void;
  className?: string;
}

// Use type for unions and intersections
export type JobStatus = 'created' | 'running' | 'completed' | 'failed';

// Const assertions for literals
export const JOB_STATUSES = ['created', 'running', 'completed', 'failed'] as const;
export type JobStatus = typeof JOB_STATUSES[number];

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
export const JobCard = memo(JobCardComponent, (prevProps, nextProps) => {
  // Custom comparison
  return prevProps.job.id === nextProps.job.id &&
         prevProps.job.status === nextProps.job.status;
});
```

#### Virtual Scrolling
```typescript
// For large lists, use react-window
import { FixedSizeList } from 'react-window';

function VirtualJobList({ jobs }: { jobs: Job[] }) {
  const Row = ({ index, style }) => (
    <div style={style}>
      <JobCard job={jobs[index]} />
    </div>
  );
  
  return (
    <FixedSizeList
      height={600}
      itemCount={jobs.length}
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
  <JobList />
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
    <li><a href="/jobs">Jobs</a></li>
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
  aria-label="Job progress"
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
  <p id="dialog-description">Are you sure you want to delete this job?</p>
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
function JobProgress({ job }: { job: Job }) {
  return (
    <>
      <div aria-live="polite" aria-atomic="true" className="sr-only">
        Job {job.name} is {job.progress}% complete
      </div>
      <ProgressBar value={job.progress} />
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

test('JobCard is accessible', async () => {
  const { container } = render(<JobCard job={mockJob} />);
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

## Future Considerations

### Scalability
- Consider server-side rendering for initial load
- Implement virtual scrolling for large lists
- Add pagination to search results
- Implement request deduplication

### Features
- Real-time collaboration features
- Advanced search filters
- Batch operations on jobs
- User preferences persistence
- Multi-language support (i18n)

### Technical Improvements
- Add comprehensive test coverage
- Implement error boundaries per feature
- Add performance monitoring (Web Vitals)
- Consider micro-frontends for scale
- Implement service workers for offline support

### Accessibility Roadmap
- Full WCAG 2.1 AA compliance
- WCAG 3.0 preparation
- Automated accessibility testing in CI/CD
- User testing with assistive technology users

## Conclusion

The Semantik frontend is a well-structured React application that effectively manages complex state, real-time updates, and user interactions. The architecture supports future growth while maintaining clean separation of concerns and type safety throughout.