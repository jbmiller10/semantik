# UI_COMPONENTS - Semantik Frontend Component Library

## 1. Component Overview

The UI_COMPONENTS module represents the React component library for the Semantik application, providing all user interface elements for document collection management, search functionality, and real-time operation monitoring. This library is built on React 19 with TypeScript, utilizing TailwindCSS for styling and following composition-based patterns for maximum reusability.

### Core Responsibilities
- **Collection Management UI**: Create, view, update, and delete document collections
- **Search Interface**: Advanced semantic and hybrid search with real-time results
- **Operation Monitoring**: WebSocket-based real-time progress tracking
- **Document Viewing**: Multi-format document preview and highlighting
- **System Feedback**: Toast notifications and error boundaries
- **Modal System**: Transactional UI flows for data modification

### Component Categories

#### Layout Components
- `Layout.tsx` - Main application shell with navigation and global modals
- `ErrorBoundary.tsx` - Application-wide error catching and recovery

#### Collection Management
- `CollectionsDashboard.tsx` - Main collections listing and filtering
- `CollectionCard.tsx` - Individual collection display with status indicators
- `CollectionDetailsModal.tsx` - Detailed collection view and management
- `CollectionOperations.tsx` - Operation history and management for a collection
- `CollectionMultiSelect.tsx` - Multi-collection selection for search

#### Modal Components
- `CreateCollectionModal.tsx` - New collection creation with chunking config
- `DeleteCollectionModal.tsx` - Safe collection deletion with confirmation
- `RenameCollectionModal.tsx` - Collection renaming interface
- `ReindexCollectionModal.tsx` - Reindexing configuration and trigger
- `AddDataToCollectionModal.tsx` - Add new data sources to collections
- `DocumentViewerModal.tsx` - Document preview wrapper modal

#### Search Components
- `SearchInterface.tsx` - Main search form with advanced options
- `SearchResults.tsx` - Search result display with highlighting
- `RerankingConfiguration.tsx` - AI reranking settings

#### Operation Tracking
- `ActiveOperationsTab.tsx` - Global operations monitoring dashboard
- `OperationProgress.tsx` - Individual operation progress display

#### Document Handling
- `DocumentViewer.tsx` - Multi-format document rendering engine

#### Chunking Strategy
- `chunking/SimplifiedChunkingStrategySelector.tsx` - Simplified strategy selection
- `chunking/ChunkingParameterTuner.tsx` - Advanced parameter configuration
- `chunking/ChunkingPreviewPanel.tsx` - Real-time chunking preview
- `chunking/ChunkingComparisonView.tsx` - Strategy comparison tool
- `chunking/ChunkingAnalyticsDashboard.tsx` - Chunking metrics visualization
- `chunking/ChunkingStrategyGuide.tsx` - User guidance modal

#### Notification System
- `Toast.tsx` - Global notification display component
- `GPUMemoryError.tsx` - Specialized GPU memory error handling

#### Development Tools
- `FeatureVerification.tsx` - Feature flag and capability verification

## 2. Architecture & Design Patterns

### Component Composition Pattern
Components follow a hierarchical composition pattern where complex UI is built from smaller, reusable pieces:

```typescript
// Example: CollectionsDashboard composes multiple sub-components
<CollectionsDashboard>
  <SearchInput />
  <FilterDropdown />
  <CollectionGrid>
    <CollectionCard /> // Repeated for each collection
  </CollectionGrid>
  <CreateCollectionModal /> // Conditionally rendered
</CollectionsDashboard>
```

### Modal Architecture
Modals follow a consistent pattern with controlled visibility through parent state:

```typescript
interface ModalProps {
  onClose: () => void;        // Required: Close handler
  onSuccess?: () => void;      // Optional: Success callback
  // Additional props specific to modal purpose
}

// Parent controls modal visibility
const [showModal, setShowModal] = useState(false);

// Modal handles its own form state and validation
```

### Real-time Updates Pattern
WebSocket integration for live updates uses custom hooks:

```typescript
// Component subscribes to operation updates
const { isConnected } = useOperationProgress(operationId, {
  onComplete: handleComplete,
  onError: handleError,
  showToasts: false  // Component controls toast display
});
```

### Error Boundary Strategy
Class-based error boundary wraps the entire application:

```typescript
class ErrorBoundary extends Component<Props, State> {
  static getDerivedStateFromError(error: Error): State
  componentDidCatch(error: Error, errorInfo: ErrorInfo): void
  // Provides fallback UI and error recovery
}
```

## 3. Key Interfaces & Contracts

### Component Props Contracts

#### Modal Base Contract
```typescript
interface BaseModalProps {
  onClose: () => void;
  onSuccess?: () => void;
}
```

#### Collection Component Props
```typescript
interface CollectionCardProps {
  collection: Collection;
}

interface CollectionOperationsProps {
  collection: Collection;
}
```

#### Search Component Props
```typescript
interface SearchInterfaceProps {
  // No required props - uses internal state management
}

interface SearchResultsProps {
  results: SearchResult[];
  loading: boolean;
  error: string | null;
  query: string;
}
```

#### Operation Tracking Props
```typescript
interface OperationProgressProps {
  operation: Operation;
  className?: string;
  showDetails?: boolean;
  onComplete?: () => void;
  onError?: (error: string) => void;
}
```

### Event Handler Patterns
```typescript
// Form submission handlers
handleSubmit: (e: React.FormEvent) => Promise<void>

// Input change handlers
handleChange: (field: string, value: any) => void

// Action handlers
handleDelete: () => Promise<void>
handleRefresh: () => void
```

## 4. Data Flow & Dependencies

### Store Connections
Components connect to Zustand stores for state management:

```typescript
// UI Store - Global UI state
const { toasts, addToast, removeToast } = useUIStore();

// Search Store - Search state and parameters
const { searchParams, updateSearchParams } = useSearchStore();

// Chunking Store - Chunking configuration
const { strategyConfig, setStrategy } = useChunkingStore();

// Auth Store - User authentication
const { user, logout } = useAuthStore();
```

### Data Fetching Pattern
React Query hooks manage server state:

```typescript
// Collections data fetching
const { data: collections, isLoading, error, refetch } = useCollections();

// Operation tracking
const { data: operations } = useQuery({
  queryKey: ['operations', collectionId],
  queryFn: () => operationsV2Api.list({ collection_id: collectionId })
});
```

### Props vs Store Data
- **Props**: Used for component-specific data passed from parent
- **Store**: Used for global application state (auth, UI, search params)
- **React Query**: Used for server state (collections, operations, documents)

## 5. Critical Implementation Details

### Modal System Implementation

#### Portal Rendering
Modals render at root level using fixed positioning:

```typescript
<div className="fixed inset-0 bg-gray-500 bg-opacity-75 flex items-center justify-center p-4 z-50">
  <div className="bg-white rounded-lg max-w-lg w-full max-h-[90vh] overflow-y-auto">
    {/* Modal content */}
  </div>
</div>
```

#### Form State Management
Modals maintain local form state with validation:

```typescript
const [formData, setFormData] = useState<CreateCollectionRequest>({
  name: '',
  description: '',
  embedding_model: DEFAULT_EMBEDDING_MODEL,
  quantization: DEFAULT_QUANTIZATION,
});

const [errors, setErrors] = useState<Record<string, string>>({});

const validateForm = (): boolean => {
  const newErrors: Record<string, string> = {};
  // Validation logic
  setErrors(newErrors);
  return Object.keys(newErrors).length === 0;
};
```

### Real-time WebSocket Updates

#### Operation Progress Tracking
```typescript
// Hook manages WebSocket connection lifecycle
useOperationProgress(operationId, {
  onComplete: () => {
    // Handle completion
    refetchCollections();
  },
  onError: (error) => {
    // Handle error
    addToast({ type: 'error', message: error });
  }
});
```

#### Message Processing
```typescript
// WebSocket message handler parses backend format
const message = JSON.parse(event.data);
if (message.type === 'operation_completed') {
  updateOperationInCache(operationId, { status: 'completed' });
}
```

### Toast Notification System

#### Auto-dismissal Logic
```typescript
addToast: (toast) => {
  const id = Date.now().toString();
  set((state) => ({ toasts: [...state.toasts, { ...toast, id }] }));
  
  if (toast.duration !== 0) {
    setTimeout(() => {
      set((state) => ({
        toasts: state.toasts.filter((t) => t.id !== id),
      }));
    }, toast.duration || 5000);
  }
}
```

## 6. Security Considerations

### XSS Prevention

#### Input Sanitization
All user inputs are sanitized before display:

```typescript
// DOMPurify for HTML content
const sanitizedHtml = window.DOMPurify.sanitize(htmlContent);

// Text content uses React's built-in escaping
<p>{userInput}</p> // Automatically escaped by React
```

#### Content Security Policy
Document viewer enforces CSP for external content:

```typescript
// Script loading with SRI hashes
const SCRIPT_CONFIGS = {
  jszip: {
    url: 'https://unpkg.com/jszip@3.10.1/dist/jszip.min.js',
    integrity: 'sha384-...',
    crossOrigin: 'anonymous'
  }
};
```

### Form Validation

#### Client-side Validation
```typescript
// Validate before submission
if (!formData.name.trim()) {
  newErrors.name = 'Collection name is required';
}
if (formData.name.length > 100) {
  newErrors.name = 'Collection name must be 100 characters or less';
}
```

#### Server Response Validation
```typescript
// Type-safe API responses
const response = await collectionsV2Api.create(formData);
if (!response.data?.id) {
  throw new Error('Invalid server response');
}
```

### Authentication Headers
All API calls include authentication:

```typescript
// Automatic auth header injection via axios interceptor
apiClient.interceptors.request.use((config) => {
  const token = getAuthToken();
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});
```

## 7. Testing Requirements

### Component Testing Strategy

#### Unit Tests
Test individual component behavior:

```typescript
describe('Toast', () => {
  it('renders error toast correctly', () => {
    render(<Toast />);
    expect(screen.getByText('Error')).toBeInTheDocument();
    expect(toastElement).toHaveClass('border-red-500');
  });
});
```

#### Integration Tests
Test component interactions:

```typescript
describe('CreateCollectionModal', () => {
  it('creates collection and adds source', async () => {
    const user = userEvent.setup();
    render(<CreateCollectionModal onClose={mockClose} onSuccess={mockSuccess} />);
    
    await user.type(screen.getByLabelText('Collection Name'), 'Test Collection');
    await user.click(screen.getByText('Create Collection'));
    
    expect(mockCreateCollection).toHaveBeenCalled();
    expect(mockSuccess).toHaveBeenCalled();
  });
});
```

#### WebSocket Tests
Mock WebSocket connections for testing:

```typescript
describe('OperationProgress - WebSocket', () => {
  it('updates progress in real-time', () => {
    const { rerender } = render(<OperationProgress operation={mockOperation} />);
    
    // Simulate WebSocket message
    mockWebSocket.send({ type: 'progress', data: { progress: 50 } });
    
    rerender(<OperationProgress operation={mockOperation} />);
    expect(screen.getByText('50%')).toBeInTheDocument();
  });
});
```

### Test Coverage Requirements
- **Component Rendering**: 100% of components must render without errors
- **User Interactions**: All interactive elements must be tested
- **Error States**: All error conditions must have test coverage
- **Accessibility**: ARIA attributes and keyboard navigation must be tested

## 8. Common Pitfalls & Best Practices

### Accessibility Best Practices

#### ARIA Labels
```typescript
// Always provide ARIA labels for interactive elements
<button aria-label="Close modal">
<input aria-describedby="input-help-text">
<div role="status" aria-live="polite">
```

#### Keyboard Navigation
```typescript
// Handle escape key for modals
useEffect(() => {
  const handleEscape = (e: KeyboardEvent) => {
    if (e.key === 'Escape' && !isSubmitting) {
      onClose();
    }
  };
  document.addEventListener('keydown', handleEscape);
  return () => document.removeEventListener('keydown', handleEscape);
}, [onClose, isSubmitting]);
```

### Performance Optimization

#### Memoization
```typescript
// Memoize expensive computations
const sortedCollections = useMemo(() => 
  [...collections].sort((a, b) => 
    new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime()
  ), [collections]);
```

#### Lazy Loading
```typescript
// Lazy load heavy components
const DocumentViewer = lazy(() => import('./DocumentViewer'));
```

#### Debouncing
```typescript
// Debounce search input
const debouncedSearch = useDebouncedCallback(
  (value: string) => {
    updateSearchParams({ query: value });
  },
  300
);
```

### Common Mistakes to Avoid

1. **Direct State Mutations**
```typescript
// BAD: Mutating state directly
state.toasts.push(newToast);

// GOOD: Creating new state
set((state) => ({ toasts: [...state.toasts, newToast] }));
```

2. **Missing Cleanup**
```typescript
// BAD: Not cleaning up subscriptions
useEffect(() => {
  const subscription = subscribe();
  // Missing cleanup
});

// GOOD: Proper cleanup
useEffect(() => {
  const subscription = subscribe();
  return () => subscription.unsubscribe();
}, []);
```

3. **Uncontrolled Forms**
```typescript
// BAD: Uncontrolled input
<input defaultValue={value} />

// GOOD: Controlled input
<input value={value} onChange={handleChange} />
```

## 9. Configuration & Environment

### Styling Configuration

#### TailwindCSS Classes
Standard utility classes for consistent styling:

```typescript
// Button styles
"px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"

// Input styles  
"border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"

// Card styles
"bg-white rounded-lg shadow-md p-6"
```

#### Custom CSS Classes
Limited custom CSS for animations:

```css
/* SimplifiedChunkingStrategySelector.css */
@keyframes slideDown {
  from { opacity: 0; transform: translateY(-10px); }
  to { opacity: 1; transform: translateY(0); }
}

.animate-slideDown {
  animation: slideDown 0.3s ease-out;
}
```

### Theme Configuration
Theme values defined in Tailwind config:

```javascript
// tailwind.config.js
module.exports = {
  theme: {
    extend: {
      colors: {
        primary: 'blue',
        danger: 'red',
        success: 'green',
        warning: 'yellow'
      }
    }
  }
}
```

### Environment Variables
```typescript
// Development-only features
{import.meta.env.DEV && (
  <Link to="/verification">Verification</Link>
)}
```

## 10. Integration Points

### Store Integration

#### Zustand Store Connections
```typescript
// UI Store - Toast notifications, modal visibility
import { useUIStore } from '../stores/uiStore';

// Auth Store - User authentication state  
import { useAuthStore } from '../stores/authStore';

// Search Store - Search parameters and results
import { useSearchStore } from '../stores/searchStore';

// Chunking Store - Chunking strategy configuration
import { useChunkingStore } from '../stores/chunkingStore';
```

### Hook Integration

#### Custom Hooks Usage
```typescript
// Collection data management
import { useCollections, useCreateCollection } from '../hooks/useCollections';

// Operation tracking
import { useOperationProgress } from '../hooks/useOperationProgress';

// WebSocket connections
import { useWebSocket } from '../hooks/useWebSocket';

// Directory scanning
import { useDirectoryScan } from '../hooks/useDirectoryScan';

// Reranking availability
import { useRerankingAvailability } from '../hooks/useRerankingAvailability';
```

### API Service Integration

#### V2 API Clients
```typescript
// Collection operations
import { collectionsV2Api } from '../services/api/v2/collections';

// Operation management
import { operationsV2Api } from '../services/api/v2/operations';

// Search functionality
import { searchV2Api } from '../services/api/v2/collections';

// Document access
import { documentsV2Api } from '../services/api/v2';
```

### Router Integration
```typescript
// React Router navigation
import { useNavigate, useLocation, Outlet } from 'react-router-dom';

// Programmatic navigation
const navigate = useNavigate();
navigate(`/collections/${collectionId}`);
```

### Utility Integration

#### Form Styling Utilities
```typescript
import { getInputClassName, getInputClassNameWithBase } from '../utils/formStyles';

// Usage
className={getInputClassName(!!errors.name, isSubmitting)}
```

#### Error Handling Utilities
```typescript
import { 
  getErrorMessage, 
  isAxiosError,
  getInsufficientMemoryErrorDetails 
} from '../utils/errorUtils';
```

#### Search Validation
```typescript
import { DEFAULT_VALIDATION_RULES } from '../utils/searchValidation';
```

## Component Interaction Flow

### Collection Creation Flow
1. User clicks "Create Collection" → `showCreateModal = true`
2. `CreateCollectionModal` renders with form
3. User fills form and submits
4. Modal validates input locally
5. API call to create collection
6. On success: WebSocket monitors INDEX operation
7. When INDEX completes: Add source if provided
8. Close modal and refresh collection list

### Search Flow
1. User enters query in `SearchInterface`
2. Selects collections via `CollectionMultiSelect`
3. Configures search parameters (reranking, hybrid mode)
4. Submits search → API call
5. Results displayed in `SearchResults`
6. Click result → Opens `DocumentViewerModal`
7. Document rendered with query highlighting

### Operation Monitoring Flow
1. Operation initiated (create, reindex, add source)
2. `useOperationProgress` hook establishes WebSocket connection
3. Real-time updates received and processed
4. UI updates via React Query cache mutations
5. Progress displayed in `OperationProgress` component
6. Completion triggers callbacks and toasts

## Performance Considerations

### Component Optimization
- Use React.memo for expensive renders
- Implement virtual scrolling for large lists
- Lazy load modals and heavy components
- Debounce search and filter inputs
- Minimize re-renders with proper dependency arrays

### Bundle Size Management
- Code split by route
- Dynamic imports for large libraries
- Tree shaking via ES modules
- Minimize TailwindCSS with PurgeCSS

### Runtime Performance
- Batch state updates
- Use CSS transforms for animations
- Optimize images and assets
- Implement progressive enhancement
- Cache API responses appropriately

## Debugging & Development

### Component DevTools
- React DevTools for component inspection
- Redux DevTools for Zustand state
- Network tab for API calls
- WebSocket frames inspector
- Console for debug logging

### Common Debug Points
```typescript
// Collection management
console.log('Collection created:', response.data);

// WebSocket messages
console.log('WS message received:', message);

// State updates
console.log('Store state updated:', state);
```

### Error Tracking
- Error boundaries catch component errors
- API errors logged with full context
- WebSocket disconnections tracked
- Form validation errors displayed inline

## Future Considerations

### Planned Enhancements
- Virtualized collection grid for large datasets
- Drag-and-drop file upload
- Batch operations UI
- Advanced search query builder
- Customizable dashboard layouts

### Technical Debt
- Migrate remaining class components to hooks
- Implement comprehensive E2E tests
- Add performance monitoring
- Enhance accessibility compliance
- Improve mobile responsiveness

### Scalability Considerations
- Implement pagination for all lists
- Add request caching strategies
- Optimize WebSocket connections
- Consider server-side rendering
- Plan for internationalization