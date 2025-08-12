# API Client Layer - Cleanroom Documentation

## 1. Component Overview

The API Client Layer serves as the foundational communication interface between the Semantik React frontend and the FastAPI backend services. This layer provides a unified, type-safe, and resilient API for all backend interactions, including REST endpoints and WebSocket connections.

### Core Responsibilities
- HTTP request/response handling via Axios
- Authentication token injection and management
- WebSocket connection management for real-time updates
- Request retry logic and error handling
- Type-safe API method definitions
- Progress tracking for long-running operations
- Request cancellation support

### Directory Structure
```
apps/webui-react/src/services/
├── api/
│   └── v2/
│       ├── client.ts          # Axios client configuration
│       ├── types.ts           # API type definitions
│       ├── index.ts           # Module exports
│       ├── auth.ts            # Authentication endpoints
│       ├── collections.ts     # Collection management
│       ├── operations.ts      # Operation tracking
│       ├── documents.ts       # Document handling
│       ├── search.ts          # Search functionality
│       ├── chunking.ts        # Advanced chunking with retry
│       ├── directoryScan.ts   # Directory scanning
│       ├── settings.ts        # Settings management
│       └── system.ts          # System status checks
└── websocket.ts               # WebSocket service implementation
```

## 2. Architecture & Design Patterns

### 2.1 Modular API Organization

The API client is organized into domain-specific modules, each responsible for a specific area of functionality:

```typescript
// Unified API object pattern
export const v2Api = {
  collections: collectionsV2Api,
  operations: operationsV2Api,
  search: searchV2Api,
};
```

### 2.2 Axios Client Configuration

The central Axios instance is configured with:
- **Base URL**: Empty string (uses relative URLs for same-origin requests)
- **Default Headers**: Content-Type: application/json
- **Request Interceptor**: Automatic Bearer token injection
- **Response Interceptor**: 401 handling and auto-logout

### 2.3 WebSocket Architecture

The WebSocket service uses an EventEmitter pattern with:
- **Secure Authentication Flow**: Token sent via message, not URL
- **Automatic Reconnection**: Exponential backoff with jitter
- **Heartbeat Mechanism**: Ping/pong for connection health
- **Message Queueing**: Buffering during disconnection

### 2.4 Request Management Pattern

Advanced modules (like chunking) implement:
- **Request Tracking**: Unique IDs for cancellation
- **Retry Logic**: Exponential backoff for transient failures
- **Progress Callbacks**: Real-time progress updates
- **Cancellation Support**: Abort ongoing requests

## 3. Key Interfaces & Contracts

### 3.1 Core Type Definitions

```typescript
// Collection domain types
export interface Collection {
  id: string;                      // UUID
  name: string;
  description?: string;
  owner_id: number;
  vector_store_name: string;
  embedding_model: string;
  quantization: string;
  chunking_strategy?: string;
  chunking_config?: Record<string, number | boolean | string>;
  status: CollectionStatus;
  status_message?: string;
  metadata?: Record<string, unknown>;
  document_count: number;
  vector_count: number;
  total_size_bytes?: number;
  created_at: string;              // ISO 8601
  updated_at: string;              // ISO 8601
}

// Operation tracking
export interface Operation {
  id: string;                      // UUID
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

// Search functionality
export interface SearchRequest {
  query: string;
  collection_uuids?: string[];
  k?: number;
  score_threshold?: number;
  search_type?: 'semantic' | 'question' | 'code' | 'hybrid';
  use_reranker?: boolean;
  rerank_model?: string | null;
  hybrid_alpha?: number;
  hybrid_mode?: 'reciprocal_rank' | 'relative_score';
  keyword_mode?: 'bm25';
}

// WebSocket messages
export interface WebSocketMessage<T = unknown> {
  type: string;
  data: T;
  timestamp: number;
  requestId?: string;
}
```

### 3.2 API Method Signatures

```typescript
// Collections API
collectionsV2Api = {
  list: (params?: PaginationParams) => Promise<AxiosResponse<CollectionListResponse>>,
  get: (uuid: string) => Promise<AxiosResponse<Collection>>,
  create: (data: CreateCollectionRequest) => Promise<AxiosResponse<Collection>>,
  update: (uuid: string, data: UpdateCollectionRequest) => Promise<AxiosResponse<Collection>>,
  delete: (uuid: string) => Promise<AxiosResponse<void>>,
  addSource: (uuid: string, data: AddSourceRequest) => Promise<AxiosResponse<Operation>>,
  removeSource: (uuid: string, data: RemoveSourceRequest) => Promise<AxiosResponse<Operation>>,
  reindex: (uuid: string, data?: ReindexRequest) => Promise<AxiosResponse<Operation>>,
}

// Chunking API with advanced features
chunkingApi = {
  preview: (request: ChunkingPreviewRequest, options?: {
    requestId?: string;
    onProgress?: ProgressCallback;
    retryConfig?: Partial<RetryConfig>;
  }) => Promise<ChunkingPreviewResponse>,
  
  compare: (request: CompareRequest, options?: RequestOptions) => Promise<ChunkingComparisonResult[]>,
  
  cancelRequest: (requestId: string, reason?: string) => boolean,
}
```

## 4. Data Flow & Dependencies

### 4.1 Request Flow

```
Component → Store Action → API Module → Axios Client → Backend
                                            ↓
                                     Request Interceptor
                                     (Token Injection)
                                            ↓
                                        HTTP Request
                                            ↓
                                     Response Interceptor
                                     (Error Handling)
                                            ↓
Component ← Store Update ← API Module ← Response
```

### 4.2 Authentication Flow

1. **Token Storage**: Zustand store persisted to localStorage
2. **Token Retrieval**: Request interceptor reads from store state
3. **Token Injection**: Added as Bearer token to Authorization header
4. **401 Handling**: Auto-logout and redirect to /login

### 4.3 WebSocket Connection Flow

```
1. Connect to WebSocket (without token in URL)
2. Send AUTH_REQUEST message with token
3. Receive AUTH_SUCCESS/AUTH_ERROR
4. Start heartbeat mechanism (if authenticated)
5. Process domain messages
6. Handle reconnection on disconnect
```

### 4.4 Dependencies

- **axios**: HTTP client library
- **zustand**: State management (auth token source)
- **EventEmitter**: WebSocket event handling
- **crypto.randomUUID**: Request ID generation

## 5. Critical Implementation Details

### 5.1 Authentication Header Injection

```typescript
// Request interceptor in client.ts
apiClient.interceptors.request.use(
  (config) => {
    const state = useAuthStore.getState();
    const token = state.token;
    
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    
    return config;
  },
  (error) => Promise.reject(error)
);
```

### 5.2 Error Response Handling

```typescript
// Response interceptor for auth errors
apiClient.interceptors.response.use(
  (response) => response,
  async (error) => {
    if (error.response?.status === 401) {
      await useAuthStore.getState().logout();
      
      if (typeof window !== 'undefined' && window.location.pathname !== '/login') {
        // Navigation handling for tests vs production
        const navigate = (window as any).__navigate;
        if (navigate) {
          navigate('/login');
        } else {
          window.location.href = '/login';
        }
      }
    }
    return Promise.reject(error);
  }
);
```

### 5.3 Retry Logic Implementation

```typescript
// Exponential backoff with jitter
function calculateDelay(attempt: number, config: RetryConfig): number {
  const delay = Math.min(
    config.baseDelay * Math.pow(2, attempt),
    config.maxDelay
  );
  return delay + Math.random() * 1000; // Add jitter
}

// Retry execution
async function executeWithRetry<T>(
  requestFn: () => Promise<T>,
  config: RetryConfig,
  attempt: number = 0
): Promise<T> {
  try {
    return await requestFn();
  } catch (error) {
    const axiosError = error as AxiosError;
    const shouldRetry = 
      attempt < config.maxRetries &&
      axiosError.response &&
      config.retryableStatuses.includes(axiosError.response.status);

    if (shouldRetry) {
      const delay = calculateDelay(attempt, config);
      await new Promise(resolve => setTimeout(resolve, delay));
      return executeWithRetry(requestFn, config, attempt + 1);
    }
    throw error;
  }
}
```

### 5.4 Request Cancellation

```typescript
class RequestManager {
  private activeRequests = new Map<string, CancelTokenSource>();

  register(id: string, source: CancelTokenSource): void {
    this.activeRequests.set(id, source);
  }

  cancel(id: string, reason?: string): boolean {
    const source = this.activeRequests.get(id);
    if (source) {
      source.cancel(reason || 'Request cancelled by user');
      this.activeRequests.delete(id);
      return true;
    }
    return false;
  }
}
```

## 6. Security Considerations

### 6.1 Token Management

- **Storage**: Tokens stored in Zustand store, persisted to localStorage
- **Transmission**: Always sent via HTTPS in production
- **Injection**: Added to headers, never in URL parameters
- **Expiration**: 401 responses trigger automatic logout

### 6.2 WebSocket Security

```typescript
// SECURE: Token sent as message, not in URL
private sendAuthentication(): void {
  const authMessage: WebSocketMessage<AuthRequestData> = {
    type: ChunkingMessageType.AUTH_REQUEST,
    data: { token: this.authenticationToken },
    timestamp: Date.now(),
  };
  
  if (this.ws?.readyState === WebSocketState.OPEN) {
    this.ws.send(JSON.stringify(authMessage));
  }
}
```

### 6.3 CORS Configuration

- Development: Vite proxy handles CORS
- Production: Same-origin requests (no CORS)

### 6.4 Input Validation

- Type-safe interfaces enforce correct data structures
- Backend validates all inputs (defense in depth)

## 7. Testing Requirements

### 7.1 Unit Testing

```typescript
// Mock Axios for API testing
jest.mock('./client', () => ({
  default: {
    get: jest.fn(),
    post: jest.fn(),
    put: jest.fn(),
    delete: jest.fn(),
    interceptors: {
      request: { use: jest.fn() },
      response: { use: jest.fn() }
    }
  }
}));

// Test authentication injection
test('should inject auth token', async () => {
  const mockToken = 'test-token';
  useAuthStore.setState({ token: mockToken });
  
  await collectionsV2Api.list();
  
  expect(apiClient.get).toHaveBeenCalledWith(
    '/api/v2/collections',
    expect.objectContaining({
      headers: expect.objectContaining({
        Authorization: `Bearer ${mockToken}`
      })
    })
  );
});
```

### 7.2 Integration Testing

- Test actual API endpoints with mock backend
- Verify retry logic with simulated failures
- Test WebSocket reconnection scenarios
- Validate progress tracking accuracy

### 7.3 Error Scenario Testing

- Network failures
- 401 unauthorized responses
- 429 rate limiting
- 500 server errors
- WebSocket disconnections
- Request cancellations

## 8. Common Pitfalls & Best Practices

### 8.1 Common Pitfalls

1. **Direct localStorage Access**: Always use Zustand store for auth state
2. **Token in URLs**: Never put tokens in query parameters
3. **Missing Error Handling**: Always handle rejected promises
4. **Synchronous Token Checks**: Use `getState()` for current token
5. **WebSocket Memory Leaks**: Always disconnect on unmount

### 8.2 Best Practices

```typescript
// GOOD: Use store for auth state
const token = useAuthStore.getState().token;

// BAD: Direct localStorage access
const token = localStorage.getItem('auth-token');

// GOOD: Handle errors properly
try {
  const response = await collectionsV2Api.create(data);
  return response.data;
} catch (error) {
  const message = handleApiError(error);
  console.error('Collection creation failed:', message);
  throw error;
}

// GOOD: Clean up WebSocket on unmount
useEffect(() => {
  const ws = getChunkingWebSocket();
  ws.connect();
  
  return () => {
    ws.disconnect();
  };
}, []);
```

### 8.3 Error Handling Patterns

```typescript
// Standardized error extraction
export function handleApiError(error: unknown): string {
  if (axios.isCancel(error)) {
    return 'Request was cancelled';
  }
  
  if (error instanceof Error && 'response' in error) {
    const axiosError = error as AxiosError<{ detail?: string }>;
    if (axiosError.response?.data?.detail) {
      return axiosError.response.data.detail;
    }
  }
  
  return 'An unexpected error occurred';
}
```

## 9. Configuration & Environment

### 9.1 Development Configuration

```typescript
// vite.config.ts proxy configuration
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
}
```

### 9.2 Production Configuration

- **Base URL**: Relative paths (same origin)
- **WebSocket URL**: Derived from window.location
- **HTTPS**: Enforced by deployment environment

### 9.3 Timeout Configuration

```typescript
const DEFAULT_RETRY_CONFIG: RetryConfig = {
  maxRetries: 3,
  baseDelay: 1000,  // 1 second
  maxDelay: 30000,  // 30 seconds
  retryableStatuses: [429, 500, 502, 503, 504],
};

// WebSocket configuration
const WS_CONFIG = {
  reconnectInterval: 3000,
  reconnectMaxAttempts: 5,
  heartbeatInterval: 30000,
  connectionTimeout: 5000,
};
```

## 10. Integration Points

### 10.1 Zustand Store Integration

```typescript
// Store initiates API calls
const collectionStore = {
  fetchCollections: async () => {
    set({ loading: true });
    try {
      const response = await collectionsV2Api.list();
      set({ collections: response.data.collections });
    } catch (error) {
      set({ error: handleApiError(error) });
    } finally {
      set({ loading: false });
    }
  }
};
```

### 10.2 Component Integration

```typescript
// Components use stores, not API directly
function CollectionList() {
  const { collections, fetchCollections } = useCollectionStore();
  
  useEffect(() => {
    fetchCollections();
  }, []);
  
  return (
    <div>
      {collections.map(c => <CollectionCard key={c.id} {...c} />)}
    </div>
  );
}
```

### 10.3 WebSocket Integration

```typescript
// Real-time updates via WebSocket
const ws = new WebSocketService({
  url: `${baseUrl}/ws/operations/${operationId}`
});

ws.on('progress', (data: OperationProgressMessage) => {
  updateOperationProgress(data);
});

ws.on('completed', (data: OperationCompleteMessage) => {
  markOperationComplete(data);
});

ws.connect();
```

### 10.4 Error Boundary Integration

```typescript
// Global error handling
class ApiErrorBoundary extends Component {
  componentDidCatch(error: Error) {
    if (error.message.includes('401')) {
      // Handle authentication errors
      this.redirectToLogin();
    }
  }
}
```

## Critical Files Reference

### Core Configuration
- `/apps/webui-react/src/services/api/v2/client.ts` - Axios instance and interceptors
- `/apps/webui-react/vite.config.ts` - Proxy configuration for development

### API Modules
- `/apps/webui-react/src/services/api/v2/collections.ts` - Collection CRUD operations
- `/apps/webui-react/src/services/api/v2/operations.ts` - Operation tracking
- `/apps/webui-react/src/services/api/v2/chunking.ts` - Advanced chunking with retry
- `/apps/webui-react/src/services/api/v2/auth.ts` - Authentication endpoints

### Type Definitions
- `/apps/webui-react/src/services/api/v2/types.ts` - API-specific types
- `/apps/webui-react/src/types/collection.ts` - Domain model types
- `/apps/webui-react/src/types/chunking.ts` - Chunking strategy types

### WebSocket Implementation
- `/apps/webui-react/src/services/websocket.ts` - WebSocket service with auth

### State Management
- `/apps/webui-react/src/stores/authStore.ts` - Authentication state and token management

## Implementation Notes for LLM Agents

1. **Always use the API modules through Zustand stores** - Never call API methods directly from components
2. **Handle all promise rejections** - Use try/catch or .catch() for all API calls
3. **Use type-safe interfaces** - Import types from types.ts files
4. **Clean up WebSocket connections** - Always disconnect on component unmount
5. **Use request IDs for cancellation** - Generate unique IDs for cancellable operations
6. **Respect retry configurations** - Don't override unless necessary
7. **Log errors appropriately** - Use console.error for debugging, user-friendly messages for UI
8. **Test auth flows** - Ensure 401 handling works correctly
9. **Monitor WebSocket health** - Use heartbeat mechanism for connection status
10. **Follow security best practices** - Never expose tokens in URLs or logs

This documentation represents the complete API client layer implementation as of the current codebase state. Any modifications should maintain backward compatibility and follow the established patterns.