# Frontend State Management Architecture

> **Location:** `apps/webui-react/src/stores/`, `services/`, `hooks/`

## Overview

State management uses a dual approach:
- **Zustand** for client state (auth, UI preferences)
- **React Query** for server state (collections, search results)

## Zustand Stores

### authStore.ts
Authentication state and tokens.

```typescript
interface AuthState {
  token: string | null;
  refreshToken: string | null;
  user: User | null;
  setAuth: (token: string, user: User, refreshToken?: string) => void;
  logout: () => Promise<void>;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      token: null,
      refreshToken: null,
      user: null,
      setAuth: (token, user, refreshToken) => set({ token, user, refreshToken }),
      logout: async () => {
        await authApi.logout();
        set({ token: null, user: null, refreshToken: null });
      },
    }),
    { name: 'auth-storage' }
  )
);
```

### searchStore.ts
Search parameters, results, and validation.

```typescript
interface SearchState {
  searchParams: SearchParams;
  results: SearchResult[];
  loading: boolean;
  error: string | null;
  validationErrors: ValidationError[];
  touched: Record<string, boolean>;
  gpuMemoryError: GPUMemoryError | null;

  validateAndUpdateSearchParams: (params: Partial<SearchParams>) => void;
  setFieldTouched: (field: string, touched?: boolean) => void;
  hasValidationErrors: () => boolean;
  getValidationError: (field: string) => string | undefined;
}
```

**Key Features:**
- Field-level validation with touched tracking
- GPU error state for reranking failures
- Abort controller for request cancellation

### uiStore.ts
UI state (toasts, modals, active tab).

```typescript
interface UIState {
  toasts: Toast[];
  activeTab: 'search' | 'collections' | 'operations';
  showDocumentViewer: { collectionId: string; docId: string } | null;
  showCollectionDetailsModal: string | null;

  addToast: (toast: Omit<Toast, 'id'>) => void;
  removeToast: (id: string) => void;
  setActiveTab: (tab: UIState['activeTab']) => void;
}
```

### chunkingStore.ts
Chunking strategy configuration.

```typescript
interface ChunkingState {
  selectedStrategy: ChunkingStrategyType;
  strategyConfig: ChunkingConfiguration;
  previewChunks: ChunkPreview[];
  comparisonResults: Record<string, ChunkingComparisonResult>;

  setStrategy: (strategy: ChunkingStrategyType) => void;
  updateConfiguration: (config: Partial<ChunkingConfiguration>) => void;
  loadPreview: (documentId: string) => Promise<void>;
  compareStrategies: (strategies: ChunkingStrategyType[]) => Promise<void>;
}
```

## React Query Integration

### Query Key Factory
```typescript
export const collectionKeys = {
  all: ['collections'] as const,
  lists: () => [...collectionKeys.all, 'list'] as const,
  list: (filters?: unknown) => [...collectionKeys.lists(), filters] as const,
  details: () => [...collectionKeys.all, 'detail'] as const,
  detail: (id: string) => [...collectionKeys.details(), id] as const,
};
```

### Query Hooks

**useCollections.ts**
```typescript
export function useCollections() {
  return useQuery({
    queryKey: collectionKeys.lists(),
    queryFn: async () => {
      const response = await collectionsV2Api.list();
      return response.data.collections;
    },
    staleTime: 5000,
    refetchInterval: (query) => {
      const hasActive = query.state.data?.some(c => c.status === 'processing');
      return hasActive ? 30000 : false;
    },
  });
}
```

**useCreateCollection.ts**
```typescript
export function useCreateCollection() {
  const queryClient = useQueryClient();
  const { addToast } = useUIStore();

  return useMutation({
    mutationFn: (data: CreateCollectionRequest) => collectionsV2Api.create(data),
    onMutate: async (newData) => {
      await queryClient.cancelQueries({ queryKey: collectionKeys.lists() });
      const previous = queryClient.getQueryData(collectionKeys.lists());
      queryClient.setQueryData(collectionKeys.lists(), old =>
        [...(old || []), { id: 'temp', ...newData, status: 'pending' }]
      );
      return { previous };
    },
    onError: (error, _vars, context) => {
      queryClient.setQueryData(collectionKeys.lists(), context?.previous);
      addToast({ type: 'error', message: error.message });
    },
    onSuccess: () => {
      addToast({ type: 'success', message: 'Collection created!' });
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: collectionKeys.lists() });
    },
  });
}
```

## API Client Layer

### Axios Configuration
`services/api/v2/client.ts`

```typescript
const apiClient = axios.create({
  baseURL: getApiBaseUrl(),
  headers: { 'Content-Type': 'application/json' },
});

// Request interceptor: Inject JWT
apiClient.interceptors.request.use((config) => {
  const token = useAuthStore.getState().token;
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Response interceptor: Handle 401
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      useAuthStore.getState().logout();
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);
```

### API Modules

**collections.ts**
```typescript
export const collectionsV2Api = {
  list: (params?: PaginationParams) =>
    apiClient.get<CollectionsResponse>('/api/v2/collections', { params }),
  get: (uuid: string) =>
    apiClient.get<Collection>(`/api/v2/collections/${uuid}`),
  create: (data: CreateCollectionRequest) =>
    apiClient.post<Collection>('/api/v2/collections', data),
  delete: (uuid: string) =>
    apiClient.delete(`/api/v2/collections/${uuid}`),
  addSource: (uuid: string, data: AddSourceRequest) =>
    apiClient.post(`/api/v2/collections/${uuid}/sources`, data),
};
```

**operations.ts**
```typescript
export const operationsV2Api = {
  get: (uuid: string) => apiClient.get<Operation>(`/api/v2/operations/${uuid}`),
  list: (params?: { status?: string }) =>
    apiClient.get<{ operations: Operation[] }>('/api/v2/operations', { params }),
  getWebSocketUrl: (operationId: string) =>
    buildWebSocketUrl(`/api/ws/operations/${operationId}`),
};
```

## WebSocket Hooks

### useWebSocket.ts
Core WebSocket connection management.

```typescript
export function useWebSocket(url: string | null, options: UseWebSocketOptions = {}) {
  const [readyState, setReadyState] = useState(WebSocket.CLOSED);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!url) return;

    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => setReadyState(WebSocket.OPEN);
    ws.onmessage = options.onMessage;
    ws.onclose = (e) => {
      setReadyState(WebSocket.CLOSED);
      if (options.autoReconnect && !e.wasClean) {
        setTimeout(() => reconnect(), reconnectInterval);
      }
    };

    return () => ws.close();
  }, [url]);

  return {
    sendMessage: (data) => wsRef.current?.send(JSON.stringify(data)),
    readyState,
    isConnected: readyState === WebSocket.OPEN,
  };
}
```

### useOperationProgress.ts
Real-time operation progress tracking.

```typescript
export function useOperationProgress(operationId: string | null, options?: Options) {
  const updateOperation = useUpdateOperationInCache();

  const wsUrl = operationId ? operationsV2Api.getWebSocketUrl(operationId) : null;

  return useWebSocket(wsUrl, {
    onMessage: (event) => {
      const message = JSON.parse(event.data);
      updateOperation(operationId!, {
        status: message.status,
        progress: message.progress,
      });

      if (message.status === 'completed') {
        options?.onComplete?.();
      }
    },
  });
}
```

### useOperationsSocket.ts
Global operations broadcast for all active operations.

```typescript
export function useOperationsSocket() {
  const updateOperation = useUpdateOperationInCache();

  return useWebSocket(operationsV2Api.getGlobalWebSocketUrl(), {
    onMessage: (event) => {
      const { operation_id, ...data } = JSON.parse(event.data);
      updateOperation(operation_id, data);
    },
  });
}
```

## Data Flow Patterns

### Server State Flow
```
API Request → React Query → Cache → Component
                ↑
        Cache Invalidation ← Mutation Success
```

### Client State Flow
```
User Action → Zustand Store Update → Component Re-render
                    ↓
              Validation (if applicable)
                    ↓
              API Call (if applicable)
```

### WebSocket Update Flow
```
WebSocket Message → Parse → Update React Query Cache → Component Re-render
```

## Extension Points

### Adding a New Store
```typescript
interface MyState {
  data: unknown;
  setData: (data: unknown) => void;
}

export const useMyStore = create<MyState>()(
  persist(
    (set) => ({
      data: null,
      setData: (data) => set({ data }),
    }),
    { name: 'my-storage' }
  )
);
```

### Adding a New Query Hook
```typescript
export const myKeys = {
  all: ['myData'] as const,
  list: () => [...myKeys.all, 'list'] as const,
};

export function useMyData() {
  return useQuery({
    queryKey: myKeys.list(),
    queryFn: () => myApi.list(),
    staleTime: 5000,
  });
}
```

### Adding a New API Module
```typescript
// services/api/v2/myApi.ts
export const myApi = {
  list: () => apiClient.get<MyResponse[]>('/api/v2/my-data'),
  create: (data: MyRequest) => apiClient.post<MyResponse>('/api/v2/my-data', data),
};

// Export in index.ts
export * from './myApi';
```
