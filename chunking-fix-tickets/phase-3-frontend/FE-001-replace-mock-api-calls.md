# FE-001: Replace Mock API Calls with Real Implementation

## Ticket Information
- **Priority**: BLOCKER
- **Estimated Time**: 4 hours
- **Dependencies**: BE-002 (Backend API must be properly structured first)
- **Risk Level**: HIGH - Feature is non-functional with mocks
- **Affected Files**:
  - `apps/webui-react/src/stores/chunkingStore.ts`
  - `apps/webui-react/src/api/chunking.ts` (new)
  - `apps/webui-react/src/types/api.ts`
  - All chunking components using the store

## Context

The chunking store currently uses mock API implementations that return simulated data. This means the entire chunking feature is non-functional - it appears to work but doesn't actually communicate with the backend.

### Current Problems

```typescript
// apps/webui-react/src/stores/chunkingStore.ts - BAD
const mockApi = {
  fetchStrategies: async () => {
    // Simulated delay
    await new Promise(resolve => setTimeout(resolve, 500));
    // Returning fake data
    return [
      { id: 'character', name: 'Character', ... },
      { id: 'semantic', name: 'Semantic', ... }
    ];
  },
  previewChunking: async () => {
    // More fake data
    return { chunks: [...], statistics: {...} };
  }
};
```

## Requirements

1. Create proper API client for chunking endpoints
2. Replace ALL mock calls with real API calls
3. Implement proper error handling and retry logic
4. Add request/response type safety
5. Implement progress tracking for long operations
6. Add request cancellation support

## Technical Details

### 1. Create API Client

```typescript
// apps/webui-react/src/api/chunking.ts

import axios, { AxiosInstance, CancelTokenSource } from 'axios';
import { 
  ChunkingStrategy,
  PreviewRequest,
  PreviewResponse,
  ApplyChunkingRequest,
  ApplyChunkingResponse,
  ChunkingOperation,
  ChunkingStatistics
} from '@/types/chunking';

class ChunkingAPIClient {
  private client: AxiosInstance;
  private cancelTokens: Map<string, CancelTokenSource>;
  
  constructor(baseURL: string = '/api/v2') {
    this.client = axios.create({
      baseURL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json'
      }
    });
    
    this.cancelTokens = new Map();
    
    // Add auth interceptor
    this.client.interceptors.request.use(
      (config) => {
        const token = localStorage.getItem('auth_token');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );
    
    // Add response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      async (error) => {
        if (error.response?.status === 401) {
          // Token expired, try refresh
          await this.refreshToken();
          return this.client.request(error.config);
        }
        return Promise.reject(this.transformError(error));
      }
    );
  }
  
  /**
   * Get available chunking strategies
   */
  async getStrategies(): Promise<ChunkingStrategy[]> {
    const response = await this.client.get<{strategies: ChunkingStrategy[]}>(
      '/chunking/strategies'
    );
    return response.data.strategies;
  }
  
  /**
   * Preview chunking with progress tracking
   */
  async previewChunking(
    request: PreviewRequest,
    onProgress?: (progress: number) => void
  ): Promise<PreviewResponse> {
    // Create cancel token
    const source = axios.CancelToken.source();
    const requestId = `preview_${Date.now()}`;
    this.cancelTokens.set(requestId, source);
    
    try {
      const response = await this.client.post<PreviewResponse>(
        '/chunking/preview',
        request,
        {
          cancelToken: source.token,
          onUploadProgress: (progressEvent) => {
            if (onProgress && progressEvent.total) {
              const progress = (progressEvent.loaded / progressEvent.total) * 100;
              onProgress(progress);
            }
          }
        }
      );
      
      return response.data;
    } finally {
      this.cancelTokens.delete(requestId);
    }
  }
  
  /**
   * Apply chunking to a document
   */
  async applyChunking(
    request: ApplyChunkingRequest
  ): Promise<ApplyChunkingResponse> {
    const response = await this.client.post<ApplyChunkingResponse>(
      '/chunking/apply',
      request
    );
    return response.data;
  }
  
  /**
   * Get operation status
   */
  async getOperationStatus(
    operationId: string
  ): Promise<ChunkingOperation> {
    const response = await this.client.get<ChunkingOperation>(
      `/chunking/operations/${operationId}`
    );
    return response.data;
  }
  
  /**
   * Compare multiple strategies
   */
  async compareStrategies(
    content: string,
    strategies: string[],
    config?: Record<string, any>
  ): Promise<Record<string, PreviewResponse>> {
    const response = await this.client.post<Record<string, PreviewResponse>>(
      '/chunking/compare',
      { content, strategies, config }
    );
    return response.data;
  }
  
  /**
   * Get chunking statistics for a collection
   */
  async getStatistics(
    collectionId: string
  ): Promise<ChunkingStatistics> {
    const response = await this.client.get<ChunkingStatistics>(
      `/chunking/statistics/${collectionId}`
    );
    return response.data;
  }
  
  /**
   * Cancel a request
   */
  cancelRequest(requestId: string): void {
    const source = this.cancelTokens.get(requestId);
    if (source) {
      source.cancel('Request cancelled by user');
      this.cancelTokens.delete(requestId);
    }
  }
  
  /**
   * Cancel all pending requests
   */
  cancelAllRequests(): void {
    this.cancelTokens.forEach(source => {
      source.cancel('All requests cancelled');
    });
    this.cancelTokens.clear();
  }
  
  private async refreshToken(): Promise<void> {
    const refreshToken = localStorage.getItem('refresh_token');
    if (!refreshToken) {
      throw new Error('No refresh token available');
    }
    
    const response = await axios.post('/api/auth/refresh', {
      refresh_token: refreshToken
    });
    
    localStorage.setItem('auth_token', response.data.access_token);
    if (response.data.refresh_token) {
      localStorage.setItem('refresh_token', response.data.refresh_token);
    }
  }
  
  private transformError(error: any): Error {
    if (axios.isCancel(error)) {
      return new Error('Request cancelled');
    }
    
    if (error.response) {
      // Server responded with error
      const message = error.response.data?.detail || 
                     error.response.data?.message || 
                     `Server error: ${error.response.status}`;
      return new Error(message);
    } else if (error.request) {
      // Request made but no response
      return new Error('Network error: No response from server');
    } else {
      // Request setup error
      return new Error(error.message || 'Request failed');
    }
  }
}

// Export singleton instance
export const chunkingApi = new ChunkingAPIClient();
```

### 2. Update Store to Use Real API

```typescript
// apps/webui-react/src/stores/chunkingStore.ts

import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { chunkingApi } from '@/api/chunking';
import { 
  ChunkingStrategy,
  PreviewResult,
  ChunkingOperation 
} from '@/types/chunking';

interface ChunkingStore {
  // State
  strategies: ChunkingStrategy[];
  selectedStrategy: string | null;
  previewResult: PreviewResult | null;
  operations: Map<string, ChunkingOperation>;
  loading: boolean;
  error: string | null;
  
  // Actions
  fetchStrategies: () => Promise<void>;
  previewChunking: (params: PreviewParams) => Promise<void>;
  applyChunking: (documentId: string) => Promise<string>;
  compareStrategies: (strategies: string[]) => Promise<void>;
  cancelPreview: () => void;
  reset: () => void;
}

export const useChunkingStore = create<ChunkingStore>()(
  devtools(
    (set, get) => ({
      // Initial state
      strategies: [],
      selectedStrategy: null,
      previewResult: null,
      operations: new Map(),
      loading: false,
      error: null,
      
      // Fetch available strategies
      fetchStrategies: async () => {
        set({ loading: true, error: null });
        
        try {
          const strategies = await chunkingApi.getStrategies();
          
          set({ 
            strategies,
            loading: false,
            selectedStrategy: strategies[0]?.id || null
          });
        } catch (error) {
          set({ 
            error: error.message,
            loading: false 
          });
          
          // Retry with exponential backoff
          await get().retryWithBackoff(
            () => chunkingApi.getStrategies(),
            3
          );
        }
      },
      
      // Preview chunking with real API
      previewChunking: async (params: PreviewParams) => {
        const { strategy, content, documentId, config } = params;
        
        set({ loading: true, error: null, previewResult: null });
        
        try {
          // Show optimistic UI
          set({ previewProgress: 0 });
          
          const result = await chunkingApi.previewChunking(
            {
              strategy,
              content,
              document_id: documentId,
              config
            },
            (progress) => {
              // Update progress in UI
              set({ previewProgress: progress });
            }
          );
          
          set({ 
            previewResult: result,
            loading: false,
            previewProgress: 100
          });
          
          // Cache result for quick access
          get().cachePreviewResult(result);
          
        } catch (error) {
          if (error.message === 'Request cancelled') {
            set({ loading: false });
          } else {
            set({ 
              error: error.message,
              loading: false 
            });
            
            // Show error toast
            get().showErrorNotification(error.message);
          }
        }
      },
      
      // Apply chunking to document
      applyChunking: async (documentId: string) => {
        const { selectedStrategy, previewResult } = get();
        
        if (!selectedStrategy) {
          throw new Error('No strategy selected');
        }
        
        set({ loading: true, error: null });
        
        try {
          const response = await chunkingApi.applyChunking({
            document_id: documentId,
            strategy: selectedStrategy,
            config: previewResult?.config
          });
          
          // Track operation
          const operation: ChunkingOperation = {
            id: response.operation_id,
            status: 'pending',
            progress: 0,
            created_at: new Date().toISOString()
          };
          
          get().operations.set(response.operation_id, operation);
          set({ loading: false });
          
          // Start polling for status
          get().pollOperationStatus(response.operation_id);
          
          return response.operation_id;
          
        } catch (error) {
          set({ 
            error: error.message,
            loading: false 
          });
          throw error;
        }
      },
      
      // Compare multiple strategies
      compareStrategies: async (strategies: string[]) => {
        const { previewResult } = get();
        
        if (!previewResult?.content) {
          throw new Error('No content to compare');
        }
        
        set({ comparingStrategies: true, comparisonResults: null });
        
        try {
          const results = await chunkingApi.compareStrategies(
            previewResult.content,
            strategies,
            previewResult.config
          );
          
          set({ 
            comparisonResults: results,
            comparingStrategies: false 
          });
          
        } catch (error) {
          set({ 
            error: error.message,
            comparingStrategies: false 
          });
        }
      },
      
      // Cancel preview request
      cancelPreview: () => {
        chunkingApi.cancelRequest('preview_current');
        set({ loading: false });
      },
      
      // Poll operation status
      pollOperationStatus: async (operationId: string) => {
        const interval = setInterval(async () => {
          try {
            const status = await chunkingApi.getOperationStatus(operationId);
            
            get().operations.set(operationId, status);
            set({ operations: new Map(get().operations) });
            
            if (status.status === 'completed' || status.status === 'failed') {
              clearInterval(interval);
              
              if (status.status === 'completed') {
                get().showSuccessNotification('Chunking completed');
              } else {
                get().showErrorNotification(`Chunking failed: ${status.error}`);
              }
            }
          } catch (error) {
            clearInterval(interval);
            console.error('Failed to poll operation status:', error);
          }
        }, 2000); // Poll every 2 seconds
      },
      
      // Utility: Retry with backoff
      retryWithBackoff: async (
        fn: () => Promise<any>,
        maxRetries: number = 3
      ) => {
        let lastError;
        
        for (let i = 0; i < maxRetries; i++) {
          try {
            const result = await fn();
            return result;
          } catch (error) {
            lastError = error;
            const delay = Math.pow(2, i) * 1000; // Exponential backoff
            await new Promise(resolve => setTimeout(resolve, delay));
          }
        }
        
        throw lastError;
      },
      
      // Reset store
      reset: () => {
        chunkingApi.cancelAllRequests();
        set({
          strategies: [],
          selectedStrategy: null,
          previewResult: null,
          operations: new Map(),
          loading: false,
          error: null
        });
      }
    }),
    {
      name: 'chunking-store'
    }
  )
);
```

### 3. Update Components to Handle Real Data

```typescript
// Example component update
const ChunkingPreviewPanel: React.FC = () => {
  const {
    previewResult,
    loading,
    error,
    previewChunking,
    cancelPreview
  } = useChunkingStore();
  
  const handlePreview = async () => {
    try {
      await previewChunking({
        strategy: selectedStrategy,
        content: documentContent,
        config: customConfig
      });
    } catch (error) {
      // Error already handled in store
      console.error('Preview failed:', error);
    }
  };
  
  // Handle loading state
  if (loading) {
    return (
      <div>
        <LoadingSpinner />
        <Button onClick={cancelPreview}>Cancel</Button>
      </div>
    );
  }
  
  // Handle error state
  if (error) {
    return (
      <ErrorAlert 
        message={error}
        onRetry={handlePreview}
      />
    );
  }
  
  // Display real results
  return (
    <div>
      {previewResult && (
        <ChunkList chunks={previewResult.chunks} />
      )}
    </div>
  );
};
```

## Acceptance Criteria

1. **API Integration**
   - [ ] All mock API calls removed
   - [ ] Real API client implemented
   - [ ] All endpoints properly typed
   - [ ] Authentication handled correctly

2. **Error Handling**
   - [ ] Network errors handled gracefully
   - [ ] Auth errors trigger token refresh
   - [ ] User-friendly error messages displayed
   - [ ] Retry logic implemented

3. **Progress Tracking**
   - [ ] Upload progress shown for large documents
   - [ ] Operation status polling works
   - [ ] Cancel functionality works

4. **Type Safety**
   - [ ] All API responses properly typed
   - [ ] No `any` types in API client
   - [ ] Request/response types match backend

## Testing Requirements

1. **Unit Tests**
   ```typescript
   describe('ChunkingAPIClient', () => {
     it('should fetch strategies', async () => {
       const strategies = await chunkingApi.getStrategies();
       expect(strategies).toHaveLength(6);
       expect(strategies[0]).toHaveProperty('id');
     });
     
     it('should handle auth errors', async () => {
       // Mock 401 response
       mockAxios.onGet().reply(401);
       
       await expect(chunkingApi.getStrategies())
         .rejects.toThrow('Authentication required');
     });
     
     it('should cancel requests', () => {
       const promise = chunkingApi.previewChunking({...});
       chunkingApi.cancelRequest('preview_current');
       
       await expect(promise).rejects.toThrow('Request cancelled');
     });
   });
   ```

2. **Integration Tests**
   - Test with real backend API
   - Test error scenarios
   - Test progress tracking
   - Test cancel functionality

## Rollback Plan

1. Keep backup of mock implementation
2. Feature flag to switch between mock/real
3. Monitor for API errors
4. Quick revert if critical issues

## Success Metrics

- Zero mock API calls remaining
- All chunking operations functional
- API response time < 2s for preview
- Error rate < 1%
- Successful auth token refresh

## Notes for LLM Agent

- Remove ALL mock implementations - no fallbacks
- Ensure proper TypeScript types throughout
- Handle all error cases explicitly
- Test with real backend before marking complete
- Verify auth flow works correctly
- Ensure progress tracking updates UI smoothly