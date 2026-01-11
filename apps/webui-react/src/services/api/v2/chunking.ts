import axios, { AxiosError } from 'axios';
import type { AxiosProgressEvent, CancelTokenSource } from 'axios';
import apiClient from './client';
// Re-export for callers migrating to the new error handling pattern
export { ApiErrorHandler, ErrorCategories, type ApiErrorCategory, ApiError } from '../../../utils/api-error-handler';
import type {
  ChunkingStrategyType,
  ChunkingConfiguration,
  ChunkingPreviewRequest,
  ChunkingPreviewResponse,
  ChunkingAnalytics,
  ChunkingPreset,
  ChunkingComparisonResult,
} from '../../../types/chunking';

/**
 * Configuration for retry logic
 */
interface RetryConfig {
  maxRetries: number;
  baseDelay: number;
  maxDelay: number;
  retryableStatuses: number[];
}

const DEFAULT_RETRY_CONFIG: RetryConfig = {
  maxRetries: 3,
  baseDelay: 1000, // 1 second
  maxDelay: 30000, // 30 seconds
  retryableStatuses: [429, 500, 502, 503, 504],
};

/**
 * Progress callback for long-running operations
 */
export type ProgressCallback = (progress: {
  loaded: number;
  total: number;
  percentage: number;
  estimatedTimeRemaining?: number;
}) => void;

/**
 * Tracks active requests for cancellation
 */
class RequestManager {
  private activeRequests = new Map<string, CancelTokenSource>();

  register(id: string, source: CancelTokenSource): void {
    this.activeRequests.set(id, source);
  }

  unregister(id: string): void {
    this.activeRequests.delete(id);
  }

  cancel(id: string, reason?: string): boolean {
    const source = this.activeRequests.get(id);
    if (source) {
      source.cancel(reason || 'Request cancelled by user');
      this.unregister(id);
      return true;
    }
    return false;
  }

  cancelAll(reason?: string): void {
    this.activeRequests.forEach((source) => {
      source.cancel(reason || 'All requests cancelled');
    });
    this.activeRequests.clear();
  }

  isActive(id: string): boolean {
    return this.activeRequests.has(id);
  }
}

const requestManager = new RequestManager();

/**
 * Exponential backoff delay calculation
 */
function calculateDelay(attempt: number, config: RetryConfig): number {
  const delay = Math.min(
    config.baseDelay * Math.pow(2, attempt),
    config.maxDelay
  );
  // Add jitter to prevent thundering herd
  return delay + Math.random() * 1000;
}

/**
 * Execute request with retry logic
 */
async function executeWithRetry<T>(
  requestFn: () => Promise<T>,
  config: RetryConfig = DEFAULT_RETRY_CONFIG,
  attempt: number = 0
): Promise<T> {
  try {
    return await requestFn();
  } catch (error) {
    const axiosError = error as AxiosError;
    
    // Check if we should retry
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

/**
 * Chunking API client with full feature support
 */
export const chunkingApi = {
  /**
   * Preview chunking for a document
   * Supports progress tracking and cancellation
   */
  preview: async (
    request: ChunkingPreviewRequest,
    options?: {
      requestId?: string;
      onProgress?: ProgressCallback;
      retryConfig?: Partial<RetryConfig>;
    }
  ): Promise<ChunkingPreviewResponse> => {
    const requestId = options?.requestId || `preview-${Date.now()}`;
    const cancelTokenSource = axios.CancelToken.source();
    
    requestManager.register(requestId, cancelTokenSource);

    try {
      const response = await executeWithRetry(
        () => apiClient.post<ChunkingPreviewResponse>(
          '/api/v2/chunking/preview',
          request,
          {
            cancelToken: cancelTokenSource.token,
            onUploadProgress: options?.onProgress
              ? (progressEvent: AxiosProgressEvent) => {
                  const percentage = progressEvent.total
                    ? Math.round((progressEvent.loaded * 100) / progressEvent.total)
                    : 0;
                  
                  options.onProgress!({
                    loaded: progressEvent.loaded,
                    total: progressEvent.total || 0,
                    percentage,
                    estimatedTimeRemaining: progressEvent.estimated,
                  });
                }
              : undefined,
          }
        ),
        { ...DEFAULT_RETRY_CONFIG, ...options?.retryConfig }
      );

      return response.data;
    } finally {
      requestManager.unregister(requestId);
    }
  },

  /**
   * Compare multiple chunking strategies
   * Returns comparison results for each strategy
   */
  compare: async (
    request: {
      documentId?: string;
      content?: string;
      strategies: Array<{
        strategy: ChunkingStrategyType;
        configuration: ChunkingConfiguration['parameters'];
      }>;
    },
    options?: {
      requestId?: string;
      onProgress?: ProgressCallback;
      retryConfig?: Partial<RetryConfig>;
    }
  ): Promise<ChunkingComparisonResult[]> => {
    const requestId = options?.requestId || `compare-${Date.now()}`;
    const cancelTokenSource = axios.CancelToken.source();
    
    requestManager.register(requestId, cancelTokenSource);

    try {
      const response = await executeWithRetry(
        () => apiClient.post<ChunkingComparisonResult[]>(
          '/api/v2/chunking/compare',
          request,
          {
            cancelToken: cancelTokenSource.token,
            onDownloadProgress: options?.onProgress
              ? (progressEvent: AxiosProgressEvent) => {
                  const percentage = progressEvent.total
                    ? Math.round((progressEvent.loaded * 100) / progressEvent.total)
                    : 0;
                  
                  options.onProgress!({
                    loaded: progressEvent.loaded,
                    total: progressEvent.total || 0,
                    percentage,
                    estimatedTimeRemaining: progressEvent.estimated,
                  });
                }
              : undefined,
          }
        ),
        { ...DEFAULT_RETRY_CONFIG, ...options?.retryConfig }
      );

      return response.data;
    } finally {
      requestManager.unregister(requestId);
    }
  },

  /**
   * Get chunking analytics
   */
  getAnalytics: async (
    params?: {
      startDate?: string;
      endDate?: string;
      collectionId?: string;
    },
    options?: {
      requestId?: string;
      retryConfig?: Partial<RetryConfig>;
    }
  ): Promise<ChunkingAnalytics> => {
    const requestId = options?.requestId || `analytics-${Date.now()}`;
    const cancelTokenSource = axios.CancelToken.source();
    
    requestManager.register(requestId, cancelTokenSource);

    try {
      const response = await executeWithRetry(
        () => apiClient.get<ChunkingAnalytics>(
          '/api/v2/chunking/analytics',
          {
            params,
            cancelToken: cancelTokenSource.token,
          }
        ),
        { ...DEFAULT_RETRY_CONFIG, ...options?.retryConfig }
      );

      return response.data;
    } finally {
      requestManager.unregister(requestId);
    }
  },

  /**
   * Get all presets (system and custom)
   */
  getPresets: async (
    options?: {
      requestId?: string;
      retryConfig?: Partial<RetryConfig>;
    }
  ): Promise<ChunkingPreset[]> => {
    const requestId = options?.requestId || `presets-${Date.now()}`;
    const cancelTokenSource = axios.CancelToken.source();
    
    requestManager.register(requestId, cancelTokenSource);

    try {
      const response = await executeWithRetry(
        () => apiClient.get<ChunkingPreset[]>(
          '/api/v2/chunking/presets',
          {
            cancelToken: cancelTokenSource.token,
          }
        ),
        { ...DEFAULT_RETRY_CONFIG, ...options?.retryConfig }
      );

      return response.data;
    } finally {
      requestManager.unregister(requestId);
    }
  },

  /**
   * Save a custom preset
   */
  savePreset: async (
    preset: Omit<ChunkingPreset, 'id'>,
    options?: {
      requestId?: string;
      retryConfig?: Partial<RetryConfig>;
    }
  ): Promise<ChunkingPreset> => {
    const requestId = options?.requestId || `save-preset-${Date.now()}`;
    const cancelTokenSource = axios.CancelToken.source();
    
    requestManager.register(requestId, cancelTokenSource);

    try {
      const response = await executeWithRetry(
        () => apiClient.post<ChunkingPreset>(
          '/api/v2/chunking/presets',
          preset,
          {
            cancelToken: cancelTokenSource.token,
          }
        ),
        { ...DEFAULT_RETRY_CONFIG, ...options?.retryConfig }
      );

      return response.data;
    } finally {
      requestManager.unregister(requestId);
    }
  },

  /**
   * Delete a custom preset
   */
  deletePreset: async (
    presetId: string,
    options?: {
      requestId?: string;
      retryConfig?: Partial<RetryConfig>;
    }
  ): Promise<void> => {
    const requestId = options?.requestId || `delete-preset-${Date.now()}`;
    const cancelTokenSource = axios.CancelToken.source();
    
    requestManager.register(requestId, cancelTokenSource);

    try {
      await executeWithRetry(
        () => apiClient.delete<void>(
          `/api/v2/chunking/presets/${presetId}`,
          {
            cancelToken: cancelTokenSource.token,
          }
        ),
        { ...DEFAULT_RETRY_CONFIG, ...options?.retryConfig }
      );
    } finally {
      requestManager.unregister(requestId);
    }
  },

  /**
   * Process a document with specific chunking strategy
   * This is for actually applying chunking to a document in a collection
   */
  process: async (
    request: {
      collectionId: string;
      documentId: string;
      strategy: ChunkingStrategyType;
      configuration: ChunkingConfiguration['parameters'];
    },
    options?: {
      requestId?: string;
      onProgress?: ProgressCallback;
      retryConfig?: Partial<RetryConfig>;
    }
  ): Promise<{ operationId: string }> => {
    const requestId = options?.requestId || `process-${Date.now()}`;
    const cancelTokenSource = axios.CancelToken.source();
    
    requestManager.register(requestId, cancelTokenSource);

    try {
      const response = await executeWithRetry(
        () => apiClient.post<{ operationId: string }>(
          `/api/v2/collections/${request.collectionId}/documents/${request.documentId}/chunk`,
          {
            strategy: request.strategy,
            configuration: request.configuration,
          },
          {
            cancelToken: cancelTokenSource.token,
            onUploadProgress: options?.onProgress
              ? (progressEvent: AxiosProgressEvent) => {
                  const percentage = progressEvent.total
                    ? Math.round((progressEvent.loaded * 100) / progressEvent.total)
                    : 0;
                  
                  options.onProgress!({
                    loaded: progressEvent.loaded,
                    total: progressEvent.total || 0,
                    percentage,
                    estimatedTimeRemaining: progressEvent.estimated,
                  });
                }
              : undefined,
          }
        ),
        { ...DEFAULT_RETRY_CONFIG, ...options?.retryConfig }
      );

      return response.data;
    } finally {
      requestManager.unregister(requestId);
    }
  },

  /**
   * Get recommended strategy for a file type
   */
  getRecommendation: async (
    fileType: string,
    options?: {
      requestId?: string;
      retryConfig?: Partial<RetryConfig>;
    }
  ): Promise<{
    strategy: ChunkingStrategyType;
    configuration: ChunkingConfiguration;
    confidence: number;
  }> => {
    const requestId = options?.requestId || `recommend-${Date.now()}`;
    const cancelTokenSource = axios.CancelToken.source();
    
    requestManager.register(requestId, cancelTokenSource);

    try {
      const response = await executeWithRetry(
        () => apiClient.get<{
          strategy: ChunkingStrategyType;
          configuration: ChunkingConfiguration;
          confidence: number;
        }>(
          '/api/v2/chunking/recommend',
          {
            params: { fileType },
            cancelToken: cancelTokenSource.token,
          }
        ),
        { ...DEFAULT_RETRY_CONFIG, ...options?.retryConfig }
      );

      return response.data;
    } finally {
      requestManager.unregister(requestId);
    }
  },

  /**
   * Cancel a specific request
   */
  cancelRequest: (requestId: string, reason?: string): boolean => {
    return requestManager.cancel(requestId, reason);
  },

  /**
   * Cancel all active requests
   */
  cancelAllRequests: (reason?: string): void => {
    requestManager.cancelAll(reason);
  },

  /**
   * Check if a request is active
   */
  isRequestActive: (requestId: string): boolean => {
    return requestManager.isActive(requestId);
  },
};

/**
 * Helper function to handle chunking API errors.
 * Provides chunking-specific error messages while delegating to ApiErrorHandler.
 * @deprecated Use ApiErrorHandler.handle() for typed errors with category
 */
export function handleChunkingError(error: unknown): string {
  // Handle axios cancellation first
  if (axios.isCancel(error)) {
    return 'Request was cancelled';
  }

  // Handle axios errors with response
  if (error instanceof Error && 'response' in error) {
    const axiosError = error as AxiosError<{ detail?: string; message?: string }>;
    const status = axiosError.response?.status;

    // First try to get the detail message from the response
    const detailMessage =
      axiosError.response?.data?.detail || axiosError.response?.data?.message;

    if (detailMessage) {
      return detailMessage;
    }

    // Provide chunking-specific fallback messages for certain status codes
    switch (status) {
      case 400:
        return 'Invalid chunking configuration';
      case 404:
        return 'Document or collection not found';
      case 413:
        return 'Document is too large for preview';
      case 429:
        return 'Too many requests. Please try again later';
      case 500:
        return 'Server error while processing chunking request';
      default:
        return `Server error: ${status || 'Unknown'}`;
    }
  }

  // Handle regular Error objects
  if (error instanceof Error && error.message) {
    return error.message;
  }

  // Fallback for unknown errors
  return 'An unexpected error occurred while processing chunking request';
}

// Export for use in stores and components
export default chunkingApi;