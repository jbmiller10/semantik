import type { AxiosRequestConfig } from 'axios';
import apiClient from './client';
import { ApiErrorHandler } from '../../../utils/api-error-handler';
import type {
  Collection,
  Operation,
  CreateCollectionRequest,
  UpdateCollectionRequest,
  AddSourceRequest,
  RemoveSourceRequest,
  ReindexRequest,
  CollectionListResponse,
  OperationListResponse,
  CollectionSyncRun,
  SyncRunListResponse,
  PaginationParams,
} from '../../../types/collection';
import type {
  DocumentListResponse,
  SearchRequest,
  SearchResponse,
  SourceListResponse,
} from './types';
import { projectionsV2Api } from './projections';

/**
 * V2 Collections API client
 * Implements the new collection-centric API endpoints
 */
export const collectionsV2Api = {
  // Collections CRUD
  list: (params?: PaginationParams) => 
    apiClient.get<CollectionListResponse>('/api/v2/collections', { params }),
    
  get: (uuid: string) => 
    apiClient.get<Collection>(`/api/v2/collections/${uuid}`),
    
  create: (data: CreateCollectionRequest) => 
    apiClient.post<Collection>('/api/v2/collections', data),
    
  update: (uuid: string, data: UpdateCollectionRequest) => 
    apiClient.put<Collection>(`/api/v2/collections/${uuid}`, data),
    
  delete: (uuid: string) => 
    apiClient.delete<void>(`/api/v2/collections/${uuid}`),
  
  // Collection operations
  addSource: (uuid: string, data: AddSourceRequest) => 
    apiClient.post<Operation>(`/api/v2/collections/${uuid}/sources`, data),
    
  removeSource: (uuid: string, data: RemoveSourceRequest) =>
    apiClient.delete<Operation>(`/api/v2/collections/${uuid}/sources`, { data }),

  listSources: (uuid: string, params?: { offset?: number; limit?: number }) =>
    apiClient.get<SourceListResponse>(`/api/v2/collections/${uuid}/sources`, { params }),

  reindex: (uuid: string, data?: ReindexRequest) => 
    apiClient.post<Operation>(`/api/v2/collections/${uuid}/reindex`, data || {}),
  
  // Collection queries
  listOperations: (uuid: string, params?: PaginationParams) =>
    apiClient.get<Operation[]>(`/api/v2/collections/${uuid}/operations`, { params }),

  listDocuments: (uuid: string, params?: PaginationParams) =>
    apiClient.get<DocumentListResponse>(`/api/v2/collections/${uuid}/documents`, { params }),

  // Sync control (collection-level)
  runSync: (uuid: string) =>
    apiClient.post<CollectionSyncRun>(`/api/v2/collections/${uuid}/sync/run`),

  pauseSync: (uuid: string) =>
    apiClient.post<Collection>(`/api/v2/collections/${uuid}/sync/pause`),

  resumeSync: (uuid: string) =>
    apiClient.post<Collection>(`/api/v2/collections/${uuid}/sync/resume`),

  listSyncRuns: (uuid: string, params?: { offset?: number; limit?: number }) =>
    apiClient.get<SyncRunListResponse>(`/api/v2/collections/${uuid}/sync/runs`, { params }),
};

/**
 * V2 Operations API client
 * For managing operations directly
 */
export const operationsV2Api = {
  get: (uuid: string) => 
    apiClient.get<Operation>(`/api/v2/operations/${uuid}`),
    
  cancel: (uuid: string) => 
    apiClient.delete<void>(`/api/v2/operations/${uuid}`),
    
  list: (
    params?: PaginationParams & {
      status?: string;
      operation_type?: string;
      per_page?: number;
      offset?: number;
    }
  ) => {
    const { limit, offset, page, per_page, status, operation_type, sort_by, sort_order } = params || {};
    const perPage = per_page ?? limit;
    const resolvedPage = page ?? (offset !== undefined && limit ? Math.floor(offset / limit) + 1 : undefined);

    const queryParams: Record<string, string | number> = {};
    if (status) queryParams.status = status;
    if (operation_type) queryParams.operation_type = operation_type;
    if (resolvedPage) queryParams.page = resolvedPage;
    if (perPage) queryParams.per_page = perPage;
    if (sort_by) queryParams.sort_by = sort_by;
    if (sort_order) queryParams.sort_order = sort_order;

    return apiClient.get<OperationListResponse>('/api/v2/operations', { params: queryParams });
  },
};

/**
 * V2 Search API client
 * Minimal implementation for existing usage
 */
export const searchV2Api = {
  search: (data: SearchRequest, config?: AxiosRequestConfig) =>
    apiClient.post<SearchResponse>('/api/v2/search', data, config),
};

/**
 * Helper function to handle API errors.
 * @deprecated Use ApiErrorHandler.getMessage() or ApiErrorHandler.handle() for typed errors
 */
export function handleApiError(error: unknown): string {
  return ApiErrorHandler.getMessage(error);
}

// Export a unified v2Api object for convenience
export const v2Api = {
  collections: collectionsV2Api,
  operations: operationsV2Api,
  projections: projectionsV2Api,
  search: searchV2Api,
};

/**
 * Wait for a collection to become ready (initial operation completes).
 * Polls the collection status until it transitions from 'pending' to 'ready' or an error state.
 *
 * @param collectionId - The collection UUID to wait for
 * @param options - Configuration options
 * @returns The collection once it's ready
 * @throws Error if the collection enters an error state or times out
 */
export async function waitForCollectionReady(
  collectionId: string,
  options: {
    /** Maximum time to wait in milliseconds (default: 30000) */
    timeout?: number;
    /** Polling interval in milliseconds (default: 500) */
    pollInterval?: number;
    /** Callback for progress updates */
    onProgress?: (status: string) => void;
  } = {}
): Promise<Collection> {
  const { timeout = 30000, pollInterval = 500, onProgress } = options;
  const startTime = Date.now();

  while (Date.now() - startTime < timeout) {
    const response = await collectionsV2Api.get(collectionId);
    const collection = response.data;

    onProgress?.(collection.status);

    if (collection.status === 'ready') {
      return collection;
    }

    if (collection.status === 'error') {
      throw new Error(collection.status_message || 'Collection initialization failed');
    }

    // Still pending or processing, wait and retry
    await new Promise(resolve => setTimeout(resolve, pollInterval));
  }

  throw new Error('Timed out waiting for collection to become ready');
}
