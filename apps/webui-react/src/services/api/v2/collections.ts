import type { AxiosRequestConfig } from 'axios';
import apiClient from './client';
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

// Helper function to handle API errors
export function handleApiError(error: unknown): string {
  if (error instanceof Error && 'response' in error) {
    const axiosError = error as { response?: { data?: { detail?: string } } };
    if (axiosError.response?.data?.detail) {
      return axiosError.response.data.detail;
    }
  }
  if (error instanceof Error && error.message) {
    return error.message;
  }
  return 'An unexpected error occurred';
}

// Export a unified v2Api object for convenience
export const v2Api = {
  collections: collectionsV2Api,
  operations: operationsV2Api,
  projections: projectionsV2Api,
  search: searchV2Api,
};
