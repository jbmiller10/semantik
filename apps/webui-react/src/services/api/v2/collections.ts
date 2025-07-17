import api from '../../api';
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
  PaginationParams,
  DocumentListResponse,
  SearchRequest,
  SearchResponse,
} from './types';

/**
 * V2 Collections API client
 * Implements the new collection-centric API endpoints
 */
export const collectionsV2Api = {
  // Collections CRUD
  list: (params?: PaginationParams) => 
    api.get<CollectionListResponse>('/api/v2/collections', { params }),
    
  get: (uuid: string) => 
    api.get<Collection>(`/api/v2/collections/${uuid}`),
    
  create: (data: CreateCollectionRequest) => 
    api.post<Collection>('/api/v2/collections', data),
    
  update: (uuid: string, data: UpdateCollectionRequest) => 
    api.put<Collection>(`/api/v2/collections/${uuid}`, data),
    
  delete: (uuid: string) => 
    api.delete<void>(`/api/v2/collections/${uuid}`),
  
  // Collection operations
  addSource: (uuid: string, data: AddSourceRequest) => 
    api.post<Operation>(`/api/v2/collections/${uuid}/sources`, data),
    
  removeSource: (uuid: string, data: RemoveSourceRequest) => 
    api.delete<Operation>(`/api/v2/collections/${uuid}/sources`, { data }),
    
  reindex: (uuid: string, data?: ReindexRequest) => 
    api.post<Operation>(`/api/v2/collections/${uuid}/reindex`, data || {}),
  
  // Collection queries
  listOperations: (uuid: string, params?: PaginationParams) => 
    api.get<OperationListResponse>(`/api/v2/collections/${uuid}/operations`, { params }),
    
  listDocuments: (uuid: string, params?: PaginationParams) => 
    api.get<DocumentListResponse>(`/api/v2/collections/${uuid}/documents`, { params }),
};

/**
 * V2 Operations API client
 * For managing operations directly
 */
export const operationsV2Api = {
  get: (uuid: string) => 
    api.get<Operation>(`/api/v2/operations/${uuid}`),
    
  cancel: (uuid: string) => 
    api.delete<void>(`/api/v2/operations/${uuid}`),
    
  list: (params?: PaginationParams & { collection_id?: string; status?: string }) => 
    api.get<OperationListResponse>('/api/v2/operations', { params }),
};

/**
 * V2 Search API client
 * Supports multi-collection search
 */
export const searchV2Api = {
  search: (data: SearchRequest) => 
    api.post<SearchResponse>('/api/v2/search', data),
    
  multiSearch: (data: SearchRequest) => 
    api.post<SearchResponse>('/api/v2/search/multi', data),
};

// Helper function to handle API errors
export function handleApiError(error: any): string {
  if (error.response?.data?.detail) {
    return error.response.data.detail;
  }
  if (error.message) {
    return error.message;
  }
  return 'An unexpected error occurred';
}

// Export a unified v2Api object for convenience
export const v2Api = {
  collections: collectionsV2Api,
  operations: operationsV2Api,
  search: searchV2Api,
};