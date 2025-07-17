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
} from '../../../types/collection';

// Re-export for convenience
export type {
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
};

// Additional v2 API specific types
export interface DocumentResponse {
  id: string;
  collection_id: string;
  source_path: string;
  file_path: string;
  chunk_count: number;
  metadata?: Record<string, any>;
  created_at: string;
  updated_at: string;
}

export interface DocumentListResponse {
  items: DocumentResponse[];
  total: number;
  page: number;
  limit: number;
}

export interface SearchRequest {
  query: string;
  collection_ids?: string[];
  top_k?: number;
  score_threshold?: number;
  search_type?: 'vector' | 'hybrid';
  hybrid_config?: {
    alpha?: number;
    mode?: 'rerank' | 'filter';
    keyword_mode?: 'any' | 'all';
  };
  rerank_config?: {
    model?: string;
    quantization?: string;
    enabled?: boolean;
  };
}

export interface SearchResult {
  id: string;
  collection_id: string;
  document_id: string;
  chunk_index: number;
  score: number;
  text: string;
  metadata?: Record<string, any>;
  highlights?: string[];
}

export interface SearchResponse {
  results: SearchResult[];
  total: number;
  query: string;
  took_ms: number;
}

export interface ErrorResponse {
  detail: string;
  status_code: number;
  type?: string;
}