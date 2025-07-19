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
  documents: DocumentResponse[];
  total: number;
  page: number;
  per_page: number;
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
  collection_name: string;
  document_id: string;
  chunk_id: string;
  chunk_index: number;
  score: number;
  original_score: number;
  reranked_score?: number;
  text: string;
  file_name: string;
  file_path: string;
  embedding_model: string;
  metadata?: Record<string, any>;
  highlights?: string[];
}

export interface SearchResponse {
  query: string;
  results: SearchResult[];
  total_results: number;
  collections_searched: Array<{
    id: string;
    name: string;
    embedding_model: string;
  }>;
  search_type: string;
  reranking_used: boolean;
  reranker_model?: string;
  // Timing metrics
  embedding_time_ms?: number;
  search_time_ms: number;
  reranking_time_ms?: number;
  total_time_ms: number;
  // Failure information
  partial_failure: boolean;
  failed_collections?: Array<{
    collection_id: string;
    collection_name: string;
    error_message: string;
  }>;
  api_version: string;
}

export interface ErrorResponse {
  detail: string;
  status_code: number;
  type?: string;
}