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
  metadata?: Record<string, unknown>;
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
  metadata?: Record<string, unknown>;
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

// Directory scan types
export interface DirectoryScanRequest {
  path: string;
  scan_id: string;
  recursive?: boolean;
  include_patterns?: string[];
  exclude_patterns?: string[];
}

export interface DirectoryScanFile {
  file_path: string;
  file_name: string;
  file_size: number;
  mime_type: string | null;
  content_hash: string;
  modified_at: string;
}

export interface DirectoryScanResponse {
  scan_id: string;
  path: string;
  files: DirectoryScanFile[];
  total_files: number;
  total_size: number;
  warnings: string[];
}

export interface DirectoryScanProgress {
  type: 'started' | 'counting' | 'progress' | 'completed' | 'error' | 'warning';
  scan_id: string;
  data: {
    files_scanned?: number;
    total_files?: number;
    current_path?: string;
    percentage?: number;
    message?: string;
    path?: string;
    recursive?: boolean;
    warnings?: string[];
    total_size?: number;
  };
}