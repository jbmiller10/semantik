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

import type {
  SearchMode,
  SparseIndexStatus,
  EnableSparseIndexRequest,
  SparseReindexJobResponse,
  SparseReindexProgress,
} from '../../../types/sparse-index';

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

// Re-export sparse index types
export type {
  SearchMode,
  SparseIndexStatus,
  EnableSparseIndexRequest,
  SparseReindexJobResponse,
  SparseReindexProgress,
};

// Document status type
export type DocumentStatus = 'pending' | 'processing' | 'completed' | 'failed' | 'deleted';

// Error category type for retry decisions
export type ErrorCategory = 'transient' | 'permanent' | 'unknown';

// Additional v2 API specific types
export interface DocumentResponse {
  id: string;
  collection_id: string;
  file_name: string;
  file_path: string;
  file_size: number;
  mime_type: string | null;
  content_hash: string;
  status: DocumentStatus;
  error_message: string | null;
  chunk_count: number;
  // Retry tracking fields
  retry_count: number;
  last_retry_at: string | null;
  error_category: ErrorCategory | null;
  metadata?: Record<string, unknown>;
  created_at: string;
  updated_at: string;
  // Legacy field - not always present in backend response
  source_path?: string;
}

export interface DocumentListResponse {
  documents: DocumentResponse[];
  total: number;
  page: number;
  per_page: number;
}

// Failed document count response
export interface FailedDocumentCountResponse {
  transient: number;
  permanent: number;
  unknown: number;
  total: number;
}

// Retry documents response
export interface RetryDocumentsResponse {
  reset_count: number;
  operation_id: string | null;
  message: string;
}

export interface SearchRequest {
  query: string;
  collection_uuids?: string[];
  k?: number;
  score_threshold?: number;
  /** Embedding mode instruction (semantic/question/code) */
  search_type?: 'semantic' | 'question' | 'code' | 'hybrid';
  use_reranker?: boolean;
  rerank_model?: string | null;

  // New sparse/hybrid search parameters
  /** Search mode: dense (vector), sparse (BM25/SPLADE), or hybrid (RRF fusion) */
  search_mode?: SearchMode;
  /** RRF constant k for hybrid search (default: 60) */
  rrf_k?: number;

  // Legacy hybrid parameters (deprecated - for backward compatibility only)
  /** @deprecated Use search_mode='hybrid' instead */
  hybrid_alpha?: number;
  /** @deprecated Use search_mode='hybrid' instead */
  hybrid_mode?: 'filter' | 'weighted';
  /** @deprecated Use search_mode='hybrid' instead */
  keyword_mode?: 'any' | 'all';

  /** Enable HyDE query expansion (None = use user preference default) */
  use_hyde?: boolean;
}

export interface SearchResult {
  id: string;
  collection_id: string;
  collection_name: string;
  document_id: string;
  chunk_id: string;
  chunk_index?: number;
  total_chunks?: number;
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

/** HyDE generation metadata returned when HyDE query expansion was used */
export interface HyDEInfo {
  /** The generated hypothetical document (if HyDE was used) */
  expanded_query: string | null;
  /** Time taken for HyDE generation in milliseconds */
  generation_time_ms: number | null;
  /** Total tokens consumed for HyDE generation */
  tokens_used: number | null;
  /** LLM provider used for HyDE (anthropic, openai, local) */
  provider: string | null;
  /** LLM model used for HyDE generation */
  model: string | null;
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
  // Sparse/hybrid search metrics
  /** Actual search mode used (may differ from requested if sparse unavailable) */
  search_mode_used?: SearchMode;
  /** Sparse search time in milliseconds */
  sparse_search_time_ms?: number;
  /** RRF fusion time in milliseconds */
  rrf_fusion_time_ms?: number;
  /** Warnings about fallbacks (e.g., sparse not enabled, fell back to dense) */
  warnings?: string[];
  // HyDE metadata
  /** Whether HyDE query expansion was used for this search */
  hyde_used: boolean;
  /** HyDE generation details (if hyde_used is true) */
  hyde_info?: HyDEInfo;
  // Failure information
  partial_failure: boolean;
  failed_collections?: Array<{
    collection_id: string;
    collection_name: string;
    error?: string;
    error_message?: string;
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

// Source types for collection sources API
export interface SourceResponse {
  id: number;
  collection_id: string;
  source_type: string;
  source_path: string;
  source_config: Record<string, unknown>;
  document_count: number;
  size_bytes: number;
  // Per-source sync telemetry
  last_run_started_at: string | null;
  last_run_completed_at: string | null;
  last_run_status: string | null;
  last_error: string | null;
  last_indexed_at: string | null;
  // Timestamps
  created_at: string;
  updated_at: string;
  // Secret indicators
  has_password: boolean;
  has_token: boolean;
  has_ssh_key: boolean;
  has_ssh_passphrase: boolean;
}

export interface SourceListResponse {
  items: SourceResponse[];
  total: number;
  offset: number;
  limit: number;
}
