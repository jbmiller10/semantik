/**
 * TypeScript types for User Preferences API.
 * Matches backend Pydantic schemas from packages/webui/api/v2/user_preferences_schemas.py
 */

/** Search mode - determines vector search strategy */
export type SearchMode = 'dense' | 'sparse' | 'hybrid';

/** Chunking strategy for document processing */
export type ChunkingStrategy = 'character' | 'recursive' | 'markdown' | 'semantic';

/** Model precision level */
export type Quantization = 'float32' | 'float16' | 'int8';

/** Sparse indexing type */
export type SparseType = 'bm25' | 'splade';

/**
 * Search preferences - controls default search behavior.
 */
export interface SearchPreferences {
  /** Default number of results to return (5-50) */
  top_k: number;
  /** Default search mode */
  mode: SearchMode;
  /** Whether to use reranker by default */
  use_reranker: boolean;
  /** RRF constant for hybrid fusion (1-100) */
  rrf_k: number;
  /** Similarity threshold for filtering results (0.0-1.0, null = no threshold) */
  similarity_threshold: number | null;
}

/**
 * Collection defaults - applied when creating new collections.
 */
export interface CollectionDefaults {
  /** Default embedding model (null = use system default) */
  embedding_model: string | null;
  /** Vector quantization level */
  quantization: Quantization;
  /** Chunking strategy */
  chunking_strategy: ChunkingStrategy;
  /** Chunk size in characters (256-4096) */
  chunk_size: number;
  /** Chunk overlap in characters (0-512) */
  chunk_overlap: number;
  /** Enable sparse indexing by default */
  enable_sparse: boolean;
  /** Sparse indexing type */
  sparse_type: SparseType;
  /** Enable hybrid search by default (requires enable_sparse) */
  enable_hybrid: boolean;
}

/**
 * Interface preferences - UI behavior settings.
 */
export interface InterfacePreferences {
  /** Data polling interval in milliseconds (10000-60000) */
  data_refresh_interval_ms: number;
  /** Maximum points for UMAP/PCA visualizations (10000-500000) */
  visualization_sample_limit: number;
  /** Enable UI animations */
  animation_enabled: boolean;
}

/**
 * Response from GET /api/v2/preferences.
 * Contains all user preferences with timestamps.
 */
export interface UserPreferencesResponse {
  search: SearchPreferences;
  collection_defaults: CollectionDefaults;
  interface: InterfacePreferences;
  /** ISO 8601 timestamp */
  created_at: string;
  /** ISO 8601 timestamp */
  updated_at: string;
}

/**
 * Request body for PUT /api/v2/preferences.
 * Supports partial updates - only include fields you want to change.
 */
export interface UserPreferencesUpdate {
  search?: Partial<SearchPreferences>;
  collection_defaults?: Partial<CollectionDefaults>;
  interface?: Partial<InterfacePreferences>;
}
