/**
 * Sparse Index Types
 *
 * Type definitions for sparse indexing (BM25/SPLADE) management
 * in the Semantik UI.
 */

/**
 * Search mode for retrieval - controls whether to use dense vectors,
 * sparse vectors (BM25/SPLADE), or hybrid (RRF fusion) search.
 */
export type SearchMode = 'dense' | 'sparse' | 'hybrid';

/**
 * Available sparse indexer plugin types
 */
export type SparseIndexerPlugin = 'bm25-local' | 'splade-local';

/**
 * BM25 configuration parameters
 */
export interface BM25Config {
  /** Term frequency saturation parameter (default: 1.5, range: 0.5-3.0) */
  k1?: number;
  /** Document length normalization parameter (default: 0.75, range: 0.0-1.0) */
  b?: number;
}

/**
 * SPLADE configuration parameters
 */
export interface SPLADEConfig {
  /** Maximum number of non-zero dimensions (optional) */
  max_sparse_length?: number;
}

/**
 * Union type for sparse indexer model configurations
 */
export type SparseModelConfig = BM25Config | SPLADEConfig | Record<string, unknown>;

/**
 * Sparse index status response from the backend
 */
export interface SparseIndexStatus {
  /** Whether sparse indexing is enabled for this collection */
  enabled: boolean;
  /** The sparse indexer plugin ID (e.g., 'bm25-local', 'splade-local') */
  plugin_id?: string;
  /** Name of the sparse collection in Qdrant */
  sparse_collection_name?: string;
  /** Plugin-specific configuration data */
  model_config_data?: SparseModelConfig;
  /** Number of documents indexed in the sparse collection */
  document_count?: number;
  /** When the sparse index was created */
  created_at?: string;
  /** When the sparse index was last updated */
  last_indexed_at?: string;
}

/**
 * Request to enable sparse indexing on a collection
 */
export interface EnableSparseIndexRequest {
  /** Sparse indexer plugin to use (e.g., 'bm25-local', 'splade-local') */
  plugin_id: SparseIndexerPlugin | string;
  /** Plugin-specific configuration */
  model_config_data?: SparseModelConfig;
  /** Whether to reindex existing documents immediately */
  reindex_existing?: boolean;
}

/**
 * Response when triggering a sparse reindex job
 */
export interface SparseReindexJobResponse {
  /** Unique job identifier */
  job_id: string;
  /** Current job status */
  status: SparseReindexStatus;
  /** Collection UUID being reindexed */
  collection_uuid: string;
  /** Plugin ID used for reindexing */
  plugin_id: string;
}

/**
 * Sparse reindex job status values
 */
export type SparseReindexStatus =
  | 'pending'
  | 'processing'
  | 'completed'
  | 'failed';

/**
 * Progress information for a sparse reindex job
 */
export interface SparseReindexProgress {
  /** Unique job identifier */
  job_id: string;
  /** Current job status */
  status: SparseReindexStatus;
  /** Progress percentage (0.0 to 1.0) */
  progress?: number;
  /** Current step description */
  current_step?: string;
  /** Total number of steps */
  total_steps?: number;
  /** Number of documents processed */
  documents_processed?: number;
  /** Total documents to process */
  total_documents?: number;
  /** Error message if status is 'failed' */
  error?: string;
}

/**
 * Default BM25 configuration values
 */
export const DEFAULT_BM25_CONFIG: Required<BM25Config> = {
  k1: 1.5,
  b: 0.75,
};

/**
 * BM25 parameter validation ranges
 */
export const BM25_PARAM_RANGES = {
  k1: { min: 0.5, max: 3.0, step: 0.1 },
  b: { min: 0.0, max: 1.0, step: 0.05 },
} as const;

/**
 * RRF (Reciprocal Rank Fusion) parameter defaults for hybrid search
 */
export const RRF_DEFAULTS = {
  /** Default RRF constant k */
  k: 60,
  /** Minimum RRF k value */
  min: 1,
  /** Maximum RRF k value */
  max: 1000,
} as const;

/**
 * Plugin display information for UI
 */
export const SPARSE_PLUGIN_INFO: Record<
  SparseIndexerPlugin,
  { name: string; description: string; requiresGPU: boolean }
> = {
  'bm25-local': {
    name: 'BM25 (Statistical)',
    description: 'Traditional keyword matching with TF-IDF scoring. Fast, CPU-based.',
    requiresGPU: false,
  },
  'splade-local': {
    name: 'SPLADE (Neural)',
    description: 'Neural sparse representations with semantic expansion. Requires GPU.',
    requiresGPU: true,
  },
};
