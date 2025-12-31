export type CollectionStatus =
  | 'pending'
  | 'ready'
  | 'processing'
  | 'error'
  | 'degraded';

export type SyncMode = 'one_time' | 'continuous';

export type SyncRunStatus = 'running' | 'success' | 'failed' | 'partial';

export type OperationType =
  | 'index'
  | 'append'
  | 'reindex'
  | 'remove_source'
  | 'delete'
  | 'projection_build';

export type OperationStatus = 
  | 'pending'
  | 'processing'
  | 'completed'
  | 'failed'
  | 'cancelled';

export interface Collection {
  id: string;                      // UUID
  name: string;
  description?: string;
  owner_id: number;
  vector_store_name: string;       // Qdrant collection name
  embedding_model: string;
  quantization: string;          // float32, float16, or int8
  chunk_size: number;              // Deprecated: for backward compatibility
  chunk_overlap: number;           // Deprecated: for backward compatibility
  chunking_strategy?: string;      // New: chunking strategy type
  chunking_config?: Record<string, number | boolean | string>; // New: strategy-specific configuration
  is_public: boolean;
  status: CollectionStatus;
  status_message?: string;
  metadata?: Record<string, unknown>;
  document_count: number;
  vector_count: number;
  total_size_bytes?: number;
  created_at: string;              // ISO 8601 string
  updated_at: string;              // ISO 8601 string

  // Sync policy (collection-level)
  sync_mode: SyncMode;
  sync_interval_minutes?: number;
  sync_paused_at?: string;         // ISO 8601 string, null if not paused
  sync_next_run_at?: string;       // ISO 8601 string

  // Sync run tracking
  sync_last_run_started_at?: string;   // ISO 8601 string
  sync_last_run_completed_at?: string; // ISO 8601 string
  sync_last_run_status?: SyncRunStatus;
  sync_last_error?: string;

  // Frontend-specific fields
  isProcessing?: boolean;          // Optimistic UI state
  activeOperation?: Operation;     // Current operation
  initial_operation_id?: string;   // Initial INDEX operation ID when collection is created
}

export interface Operation {
  id: string;                      // UUID
  collection_id: string;
  type: OperationType;
  status: OperationStatus;
  config: Record<string, unknown>;     // Operation-specific configuration
  error_message?: string;
  created_at: string;              // ISO 8601 string
  started_at?: string;             // ISO 8601 string
  completed_at?: string;           // ISO 8601 string
  
  // Frontend-specific fields
  progress?: number;               // 0-100 for progress tracking
  eta?: number;                    // Estimated time remaining in seconds
}

// Request types for API calls
export interface CreateCollectionRequest {
  name: string;
  description?: string;
  embedding_model?: string;
  quantization?: string;         // float32, float16, or int8
  chunk_size?: number;           // Deprecated: use chunking_config instead
  chunk_overlap?: number;         // Deprecated: use chunking_config instead
  chunking_strategy?: string;     // New: chunking strategy type
  chunking_config?: Record<string, number | boolean | string>; // New: strategy-specific configuration
  is_public?: boolean;
  metadata?: Record<string, unknown>;
  // Sync configuration
  sync_mode?: SyncMode;           // 'one_time' (default) or 'continuous'
  sync_interval_minutes?: number; // Required for continuous mode, minimum 15
}

export interface UpdateCollectionRequest {
  name?: string;
  description?: string;
  is_public?: boolean;
  metadata?: Record<string, unknown>;
  // Sync configuration updates
  sync_mode?: SyncMode;
  sync_interval_minutes?: number;
}

// Sync run tracking
export interface CollectionSyncRun {
  id: number;
  collection_id: string;
  triggered_by: 'scheduler' | 'manual';
  started_at: string;              // ISO 8601 string
  completed_at?: string;           // ISO 8601 string
  status: SyncRunStatus;
  expected_sources: number;
  completed_sources: number;
  failed_sources: number;
  partial_sources: number;
  error_summary?: string;
}

export interface SyncRunListResponse {
  items: CollectionSyncRun[];
  total: number;
  offset: number;
  limit: number;
}

export interface AddSourceRequest {
  /** @deprecated Use source_type + source_config instead */
  source_path?: string;
  /** Source type (e.g., "directory", "git", "imap"). Defaults to "directory" on backend. */
  source_type?: string;
  /** Source-specific configuration (e.g., { path: "/data/docs" } for directory) */
  source_config?: Record<string, unknown>;
  /** Secrets for authentication (e.g., tokens, passwords, SSH keys) - encrypted at rest */
  secrets?: Record<string, string>;
  config?: {
    chunk_size?: number;         // Deprecated: use chunking_config instead
    chunk_overlap?: number;      // Deprecated: use chunking_config instead
    chunking_strategy?: string;  // New: chunking strategy type
    chunking_config?: Record<string, number | boolean | string>; // New: strategy-specific configuration
    metadata?: Record<string, unknown>;
  };
}

export interface RemoveSourceRequest {
  source_path: string;
}

export interface ReindexRequest {
  embedding_model?: string;
  quantization?: string;         // float32, float16, or int8
  chunk_size?: number;           // Deprecated: use chunking_config instead
  chunk_overlap?: number;         // Deprecated: use chunking_config instead
  chunking_strategy?: string;     // New: chunking strategy type
  chunking_config?: Record<string, number | boolean | string>; // New: strategy-specific configuration
}

// Response types
export interface CollectionListResponse {
  collections: Collection[];
  total: number;
  page: number;
  per_page: number;
}

export interface OperationListResponse {
  operations: Operation[];
  total: number;
  page: number;
  per_page: number;
}

export interface PaginationParams {
  page?: number;
  limit?: number;
  sort_by?: string;
  sort_order?: 'asc' | 'desc';
}

// WebSocket message types for real-time updates
export interface OperationProgressMessage {
  operation_id: string;
  status: OperationStatus;
  progress?: number;
  message?: string;
  error?: string;
  metadata?: Record<string, unknown>;
}

export interface CollectionStatusMessage {
  collection_id: string;
  status: CollectionStatus;
  message?: string;
  stats?: {
    document_count: number;
    vector_count: number;
    total_size_bytes: number;
  };
}