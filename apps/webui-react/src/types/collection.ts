export type CollectionStatus = 
  | 'pending'
  | 'ready'
  | 'processing'
  | 'error'
  | 'degraded';

export type OperationType = 
  | 'index'
  | 'append'
  | 'reindex'
  | 'remove_source'
  | 'delete';

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
}

export interface UpdateCollectionRequest {
  name?: string;
  description?: string;
  is_public?: boolean;
  metadata?: Record<string, unknown>;
}

export interface AddSourceRequest {
  /** @deprecated Use source_type + source_config instead */
  source_path?: string;
  /** Source type (e.g., "directory", "web", "slack"). Defaults to "directory" on backend. */
  source_type?: string;
  /** Source-specific configuration (e.g., { path: "/data/docs" } for directory) */
  source_config?: Record<string, unknown>;
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