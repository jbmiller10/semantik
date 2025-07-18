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
  chunk_size: number;
  chunk_overlap: number;
  is_public: boolean;
  status: CollectionStatus;
  status_message?: string;
  metadata?: Record<string, any>;
  document_count: number;
  vector_count: number;
  total_size_bytes?: number;
  created_at: string;              // ISO 8601 string
  updated_at: string;              // ISO 8601 string
  
  // Frontend-specific fields
  isProcessing?: boolean;          // Optimistic UI state
  activeOperation?: Operation;     // Current operation
}

export interface Operation {
  id: string;                      // UUID
  collection_id: string;
  type: OperationType;
  status: OperationStatus;
  config: Record<string, any>;     // Operation-specific configuration
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
  chunk_size?: number;
  chunk_overlap?: number;
  is_public?: boolean;
  metadata?: Record<string, any>;
}

export interface UpdateCollectionRequest {
  name?: string;
  description?: string;
  is_public?: boolean;
  metadata?: Record<string, any>;
}

export interface AddSourceRequest {
  source_path: string;
  config?: {
    chunk_size?: number;
    chunk_overlap?: number;
    metadata?: Record<string, any>;
  };
}

export interface RemoveSourceRequest {
  source_path: string;
}

export interface ReindexRequest {
  embedding_model?: string;
  chunk_size?: number;
  chunk_overlap?: number;
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
  metadata?: Record<string, any>;
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