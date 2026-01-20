/**
 * TypeScript types for Model Manager feature.
 * These types match the backend Pydantic schemas in model_manager_schemas.py
 */

// Enums

export type ModelType = 'embedding' | 'llm' | 'reranker' | 'splade';

export type TaskStatus =
  | 'pending'
  | 'running'
  | 'completed'
  | 'failed'
  | 'already_installed'
  | 'not_installed';

export type OperationType = 'download' | 'delete';

export type ConflictType =
  | 'cross_op_exclusion'
  | 'in_use_block'
  | 'requires_confirmation';

// Interfaces

export interface CacheSizeInfo {
  total_cache_size_mb: number;
  managed_cache_size_mb: number;
  unmanaged_cache_size_mb: number;
  unmanaged_repo_count: number;
}

export interface EmbeddingModelDetails {
  dimension: number | null;
  max_sequence_length: number | null;
  pooling_method: string | null;
  is_asymmetric: boolean;
  query_prefix: string;
  document_prefix: string;
  default_query_instruction: string;
}

export interface LLMModelDetails {
  context_window: number | null;
}

export interface CuratedModelResponse {
  id: string;
  name: string;
  description: string;
  model_type: ModelType;
  memory_mb: Record<string, number>;
  is_installed: boolean;
  size_on_disk_mb: number | null;
  used_by_collections: string[];
  active_download_task_id: string | null;
  active_delete_task_id: string | null;
  embedding_details: EmbeddingModelDetails | null;
  llm_details: LLMModelDetails | null;
}

export interface ModelListResponse {
  models: CuratedModelResponse[];
  cache_size: CacheSizeInfo | null;
  hf_cache_scan_error: string | null;
}

export interface TaskResponse {
  task_id: string | null;
  model_id: string;
  operation: OperationType;
  status: TaskStatus;
  warnings: string[];
}

export interface TaskProgressResponse {
  task_id: string;
  model_id: string;
  operation: OperationType;
  status: TaskStatus;
  bytes_downloaded: number;
  bytes_total: number;
  error: string | null;
  updated_at: number;
}

export interface ModelUsageResponse {
  model_id: string;
  is_installed: boolean;
  size_on_disk_mb: number | null;
  estimated_freed_size_mb: number | null;
  blocked_by_collections: string[];
  user_preferences_count: number;
  llm_config_count: number;
  is_default_embedding_model: boolean;
  loaded_in_vecpipe: boolean;
  loaded_vecpipe_model_types: string[];
  hf_cache_scan_error: string | null;
  vecpipe_query_error: string | null;
  warnings: string[];
  can_delete: boolean;
  requires_confirmation: boolean;
}

export interface ModelManagerConflictResponse {
  conflict_type: ConflictType;
  detail: string;
  model_id: string;
  active_operation: OperationType | null;
  active_task_id: string | null;
  blocked_by_collections: string[];
  requires_confirmation: boolean;
  warnings: string[];
}

// Helper constants

export const MODEL_TYPE_LABELS: Record<ModelType, string> = {
  embedding: 'Embedding',
  llm: 'Local LLM',
  reranker: 'Reranker',
  splade: 'SPLADE',
};

export const MODEL_TYPE_ORDER: ModelType[] = ['embedding', 'llm', 'reranker', 'splade'];

// Terminal status constants

/**
 * Task statuses that indicate the task has completed (success or failure).
 * Used to stop polling for progress updates.
 */
export const TERMINAL_TASK_STATUSES: TaskStatus[] = [
  'completed',
  'failed',
  'already_installed',
  'not_installed',
];

/**
 * Check if a task status indicates the task is terminal (completed or failed).
 */
export function isTerminalStatus(status: TaskStatus | undefined | null): boolean {
  return !!status && TERMINAL_TASK_STATUSES.includes(status);
}

// Helper functions

/**
 * Groups models by their model_type.
 * Returns a record where keys are model types and values are arrays of models.
 */
export function groupModelsByType(
  models: CuratedModelResponse[]
): Record<ModelType, CuratedModelResponse[]> {
  const grouped: Record<ModelType, CuratedModelResponse[]> = {
    embedding: [],
    llm: [],
    reranker: [],
    splade: [],
  };

  for (const model of models) {
    if (model.model_type in grouped) {
      grouped[model.model_type].push(model);
    }
  }

  return grouped;
}
