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
}

export interface TaskResponse {
  task_id: string | null;
  model_id: string;
  operation: string;
  status: TaskStatus;
  warnings: string[];
}

export interface TaskProgressResponse {
  task_id: string;
  model_id: string;
  operation: string;
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
  warnings: string[];
  can_delete: boolean;
  requires_confirmation: boolean;
}

export interface ModelManagerConflictResponse {
  conflict_type: ConflictType;
  detail: string;
  model_id: string;
  active_operation: string | null;
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
