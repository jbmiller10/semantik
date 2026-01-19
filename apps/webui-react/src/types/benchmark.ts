/**
 * TypeScript types for benchmark management API.
 * Matches backend schemas from packages/webui/api/v2/benchmark_schemas.py
 */

// =============================================================================
// Status Types
// =============================================================================

export type BenchmarkStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
export type BenchmarkRunStatus = 'pending' | 'indexing' | 'evaluating' | 'completed' | 'failed';
export type MappingStatus = 'pending' | 'resolved' | 'partial';
export type SearchMode = 'dense' | 'sparse' | 'hybrid';

// =============================================================================
// Dataset Types
// =============================================================================

export interface BenchmarkDataset {
  id: string;
  name: string;
  description: string | null;
  owner_id: number;
  query_count: number;
  schema_version: string;
  created_at: string;
  updated_at: string | null;
}

export interface DatasetListResponse {
  datasets: BenchmarkDataset[];
  total: number;
  page: number;
  per_page: number;
}

export interface DatasetUploadRequest {
  name: string;
  description?: string;
}

// =============================================================================
// Mapping Types
// =============================================================================

export interface DatasetMapping {
  id: number;
  dataset_id: string;
  collection_id: string;
  mapping_status: MappingStatus;
  mapped_count: number;
  total_count: number;
  created_at: string;
  resolved_at: string | null;
}

export interface MappingCreateRequest {
  collection_id: string;
}

export interface MappingResolveResponse {
  id: number;
  operation_uuid: string | null;
  mapping_status: MappingStatus;
  mapped_count: number;
  total_count: number;
  unresolved: Array<Record<string, unknown>>;
}

// =============================================================================
// Config Matrix Types
// =============================================================================

export interface ConfigMatrixItem {
  search_modes: SearchMode[];
  use_reranker: boolean[];
  top_k_values: number[];
  rrf_k_values: number[];
  score_thresholds: Array<number | null>;
  primary_k?: number;
  k_values_for_metrics?: number[];
}

// =============================================================================
// Benchmark Types
// =============================================================================

export interface Benchmark {
  id: string;
  name: string;
  description: string | null;
  owner_id: number;
  mapping_id: number;
  status: BenchmarkStatus;
  total_runs: number;
  completed_runs: number;
  failed_runs: number;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  operation_uuid: string | null;
}

export interface BenchmarkListResponse {
  benchmarks: Benchmark[];
  total: number;
  page: number;
  per_page: number;
}

export interface BenchmarkCreateRequest {
  name: string;
  description?: string;
  mapping_id: number;
  config_matrix: ConfigMatrixItem;
  top_k?: number;
  metrics_to_compute?: string[];
}

export interface BenchmarkStartResponse {
  id: string;
  status: BenchmarkStatus;
  operation_uuid: string;
  message: string;
}

// =============================================================================
// Results Types
// =============================================================================

export interface RunTiming {
  indexing_ms: number | null;
  evaluation_ms: number | null;
  total_ms: number | null;
}

export interface BenchmarkRunMetrics {
  mrr: number | null;
  ap?: number | null;
  precision?: Record<string, number>;
  recall?: Record<string, number>;
  ndcg?: Record<string, number>;
}

export interface BenchmarkRun {
  id: string;
  run_order: number;
  config_hash: string;
  config: Record<string, unknown>;
  status: BenchmarkRunStatus;
  error_message: string | null;
  metrics: BenchmarkRunMetrics;
  metrics_flat: Record<string, number>;
  timing: RunTiming;
}

export interface BenchmarkResultsResponse {
  benchmark_id: string;
  primary_k: number;
  k_values_for_metrics: number[];
  runs: BenchmarkRun[];
  summary: Record<string, unknown>;
  total_runs: number;
}

export interface QueryResult {
  query_id: number;
  query_key: string;
  query_text: string;
  retrieved_doc_ids: string[];
  precision_at_k: number | null;
  recall_at_k: number | null;
  reciprocal_rank: number | null;
  ndcg_at_k: number | null;
  search_time_ms: number | null;
  rerank_time_ms: number | null;
}

export interface RunQueryResultsResponse {
  run_id: string;
  results: QueryResult[];
  total: number;
  page: number;
  per_page: number;
}

// =============================================================================
// Pagination Types
// =============================================================================

export interface BenchmarkPaginationParams {
  page?: number;
  per_page?: number;
}
