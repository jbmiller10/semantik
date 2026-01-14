/**
 * TypeScript types for System Settings API.
 * Matches backend Pydantic schemas from packages/webui/api/v2/system_settings.py
 */

/**
 * System setting value with metadata.
 * Tracks when and by whom a setting was last modified.
 */
export interface SystemSettingValue {
  /** The setting value (can be any JSON-serializable type) */
  value: unknown;
  /** ISO 8601 timestamp of last update, null if never modified */
  updated_at: string | null;
  /** User ID who last updated this setting, null if never modified */
  updated_by: number | null;
}

/**
 * Response from GET /api/v2/system-settings.
 * Contains all settings with their metadata.
 */
export interface SystemSettingsResponse {
  settings: Record<string, SystemSettingValue>;
}

/**
 * Request body for PATCH /api/v2/system-settings.
 * Only include settings you want to change.
 */
export interface SystemSettingsUpdateRequest {
  settings: Record<string, unknown>;
}

/**
 * Response from PATCH /api/v2/system-settings.
 * Returns the updated settings with metadata.
 */
export interface SystemSettingsUpdateResponse {
  /** Keys that were updated */
  updated: string[];
  /** All settings with their metadata */
  settings: Record<string, SystemSettingValue>;
}

/**
 * Response from GET /api/v2/system-settings/effective.
 * Contains resolved effective values (DB value or env fallback).
 */
export interface EffectiveSettingsResponse {
  settings: Record<string, unknown>;
}

/**
 * Response from GET /api/v2/system-settings/defaults.
 * Contains system default values for all settings.
 */
export interface DefaultSettingsResponse {
  defaults: Record<string, unknown>;
}

// ============================================================================
// Typed Setting Groups for Frontend Components
// ============================================================================

/**
 * Resource limits settings - controls per-user quotas.
 */
export interface ResourceLimitsSettings {
  /** Maximum collections a user can create (default: 10) */
  max_collections_per_user: number;
  /** Maximum storage in GB per user (default: 50) */
  max_storage_gb_per_user: number;
  /** Maximum document size in MB (default: 100) */
  max_document_size_mb: number;
  /** Maximum artifact size in MB (default: 50) */
  max_artifact_size_mb: number;
}

/**
 * Performance tuning settings.
 */
export interface PerformanceSettings {
  /** Cache TTL in seconds (default: 300) */
  cache_ttl_seconds: number;
  /** Model unload timeout in seconds (default: 300) */
  model_unload_timeout_seconds: number;
}

/**
 * GPU and memory management settings.
 */
export interface GpuMemorySettings {
  /** GPU memory reserve percent (default: 0.10) */
  gpu_memory_reserve_percent: number;
  /** GPU memory max percent (default: 0.90) */
  gpu_memory_max_percent: number;
  /** CPU memory reserve percent (default: 0.20) */
  cpu_memory_reserve_percent: number;
  /** CPU memory max percent (default: 0.50) */
  cpu_memory_max_percent: number;
  /** Enable CPU offload (default: true) */
  enable_cpu_offload: boolean;
  /** Eviction idle threshold in seconds (default: 120) */
  eviction_idle_threshold_seconds: number;
}

/**
 * Search and reranking tuning settings.
 */
export interface SearchRerankSettings {
  /** Rerank candidate multiplier (default: 5) */
  rerank_candidate_multiplier: number;
  /** Minimum candidates for reranking (default: 20) */
  rerank_min_candidates: number;
  /** Maximum candidates for reranking (default: 200) */
  rerank_max_candidates: number;
  /** Hybrid weight for reranking (default: 0.3) */
  rerank_hybrid_weight: number;
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Extract typed resource limits from effective settings.
 */
export function extractResourceLimits(
  settings: Record<string, unknown>
): ResourceLimitsSettings {
  return {
    max_collections_per_user: (settings.max_collections_per_user as number) ?? 10,
    max_storage_gb_per_user: (settings.max_storage_gb_per_user as number) ?? 50,
    max_document_size_mb: (settings.max_document_size_mb as number) ?? 100,
    max_artifact_size_mb: (settings.max_artifact_size_mb as number) ?? 50,
  };
}

/**
 * Extract typed performance settings from effective settings.
 */
export function extractPerformanceSettings(
  settings: Record<string, unknown>
): PerformanceSettings {
  return {
    cache_ttl_seconds: (settings.cache_ttl_seconds as number) ?? 300,
    model_unload_timeout_seconds: (settings.model_unload_timeout_seconds as number) ?? 300,
  };
}

/**
 * Extract typed GPU/memory settings from effective settings.
 */
export function extractGpuMemorySettings(
  settings: Record<string, unknown>
): GpuMemorySettings {
  return {
    gpu_memory_reserve_percent: (settings.gpu_memory_reserve_percent as number) ?? 0.1,
    gpu_memory_max_percent: (settings.gpu_memory_max_percent as number) ?? 0.9,
    cpu_memory_reserve_percent: (settings.cpu_memory_reserve_percent as number) ?? 0.2,
    cpu_memory_max_percent: (settings.cpu_memory_max_percent as number) ?? 0.5,
    enable_cpu_offload: (settings.enable_cpu_offload as boolean) ?? true,
    eviction_idle_threshold_seconds: (settings.eviction_idle_threshold_seconds as number) ?? 120,
  };
}

/**
 * Extract typed search/rerank settings from effective settings.
 */
export function extractSearchRerankSettings(
  settings: Record<string, unknown>
): SearchRerankSettings {
  return {
    rerank_candidate_multiplier: (settings.rerank_candidate_multiplier as number) ?? 5,
    rerank_min_candidates: (settings.rerank_min_candidates as number) ?? 20,
    rerank_max_candidates: (settings.rerank_max_candidates as number) ?? 200,
    rerank_hybrid_weight: (settings.rerank_hybrid_weight as number) ?? 0.3,
  };
}
