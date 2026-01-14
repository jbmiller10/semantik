/**
 * TypeScript types for System API.
 * Used for system info, health checks, and GPU status.
 */

/**
 * System information from GET /api/v2/system/info.
 * Contains version, environment, limits, and rate limits.
 */
export interface SystemInfo {
  /** Application version */
  version: string;
  /** Environment (development/production) */
  environment: string;
  /** Python runtime version */
  python_version: string;
  /** Resource limits */
  limits: {
    /** Maximum collections per user */
    max_collections_per_user: number;
    /** Maximum storage in GB per user */
    max_storage_gb_per_user: number;
  };
  /** Rate limits (formatted strings) */
  rate_limits: {
    chunking_preview: string;
    plugin_install: string;
    llm_test: string;
  };
}

/**
 * Individual service health status.
 */
export interface ServiceHealth {
  /** Service health status */
  status: 'healthy' | 'unhealthy';
  /** Optional status message */
  message?: string;
}

/**
 * Health status for all backend services.
 * From GET /api/v2/system/health.
 * Always returns 200 with per-service status.
 */
export interface SystemHealth {
  postgres: ServiceHealth;
  redis: ServiceHealth;
  qdrant: ServiceHealth;
  vecpipe: ServiceHealth;
}

/**
 * System status including GPU availability.
 * From GET /api/v2/system/status.
 */
export interface SystemStatus {
  /** Whether GPU is available for compute */
  gpu_available: boolean;
  /** Whether reranking is available */
  reranking_available: boolean;
  /** List of available reranking models */
  available_reranking_models: string[];
  /** Number of CUDA devices */
  cuda_device_count: number;
  /** CUDA device name (null if no GPU) */
  cuda_device_name: string | null;
}
