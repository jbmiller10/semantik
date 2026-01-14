/**
 * TypeScript types for LLM settings API.
 * Matches backend Pydantic schemas from packages/webui/api/v2/llm_schemas.py
 */

/** Provider type union - matches backend Literal["anthropic", "openai", "local"] */
export type LLMProviderType = 'anthropic' | 'openai' | 'local';

/** Quantization options for local models */
export type LocalQuantization = 'int4' | 'int8';

/**
 * Request body for updating LLM settings.
 * API keys are write-only (never returned in responses).
 */
export interface LLMSettingsUpdate {
  /** High-quality tier provider (NULL = use defaults from model registry) */
  high_quality_provider?: LLMProviderType | null;
  /** High-quality tier model ID */
  high_quality_model?: string | null;
  /** Low-quality tier provider (NULL = use defaults from model registry) */
  low_quality_provider?: LLMProviderType | null;
  /** Low-quality tier model ID */
  low_quality_model?: string | null;
  /** Anthropic API key (write-only, shared across tiers) */
  anthropic_api_key?: string | null;
  /** OpenAI API key (write-only, shared across tiers) */
  openai_api_key?: string | null;
  /** Local model quantization for high quality tier */
  local_high_quantization?: LocalQuantization | null;
  /** Local model quantization for low quality tier */
  local_low_quantization?: LocalQuantization | null;
  /** Default temperature (0.0 - 2.0) */
  default_temperature?: number | null;
  /** Default max tokens (1 - 200000) */
  default_max_tokens?: number | null;
}

/**
 * Response for GET/PUT /api/v2/llm/settings.
 * Contains has_key booleans instead of actual API keys.
 */
export interface LLMSettingsResponse {
  high_quality_provider: string | null;
  high_quality_model: string | null;
  low_quality_provider: string | null;
  low_quality_model: string | null;
  /** True if Anthropic API key is configured */
  anthropic_has_key: boolean;
  /** True if OpenAI API key is configured */
  openai_has_key: boolean;
  /** Local model quantization for high quality tier */
  local_high_quantization: string | null;
  /** Local model quantization for low quality tier */
  local_low_quantization: string | null;
  default_temperature: number | null;
  default_max_tokens: number | null;
  /** ISO 8601 timestamp */
  created_at: string;
  /** ISO 8601 timestamp */
  updated_at: string;
}

/** Available model info from curated registry or provider API */
export interface AvailableModel {
  /** Model ID (e.g., "claude-opus-4-5-20251101") */
  id: string;
  /** Short name (e.g., "Opus 4.5") */
  name: string;
  /** Display name with provider prefix (e.g., "Claude - Opus 4.5") */
  display_name: string;
  /** Provider identifier ("anthropic" or "openai") */
  provider: string;
  /** Recommended tier ("high" or "low") */
  tier_recommendation: string;
  /** Context window size in tokens */
  context_window: number;
  /** Human-readable description */
  description: string;
  /** True for curated models, false for API-fetched models */
  is_curated: boolean;
  /** Memory requirements per quantization (local models only) */
  memory_mb?: Record<string, number> | null;
}

/** Response for GET /api/v2/llm/models */
export interface AvailableModelsResponse {
  models: AvailableModel[];
}

/** Request body for POST /api/v2/llm/test */
export interface LLMTestRequest {
  provider: LLMProviderType;
  api_key: string;
}

/** Request body for POST /api/v2/llm/test (local provider variant) */
export interface LLMTestRequestLocal {
  provider: 'local';
}

/** Response for POST /api/v2/llm/test */
export interface LLMTestResponse {
  success: boolean;
  message: string;
  /** Model used for testing (null if test failed before model selection) */
  model_tested: string | null;
}

/** Token usage breakdown by category */
export interface TokenUsageBreakdown {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
}

/** Response for GET /api/v2/llm/usage */
export interface TokenUsageResponse {
  total_input_tokens: number;
  total_output_tokens: number;
  total_tokens: number;
  /** Usage breakdown by feature (e.g., "hyde", "summary") */
  by_feature: Record<string, TokenUsageBreakdown>;
  /** Usage breakdown by provider (e.g., "anthropic", "openai") */
  by_provider: Record<string, TokenUsageBreakdown>;
  /** Number of LLM usage events in the period */
  event_count: number;
  /** Number of days included in the statistics */
  period_days: number;
}
