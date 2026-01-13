/**
 * LLM settings API client.
 * Implements endpoints for LLM provider configuration, model listing,
 * API key testing, and usage statistics.
 */
import apiClient from './client';
import type {
  LLMSettingsUpdate,
  LLMSettingsResponse,
  AvailableModelsResponse,
  LLMTestRequest,
  LLMTestResponse,
  TokenUsageResponse,
  LLMProviderType,
} from '../../../types/llm';

export const llmApi = {
  /**
   * Get current user's LLM configuration.
   * Returns 404 if not configured.
   */
  getSettings: () => apiClient.get<LLMSettingsResponse>('/api/v2/llm/settings'),

  /**
   * Create or update LLM settings.
   * API keys are write-only (never returned in responses).
   * @param data Settings to update (partial updates supported)
   */
  updateSettings: (data: LLMSettingsUpdate) =>
    apiClient.put<LLMSettingsResponse>('/api/v2/llm/settings', data),

  /**
   * List available models from curated registry.
   * No API key required (static list).
   */
  getModels: () => apiClient.get<AvailableModelsResponse>('/api/v2/llm/models'),

  /**
   * Test API key validity without saving.
   * Rate limited: 5 requests/minute per user.
   * @param data Provider and API key to test
   */
  testKey: (data: LLMTestRequest) =>
    apiClient.post<LLMTestResponse>('/api/v2/llm/test', data),

  /**
   * Get token usage statistics for the current user.
   * @param days Number of days to include (1-365, default 30)
   */
  getUsage: (days: number = 30) =>
    apiClient.get<TokenUsageResponse>('/api/v2/llm/usage', { params: { days } }),

  /**
   * Refresh models from provider API.
   * Fetches available models directly from the provider (Anthropic/OpenAI).
   * Rate limited: 5 requests/minute per user.
   * @param provider Provider to fetch models from
   * @param apiKey API key for the provider
   */
  refreshModels: (provider: LLMProviderType, apiKey: string) =>
    apiClient.get<AvailableModelsResponse>('/api/v2/llm/models/refresh', {
      params: { provider, api_key: apiKey },
    }),
};
