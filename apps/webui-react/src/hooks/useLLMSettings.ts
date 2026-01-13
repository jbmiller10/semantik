/**
 * React Query hooks for LLM settings management.
 * Provides hooks for fetching settings, updating settings, listing models,
 * testing API keys, and getting usage statistics.
 */
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { llmApi } from '../services/api/v2/llm';
import { useUIStore } from '../stores/uiStore';
import { ApiErrorHandler } from '../utils/api-error-handler';
import type {
  LLMSettingsUpdate,
  LLMSettingsResponse,
  LLMTestRequest,
  LLMProviderType,
} from '../types/llm';

/**
 * Query key factory for LLM queries.
 * Enables hierarchical cache invalidation.
 */
export const llmKeys = {
  all: ['llm'] as const,
  settings: () => [...llmKeys.all, 'settings'] as const,
  models: () => [...llmKeys.all, 'models'] as const,
  usage: (days: number) => [...llmKeys.all, 'usage', days] as const,
};

/**
 * Hook to fetch current user's LLM settings.
 * Returns undefined data if not configured (404).
 */
export function useLLMSettings() {
  return useQuery({
    queryKey: llmKeys.settings(),
    queryFn: async () => {
      const response = await llmApi.getSettings();
      return response.data;
    },
    retry: (failureCount, error) => {
      // Don't retry 404s - user just hasn't configured yet
      const apiError = ApiErrorHandler.handle(error);
      if (apiError.statusCode === 404) {
        return false;
      }
      return failureCount < 3;
    },
  });
}

/**
 * Hook to update or create LLM settings.
 * Shows toast notifications on success/error.
 */
export function useUpdateLLMSettings() {
  const queryClient = useQueryClient();
  const { addToast } = useUIStore();

  return useMutation<LLMSettingsResponse, Error, LLMSettingsUpdate>({
    mutationFn: async (data: LLMSettingsUpdate) => {
      const response = await llmApi.updateSettings(data);
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: llmKeys.settings() });
      addToast({
        type: 'success',
        message: 'LLM settings saved successfully',
      });
    },
    onError: (error) => {
      const apiError = ApiErrorHandler.handle(error);
      addToast({
        type: 'error',
        message: apiError.message || 'Failed to save LLM settings',
      });
    },
  });
}

/**
 * Hook to fetch available LLM models from curated registry.
 * Static list - uses infinite staleTime.
 */
export function useLLMModels() {
  return useQuery({
    queryKey: llmKeys.models(),
    queryFn: async () => {
      const response = await llmApi.getModels();
      return response.data;
    },
    staleTime: Infinity, // Static curated list, only changes on deployment
    gcTime: 30 * 60 * 1000, // Keep in cache for 30 minutes
  });
}

/**
 * Hook to test API key validity without saving.
 * Shows toast with test result.
 * Rate limited: 5 requests/minute.
 */
export function useTestLLMKey() {
  const { addToast } = useUIStore();

  return useMutation({
    mutationFn: async (data: LLMTestRequest) => {
      const response = await llmApi.testKey(data);
      return response.data;
    },
    onSuccess: (data) => {
      if (data.success) {
        addToast({
          type: 'success',
          message: data.message,
        });
      } else {
        addToast({
          type: 'error',
          message: data.message,
        });
      }
    },
    onError: (error) => {
      const apiError = ApiErrorHandler.handle(error);
      // Handle rate limit specifically
      if (apiError.statusCode === 429) {
        addToast({
          type: 'error',
          message: 'Rate limit exceeded. Please wait before testing again.',
        });
      } else {
        addToast({
          type: 'error',
          message: apiError.message || 'Failed to test API key',
        });
      }
    },
  });
}

/**
 * Hook to fetch token usage statistics.
 * Only fetches if user has at least one provider configured.
 * @param days Number of days to include (default 30)
 * @param settings Current LLM settings (used to determine if query should run)
 */
export function useLLMUsage(days: number = 30, settings?: LLMSettingsResponse) {
  return useQuery({
    queryKey: llmKeys.usage(days),
    queryFn: async () => {
      const response = await llmApi.getUsage(days);
      return response.data;
    },
    // Only fetch if user has configured at least one provider
    enabled: !!settings?.anthropic_has_key || !!settings?.openai_has_key,
    staleTime: 60 * 1000, // Consider stale after 1 minute
  });
}

/**
 * Hook to refresh models from provider API.
 * Fetches available models directly from the provider.
 * Rate limited: 5 requests/minute.
 */
export function useRefreshLLMModels() {
  const queryClient = useQueryClient();
  const { addToast } = useUIStore();

  return useMutation({
    mutationFn: async ({
      provider,
      apiKey,
    }: {
      provider: LLMProviderType;
      apiKey: string;
    }) => {
      const response = await llmApi.refreshModels(provider, apiKey);
      return response.data;
    },
    onSuccess: (data) => {
      // Update the models cache with the refreshed data
      queryClient.setQueryData(llmKeys.models(), data);
      addToast({
        type: 'success',
        message: `Loaded ${data.models.length} models from provider`,
      });
    },
    onError: (error) => {
      const apiError = ApiErrorHandler.handle(error);
      if (apiError.statusCode === 429) {
        addToast({
          type: 'error',
          message: 'Rate limit exceeded. Please wait before refreshing again.',
        });
      } else {
        addToast({
          type: 'error',
          message: apiError.message || 'Failed to refresh models',
        });
      }
    },
  });
}
