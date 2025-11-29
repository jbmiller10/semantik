import { useQuery } from '@tanstack/react-query';
import { modelsApi } from '../services/api/v2/models';
import type { ModelsResponse } from '../services/api/v2/models';

// Query key factory for consistent key generation
export const modelKeys = {
  all: ['models'] as const,
  list: () => [...modelKeys.all, 'list'] as const,
};

/**
 * Hook to fetch all available embedding models including plugin models.
 *
 * Models are cached for 5 minutes since they rarely change during a session.
 * Returns the full ModelsResponse including current_device and using_real_embeddings.
 *
 * @example
 * const { data, isLoading, error } = useEmbeddingModels();
 * if (data?.models) {
 *   Object.entries(data.models)
 *     .sort(([a], [b]) => a.localeCompare(b))
 *     .map(([modelName, config]) => ...)
 * }
 */
export function useEmbeddingModels() {
  return useQuery<ModelsResponse>({
    queryKey: modelKeys.list(),
    queryFn: modelsApi.getModels,
    staleTime: 5 * 60 * 1000, // 5 minutes - models rarely change
    gcTime: 10 * 60 * 1000, // Keep in cache for 10 minutes
  });
}
