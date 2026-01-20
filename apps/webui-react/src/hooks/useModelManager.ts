/**
 * React Query hooks for Model Manager feature.
 * Note: This is separate from useModels.ts which handles legacy embedding models.
 */
import { useQuery } from '@tanstack/react-query';
import { modelManagerApi, type ListModelsParams } from '../services/api/v2/model-manager';
import type { ModelListResponse, ModelType } from '../types/model-manager';

// Query key factory for consistent key generation
export const modelManagerKeys = {
  all: ['model-manager'] as const,
  list: (params?: ListModelsParams) => [...modelManagerKeys.all, 'list', params ?? {}] as const,
  usage: (modelId: string) => [...modelManagerKeys.all, 'usage', modelId] as const,
  task: (taskId: string) => [...modelManagerKeys.all, 'task', taskId] as const,
};

export interface UseModelManagerModelsOptions {
  modelType?: ModelType;
  installedOnly?: boolean;
  includeCacheSize?: boolean;
  enabled?: boolean;
}

/**
 * Hook to fetch curated models from the model manager.
 *
 * @example
 * // Fetch all models with cache size info
 * const { data, isLoading, refetch } = useModelManagerModels({ includeCacheSize: true });
 *
 * // Fetch only installed embedding models
 * const { data } = useModelManagerModels({ modelType: 'embedding', installedOnly: true });
 */
export function useModelManagerModels(options?: UseModelManagerModelsOptions) {
  const params: ListModelsParams = {
    model_type: options?.modelType,
    installed_only: options?.installedOnly,
    include_cache_size: options?.includeCacheSize,
  };

  return useQuery<ModelListResponse>({
    queryKey: modelManagerKeys.list(params),
    queryFn: () => modelManagerApi.listModels(params),
    staleTime: 2 * 60 * 1000, // 2 minutes - models change more frequently during downloads
    gcTime: 5 * 60 * 1000, // Keep in cache for 5 minutes
    enabled: options?.enabled ?? true,
  });
}

// Phase 2B/2C hooks - placeholders for future implementation
// export function useStartModelDownload() { ... }
// export function useModelUsage(modelId: string) { ... }
// export function useDeleteModel() { ... }
// export function useTaskProgress(taskId: string | null) { ... }
