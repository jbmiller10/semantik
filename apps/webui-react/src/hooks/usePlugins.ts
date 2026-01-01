import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { pluginsApi } from '../services/api/v2/plugins';
import type {
  PluginInfo,
  PluginListFilters,
  PluginManifest,
  PluginConfigSchema,
  PluginStatusResponse,
  PluginHealthResponse,
} from '../types/plugin';

/**
 * Query key factory for plugin queries
 */
export const pluginKeys = {
  all: ['plugins'] as const,
  list: (filters?: PluginListFilters) => [...pluginKeys.all, 'list', filters] as const,
  detail: (pluginId: string) => [...pluginKeys.all, 'detail', pluginId] as const,
  manifest: (pluginId: string) => [...pluginKeys.all, 'manifest', pluginId] as const,
  configSchema: (pluginId: string) => [...pluginKeys.all, 'config-schema', pluginId] as const,
  health: (pluginId: string) => [...pluginKeys.all, 'health', pluginId] as const,
};

/**
 * Hook to fetch all plugins with optional filtering
 * @param filters Optional filters for type, enabled status, and health inclusion
 */
export function usePlugins(filters?: PluginListFilters) {
  return useQuery({
    queryKey: pluginKeys.list(filters),
    queryFn: async (): Promise<PluginInfo[]> => {
      const response = await pluginsApi.list(filters);
      return response.data.plugins;
    },
    staleTime: 30 * 1000, // Cache for 30 seconds (plugins can change)
    gcTime: 5 * 60 * 1000, // Keep in cache for 5 minutes
  });
}

/**
 * Hook to fetch a specific plugin's details
 * @param pluginId The plugin identifier
 */
export function usePlugin(pluginId: string) {
  return useQuery({
    queryKey: pluginKeys.detail(pluginId),
    queryFn: async (): Promise<PluginInfo> => {
      const response = await pluginsApi.get(pluginId);
      return response.data;
    },
    enabled: !!pluginId,
    staleTime: 30 * 1000,
  });
}

/**
 * Hook to fetch a plugin's manifest
 * @param pluginId The plugin identifier
 */
export function usePluginManifest(pluginId: string) {
  return useQuery({
    queryKey: pluginKeys.manifest(pluginId),
    queryFn: async (): Promise<PluginManifest> => {
      const response = await pluginsApi.getManifest(pluginId);
      return response.data;
    },
    enabled: !!pluginId,
    staleTime: 5 * 60 * 1000, // Manifests don't change often
  });
}

/**
 * Hook to fetch a plugin's configuration schema
 * @param pluginId The plugin identifier
 */
export function usePluginConfigSchema(pluginId: string) {
  return useQuery({
    queryKey: pluginKeys.configSchema(pluginId),
    queryFn: async (): Promise<PluginConfigSchema | null> => {
      const response = await pluginsApi.getConfigSchema(pluginId);
      return response.data;
    },
    enabled: !!pluginId,
    staleTime: 5 * 60 * 1000, // Schemas don't change often
  });
}

/**
 * Hook to fetch a plugin's health status
 * @param pluginId The plugin identifier
 */
export function usePluginHealth(pluginId: string) {
  return useQuery({
    queryKey: pluginKeys.health(pluginId),
    queryFn: async (): Promise<PluginHealthResponse> => {
      const response = await pluginsApi.checkHealth(pluginId);
      return response.data;
    },
    enabled: !!pluginId,
    staleTime: 60 * 1000, // Health status cached for 1 minute
  });
}

/**
 * Hook to enable a plugin
 * Invalidates plugin list on success
 */
export function useEnablePlugin() {
  const queryClient = useQueryClient();

  return useMutation<PluginStatusResponse, Error, string>({
    mutationFn: async (pluginId: string) => {
      const response = await pluginsApi.enable(pluginId);
      return response.data;
    },
    onSuccess: (_data, pluginId) => {
      // Invalidate both the list and the specific plugin
      queryClient.invalidateQueries({ queryKey: pluginKeys.all });
      queryClient.invalidateQueries({ queryKey: pluginKeys.detail(pluginId) });
    },
  });
}

/**
 * Hook to disable a plugin
 * Invalidates plugin list on success
 */
export function useDisablePlugin() {
  const queryClient = useQueryClient();

  return useMutation<PluginStatusResponse, Error, string>({
    mutationFn: async (pluginId: string) => {
      const response = await pluginsApi.disable(pluginId);
      return response.data;
    },
    onSuccess: (_data, pluginId) => {
      // Invalidate both the list and the specific plugin
      queryClient.invalidateQueries({ queryKey: pluginKeys.all });
      queryClient.invalidateQueries({ queryKey: pluginKeys.detail(pluginId) });
    },
  });
}

/**
 * Hook to update plugin configuration
 * Invalidates plugin data on success
 */
export function useUpdatePluginConfig() {
  const queryClient = useQueryClient();

  return useMutation<
    PluginInfo,
    Error,
    { pluginId: string; config: Record<string, unknown> }
  >({
    mutationFn: async ({ pluginId, config }) => {
      const response = await pluginsApi.updateConfig(pluginId, config);
      return response.data;
    },
    onSuccess: (data, { pluginId }) => {
      // Update the cache with the returned plugin info
      queryClient.setQueryData(pluginKeys.detail(pluginId), data);
      // Also invalidate the list to ensure it's fresh
      queryClient.invalidateQueries({ queryKey: pluginKeys.list() });
    },
  });
}

/**
 * Hook to manually refresh a plugin's health status
 */
export function useRefreshPluginHealth() {
  const queryClient = useQueryClient();

  return useMutation<PluginHealthResponse, Error, string>({
    mutationFn: async (pluginId: string) => {
      const response = await pluginsApi.checkHealth(pluginId);
      return response.data;
    },
    onSuccess: (data, pluginId) => {
      // Update the health cache
      queryClient.setQueryData(pluginKeys.health(pluginId), data);
      // Also invalidate the detail to get updated health_status
      queryClient.invalidateQueries({ queryKey: pluginKeys.detail(pluginId) });
      queryClient.invalidateQueries({ queryKey: pluginKeys.list() });
    },
  });
}
