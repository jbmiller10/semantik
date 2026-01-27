import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { pluginsApi } from '../services/api/v2/plugins';
import type {
  PluginInfo,
  PluginListFilters,
  PluginManifest,
  PluginConfigSchema,
  PluginStatusResponse,
  PluginHealthResponse,
  AvailablePluginsListResponse,
  AvailablePluginFilters,
  PluginInstallRequest,
  PluginInstallResponse,
  PipelinePluginInfo,
  PipelinePluginFilters,
} from '../types/plugin';

// Re-export PluginConfigSchema for convenience
export type { PluginConfigSchema } from '../types/plugin';

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
  available: (filters?: AvailablePluginFilters) =>
    [...pluginKeys.all, 'available', filters] as const,
  pipeline: (filters?: PipelinePluginFilters) =>
    [...pluginKeys.all, 'pipeline', filters] as const,
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
      // Invalidate all list queries regardless of filters
      queryClient.invalidateQueries({ queryKey: [...pluginKeys.all, 'list'] });
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
      // Invalidate all list queries regardless of filters (e.g., include_health)
      queryClient.invalidateQueries({ queryKey: [...pluginKeys.all, 'list'] });
    },
  });
}

// --- Pipeline Plugins (for wizard) ---

/**
 * Hook to fetch all plugins (builtin + external) for pipeline configuration
 * Used by the wizard's pipeline editor to show available plugins for each stage.
 * @param filters Optional filters for plugin type
 */
export function usePipelinePlugins(filters?: PipelinePluginFilters) {
  return useQuery({
    queryKey: pluginKeys.pipeline(filters),
    queryFn: async (): Promise<PipelinePluginInfo[]> => {
      const response = await pluginsApi.listPipeline(filters);
      return response.data.plugins;
    },
    staleTime: 60 * 1000, // Cache for 60 seconds (plugins don't change often)
    gcTime: 5 * 60 * 1000, // Keep in cache for 5 minutes
  });
}

/**
 * Hook to fetch a plugin's configuration schema for pipeline configuration
 * Works for both builtin and external plugins.
 * @param pluginId The plugin identifier
 */
export function usePipelinePluginConfigSchema(pluginId: string) {
  return useQuery({
    queryKey: [...pluginKeys.all, 'pipeline-config-schema', pluginId] as const,
    queryFn: async (): Promise<PluginConfigSchema | null> => {
      const response = await pluginsApi.getPipelineConfigSchema(pluginId);
      return response.data;
    },
    enabled: !!pluginId,
    staleTime: 5 * 60 * 1000, // Schemas don't change often
  });
}

// --- Available Plugins (from registry) ---

/**
 * Hook to fetch available plugins from the remote registry
 * @param filters Optional filters for type and verified status
 */
export function useAvailablePlugins(filters?: AvailablePluginFilters) {
  return useQuery({
    queryKey: pluginKeys.available(filters),
    queryFn: async (): Promise<AvailablePluginsListResponse> => {
      const response = await pluginsApi.listAvailable(filters);
      return response.data;
    },
    staleTime: 5 * 60 * 1000, // Cache for 5 minutes (registry data)
    gcTime: 30 * 60 * 1000, // Keep in cache for 30 minutes
  });
}

/**
 * Hook to force refresh the available plugins registry
 */
export function useRefreshAvailablePlugins() {
  const queryClient = useQueryClient();

  return useMutation<AvailablePluginsListResponse, Error, void>({
    mutationFn: async () => {
      const response = await pluginsApi.refreshAvailable();
      return response.data;
    },
    onSuccess: (data) => {
      // Update the cache with fresh data
      queryClient.setQueryData(pluginKeys.available(undefined), data);
      // Invalidate filtered queries
      queryClient.invalidateQueries({
        queryKey: [...pluginKeys.all, 'available'],
      });
    },
  });
}

// --- Plugin Installation ---

/**
 * Hook to install a plugin from the registry
 * Invalidates available plugins query on success to update install state
 */
export function usePluginInstall() {
  const queryClient = useQueryClient();

  return useMutation<PluginInstallResponse, Error, PluginInstallRequest>({
    mutationFn: async (request: PluginInstallRequest) => {
      const response = await pluginsApi.install(request);
      return response.data;
    },
    onSuccess: () => {
      // Invalidate available plugins to update is_installed and pending_restart state
      queryClient.invalidateQueries({
        queryKey: [...pluginKeys.all, 'available'],
      });
    },
  });
}

/**
 * Hook to uninstall an installed plugin
 * Invalidates both available and installed plugins queries on success
 */
export function usePluginUninstall() {
  const queryClient = useQueryClient();

  return useMutation<PluginInstallResponse, Error, string>({
    mutationFn: async (pluginId: string) => {
      const response = await pluginsApi.uninstall(pluginId);
      return response.data;
    },
    onSuccess: () => {
      // Invalidate all plugin queries
      queryClient.invalidateQueries({ queryKey: pluginKeys.all });
    },
  });
}
