import apiClient from './client';
import type {
  PluginInfo,
  PluginListResponse,
  PluginListFilters,
  PluginManifest,
  PluginStatusResponse,
  PluginHealthResponse,
  PluginConfigSchema,
  PluginInstallRequest,
  PluginInstallResponse,
  AvailablePluginsListResponse,
  AvailablePluginFilters,
  PipelinePluginListResponse,
  PipelinePluginFilters,
} from '../../../types/plugin';

/**
 * V2 Plugins API client
 * Provides plugin management endpoints for discovery, configuration, and health
 */
export const pluginsApi = {
  /**
   * List all installed external plugins
   * @param filters Optional filters for type, enabled status, and health inclusion
   */
  list: (filters?: PluginListFilters) =>
    apiClient.get<PluginListResponse>('/api/v2/plugins', { params: filters }),

  /**
   * Get detailed info for a specific plugin
   * @param pluginId The plugin identifier
   */
  get: (pluginId: string) =>
    apiClient.get<PluginInfo>(`/api/v2/plugins/${pluginId}`),

  /**
   * Get the manifest for a plugin
   * @param pluginId The plugin identifier
   */
  getManifest: (pluginId: string) =>
    apiClient.get<PluginManifest>(`/api/v2/plugins/${pluginId}/manifest`),

  /**
   * Get JSON Schema for plugin configuration
   * @param pluginId The plugin identifier
   * @returns The config schema, or null if plugin has no configuration
   */
  getConfigSchema: (pluginId: string) =>
    apiClient.get<PluginConfigSchema | null>(`/api/v2/plugins/${pluginId}/config-schema`),

  /**
   * Enable a plugin (requires service restart to take effect)
   * @param pluginId The plugin identifier
   */
  enable: (pluginId: string) =>
    apiClient.post<PluginStatusResponse>(`/api/v2/plugins/${pluginId}/enable`),

  /**
   * Disable a plugin (requires service restart to take effect)
   * @param pluginId The plugin identifier
   */
  disable: (pluginId: string) =>
    apiClient.post<PluginStatusResponse>(`/api/v2/plugins/${pluginId}/disable`),

  /**
   * Update plugin configuration (validated against schema if present)
   * Requires service restart to take effect
   * @param pluginId The plugin identifier
   * @param config The new configuration values
   */
  updateConfig: (pluginId: string, config: Record<string, unknown>) =>
    apiClient.put<PluginInfo>(`/api/v2/plugins/${pluginId}/config`, { config }),

  /**
   * Run a health check for a plugin
   * @param pluginId The plugin identifier
   */
  checkHealth: (pluginId: string) =>
    apiClient.get<PluginHealthResponse>(`/api/v2/plugins/${pluginId}/health`),

  // --- Pipeline Plugins (for wizard) ---

  /**
   * List all plugins (builtin + external) for pipeline configuration
   * @param filters Optional filters for plugin type
   */
  listPipeline: (filters?: PipelinePluginFilters) =>
    apiClient.get<PipelinePluginListResponse>('/api/v2/plugins/pipeline', {
      params: filters,
    }),

  /**
   * Get JSON Schema for any plugin configuration (builtin or external)
   * @param pluginId The plugin identifier
   * @returns The config schema, or null if plugin has no configuration
   */
  getPipelineConfigSchema: (pluginId: string) =>
    apiClient.get<PluginConfigSchema | null>(`/api/v2/plugins/pipeline/${pluginId}/config-schema`),

  // --- Available Plugins (from registry) ---

  /**
   * List available plugins from the remote registry
   * @param filters Optional filters for type and verified status
   */
  listAvailable: (filters?: AvailablePluginFilters) =>
    apiClient.get<AvailablePluginsListResponse>('/api/v2/plugins/available', {
      params: filters,
    }),

  /**
   * Force refresh the available plugins registry cache
   */
  refreshAvailable: () =>
    apiClient.post<AvailablePluginsListResponse>('/api/v2/plugins/available/refresh'),

  // --- Plugin Installation ---

  /**
   * Install a plugin from the registry (admin only)
   * @param request The install request with plugin_id and optional version
   */
  install: (request: PluginInstallRequest) =>
    apiClient.post<PluginInstallResponse>('/api/v2/plugins/install', request),

  /**
   * Uninstall an installed plugin (admin only)
   * @param pluginId The plugin identifier
   */
  uninstall: (pluginId: string) =>
    apiClient.delete<PluginInstallResponse>(`/api/v2/plugins/${pluginId}/uninstall`),
};
