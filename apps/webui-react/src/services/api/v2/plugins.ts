import apiClient from './client';
import type {
  PluginInfo,
  PluginListResponse,
  PluginListFilters,
  PluginManifest,
  PluginStatusResponse,
  PluginHealthResponse,
  PluginConfigSchema,
} from '../../../types/plugin';

/**
 * Response wrapper for single plugin
 */
interface PluginDetailResponse extends PluginInfo {}

/**
 * Response wrapper for plugin manifest
 */
interface PluginManifestResponse extends PluginManifest {}

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
    apiClient.get<PluginDetailResponse>(`/api/v2/plugins/${pluginId}`),

  /**
   * Get the manifest for a plugin
   * @param pluginId The plugin identifier
   */
  getManifest: (pluginId: string) =>
    apiClient.get<PluginManifestResponse>(`/api/v2/plugins/${pluginId}/manifest`),

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
};
