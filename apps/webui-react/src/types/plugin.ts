/**
 * Plugin type definitions for the plugin management UI.
 * These types match the backend plugin schemas.
 */

/**
 * Plugin manifest containing metadata about a plugin
 */
export interface PluginManifest {
  id: string;
  type: PluginType;
  version: string;
  display_name: string;
  description: string;
  author?: string | null;
  license?: string | null;
  homepage?: string | null;
  requires: string[];
  semantik_version?: string | null;
  capabilities: Record<string, unknown>;
}

/**
 * Available plugin types
 */
export type PluginType = 'embedding' | 'chunking' | 'connector' | 'reranker' | 'extractor' | 'parser';

/**
 * Health status of a plugin
 */
export type HealthStatus = 'healthy' | 'unhealthy' | 'unknown';

/**
 * Full plugin information including config and health
 */
export interface PluginInfo {
  id: string;
  type: PluginType;
  version: string;
  manifest: PluginManifest;
  enabled: boolean;
  config: Record<string, unknown>;
  health_status?: HealthStatus | null;
  last_health_check?: string | null;
  error_message?: string | null;
  requires_restart?: boolean | null;
}

/**
 * Response from listing plugins
 */
export interface PluginListResponse {
  plugins: PluginInfo[];
}

/**
 * Request to update plugin configuration
 */
export interface PluginConfigUpdateRequest {
  config: Record<string, unknown>;
}

/**
 * Response from enable/disable operations
 */
export interface PluginStatusResponse {
  plugin_id: string;
  enabled: boolean;
  requires_restart: boolean;
}

/**
 * Response from health check
 */
export interface PluginHealthResponse {
  plugin_id: string;
  health_status?: HealthStatus | null;
  last_health_check?: string | null;
  error_message?: string | null;
}

/**
 * Filters for listing plugins
 */
export interface PluginListFilters {
  type?: PluginType;
  enabled?: boolean;
  include_health?: boolean;
}

/**
 * JSON Schema property definition for plugin config forms
 * Supports a subset of JSON Schema Draft 7
 */
export interface JsonSchemaProperty {
  type: 'string' | 'number' | 'integer' | 'boolean' | 'array' | 'object';
  title?: string;
  description?: string;
  default?: unknown;
  enum?: string[];
  const?: unknown;
  minLength?: number;
  maxLength?: number;
  pattern?: string;
  minimum?: number;
  maximum?: number;
  exclusiveMinimum?: number;
  exclusiveMaximum?: number;
  items?: JsonSchemaProperty;
  minItems?: number;
  maxItems?: number;
  uniqueItems?: boolean;
  properties?: Record<string, JsonSchemaProperty>;
  required?: string[];
  additionalProperties?: boolean | JsonSchemaProperty;
}

/**
 * JSON Schema for plugin configuration
 */
export interface PluginConfigSchema {
  type: 'object';
  title?: string;
  description?: string;
  properties?: Record<string, JsonSchemaProperty>;
  required?: string[];
  additionalProperties?: boolean;
}

/**
 * Plugin type display names for the UI
 */
export const PLUGIN_TYPE_LABELS: Record<PluginType, string> = {
  embedding: 'Embedding Providers',
  chunking: 'Chunking Strategies',
  connector: 'Connectors',
  reranker: 'Rerankers',
  extractor: 'Extractors',
  parser: 'Parsers',
};

/**
 * Plugin type sort order for consistent display
 */
export const PLUGIN_TYPE_ORDER: PluginType[] = [
  'parser',
  'embedding',
  'reranker',
  'extractor',
  'chunking',
  'connector',
];

/**
 * Group plugins by their type
 */
export function groupPluginsByType(
  plugins: PluginInfo[]
): Record<PluginType, PluginInfo[]> {
  const grouped: Record<PluginType, PluginInfo[]> = {
    embedding: [],
    chunking: [],
    connector: [],
    reranker: [],
    extractor: [],
    parser: [],
  };

  for (const plugin of plugins) {
    if (plugin.type in grouped) {
      grouped[plugin.type].push(plugin);
    }
  }

  // Sort plugins within each type by display name
  for (const type of Object.keys(grouped) as PluginType[]) {
    grouped[type].sort((a, b) =>
      a.manifest.display_name.localeCompare(b.manifest.display_name)
    );
  }

  return grouped;
}

// --- Pipeline Plugins (for wizard) ---

/**
 * Simplified plugin info for pipeline configuration.
 * Used by the wizard to show all available plugins (builtin + external)
 * for each pipeline stage.
 */
export interface PipelinePluginInfo {
  id: string;
  type: PluginType;
  display_name: string;
  description: string;
  source: 'builtin' | 'external';
  enabled: boolean;
}

/**
 * Response from listing pipeline plugins
 */
export interface PipelinePluginListResponse {
  plugins: PipelinePluginInfo[];
}

/**
 * Filters for listing pipeline plugins
 */
export interface PipelinePluginFilters {
  plugin_type?: PluginType;
}

// --- Available Plugins (from registry) ---

/**
 * Available plugin from the remote registry (not installed)
 */
export interface AvailablePlugin {
  id: string;
  type: PluginType;
  name: string;
  description: string;
  author: string;
  repository: string;
  pypi?: string | null;
  verified: boolean;
  min_semantik_version?: string | null;
  tags: string[];
  is_compatible: boolean;
  compatibility_message?: string | null;
  is_installed: boolean;
  pending_restart: boolean;
  install_command: string;
}

/**
 * Request to install a plugin
 */
export interface PluginInstallRequest {
  plugin_id: string;
  version?: string | null;
}

/**
 * Response from install/uninstall operations
 */
export interface PluginInstallResponse {
  success: boolean;
  message: string;
  restart_required: boolean;
}

/**
 * Response from listing available plugins
 */
export interface AvailablePluginsListResponse {
  plugins: AvailablePlugin[];
  registry_version?: string | null;
  last_updated?: string | null;
  registry_source?: string | null;
  semantik_version: string;
}

/**
 * Filters for listing available plugins
 */
export interface AvailablePluginFilters {
  plugin_type?: PluginType;
  verified_only?: boolean;
  force_refresh?: boolean;
}

/**
 * Group available plugins by their type
 */
export function groupAvailablePluginsByType(
  plugins: AvailablePlugin[]
): Record<PluginType, AvailablePlugin[]> {
  const grouped: Record<PluginType, AvailablePlugin[]> = {
    embedding: [],
    chunking: [],
    connector: [],
    reranker: [],
    extractor: [],
    parser: [],
  };

  for (const plugin of plugins) {
    if (plugin.type in grouped) {
      grouped[plugin.type].push(plugin);
    }
  }

  // Sort plugins within each type by name
  for (const type of Object.keys(grouped) as PluginType[]) {
    grouped[type].sort((a, b) => a.name.localeCompare(b.name));
  }

  return grouped;
}
