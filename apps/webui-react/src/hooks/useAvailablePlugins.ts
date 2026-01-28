/**
 * Hook to fetch available plugins for a specific pipeline node type.
 * Used by the node picker popover to show plugins that can be added at each tier.
 */

import { useMemo } from 'react';
import { usePipelinePlugins } from './usePlugins';
import { nodeTypeToPluginType } from '@/utils/pipelinePluginMapping';
import type { NodeType } from '@/types/pipeline';
import type { PipelinePluginInfo } from '@/types/plugin';

export interface AvailablePluginOption {
  id: string;
  name: string;
  description: string;
}

export interface UseAvailablePluginsResult {
  plugins: AvailablePluginOption[];
  isLoading: boolean;
  error: Error | null;
  refetch: () => void;
}

/**
 * Fetch available plugins for a given pipeline node type.
 * Returns a simplified list suitable for the node picker UI.
 */
export function useAvailablePlugins(type: NodeType): UseAvailablePluginsResult {
  const pluginType = nodeTypeToPluginType(type);

  const { data: allPlugins, isLoading, error, refetch } = usePipelinePlugins({
    plugin_type: pluginType,
  });

  const plugins = useMemo((): AvailablePluginOption[] => {
    if (!allPlugins) return [];

    // Filter to enabled plugins only and map to simplified format
    return allPlugins
      .filter((plugin: PipelinePluginInfo) => plugin.enabled)
      .map((plugin: PipelinePluginInfo) => ({
        id: plugin.id,
        name: plugin.display_name,
        description: plugin.description,
      }));
  }, [allPlugins]);

  return {
    plugins,
    isLoading,
    error: error as Error | null,
    refetch,
  };
}

export default useAvailablePlugins;
