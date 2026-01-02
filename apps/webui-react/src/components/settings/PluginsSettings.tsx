import { useState } from 'react';
import type { ReactElement } from 'react';
import {
  usePlugins,
  useEnablePlugin,
  useDisablePlugin,
  useRefreshPluginHealth,
} from '../../hooks/usePlugins';
import type { PluginInfo, PluginType } from '../../types/plugin';
import { PLUGIN_TYPE_LABELS, PLUGIN_TYPE_ORDER, groupPluginsByType } from '../../types/plugin';
import PluginCard from '../plugins/PluginCard';
import PluginConfigModal from '../plugins/PluginConfigModal';
import AvailablePluginsTab from '../plugins/AvailablePluginsTab';

type PluginsTab = 'installed' | 'available';

/**
 * Get icon for plugin type
 */
function getPluginTypeIcon(type: PluginType): ReactElement {
  switch (type) {
    case 'embedding':
      return (
        <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z"
          />
        </svg>
      );
    case 'chunking':
      return (
        <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M14.121 14.121L19 19m-7-7l7-7m-7 7l-2.879 2.879M12 12L9.121 9.121m0 5.758a3 3 0 10-4.243 4.243 3 3 0 004.243-4.243zm0-5.758a3 3 0 10-4.243-4.243 3 3 0 004.243 4.243z"
          />
        </svg>
      );
    case 'connector':
      return (
        <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M17 14v6m-3-3h6M6 10h2a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v2a2 2 0 002 2zm10 0h2a2 2 0 002-2V6a2 2 0 00-2-2h-2a2 2 0 00-2 2v2a2 2 0 002 2zM6 20h2a2 2 0 002-2v-2a2 2 0 00-2-2H6a2 2 0 00-2 2v2a2 2 0 002 2z"
          />
        </svg>
      );
    case 'reranker':
      return (
        <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M7 16V4m0 0L3 8m4-4l4 4m6 0v12m0 0l4-4m-4 4l-4-4"
          />
        </svg>
      );
    case 'extractor':
      return (
        <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z"
          />
        </svg>
      );
    default:
      return (
        <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M4 6h16M4 12h16M4 18h16"
          />
        </svg>
      );
  }
}

function PluginsSettings() {
  const [activeTab, setActiveTab] = useState<PluginsTab>('installed');
  const [selectedPlugin, setSelectedPlugin] = useState<PluginInfo | null>(null);
  const [enablingPluginId, setEnablingPluginId] = useState<string | null>(null);
  const [disablingPluginId, setDisablingPluginId] = useState<string | null>(null);
  const [refreshingHealthId, setRefreshingHealthId] = useState<string | null>(null);

  // Fetch plugins with health status
  const { data: plugins, isLoading, error, refetch } = usePlugins({ include_health: true });

  // Mutations
  const enableMutation = useEnablePlugin();
  const disableMutation = useDisablePlugin();
  const refreshHealthMutation = useRefreshPluginHealth();

  const handleEnable = async (pluginId: string) => {
    setEnablingPluginId(pluginId);
    try {
      await enableMutation.mutateAsync(pluginId);
    } finally {
      setEnablingPluginId(null);
    }
  };

  const handleDisable = async (pluginId: string) => {
    setDisablingPluginId(pluginId);
    try {
      await disableMutation.mutateAsync(pluginId);
    } finally {
      setDisablingPluginId(null);
    }
  };

  const handleRefreshHealth = async (pluginId: string) => {
    setRefreshingHealthId(pluginId);
    try {
      await refreshHealthMutation.mutateAsync(pluginId);
    } finally {
      setRefreshingHealthId(null);
    }
  };

  const handleConfigure = (plugin: PluginInfo) => {
    setSelectedPlugin(plugin);
  };

  const handleCloseConfigModal = () => {
    setSelectedPlugin(null);
  };

  // Group plugins by type (memoized by caller)
  const groupedPlugins = plugins ? groupPluginsByType(plugins) : null;

  // Render installed plugins content
  const renderInstalledContent = () => {
    // Loading state
    if (isLoading) {
      return (
        <div className="flex items-center justify-center py-12">
          <svg className="animate-spin h-8 w-8 text-gray-400" fill="none" viewBox="0 0 24 24">
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
            />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
            />
          </svg>
          <span className="ml-3 text-gray-500">Loading plugins...</span>
        </div>
      );
    }

    // Error state
    if (error) {
      return (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex">
            <svg
              className="h-5 w-5 text-red-400"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
            <div className="ml-3">
              <h3 className="text-sm font-medium text-red-800">Error loading plugins</h3>
              <p className="mt-1 text-sm text-red-700">{error.message}</p>
              <button
                onClick={() => refetch()}
                className="mt-2 text-sm font-medium text-red-600 hover:text-red-500"
              >
                Try again
              </button>
            </div>
          </div>
        </div>
      );
    }

    // Empty state
    if (!plugins || plugins.length === 0) {
      return (
        <div className="text-center py-12">
          <svg
            className="mx-auto h-12 w-12 text-gray-400"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M17 14v6m-3-3h6M6 10h2a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v2a2 2 0 002 2zm10 0h2a2 2 0 002-2V6a2 2 0 00-2-2h-2a2 2 0 00-2 2v2a2 2 0 002 2zM6 20h2a2 2 0 002-2v-2a2 2 0 00-2-2H6a2 2 0 00-2 2v2a2 2 0 002 2z"
            />
          </svg>
          <h3 className="mt-2 text-sm font-medium text-gray-900">No plugins installed</h3>
          <p className="mt-1 text-sm text-gray-500">
            Check the Available tab for plugins you can install.
          </p>
        </div>
      );
    }

    return (
      <div className="space-y-6">
        {/* Header with refresh button */}
        <div className="flex items-center justify-between">
          <p className="text-sm text-gray-500">
            Manage installed plugins. Changes require a service restart to take effect.
          </p>
          <button
            onClick={() => refetch()}
            className="inline-flex items-center px-3 py-1.5 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
          >
            <svg
              className="h-4 w-4 mr-1.5"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
              />
            </svg>
            Refresh
          </button>
        </div>

        {/* Plugin Groups */}
        {groupedPlugins &&
          PLUGIN_TYPE_ORDER.map((type) => {
            const typePlugins = groupedPlugins[type];
            if (typePlugins.length === 0) return null;

            return (
              <div key={type} className="bg-white shadow rounded-lg overflow-hidden">
                <div className="px-4 py-3 bg-gray-50 border-b border-gray-200">
                  <div className="flex items-center gap-2">
                    <span className="text-gray-400">{getPluginTypeIcon(type)}</span>
                    <h3 className="text-sm font-medium text-gray-900">
                      {PLUGIN_TYPE_LABELS[type]}
                    </h3>
                    <span className="text-xs text-gray-400">({typePlugins.length})</span>
                  </div>
                </div>
                <div className="divide-y divide-gray-100">
                  {typePlugins.map((plugin) => (
                    <div key={plugin.id} className="p-4">
                      <PluginCard
                        plugin={plugin}
                        onEnable={handleEnable}
                        onDisable={handleDisable}
                        onConfigure={handleConfigure}
                        onRefreshHealth={handleRefreshHealth}
                        isEnabling={enablingPluginId === plugin.id}
                        isDisabling={disablingPluginId === plugin.id}
                        isRefreshingHealth={refreshingHealthId === plugin.id}
                      />
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
      </div>
    );
  };

  return (
    <div className="space-y-6">
      {/* Tab Navigation */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8" aria-label="Plugins tabs">
          <button
            onClick={() => setActiveTab('installed')}
            className={`
              whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm
              ${
                activeTab === 'installed'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }
            `}
          >
            Installed
            {plugins && plugins.length > 0 && (
              <span className="ml-2 text-xs text-gray-400">({plugins.length})</span>
            )}
          </button>
          <button
            onClick={() => setActiveTab('available')}
            className={`
              whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm
              ${
                activeTab === 'available'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }
            `}
          >
            Available
          </button>
        </nav>
      </div>

      {/* Tab Content */}
      {activeTab === 'installed' && renderInstalledContent()}
      {activeTab === 'available' && <AvailablePluginsTab />}

      {/* Configuration Modal */}
      {selectedPlugin && (
        <PluginConfigModal plugin={selectedPlugin} onClose={handleCloseConfigModal} />
      )}
    </div>
  );
}

export default PluginsSettings;
