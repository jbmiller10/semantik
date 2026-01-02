import { useState, useMemo } from 'react';
import {
  useAvailablePlugins,
  useRefreshAvailablePlugins,
} from '../../hooks/usePlugins';
import type { PluginType } from '../../types/plugin';
import {
  PLUGIN_TYPE_LABELS,
  PLUGIN_TYPE_ORDER,
  groupAvailablePluginsByType,
} from '../../types/plugin';
import AvailablePluginCard from './AvailablePluginCard';

/**
 * Tab component for browsing available plugins from the registry.
 * Provides search, filtering, and displays plugins grouped by type.
 */
function AvailablePluginsTab() {
  const [searchQuery, setSearchQuery] = useState('');
  const [filterType, setFilterType] = useState<PluginType | 'all'>('all');
  const [verifiedOnly, setVerifiedOnly] = useState(false);

  const { data, isLoading, error, refetch } = useAvailablePlugins();
  const refreshMutation = useRefreshAvailablePlugins();

  const handleRefresh = async () => {
    await refreshMutation.mutateAsync();
  };

  // Filter and search plugins
  const filteredPlugins = useMemo(() => {
    if (!data?.plugins) return [];

    let plugins = data.plugins;

    // Filter by type
    if (filterType !== 'all') {
      plugins = plugins.filter((p) => p.type === filterType);
    }

    // Filter by verified
    if (verifiedOnly) {
      plugins = plugins.filter((p) => p.verified);
    }

    // Search
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      plugins = plugins.filter(
        (p) =>
          p.name.toLowerCase().includes(query) ||
          p.description.toLowerCase().includes(query) ||
          p.author.toLowerCase().includes(query) ||
          p.tags.some((t) => t.toLowerCase().includes(query))
      );
    }

    return plugins;
  }, [data?.plugins, filterType, verifiedOnly, searchQuery]);

  // Group plugins by type for display
  const groupedPlugins = useMemo(() => {
    return groupAvailablePluginsByType(filteredPlugins);
  }, [filteredPlugins]);

  // Loading state
  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <svg
          className="animate-spin h-8 w-8 text-gray-400"
          fill="none"
          viewBox="0 0 24 24"
        >
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
        <span className="ml-3 text-gray-500">Loading available plugins...</span>
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
            <h3 className="text-sm font-medium text-red-800">
              Failed to load plugin registry
            </h3>
            <p className="mt-1 text-sm text-red-700">
              Could not fetch the plugin registry. Check your network connection.
            </p>
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

  return (
    <div className="space-y-6">
      {/* Header with search and filters */}
      <div className="flex flex-col sm:flex-row gap-4">
        {/* Search */}
        <div className="flex-1 relative">
          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
            <svg
              className="h-5 w-5 text-gray-400"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
              />
            </svg>
          </div>
          <input
            type="text"
            placeholder="Search plugins..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="block w-full pl-10 pr-3 py-2 border border-gray-300 rounded-md leading-5 bg-white placeholder-gray-500 focus:outline-none focus:placeholder-gray-400 focus:ring-1 focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
          />
        </div>

        {/* Type filter */}
        <select
          value={filterType}
          onChange={(e) => setFilterType(e.target.value as PluginType | 'all')}
          className="block w-full sm:w-auto pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md"
        >
          <option value="all">All Types</option>
          {PLUGIN_TYPE_ORDER.map((type) => (
            <option key={type} value={type}>
              {PLUGIN_TYPE_LABELS[type]}
            </option>
          ))}
        </select>

        {/* Verified only toggle */}
        <label className="inline-flex items-center">
          <input
            type="checkbox"
            checked={verifiedOnly}
            onChange={(e) => setVerifiedOnly(e.target.checked)}
            className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
          />
          <span className="ml-2 text-sm text-gray-600">Verified only</span>
        </label>

        {/* Refresh button */}
        <button
          onClick={handleRefresh}
          disabled={refreshMutation.isPending}
          className="inline-flex items-center px-3 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50"
        >
          <svg
            className={`h-4 w-4 mr-1.5 ${refreshMutation.isPending ? 'animate-spin' : ''}`}
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
          {refreshMutation.isPending ? 'Refreshing...' : 'Refresh'}
        </button>
      </div>

      {/* Registry info */}
      {data && (
        <div className="text-xs text-gray-400 flex items-center gap-4">
          <span>Registry v{data.registry_version}</span>
          {data.last_updated && (
            <span>
              Updated: {new Date(data.last_updated).toLocaleDateString()}
            </span>
          )}
          <span>Semantik v{data.semantik_version}</span>
          {data.registry_source && (
            <span className="text-gray-300">({data.registry_source})</span>
          )}
        </div>
      )}

      {/* Empty state */}
      {filteredPlugins.length === 0 && (
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
              d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M12 12h.01M12 12h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
          <h3 className="mt-2 text-sm font-medium text-gray-900">
            No plugins found
          </h3>
          <p className="mt-1 text-sm text-gray-500">
            {searchQuery
              ? 'Try a different search term.'
              : 'No plugins match the current filters.'}
          </p>
        </div>
      )}

      {/* Plugin Groups */}
      {PLUGIN_TYPE_ORDER.map((type) => {
        const typePlugins = groupedPlugins[type];
        if (typePlugins.length === 0) return null;

        return (
          <div key={type} className="space-y-3">
            <h3 className="text-sm font-medium text-gray-700">
              {PLUGIN_TYPE_LABELS[type]}
              <span className="ml-2 text-gray-400">({typePlugins.length})</span>
            </h3>
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {typePlugins.map((plugin) => (
                <AvailablePluginCard key={plugin.id} plugin={plugin} />
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default AvailablePluginsTab;
