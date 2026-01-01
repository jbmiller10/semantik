import { useState } from 'react';
import type { PluginInfo, HealthStatus } from '../../types/plugin';

interface PluginCardProps {
  plugin: PluginInfo;
  onEnable: (pluginId: string) => void;
  onDisable: (pluginId: string) => void;
  onConfigure: (plugin: PluginInfo) => void;
  onRefreshHealth: (pluginId: string) => void;
  isEnabling?: boolean;
  isDisabling?: boolean;
  isRefreshingHealth?: boolean;
}

/**
 * Get health status color classes
 */
function getHealthStatusClasses(status: HealthStatus | null | undefined): {
  dot: string;
  text: string;
  label: string;
} {
  switch (status) {
    case 'healthy':
      return {
        dot: 'bg-green-400',
        text: 'text-green-700',
        label: 'Healthy',
      };
    case 'unhealthy':
      return {
        dot: 'bg-red-400',
        text: 'text-red-700',
        label: 'Unhealthy',
      };
    default:
      return {
        dot: 'bg-gray-400',
        text: 'text-gray-500',
        label: 'Unknown',
      };
  }
}

function PluginCard({
  plugin,
  onEnable,
  onDisable,
  onConfigure,
  onRefreshHealth,
  isEnabling = false,
  isDisabling = false,
  isRefreshingHealth = false,
}: PluginCardProps) {
  const [showDisableConfirm, setShowDisableConfirm] = useState(false);

  const healthClasses = getHealthStatusClasses(plugin.health_status as HealthStatus);
  const isLoading = isEnabling || isDisabling;

  const handleToggle = () => {
    if (plugin.enabled) {
      setShowDisableConfirm(true);
    } else {
      onEnable(plugin.id);
    }
  };

  const handleConfirmDisable = () => {
    setShowDisableConfirm(false);
    onDisable(plugin.id);
  };

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4 hover:shadow-sm transition-shadow">
      <div className="flex items-start justify-between">
        {/* Plugin Info */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <h4 className="text-sm font-medium text-gray-900 truncate">
              {plugin.manifest.display_name}
            </h4>
            <span className="text-xs text-gray-400">v{plugin.version}</span>
            {plugin.requires_restart && (
              <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-yellow-100 text-yellow-800">
                Restart required
              </span>
            )}
          </div>
          <p className="mt-1 text-sm text-gray-500 line-clamp-2">
            {plugin.manifest.description}
          </p>

          {/* Health Status */}
          <div className="mt-2 flex items-center gap-3">
            <button
              onClick={() => onRefreshHealth(plugin.id)}
              disabled={isRefreshingHealth}
              className="flex items-center gap-1.5 text-xs hover:underline disabled:opacity-50"
            >
              <span
                className={`inline-block w-2 h-2 rounded-full ${healthClasses.dot} ${
                  isRefreshingHealth ? 'animate-pulse' : ''
                }`}
              />
              <span className={healthClasses.text}>{healthClasses.label}</span>
              {isRefreshingHealth && (
                <svg
                  className="animate-spin h-3 w-3 text-gray-400"
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
              )}
            </button>

            {plugin.error_message && (
              <span className="text-xs text-red-600 truncate" title={plugin.error_message}>
                {plugin.error_message}
              </span>
            )}
          </div>

          {/* Author and Homepage */}
          {(plugin.manifest.author || plugin.manifest.homepage) && (
            <div className="mt-2 flex items-center gap-3 text-xs text-gray-400">
              {plugin.manifest.author && <span>by {plugin.manifest.author}</span>}
              {plugin.manifest.homepage && (
                <a
                  href={plugin.manifest.homepage}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-blue-500 hover:underline"
                >
                  Homepage
                </a>
              )}
            </div>
          )}
        </div>

        {/* Actions */}
        <div className="flex items-center gap-2 ml-4">
          {/* Configure Button */}
          <button
            onClick={() => onConfigure(plugin)}
            className="inline-flex items-center px-2.5 py-1.5 border border-gray-300 shadow-sm text-xs font-medium rounded text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
          >
            <svg
              className="h-3.5 w-3.5 mr-1"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"
              />
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
              />
            </svg>
            Configure
          </button>

          {/* Enable/Disable Toggle */}
          <button
            onClick={handleToggle}
            disabled={isLoading}
            className={`
              relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent
              transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2
              disabled:opacity-50 disabled:cursor-not-allowed
              ${plugin.enabled ? 'bg-blue-600' : 'bg-gray-200'}
            `}
            role="switch"
            aria-checked={plugin.enabled}
            aria-label={plugin.enabled ? 'Disable plugin' : 'Enable plugin'}
          >
            <span
              className={`
                pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0
                transition duration-200 ease-in-out
                ${plugin.enabled ? 'translate-x-5' : 'translate-x-0'}
              `}
            >
              {isLoading && (
                <svg
                  className="absolute inset-0 h-5 w-5 animate-spin text-gray-400"
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
              )}
            </span>
          </button>
        </div>
      </div>

      {/* Disable Confirmation */}
      {showDisableConfirm && (
        <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-md">
          <p className="text-sm text-yellow-800">
            Disabling this plugin requires a service restart. Are you sure?
          </p>
          <div className="mt-2 flex gap-2">
            <button
              onClick={handleConfirmDisable}
              disabled={isDisabling}
              className="inline-flex items-center px-2.5 py-1.5 border border-transparent text-xs font-medium rounded text-white bg-yellow-600 hover:bg-yellow-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-yellow-500 disabled:opacity-50"
            >
              {isDisabling ? 'Disabling...' : 'Yes, disable'}
            </button>
            <button
              onClick={() => setShowDisableConfirm(false)}
              className="inline-flex items-center px-2.5 py-1.5 border border-gray-300 text-xs font-medium rounded text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-500"
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default PluginCard;
