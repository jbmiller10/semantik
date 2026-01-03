import { useState } from 'react';
import type { AvailablePlugin } from '../../types/plugin';
import { usePluginInstall, usePluginUninstall } from '../../hooks/usePlugins';
import { useAuthStore } from '../../stores/authStore';

interface AvailablePluginCardProps {
  plugin: AvailablePlugin;
}

/**
 * Card component for displaying an available (not installed) plugin from the registry.
 * Shows plugin info, verification status, compatibility, and install/uninstall buttons.
 */
function AvailablePluginCard({ plugin }: AvailablePluginCardProps) {
  const [resultMessage, setResultMessage] = useState<{
    type: 'success' | 'error';
    text: string;
  } | null>(null);

  const user = useAuthStore((state) => state.user);
  const isAdmin = user?.is_superuser ?? false;

  const installMutation = usePluginInstall();
  const uninstallMutation = usePluginUninstall();

  const isLoading = installMutation.isPending || uninstallMutation.isPending;

  const handleInstall = async () => {
    setResultMessage(null);
    try {
      const result = await installMutation.mutateAsync({ plugin_id: plugin.id });
      if (result.success) {
        setResultMessage({ type: 'success', text: result.message });
      } else {
        setResultMessage({ type: 'error', text: result.message });
      }
    } catch (error) {
      setResultMessage({
        type: 'error',
        text: error instanceof Error ? error.message : 'Installation failed',
      });
    }
  };

  const handleUninstall = async () => {
    setResultMessage(null);
    try {
      const result = await uninstallMutation.mutateAsync(plugin.id);
      if (result.success) {
        setResultMessage({ type: 'success', text: result.message });
      } else {
        setResultMessage({ type: 'error', text: result.message });
      }
    } catch (error) {
      setResultMessage({
        type: 'error',
        text: error instanceof Error ? error.message : 'Uninstallation failed',
      });
    }
  };

  return (
    <div
      className={`
        bg-white border rounded-lg p-4 transition-shadow
        ${plugin.is_installed || plugin.pending_restart ? 'border-green-200 bg-green-50' : 'border-gray-200 hover:shadow-sm'}
        ${!plugin.is_compatible ? 'opacity-60' : ''}
      `}
    >
      <div className="flex items-start justify-between">
        {/* Plugin Info */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <h4 className="text-sm font-medium text-gray-900">{plugin.name}</h4>

            {/* Verified badge */}
            {plugin.verified ? (
              <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-green-100 text-green-800">
                <svg
                  className="w-3 h-3 mr-1"
                  fill="currentColor"
                  viewBox="0 0 20 20"
                >
                  <path
                    fillRule="evenodd"
                    d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                    clipRule="evenodd"
                  />
                </svg>
                Verified
              </span>
            ) : (
              <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-yellow-100 text-yellow-800">
                Unverified
              </span>
            )}

            {/* Installed badge */}
            {plugin.is_installed && (
              <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-blue-100 text-blue-800">
                Installed
              </span>
            )}

            {/* Pending restart badge */}
            {plugin.pending_restart && (
              <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-orange-100 text-orange-800">
                <svg
                  className="w-3 h-3 mr-1"
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
                Restart Required
              </span>
            )}

            {/* Incompatible badge */}
            {!plugin.is_compatible && (
              <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-red-100 text-red-800">
                Incompatible
              </span>
            )}
          </div>

          <p className="mt-1 text-sm text-gray-500 line-clamp-2">
            {plugin.description}
          </p>

          {/* Compatibility message */}
          {!plugin.is_compatible && plugin.compatibility_message && (
            <p className="mt-1 text-xs text-red-600">
              {plugin.compatibility_message}
            </p>
          )}

          {/* Author and links */}
          <div className="mt-2 flex items-center gap-3 text-xs text-gray-400">
            <span>by {plugin.author}</span>
            <a
              href={plugin.repository}
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-blue-500 hover:underline inline-flex items-center"
            >
              <svg
                className="w-3 h-3 mr-1"
                fill="currentColor"
                viewBox="0 0 20 20"
              >
                <path
                  fillRule="evenodd"
                  d="M10 0C4.477 0 0 4.477 0 10c0 4.42 2.865 8.166 6.839 9.489.5.092.682-.217.682-.482 0-.237-.008-.866-.013-1.7-2.782.604-3.369-1.341-3.369-1.341-.454-1.155-1.11-1.462-1.11-1.462-.908-.62.069-.608.069-.608 1.003.07 1.531 1.03 1.531 1.03.892 1.529 2.341 1.087 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.11-4.555-4.943 0-1.091.39-1.984 1.029-2.683-.103-.253-.446-1.27.098-2.647 0 0 .84-.269 2.75 1.025A9.578 9.578 0 0110 4.836c.85.004 1.705.115 2.504.337 1.909-1.294 2.747-1.025 2.747-1.025.546 1.377.203 2.394.1 2.647.64.699 1.028 1.592 1.028 2.683 0 3.842-2.339 4.687-4.566 4.935.359.309.678.919.678 1.852 0 1.336-.012 2.415-.012 2.743 0 .267.18.578.688.48C17.137 18.163 20 14.418 20 10c0-5.523-4.477-10-10-10z"
                  clipRule="evenodd"
                />
              </svg>
              GitHub
            </a>
          </div>

          {/* Tags */}
          {plugin.tags.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-1">
              {plugin.tags.map((tag) => (
                <span
                  key={tag}
                  className="inline-flex items-center px-2 py-0.5 rounded text-xs bg-gray-100 text-gray-600"
                >
                  {tag}
                </span>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Result message */}
      {resultMessage && (
        <div
          className={`mt-3 p-2 rounded text-sm ${
            resultMessage.type === 'success'
              ? 'bg-green-100 text-green-800'
              : 'bg-red-100 text-red-800'
          }`}
        >
          {resultMessage.text}
        </div>
      )}

      {/* Install/Uninstall buttons */}
      {isAdmin && plugin.is_compatible && (
        <div className="mt-3 flex items-center gap-2">
          {!plugin.is_installed && !plugin.pending_restart && (
            <button
              onClick={handleInstall}
              disabled={isLoading}
              className={`
                inline-flex items-center px-3 py-1.5 rounded-md text-sm font-medium
                ${
                  isLoading
                    ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                    : 'bg-blue-600 text-white hover:bg-blue-700'
                }
              `}
            >
              {installMutation.isPending ? (
                <>
                  <svg
                    className="animate-spin -ml-0.5 mr-2 h-4 w-4"
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
                  Installing...
                </>
              ) : (
                <>
                  <svg
                    className="w-4 h-4 mr-1.5"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
                    />
                  </svg>
                  Install
                </>
              )}
            </button>
          )}

          {(plugin.is_installed || plugin.pending_restart) && (
            <button
              onClick={handleUninstall}
              disabled={isLoading}
              className={`
                inline-flex items-center px-3 py-1.5 rounded-md text-sm font-medium
                ${
                  isLoading
                    ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                    : 'bg-red-100 text-red-700 hover:bg-red-200'
                }
              `}
            >
              {uninstallMutation.isPending ? (
                <>
                  <svg
                    className="animate-spin -ml-0.5 mr-2 h-4 w-4"
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
                  Uninstalling...
                </>
              ) : (
                <>
                  <svg
                    className="w-4 h-4 mr-1.5"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                    />
                  </svg>
                  Uninstall
                </>
              )}
            </button>
          )}
        </div>
      )}

      {/* Non-admin message */}
      {!isAdmin && plugin.is_compatible && !plugin.is_installed && !plugin.pending_restart && (
        <div className="mt-3 text-xs text-gray-500 italic">
          Admin access required to install plugins
        </div>
      )}
    </div>
  );
}

export default AvailablePluginCard;
