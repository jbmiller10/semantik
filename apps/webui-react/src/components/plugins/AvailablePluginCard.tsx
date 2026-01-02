import { useState } from 'react';
import type { AvailablePlugin } from '../../types/plugin';

interface AvailablePluginCardProps {
  plugin: AvailablePlugin;
}

/**
 * Card component for displaying an available (not installed) plugin from the registry.
 * Shows plugin info, verification status, compatibility, and install instructions.
 */
function AvailablePluginCard({ plugin }: AvailablePluginCardProps) {
  const [copied, setCopied] = useState(false);

  const handleCopyInstallCommand = async () => {
    try {
      await navigator.clipboard.writeText(plugin.install_command);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback: ignore if clipboard not available
    }
  };

  return (
    <div
      className={`
        bg-white border rounded-lg p-4 transition-shadow
        ${plugin.is_installed ? 'border-green-200 bg-green-50' : 'border-gray-200 hover:shadow-sm'}
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

      {/* Install command */}
      {!plugin.is_installed && plugin.is_compatible && (
        <div className="mt-3 flex items-center gap-2">
          <div className="flex-1 flex items-center gap-2 bg-gray-100 rounded-md px-3 py-1.5">
            <code className="text-xs text-gray-700 font-mono truncate flex-1">
              {plugin.install_command}
            </code>
            <button
              onClick={handleCopyInstallCommand}
              className="text-gray-400 hover:text-gray-600 flex-shrink-0"
              title="Copy to clipboard"
            >
              {copied ? (
                <svg
                  className="w-4 h-4 text-green-500"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M5 13l4 4L19 7"
                  />
                </svg>
              ) : (
                <svg
                  className="w-4 h-4"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3"
                  />
                </svg>
              )}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default AvailablePluginCard;
