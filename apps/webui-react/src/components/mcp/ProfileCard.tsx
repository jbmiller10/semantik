import { useState } from 'react';
import type { MCPProfile } from '../../types/mcp-profile';
import { SEARCH_TYPE_LABELS, SEARCH_MODE_LABELS } from '../../types/mcp-profile';
import { useToggleMCPProfileEnabled } from '../../hooks/useMCPProfiles';

interface ProfileCardProps {
  profile: MCPProfile;
  onEdit: () => void;
  onDelete: () => void;
  onViewConfig: () => void;
}

export default function ProfileCard({
  profile,
  onEdit,
  onDelete,
  onViewConfig,
}: ProfileCardProps) {
  const toggleEnabled = useToggleMCPProfileEnabled();
  const [isToggling, setIsToggling] = useState(false);
  const [toggleErrorMessage, setToggleErrorMessage] = useState<string | null>(null);

  const handleToggleEnabled = async () => {
    setIsToggling(true);
    setToggleErrorMessage(null);
    try {
      await toggleEnabled.mutateAsync({
        profileId: profile.id,
        enabled: !profile.enabled,
        profileName: profile.name,
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Toggle failed';
      console.error('Profile toggle failed:', error);
      setToggleErrorMessage(message);
      // Clear error after 5 seconds
      setTimeout(() => setToggleErrorMessage(null), 5000);
    } finally {
      setIsToggling(false);
    }
  };

  return (
    <div className="bg-white border border-gray-200 rounded-lg shadow-sm hover:shadow-md transition-shadow">
      {/* Card Header */}
      <div className="p-4 border-b border-gray-100">
        <div className="flex items-start justify-between">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <h3 className="text-lg font-semibold text-gray-900 truncate">
                {profile.name}
              </h3>
              <span
                className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${
                  profile.enabled
                    ? 'bg-green-100 text-green-800'
                    : 'bg-gray-100 text-gray-600'
                }`}
              >
                {profile.enabled ? 'Enabled' : 'Disabled'}
              </span>
            </div>
            <p className="mt-1 text-sm text-gray-500 line-clamp-2">
              {profile.description}
            </p>
          </div>

          {/* Enable/Disable Toggle */}
          <div className="flex items-center ml-4">
            {toggleErrorMessage && (
              <span className="text-xs text-red-500 mr-2 max-w-32 truncate" title={toggleErrorMessage}>
                {toggleErrorMessage}
              </span>
            )}
            <button
              onClick={handleToggleEnabled}
              disabled={isToggling}
              className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 ${
                profile.enabled ? 'bg-blue-600' : 'bg-gray-200'
              } ${isToggling ? 'opacity-50 cursor-not-allowed' : ''} ${toggleErrorMessage ? 'ring-2 ring-red-300' : ''}`}
              role="switch"
              aria-checked={profile.enabled}
              aria-label={`${profile.enabled ? 'Disable' : 'Enable'} profile`}
            >
              <span
                className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
                  profile.enabled ? 'translate-x-5' : 'translate-x-0'
                }`}
              />
            </button>
          </div>
        </div>
      </div>

      {/* Card Body */}
      <div className="p-4 space-y-3">
        {/* Collections */}
        <div>
          <span className="text-xs font-medium text-gray-500 uppercase tracking-wider">
            Collections
          </span>
          <div className="mt-1 flex flex-wrap gap-1">
            {profile.collections.length === 0 ? (
              <span className="text-sm text-gray-400 italic">
                No collections
              </span>
            ) : (
              profile.collections.map((collection) => (
                <span
                  key={collection.id}
                  className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800"
                >
                  {collection.name}
                </span>
              ))
            )}
          </div>
        </div>

        {/* Search Settings */}
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-gray-500">Search Type:</span>{' '}
            <span className="font-medium text-gray-900">
              {SEARCH_TYPE_LABELS[profile.search_type]}
            </span>
          </div>
          <div>
            <span className="text-gray-500">Results:</span>{' '}
            <span className="font-medium text-gray-900">
              {profile.result_count}
            </span>
          </div>
          <div>
            <span className="text-gray-500">Reranker:</span>{' '}
            <span className="font-medium text-gray-900">
              {profile.use_reranker ? 'Yes' : 'No'}
            </span>
          </div>
          <div>
            <span className="text-gray-500">Mode:</span>{' '}
            <span className="font-medium text-gray-900">
              {SEARCH_MODE_LABELS[profile.search_mode]}
            </span>
          </div>
          {profile.search_type === 'hybrid' && profile.hybrid_alpha !== null && (
            <div>
              <span className="text-gray-500">Hybrid Alpha:</span>{' '}
              <span className="font-medium text-gray-900">
                {profile.hybrid_alpha}
              </span>
            </div>
          )}
          {profile.search_mode === 'hybrid' && profile.rrf_k !== null && (
            <div>
              <span className="text-gray-500">RRF k:</span>{' '}
              <span className="font-medium text-gray-900">
                {profile.rrf_k}
              </span>
            </div>
          )}
        </div>

        {/* Tool Name */}
        <div className="pt-2 border-t border-gray-100">
          <span className="text-xs font-medium text-gray-500">MCP Tool Name</span>
          <code className="mt-1 block text-sm font-mono bg-gray-50 px-2 py-1 rounded text-gray-700">
            search_{profile.name}
          </code>
        </div>
      </div>

      {/* Card Footer - Actions */}
      <div className="px-4 py-3 bg-gray-50 border-t border-gray-100 rounded-b-lg flex justify-between items-center">
        <div className="flex gap-2">
          <button
            onClick={onEdit}
            className="inline-flex items-center px-3 py-1.5 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-1"
          >
            <svg
              className="w-4 h-4 mr-1"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"
              />
            </svg>
            Edit
          </button>
          <button
            onClick={onViewConfig}
            className="inline-flex items-center px-3 py-1.5 text-sm font-medium text-blue-700 bg-blue-50 border border-blue-200 rounded-md hover:bg-blue-100 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-1"
          >
            <svg
              className="w-4 h-4 mr-1"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"
              />
            </svg>
            Connection Info
          </button>
        </div>
        <button
          onClick={onDelete}
          className="inline-flex items-center px-3 py-1.5 text-sm font-medium text-red-600 hover:text-red-700 hover:bg-red-50 rounded-md focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-1"
        >
          <svg
            className="w-4 h-4 mr-1"
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
          Delete
        </button>
      </div>
    </div>
  );
}
