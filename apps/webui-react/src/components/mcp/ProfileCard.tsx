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
    <div className="bg-[var(--bg-secondary)] border border-[var(--border)] rounded-lg shadow-sm hover:shadow-md transition-shadow">
      {/* Card Header */}
      <div className="p-4 border-b border-[var(--border)]">
        <div className="flex items-start justify-between">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <h3 className="text-lg font-semibold text-[var(--text-primary)] truncate">
                {profile.name}
              </h3>
              <span
                className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${
                  profile.enabled
                    ? 'bg-green-500/20 text-green-400'
                    : 'bg-gray-500/20 text-gray-400'
                }`}
              >
                {profile.enabled ? 'Enabled' : 'Disabled'}
              </span>
            </div>
            <p className="mt-1 text-sm text-[var(--text-secondary)] line-clamp-2">
              {profile.description}
            </p>
          </div>

          {/* Enable/Disable Toggle */}
          <div className="flex items-center ml-4">
            {toggleErrorMessage && (
              <span className="text-xs text-red-400 mr-2 max-w-32 truncate" title={toggleErrorMessage}>
                {toggleErrorMessage}
              </span>
            )}
            <button
              onClick={handleToggleEnabled}
              disabled={isToggling}
              className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-gray-400 dark:focus:ring-white focus:ring-offset-2 focus:ring-offset-[var(--bg-primary)] ${
                profile.enabled ? 'bg-gray-600 dark:bg-white' : 'bg-gray-300 dark:bg-gray-600'
              } ${isToggling ? 'opacity-50 cursor-not-allowed' : ''} ${toggleErrorMessage ? 'ring-2 ring-red-400/50' : ''}`}
              role="switch"
              aria-checked={profile.enabled}
              aria-label={`${profile.enabled ? 'Disable' : 'Enable'} profile`}
            >
              <span
                className={`pointer-events-none inline-block h-5 w-5 transform rounded-full shadow ring-0 transition duration-200 ease-in-out ${
                  profile.enabled ? 'translate-x-5 bg-white dark:bg-gray-800' : 'translate-x-0 bg-white'
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
          <span className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">
            Collections
          </span>
          <div className="mt-1 flex flex-wrap gap-1">
            {profile.collections.length === 0 ? (
              <span className="text-sm text-[var(--text-muted)] italic">
                No collections
              </span>
            ) : (
              profile.collections.map((collection) => (
                <span
                  key={collection.id}
                  className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-gray-200 dark:bg-white/20 text-gray-700 dark:text-gray-200"
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
            <span className="text-[var(--text-secondary)]">Search Type:</span>{' '}
            <span className="font-medium text-[var(--text-primary)]">
              {SEARCH_TYPE_LABELS[profile.search_type]}
            </span>
          </div>
          <div>
            <span className="text-[var(--text-secondary)]">Results:</span>{' '}
            <span className="font-medium text-[var(--text-primary)]">
              {profile.result_count}
            </span>
          </div>
          <div>
            <span className="text-[var(--text-secondary)]">Reranker:</span>{' '}
            <span className="font-medium text-[var(--text-primary)]">
              {profile.use_reranker ? 'Yes' : 'No'}
            </span>
          </div>
          <div>
            <span className="text-[var(--text-secondary)]">Mode:</span>{' '}
            <span className="font-medium text-[var(--text-primary)]">
              {SEARCH_MODE_LABELS[profile.search_mode]}
            </span>
          </div>
          {profile.search_type === 'hybrid' && profile.hybrid_alpha !== null && (
            <div>
              <span className="text-[var(--text-secondary)]">Hybrid Alpha:</span>{' '}
              <span className="font-medium text-[var(--text-primary)]">
                {profile.hybrid_alpha}
              </span>
            </div>
          )}
          {profile.search_mode === 'hybrid' && profile.rrf_k !== null && (
            <div>
              <span className="text-[var(--text-secondary)]">RRF k:</span>{' '}
              <span className="font-medium text-[var(--text-primary)]">
                {profile.rrf_k}
              </span>
            </div>
          )}
        </div>

        {/* Tool Name */}
        <div className="pt-2 border-t border-[var(--border)]">
          <span className="text-xs font-medium text-[var(--text-muted)]">MCP Tool Name</span>
          <code className="mt-1 block text-sm font-mono bg-[var(--bg-tertiary)] px-2 py-1 rounded text-[var(--text-primary)]">
            search_{profile.name}
          </code>
        </div>
      </div>

      {/* Card Footer - Actions */}
      <div className="px-4 py-3 bg-[var(--bg-tertiary)] border-t border-[var(--border)] rounded-b-lg flex justify-between items-center">
        <div className="flex gap-2">
          <button
            onClick={onEdit}
            className="inline-flex items-center px-3 py-1.5 text-sm font-medium text-[var(--text-secondary)] bg-[var(--bg-secondary)] border border-[var(--border)] rounded-md hover:bg-[var(--bg-primary)] focus:outline-none focus:ring-2 focus:ring-gray-400 dark:focus:ring-white focus:ring-offset-1 focus:ring-offset-[var(--bg-tertiary)]"
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
            className="inline-flex items-center px-3 py-1.5 text-sm font-medium text-gray-700 dark:text-gray-200 bg-gray-100 dark:bg-white/10 border border-gray-300 dark:border-white/20 rounded-md hover:bg-gray-200 dark:hover:bg-white/20 focus:outline-none focus:ring-2 focus:ring-gray-400 dark:focus:ring-white focus:ring-offset-1 focus:ring-offset-[var(--bg-tertiary)]"
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
          className="inline-flex items-center px-3 py-1.5 text-sm font-medium text-red-400 hover:text-red-300 hover:bg-red-500/10 rounded-md focus:outline-none focus:ring-2 focus:ring-red-400 focus:ring-offset-1 focus:ring-offset-[var(--bg-tertiary)]"
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
