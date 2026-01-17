import { useState } from 'react';
import type { ApiKeyResponse } from '../../../types/api-key';
import {
  getApiKeyStatus,
  formatRelativeTime,
  truncateKeyId,
} from '../../../types/api-key';
import { useRevokeApiKey } from '../../../hooks/useApiKeys';

interface ApiKeyCardProps {
  apiKey: ApiKeyResponse;
}

export default function ApiKeyCard({ apiKey }: ApiKeyCardProps) {
  const revokeKey = useRevokeApiKey();
  const [isToggling, setIsToggling] = useState(false);

  const status = getApiKeyStatus(apiKey);

  const handleToggle = async () => {
    setIsToggling(true);
    try {
      await revokeKey.mutateAsync({
        keyId: apiKey.id,
        isActive: !apiKey.is_active,
        keyName: apiKey.name,
      });
    } catch (error) {
      console.error('Failed to toggle API key:', error);
    } finally {
      setIsToggling(false);
    }
  };

  // Status badge styling based on status
  const statusBadgeStyles = {
    active: 'bg-green-500/20 text-green-400',
    disabled: 'bg-gray-500/20 text-gray-400',
    expired: 'bg-amber-500/20 text-amber-400',
  };

  const statusLabels = {
    active: 'Active',
    disabled: 'Disabled',
    expired: 'Expired',
  };

  return (
    <div className="bg-[var(--bg-secondary)] border border-[var(--border)] rounded-lg shadow-sm hover:shadow-md transition-shadow">
      {/* Card Header */}
      <div className="p-4 border-b border-[var(--border)]">
        <div className="flex items-start justify-between">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <h3 className="text-lg font-semibold text-[var(--text-primary)] truncate">
                {apiKey.name}
              </h3>
              <span
                className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${statusBadgeStyles[status]}`}
              >
                {statusLabels[status]}
              </span>
            </div>
            <code className="mt-1 text-sm font-mono text-[var(--text-secondary)]">
              {truncateKeyId(apiKey.id)}
            </code>
          </div>

          {/* Toggle Button (only show if not expired) */}
          {status !== 'expired' && (
            <button
              onClick={handleToggle}
              disabled={isToggling}
              className={`px-3 py-1.5 text-sm font-medium rounded-md transition-colors focus:outline-none focus:ring-2 focus:ring-gray-400 dark:focus:ring-white focus:ring-offset-1 focus:ring-offset-[var(--bg-secondary)] ${
                apiKey.is_active
                  ? 'text-red-400 hover:text-red-300 hover:bg-red-500/10'
                  : 'text-green-400 hover:text-green-300 hover:bg-green-500/10'
              } ${isToggling ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
              {isToggling ? (
                <span className="flex items-center">
                  <svg
                    className="animate-spin h-4 w-4 mr-1"
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
                  ...
                </span>
              ) : apiKey.is_active ? (
                'Revoke'
              ) : (
                'Reactivate'
              )}
            </button>
          )}
        </div>
      </div>

      {/* Card Body - Metadata */}
      <div className="p-4 space-y-2 text-sm">
        <div className="flex justify-between">
          <span className="text-[var(--text-secondary)]">Created</span>
          <span className="text-[var(--text-primary)]">
            {formatRelativeTime(apiKey.created_at)}
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-[var(--text-secondary)]">Last used</span>
          <span className="text-[var(--text-primary)]">
            {apiKey.last_used_at ? formatRelativeTime(apiKey.last_used_at) : 'Never'}
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-[var(--text-secondary)]">Expires</span>
          <span
            className={`${
              status === 'expired' ? 'text-amber-400' : 'text-[var(--text-primary)]'
            }`}
          >
            {apiKey.expires_at ? formatRelativeTime(apiKey.expires_at) : 'Never'}
          </span>
        </div>
      </div>
    </div>
  );
}
