import { useState } from 'react';
import { useApiKeys } from '../../hooks/useApiKeys';
import type { ApiKeyCreateResponse } from '../../types/api-key';
import ApiKeyCard from './api-keys/ApiKeyCard';
import CreateApiKeyModal from './api-keys/CreateApiKeyModal';
import ApiKeyCreatedModal from './api-keys/ApiKeyCreatedModal';

export default function ApiKeysSettings() {
  const { data: apiKeys, isLoading, error, refetch } = useApiKeys();

  // Modal state
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [createdKey, setCreatedKey] = useState<ApiKeyCreateResponse | null>(null);

  // Handle successful key creation
  const handleCreateSuccess = (response: ApiKeyCreateResponse) => {
    setShowCreateModal(false);
    setCreatedKey(response);
  };

  // Loading state
  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <svg
          className="animate-spin h-8 w-8 text-[var(--text-muted)]"
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
        <span className="ml-3 text-[var(--text-secondary)]">Loading API keys...</span>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4">
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
            <h3 className="text-sm font-medium text-red-300">
              Error loading API keys
            </h3>
            <p className="mt-1 text-sm text-red-400">
              {error instanceof Error ? error.message : 'Unknown error occurred'}
            </p>
            <button
              onClick={() => refetch()}
              className="mt-2 text-sm font-medium text-red-400 hover:text-red-300"
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
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-medium text-[var(--text-primary)]">API Keys</h3>
          <p className="mt-1 text-sm text-[var(--text-secondary)]">
            Manage API keys for programmatic access
          </p>
        </div>
        <button
          onClick={() => setShowCreateModal(true)}
          className="inline-flex items-center px-4 py-2 text-sm font-medium text-gray-900 dark:text-gray-900 bg-gray-200 dark:bg-white border border-transparent rounded-md hover:bg-gray-300 dark:hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 focus:ring-offset-[var(--bg-primary)]"
        >
          <svg
            className="w-4 h-4 mr-2"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 4v16m8-8H4"
            />
          </svg>
          Create API Key
        </button>
      </div>

      {/* Info Box */}
      <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
        <div className="flex">
          <svg
            className="h-5 w-5 text-blue-400 flex-shrink-0"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
          <div className="ml-3">
            <h3 className="text-sm font-medium text-blue-300">
              What are API Keys?
            </h3>
            <p className="mt-1 text-sm text-blue-400">
              API keys provide secure, long-lived access to the Semantik API without
              requiring user credentials. Use them for MCP servers, automation scripts,
              and other programmatic integrations. Keys can be revoked at any time.
            </p>
          </div>
        </div>
      </div>

      {/* Empty State */}
      {(!apiKeys || apiKeys.length === 0) && (
        <div className="text-center py-12 bg-[var(--bg-tertiary)] rounded-lg border-2 border-dashed border-[var(--border)]">
          <svg
            className="mx-auto h-12 w-12 text-[var(--text-muted)]"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z"
            />
          </svg>
          <h3 className="mt-2 text-sm font-medium text-[var(--text-primary)]">
            No API keys
          </h3>
          <p className="mt-1 text-sm text-[var(--text-secondary)]">
            Create an API key to enable programmatic access to your collections.
          </p>
          <button
            onClick={() => setShowCreateModal(true)}
            className="mt-4 inline-flex items-center px-4 py-2 text-sm font-medium text-gray-900 dark:text-gray-900 bg-gray-200 dark:bg-white border border-transparent rounded-md hover:bg-gray-300 dark:hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 focus:ring-offset-[var(--bg-primary)]"
          >
            <svg
              className="w-4 h-4 mr-2"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 4v16m8-8H4"
              />
            </svg>
            Create API Key
          </button>
        </div>
      )}

      {/* API Key List */}
      {apiKeys && apiKeys.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {apiKeys.map((key) => (
            <ApiKeyCard key={key.id} apiKey={key} />
          ))}
        </div>
      )}

      {/* Modals */}
      {showCreateModal && (
        <CreateApiKeyModal
          onClose={() => setShowCreateModal(false)}
          onSuccess={handleCreateSuccess}
        />
      )}

      {createdKey && (
        <ApiKeyCreatedModal
          apiKey={createdKey}
          onClose={() => setCreatedKey(null)}
        />
      )}
    </div>
  );
}
