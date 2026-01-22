import { useState } from 'react';
import { useApiKeys, useCreateApiKey } from '../../hooks/useApiKeys';
import type { ApiKeyResponse } from '../../types/api-key';

interface ApiKeySelectorProps {
  profileName: string;
  onKeySelected: (apiKey: string) => void;
}

/**
 * Filters API keys to only include active and non-expired keys.
 */
function filterActiveKeys(keys: ApiKeyResponse[] | undefined): ApiKeyResponse[] {
  if (!keys) return [];

  const now = new Date();
  return keys.filter((key) => {
    // Must be active
    if (!key.is_active) return false;

    // Must not be expired
    if (key.expires_at && new Date(key.expires_at) < now) return false;

    return true;
  });
}

const CREATE_NEW_VALUE = '__create_new__';

export default function ApiKeySelector({
  profileName,
  onKeySelected,
}: ApiKeySelectorProps) {
  const { data: apiKeys, isLoading, refetch } = useApiKeys();
  const createApiKey = useCreateApiKey();
  const [selectedValue, setSelectedValue] = useState<string>('');
  const [error, setError] = useState<string | null>(null);

  const activeKeys = filterActiveKeys(apiKeys);

  const handleChange = async (e: React.ChangeEvent<HTMLSelectElement>) => {
    const value = e.target.value;
    setSelectedValue(value);
    setError(null);

    if (value === CREATE_NEW_VALUE) {
      try {
        const result = await createApiKey.mutateAsync({
          name: `MCP: ${profileName}`,
          expires_in_days: null,
        });
        onKeySelected(result.api_key);
      } catch (err) {
        setError('Failed to create API key');
        setSelectedValue('');
        console.error('Failed to create API key:', err);
      }
    } else if (value) {
      // For existing keys, the user selected an existing key by ID
      // Since API keys are shown only once at creation, we can't get the actual key value
      // The user needs to use the original key they created
      // For now, we just mark it as selected but inform via the existing key note below
    }
  };

  const isCreating = createApiKey.isPending;

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 text-sm text-[var(--text-muted)]">
        <svg
          className="animate-spin h-4 w-4"
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
        <span>Loading API keys...</span>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <label className="block text-sm font-medium text-[var(--text-secondary)]">
          API Key
        </label>
        <button
          type="button"
          onClick={() => refetch()}
          aria-label="Refresh API keys"
          className="p-1 text-[var(--text-muted)] hover:text-[var(--text-primary)] focus:outline-none focus:ring-2 focus:ring-gray-400 dark:focus:ring-white rounded transition-colors"
        >
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
              d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
            />
          </svg>
        </button>
      </div>

      <div className="flex items-center gap-2">
        <select
          value={selectedValue}
          onChange={handleChange}
          disabled={isCreating}
          className="flex-1 px-3 py-2 bg-[var(--bg-tertiary)] border border-[var(--border)] rounded-lg text-[var(--text-primary)] focus:outline-none focus:ring-2 focus:ring-gray-400 dark:focus:ring-white disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <option value="">Select an API key...</option>
          {activeKeys.map((key) => (
            <option key={key.id} value={key.id}>
              {key.name}
            </option>
          ))}
          {activeKeys.length > 0 && (
            <option disabled>────────────</option>
          )}
          <option value={CREATE_NEW_VALUE}>+ Create new key</option>
        </select>

        {isCreating && (
          <div className="flex items-center gap-2 text-sm text-[var(--text-muted)]">
            <svg
              className="animate-spin h-4 w-4"
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
            <span>Creating...</span>
          </div>
        )}
      </div>

      {error && (
        <p className="text-sm text-red-500">{error}</p>
      )}

      {selectedValue && selectedValue !== CREATE_NEW_VALUE && (
        <p className="text-xs text-[var(--text-muted)]">
          Note: For existing keys, you'll need to use the original key value from when it was created.
          Consider creating a new key for this MCP profile.
        </p>
      )}
    </div>
  );
}
