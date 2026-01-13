/**
 * Resource Limits Settings component.
 * Admin-only settings for per-user quotas and limits.
 */
import { useState, useEffect, useCallback } from 'react';
import { useEffectiveSettings, useUpdateSystemSettings, useResetSettingsToDefaults } from '../../hooks/useSystemSettings';
import { getInputClassName } from '../../utils/formStyles';
import { extractResourceLimits } from '../../types/system-settings';
import type { ResourceLimitsSettings as ResourceLimitsSettingsType } from '../../types/system-settings';

const RESOURCE_LIMIT_KEYS = [
  'max_collections_per_user',
  'max_storage_gb_per_user',
  'max_document_size_mb',
  'max_artifact_size_mb',
];

export default function ResourceLimitsSettings() {
  const { data: effectiveSettings, isLoading, error } = useEffectiveSettings();
  const updateMutation = useUpdateSystemSettings();
  const resetMutation = useResetSettingsToDefaults();

  const [formState, setFormState] = useState<ResourceLimitsSettingsType>({
    max_collections_per_user: 10,
    max_storage_gb_per_user: 50,
    max_document_size_mb: 100,
    max_artifact_size_mb: 50,
  });

  // Initialize form state from effective settings
  useEffect(() => {
    if (effectiveSettings?.settings) {
      setFormState(extractResourceLimits(effectiveSettings.settings));
    }
  }, [effectiveSettings]);

  const handleChange = useCallback(
    <K extends keyof ResourceLimitsSettingsType>(field: K, value: ResourceLimitsSettingsType[K]) => {
      setFormState((prev) => ({ ...prev, [field]: value }));
    },
    []
  );

  const handleSave = useCallback(async () => {
    await updateMutation.mutateAsync({
      settings: {
        max_collections_per_user: formState.max_collections_per_user,
        max_storage_gb_per_user: formState.max_storage_gb_per_user,
        max_document_size_mb: formState.max_document_size_mb,
        max_artifact_size_mb: formState.max_artifact_size_mb,
      },
    });
  }, [formState, updateMutation]);

  const handleReset = useCallback(async () => {
    await resetMutation.mutateAsync(RESOURCE_LIMIT_KEYS);
  }, [resetMutation]);

  // Loading state
  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-8">
        <svg className="animate-spin h-6 w-6 text-gray-400" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
        </svg>
        <span className="ml-2 text-gray-500">Loading resource limits...</span>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <div className="flex">
          <svg className="h-5 w-5 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <div className="ml-3">
            <h3 className="text-sm font-medium text-red-800">Error loading settings</h3>
            <p className="mt-1 text-sm text-red-700">{error.message}</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Info box */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <div className="flex">
          <svg className="h-5 w-5 text-blue-400 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <div className="ml-3">
            <p className="text-sm text-blue-700">
              These limits apply to all users. Changes take effect immediately for new operations.
            </p>
          </div>
        </div>
      </div>

      {/* Settings Form */}
      <div className="space-y-4">
        {/* Max Collections */}
        <div>
          <label className="block text-sm font-medium text-gray-700">
            Max Collections per User
          </label>
          <input
            type="number"
            min={1}
            max={1000}
            value={formState.max_collections_per_user}
            onChange={(e) => handleChange('max_collections_per_user', parseInt(e.target.value, 10) || 10)}
            className={getInputClassName(false, false)}
          />
          <p className="mt-1 text-xs text-gray-500">
            Maximum number of collections a user can create (1-1000)
          </p>
        </div>

        {/* Max Storage */}
        <div>
          <label className="block text-sm font-medium text-gray-700">
            Max Storage per User (GB)
          </label>
          <input
            type="number"
            min={1}
            max={10000}
            value={formState.max_storage_gb_per_user}
            onChange={(e) => handleChange('max_storage_gb_per_user', parseInt(e.target.value, 10) || 50)}
            className={getInputClassName(false, false)}
          />
          <p className="mt-1 text-xs text-gray-500">
            Maximum storage in gigabytes per user (1-10000)
          </p>
        </div>

        {/* Max Document Size */}
        <div>
          <label className="block text-sm font-medium text-gray-700">
            Max Document Size (MB)
          </label>
          <input
            type="number"
            min={1}
            max={1000}
            value={formState.max_document_size_mb}
            onChange={(e) => handleChange('max_document_size_mb', parseInt(e.target.value, 10) || 100)}
            className={getInputClassName(false, false)}
          />
          <p className="mt-1 text-xs text-gray-500">
            Maximum size of a single document in megabytes (1-1000)
          </p>
        </div>

        {/* Max Artifact Size */}
        <div>
          <label className="block text-sm font-medium text-gray-700">
            Max Artifact Size (MB)
          </label>
          <input
            type="number"
            min={1}
            max={500}
            value={formState.max_artifact_size_mb}
            onChange={(e) => handleChange('max_artifact_size_mb', parseInt(e.target.value, 10) || 50)}
            className={getInputClassName(false, false)}
          />
          <p className="mt-1 text-xs text-gray-500">
            Maximum size of generated artifacts (parquet files) in megabytes (1-500)
          </p>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex justify-between pt-4 border-t border-gray-200">
        <button
          type="button"
          onClick={handleReset}
          disabled={resetMutation.isPending}
          className="inline-flex items-center px-4 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {resetMutation.isPending ? 'Resetting...' : 'Reset to Defaults'}
        </button>
        <button
          type="button"
          onClick={handleSave}
          disabled={updateMutation.isPending}
          className="inline-flex items-center px-4 py-2 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {updateMutation.isPending ? (
            <>
              <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
              Saving...
            </>
          ) : (
            'Save Settings'
          )}
        </button>
      </div>
    </div>
  );
}
