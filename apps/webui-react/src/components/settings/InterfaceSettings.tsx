/**
 * Interface Settings component.
 * Allows users to configure UI behavior preferences.
 */
import { useState, useEffect, useCallback } from 'react';
import {
  usePreferences,
  useUpdatePreferences,
  useResetInterfacePreferences,
} from '../../hooks/usePreferences';

interface InterfaceFormState {
  data_refresh_interval_ms: number;
  visualization_sample_limit: number;
  animation_enabled: boolean;
}

const DEFAULT_FORM_STATE: InterfaceFormState = {
  data_refresh_interval_ms: 30000,
  visualization_sample_limit: 200000,
  animation_enabled: true,
};

/** Convert ms to seconds for display */
const msToSeconds = (ms: number) => Math.round(ms / 1000);

/** Convert seconds to ms for storage */
const secondsToMs = (s: number) => s * 1000;

/** Format sample limit for display (e.g., 200000 -> "200K") */
const formatSampleLimit = (n: number) => {
  if (n >= 1000) {
    return `${Math.round(n / 1000)}K`;
  }
  return n.toString();
};

export default function InterfaceSettings() {
  const { data: preferences, isLoading, error } = usePreferences();
  const updateMutation = useUpdatePreferences();
  const resetMutation = useResetInterfacePreferences();

  const [formState, setFormState] = useState<InterfaceFormState>(DEFAULT_FORM_STATE);

  // Initialize form state from preferences
  useEffect(() => {
    if (preferences?.interface) {
      setFormState({
        data_refresh_interval_ms: preferences.interface.data_refresh_interval_ms,
        visualization_sample_limit: preferences.interface.visualization_sample_limit,
        animation_enabled: preferences.interface.animation_enabled,
      });
    }
  }, [preferences]);

  const handleChange = useCallback(
    <K extends keyof InterfaceFormState>(field: K, value: InterfaceFormState[K]) => {
      setFormState((prev) => ({ ...prev, [field]: value }));
    },
    []
  );

  const handleSave = useCallback(async () => {
    await updateMutation.mutateAsync({
      interface: {
        data_refresh_interval_ms: formState.data_refresh_interval_ms,
        visualization_sample_limit: formState.visualization_sample_limit,
        animation_enabled: formState.animation_enabled,
      },
    });
  }, [formState, updateMutation]);

  const handleReset = useCallback(async () => {
    await resetMutation.mutateAsync();
  }, [resetMutation]);

  // Loading state
  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <svg className="animate-spin h-8 w-8 text-[var(--text-muted)]" fill="none" viewBox="0 0 24 24">
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
        <span className="ml-3 text-[var(--text-secondary)]">Loading interface preferences...</span>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <div className="flex">
          <svg className="h-5 w-5 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
          <div className="ml-3">
            <h3 className="text-sm font-medium text-red-800">Error loading preferences</h3>
            <p className="mt-1 text-sm text-red-700">{error.message}</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h3 className="text-lg leading-6 font-medium text-[var(--text-primary)]">Interface Preferences</h3>
        <p className="mt-1 text-sm text-[var(--text-secondary)]">
          Configure UI behavior settings like data refresh intervals and visualizations.
        </p>
      </div>

      {/* Info box */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
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
            <p className="text-sm text-blue-700">
              These settings control how the UI behaves. Changes take effect immediately after saving.
            </p>
          </div>
        </div>
      </div>

      {/* Interface Settings Form */}
      <div className="bg-[var(--bg-secondary)] shadow rounded-lg border border-[var(--border)]">
        <div className="px-4 py-5 sm:p-6">
          <div className="space-y-6">
            {/* Data Refresh Interval */}
            <div>
              <label className="block text-sm font-medium text-[var(--text-primary)]">
                Data Refresh Interval
              </label>
              <div className="mt-2">
                <input
                  type="range"
                  min={10}
                  max={60}
                  step={5}
                  value={msToSeconds(formState.data_refresh_interval_ms)}
                  onChange={(e) =>
                    handleChange('data_refresh_interval_ms', secondsToMs(parseInt(e.target.value, 10)))
                  }
                  className="w-full h-2 bg-[var(--bg-tertiary)] rounded-lg appearance-none cursor-pointer"
                />
                <div className="flex justify-between text-xs text-[var(--text-secondary)] mt-1">
                  <span>10s</span>
                  <span className="font-medium text-[var(--text-primary)]">
                    {msToSeconds(formState.data_refresh_interval_ms)}s
                  </span>
                  <span>60s</span>
                </div>
              </div>
              <p className="mt-1 text-xs text-[var(--text-secondary)]">
                How often to automatically refresh data from the server (10-60 seconds)
              </p>
            </div>

            {/* Visualization Sample Limit */}
            <div>
              <label className="block text-sm font-medium text-[var(--text-primary)]">
                Visualization Sample Limit
              </label>
              <div className="mt-2">
                <input
                  type="range"
                  min={10000}
                  max={500000}
                  step={10000}
                  value={formState.visualization_sample_limit}
                  onChange={(e) =>
                    handleChange('visualization_sample_limit', parseInt(e.target.value, 10))
                  }
                  className="w-full h-2 bg-[var(--bg-tertiary)] rounded-lg appearance-none cursor-pointer"
                />
                <div className="flex justify-between text-xs text-[var(--text-secondary)] mt-1">
                  <span>10K</span>
                  <span className="font-medium text-[var(--text-primary)]">
                    {formatSampleLimit(formState.visualization_sample_limit)}
                  </span>
                  <span>500K</span>
                </div>
              </div>
              <p className="mt-1 text-xs text-[var(--text-secondary)]">
                Maximum points to display in UMAP/PCA visualizations. Higher values show more detail
                but may impact performance.
              </p>
            </div>

            {/* Animation Enabled */}
            <div className="flex items-start">
              <div className="flex items-center h-5">
                <input
                  type="checkbox"
                  checked={formState.animation_enabled}
                  onChange={(e) => handleChange('animation_enabled', e.target.checked)}
                  className="h-4 w-4 text-[var(--accent-primary)] border-[var(--border)] rounded focus:ring-[var(--accent-primary)]"
                />
              </div>
              <div className="ml-3 text-sm">
                <label className="font-medium text-[var(--text-primary)]">Enable Animations</label>
                <p className="text-[var(--text-secondary)]">
                  Show smooth transitions and animations throughout the interface. Disable for better
                  performance on slower devices.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex justify-between">
        <button
          type="button"
          onClick={handleReset}
          disabled={resetMutation.isPending}
          className="inline-flex items-center px-4 py-2 border border-[var(--border)] shadow-sm text-sm font-medium rounded-md text-[var(--text-primary)] bg-[var(--bg-secondary)] hover:bg-[var(--bg-tertiary)] focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-[var(--accent-primary)] disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {resetMutation.isPending ? 'Resetting...' : 'Reset to Defaults'}
        </button>
        <button
          type="button"
          onClick={handleSave}
          disabled={updateMutation.isPending}
          className="inline-flex items-center px-4 py-2 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-[var(--accent-primary)] hover:opacity-90 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-[var(--accent-primary)] disabled:opacity-50 disabled:cursor-not-allowed"
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
            'Save Preferences'
          )}
        </button>
      </div>
    </div>
  );
}
