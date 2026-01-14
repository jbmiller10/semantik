/**
 * Performance Settings component.
 * Admin-only settings for cache, model unloading, and search tuning.
 */
import { useState, useEffect, useCallback } from 'react';
import { useEffectiveSettings, useUpdateSystemSettings, useResetSettingsToDefaults } from '../../hooks/useSystemSettings';
import { getInputClassName } from '../../utils/formStyles';
import { extractPerformanceSettings } from '../../types/system-settings';
import type { PerformanceSettings as PerformanceSettingsType } from '../../types/system-settings';

const PERFORMANCE_KEYS = [
  'cache_ttl_seconds',
  'model_unload_timeout_seconds',
];

export default function PerformanceSettings() {
  const { data: effectiveSettings, isLoading, error } = useEffectiveSettings();
  const updateMutation = useUpdateSystemSettings();
  const resetMutation = useResetSettingsToDefaults();

  const [formState, setFormState] = useState<PerformanceSettingsType>({
    cache_ttl_seconds: 300,
    model_unload_timeout_seconds: 300,
  });

  // Initialize form state from effective settings
  useEffect(() => {
    if (effectiveSettings?.settings) {
      setFormState(extractPerformanceSettings(effectiveSettings.settings));
    }
  }, [effectiveSettings]);

  const handleChange = useCallback(
    <K extends keyof PerformanceSettingsType>(field: K, value: PerformanceSettingsType[K]) => {
      setFormState((prev) => ({ ...prev, [field]: value }));
    },
    []
  );

  const handleSave = useCallback(async () => {
    await updateMutation.mutateAsync({
      settings: {
        cache_ttl_seconds: formState.cache_ttl_seconds,
        model_unload_timeout_seconds: formState.model_unload_timeout_seconds,
      },
    });
  }, [formState, updateMutation]);

  const handleReset = useCallback(async () => {
    await resetMutation.mutateAsync(PERFORMANCE_KEYS);
  }, [resetMutation]);

  // Loading state
  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-8">
        <svg className="animate-spin h-6 w-6 text-gray-400" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
        </svg>
        <span className="ml-2 text-gray-500">Loading performance settings...</span>
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
      <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
        <div className="flex">
          <svg className="h-5 w-5 text-amber-400 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
          <div className="ml-3">
            <p className="text-sm text-amber-700">
              Changing these settings may impact system performance.
              Lower cache TTLs increase database load; higher multipliers increase memory usage.
            </p>
          </div>
        </div>
      </div>

      {/* Settings Form */}
      <div className="space-y-4">
        {/* Cache TTL */}
        <div>
          <label className="block text-sm font-medium text-gray-700">
            Cache TTL (seconds)
          </label>
          <input
            type="number"
            min={30}
            max={3600}
            value={formState.cache_ttl_seconds}
            onChange={(e) => handleChange('cache_ttl_seconds', parseInt(e.target.value, 10) || 300)}
            className={getInputClassName(false, false)}
          />
          <p className="mt-1 text-xs text-gray-500">
            How long to cache frequently accessed data (30-3600 seconds)
          </p>
        </div>

        {/* Model Unload Timeout */}
        <div>
          <label className="block text-sm font-medium text-gray-700">
            Model Unload Timeout (seconds)
          </label>
          <input
            type="number"
            min={60}
            max={3600}
            value={formState.model_unload_timeout_seconds}
            onChange={(e) => handleChange('model_unload_timeout_seconds', parseInt(e.target.value, 10) || 300)}
            className={getInputClassName(false, false)}
          />
          <p className="mt-1 text-xs text-gray-500">
            Time before idle models are unloaded from GPU memory (60-3600 seconds)
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
