/**
 * GPU & Memory Settings component.
 * Admin-only settings for GPU memory management and CPU offloading.
 */
import { useState, useEffect, useCallback } from 'react';
import { useEffectiveSettings, useUpdateSystemSettings, useResetSettingsToDefaults } from '../../hooks/useSystemSettings';
import { getInputClassName } from '../../utils/formStyles';
import { extractGpuMemorySettings } from '../../types/system-settings';
import type { GpuMemorySettings as GpuMemorySettingsType } from '../../types/system-settings';

const GPU_MEMORY_KEYS = [
  'gpu_memory_max_percent',
  'cpu_memory_max_percent',
  'enable_cpu_offload',
  'eviction_idle_threshold_seconds',
];

export default function GpuMemorySettings() {
  const { data: effectiveSettings, isLoading, error } = useEffectiveSettings();
  const updateMutation = useUpdateSystemSettings();
  const resetMutation = useResetSettingsToDefaults();

  const [formState, setFormState] = useState<GpuMemorySettingsType>({
    gpu_memory_max_percent: 0.9,
    cpu_memory_max_percent: 0.5,
    enable_cpu_offload: true,
    eviction_idle_threshold_seconds: 120,
  });

  // Initialize form state from effective settings
  useEffect(() => {
    if (effectiveSettings?.settings) {
      setFormState(extractGpuMemorySettings(effectiveSettings.settings));
    }
  }, [effectiveSettings]);

  const handleChange = useCallback(
    <K extends keyof GpuMemorySettingsType>(field: K, value: GpuMemorySettingsType[K]) => {
      setFormState((prev) => ({ ...prev, [field]: value }));
    },
    []
  );

  const handleSave = useCallback(async () => {
    await updateMutation.mutateAsync({
      settings: {
        gpu_memory_max_percent: formState.gpu_memory_max_percent,
        cpu_memory_max_percent: formState.cpu_memory_max_percent,
        enable_cpu_offload: formState.enable_cpu_offload,
        eviction_idle_threshold_seconds: formState.eviction_idle_threshold_seconds,
      },
    });
  }, [formState, updateMutation]);

  const handleReset = useCallback(async () => {
    await resetMutation.mutateAsync(GPU_MEMORY_KEYS);
  }, [resetMutation]);

  // Loading state
  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-8">
        <svg className="animate-spin h-6 w-6 text-gray-400" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
        </svg>
        <span className="ml-2 text-gray-500">Loading GPU/memory settings...</span>
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
              <strong>Caution:</strong> Incorrect memory settings can cause out-of-memory errors or system instability.
              Changes require a service restart to take full effect.
            </p>
          </div>
        </div>
      </div>

      {/* GPU Memory Section */}
      <div>
        <h4 className="text-sm font-medium text-gray-900 mb-3">GPU Memory</h4>
        <div className="space-y-4 ml-4">
          {/* GPU Memory Limit */}
          <div>
            <label className="block text-sm font-medium text-gray-700">
              GPU Memory Limit ({(formState.gpu_memory_max_percent * 100).toFixed(0)}%)
            </label>
            <input
              type="range"
              min={0.5}
              max={1}
              step={0.05}
              value={formState.gpu_memory_max_percent}
              onChange={(e) => handleChange('gpu_memory_max_percent', parseFloat(e.target.value))}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
            />
            <p className="mt-1 text-xs text-gray-500">
              Maximum GPU memory percentage the application can use (50-100%)
            </p>
          </div>
        </div>
      </div>

      {/* CPU Memory Section */}
      <div>
        <h4 className="text-sm font-medium text-gray-900 mb-3">CPU Memory (Warm Models)</h4>
        <div className="space-y-4 ml-4">
          {/* CPU Memory Limit */}
          <div>
            <label className="block text-sm font-medium text-gray-700">
              CPU Memory Limit ({(formState.cpu_memory_max_percent * 100).toFixed(0)}%)
            </label>
            <input
              type="range"
              min={0.3}
              max={0.9}
              step={0.05}
              value={formState.cpu_memory_max_percent}
              onChange={(e) => handleChange('cpu_memory_max_percent', parseFloat(e.target.value))}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
            />
            <p className="mt-1 text-xs text-gray-500">
              Maximum CPU memory for warm models when offloaded from GPU (30-90%)
            </p>
          </div>
        </div>
      </div>

      {/* Offloading & Eviction Section */}
      <div>
        <h4 className="text-sm font-medium text-gray-900 mb-3">Offloading & Eviction</h4>
        <div className="space-y-4 ml-4">
          {/* Enable CPU Offload */}
          <div className="flex items-start">
            <div className="flex items-center h-5">
              <input
                type="checkbox"
                checked={formState.enable_cpu_offload}
                onChange={(e) => handleChange('enable_cpu_offload', e.target.checked)}
                className="h-4 w-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
              />
            </div>
            <div className="ml-3 text-sm">
              <label className="font-medium text-gray-700">Enable CPU Offload</label>
              <p className="text-gray-500">
                Automatically offload models to CPU when GPU memory is low
              </p>
            </div>
          </div>

          {/* Eviction Idle Threshold */}
          <div>
            <label className="block text-sm font-medium text-gray-700">
              Eviction Idle Threshold (seconds)
            </label>
            <input
              type="number"
              min={30}
              max={600}
              value={formState.eviction_idle_threshold_seconds}
              onChange={(e) => handleChange('eviction_idle_threshold_seconds', parseInt(e.target.value, 10) || 120)}
              className={getInputClassName(false, false)}
            />
            <p className="mt-1 text-xs text-gray-500">
              Time before idle models are evicted from memory (30-600 seconds)
            </p>
          </div>
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
