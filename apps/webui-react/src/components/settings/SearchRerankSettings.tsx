/**
 * Search & Reranking Settings component.
 * Admin-only settings for reranking behavior and search tuning.
 */
import { useState, useEffect, useCallback } from 'react';
import { useEffectiveSettings, useUpdateSystemSettings, useResetSettingsToDefaults } from '../../hooks/useSystemSettings';
import { getInputClassName } from '../../utils/formStyles';
import { extractSearchRerankSettings } from '../../types/system-settings';
import type { SearchRerankSettings as SearchRerankSettingsType } from '../../types/system-settings';

const SEARCH_RERANK_KEYS = [
  'rerank_candidate_multiplier',
  'rerank_min_candidates',
  'rerank_max_candidates',
  'rerank_hybrid_weight',
];

export default function SearchRerankSettings() {
  const { data: effectiveSettings, isLoading, error } = useEffectiveSettings();
  const updateMutation = useUpdateSystemSettings();
  const resetMutation = useResetSettingsToDefaults();

  const [formState, setFormState] = useState<SearchRerankSettingsType>({
    rerank_candidate_multiplier: 5,
    rerank_min_candidates: 20,
    rerank_max_candidates: 200,
    rerank_hybrid_weight: 0.3,
  });

  // Initialize form state from effective settings
  useEffect(() => {
    if (effectiveSettings?.settings) {
      setFormState(extractSearchRerankSettings(effectiveSettings.settings));
    }
  }, [effectiveSettings]);

  const handleChange = useCallback(
    <K extends keyof SearchRerankSettingsType>(field: K, value: SearchRerankSettingsType[K]) => {
      setFormState((prev) => ({ ...prev, [field]: value }));
    },
    []
  );

  const handleSave = useCallback(async () => {
    await updateMutation.mutateAsync({
      settings: {
        rerank_candidate_multiplier: formState.rerank_candidate_multiplier,
        rerank_min_candidates: formState.rerank_min_candidates,
        rerank_max_candidates: formState.rerank_max_candidates,
        rerank_hybrid_weight: formState.rerank_hybrid_weight,
      },
    });
  }, [formState, updateMutation]);

  const handleReset = useCallback(async () => {
    await resetMutation.mutateAsync(SEARCH_RERANK_KEYS);
  }, [resetMutation]);

  // Loading state
  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-8">
        <svg className="animate-spin h-6 w-6 text-[var(--text-muted)]" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
        </svg>
        <span className="ml-2 text-[var(--text-secondary)]">Loading search settings...</span>
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
              These settings control how the reranking model processes search results.
              Higher candidate counts improve quality but increase GPU memory usage and latency.
            </p>
          </div>
        </div>
      </div>

      {/* Settings Form */}
      <div className="space-y-4">
        {/* Candidate Multiplier */}
        <div>
          <label className="block text-sm font-medium text-[var(--text-primary)]">
            Rerank Candidate Multiplier
          </label>
          <input
            type="number"
            min={1}
            max={20}
            value={formState.rerank_candidate_multiplier}
            onChange={(e) => handleChange('rerank_candidate_multiplier', parseInt(e.target.value, 10) || 5)}
            className={getInputClassName(false, false)}
          />
          <p className="mt-1 text-xs text-[var(--text-secondary)]">
            Multiplier applied to requested results to get initial candidates (1-20).
            E.g., requesting 10 results with multiplier 5 fetches 50 candidates for reranking.
          </p>
        </div>

        {/* Min Candidates */}
        <div>
          <label className="block text-sm font-medium text-[var(--text-primary)]">
            Minimum Candidates
          </label>
          <input
            type="number"
            min={5}
            max={100}
            value={formState.rerank_min_candidates}
            onChange={(e) => handleChange('rerank_min_candidates', parseInt(e.target.value, 10) || 20)}
            className={getInputClassName(false, false)}
          />
          <p className="mt-1 text-xs text-[var(--text-secondary)]">
            Minimum number of candidates to fetch for reranking (5-100)
          </p>
        </div>

        {/* Max Candidates */}
        <div>
          <label className="block text-sm font-medium text-[var(--text-primary)]">
            Maximum Candidates
          </label>
          <input
            type="number"
            min={50}
            max={500}
            value={formState.rerank_max_candidates}
            onChange={(e) => handleChange('rerank_max_candidates', parseInt(e.target.value, 10) || 200)}
            className={getInputClassName(false, false)}
          />
          <p className="mt-1 text-xs text-[var(--text-secondary)]">
            Maximum number of candidates to rerank (50-500). Higher values use more GPU memory.
          </p>
        </div>

        {/* Hybrid Weight */}
        <div>
          <label className="block text-sm font-medium text-[var(--text-primary)]">
            Hybrid Weight ({formState.rerank_hybrid_weight.toFixed(2)})
          </label>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={formState.rerank_hybrid_weight}
            onChange={(e) => handleChange('rerank_hybrid_weight', parseFloat(e.target.value))}
            className="w-full h-2 bg-[var(--bg-tertiary)] rounded-lg appearance-none cursor-pointer"
          />
          <div className="flex justify-between text-xs text-[var(--text-secondary)] mt-1">
            <span>Dense only (0.0)</span>
            <span>Balanced (0.5)</span>
            <span>Sparse only (1.0)</span>
          </div>
          <p className="mt-1 text-xs text-[var(--text-secondary)]">
            Weight given to sparse (keyword) scores vs dense (semantic) scores in hybrid mode
          </p>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex justify-between pt-4 border-t border-[var(--border)]">
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
            'Save Settings'
          )}
        </button>
      </div>
    </div>
  );
}
