/**
 * Search Preferences Settings component.
 * Allows users to configure default search behavior.
 */
import { useState, useEffect, useCallback } from 'react';
import {
  usePreferences,
  useUpdatePreferences,
  useResetSearchPreferences,
} from '../../hooks/usePreferences';
import { useSystemStatus } from '../../hooks/useSystemInfo';
import { getInputClassName } from '../../utils/formStyles';
import type { SearchMode } from '../../types/preferences';

import type { HyDEQualityTier } from '../../types/preferences';

interface SearchFormState {
  top_k: number;
  mode: SearchMode;
  use_reranker: boolean;
  rrf_k: number;
  similarity_threshold: string; // String for form input, empty = null
  use_hyde: boolean;
  hyde_quality_tier: HyDEQualityTier;
  hyde_timeout_seconds: number;
}

const DEFAULT_FORM_STATE: SearchFormState = {
  top_k: 10,
  mode: 'dense',
  use_reranker: false,
  rrf_k: 60,
  similarity_threshold: '',
  use_hyde: false,
  hyde_quality_tier: 'low',
  hyde_timeout_seconds: 10,
};

export default function SearchPreferencesSettings() {
  const { data: preferences, isLoading, error } = usePreferences();
  const { data: systemStatus } = useSystemStatus();
  const updateMutation = useUpdatePreferences();
  const resetMutation = useResetSearchPreferences();

  const [formState, setFormState] = useState<SearchFormState>(DEFAULT_FORM_STATE);

  // Initialize form state from preferences
  useEffect(() => {
    if (preferences?.search) {
      setFormState({
        top_k: preferences.search.top_k,
        mode: preferences.search.mode,
        use_reranker: preferences.search.use_reranker,
        rrf_k: preferences.search.rrf_k,
        similarity_threshold:
          preferences.search.similarity_threshold !== null
            ? preferences.search.similarity_threshold.toString()
            : '',
        use_hyde: preferences.search.use_hyde,
        hyde_quality_tier: preferences.search.hyde_quality_tier,
        hyde_timeout_seconds: preferences.search.hyde_timeout_seconds,
      });
    }
  }, [preferences]);

  const handleChange = useCallback(
    <K extends keyof SearchFormState>(field: K, value: SearchFormState[K]) => {
      setFormState((prev) => ({ ...prev, [field]: value }));
    },
    []
  );

  const handleSave = useCallback(async () => {
    const threshold = formState.similarity_threshold
      ? parseFloat(formState.similarity_threshold)
      : null;

    await updateMutation.mutateAsync({
      search: {
        top_k: formState.top_k,
        mode: formState.mode,
        use_reranker: formState.use_reranker,
        rrf_k: formState.rrf_k,
        similarity_threshold: threshold,
        use_hyde: formState.use_hyde,
        hyde_quality_tier: formState.hyde_quality_tier,
        hyde_timeout_seconds: formState.hyde_timeout_seconds,
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
        <span className="ml-3 text-[var(--text-secondary)]">Loading search preferences...</span>
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
        <h3 className="text-lg leading-6 font-medium text-[var(--text-primary)]">Search Preferences</h3>
        <p className="mt-1 text-sm text-[var(--text-secondary)]">
          Configure default search behavior. These settings apply when performing semantic searches.
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
              These defaults will be applied when you open the search interface.
              You can override them for individual searches.
            </p>
          </div>
        </div>
      </div>

      {/* Search Settings Form */}
      <div className="bg-[var(--bg-secondary)] shadow rounded-lg border border-[var(--border)]">
        <div className="px-4 py-5 sm:p-6">
          <div className="space-y-6">
            {/* Results Count */}
            <div>
              <label className="block text-sm font-medium text-[var(--text-primary)]">
                Default Results Count
              </label>
              <input
                type="number"
                min={1}
                max={250}
                value={formState.top_k}
                onChange={(e) => handleChange('top_k', parseInt(e.target.value, 10) || 10)}
                className={getInputClassName(false, false)}
              />
              <p className="mt-1 text-xs text-[var(--text-secondary)]">
                Number of results to return (1-250)
              </p>
            </div>

            {/* Search Mode */}
            <div>
              <label className="block text-sm font-medium text-[var(--text-primary)] mb-2">
                Search Mode
              </label>
              <div className="flex space-x-2">
                <button
                  type="button"
                  onClick={() => handleChange('mode', 'dense')}
                  className={`flex-1 px-4 py-2 text-sm font-medium rounded-md border ${
                    formState.mode === 'dense'
                      ? 'bg-[var(--accent-primary)]/10 border-[var(--accent-primary)] text-[var(--accent-primary)]'
                      : 'bg-[var(--bg-secondary)] border-[var(--border)] text-[var(--text-primary)] hover:bg-[var(--bg-tertiary)]'
                  }`}
                >
                  Dense
                </button>
                <button
                  type="button"
                  onClick={() => handleChange('mode', 'sparse')}
                  className={`flex-1 px-4 py-2 text-sm font-medium rounded-md border ${
                    formState.mode === 'sparse'
                      ? 'bg-[var(--accent-primary)]/10 border-[var(--accent-primary)] text-[var(--accent-primary)]'
                      : 'bg-[var(--bg-secondary)] border-[var(--border)] text-[var(--text-primary)] hover:bg-[var(--bg-tertiary)]'
                  }`}
                >
                  Sparse
                </button>
                <button
                  type="button"
                  onClick={() => handleChange('mode', 'hybrid')}
                  className={`flex-1 px-4 py-2 text-sm font-medium rounded-md border ${
                    formState.mode === 'hybrid'
                      ? 'bg-[var(--accent-primary)]/10 border-[var(--accent-primary)] text-[var(--accent-primary)]'
                      : 'bg-[var(--bg-secondary)] border-[var(--border)] text-[var(--text-primary)] hover:bg-[var(--bg-tertiary)]'
                  }`}
                >
                  Hybrid
                </button>
              </div>
              <p className="mt-1 text-xs text-[var(--text-secondary)]">
                Dense uses vector embeddings, Sparse uses keyword matching, Hybrid combines both
              </p>
            </div>

            {/* RRF Constant - only shown when mode is hybrid */}
            {formState.mode === 'hybrid' && (
              <div>
                <label className="block text-sm font-medium text-[var(--text-primary)]">
                  RRF Constant (k)
                </label>
                <input
                  type="number"
                  min={1}
                  max={100}
                  value={formState.rrf_k}
                  onChange={(e) => handleChange('rrf_k', parseInt(e.target.value, 10) || 60)}
                  className={getInputClassName(false, false)}
                />
                <p className="mt-1 text-xs text-[var(--text-secondary)]">
                  Reciprocal Rank Fusion constant for combining dense and sparse results (1-100)
                </p>
              </div>
            )}

            {/* Use Reranker */}
            <div className="flex items-start">
              <div className="flex items-center h-5">
                <input
                  type="checkbox"
                  checked={formState.use_reranker}
                  onChange={(e) => handleChange('use_reranker', e.target.checked)}
                  disabled={!systemStatus?.reranking_available}
                  className="h-4 w-4 text-[var(--accent-primary)] border-[var(--border)] rounded focus:ring-[var(--accent-primary)] disabled:opacity-50"
                />
              </div>
              <div className="ml-3 text-sm">
                <label className="font-medium text-[var(--text-primary)]">
                  Use Reranker
                  {!systemStatus?.reranking_available && (
                    <span className="ml-2 text-[var(--text-muted)] text-xs">(not available)</span>
                  )}
                </label>
                <p className="text-[var(--text-secondary)]">
                  Apply neural reranking to improve result quality (requires GPU)
                </p>
              </div>
            </div>

            {/* Similarity Threshold */}
            <div>
              <label className="block text-sm font-medium text-[var(--text-primary)]">
                Similarity Threshold
              </label>
              <input
                type="number"
                min={0}
                max={1}
                step={0.05}
                value={formState.similarity_threshold}
                onChange={(e) => handleChange('similarity_threshold', e.target.value)}
                placeholder="No threshold"
                className={getInputClassName(false, false)}
              />
              <p className="mt-1 text-xs text-[var(--text-secondary)]">
                Minimum similarity score (0.0-1.0). Leave empty for no threshold.
              </p>
            </div>

            {/* HyDE Query Expansion */}
            <div className="space-y-4 pt-4 border-t border-[var(--border)]">
              <h4 className="text-sm font-medium text-[var(--text-primary)]">HyDE Query Expansion</h4>

              {/* Enable HyDE Toggle */}
              <div className="flex items-start">
                <div className="flex items-center h-5">
                  <input
                    type="checkbox"
                    checked={formState.use_hyde}
                    onChange={(e) => handleChange('use_hyde', e.target.checked)}
                    className="h-4 w-4 text-[var(--accent-primary)] border-[var(--border)] rounded focus:ring-[var(--accent-primary)]"
                  />
                </div>
                <div className="ml-3 text-sm">
                  <label className="font-medium text-[var(--text-primary)]">Enable HyDE by default</label>
                  <p className="text-[var(--text-secondary)]">
                    Generate hypothetical documents for improved search quality
                  </p>
                </div>
              </div>

              {formState.use_hyde && (
                <>
                  {/* Quality Tier */}
                  <div>
                    <label className="block text-sm font-medium text-[var(--text-primary)] mb-2">
                      Quality Tier
                    </label>
                    <div className="flex space-x-2">
                      <button
                        type="button"
                        onClick={() => handleChange('hyde_quality_tier', 'low')}
                        className={`flex-1 px-4 py-2 text-sm font-medium rounded-md border ${
                          formState.hyde_quality_tier === 'low'
                            ? 'bg-[var(--accent-primary)]/10 border-[var(--accent-primary)] text-[var(--accent-primary)]'
                            : 'bg-[var(--bg-secondary)] border-[var(--border)] text-[var(--text-primary)] hover:bg-[var(--bg-tertiary)]'
                        }`}
                      >
                        Low (Faster)
                      </button>
                      <button
                        type="button"
                        onClick={() => handleChange('hyde_quality_tier', 'high')}
                        className={`flex-1 px-4 py-2 text-sm font-medium rounded-md border ${
                          formState.hyde_quality_tier === 'high'
                            ? 'bg-[var(--accent-primary)]/10 border-[var(--accent-primary)] text-[var(--accent-primary)]'
                            : 'bg-[var(--bg-secondary)] border-[var(--border)] text-[var(--text-primary)] hover:bg-[var(--bg-tertiary)]'
                        }`}
                      >
                        High (Better)
                      </button>
                    </div>
                    <p className="mt-1 text-xs text-[var(--text-secondary)]">
                      Low tier recommended for faster responses
                    </p>
                  </div>

                  {/* Timeout */}
                  <div>
                    <label className="block text-sm font-medium text-[var(--text-primary)]">
                      Timeout
                    </label>
                    <div className="flex items-center space-x-3">
                      <input
                        type="range"
                        min={3}
                        max={60}
                        value={formState.hyde_timeout_seconds}
                        onChange={(e) => handleChange('hyde_timeout_seconds', parseInt(e.target.value, 10))}
                        className="flex-1"
                      />
                      <span className="text-sm text-[var(--text-secondary)] w-12">
                        {formState.hyde_timeout_seconds}s
                      </span>
                    </div>
                    <p className="mt-1 text-xs text-[var(--text-secondary)]">
                      Max time for HyDE generation (3-60s)
                    </p>
                  </div>
                </>
              )}
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
