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

interface SearchFormState {
  top_k: number;
  mode: SearchMode;
  use_reranker: boolean;
  rrf_k: number;
  similarity_threshold: string; // String for form input, empty = null
}

const DEFAULT_FORM_STATE: SearchFormState = {
  top_k: 10,
  mode: 'dense',
  use_reranker: false,
  rrf_k: 60,
  similarity_threshold: '',
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
        <svg className="animate-spin h-8 w-8 text-gray-400" fill="none" viewBox="0 0 24 24">
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
        <span className="ml-3 text-gray-500">Loading search preferences...</span>
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
        <h3 className="text-lg leading-6 font-medium text-gray-900">Search Preferences</h3>
        <p className="mt-1 text-sm text-gray-500">
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
      <div className="bg-white shadow rounded-lg">
        <div className="px-4 py-5 sm:p-6">
          <div className="space-y-6">
            {/* Results Count */}
            <div>
              <label className="block text-sm font-medium text-gray-700">
                Default Results Count
              </label>
              <input
                type="number"
                min={5}
                max={50}
                value={formState.top_k}
                onChange={(e) => handleChange('top_k', parseInt(e.target.value, 10) || 10)}
                className={getInputClassName(false, false)}
              />
              <p className="mt-1 text-xs text-gray-500">
                Number of results to return (5-50)
              </p>
            </div>

            {/* Search Mode */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Search Mode
              </label>
              <div className="flex space-x-2">
                <button
                  type="button"
                  onClick={() => handleChange('mode', 'dense')}
                  className={`flex-1 px-4 py-2 text-sm font-medium rounded-md border ${
                    formState.mode === 'dense'
                      ? 'bg-blue-100 border-blue-500 text-blue-700'
                      : 'bg-white border-gray-300 text-gray-700 hover:bg-gray-50'
                  }`}
                >
                  Dense
                </button>
                <button
                  type="button"
                  onClick={() => handleChange('mode', 'sparse')}
                  className={`flex-1 px-4 py-2 text-sm font-medium rounded-md border ${
                    formState.mode === 'sparse'
                      ? 'bg-blue-100 border-blue-500 text-blue-700'
                      : 'bg-white border-gray-300 text-gray-700 hover:bg-gray-50'
                  }`}
                >
                  Sparse
                </button>
                <button
                  type="button"
                  onClick={() => handleChange('mode', 'hybrid')}
                  className={`flex-1 px-4 py-2 text-sm font-medium rounded-md border ${
                    formState.mode === 'hybrid'
                      ? 'bg-blue-100 border-blue-500 text-blue-700'
                      : 'bg-white border-gray-300 text-gray-700 hover:bg-gray-50'
                  }`}
                >
                  Hybrid
                </button>
              </div>
              <p className="mt-1 text-xs text-gray-500">
                Dense uses vector embeddings, Sparse uses keyword matching, Hybrid combines both
              </p>
            </div>

            {/* RRF Constant - only shown when mode is hybrid */}
            {formState.mode === 'hybrid' && (
              <div>
                <label className="block text-sm font-medium text-gray-700">
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
                <p className="mt-1 text-xs text-gray-500">
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
                  className="h-4 w-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500 disabled:opacity-50"
                />
              </div>
              <div className="ml-3 text-sm">
                <label className="font-medium text-gray-700">
                  Use Reranker
                  {!systemStatus?.reranking_available && (
                    <span className="ml-2 text-gray-400 text-xs">(not available)</span>
                  )}
                </label>
                <p className="text-gray-500">
                  Apply neural reranking to improve result quality (requires GPU)
                </p>
              </div>
            </div>

            {/* Similarity Threshold */}
            <div>
              <label className="block text-sm font-medium text-gray-700">
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
              <p className="mt-1 text-xs text-gray-500">
                Minimum similarity score (0.0-1.0). Leave empty for no threshold.
              </p>
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
            'Save Preferences'
          )}
        </button>
      </div>
    </div>
  );
}
