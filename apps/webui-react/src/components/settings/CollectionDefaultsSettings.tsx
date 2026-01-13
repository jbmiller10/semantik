/**
 * Collection Defaults Settings component.
 * Allows users to configure default settings for new collections.
 */
import { useState, useEffect, useCallback } from 'react';
import {
  usePreferences,
  useUpdatePreferences,
  useResetCollectionDefaults,
} from '../../hooks/usePreferences';
import { useEmbeddingModels } from '../../hooks/useModels';
import { getInputClassName } from '../../utils/formStyles';
import type { ChunkingStrategy, Quantization, SparseType } from '../../types/preferences';

interface CollectionFormState {
  embedding_model: string;
  quantization: Quantization;
  chunking_strategy: ChunkingStrategy;
  chunk_size: number;
  chunk_overlap: number;
  enable_sparse: boolean;
  sparse_type: SparseType;
  enable_hybrid: boolean;
}

const DEFAULT_FORM_STATE: CollectionFormState = {
  embedding_model: '',
  quantization: 'none',
  chunking_strategy: 'recursive',
  chunk_size: 1024,
  chunk_overlap: 200,
  enable_sparse: false,
  sparse_type: 'bm25',
  enable_hybrid: false,
};

export default function CollectionDefaultsSettings() {
  const { data: preferences, isLoading, error } = usePreferences();
  const { data: modelsData, isLoading: modelsLoading } = useEmbeddingModels();
  const updateMutation = useUpdatePreferences();
  const resetMutation = useResetCollectionDefaults();

  const [formState, setFormState] = useState<CollectionFormState>(DEFAULT_FORM_STATE);
  const [validationError, setValidationError] = useState<string | null>(null);

  // Initialize form state from preferences
  useEffect(() => {
    if (preferences?.collection_defaults) {
      setFormState({
        embedding_model: preferences.collection_defaults.embedding_model || '',
        quantization: preferences.collection_defaults.quantization,
        chunking_strategy: preferences.collection_defaults.chunking_strategy,
        chunk_size: preferences.collection_defaults.chunk_size,
        chunk_overlap: preferences.collection_defaults.chunk_overlap,
        enable_sparse: preferences.collection_defaults.enable_sparse,
        sparse_type: preferences.collection_defaults.sparse_type,
        enable_hybrid: preferences.collection_defaults.enable_hybrid,
      });
    }
  }, [preferences]);

  // Validate hybrid requires sparse
  useEffect(() => {
    if (formState.enable_hybrid && !formState.enable_sparse) {
      setValidationError('Hybrid search requires sparse indexing to be enabled');
    } else {
      setValidationError(null);
    }
  }, [formState.enable_hybrid, formState.enable_sparse]);

  const handleChange = useCallback(
    <K extends keyof CollectionFormState>(field: K, value: CollectionFormState[K]) => {
      setFormState((prev) => {
        const newState = { ...prev, [field]: value };
        // If disabling sparse, also disable hybrid
        if (field === 'enable_sparse' && !value) {
          newState.enable_hybrid = false;
        }
        return newState;
      });
    },
    []
  );

  const handleSave = useCallback(async () => {
    if (validationError) return;

    await updateMutation.mutateAsync({
      collection_defaults: {
        embedding_model: formState.embedding_model || null,
        quantization: formState.quantization,
        chunking_strategy: formState.chunking_strategy,
        chunk_size: formState.chunk_size,
        chunk_overlap: formState.chunk_overlap,
        enable_sparse: formState.enable_sparse,
        sparse_type: formState.sparse_type,
        enable_hybrid: formState.enable_hybrid,
      },
    });
  }, [formState, validationError, updateMutation]);

  const handleReset = useCallback(async () => {
    await resetMutation.mutateAsync();
  }, [resetMutation]);

  // Get sorted model names
  const modelNames = modelsData?.models
    ? Object.keys(modelsData.models).sort((a, b) => a.localeCompare(b))
    : [];

  // Loading state
  if (isLoading || modelsLoading) {
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
        <span className="ml-3 text-gray-500">Loading collection defaults...</span>
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
        <h3 className="text-lg leading-6 font-medium text-gray-900">Collection Defaults</h3>
        <p className="mt-1 text-sm text-gray-500">
          Configure default settings applied when creating new collections.
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
              These defaults will pre-populate the collection creation form.
              You can override them for individual collections.
            </p>
          </div>
        </div>
      </div>

      {/* Embedding Settings */}
      <div className="bg-white shadow rounded-lg">
        <div className="px-4 py-5 sm:p-6">
          <h4 className="text-md font-medium text-gray-900 mb-4">Embedding Settings</h4>

          <div className="space-y-4">
            {/* Embedding Model */}
            <div>
              <label className="block text-sm font-medium text-gray-700">
                Default Embedding Model
              </label>
              <select
                value={formState.embedding_model}
                onChange={(e) => handleChange('embedding_model', e.target.value)}
                className={getInputClassName(false, false)}
              >
                <option value="">Use system default</option>
                {modelNames.map((modelName) => (
                  <option key={modelName} value={modelName}>
                    {modelName}
                  </option>
                ))}
              </select>
              <p className="mt-1 text-xs text-gray-500">
                The embedding model used to vectorize documents
              </p>
            </div>

            {/* Quantization */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Vector Quantization
              </label>
              <div className="flex space-x-2">
                <button
                  type="button"
                  onClick={() => handleChange('quantization', 'none')}
                  className={`flex-1 px-4 py-2 text-sm font-medium rounded-md border ${
                    formState.quantization === 'none'
                      ? 'bg-blue-100 border-blue-500 text-blue-700'
                      : 'bg-white border-gray-300 text-gray-700 hover:bg-gray-50'
                  }`}
                >
                  None
                </button>
                <button
                  type="button"
                  onClick={() => handleChange('quantization', 'scalar')}
                  className={`flex-1 px-4 py-2 text-sm font-medium rounded-md border ${
                    formState.quantization === 'scalar'
                      ? 'bg-blue-100 border-blue-500 text-blue-700'
                      : 'bg-white border-gray-300 text-gray-700 hover:bg-gray-50'
                  }`}
                >
                  Scalar
                </button>
                <button
                  type="button"
                  onClick={() => handleChange('quantization', 'binary')}
                  className={`flex-1 px-4 py-2 text-sm font-medium rounded-md border ${
                    formState.quantization === 'binary'
                      ? 'bg-blue-100 border-blue-500 text-blue-700'
                      : 'bg-white border-gray-300 text-gray-700 hover:bg-gray-50'
                  }`}
                >
                  Binary
                </button>
              </div>
              <p className="mt-1 text-xs text-gray-500">
                Reduces storage at the cost of some accuracy
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Chunking Settings */}
      <div className="bg-white shadow rounded-lg">
        <div className="px-4 py-5 sm:p-6">
          <h4 className="text-md font-medium text-gray-900 mb-4">Chunking Settings</h4>

          <div className="space-y-4">
            {/* Chunking Strategy */}
            <div>
              <label className="block text-sm font-medium text-gray-700">
                Chunking Strategy
              </label>
              <select
                value={formState.chunking_strategy}
                onChange={(e) => handleChange('chunking_strategy', e.target.value as ChunkingStrategy)}
                className={getInputClassName(false, false)}
              >
                <option value="character">Character</option>
                <option value="recursive">Recursive</option>
                <option value="markdown">Markdown</option>
                <option value="semantic">Semantic</option>
              </select>
              <p className="mt-1 text-xs text-gray-500">
                How documents are split into chunks for embedding
              </p>
            </div>

            {/* Chunk Size and Overlap */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700">
                  Chunk Size
                </label>
                <input
                  type="number"
                  min={256}
                  max={4096}
                  value={formState.chunk_size}
                  onChange={(e) => handleChange('chunk_size', parseInt(e.target.value, 10) || 1024)}
                  className={getInputClassName(false, false)}
                />
                <p className="mt-1 text-xs text-gray-500">
                  Characters per chunk (256-4096)
                </p>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700">
                  Chunk Overlap
                </label>
                <input
                  type="number"
                  min={0}
                  max={512}
                  value={formState.chunk_overlap}
                  onChange={(e) => handleChange('chunk_overlap', parseInt(e.target.value, 10) || 0)}
                  className={getInputClassName(false, false)}
                />
                <p className="mt-1 text-xs text-gray-500">
                  Overlap between chunks (0-512)
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Sparse Indexing Settings */}
      <div className="bg-white shadow rounded-lg">
        <div className="px-4 py-5 sm:p-6">
          <h4 className="text-md font-medium text-gray-900 mb-4">Sparse Indexing</h4>

          <div className="space-y-4">
            {/* Enable Sparse Indexing */}
            <div className="flex items-start">
              <div className="flex items-center h-5">
                <input
                  type="checkbox"
                  checked={formState.enable_sparse}
                  onChange={(e) => handleChange('enable_sparse', e.target.checked)}
                  className="h-4 w-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                />
              </div>
              <div className="ml-3 text-sm">
                <label className="font-medium text-gray-700">Enable Sparse Indexing</label>
                <p className="text-gray-500">
                  Create keyword-based index in addition to vector embeddings
                </p>
              </div>
            </div>

            {/* Sparse Type - only shown when sparse is enabled */}
            {formState.enable_sparse && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Sparse Index Type
                </label>
                <div className="flex space-x-2">
                  <button
                    type="button"
                    onClick={() => handleChange('sparse_type', 'bm25')}
                    className={`flex-1 px-4 py-2 text-sm font-medium rounded-md border ${
                      formState.sparse_type === 'bm25'
                        ? 'bg-blue-100 border-blue-500 text-blue-700'
                        : 'bg-white border-gray-300 text-gray-700 hover:bg-gray-50'
                    }`}
                  >
                    BM25
                  </button>
                  <button
                    type="button"
                    onClick={() => handleChange('sparse_type', 'splade')}
                    className={`flex-1 px-4 py-2 text-sm font-medium rounded-md border ${
                      formState.sparse_type === 'splade'
                        ? 'bg-blue-100 border-blue-500 text-blue-700'
                        : 'bg-white border-gray-300 text-gray-700 hover:bg-gray-50'
                    }`}
                  >
                    SPLADE
                  </button>
                </div>
                <p className="mt-1 text-xs text-gray-500">
                  BM25 is keyword-based, SPLADE uses learned sparse representations
                </p>
              </div>
            )}

            {/* Enable Hybrid Search */}
            <div className="flex items-start">
              <div className="flex items-center h-5">
                <input
                  type="checkbox"
                  checked={formState.enable_hybrid}
                  onChange={(e) => handleChange('enable_hybrid', e.target.checked)}
                  disabled={!formState.enable_sparse}
                  className="h-4 w-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500 disabled:opacity-50"
                />
              </div>
              <div className="ml-3 text-sm">
                <label className={`font-medium ${formState.enable_sparse ? 'text-gray-700' : 'text-gray-400'}`}>
                  Enable Hybrid Search by Default
                </label>
                <p className={formState.enable_sparse ? 'text-gray-500' : 'text-gray-400'}>
                  Combine dense and sparse results for better search quality
                  {!formState.enable_sparse && ' (requires sparse indexing)'}
                </p>
              </div>
            </div>

            {/* Validation Error */}
            {validationError && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-3">
                <p className="text-sm text-red-700">{validationError}</p>
              </div>
            )}
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
          {resetMutation.isPending ? 'Resetting...' : 'Reset to System Defaults'}
        </button>
        <button
          type="button"
          onClick={handleSave}
          disabled={updateMutation.isPending || !!validationError}
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
            'Save Defaults'
          )}
        </button>
      </div>
    </div>
  );
}
