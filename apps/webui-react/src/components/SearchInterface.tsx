import { useEffect, useRef } from 'react';
import { useSearchStore } from '../stores/searchStore';
import { useUIStore } from '../stores/uiStore';
import { useCollections } from '../hooks/useCollections';
import { searchV2Api } from '../services/api/v2/collections';
import SearchResults from './SearchResults';
import { CollectionMultiSelect } from './CollectionMultiSelect';
import { isAxiosError, getErrorMessage, getInsufficientMemoryErrorDetails } from '../utils/errorUtils';
import { RerankingConfiguration } from './RerankingConfiguration';
import { DEFAULT_VALIDATION_RULES } from '../utils/searchValidation';
import { useRerankingAvailability } from '../hooks/useRerankingAvailability';

function SearchInterface() {
  const {
    searchParams,
    updateSearchParams,
    validateAndUpdateSearchParams,
    setResults,
    setLoading,
    setError,
    setRerankingMetrics,
    setFailedCollections,
    setPartialFailure,
    hasValidationErrors,
    getValidationError,
  } = useSearchStore();
  const addToast = useUIStore((state) => state.addToast);
  
  // Use React Query hook to fetch collections
  const { data: collections = [], refetch: refetchCollections } = useCollections();
  
  // Check reranking availability
  useRerankingAvailability();

  const statusUpdateIntervalRef = useRef<number | null>(null);

  // Check if any collections are processing and set up auto-refresh
  useEffect(() => {
    const hasProcessing = collections.some(
      (col) => col.status === 'processing' || col.status === 'pending'
    );

    if (hasProcessing && !statusUpdateIntervalRef.current) {
      statusUpdateIntervalRef.current = window.setInterval(() => {
        refetchCollections();
      }, 5000);
    } else if (!hasProcessing && statusUpdateIntervalRef.current) {
      window.clearInterval(statusUpdateIntervalRef.current);
      statusUpdateIntervalRef.current = null;
    }

    return () => {
      if (statusUpdateIntervalRef.current) {
        window.clearInterval(statusUpdateIntervalRef.current);
      }
    };
  }, [collections, refetchCollections]);

  const handleSelectSmallerModel = (model: string) => {
    if (model === 'disabled') {
      // Disable reranking entirely
      updateSearchParams({ useReranker: false });
      setError(null);
      delete (window as any).__gpuMemoryError;
      addToast({ 
        type: 'info', 
        message: 'Reranking disabled. Try searching again.' 
      });
    } else {
      // Switch to a smaller model
      updateSearchParams({ rerankModel: model });
      setError(null);
      delete (window as any).__gpuMemoryError;
      addToast({ 
        type: 'info', 
        message: `Switched to ${model.split('/').pop()}. Try searching again.` 
      });
    }
  };

  // Make the handler available globally for SearchResults
  useEffect(() => {
    (window as any).__handleSelectSmallerModel = handleSelectSmallerModel;
    return () => {
      delete (window as any).__handleSelectSmallerModel;
    };
  }, []);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();

    // Validate before search
    if (hasValidationErrors()) {
      addToast({ type: 'error', message: 'Please fix validation errors before searching' });
      return;
    }

    if (!searchParams.query.trim()) {
      addToast({ type: 'error', message: 'Please enter a search query' });
      return;
    }

    if (searchParams.selectedCollections.length === 0) {
      addToast({ type: 'error', message: 'Please select at least one collection' });
      return;
    }

    setLoading(true);
    setError(null);
    setFailedCollections([]);
    setPartialFailure(false);

    try {
      // Use v2 search API with multiple collections
      const response = await searchV2Api.search({
        query: searchParams.query,
        collection_uuids: searchParams.selectedCollections,
        k: searchParams.topK,
        score_threshold: searchParams.scoreThreshold,
        search_type: searchParams.searchType,
        use_reranker: searchParams.useReranker,
        rerank_model: searchParams.useReranker ? searchParams.rerankModel : null,
        hybrid_alpha: searchParams.searchType === 'hybrid' ? searchParams.hybridAlpha : undefined,
        hybrid_mode: searchParams.searchType === 'hybrid' ? searchParams.hybridMode : undefined,
        keyword_mode: searchParams.searchType === 'hybrid' ? searchParams.keywordMode : undefined,
      });

      // Map results to match the search store's SearchResult type
      const mappedResults = response.data.results.map((result) => ({
        doc_id: result.document_id,
        chunk_id: result.chunk_id,
        score: result.score,
        content: result.text,
        file_path: result.file_path,
        file_name: result.file_name,
        chunk_index: result.chunk_index,
        total_chunks: (typeof result.metadata?.total_chunks === 'number' ? result.metadata.total_chunks : 1),
        collection_id: result.collection_id,
        collection_name: result.collection_name,
      }));

      setResults(mappedResults);
      
      // Handle partial failures
      if (response.data.partial_failure) {
        setPartialFailure(true);
        setFailedCollections(response.data.failed_collections || []);
        
        // Show warning toast for partial failures
        const failedCount = response.data.failed_collections?.length || 0;
        addToast({ 
          type: 'warning', 
          message: `Search completed with ${failedCount} collection(s) failing. Check the results for details.` 
        });
      }
      
      // Store reranking metrics if present
      if (response.data.reranking_used !== undefined) {
        setRerankingMetrics({
          rerankingUsed: response.data.reranking_used,
          rerankerModel: response.data.reranker_model,
          rerankingTimeMs: response.data.reranking_time_ms,
        });
      } else {
        setRerankingMetrics(null);
      }
    } catch (error) {
      const memoryErrorDetails = getInsufficientMemoryErrorDetails(error);
      
      if (memoryErrorDetails) {
        // Handle insufficient memory error specifically
        // Store a special error marker that SearchResults can detect
        setError('GPU_MEMORY_ERROR');
        // Store the memory error details in a way SearchResults can access
        (window as any).__gpuMemoryError = {
          message: memoryErrorDetails.message,
          suggestion: memoryErrorDetails.suggestion,
          currentModel: searchParams.rerankModel
        };
        addToast({ 
          type: 'error', 
          message: 'Insufficient GPU memory for reranking. See below for options.' 
        });
      } else {
        // Handle other errors
        const errorMessage = getErrorMessage(error);
        setError(errorMessage);
        addToast({ type: 'error', message: isAxiosError(error) ? 'Search failed' : errorMessage });
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Search Form */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-2xl font-semibold mb-6">Search Documents</h2>
        {hasValidationErrors() && (
          <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-md" role="alert">
            <p className="text-sm text-red-700 font-medium">Please fix the following errors:</p>
            <ul className="mt-2 text-sm text-red-600 space-y-1">
              {getValidationError('query') && <li>• {getValidationError('query')}</li>}
              {getValidationError('collections') && <li>• {getValidationError('collections')}</li>}
              {getValidationError('topK') && <li>• {getValidationError('topK')}</li>}
              {getValidationError('scoreThreshold') && <li>• {getValidationError('scoreThreshold')}</li>}
              {getValidationError('hybridAlpha') && <li>• {getValidationError('hybridAlpha')}</li>}
            </ul>
          </div>
        )}
        <form onSubmit={handleSearch} className="space-y-4">
          {/* Search Query */}
          <div>
            <label htmlFor="query" className="block text-sm font-medium text-gray-700 mb-2">
              Search Query
              {searchParams.query && (
                <span className="ml-2 text-xs text-gray-500">
                  ({searchParams.query.length}/{DEFAULT_VALIDATION_RULES.query.maxLength})
                </span>
              )}
            </label>
            <input
              type="text"
              id="query"
              value={searchParams.query}
              onChange={(e) => validateAndUpdateSearchParams({ query: e.target.value })}
              className={`w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 ${
                getValidationError('query') 
                  ? 'border-red-500 focus:ring-red-500' 
                  : 'border-gray-300 focus:ring-blue-500'
              }`}
              placeholder="Enter your search query..."
              aria-label="Search query"
              aria-required="true"
              aria-invalid={!!getValidationError('query')}
              aria-describedby={getValidationError('query') ? 'query-error' : 'query-tips'}
            />
            {getValidationError('query') && (
              <p id="query-error" className="mt-1 text-sm text-red-600" role="alert">
                {getValidationError('query')}
              </p>
            )}
            <div id="query-tips" className="mt-2 text-xs text-gray-600" role="note">
              <p className="font-medium">Search Tips:</p>
              <ul className="list-disc list-inside space-y-1 text-gray-500">
                <li>Use descriptive keywords for better results</li>
                <li>Questions often work well (e.g., "How do I...?")</li>
                <li>Try different phrasings if you don't find what you need</li>
              </ul>
            </div>
          </div>

          {/* Grid for Collection and Results count */}
          <div className="grid grid-cols-2 gap-4">
            {/* Collection Selection */}
            <div>
              <label id="collections-label" className="block text-sm font-medium text-gray-700 mb-2">
                Collections
                <button
                  type="button"
                  onClick={() => refetchCollections()}
                  className="ml-2 text-blue-600 hover:text-blue-800 text-xs"
                  aria-label="Refresh collections list"
                >
                  (refresh)
                </button>
              </label>
              <div role="group" aria-labelledby="collections-label" aria-describedby="collections-help">
                <CollectionMultiSelect
                  collections={collections}
                  selectedCollections={searchParams.selectedCollections}
                  onChange={(selected) => validateAndUpdateSearchParams({ selectedCollections: selected })}
                  placeholder="Select collections to search..."
                />
              </div>
              {getValidationError('collections') ? (
                <p id="collections-error" className="mt-1 text-sm text-red-600" role="alert">
                  {getValidationError('collections')}
                </p>
              ) : (
                <p id="collections-help" className="mt-1 text-xs text-gray-500">
                  Search across multiple collections simultaneously
                </p>
              )}
            </div>

            {/* Number of Results */}
            <div>
              <label htmlFor="topK" className="block text-sm font-medium text-gray-700 mb-2">
                Number of Results
              </label>
              <span id="topK-help" className="sr-only">Specify how many search results to return, between 1 and 100</span>
              <input
                type="number"
                id="topK"
                min="1"
                max="100"
                value={searchParams.topK}
                onChange={(e) => {
                  const value = parseInt(e.target.value);
                  if (!isNaN(value)) {
                    validateAndUpdateSearchParams({ topK: value });
                  }
                }}
                className={`w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 ${
                  getValidationError('topK')
                    ? 'border-red-500 focus:ring-red-500'
                    : 'border-gray-300 focus:ring-blue-500'
                }`}
                aria-label="Number of search results"
                aria-describedby={getValidationError('topK') ? 'topK-error' : 'topK-help'}
                aria-invalid={!!getValidationError('topK')}
              />
              {getValidationError('topK') && (
                <p id="topK-error" className="mt-1 text-sm text-red-600" role="alert">
                  {getValidationError('topK')}
                </p>
              )}
            </div>
          </div>

          {/* Score Threshold */}
          <div>
            <label htmlFor="scoreThreshold" className="block text-sm font-medium text-gray-700 mb-2">
              Minimum Score Threshold
            </label>
            <div className="flex items-center space-x-2">
              <input
                type="number"
                id="scoreThreshold"
                min="0"
                max="1"
                step="0.1"
                value={searchParams.scoreThreshold}
                onChange={(e) => {
                  const value = parseFloat(e.target.value);
                  if (!isNaN(value)) {
                    validateAndUpdateSearchParams({ scoreThreshold: value });
                  }
                }}
                className={`w-32 px-3 py-2 border rounded-md focus:outline-none focus:ring-2 ${
                  getValidationError('scoreThreshold')
                    ? 'border-red-500 focus:ring-red-500'
                    : 'border-gray-300 focus:ring-blue-500'
                }`}
                aria-label="Minimum score threshold (0-1)"
                aria-describedby={getValidationError('scoreThreshold') ? 'scoreThreshold-error' : 'scoreThreshold-help'}
                aria-invalid={!!getValidationError('scoreThreshold')}
              />
              <span className="text-sm text-gray-600">
                (0.0 = Show all, 1.0 = Perfect matches only)
              </span>
            </div>
            {getValidationError('scoreThreshold') && (
              <p id="scoreThreshold-error" className="mt-1 text-sm text-red-600" role="alert">
                {getValidationError('scoreThreshold')}
              </p>
            )}
            <p id="scoreThreshold-help" className="mt-1 text-xs text-gray-500">
              Only show results with similarity scores above this threshold
            </p>
          </div>

          {/* Search Mode Toggle */}
          <div className="bg-gray-50 rounded-lg p-4 space-y-3">
            <div className="flex items-center justify-between">
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  id="use-hybrid-search"
                  checked={searchParams.searchType === 'hybrid'}
                  onChange={(e) => validateAndUpdateSearchParams({ searchType: e.target.checked ? 'hybrid' : 'semantic' })}
                  className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 focus:ring-2"
                  aria-describedby="hybrid-search-description"
                />
                <span className="text-sm text-gray-700">
                  Use Hybrid Search (combines vector similarity with keyword matching)
                </span>
              </label>
            </div>

            {/* Hybrid Search Options */}
            {searchParams.searchType === 'hybrid' && (
              <div className="space-y-3">
                <div className="grid grid-cols-2 gap-4">
                  {/* Hybrid Mode */}
                  <div>
                    <label htmlFor="hybrid-mode" className="block text-xs font-medium text-gray-600 mb-1">
                      Hybrid Mode
                    </label>
                    <select
                      id="hybrid-mode"
                      value={searchParams.hybridMode || 'reciprocal_rank'}
                      onChange={(e) => validateAndUpdateSearchParams({ hybridMode: e.target.value as 'reciprocal_rank' | 'relative_score' })}
                      className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500"
                      aria-label="Hybrid search mode"
                    >
                      <option value="reciprocal_rank">Reciprocal Rank Fusion</option>
                      <option value="relative_score">Relative Score Fusion</option>
                    </select>
                  </div>

                  {/* Keyword Mode */}
                  <div>
                    <label htmlFor="keyword-mode" className="block text-xs font-medium text-gray-600 mb-1">
                      Keyword Matching
                    </label>
                    <select
                      id="keyword-mode"
                      value={searchParams.keywordMode || 'bm25'}
                      onChange={(e) => validateAndUpdateSearchParams({ keywordMode: e.target.value as 'bm25' })}
                      className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500"
                      aria-label="Keyword matching algorithm"
                    >
                      <option value="bm25">BM25 Ranking</option>
                    </select>
                  </div>
                </div>

                {/* Hybrid Alpha Slider */}
                <div className="mt-3">
                  <label htmlFor="hybrid-alpha" className="block text-xs font-medium text-gray-600 mb-1">
                    Hybrid Alpha (Vector vs Keyword Weight): {searchParams.hybridAlpha || 0.7}
                  </label>
                  <div className="flex items-center space-x-2">
                    <span className="text-xs text-gray-500">Keyword</span>
                    <input
                      type="range"
                      id="hybrid-alpha"
                      min="0"
                      max="1"
                      step="0.1"
                      value={searchParams.hybridAlpha || 0.7}
                      onChange={(e) => {
                        const value = parseFloat(e.target.value);
                        if (!isNaN(value)) {
                          validateAndUpdateSearchParams({ hybridAlpha: value });
                        }
                      }}
                      className="flex-1"
                      aria-label="Hybrid search alpha value"
                      aria-valuemin={0}
                      aria-valuemax={1}
                      aria-valuenow={searchParams.hybridAlpha || 0.7}
                    />
                    <span className="text-xs text-gray-500">Vector</span>
                  </div>
                  {getValidationError('hybridAlpha') && (
                    <p id="hybrid-alpha-error" className="mt-1 text-sm text-red-600" role="alert">
                      {getValidationError('hybridAlpha')}
                    </p>
                  )}
                  <p className="text-xs text-gray-400 mt-1">
                    0.0 = Pure keyword search, 1.0 = Pure vector search, 0.7 = Balanced
                  </p>
                </div>

                <p id="hybrid-search-description" className="text-xs text-gray-500 mt-3" role="note">
                  <strong>Reciprocal Rank:</strong> Combines rankings from both methods.
                  <br />
                  <strong>Relative Score:</strong> Combines normalized scores from both methods.
                </p>
              </div>
            )}
          </div>

          {/* Reranking Options */}
          <RerankingConfiguration
            enabled={searchParams.useReranker}
            model={searchParams.rerankModel}
            quantization={searchParams.rerankQuantization}
            onChange={validateAndUpdateSearchParams}
          />

          {/* Submit Button */}
          <div className="pt-2">
            <button
              type="submit"
              disabled={searchParams.selectedCollections.length === 0 || hasValidationErrors()}
              className="w-full px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              aria-label="Perform search"
              aria-disabled={searchParams.selectedCollections.length === 0 || hasValidationErrors()}
            >
              Search
            </button>
          </div>
        </form>
      </div>

      {/* Search Results */}
      <SearchResults />
    </div>
  );
}

export default SearchInterface;