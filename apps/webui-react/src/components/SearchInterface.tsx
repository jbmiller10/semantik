import { useEffect, useRef } from 'react';
import { AxiosError } from 'axios';
import { useSearchStore } from '../stores/searchStore';
import { useUIStore } from '../stores/uiStore';
import { useCollections } from '../hooks/useCollections';
import { searchV2Api } from '../services/api/v2/collections';
import SearchResults from './SearchResults';
import { CollectionMultiSelect } from './CollectionMultiSelect';

function SearchInterface() {
  const {
    searchParams,
    updateSearchParams,
    setResults,
    setLoading,
    setError,
    setRerankingMetrics,
    setFailedCollections,
    setPartialFailure,
  } = useSearchStore();
  const addToast = useUIStore((state) => state.addToast);
  
  // Use React Query hook to fetch collections
  const { data: collections = [], refetch: refetchCollections } = useCollections();

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

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();

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
      if (error instanceof AxiosError) {
        const errorDetail = error.response?.data?.detail;
        
        // Handle insufficient memory error specifically
        if (error.response?.status === 507 && 
            typeof errorDetail === 'object' && 
            errorDetail && 
            'error' in errorDetail && 
            (errorDetail as { error: string }).error === 'insufficient_memory') {
          const memoryError = errorDetail as { error: string; message?: string; suggestion?: string };
          const errorMessage = memoryError.message || 'Insufficient GPU memory for reranking';
          const suggestion = memoryError.suggestion || 'Try using a smaller model or different quantization';
          
          setError(`${errorMessage}\n\n${suggestion}`);
          addToast({ 
            type: 'error', 
            message: 'Insufficient GPU memory for reranking. Check the error message for suggestions.' 
          });
        } else {
          // Handle other errors
          const errorMessage = typeof errorDetail === 'string' ? errorDetail : 
                             (typeof errorDetail === 'object' && errorDetail && 'message' in errorDetail) 
                               ? (errorDetail as { message: string }).message 
                               : 'Search failed';
          setError(errorMessage);
          addToast({ type: 'error', message: 'Search failed' });
        }
      } else {
        setError('An unexpected error occurred');
        addToast({ type: 'error', message: 'Search failed' });
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
        <form onSubmit={handleSearch} className="space-y-4">
          {/* Search Query */}
          <div>
            <label htmlFor="query" className="block text-sm font-medium text-gray-700 mb-2">
              Search Query
            </label>
            <input
              type="text"
              id="query"
              value={searchParams.query}
              onChange={(e) => updateSearchParams({ query: e.target.value })}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="Enter your search query..."
            />
            <div className="mt-2 text-xs text-gray-600">
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
              <label htmlFor="collection" className="block text-sm font-medium text-gray-700 mb-2">
                Collections
                <button
                  type="button"
                  onClick={() => refetchCollections()}
                  className="ml-2 text-blue-600 hover:text-blue-800 text-xs"
                >
                  (refresh)
                </button>
              </label>
              <CollectionMultiSelect
                collections={collections}
                selectedCollections={searchParams.selectedCollections}
                onChange={(selected) => updateSearchParams({ selectedCollections: selected })}
                placeholder="Select collections to search..."
              />
              <p className="mt-1 text-xs text-gray-500">
                Search across multiple collections simultaneously
              </p>
            </div>

            {/* Number of Results */}
            <div>
              <label htmlFor="topK" className="block text-sm font-medium text-gray-700 mb-2">
                Number of Results
              </label>
              <input
                type="number"
                id="topK"
                min="1"
                max="100"
                value={searchParams.topK}
                onChange={(e) => updateSearchParams({ topK: parseInt(e.target.value) })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>

          {/* Search Mode Toggle */}
          <div className="bg-gray-50 rounded-lg p-4 space-y-3">
            <div className="flex items-center justify-between">
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  id="use-hybrid-search"
                  checked={searchParams.searchType === 'hybrid'}
                  onChange={(e) => updateSearchParams({ searchType: e.target.checked ? 'hybrid' : 'semantic' })}
                  className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 focus:ring-2"
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
                      onChange={(e) => updateSearchParams({ hybridMode: e.target.value as 'reciprocal_rank' | 'relative_score' })}
                      className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500"
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
                      onChange={(e) => updateSearchParams({ keywordMode: e.target.value as 'bm25' })}
                      className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500"
                    >
                      <option value="bm25">BM25 Ranking</option>
                    </select>
                  </div>
                </div>

                <p className="text-xs text-gray-500">
                  <strong>Rerank:</strong> Searches with vectors first, then uses keywords to reorder results.
                  <br />
                  <strong>Filter:</strong> Uses keywords to narrow down the search space before vector matching.
                </p>
              </div>
            )}
          </div>

          {/* Reranking Options */}
          <div className="mb-4">
            <div className="bg-gray-50 rounded-lg p-4">
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={searchParams.useReranker}
                  onChange={(e) => updateSearchParams({ useReranker: e.target.checked })}
                  className="w-4 h-4 text-blue-600 rounded"
                />
                <span className="text-sm font-medium text-gray-700">
                  Enable Cross-Encoder Reranking
                </span>
              </label>
              
              {searchParams.useReranker && (
                <div className="mt-3 ml-6">
                  <p className="text-xs text-gray-600 mb-3">
                    Reranking uses a more sophisticated model to re-score the top search results, 
                    improving accuracy at the cost of slightly increased latency.
                  </p>
                  
                  <div className="space-y-3">
                    {/* Reranker Model Selection */}
                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <label className="block text-xs text-gray-700 mb-1">
                          Reranker Model
                        </label>
                        <select
                          value={searchParams.rerankModel || 'auto'}
                          onChange={(e) => updateSearchParams({ 
                            rerankModel: e.target.value === 'auto' ? undefined : e.target.value 
                          })}
                          className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500"
                        >
                          <option value="auto">Auto-select</option>
                          <option value="Qwen/Qwen3-Reranker-0.6B">0.6B (Fastest)</option>
                          <option value="Qwen/Qwen3-Reranker-4B">4B (Balanced)</option>
                          <option value="Qwen/Qwen3-Reranker-8B">8B (Most Accurate)</option>
                        </select>
                      </div>
                      
                      <div>
                        <label className="block text-xs text-gray-700 mb-1">
                          Quantization
                        </label>
                        <select
                          value={searchParams.rerankQuantization || 'auto'}
                          onChange={(e) => updateSearchParams({ 
                            rerankQuantization: e.target.value === 'auto' ? undefined : e.target.value 
                          })}
                          className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500"
                        >
                          <option value="auto">Auto (match embedding)</option>
                          <option value="float32">Float32 (Full precision)</option>
                          <option value="float16">Float16 (Balanced)</option>
                          <option value="int8">Int8 (Low memory)</option>
                        </select>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Submit Button */}
          <div className="pt-2">
            <button
              type="submit"
              disabled={searchParams.selectedCollections.length === 0}
              className="w-full px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
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