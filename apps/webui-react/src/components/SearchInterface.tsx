import { useEffect, useState, useRef } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useSearchStore } from '../stores/searchStore';
import { useUIStore } from '../stores/uiStore';
import { searchApi, jobsApi } from '../services/api';
import SearchResults from './SearchResults';

interface CollectionStatus {
  exists: boolean;
  point_count: number;
  status: string;
}

function SearchInterface() {
  const {
    searchParams,
    updateSearchParams,
    setResults,
    setLoading,
    setError,
    setCollections,
    collections,
  } = useSearchStore();
  const addToast = useUIStore((state) => state.addToast);

  const [collectionStatuses, setCollectionStatuses] = useState<Record<string, CollectionStatus>>({});
  const statusUpdateIntervalRef = useRef<number | null>(null);

  // Fetch collections (jobs)
  const { data: jobsData } = useQuery({
    queryKey: ['jobs'],
    queryFn: async () => {
      const response = await jobsApi.list();
      return response.data;
    },
  });

  // Fetch collection statuses
  const fetchCollectionStatuses = async () => {
    try {
      const response = await jobsApi.getCollectionsStatus();
      setCollectionStatuses(response.data);
      
      // Check if any collections are processing
      const hasProcessing = Object.values(response.data as Record<string, CollectionStatus>).some(
        (status) => status.status === 'processing' || status.status === 'created'
      );
      
      // Set up periodic updates if collections are processing
      if (hasProcessing && !statusUpdateIntervalRef.current) {
        statusUpdateIntervalRef.current = window.setInterval(fetchCollectionStatuses, 5000);
      } else if (!hasProcessing && statusUpdateIntervalRef.current) {
        window.clearInterval(statusUpdateIntervalRef.current);
        statusUpdateIntervalRef.current = null;
      }
    } catch (error) {
      console.error('Failed to fetch collection statuses:', error);
    }
  };

  // Initial fetch of collection statuses
  useEffect(() => {
    fetchCollectionStatuses();
    
    // Cleanup interval on unmount
    return () => {
      if (statusUpdateIntervalRef.current) {
        window.clearInterval(statusUpdateIntervalRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (jobsData) {
      const collections = jobsData.map((job: any) => `job_${job.id}`);
      setCollections(collections);
      // Set first collection as default if none selected
      if (!searchParams.collection && collections.length > 0) {
        updateSearchParams({ collection: collections[0] });
      }
    }
  }, [jobsData, searchParams.collection, setCollections, updateSearchParams]);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!searchParams.query.trim()) {
      addToast({ type: 'error', message: 'Please enter a search query' });
      return;
    }

    if (!searchParams.collection) {
      addToast({ type: 'error', message: 'Please select a collection' });
      return;
    }

    // No need to check if collection is ready since we only show ready collections

    setLoading(true);
    setError(null);

    try {
      const response = await searchApi.search({
        query: searchParams.query,
        collection: searchParams.collection,
        top_k: searchParams.topK,
        score_threshold: searchParams.scoreThreshold,
        search_type: searchParams.searchType,
        rerank_model: searchParams.rerankModel,
        hybrid_alpha: searchParams.hybridAlpha,
        hybrid_mode: searchParams.hybridMode,
        keyword_mode: searchParams.keywordMode,
      });

      setResults(response.data.results);
    } catch (error: any) {
      setError(error.response?.data?.detail || 'Search failed');
      addToast({ type: 'error', message: 'Search failed' });
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
                Collection
                <button
                  type="button"
                  onClick={() => fetchCollectionStatuses()}
                  className="ml-2 text-blue-600 hover:text-blue-800 text-xs"
                >
                  (refresh)
                </button>
              </label>
              <select
                id="collection"
                value={searchParams.collection}
                onChange={(e) => updateSearchParams({ collection: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="">Select a collection</option>
                {collections
                  .filter((collection) => {
                    const jobId = collection.replace('job_', '');
                    const status = collectionStatuses[jobId];
                    // Only show collections that are ready (completed with vectors)
                    return status?.status === 'completed' && status?.point_count > 0;
                  })
                  .map((collection) => {
                    const jobId = collection.replace('job_', '');
                    const job = jobsData?.find((j: any) => j.id === jobId);
                    const status = collectionStatuses[jobId];
                    
                    return (
                      <option 
                        key={collection} 
                        value={collection}
                      >
                        {job?.name || collection} ({status.point_count.toLocaleString()} vectors)
                      </option>
                    );
                  })}
              </select>
              <p className="mt-1 text-xs text-gray-500">
                Collections are created from your jobs
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
                  onChange={(e) => updateSearchParams({ searchType: e.target.checked ? 'hybrid' : 'vector' })}
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
                      value={searchParams.hybridMode || 'rerank'}
                      onChange={(e) => updateSearchParams({ hybridMode: e.target.value as 'rerank' | 'filter' })}
                      className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500"
                    >
                      <option value="rerank">Rerank (More Accurate)</option>
                      <option value="filter">Filter (Faster)</option>
                    </select>
                  </div>

                  {/* Keyword Mode */}
                  <div>
                    <label htmlFor="keyword-mode" className="block text-xs font-medium text-gray-600 mb-1">
                      Keyword Matching
                    </label>
                    <select
                      id="keyword-mode"
                      value={searchParams.keywordMode || 'any'}
                      onChange={(e) => updateSearchParams({ keywordMode: e.target.value as 'any' | 'all' })}
                      className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500"
                    >
                      <option value="any">Any Keywords</option>
                      <option value="all">All Keywords</option>
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

          {/* Submit Button */}
          <div className="pt-2">
            <button
              type="submit"
              disabled={!searchParams.collection}
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