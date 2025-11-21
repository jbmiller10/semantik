import { useEffect, useRef } from 'react';
import { useSearchStore } from '../stores/searchStore';
import { useCollections } from '../hooks/useCollections';
import SearchResults from './SearchResults';
import SearchForm from './search/SearchForm';
import { useRerankingAvailability } from '../hooks/useRerankingAvailability';

function SearchInterface() {
  const {
    validateAndUpdateSearchParams,
  } = useSearchStore();

  // Use React Query hook to fetch collections
  const { data: collections = [], refetch: refetchCollections } = useCollections();

  // Check reranking availability
  useRerankingAvailability();

  const statusUpdateIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const isTestEnv = import.meta.env.MODE === 'test';

  // Check if any collections are processing and set up auto-refresh
  useEffect(() => {
    if (isTestEnv) {
      return () => {
        if (statusUpdateIntervalRef.current) {
          clearInterval(statusUpdateIntervalRef.current);
          statusUpdateIntervalRef.current = null;
        }
      };
    }

    const hasProcessing = collections.some(
      (col) => col.status === 'processing' || col.status === 'pending'
    );

    if (hasProcessing && !statusUpdateIntervalRef.current) {
      statusUpdateIntervalRef.current = setInterval(() => {
        refetchCollections();
      }, 5000);
    } else if (!hasProcessing && statusUpdateIntervalRef.current) {
      if (statusUpdateIntervalRef.current) {
        clearInterval(statusUpdateIntervalRef.current);
        statusUpdateIntervalRef.current = null;
      }
    }

    return () => {
      if (statusUpdateIntervalRef.current) {
        clearInterval(statusUpdateIntervalRef.current);
        statusUpdateIntervalRef.current = null;
      }
    };
  }, [collections, isTestEnv, refetchCollections]);

  const handleSelectSmallerModel = (model: string) => {
    validateAndUpdateSearchParams({ rerankModel: model });
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-200">
        <h2 className="text-2xl font-bold text-gray-900 mb-6">Search Collections</h2>
        <SearchForm collections={collections} />
      </div>

      <SearchResults onSelectSmallerModel={handleSelectSmallerModel} />
    </div>
  );
}

export default SearchInterface;
