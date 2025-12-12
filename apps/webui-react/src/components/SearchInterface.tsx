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
  // Note: Polling for processing collections is handled by React Query's refetchInterval in useCollections
  const { data: collections = [] } = useCollections();

  // Check reranking availability
  useRerankingAvailability();

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
