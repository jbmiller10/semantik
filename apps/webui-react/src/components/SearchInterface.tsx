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
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-0 space-y-8 animate-fade-in">
      <div className="glass-panel rounded-2xl p-8">
        <div className="mb-6">
          <h2 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-brand-700 to-accent-600">
            Search Collections
          </h2>
          <p className="mt-2 text-sm text-gray-500 font-medium">
            Configure your search parameters and find relevant documents
          </p>
        </div>
        <SearchForm collections={collections} />
      </div>

      <SearchResults onSelectSmallerModel={handleSelectSmallerModel} />
    </div>
  );
}

export default SearchInterface;
