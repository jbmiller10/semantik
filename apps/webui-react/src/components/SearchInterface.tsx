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
      <div className="panel rounded-xl p-8">
        <div className="mb-6">
          <h2 className="text-2xl font-serif font-semibold text-[var(--text-primary)] tracking-tight">
            Search Knowledge Base
          </h2>
          <p className="mt-1 text-sm text-[var(--text-secondary)]">
            Find documents and semantic context across all collections
          </p>
        </div>
        <SearchForm collections={collections} />
      </div>

      <SearchResults onSelectSmallerModel={handleSelectSmallerModel} />
    </div>
  );
}

export default SearchInterface;
