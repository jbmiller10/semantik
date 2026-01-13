import { useState } from 'react';
import { useCollections } from '../hooks/useCollections';
import CollectionCard from './CollectionCard';
import CreateCollectionModal from './CreateCollectionModal';

function CollectionsDashboard() {
  const [searchQuery, setSearchQuery] = useState('');
  const [filterStatus, setFilterStatus] = useState<string>('all');
  const [showCreateModal, setShowCreateModal] = useState(false);

  // Use React Query hook to fetch collections
  const { data: collections = [], isLoading, error, refetch } = useCollections();

  // Filter collections
  const filteredCollections = collections.filter(collection => {
    // Search filter
    const matchesSearch = !searchQuery ||
      collection.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      collection.description?.toLowerCase().includes(searchQuery.toLowerCase());

    // Status filter
    const matchesStatus = filterStatus === 'all' || collection.status === filterStatus;

    return matchesSearch && matchesStatus;
  });

  // Sort collections by updated_at (most recent first)
  const sortedCollections = [...filteredCollections].sort((a, b) =>
    new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime()
  );

  const handleCreateSuccess = () => {
    setShowCreateModal(false);
    // Toast is shown by the modal itself
    // React Query will automatically refetch due to query invalidation
  };

  if (error && collections.length === 0) {
    return (
      <div className="text-center py-12">
        <p className="text-red-600 mb-4">Failed to load collections</p>
        <button
          onClick={() => refetch()}
          className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
        >
          Retry
        </button>
      </div>
    );
  }

  if (isLoading && collections.length === 0) {
    return (
      <div className="flex justify-center py-12">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600" role="status" aria-label="Loading collections"></div>
      </div>
    );
  }

  return (
    <div>
      {/* Header */}
      <div className="mb-8 p-6 glass-panel rounded-2xl">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6">
          <div>
            <h2 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-brand-700 to-accent-600">
              Collections
            </h2>
            <p className="mt-1 text-sm text-gray-500 font-medium">
              Manage your document collections and knowledge bases
            </p>
          </div>
          <button
            onClick={() => setShowCreateModal(true)}
            className="inline-flex items-center px-5 py-2.5 border border-transparent text-sm font-semibold rounded-xl shadow-lg shadow-brand-500/30 text-white bg-gradient-to-r from-brand-600 to-brand-500 hover:from-brand-500 hover:to-brand-400 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-brand-500 transition-all duration-200 transform hover:-translate-y-0.5"
          >
            <svg className="mr-2 -ml-1 h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            Create Collection
          </button>
        </div>

        {/* Search and Filters */}
        <div className="flex flex-col sm:flex-row gap-4">
          <div className="flex-1">
            <label htmlFor="collection-search" className="sr-only">Search collections by name or description</label>
            <div className="relative group">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <svg className="h-5 w-5 text-gray-400 group-focus-within:text-brand-500 transition-colors" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
              </div>
              <input
                type="text"
                id="collection-search"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="block w-full pl-10 pr-3 py-2.5 border border-gray-200 rounded-xl leading-5 bg-white/50 placeholder-gray-400 focus:outline-none focus:bg-white focus:ring-2 focus:ring-brand-500/20 focus:border-brand-500 transition-all duration-200 sm:text-sm"
                placeholder="Search collections..."
              />
            </div>
          </div>

          <div>
            <label htmlFor="status-filter" className="sr-only">Filter collections by status</label>
            <select
              id="status-filter"
              value={filterStatus}
              onChange={(e) => setFilterStatus(e.target.value)}
              className="block w-full sm:w-auto pl-3 pr-10 py-2.5 text-sm border-gray-200 focus:outline-none focus:ring-2 focus:ring-brand-500/20 focus:border-brand-500 rounded-xl bg-white/50 focus:bg-white transition-all duration-200"
            >
              <option value="all">All Status</option>
              <option value="pending">Pending</option>
              <option value="ready">Ready</option>
              <option value="processing">Processing</option>
              <option value="error">Error</option>
              <option value="degraded">Degraded</option>
            </select>
          </div>
        </div>

        {/* Results count */}
        {searchQuery || filterStatus !== 'all' ? (
          <p className="mt-2 text-sm text-gray-500 font-medium ml-1">
            Found {sortedCollections.length} collection{sortedCollections.length !== 1 ? 's' : ''}
          </p>
        ) : null}
      </div>

      {/* Empty State */}
      {collections.length === 0 ? (
        <div className="text-center py-16 glass-card rounded-2xl border-2 border-dashed border-gray-300/50">
          <div className="mx-auto h-16 w-16 bg-brand-50 rounded-full flex items-center justify-center mb-4">
            <svg className="h-8 w-8 text-brand-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
            </svg>
          </div>
          <h3 className="text-lg font-semibold text-gray-900">No collections yet</h3>
          <p className="mt-1 text-sm text-gray-500 max-w-sm mx-auto">Get started by creating your first collection to begin embedding your documents.</p>
          <div className="mt-6">
            <button
              onClick={() => setShowCreateModal(true)}
              className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-xl text-brand-700 bg-brand-50 hover:bg-brand-100 transition-colors"
            >
              <svg className="-ml-1 mr-2 h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
              Create Collection
            </button>
          </div>
        </div>
      ) : sortedCollections.length === 0 ? (
        <div className="text-center py-16 glass-panel rounded-2xl">
          <p className="text-gray-500 font-medium">No collections match your search criteria.</p>
        </div>
      ) : (
        /* Collection Grid */
        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3 animate-slide-up">
          {sortedCollections.map((collection) => (
            <CollectionCard key={collection.id} collection={collection} />
          ))}
        </div>
      )}

      {/* Create Collection Modal */}
      {showCreateModal && (
        <CreateCollectionModal
          onClose={() => setShowCreateModal(false)}
          onSuccess={handleCreateSuccess}
        />
      )}
    </div>
  );
}

export default CollectionsDashboard;