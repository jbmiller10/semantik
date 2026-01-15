import { useState } from 'react';
import { useCollections } from '../hooks/useCollections';
import { useAnimationEnabled } from '../contexts/AnimationContext';
import { withAnimation } from '../utils/animationClasses';
import CollectionCard from './CollectionCard';
import CreateCollectionModal from './CreateCollectionModal';

function CollectionsDashboard() {
  const [searchQuery, setSearchQuery] = useState('');
  const [filterStatus, setFilterStatus] = useState<string>('all');
  const [showCreateModal, setShowCreateModal] = useState(false);
  const animationEnabled = useAnimationEnabled();

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
        <p className="text-red-400 mb-4">Failed to load collections</p>
        <button
          onClick={() => refetch()}
          className="px-4 py-2 bg-signal-600 text-white rounded hover:bg-signal-700 transition-colors"
        >
          Retry
        </button>
      </div>
    );
  }

  if (isLoading && collections.length === 0) {
    return (
      <div className="flex justify-center py-12">
        <div className={withAnimation('rounded-full h-8 w-8 border-b-2 border-signal-500', animationEnabled, 'animate-spin')} role="status" aria-label="Loading collections"></div>
      </div>
    );
  }

  return (
    <div>
      {/* Header */}
      <div className="mb-8 p-6 glass-panel rounded-2xl">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6">
          <div>
            <h2 className="text-3xl font-bold text-white tracking-tight">
              Collections
            </h2>
            <p className="mt-1 text-sm text-gray-400 font-medium">
              Manage your document collections and knowledge bases
            </p>
          </div>
          <button
            onClick={() => setShowCreateModal(true)}
            className="inline-flex items-center px-5 py-2.5 border border-transparent text-sm font-bold uppercase tracking-wide rounded-xl shadow-lg shadow-signal-600/20 text-white bg-signal-600 hover:bg-signal-500 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-void-900 focus:ring-signal-500 transition-all duration-200 transform hover:-translate-y-0.5"
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
                <svg className="h-5 w-5 text-gray-500 group-focus-within:text-signal-500 transition-colors" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
              </div>
              <input
                type="text"
                id="collection-search"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="block w-full pl-10 pr-3 py-2.5 input-glass rounded-xl text-sm"
                placeholder="Search collections..."
              />
            </div>
          </div>

          <div>
            <label htmlFor="status-filter" className="sr-only">Filter collections by status</label>
            <div className="relative">
              <select
                id="status-filter"
                value={filterStatus}
                onChange={(e) => setFilterStatus(e.target.value)}
                className="block w-full sm:w-auto pl-3 pr-10 py-2.5 text-sm input-glass rounded-xl appearance-none cursor-pointer"
              >
                <option value="all">All Status</option>
                <option value="pending">Pending</option>
                <option value="ready">Ready</option>
                <option value="processing">Processing</option>
                <option value="error">Error</option>
                <option value="degraded">Degraded</option>
              </select>
              <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-gray-400">
                <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
                </svg>
              </div>
            </div>
          </div>
        </div>

        {/* Results count */}
        {searchQuery || filterStatus !== 'all' ? (
          <p className="mt-2 text-xs font-medium text-gray-500 ml-1 uppercase tracking-wide">
            Found {sortedCollections.length} collection{sortedCollections.length !== 1 ? 's' : ''}
          </p>
        ) : null}
      </div>

      {/* Empty State */}
      {collections.length === 0 ? (
        <div className="text-center py-16 glass-card rounded-2xl border border-white/5 bg-void-900/30">
          <div className="mx-auto h-16 w-16 bg-void-800 rounded-full flex items-center justify-center mb-4 ring-1 ring-white/10">
            <svg className="h-8 w-8 text-signal-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
            </svg>
          </div>
          <h3 className="text-lg font-bold text-white">No collections yet</h3>
          <p className="mt-1 text-sm text-gray-400 max-w-sm mx-auto">Get started by creating your first collection to begin embedding your documents.</p>
          <div className="mt-6">
            <button
              onClick={() => setShowCreateModal(true)}
              className="inline-flex items-center px-4 py-2 border border-white/10 text-sm font-bold uppercase tracking-wide rounded-xl text-white bg-void-800 hover:bg-void-700 transition-colors"
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
          <p className="text-gray-400 font-medium">No collections match your search criteria.</p>
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