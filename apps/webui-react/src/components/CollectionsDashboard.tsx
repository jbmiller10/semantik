import { useState } from 'react';
import { useCollections } from '../hooks/useCollections';
import { useInProgressConversations } from '../hooks/useInProgressConversations';
import { useAnimationEnabled } from '../contexts/AnimationContext';
import { withAnimation } from '../utils/animationClasses';
import CollectionCard from './CollectionCard';
import { CollectionWizard } from './wizard';

function CollectionsDashboard() {
  const [searchQuery, setSearchQuery] = useState('');
  const [filterStatus, setFilterStatus] = useState<string>('all');
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [resumeConversationId, setResumeConversationId] = useState<string | undefined>(undefined);
  const animationEnabled = useAnimationEnabled();

  // Use React Query hook to fetch collections
  const { data: collections = [], isLoading, error, refetch } = useCollections();

  // Fetch in-progress conversations for resume banner
  const { conversations: inProgressConversations } = useInProgressConversations();

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
    setResumeConversationId(undefined);
    // Toast is shown by the modal itself
    // React Query will automatically refetch due to query invalidation
  };

  if (error && collections.length === 0) {
    return (
      <div className="text-center py-12">
        <p className="text-error mb-4">Failed to load collections</p>
        <button
          onClick={() => refetch()}
          className="btn-primary"
        >
          Retry
        </button>
      </div>
    );
  }

  if (isLoading && collections.length === 0) {
    return (
      <div className="flex justify-center py-12">
        <div className={withAnimation('rounded-full h-8 w-8 border-b-2 border-ink-900 dark:border-paper-100', animationEnabled, 'animate-spin')} role="status" aria-label="Loading collections"></div>
      </div>
    );
  }

  return (
    <div>
      {/* Header */}
      <div className="mb-8 p-6 panel rounded-xl">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6">
          <div>
            <h2 className="text-2xl font-serif font-semibold text-[var(--text-primary)] tracking-tight">
              Collections
            </h2>
            <p className="mt-1 text-sm text-[var(--text-secondary)]">
              Manage your document collections and knowledge bases
            </p>
          </div>
          <button
            onClick={() => setShowCreateModal(true)}
            className="btn-primary inline-flex items-center"
          >
            <svg className="mr-2 -ml-1 h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            New Collection
          </button>
        </div>

        {/* Search and Filters */}
        <div className="flex flex-col sm:flex-row gap-4">
          <div className="flex-1">
            <label htmlFor="collection-search" className="sr-only">Search collections by name or description</label>
            <div className="relative group">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <svg className="h-5 w-5 text-[var(--text-muted)] group-focus-within:text-[var(--accent-primary)] transition-colors" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
              </div>
              <input
                type="text"
                id="collection-search"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="input-field w-full pl-10 pr-3 py-2.5 rounded-lg"
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
                className="input-field w-full sm:w-auto pl-3 pr-10 py-2.5 rounded-lg appearance-none cursor-pointer"
              >
                <option value="all">All Status</option>
                <option value="pending">Pending</option>
                <option value="ready">Ready</option>
                <option value="processing">Processing</option>
                <option value="error">Error</option>
              </select>
              <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-[var(--text-muted)]">
                <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
                </svg>
              </div>
            </div>
          </div>
        </div>

        {/* Results count */}
        {searchQuery || filterStatus !== 'all' ? (
          <p className="mt-2 text-xs font-medium text-[var(--text-muted)] ml-1 uppercase tracking-wide">
            Found {sortedCollections.length} collection{sortedCollections.length !== 1 ? 's' : ''}
          </p>
        ) : null}
      </div>

      {/* In-Progress Conversations Banner */}
      {inProgressConversations.length > 0 && (
        <div className="mb-4 p-3 bg-blue-500/10 border border-blue-500/30 rounded-lg">
          <div className="flex items-center justify-between">
            <span className="text-sm text-blue-400">
              You have {inProgressConversations.length} in-progress collection setup{inProgressConversations.length > 1 ? 's' : ''}
            </span>
            <button
              onClick={() => {
                setResumeConversationId(inProgressConversations[0].id);
                setShowCreateModal(true);
              }}
              className="text-sm text-blue-400 hover:text-blue-300 underline"
            >
              Continue setup →
            </button>
          </div>
        </div>
      )}

      {/* Empty State */}
      {collections.length === 0 ? (
        <div className="text-center py-16 card rounded-xl">
          <div className="mx-auto h-16 w-16 bg-[var(--bg-tertiary)] rounded-full flex items-center justify-center mb-4 border border-[var(--border)]">
            <svg className="h-8 w-8 text-[var(--accent-primary)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
            </svg>
          </div>
          <h3 className="text-lg font-semibold text-[var(--text-primary)]">No collections yet</h3>
          <p className="mt-1 text-sm text-[var(--text-secondary)] max-w-sm mx-auto">Get started by creating your first collection to begin embedding your documents.</p>
          <div className="mt-6">
            <button
              onClick={() => setShowCreateModal(true)}
              className="btn-secondary inline-flex items-center"
            >
              <svg className="-ml-1 mr-2 h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
              Create Collection
            </button>
          </div>
        </div>
      ) : sortedCollections.length === 0 ? (
        <div className="text-center py-16 panel rounded-xl">
          <p className="text-[var(--text-secondary)] font-medium">No collections match your search criteria.</p>
        </div>
      ) : (
        /* Collection Grid */
        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3 animate-slide-up">
          {sortedCollections.map((collection) => (
            <CollectionCard key={collection.id} collection={collection} />
          ))}
        </div>
      )}

      {/* Create Collection Wizard */}
      {showCreateModal && (
        <CollectionWizard
          onClose={() => {
            setShowCreateModal(false);
            setResumeConversationId(undefined);
          }}
          onSuccess={handleCreateSuccess}
          resumeConversationId={resumeConversationId}
        />
      )}
    </div>
  );
}

export default CollectionsDashboard;
