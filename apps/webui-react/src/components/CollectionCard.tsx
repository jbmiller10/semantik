import { useUIStore } from '../stores/uiStore';
import type { Collection, CollectionStatus } from '../types/collection';

interface CollectionCardProps {
  collection: Collection;
}

function CollectionCard({ collection }: CollectionCardProps) {
  const { setShowCollectionDetailsModal } = useUIStore();

  const formatDate = (dateString: string | null | undefined) => {
    if (!dateString) {
      return 'N/A';
    }
    try {
      const date = new Date(dateString);
      return date.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
      });
    } catch (error) {
      console.error('Error formatting date:', error);
      return 'Invalid date';
    }
  };

  const formatNumber = (num: number | null | undefined) => {
    if (num === null || num === undefined) {
      return '0';
    }
    return num.toLocaleString();
  };

  const getModelDisplayName = (modelName: string) => {
    // Extract just the model name from the full path
    const parts = modelName.split('/');
    return parts[parts.length - 1] || modelName;
  };

  const getStatusColor = (status: CollectionStatus) => {
    switch (status) {
      case 'ready':
        return 'bg-green-100 text-green-800 border-green-200';
      case 'processing':
        return 'bg-blue-100 text-blue-800 border-blue-200';
      case 'error':
        return 'bg-red-100 text-red-800 border-red-200';
      case 'degraded':
        return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'pending':
        return 'bg-gray-100 text-gray-800 border-gray-200';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const getStatusIcon = (status: CollectionStatus) => {
    switch (status) {
      case 'ready':
        return (
          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
          </svg>
        );
      case 'processing':
        return (
          <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
        );
      case 'error':
        return (
          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
          </svg>
        );
      case 'degraded':
        return (
          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
          </svg>
        );
      default:
        return null;
    }
  };

  const isProcessing = collection.status === 'processing' || collection.isProcessing || collection.activeOperation;

  return (
    <div
      data-testid="collection-card"
      className={`relative group glass-card rounded-2xl overflow-hidden hover:-translate-y-1 ${isProcessing ? 'ring-2 ring-brand-400 ring-opacity-50' : ''
        }`}
    >
      {/* Processing indicator bar */}
      {isProcessing && (
        <div className="absolute top-0 left-0 right-0 h-1 bg-brand-100/50" role="progressbar">
          <div className="h-full bg-gradient-to-r from-brand-500 to-accent-500 animate-pulse" style={{ width: `${collection.activeOperation?.progress || 50}%` }} />
        </div>
      )}

      <div className="p-6">
        <div className="flex items-start justify-between mb-4">
          <div className="flex-1 min-w-0 pr-4">
            <h3 className="text-lg font-bold text-gray-900 truncate" title={collection.name}>
              {collection.name}
            </h3>
            {collection.description && (
              <p className="mt-1 text-sm text-gray-500 truncate" title={collection.description}>
                {collection.description}
              </p>
            )}
            <div className="mt-2 text-xs font-mono text-brand-600/80 bg-brand-50 px-2 py-1 rounded-md inline-block">
              {getModelDisplayName(collection.embedding_model)}
            </div>
          </div>
          <div className="flex-shrink-0">
            <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold shadow-sm ${getStatusColor(collection.status)}`}>
              {getStatusIcon(collection.status)}
              <span className="capitalize">{collection.status}</span>
            </span>
          </div>
        </div>

        {/* Active operation message */}
        {collection.activeOperation && (
          <div className="mb-4 p-3 bg-brand-50/80 rounded-xl border border-brand-100">
            <p className="text-xs font-medium text-brand-700 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-brand-500 animate-pulse"></span>
              {collection.activeOperation.type === 'index' && 'Indexing documents...'}
              {collection.activeOperation.type === 'append' && 'Adding new documents...'}
              {collection.activeOperation.type === 'reindex' && 'Reindexing collection...'}
              {collection.activeOperation.type === 'remove_source' && 'Removing documents...'}
              {collection.activeOperation.type === 'delete' && 'Deleting collection...'}
            </p>
          </div>
        )}

        {/* Status message if error or degraded */}
        {collection.status_message && (collection.status === 'error' || collection.status === 'degraded') && (
          <div className={`mb-4 p-3 rounded-xl border ${collection.status === 'error' ? 'bg-red-50 border-red-100' : 'bg-yellow-50 border-yellow-100'}`}>
            <p className={`text-xs font-medium ${collection.status === 'error' ? 'text-red-700' : 'text-yellow-700'}`}>
              {collection.status_message}
            </p>
          </div>
        )}

        <div className="grid grid-cols-2 gap-4 py-4 border-t border-gray-100">
          <div>
            <dt className="text-xs font-medium text-gray-400 uppercase tracking-wider">Documents</dt>
            <dd className="mt-0.5 text-sm font-bold text-gray-900">
              {formatNumber(collection.document_count)}
            </dd>
          </div>
          <div>
            <dt className="text-xs font-medium text-gray-400 uppercase tracking-wider">Vectors</dt>
            <dd className="mt-0.5 text-sm font-bold text-gray-900">
              {formatNumber(collection.vector_count)}
            </dd>
          </div>
        </div>

        <div className="flex items-center justify-between pt-4 mt-2 border-t border-gray-100">
          <div className="text-xs text-gray-400">
            Updated {formatDate(collection.updated_at)}
          </div>
          <button
            onClick={() => {
              setShowCollectionDetailsModal(collection.id);
            }}
            className="text-sm font-semibold text-brand-600 hover:text-brand-800 transition-colors flex items-center group/btn"
            disabled={!!isProcessing}
          >
            Manage
            <svg className="ml-1 w-4 h-4 transform group-hover/btn:translate-x-0.5 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
}

export default CollectionCard;
