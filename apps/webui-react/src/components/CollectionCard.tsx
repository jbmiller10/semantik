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
        return 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20';
      case 'processing':
        return 'bg-blue-500/10 text-blue-400 border-blue-500/20';
      case 'error':
        return 'bg-red-500/10 text-red-400 border-red-500/20';
      case 'degraded':
        return 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20';
      case 'pending':
        return 'bg-gray-500/10 text-gray-400 border-gray-500/20';
      default:
        return 'bg-gray-500/10 text-gray-400 border-gray-500/20';
    }
  };

  const getStatusIcon = (status: CollectionStatus) => {
    switch (status) {
      case 'ready':
        return (
          <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
          </svg>
        );
      case 'processing':
        return (
          <svg className="animate-spin h-3.5 w-3.5" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
        );
      case 'error':
        return (
          <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
          </svg>
        );
      case 'degraded':
        return (
          <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 20 20">
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
      className={`relative group glass-card rounded-xl overflow-hidden hover:-translate-y-1 ${isProcessing ? 'ring-1 ring-signal-500/50' : ''
        }`}
    >
      {/* Processing indicator bar */}
      {isProcessing && (
        <div className="absolute top-0 left-0 right-0 h-0.5 bg-void-800" role="progressbar">
          <div className="h-full bg-gradient-to-r from-signal-600 to-signal-400 animate-pulse" style={{ width: `${collection.activeOperation?.progress || 50}%` }} />
        </div>
      )}

      <div className="p-6">
        <div className="flex items-start justify-between mb-4">
          <div className="flex-1 min-w-0 pr-4">
            <h3 className="text-lg font-bold text-white truncate tracking-tight" title={collection.name}>
              {collection.name}
            </h3>
            {collection.description && (
              <p className="mt-1 text-sm text-gray-400 truncate" title={collection.description}>
                {collection.description}
              </p>
            )}
            <div className="mt-3 text-[10px] font-mono uppercase tracking-wider text-signal-300 bg-signal-500/10 border border-signal-500/20 px-2 py-1 rounded inline-block">
              {getModelDisplayName(collection.embedding_model)}
            </div>
          </div>
          <div className="flex-shrink-0">
            <span className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider border ${getStatusColor(collection.status)}`}>
              {getStatusIcon(collection.status)}
              <span className="capitalize">{collection.status}</span>
            </span>
          </div>
        </div>

        {/* Active operation message */}
        {collection.activeOperation && (
          <div className="mb-4 p-3 bg-void-800/50 rounded-lg border border-white/5">
            <p className="text-xs font-medium text-signal-300 flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-signal-500 animate-pulse"></span>
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
          <div className={`mb-4 p-3 rounded-lg border ${collection.status === 'error' ? 'bg-red-500/10 border-red-500/20' : 'bg-yellow-500/10 border-yellow-500/20'}`}>
            <p className={`text-xs font-medium ${collection.status === 'error' ? 'text-red-300' : 'text-yellow-300'}`}>
              {collection.status_message}
            </p>
          </div>
        )}

        <div className="grid grid-cols-2 gap-4 py-4 border-t border-white/5 mt-4">
          <div>
            <dt className="text-[10px] font-bold text-gray-500 uppercase tracking-widest">Documents</dt>
            <dd className="mt-0.5 text-sm font-bold text-gray-200">
              {formatNumber(collection.document_count)}
            </dd>
          </div>
          <div>
            <dt className="text-[10px] font-bold text-gray-500 uppercase tracking-widest">Vectors</dt>
            <dd className="mt-0.5 text-sm font-bold text-gray-200">
              {formatNumber(collection.vector_count)}
            </dd>
          </div>
        </div>

        <div className="flex items-center justify-between pt-4 mt-2 border-t border-white/5">
          <div className="text-xs text-gray-500 font-medium">
            Updated {formatDate(collection.updated_at)}
          </div>
          <button
            onClick={() => {
              setShowCollectionDetailsModal(collection.id);
            }}
            className="text-xs font-bold uppercase tracking-wider text-signal-400 hover:text-signal-300 transition-colors flex items-center group/btn"
            disabled={!!isProcessing}
          >
            Manage
            <svg className="ml-1 w-3.5 h-3.5 transform group-hover/btn:translate-x-0.5 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
}

export default CollectionCard;
