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
  const cardBorderColor = isProcessing ? 'border-blue-500' : collection.status === 'error' ? 'border-red-300' : 'border-gray-200';
  const cardBackground = isProcessing ? 'bg-blue-50' : collection.status === 'error' ? 'bg-red-50' : 'bg-white';

  return (
    <div className={`relative rounded-lg border-2 shadow-sm hover:shadow-lg transition-all ${cardBorderColor} ${cardBackground} overflow-hidden`}>
      {/* Processing indicator bar */}
      {isProcessing && (
        <div className="absolute top-0 left-0 right-0 h-1 bg-blue-200" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={collection.activeOperation?.progress || 50} aria-label="Operation progress">
          <div className="h-full bg-blue-600 animate-pulse" style={{ width: `${collection.activeOperation?.progress || 50}%` }} />
        </div>
      )}

      <div className="p-5">
        <div className="flex items-start justify-between">
          <div className="flex-1 min-w-0">
            <h3 className="text-lg font-medium text-gray-900 truncate" title={collection.name}>
              {collection.name}
            </h3>
            {collection.description && (
              <p className="mt-1 text-sm text-gray-500 truncate" title={collection.description}>
                {collection.description}
              </p>
            )}
            <p className="mt-1 text-xs text-gray-500">
              {getModelDisplayName(collection.embedding_model)}
            </p>
          </div>
          <div className="ml-2 flex-shrink-0">
            <span className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(collection.status)}`} role="status" aria-label={`Collection status: ${collection.status}`}>
              <span aria-hidden="true">{getStatusIcon(collection.status)}</span>
              {collection.status}
            </span>
          </div>
        </div>

        {/* Active operation message */}
        {collection.activeOperation && (
          <div className="mt-3 p-2 bg-blue-100 rounded-md">
            <p className="text-xs text-blue-700">
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
          <div className={`mt-3 p-2 rounded-md ${collection.status === 'error' ? 'bg-red-100' : 'bg-yellow-100'}`}>
            <p className={`text-xs ${collection.status === 'error' ? 'text-red-700' : 'text-yellow-700'}`}>
              {collection.status_message}
            </p>
          </div>
        )}

        <div className="mt-4 grid grid-cols-2 gap-4">
          <div>
            <dt className="text-xs font-medium text-gray-500">Documents</dt>
            <dd className="mt-1 text-sm font-semibold text-gray-900">
              {formatNumber(collection.document_count)}
            </dd>
          </div>
          <div>
            <dt className="text-xs font-medium text-gray-500">Vectors</dt>
            <dd className="mt-1 text-sm font-semibold text-gray-900">
              {formatNumber(collection.vector_count)}
            </dd>
          </div>
        </div>

        <div className="mt-4">
          <dt className="text-xs font-medium text-gray-500">Last updated</dt>
          <dd className="mt-1 text-sm text-gray-900">{formatDate(collection.updated_at)}</dd>
        </div>
      </div>

      <div className="bg-gray-50 px-5 py-3">
        <button
          onClick={() => {
            console.log('Manage button clicked for collection:', collection.id);
            setShowCollectionDetailsModal(collection.id);
          }}
          className="w-full flex justify-center items-center px-4 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
          disabled={!!isProcessing}
        >
          <svg
            className="mr-2 -ml-1 h-4 w-4 text-gray-500"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"
            />
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
            />
          </svg>
          Manage
        </button>
      </div>
    </div>
  );
}

export default CollectionCard;