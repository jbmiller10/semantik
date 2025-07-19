import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { collectionsV2Api } from '../services/api/v2/collections';
import { useUIStore } from '../stores/uiStore';

interface CollectionStats {
  total_files: number;
  total_vectors: number;
  total_size: number;
  job_count: number;
}

interface DeleteCollectionModalProps {
  collectionId: string;
  collectionName: string;
  stats: CollectionStats;
  onClose: () => void;
  onSuccess: () => void;
}

function DeleteCollectionModal({
  collectionId,
  collectionName,
  stats,
  onClose,
  onSuccess,
}: DeleteCollectionModalProps) {
  const { addToast } = useUIStore();
  const [confirmText, setConfirmText] = useState('');
  const [showDetails, setShowDetails] = useState(false);

  const mutation = useMutation({
    mutationFn: async () => {
      return collectionsV2Api.delete(collectionId);
    },
    onSuccess: () => {
      addToast({
        type: 'success',
        message: `Collection "${collectionName}" deleted successfully`,
      });
      onSuccess();
    },
    onError: (error: any) => {
      const message = error.response?.data?.detail || 'Failed to delete collection';
      addToast({ type: 'error', message });
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (confirmText === 'DELETE') {
      mutation.mutate();
    }
  };

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <>
      <div className="fixed inset-0 bg-black bg-opacity-50 z-[60]" onClick={onClose} />
      <div className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 bg-white rounded-lg shadow-xl z-[60] w-full max-w-lg">
        <div className="px-6 py-4 border-b">
          <h3 className="text-lg font-medium text-gray-900">Delete Collection</h3>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="px-6 py-4 space-y-4">
            <div className="bg-red-50 border-l-4 border-red-400 p-4">
              <div className="flex">
                <div className="flex-shrink-0">
                  <svg className="h-5 w-5 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                  </svg>
                </div>
                <div className="ml-3">
                  <h3 className="text-sm font-medium text-red-800">
                    This action cannot be undone
                  </h3>
                  <div className="mt-2 text-sm text-red-700">
                    <p>You are about to permanently delete the collection "{collectionName}".</p>
                  </div>
                </div>
              </div>
            </div>

            <div>
              <button
                type="button"
                onClick={() => setShowDetails(!showDetails)}
                className="flex items-center text-sm text-gray-600 hover:text-gray-900"
              >
                <svg
                  className={`h-4 w-4 mr-1 transform transition-transform ${showDetails ? 'rotate-90' : ''}`}
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
                What will be deleted?
              </button>
              
              {showDetails && (
                <div className="mt-2 bg-gray-50 rounded-lg p-4 text-sm">
                  <dl className="space-y-2">
                    <div className="flex justify-between">
                      <dt className="text-gray-500">Jobs:</dt>
                      <dd className="font-medium text-gray-900">{stats.job_count}</dd>
                    </div>
                    <div className="flex justify-between">
                      <dt className="text-gray-500">Documents:</dt>
                      <dd className="font-medium text-gray-900">{stats.total_files.toLocaleString()}</dd>
                    </div>
                    <div className="flex justify-between">
                      <dt className="text-gray-500">Vectors:</dt>
                      <dd className="font-medium text-gray-900">{stats.total_vectors.toLocaleString()}</dd>
                    </div>
                    <div className="flex justify-between">
                      <dt className="text-gray-500">Storage:</dt>
                      <dd className="font-medium text-gray-900">{formatBytes(stats.total_size)}</dd>
                    </div>
                  </dl>
                  <p className="mt-3 text-xs text-gray-600">
                    All database records, vector embeddings, and associated files will be permanently removed.
                  </p>
                </div>
              )}
            </div>

            <div>
              <label htmlFor="confirm-text" className="block text-sm font-medium text-gray-700">
                Type <span className="font-mono font-bold">DELETE</span> to confirm
              </label>
              <input
                type="text"
                id="confirm-text"
                value={confirmText}
                onChange={(e) => setConfirmText(e.target.value)}
                className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-red-500 focus:border-red-500 sm:text-sm"
                placeholder="Type DELETE here"
                autoComplete="off"
                required
              />
            </div>
          </div>

          <div className="px-6 py-4 border-t flex justify-end space-x-3">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 hover:bg-gray-50"
              disabled={mutation.isPending}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="px-4 py-2 bg-red-600 text-white rounded-md text-sm font-medium hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed"
              disabled={mutation.isPending || confirmText !== 'DELETE'}
            >
              {mutation.isPending ? 'Deleting...' : 'Delete Collection'}
            </button>
          </div>
        </form>
      </div>
    </>
  );
}

export default DeleteCollectionModal;