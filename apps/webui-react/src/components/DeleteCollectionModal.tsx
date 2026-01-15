import { useState } from 'react';
import { useDeleteCollection } from '../hooks/useCollections';

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
  const [confirmText, setConfirmText] = useState('');
  const [showDetails, setShowDetails] = useState(false);

  const deleteCollectionMutation = useDeleteCollection();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (confirmText === 'DELETE') {
      deleteCollectionMutation.mutate(collectionId, {
        onSuccess: () => {
          onSuccess();
          onClose();
        },
      });
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
      <div className="fixed inset-0 bg-void-950/80 backdrop-blur-sm z-[60]" onClick={onClose} />
      <div className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 glass-panel border border-white/10 rounded-2xl shadow-2xl z-[60] w-full max-w-lg">
        <div className="px-6 py-4 border-b border-white/10 bg-void-900/50">
          <h3 className="text-xl font-bold text-white tracking-tight">Delete Collection</h3>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="px-6 py-4 space-y-4">
            <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-4">
              <div className="flex">
                <div className="flex-shrink-0">
                  <svg className="h-5 w-5 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                  </svg>
                </div>
                <div className="ml-3">
                  <h3 className="text-sm font-bold text-red-400">
                    This action cannot be undone
                  </h3>
                  <div className="mt-2 text-sm text-red-300/80">
                    <p>You are about to permanently delete the collection <span className="text-white font-mono">{collectionName}</span>.</p>
                  </div>
                </div>
              </div>
            </div>

            <div>
              <button
                type="button"
                onClick={() => setShowDetails(!showDetails)}
                className="flex items-center text-sm font-bold text-gray-400 hover:text-white transition-colors"
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
                <div className="mt-2 bg-void-900/50 border border-white/5 rounded-xl p-4 text-sm">
                  <dl className="space-y-2">
                    <div className="flex justify-between">
                      <dt className="text-gray-500">Jobs:</dt>
                      <dd className="font-bold text-white">{stats.job_count}</dd>
                    </div>
                    <div className="flex justify-between">
                      <dt className="text-gray-500">Documents:</dt>
                      <dd className="font-bold text-white">{stats.total_files.toLocaleString()}</dd>
                    </div>
                    <div className="flex justify-between">
                      <dt className="text-gray-500">Vectors:</dt>
                      <dd className="font-bold text-white">{stats.total_vectors.toLocaleString()}</dd>
                    </div>
                    <div className="flex justify-between">
                      <dt className="text-gray-500">Storage:</dt>
                      <dd className="font-bold text-white">{formatBytes(stats.total_size)}</dd>
                    </div>
                  </dl>
                  <p className="mt-3 text-xs text-gray-500">
                    All database records, vector embeddings, and associated files will be permanently removed.
                  </p>
                </div>
              )}
            </div>

            <div>
              <label htmlFor="confirm-text" className="block text-xs font-bold text-gray-500 uppercase tracking-wider mb-2">
                Type <span className="font-mono text-red-400">DELETE</span> to confirm
              </label>
              <input
                type="text"
                id="confirm-text"
                value={confirmText}
                onChange={(e) => setConfirmText(e.target.value)}
                className="mt-1 block w-full bg-void-950/50 border border-white/10 rounded-xl px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-red-500/50 focus:border-transparent placeholder-gray-600 sm:text-sm"
                placeholder="Type DELETE here"
                autoComplete="off"
                autoFocus
                required
              />
            </div>
          </div>

          <div className="px-6 py-4 border-t border-white/10 flex justify-end space-x-3 bg-void-900/30">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 border border-white/10 rounded-xl text-sm font-medium text-gray-400 hover:bg-white/5 hover:text-white transition-colors"
              disabled={deleteCollectionMutation.isPending}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="px-4 py-2 bg-red-600 text-white rounded-xl text-sm font-bold shadow-lg shadow-red-600/20 hover:bg-red-500 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
              disabled={deleteCollectionMutation.isPending || confirmText !== 'DELETE'}
            >
              {deleteCollectionMutation.isPending ? 'Deleting...' : 'Delete Collection'}
            </button>
          </div>
        </form>
      </div>
    </>
  );
}

export default DeleteCollectionModal;