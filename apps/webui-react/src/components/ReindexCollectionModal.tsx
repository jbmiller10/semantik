import { useState } from 'react';
import { useUIStore } from '../stores/uiStore';
import { useCollectionStore } from '../stores/collectionStore';
import type { ReindexRequest } from '../types/collection';

interface ReindexCollectionModalProps {
  collectionName: string;
  configChanges: {
    chunk_size?: number;
    chunk_overlap?: number;
    instruction?: string;
  };
  onClose: () => void;
  onSuccess: () => void;
}

function ReindexCollectionModal({ collectionName, configChanges, onClose, onSuccess }: ReindexCollectionModalProps) {
  const { addToast } = useUIStore();
  const { collections, reindexCollection } = useCollectionStore();
  const [confirmText, setConfirmText] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Find the collection by name to get its ID
  const collection = Array.from(collections.values()).find(c => c.name === collectionName);
  
  const expectedConfirmText = `reindex ${collectionName}`;
  const isConfirmValid = confirmText === expectedConfirmText;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!isConfirmValid || isSubmitting || !collection) return;

    setIsSubmitting(true);
    try {
      // Prepare the reindex request (only chunk_size and chunk_overlap are supported)
      const request: ReindexRequest = {};
      if (configChanges.chunk_size !== undefined) {
        request.chunk_size = configChanges.chunk_size;
      }
      if (configChanges.chunk_overlap !== undefined) {
        request.chunk_overlap = configChanges.chunk_overlap;
      }
      
      // Call the store method to start re-indexing
      await reindexCollection(collection.id, request);
      
      addToast({
        type: 'success',
        message: `Re-indexing started for collection "${collectionName}". This may take some time.`,
      });
      
      onSuccess();
    } catch (error) {
      console.error('Failed to start re-indexing:', error);
      addToast({
        type: 'error',
        message: 'Failed to start re-indexing. Please try again.',
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleCancel = () => {
    if (!isSubmitting) {
      onClose();
    }
  };

  // Handle escape key
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape' && !isSubmitting) {
      onClose();
    }
  };

  return (
    <>
      <div 
        className="fixed inset-0 bg-black bg-opacity-50 z-[60]" 
        onClick={handleCancel}
      />
      <div className="fixed inset-0 z-[60] overflow-y-auto">
        <div className="flex min-h-full items-center justify-center p-4">
          <div 
            className="relative bg-white rounded-lg shadow-xl max-w-md w-full p-6"
            onKeyDown={handleKeyDown}
          >
            <h2 className="text-lg font-semibold text-gray-900 mb-4">
              Re-index Collection
            </h2>

            <div className="mb-6">
              <div className="bg-red-50 border border-red-200 rounded-md p-4">
                <div className="flex">
                  <div className="flex-shrink-0">
                    <svg className="h-5 w-5 text-red-400" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                    </svg>
                  </div>
                  <div className="ml-3">
                    <h3 className="text-sm font-medium text-red-800">
                      Warning: This action cannot be undone
                    </h3>
                    <div className="mt-2 text-sm text-red-700">
                      <p>Re-indexing will:</p>
                      <ul className="list-disc list-inside mt-1">
                        <li>Delete all existing vectors for this collection</li>
                        <li>Re-process all documents with the new configuration</li>
                        <li>Make the collection unavailable during processing</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>

              <div className="mt-4">
                <h4 className="text-sm font-medium text-gray-700 mb-2">Configuration Changes:</h4>
                <ul className="text-sm text-gray-600 space-y-1">
                  {configChanges.chunk_size !== undefined && (
                    <li>• Chunk size: {configChanges.chunk_size} tokens</li>
                  )}
                  {configChanges.chunk_overlap !== undefined && (
                    <li>• Chunk overlap: {configChanges.chunk_overlap} tokens</li>
                  )}
                  {configChanges.instruction !== undefined && (
                    <li>• Embedding instruction: {configChanges.instruction || '(removed)'}</li>
                  )}
                </ul>
              </div>
            </div>

            <form onSubmit={handleSubmit}>
              <div className="mb-4">
                <label htmlFor="confirm" className="block text-sm font-medium text-gray-700 mb-2">
                  To confirm, type <span className="font-mono bg-gray-100 px-1 py-0.5 rounded">reindex {collectionName}</span>
                </label>
                <input
                  id="confirm"
                  type="text"
                  value={confirmText}
                  onChange={(e) => setConfirmText(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-red-500 focus:border-red-500"
                  placeholder="Type the confirmation text"
                  autoComplete="off"
                  autoFocus
                />
              </div>

              <div className="flex gap-3 justify-end">
                <button
                  type="button"
                  onClick={handleCancel}
                  disabled={isSubmitting}
                  className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={!isConfirmValid || isSubmitting}
                  className="px-4 py-2 text-sm font-medium text-white bg-red-600 border border-transparent rounded-md hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isSubmitting ? 'Starting Re-index...' : 'Re-index Collection'}
                </button>
              </div>
            </form>
          </div>
        </div>
      </div>
    </>
  );
}

export default ReindexCollectionModal;