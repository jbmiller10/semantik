import { useState } from 'react';
import { useCollectionStore } from '../stores/collectionStore';
import { useUIStore } from '../stores/uiStore';
import { useNavigate } from 'react-router-dom';
import { getInputClassName } from '../utils/formStyles';
import type { Collection } from '../types/collection';

interface AddDataToCollectionModalProps {
  collection: Collection;
  onClose: () => void;
  onSuccess: () => void;
}

function AddDataToCollectionModal({
  collection,
  onClose,
  onSuccess,
}: AddDataToCollectionModalProps) {
  const { addSource } = useCollectionStore();
  const { addToast } = useUIStore();
  const navigate = useNavigate();
  const [sourcePath, setSourcePath] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!sourcePath.trim()) {
      addToast({ type: 'error', message: 'Please enter a directory path' });
      return;
    }
    
    setIsSubmitting(true);
    
    try {
      await addSource(collection.id, sourcePath.trim(), {
        chunk_size: collection.chunk_size,
        chunk_overlap: collection.chunk_overlap,
      });
      
      // Navigate to collection detail page to show operation progress
      navigate(`/collections/${collection.id}`);
      addToast({
        message: 'Data source added, indexing started',
        type: 'success'
      });
      onSuccess();
    } catch (error) {
      addToast({
        message: error instanceof Error ? error.message : 'Failed to add data source',
        type: 'error'
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <>
      <div className="fixed inset-0 bg-black bg-opacity-50 z-[60]" onClick={onClose} />
      <div className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 bg-white rounded-lg shadow-xl z-[60] w-full max-w-lg">
        <div className="px-6 py-4 border-b">
          <h3 className="text-lg font-medium text-gray-900">Add Data to Collection</h3>
          <p className="mt-1 text-sm text-gray-500">
            Add new documents to "{collection.name}" using the same settings
          </p>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="px-6 py-4 space-y-4">
            {/* Source Path Input */}
            <div>
              <label htmlFor="sourcePath" className="block text-sm font-medium text-gray-700">
                Source Directory Path
              </label>
              <input
                type="text"
                id="sourcePath"
                value={sourcePath}
                onChange={(e) => setSourcePath(e.target.value)}
                placeholder="/path/to/documents"
                className={getInputClassName(false, isSubmitting)}
                required
                autoFocus
              />
              <p className="mt-1 text-xs text-gray-500">
                All files in this directory will be scanned and added to the collection
              </p>
            </div>

            {/* Settings Summary */}
            <div className="bg-gray-50 rounded-lg p-4">
              <h4 className="text-sm font-medium text-gray-900 mb-2">Collection Settings</h4>
              <dl className="text-xs space-y-1">
                <div className="flex justify-between">
                  <dt className="text-gray-500">Embedding Model:</dt>
                  <dd className="text-gray-900 font-mono text-xs">{collection.embedding_model}</dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-gray-500">Chunk Size:</dt>
                  <dd className="text-gray-900">{collection.chunk_size} characters</dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-gray-500">Chunk Overlap:</dt>
                  <dd className="text-gray-900">{collection.chunk_overlap} characters</dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-gray-500">Status:</dt>
                  <dd className="text-gray-900 capitalize">{collection.status}</dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-gray-500">Documents:</dt>
                  <dd className="text-gray-900">{collection.document_count}</dd>
                </div>
              </dl>
            </div>

            <div className="bg-blue-50 rounded-lg p-4">
              <div className="flex">
                <div className="flex-shrink-0">
                  <svg className="h-5 w-5 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
                <div className="ml-3">
                  <p className="text-sm text-blue-800">
                    Duplicate files will be automatically skipped. Only new or modified files will be processed.
                  </p>
                </div>
              </div>
            </div>
          </div>

          <div className="px-6 py-4 border-t flex justify-end space-x-3">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 hover:bg-gray-50"
              disabled={isSubmitting}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="px-4 py-2 bg-blue-600 text-white rounded-md text-sm font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
              disabled={isSubmitting}
            >
              {isSubmitting ? 'Adding Source...' : 'Add Data'}
            </button>
          </div>
        </form>
      </div>
    </>
  );
}

export default AddDataToCollectionModal;