import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { collectionsApi } from '../services/api';
import { useUIStore } from '../stores/uiStore';

interface Configuration {
  model_name: string;
  chunk_size: number;
  chunk_overlap: number;
  quantization: string;
  vector_dim: number | null;
  instruction: string | null;
}

interface AddDataToCollectionModalProps {
  collectionName: string;
  configuration: Configuration;
  onClose: () => void;
  onSuccess: () => void;
}

function AddDataToCollectionModal({
  collectionName,
  configuration,
  onClose,
  onSuccess,
}: AddDataToCollectionModalProps) {
  const { addToast } = useUIStore();
  const [directory, setDirectory] = useState('');
  const [description, setDescription] = useState('');

  const mutation = useMutation({
    mutationFn: async () => {
      return collectionsApi.addData(collectionName, directory, description);
    },
    onSuccess: () => {
      onSuccess();
    },
    onError: (error: any) => {
      const message = error.response?.data?.detail || 'Failed to create job';
      addToast({ type: 'error', message });
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!directory.trim()) {
      addToast({ type: 'error', message: 'Please enter a directory path' });
      return;
    }
    mutation.mutate();
  };

  return (
    <>
      <div className="fixed inset-0 bg-black bg-opacity-50 z-[60]" onClick={onClose} />
      <div className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 bg-white rounded-lg shadow-xl z-[60] w-full max-w-lg">
        <div className="px-6 py-4 border-b">
          <h3 className="text-lg font-medium text-gray-900">Add Data to Collection</h3>
          <p className="mt-1 text-sm text-gray-500">
            Add new documents to "{collectionName}" using the same settings
          </p>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="px-6 py-4 space-y-4">
            {/* Directory Input */}
            <div>
              <label htmlFor="directory" className="block text-sm font-medium text-gray-700">
                Directory Path
              </label>
              <input
                type="text"
                id="directory"
                value={directory}
                onChange={(e) => setDirectory(e.target.value)}
                placeholder="/path/to/documents"
                className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                required
              />
              <p className="mt-1 text-xs text-gray-500">
                All files in this directory will be scanned and added to the collection
              </p>
            </div>

            {/* Description Input */}
            <div>
              <label htmlFor="description" className="block text-sm font-medium text-gray-700">
                Description (Optional)
              </label>
              <textarea
                id="description"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                rows={2}
                className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                placeholder="Brief description of what you're adding..."
              />
            </div>

            {/* Settings Summary */}
            <div className="bg-gray-50 rounded-lg p-4">
              <h4 className="text-sm font-medium text-gray-900 mb-2">Inherited Settings</h4>
              <dl className="text-xs space-y-1">
                <div className="flex justify-between">
                  <dt className="text-gray-500">Model:</dt>
                  <dd className="text-gray-900">{configuration.model_name}</dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-gray-500">Chunk Size:</dt>
                  <dd className="text-gray-900">{configuration.chunk_size} tokens</dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-gray-500">Chunk Overlap:</dt>
                  <dd className="text-gray-900">{configuration.chunk_overlap} tokens</dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-gray-500">Quantization:</dt>
                  <dd className="text-gray-900">{configuration.quantization}</dd>
                </div>
                {configuration.vector_dim && (
                  <div className="flex justify-between">
                    <dt className="text-gray-500">Vector Dimensions:</dt>
                    <dd className="text-gray-900">{configuration.vector_dim}</dd>
                  </div>
                )}
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
              disabled={mutation.isPending}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="px-4 py-2 bg-blue-600 text-white rounded-md text-sm font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
              disabled={mutation.isPending}
            >
              {mutation.isPending ? 'Creating Job...' : 'Add Data'}
            </button>
          </div>
        </form>
      </div>
    </>
  );
}

export default AddDataToCollectionModal;