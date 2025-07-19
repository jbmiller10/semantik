import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { collectionsApi } from '../services/api';
import { useUIStore } from '../stores/uiStore';

interface RenameCollectionModalProps {
  currentName: string;
  onClose: () => void;
  onSuccess: (newName: string) => void;
}

function RenameCollectionModal({
  currentName,
  onClose,
  onSuccess,
}: RenameCollectionModalProps) {
  const { addToast } = useUIStore();
  const [newName, setNewName] = useState(currentName);
  const [error, setError] = useState('');

  const mutation = useMutation({
    mutationFn: async () => {
      if (newName === currentName) {
        throw new Error('New name must be different from current name');
      }
      return collectionsApi.rename(currentName, newName);
    },
    onSuccess: () => {
      onSuccess(newName);
    },
    onError: (error: any) => {
      const message = error.response?.data?.detail || error.message || 'Failed to rename collection';
      setError(message);
      addToast({ type: 'error', message });
    },
  });

  const validateName = (name: string) => {
    if (!name.trim()) {
      setError('Collection name cannot be empty');
      return false;
    }
    
    const invalidChars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*'];
    for (const char of invalidChars) {
      if (name.includes(char)) {
        setError(`Collection name cannot contain "${char}"`);
        return false;
      }
    }
    
    setError('');
    return true;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (validateName(newName)) {
      mutation.mutate();
    }
  };

  const handleNameChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setNewName(value);
    validateName(value);
  };

  return (
    <>
      <div className="fixed inset-0 bg-black bg-opacity-50 z-[60]" onClick={onClose} />
      <div className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 bg-white rounded-lg shadow-xl z-[60] w-full max-w-md">
        <div className="px-6 py-4 border-b">
          <h3 className="text-lg font-medium text-gray-900">Rename Collection</h3>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="px-6 py-4 space-y-4">
            <div>
              <label htmlFor="current-name" className="block text-sm font-medium text-gray-700">
                Current Name
              </label>
              <input
                type="text"
                id="current-name"
                value={currentName}
                className="mt-1 block w-full border-gray-300 rounded-md shadow-sm bg-gray-50 sm:text-sm"
                disabled
              />
            </div>

            <div>
              <label htmlFor="new-name" className="block text-sm font-medium text-gray-700">
                New Name
              </label>
              <input
                type="text"
                id="new-name"
                value={newName}
                onChange={handleNameChange}
                className={`mt-1 block w-full rounded-md shadow-sm sm:text-sm ${
                  error
                    ? 'border-red-300 focus:ring-red-500 focus:border-red-500'
                    : 'border-gray-300 focus:ring-blue-500 focus:border-blue-500'
                }`}
                required
                autoFocus
              />
              {error && (
                <p className="mt-1 text-sm text-red-600">{error}</p>
              )}
            </div>

            <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4">
              <div className="flex">
                <div className="flex-shrink-0">
                  <svg className="h-5 w-5 text-yellow-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                  </svg>
                </div>
                <div className="ml-3">
                  <p className="text-sm text-yellow-700">
                    This will only change the display name. The underlying data and vector collections will remain unchanged.
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
              disabled={mutation.isPending || !!error || newName === currentName}
            >
              {mutation.isPending ? 'Renaming...' : 'Rename'}
            </button>
          </div>
        </form>
      </div>
    </>
  );
}

export default RenameCollectionModal;