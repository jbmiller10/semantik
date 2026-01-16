import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { AxiosError } from 'axios';
import { collectionsV2Api } from '../services/api/v2/collections';
import { useUIStore } from '../stores/uiStore';

interface RenameCollectionModalProps {
  collectionId: string;
  currentName: string;
  onClose: () => void;
  onSuccess: (newName: string) => void;
}

function RenameCollectionModal({
  collectionId,
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
      return collectionsV2Api.update(collectionId, { name: newName });
    },
    onSuccess: () => {
      onSuccess(newName);
    },
    onError: (error: Error | AxiosError) => {
      let message = 'Failed to rename collection';
      if (error instanceof AxiosError && error.response?.data?.detail) {
        message = error.response.data.detail;
      } else if (error instanceof Error) {
        message = error.message;
      }
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
      <div className="fixed inset-0 bg-black/50 dark:bg-black/80 backdrop-blur-sm z-[60]" onClick={onClose} />
      <div className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 bg-[var(--bg-primary)] border border-[var(--border)] rounded-2xl shadow-2xl z-[60] w-full max-w-md">
        <div className="px-6 py-4 border-b border-[var(--border)] bg-[var(--bg-secondary)]">
          <h3 className="text-xl font-bold text-[var(--text-primary)] tracking-tight">Rename Collection</h3>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="px-6 py-4 space-y-4">
            <div>
              <label htmlFor="current-name" className="block text-xs font-bold text-[var(--text-muted)] uppercase tracking-wider mb-2">
                Current Name
              </label>
              <input
                type="text"
                id="current-name"
                value={currentName}
                className="mt-1 block w-full border border-[var(--border)] rounded-xl bg-[var(--bg-secondary)] text-[var(--text-muted)] sm:text-sm px-4 py-2"
                disabled
              />
            </div>

            <div>
              <label htmlFor="new-name" className="block text-xs font-bold text-[var(--text-muted)] uppercase tracking-wider mb-2">
                New Name
              </label>
              <input
                type="text"
                id="new-name"
                value={newName}
                onChange={handleNameChange}
                className={`mt-1 block w-full rounded-xl shadow-sm sm:text-sm bg-[var(--bg-tertiary)] px-4 py-2 text-[var(--text-primary)] placeholder-[var(--text-muted)] focus:outline-none focus:ring-2 ${error
                    ? 'border border-red-500/50 focus:ring-red-500/50'
                    : 'border border-[var(--border)] focus:ring-signal-500/50'
                  }`}
                required
                autoFocus
                placeholder="Enter new name"
              />
              {error && (
                <p className="mt-1 text-sm text-red-400">{error}</p>
              )}
            </div>

            <div className="bg-amber-500/10 border border-amber-500/20 rounded-xl p-4">
              <div className="flex">
                <div className="flex-shrink-0">
                  <svg className="h-5 w-5 text-amber-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                  </svg>
                </div>
                <div className="ml-3">
                  <p className="text-sm text-amber-300/80">
                    This will only change the display name. The underlying data and vector collections will remain unchanged.
                  </p>
                </div>
              </div>
            </div>
          </div>

          <div className="px-6 py-4 border-t border-[var(--border)] flex justify-end space-x-3 bg-[var(--bg-secondary)]">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 border border-[var(--border)] rounded-xl text-sm font-medium text-[var(--text-secondary)] hover:bg-[var(--bg-tertiary)] hover:text-[var(--text-primary)] transition-colors"
              disabled={mutation.isPending}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="px-4 py-2 bg-signal-600 text-white rounded-xl text-sm font-bold shadow-lg shadow-signal-600/20 hover:bg-signal-500 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
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