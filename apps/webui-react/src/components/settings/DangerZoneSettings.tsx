/**
 * Danger Zone Settings component.
 * Admin-only destructive operations with typed confirmation.
 */
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useQueryClient } from '@tanstack/react-query';
import { settingsApi } from '../../services/api/v2';
import { getErrorMessage } from '../../utils/errorUtils';
import { AlertTriangle } from 'lucide-react';

const CONFIRMATION_TEXT = 'RESET';

export default function DangerZoneSettings() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [showConfirmDialog, setShowConfirmDialog] = useState(false);
  const [confirmText, setConfirmText] = useState('');
  const [resetting, setResetting] = useState(false);

  const handleReset = async () => {
    if (confirmText !== CONFIRMATION_TEXT) return;

    try {
      setResetting(true);
      await settingsApi.resetDatabase();

      // Clear all React Query caches
      queryClient.clear();

      // Show success message and redirect
      alert('Database reset successfully!');
      setShowConfirmDialog(false);
      setConfirmText('');

      // Redirect to home page
      navigate('/');
    } catch (error) {
      console.error('Failed to reset database:', error);
      const errorMessage = getErrorMessage(error);
      alert(`Failed to reset database: ${errorMessage}`);
    } finally {
      setResetting(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Warning box */}
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <div className="flex">
          <AlertTriangle className="h-5 w-5 text-red-400 flex-shrink-0" />
          <div className="ml-3">
            <p className="text-sm text-red-700">
              <strong>Warning:</strong> Operations in this section are destructive and cannot be undone.
              Please proceed with extreme caution.
            </p>
          </div>
        </div>
      </div>

      {/* Reset Database */}
      <div className="bg-[var(--bg-secondary)] border border-red-200 dark:border-red-900 rounded-lg p-4">
        <h4 className="font-medium text-[var(--text-primary)] mb-2">Reset Database</h4>
        <p className="text-sm text-[var(--text-secondary)] mb-3">
          This will permanently delete all collections, files, and associated data from the database.
          All Qdrant collections and parquet files will also be removed.
        </p>
        <button
          onClick={() => setShowConfirmDialog(true)}
          className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-red-600 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500"
        >
          <svg className="h-4 w-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
          </svg>
          Reset Database
        </button>
      </div>

      {/* Confirmation Dialog */}
      {showConfirmDialog && (
        <div className="fixed inset-0 bg-black/50 overflow-y-auto h-full w-full z-50 flex items-center justify-center">
          <div className="relative mx-auto p-5 border border-[var(--border)] w-96 shadow-lg rounded-md bg-[var(--bg-primary)]">
            <div className="mt-3 text-center">
              <div className="mx-auto flex items-center justify-center h-12 w-12 rounded-full bg-red-100 dark:bg-red-900/30">
                <AlertTriangle className="h-6 w-6 text-red-600 dark:text-red-400" />
              </div>
              <h3 className="text-lg leading-6 font-medium text-[var(--text-primary)] mt-4">
                Confirm Database Reset
              </h3>
              <div className="mt-2 px-7 py-3">
                <p className="text-sm text-[var(--text-secondary)]">
                  Are you sure you want to reset the database? This will permanently delete all
                  collections, files, and embeddings.
                </p>
                <p className="text-sm text-red-600 dark:text-red-400 font-semibold mt-2">
                  Type "{CONFIRMATION_TEXT}" to confirm:
                </p>
                <input
                  type="text"
                  value={confirmText}
                  onChange={(e) => setConfirmText(e.target.value)}
                  className="mt-2 w-full px-3 py-2 border border-[var(--border)] rounded-md bg-[var(--bg-secondary)] text-[var(--text-primary)] focus:outline-none focus:ring-2 focus:ring-red-500"
                  placeholder={`Type ${CONFIRMATION_TEXT}`}
                  autoFocus
                />
              </div>
              <div className="items-center px-4 py-3">
                <button
                  onClick={handleReset}
                  disabled={confirmText !== CONFIRMATION_TEXT || resetting}
                  className="px-4 py-2 bg-red-600 text-white text-base font-medium rounded-md w-full shadow-sm hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-red-500 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {resetting ? (
                    <span className="flex items-center justify-center">
                      <svg className="animate-spin h-4 w-4 mr-2" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                      </svg>
                      Resetting...
                    </span>
                  ) : (
                    'Reset Database'
                  )}
                </button>
                <button
                  onClick={() => {
                    setShowConfirmDialog(false);
                    setConfirmText('');
                  }}
                  className="mt-3 px-4 py-2 bg-[var(--bg-tertiary)] text-[var(--text-primary)] text-base font-medium rounded-md w-full shadow-sm hover:bg-[var(--border)] focus:outline-none focus:ring-2 focus:ring-[var(--border)]"
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
