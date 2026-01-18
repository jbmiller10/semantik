import { useEffect, useState, useRef } from 'react';
import type { MCPProfile } from '../../types/mcp-profile';
import { useDeleteMCPProfile } from '../../hooks/useMCPProfiles';

interface DeleteConfirmModalProps {
  profile: MCPProfile;
  onClose: () => void;
}

export default function DeleteConfirmModal({
  profile,
  onClose,
}: DeleteConfirmModalProps) {
  const deleteProfile = useDeleteMCPProfile();
  const [isDeleting, setIsDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  // Handle escape key to close modal
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !isDeleting) {
        onClose();
      }
    };
    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [onClose, isDeleting]);

  // Focus trap for accessibility
  const modalRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const modal = modalRef.current;
    if (!modal) return;

    const focusableElements = modal.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    const firstElement = focusableElements[0] as HTMLElement;
    const lastElement = focusableElements[focusableElements.length - 1] as HTMLElement;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key !== 'Tab') return;

      if (e.shiftKey && document.activeElement === firstElement) {
        e.preventDefault();
        lastElement?.focus();
      } else if (!e.shiftKey && document.activeElement === lastElement) {
        e.preventDefault();
        firstElement?.focus();
      }
    };

    // Focus first focusable element on mount
    firstElement?.focus();

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, []);

  const handleDelete = async () => {
    setIsDeleting(true);
    setDeleteError(null);
    try {
      await deleteProfile.mutateAsync({
        profileId: profile.id,
        profileName: profile.name,
      });
      onClose();
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Delete failed';
      setDeleteError(errorMessage);
    } finally {
      setIsDeleting(false);
    }
  };

  return (
    <>
      <div
        className="fixed inset-0 bg-black/50 dark:bg-black/80 z-[60]"
        onClick={isDeleting ? undefined : onClose}
      />
      <div
        ref={modalRef}
        className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 panel rounded-xl shadow-2xl z-[60] w-full max-w-md"
        role="dialog"
        aria-modal="true"
        aria-labelledby="delete-modal-title"
      >
        {/* Content */}
        <div className="p-6">
          <div className="flex items-center justify-center w-12 h-12 mx-auto bg-red-500/20 rounded-full">
            <svg
              className="w-6 h-6 text-red-500"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
              />
            </svg>
          </div>

          <h3
            id="delete-modal-title"
            className="mt-4 text-lg font-semibold text-[var(--text-primary)] text-center"
          >
            Delete Profile
          </h3>

          <p className="mt-2 text-sm text-[var(--text-muted)] text-center">
            Are you sure you want to delete the profile{' '}
            <span className="font-medium text-[var(--text-primary)]">"{profile.name}"</span>?
            This action cannot be undone.
          </p>

          <p className="mt-2 text-xs text-[var(--text-muted)] text-center">
            This will remove the MCP tool{' '}
            <code className="bg-[var(--bg-tertiary)] px-1 rounded text-[var(--text-secondary)]">search_{profile.name}</code>{' '}
            from connected clients.
          </p>

          {deleteError && (
            <p className="mt-4 text-sm text-red-500 text-center">{deleteError}</p>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 bg-[var(--bg-secondary)] border-t border-[var(--border)] flex justify-end gap-3 rounded-b-xl">
          <button
            onClick={onClose}
            disabled={isDeleting}
            className="px-4 py-2 text-sm font-medium text-[var(--text-secondary)] bg-[var(--bg-tertiary)] border border-[var(--border)] rounded-lg hover:bg-[var(--bg-primary)] focus:outline-none focus:ring-2 focus:ring-gray-400 dark:focus:ring-white focus:ring-offset-1 focus:ring-offset-[var(--bg-primary)] transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleDelete}
            disabled={isDeleting}
            className="inline-flex items-center px-4 py-2 text-sm font-medium text-white bg-red-600 border border-transparent rounded-lg hover:bg-red-500 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-1 focus:ring-offset-[var(--bg-primary)] disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {isDeleting && (
              <svg
                className="animate-spin -ml-1 mr-2 h-4 w-4 text-white"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                />
              </svg>
            )}
            Delete Profile
          </button>
        </div>
      </div>
    </>
  );
}
