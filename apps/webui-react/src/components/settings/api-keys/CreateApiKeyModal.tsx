import { useState, useEffect, useRef } from 'react';
import type { ApiKeyCreate, ApiKeyCreateResponse } from '../../../types/api-key';
import { useCreateApiKey } from '../../../hooks/useApiKeys';
import { AxiosError } from 'axios';

interface CreateApiKeyModalProps {
  onClose: () => void;
  onSuccess: (response: ApiKeyCreateResponse) => void;
}

// Expiration options in days
const EXPIRATION_OPTIONS = [
  { label: '30 days', value: 30 },
  { label: '90 days', value: 90 },
  { label: '365 days', value: 365 },
  { label: '10 years', value: 3650 },
] as const;

export default function CreateApiKeyModal({
  onClose,
  onSuccess,
}: CreateApiKeyModalProps) {
  const createApiKey = useCreateApiKey();
  const [name, setName] = useState('');
  const [expiresInDays, setExpiresInDays] = useState<number>(365);
  const [nameError, setNameError] = useState<string | null>(null);
  const modalRef = useRef<HTMLDivElement>(null);

  // Handle escape key to close modal
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };
    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [onClose]);

  // Focus trap for accessibility
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

  const validateName = (value: string): boolean => {
    if (!value.trim()) {
      setNameError('Name is required');
      return false;
    }
    if (value.length > 100) {
      setNameError('Name must be 100 characters or less');
      return false;
    }
    setNameError(null);
    return true;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!validateName(name)) {
      return;
    }

    const data: ApiKeyCreate = {
      name: name.trim(),
      expires_in_days: expiresInDays,
    };

    try {
      const response = await createApiKey.mutateAsync(data);
      onSuccess(response);
    } catch (error) {
      // Check for duplicate name error
      if (error instanceof AxiosError && error.response?.status === 409) {
        setNameError('A key with this name already exists');
      }
      // Other errors are handled by the hook
    }
  };

  return (
    <>
      <div
        className="fixed inset-0 bg-black/50 dark:bg-black/80 z-[60]"
        onClick={onClose}
      />
      <div
        ref={modalRef}
        className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 panel rounded-xl shadow-2xl z-[60] w-full max-w-md"
        role="dialog"
        aria-modal="true"
        aria-labelledby="create-api-key-title"
      >
        {/* Header */}
        <div className="px-6 py-4 border-b border-[var(--border)] flex items-center justify-between">
          <h2
            id="create-api-key-title"
            className="text-xl font-semibold text-[var(--text-primary)]"
          >
            Create API Key
          </h2>
          <button
            onClick={onClose}
            className="text-[var(--text-muted)] hover:text-[var(--text-primary)] focus:outline-none focus:ring-2 focus:ring-gray-400 dark:focus:ring-white rounded transition-colors"
          >
            <svg
              className="w-6 h-6"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="px-6 py-4 space-y-4">
          {/* Name Field */}
          <div>
            <label
              htmlFor="key-name"
              className="block text-sm font-medium text-[var(--text-secondary)] mb-1"
            >
              Name
            </label>
            <input
              type="text"
              id="key-name"
              value={name}
              onChange={(e) => {
                setName(e.target.value);
                if (nameError) validateName(e.target.value);
              }}
              onBlur={() => validateName(name)}
              placeholder="e.g., Claude Desktop, Production MCP"
              className={`w-full px-3 py-2 bg-[var(--bg-tertiary)] border rounded-lg text-[var(--text-primary)] placeholder-[var(--text-muted)] focus:outline-none focus:ring-2 focus:ring-gray-400 dark:focus:ring-white ${
                nameError ? 'border-red-500' : 'border-[var(--border)]'
              }`}
              maxLength={100}
            />
            {nameError && (
              <p className="mt-1 text-sm text-red-400">{nameError}</p>
            )}
          </div>

          {/* Expiration Field */}
          <div>
            <label
              htmlFor="key-expiration"
              className="block text-sm font-medium text-[var(--text-secondary)] mb-1"
            >
              Expiration
            </label>
            <select
              id="key-expiration"
              value={expiresInDays}
              onChange={(e) => setExpiresInDays(Number(e.target.value))}
              className="w-full px-3 py-2 bg-[var(--bg-tertiary)] border border-[var(--border)] rounded-lg text-[var(--text-primary)] focus:outline-none focus:ring-2 focus:ring-gray-400 dark:focus:ring-white"
            >
              {EXPIRATION_OPTIONS.map((option) => (
                <option
                  key={option.value}
                  value={option.value}
                >
                  {option.label}
                </option>
              ))}
            </select>
          </div>

          {/* Footer */}
          <div className="flex justify-end gap-3 pt-4">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-sm font-medium text-[var(--text-secondary)] bg-[var(--bg-tertiary)] border border-[var(--border)] rounded-lg hover:bg-[var(--bg-primary)] focus:outline-none focus:ring-2 focus:ring-gray-400 dark:focus:ring-white transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={createApiKey.isPending || !name.trim()}
              className="px-4 py-2 text-sm font-medium text-gray-900 dark:text-gray-900 bg-gray-200 dark:bg-white border border-transparent rounded-lg hover:bg-gray-300 dark:hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 focus:ring-offset-[var(--bg-primary)] disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {createApiKey.isPending ? (
                <span className="flex items-center">
                  <svg
                    className="animate-spin h-4 w-4 mr-2"
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
                  Creating...
                </span>
              ) : (
                'Create Key'
              )}
            </button>
          </div>
        </form>
      </div>
    </>
  );
}
