import { useState, useRef, useEffect } from 'react';
import type { ApiKeyCreateResponse } from '../../../types/api-key';

interface ApiKeyCreatedModalProps {
  apiKey: ApiKeyCreateResponse;
  onClose: () => void;
}

export default function ApiKeyCreatedModal({
  apiKey,
  onClose,
}: ApiKeyCreatedModalProps) {
  const [showKey, setShowKey] = useState(false);
  const [copied, setCopied] = useState(false);
  const modalRef = useRef<HTMLDivElement>(null);

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
      // Don't close on Escape - force user to acknowledge
      if (e.key === 'Escape') {
        e.preventDefault();
        return;
      }

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

  /**
   * Copy text to clipboard with fallback for older browsers.
   */
  const copyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(apiKey.api_key);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (clipboardError) {
      // Clipboard API failed - use execCommand fallback
      console.warn('Clipboard API failed, falling back to execCommand:', clipboardError);
      const textArea = document.createElement('textarea');
      textArea.value = apiKey.api_key;
      document.body.appendChild(textArea);
      textArea.select();
      const success = document.execCommand('copy');
      document.body.removeChild(textArea);
      if (success) {
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      } else {
        console.error('Both clipboard methods failed');
      }
    }
  };

  // Mask the key (show first and last few characters)
  const maskedKey = showKey
    ? apiKey.api_key
    : `${apiKey.api_key.substring(0, 15)}${'*'.repeat(20)}${apiKey.api_key.substring(apiKey.api_key.length - 4)}`;

  return (
    <>
      {/* Backdrop - no onClick to prevent closing */}
      <div className="fixed inset-0 bg-black/50 dark:bg-black/80 z-[70]" />
      <div
        ref={modalRef}
        className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 panel rounded-xl shadow-2xl z-[70] w-full max-w-lg"
        role="dialog"
        aria-modal="true"
        aria-labelledby="key-created-title"
      >
        {/* Header */}
        <div className="px-6 py-4 border-b border-[var(--border)]">
          <h2
            id="key-created-title"
            className="text-xl font-semibold text-[var(--text-primary)]"
          >
            API Key Created
          </h2>
        </div>

        {/* Content */}
        <div className="px-6 py-4 space-y-4">
          {/* Warning Banner */}
          <div className="bg-amber-500/10 border border-amber-500/20 rounded-lg p-4">
            <div className="flex">
              <svg
                className="h-5 w-5 text-amber-500 flex-shrink-0"
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
              <div className="ml-3">
                <h3 className="text-sm font-medium text-amber-600 dark:text-amber-400">
                  Copy your API key now
                </h3>
                <p className="mt-1 text-sm text-amber-700 dark:text-amber-300/80">
                  This is the only time your API key will be shown. Store it securely.
                </p>
              </div>
            </div>
          </div>

          {/* Key Name */}
          <div>
            <label className="block text-sm font-medium text-[var(--text-secondary)] mb-1">
              Name
            </label>
            <p className="text-[var(--text-primary)]">{apiKey.name}</p>
          </div>

          {/* Key Display */}
          <div>
            <label className="block text-sm font-medium text-[var(--text-secondary)] mb-1">
              API Key
            </label>
            <div className="flex items-center gap-2">
              <div className="flex-1 bg-[var(--bg-tertiary)] border border-[var(--border)] rounded-lg p-3 font-mono text-sm text-[var(--text-primary)] break-all">
                {maskedKey}
              </div>
              <div className="flex flex-col gap-1">
                <button
                  onClick={() => setShowKey(!showKey)}
                  className="p-2 text-[var(--text-secondary)] bg-[var(--bg-tertiary)] border border-[var(--border)] rounded-lg hover:bg-[var(--bg-primary)] focus:outline-none focus:ring-2 focus:ring-gray-400 dark:focus:ring-white transition-colors"
                  title={showKey ? 'Hide key' : 'Show key'}
                >
                  {showKey ? (
                    <svg
                      className="w-5 h-5"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21"
                      />
                    </svg>
                  ) : (
                    <svg
                      className="w-5 h-5"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                      />
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
                      />
                    </svg>
                  )}
                </button>
                <button
                  onClick={copyToClipboard}
                  className={`p-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-gray-400 dark:focus:ring-white transition-colors ${
                    copied
                      ? 'bg-green-500/20 border-green-500/30 text-green-400'
                      : 'bg-[var(--bg-tertiary)] border-[var(--border)] text-[var(--text-secondary)] hover:bg-[var(--bg-primary)]'
                  }`}
                  title="Copy to clipboard"
                >
                  {copied ? (
                    <svg
                      className="w-5 h-5"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M5 13l4 4L19 7"
                      />
                    </svg>
                  ) : (
                    <svg
                      className="w-5 h-5"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"
                      />
                    </svg>
                  )}
                </button>
              </div>
            </div>
          </div>

          {/* Usage Instructions */}
          <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
            <div className="flex">
              <svg
                className="h-5 w-5 text-blue-400 flex-shrink-0"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
              <div className="ml-3">
                <h3 className="text-sm font-medium text-blue-300">Usage</h3>
                <p className="mt-1 text-sm text-blue-400">
                  Include this key in the Authorization header:
                </p>
                <code className="mt-2 block bg-blue-500/20 px-2 py-1 rounded text-xs font-mono text-blue-300">
                  Authorization: Bearer {apiKey.api_key.substring(0, 15)}...
                </code>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-4 bg-[var(--bg-secondary)] border-t border-[var(--border)] flex justify-end rounded-b-xl">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm font-medium text-gray-900 dark:text-gray-900 bg-gray-200 dark:bg-white border border-transparent rounded-lg hover:bg-gray-300 dark:hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 focus:ring-offset-[var(--bg-primary)] transition-colors"
          >
            I've copied my key
          </button>
        </div>
      </div>
    </>
  );
}
