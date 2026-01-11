import { useEffect, useState, useRef } from 'react';
import type { MCPProfile } from '../../types/mcp-profile';
import { useMCPProfileConfig } from '../../hooks/useMCPProfiles';

interface ConfigModalProps {
  profile: MCPProfile;
  onClose: () => void;
}

export default function ConfigModal({ profile, onClose }: ConfigModalProps) {
  const { data: config, isLoading, error } = useMCPProfileConfig(profile.id);
  const [copied, setCopied] = useState<string | null>(null);

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

  /**
   * Copy text to clipboard with fallback for older browsers.
   * @param text - The string to copy to clipboard
   * @param label - Unique identifier for tracking copied state (e.g., 'toolName', 'config')
   */
  const copyToClipboard = async (text: string, label: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(label);
      setTimeout(() => setCopied(null), 2000);
    } catch (clipboardError) {
      // Clipboard API failed - use execCommand fallback (deprecated but widely supported)
      console.warn('Clipboard API failed, falling back to execCommand:', clipboardError);
      const textArea = document.createElement('textarea');
      textArea.value = text;
      document.body.appendChild(textArea);
      textArea.select();
      const success = document.execCommand('copy');
      document.body.removeChild(textArea);
      if (success) {
        setCopied(label);
        setTimeout(() => setCopied(null), 2000);
      } else {
        console.error('Both clipboard methods failed');
        setCopied(`${label}-error`);
        setTimeout(() => setCopied(null), 2000);
      }
    }
  };

  const toolName = `search_${profile.name}`;

  const configJson = config
    ? JSON.stringify(
        {
          [config.server_name]: {
            command: config.command,
            args: config.args,
            env: config.env,
          },
        },
        null,
        2
      )
    : '';

  return (
    <>
      <div
        className="fixed inset-0 bg-black bg-opacity-50 z-[60]"
        onClick={onClose}
      />
      <div
        ref={modalRef}
        className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 bg-white rounded-lg shadow-xl z-[60] w-full max-w-2xl max-h-[90vh] overflow-y-auto"
        role="dialog"
        aria-modal="true"
        aria-labelledby="config-modal-title"
      >
        {/* Header */}
        <div className="px-6 py-4 border-b border-gray-200 flex items-center justify-between">
          <div>
            <h2
              id="config-modal-title"
              className="text-xl font-semibold text-gray-900"
            >
              Connection Info
            </h2>
            <p className="text-sm text-gray-500 mt-1">
              Configure MCP client to use "{profile.name}" profile
            </p>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 rounded"
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

        {/* Content */}
        <div className="px-6 py-4 space-y-6">
          {/* Loading State */}
          {isLoading && (
            <div className="flex items-center justify-center py-8">
              <svg
                className="animate-spin h-8 w-8 text-gray-400"
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
              <span className="ml-3 text-gray-500">Loading configuration...</span>
            </div>
          )}

          {/* Error State */}
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <div className="flex">
                <svg
                  className="h-5 w-5 text-red-400"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
                <div className="ml-3">
                  <h3 className="text-sm font-medium text-red-800">
                    Error loading configuration
                  </h3>
                  <p className="mt-1 text-sm text-red-700">
                    {error instanceof Error ? error.message : 'Unknown error'}
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Config Content */}
          {config && (
            <>
              {/* Tool Name */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  MCP Tool Name
                </label>
                <div className="flex items-center gap-2">
                  <code className="flex-1 text-sm bg-gray-100 px-4 py-2 rounded-md font-mono text-gray-800">
                    {toolName}
                  </code>
                  <button
                    onClick={() => copyToClipboard(toolName, 'toolName')}
                    className="inline-flex items-center px-3 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    {copied === 'toolName-error' ? (
                      <>
                        <svg
                          className="w-4 h-4 mr-1 text-red-500"
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
                        Failed
                      </>
                    ) : copied === 'toolName' ? (
                      <>
                        <svg
                          className="w-4 h-4 mr-1 text-green-500"
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
                        Copied
                      </>
                    ) : (
                      <>
                        <svg
                          className="w-4 h-4 mr-1"
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
                        Copy
                      </>
                    )}
                  </button>
                </div>
                <p className="mt-1 text-xs text-gray-500">
                  Claude will use this tool name to search this profile's collections
                </p>
              </div>

              {/* Config File Locations */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Config File Location
                </label>
                <div className="space-y-2 text-sm">
                  <div className="flex items-start gap-2">
                    <span className="text-gray-500 w-16 flex-shrink-0">macOS:</span>
                    <code className="bg-gray-100 px-2 py-1 rounded text-xs font-mono text-gray-700 break-all">
                      ~/Library/Application Support/Claude/claude_desktop_config.json
                    </code>
                  </div>
                  <div className="flex items-start gap-2">
                    <span className="text-gray-500 w-16 flex-shrink-0">Linux:</span>
                    <code className="bg-gray-100 px-2 py-1 rounded text-xs font-mono text-gray-700 break-all">
                      ~/.config/Claude/claude_desktop_config.json
                    </code>
                  </div>
                  <div className="flex items-start gap-2">
                    <span className="text-gray-500 w-16 flex-shrink-0">Windows:</span>
                    <code className="bg-gray-100 px-2 py-1 rounded text-xs font-mono text-gray-700 break-all">
                      %APPDATA%\Claude\claude_desktop_config.json
                    </code>
                  </div>
                </div>
              </div>

              {/* JSON Config */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="block text-sm font-medium text-gray-700">
                    Add to mcpServers
                  </label>
                  <button
                    onClick={() => copyToClipboard(configJson, 'config')}
                    className="inline-flex items-center px-2 py-1 text-xs font-medium text-gray-600 hover:text-gray-900 focus:outline-none"
                  >
                    {copied === 'config-error' ? (
                      <>
                        <svg
                          className="w-3 h-3 mr-1 text-red-500"
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
                        Failed!
                      </>
                    ) : copied === 'config' ? (
                      <>
                        <svg
                          className="w-3 h-3 mr-1 text-green-500"
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
                        Copied!
                      </>
                    ) : (
                      <>
                        <svg
                          className="w-3 h-3 mr-1"
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
                        Copy JSON
                      </>
                    )}
                  </button>
                </div>
                <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg text-sm font-mono overflow-x-auto">
                  {configJson}
                </pre>
              </div>

              {/* Token Warning */}
              <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
                <div className="flex">
                  <svg
                    className="h-5 w-5 text-amber-400 flex-shrink-0"
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
                    <h3 className="text-sm font-medium text-amber-800">
                      Replace the auth token
                    </h3>
                    <p className="mt-1 text-sm text-amber-700">
                      Replace{' '}
                      <code className="bg-amber-100 px-1 rounded text-xs">
                        &lt;your-access-token-or-api-key&gt;
                      </code>{' '}
                      with a valid API key or access token. API keys are recommended
                      for persistent MCP clients.
                    </p>
                  </div>
                </div>
              </div>

              {/* Usage Note */}
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
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
                    <h3 className="text-sm font-medium text-blue-800">
                      How it works
                    </h3>
                    <p className="mt-1 text-sm text-blue-700">
                      After configuration, Claude Desktop will have access to a{' '}
                      <code className="bg-blue-100 px-1 rounded text-xs">
                        {toolName}
                      </code>{' '}
                      tool that searches your selected collections. Restart Claude
                      Desktop after updating the config.
                    </p>
                  </div>
                </div>
              </div>
            </>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 bg-gray-50 border-t border-gray-200 flex justify-end rounded-b-lg">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-1"
          >
            Close
          </button>
        </div>
      </div>
    </>
  );
}
