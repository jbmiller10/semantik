/**
 * Modal for starting guided pipeline setup.
 * User selects a source to configure via the agent chat.
 */

import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useCollections } from '../../hooks/useCollections';
import { useCreateConversation } from '../../hooks/useAgentConversation';

interface Source {
  id: number;
  source_type: string;
  source_path: string;
  collection_id: string;
  collection_name: string;
}

interface GuidedSetupModalProps {
  onClose: () => void;
}

export function GuidedSetupModal({ onClose }: GuidedSetupModalProps) {
  const navigate = useNavigate();
  const [selectedSourceId, setSelectedSourceId] = useState<number | null>(null);
  const [sources, setSources] = useState<Source[]>([]);
  const [isLoadingSources, setIsLoadingSources] = useState(true);

  // Get collections to extract sources
  const { data: collections = [], isLoading: collectionsLoading } = useCollections();
  const createConversation = useCreateConversation();

  // Extract sources from collections
  // Note: This is a simplified version - in a real implementation,
  // we might have a dedicated API endpoint to list sources
  useEffect(() => {
    if (!collectionsLoading && collections.length > 0) {
      // For now, we'll create placeholder sources from collections
      // In real implementation, this would fetch actual sources
      const extractedSources: Source[] = [];
      // TODO: Implement source extraction or dedicated API endpoint
      setSources(extractedSources);
      setIsLoadingSources(false);
    } else if (!collectionsLoading) {
      setIsLoadingSources(false);
    }
  }, [collections, collectionsLoading]);

  const handleStart = async () => {
    if (!selectedSourceId) return;

    createConversation.mutate(
      { source_id: selectedSourceId },
      {
        onSuccess: (data) => {
          onClose();
          navigate(`/agent/${data.id}`);
        },
      }
    );
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/50"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="relative bg-[var(--bg-primary)] rounded-xl shadow-xl w-full max-w-lg mx-4 max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="px-6 py-4 border-b border-[var(--border)]">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xl font-semibold text-[var(--text-primary)]">
                Guided Pipeline Setup
              </h2>
              <p className="text-sm text-[var(--text-secondary)] mt-1">
                Chat with an AI assistant to configure your pipeline
              </p>
            </div>
            <button
              onClick={onClose}
              className="p-2 rounded-lg hover:bg-[var(--bg-tertiary)] text-[var(--text-muted)] transition-colors"
              aria-label="Close"
            >
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
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="p-6">
          {/* Feature explanation */}
          <div className="mb-6 p-4 bg-blue-500/10 border border-blue-500/30 rounded-lg">
            <div className="flex gap-3">
              <svg
                className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5"
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
              <div>
                <p className="text-sm text-blue-400 font-medium mb-1">
                  How it works
                </p>
                <p className="text-sm text-blue-300/80">
                  The assistant will analyze your source documents and help you choose
                  the best embedding model, chunking strategy, and other settings through
                  a natural conversation.
                </p>
              </div>
            </div>
          </div>

          {/* Source input */}
          <div className="mb-6">
            <label
              htmlFor="source-id"
              className="block text-sm font-medium text-[var(--text-primary)] mb-2"
            >
              Source ID
            </label>
            <p className="text-xs text-[var(--text-muted)] mb-3">
              Enter the ID of a source you've already added to a collection.
              The assistant will help you optimize its pipeline settings.
            </p>
            <input
              type="number"
              id="source-id"
              value={selectedSourceId || ''}
              onChange={(e) => setSelectedSourceId(e.target.value ? parseInt(e.target.value, 10) : null)}
              placeholder="Enter source ID..."
              className="input-field w-full"
              min={1}
            />
          </div>

          {/* Existing sources list (when available) */}
          {sources.length > 0 && (
            <div className="mb-6">
              <label className="block text-sm font-medium text-[var(--text-primary)] mb-2">
                Or select an existing source
              </label>
              <div className="space-y-2 max-h-48 overflow-y-auto">
                {sources.map((source) => (
                  <button
                    key={source.id}
                    onClick={() => setSelectedSourceId(source.id)}
                    className={`w-full text-left p-3 rounded-lg border transition-colors ${
                      selectedSourceId === source.id
                        ? 'border-gray-400 dark:border-white bg-gray-100 dark:bg-white/10'
                        : 'border-[var(--border)] hover:border-[var(--border-subtle)] hover:bg-[var(--bg-tertiary)]'
                    }`}
                  >
                    <p className="text-sm font-medium text-[var(--text-primary)]">
                      {source.source_path}
                    </p>
                    <p className="text-xs text-[var(--text-muted)] mt-0.5">
                      {source.source_type} - Collection: {source.collection_name}
                    </p>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* No sources state */}
          {!isLoadingSources && sources.length === 0 && (
            <div className="text-center py-4 text-sm text-[var(--text-muted)]">
              <p>
                No sources found. Enter a source ID above or create a collection
                with a source first.
              </p>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-[var(--border)] flex justify-end gap-3">
          <button
            onClick={onClose}
            className="btn-secondary"
          >
            Cancel
          </button>
          <button
            onClick={handleStart}
            disabled={!selectedSourceId || createConversation.isPending}
            className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {createConversation.isPending ? (
              <>
                <svg
                  className="w-4 h-4 animate-spin mr-2"
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
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                  />
                </svg>
                Starting...
              </>
            ) : (
              <>
                <svg
                  className="w-4 h-4 mr-2"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
                  />
                </svg>
                Start Conversation
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}

export default GuidedSetupModal;
