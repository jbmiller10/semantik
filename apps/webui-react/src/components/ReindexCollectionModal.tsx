import { useState } from 'react';
import { AxiosError } from 'axios';
import { useUIStore } from '../stores/uiStore';
import { useReindexCollection } from '../hooks/useCollectionOperations';
import { useNavigate } from 'react-router-dom';
import { useChunkingStore } from '../stores/chunkingStore';
import { SimplifiedChunkingStrategySelector } from './chunking/SimplifiedChunkingStrategySelector';
import { CHUNKING_STRATEGIES } from '../types/chunking';
import type { Collection, ReindexRequest } from '../types/collection';
import type { ChunkingStrategyType } from '../types/chunking';

interface ReindexCollectionModalProps {
  collection: Collection;
  configChanges: {
    embedding_model?: string;
    chunk_size?: number;
    chunk_overlap?: number;
    instruction?: string;
  };
  onClose: () => void;
  onSuccess: () => void;
}

function ReindexCollectionModal({ collection, configChanges, onClose, onSuccess }: ReindexCollectionModalProps) {
  const { addToast } = useUIStore();
  const reindexCollectionMutation = useReindexCollection();
  const navigate = useNavigate();
  const { strategyConfig, setStrategy } = useChunkingStore();
  const [confirmText, setConfirmText] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showStrategySelector, setShowStrategySelector] = useState(false);
  const [selectedStrategy, setSelectedStrategy] = useState<ChunkingStrategyType | null>(null);

  const expectedConfirmText = `reindex ${collection.name}`;
  const isConfirmValid = confirmText === expectedConfirmText;
  
  // Calculate what's changing
  const hasModelChange = configChanges.embedding_model !== undefined && configChanges.embedding_model !== collection.embedding_model;
  const hasInstructionChange = configChanges.instruction !== undefined;
  const hasStrategyChange = selectedStrategy !== null && selectedStrategy !== collection.chunking_strategy;
  
  const totalChanges = [hasModelChange, hasInstructionChange, hasStrategyChange].filter(Boolean).length;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!isConfirmValid || isSubmitting) return;

    setIsSubmitting(true);
    try {
      // Prepare the reindex request
      const request: ReindexRequest = {};
      if (configChanges.embedding_model !== undefined) {
        request.embedding_model = configChanges.embedding_model;
      }
      
      // Handle chunking strategy changes
      if (hasStrategyChange && selectedStrategy) {
        request.chunking_strategy = selectedStrategy;
        request.chunking_config = strategyConfig.parameters;
      }
      
      // Call the mutation to start re-indexing
      await reindexCollectionMutation.mutateAsync({
        collectionId: collection.id,
        config: request
      });
      
      // Navigate to collection detail page to show operation progress
      navigate(`/collections/${collection.id}`);
      // Toast is already shown by the mutation
      
      onSuccess();
    } catch (error) {
      // Error handling is already done by the mutation
      // This catch block is for any unexpected errors
      if (!reindexCollectionMutation.isError) {
        console.error('Failed to start re-indexing:', error);
        
        // Provide specific error messages based on error type
        let errorMessage = 'Failed to start re-indexing. Please try again.';
        
        if (error instanceof AxiosError) {
          if (error.response?.status === 403) {
            errorMessage = 'You do not have permission to re-index this collection.';
          } else if (error.response?.status === 404) {
            errorMessage = 'Collection not found. It may have been deleted.';
          } else if (error.response?.status === 409) {
            errorMessage = 'Another operation is already in progress for this collection.';
          } else if (error.response?.data?.detail) {
            errorMessage = error.response.data.detail;
          } else if (error.message) {
            errorMessage = error.message;
          }
        } else if (error instanceof Error) {
          errorMessage = error.message;
        }
        
        addToast({
          type: 'error',
          message: errorMessage,
        });
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleCancel = () => {
    if (!isSubmitting) {
      onClose();
    }
  };

  // Handle escape key
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape' && !isSubmitting) {
      onClose();
    }
  };

  return (
    <>
      <div 
        className="fixed inset-0 bg-black bg-opacity-50 z-[60]" 
        onClick={handleCancel}
      />
      <div className="fixed inset-0 z-[60] overflow-y-auto">
        <div className="flex min-h-full items-center justify-center p-4">
          <div 
            className="relative bg-white rounded-lg shadow-xl max-w-md w-full p-6"
            onKeyDown={handleKeyDown}
          >
            <h2 className="text-lg font-semibold text-gray-900 mb-4">
              Re-index Collection: {collection.name}
            </h2>

            <div className="mb-6">
              <div className="bg-red-50 border border-red-200 rounded-md p-4" role="alert">
                <div className="flex">
                  <div className="flex-shrink-0">
                    <svg className="h-5 w-5 text-red-400" fill="currentColor" viewBox="0 0 20 20" aria-hidden="true">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                    </svg>
                  </div>
                  <div className="ml-3">
                    <h3 className="text-sm font-medium text-red-800">
                      Warning: This action cannot be undone
                    </h3>
                    <div className="mt-2 text-sm text-red-700">
                      <p>Re-indexing will:</p>
                      <ul className="list-disc list-inside mt-1">
                        <li>Delete all existing vectors ({collection.vector_count} vectors)</li>
                        <li>Re-process all documents ({collection.document_count} documents) with new settings</li>
                        <li>Make the collection unavailable during processing</li>
                        {hasModelChange && (
                          <li className="font-semibold">Change the embedding model (requires complete re-embedding)</li>
                        )}
                      </ul>
                    </div>
                  </div>
                </div>
              </div>

              {/* Chunking Strategy Selector */}
              <div className="mt-4 mb-4">
                <button
                  type="button"
                  onClick={() => setShowStrategySelector(!showStrategySelector)}
                  className="text-sm text-blue-600 hover:text-blue-700 font-medium"
                >
                  {showStrategySelector ? 'Hide' : 'Change'} Chunking Strategy
                </button>
                
                {showStrategySelector && (
                  <div className="mt-3 p-4 border border-gray-200 rounded-lg bg-gray-50">
                    <SimplifiedChunkingStrategySelector
                      onStrategyChange={(strategy) => {
                        setSelectedStrategy(strategy);
                        setStrategy(strategy);
                      }}
                      disabled={isSubmitting}
                    />
                  </div>
                )}
              </div>

              <div className="mt-4">
                <h4 className="text-sm font-medium text-gray-700 mb-3">Configuration Changes ({totalChanges} change{totalChanges !== 1 ? 's' : ''}):</h4>
                <div className="space-y-2 bg-gray-50 rounded-lg p-3">
                  {hasModelChange && (
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-600">Embedding Model:</span>
                      <div className="flex items-center gap-2">
                        <span className="text-red-600 line-through">{collection.embedding_model}</span>
                        <span className="text-gray-400">→</span>
                        <span className="text-green-600 font-medium">{configChanges.embedding_model}</span>
                      </div>
                    </div>
                  )}
                  {hasStrategyChange && selectedStrategy && (
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-600">Chunking Strategy:</span>
                      <div className="flex items-center gap-2">
                        <span className="text-red-600 line-through">{CHUNKING_STRATEGIES[collection.chunking_strategy as ChunkingStrategyType]?.name || 'None'}</span>
                        <span className="text-gray-400">→</span>
                        <span className="text-green-600 font-medium">
                          {CHUNKING_STRATEGIES[selectedStrategy]?.name}
                        </span>
                      </div>
                    </div>
                  )}
                  {hasInstructionChange && (
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-600">Embedding Instruction:</span>
                      <span className="text-green-600 font-medium">Updated</span>
                    </div>
                  )}
                  
                </div>
                
                {/* Impact summary */}
                <div className="mt-3 text-sm text-gray-600">
                  <p className="font-medium">Estimated impact:</p>
                  <ul className="mt-1 list-disc list-inside text-xs">
                    <li>Processing time: ~{Math.ceil(collection.document_count / 100)} minutes (estimate)</li>
                    <li>Collection will be read-only during re-indexing</li>
                    {hasModelChange && (
                      <li>Model change may significantly affect search results</li>
                    )}
                  </ul>
                </div>
              </div>
            </div>

            <form onSubmit={handleSubmit}>
              <div className="mb-4">
                <label htmlFor="confirm-reindex" className="block text-sm font-medium text-gray-700 mb-2">
                  To confirm, type <span className="font-mono bg-gray-100 px-1 py-0.5 rounded">reindex {collection.name}</span>
                </label>
                <input
                  id="confirm-reindex"
                  type="text"
                  value={confirmText}
                  onChange={(e) => setConfirmText(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-red-500 focus:border-red-500"
                  placeholder="Type the confirmation text"
                  autoComplete="off"
                  autoFocus
                  aria-label="Confirmation text for re-indexing"
                  aria-describedby="confirm-help-text"
                />
                <p id="confirm-help-text" className="sr-only">
                  Type the exact phrase shown above to confirm the re-index operation
                </p>
              </div>

              <div className="flex gap-3 justify-end">
                <button
                  type="button"
                  onClick={handleCancel}
                  disabled={isSubmitting || reindexCollectionMutation.isPending}
                  className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={!isConfirmValid || isSubmitting || reindexCollectionMutation.isPending}
                  className="px-4 py-2 text-sm font-medium text-white bg-red-600 border border-transparent rounded-md hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isSubmitting || reindexCollectionMutation.isPending ? 'Starting Re-index...' : 'Re-index Collection'}
                </button>
              </div>
            </form>
          </div>
        </div>
      </div>
    </>
  );
}

export default ReindexCollectionModal;
