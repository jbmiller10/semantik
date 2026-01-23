import { useState, useEffect, useMemo } from 'react';
import { useSearchStore, type SearchResult } from '../stores/searchStore';
import { useUIStore } from '../stores/uiStore';
import { AlertTriangle, ChevronRight, FileText, Layers, Sparkles } from 'lucide-react';
import { GPUMemoryError } from './GPUMemoryError';

interface SearchResultsProps {
  onSelectSmallerModel?: (model: string) => void;
}

function SearchResults({ onSelectSmallerModel }: SearchResultsProps = {}) {
  const {
    results,
    loading,
    error,
    rerankingMetrics,
    failedCollections,
    partialFailure,
    hydeUsed,
    hydeInfo
  } = useSearchStore();
  const gpuMemoryError = useSearchStore((state) => state.gpuMemoryError);
  const setShowDocumentViewer = useUIStore((state) => state.setShowDocumentViewer);
  const [expandedDocs, setExpandedDocs] = useState<Set<string>>(new Set());
  const [expandedCollections, setExpandedCollections] = useState<Set<string>>(new Set());
  const [showHydeQuery, setShowHydeQuery] = useState(false);

  // Group results first by collection, then by document
  // Note: Results without a collection_id are grouped under 'unknown' for consistency
  // useMemo to prevent unnecessary recalculations and useEffect triggers
  const groupedByCollection = useMemo(() => results.reduce((acc, result) => {
    const collectionId = result.collection_id || 'unknown';
    const collectionName = result.collection_name || 'Unknown Collection';

    if (!acc[collectionId]) {
      acc[collectionId] = {
        name: collectionName,
        documents: {},
        totalResults: 0,
      };
    }

    if (!acc[collectionId].documents[result.doc_id]) {
      acc[collectionId].documents[result.doc_id] = {
        file_path: result.file_path,
        file_name: result.file_name,
        chunks: [],
      };
    }

    acc[collectionId].documents[result.doc_id].chunks.push(result);
    acc[collectionId].totalResults++;

    return acc;
  }, {} as Record<string, {
    name: string;
    documents: Record<string, {
      file_path: string;
      file_name: string;
      chunks: SearchResult[];
    }>;
    totalResults: number;
  }>), [results]);

  // Auto-expand all collections by default if there are results
  useEffect(() => {
    if (expandedCollections.size === 0 && Object.keys(groupedByCollection).length > 0) {
      setExpandedCollections(new Set(Object.keys(groupedByCollection)));
    }
  }, [groupedByCollection, expandedCollections.size]);

  const handleViewDocument = (collectionId: string | undefined, docId: string, chunkId?: string) => {
    // Ensure we always have a valid collection ID, defaulting to 'unknown' if missing
    const safeCollectionId = collectionId || 'unknown';
    setShowDocumentViewer({ collectionId: safeCollectionId, docId, chunkId });
  };

  const toggleDocExpansion = (docId: string) => {
    const newExpanded = new Set(expandedDocs);
    if (newExpanded.has(docId)) {
      newExpanded.delete(docId);
    } else {
      newExpanded.add(docId);
    }
    setExpandedDocs(newExpanded);
  };

  const toggleCollection = (collectionId: string) => {
    const newExpanded = new Set(expandedCollections);
    if (newExpanded.has(collectionId)) {
      newExpanded.delete(collectionId);
    } else {
      newExpanded.add(collectionId);
    }
    setExpandedCollections(newExpanded);
  };

  if (loading) {
    return (
      <div className="bg-[var(--bg-secondary)] rounded-lg shadow-md p-6 border border-[var(--border)]">
        <div className="flex justify-center items-center py-12">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-[var(--accent-primary)]"></div>
          <span className="ml-3 text-[var(--text-secondary)]">Searching...</span>
        </div>
      </div>
    );
  }

  if (error) {
    // Check if this is a GPU memory error
    if (error === 'GPU_MEMORY_ERROR' && gpuMemoryError) {
      return (
        <div className="bg-[var(--bg-secondary)] rounded-lg shadow-md p-6 border border-[var(--border)]">
          <GPUMemoryError
            suggestion={gpuMemoryError.suggestion}
            currentModel={gpuMemoryError.currentModel}
            onSelectSmallerModel={(model) => onSelectSmallerModel?.(model)}
          />
        </div>
      );
    }

    // Regular error display
    return (
      <div className="bg-[var(--bg-secondary)] rounded-lg shadow-md p-6 border border-[var(--border)]">
        <div className="bg-red-500/10 border border-red-500/30 rounded-md p-4">
          <p className="text-sm text-red-400">{error}</p>
        </div>
      </div>
    );
  }

  if (results.length === 0 && failedCollections.length === 0) {
    return null;
  }

  return (
    <div className="space-y-4">
      {/* Warnings for failed collections */}
      {partialFailure && failedCollections.length > 0 && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
          <div className="flex">
            <AlertTriangle className="h-5 w-5 text-yellow-400 mt-0.5" />
            <div className="ml-3">
              <h3 className="text-sm font-medium text-yellow-800">
                Partial Search Failure
              </h3>
              <div className="mt-2 text-sm text-yellow-700">
                <p>The following collections could not be searched:</p>
                <ul className="mt-1 list-disc list-inside">
                  {failedCollections.map((failed) => (
                    <li key={failed.collection_id}>
                      <span className="font-medium">{failed.collection_name}</span>: {failed.error_message ?? failed.error ?? 'Unknown error'}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* HyDE Query Expansion Info */}
      {hydeUsed && hydeInfo && (
        <div className="bg-[var(--bg-secondary)] rounded-lg shadow-md border border-[var(--border)]">
          <button
            type="button"
            onClick={() => setShowHydeQuery(!showHydeQuery)}
            className="w-full px-6 py-4 flex items-center justify-between text-left hover:bg-[var(--bg-tertiary)] transition-colors rounded-lg"
          >
            <div className="flex items-center space-x-2">
              <Sparkles className="h-5 w-5 text-purple-400" />
              <span className="font-medium text-[var(--text-primary)]">HyDE Query Expansion</span>
              <span className="text-xs text-purple-400 bg-purple-500/15 px-2 py-0.5 rounded-full">Used</span>
            </div>
            <ChevronRight className={`h-5 w-5 text-[var(--text-muted)] transform transition-transform ${showHydeQuery ? 'rotate-90' : ''}`} />
          </button>
          {showHydeQuery && hydeInfo.expanded_query && (
            <div className="px-6 pb-4 border-t border-[var(--border-subtle)]">
              <div className="mt-3">
                <p className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wide mb-2">Generated Hypothetical Document</p>
                <pre className="bg-[var(--bg-tertiary)] p-4 rounded-lg text-sm whitespace-pre-wrap font-mono text-[var(--text-secondary)] border border-[var(--border-subtle)]">
                  {hydeInfo.expanded_query}
                </pre>
              </div>
              <div className="mt-3 flex items-center justify-between text-xs text-[var(--text-muted)]">
                <span>
                  {hydeInfo.provider && hydeInfo.model && `${hydeInfo.provider} / ${hydeInfo.model}`}
                </span>
                <span>
                  {hydeInfo.tokens_used && `${hydeInfo.tokens_used} tokens`}
                  {hydeInfo.generation_time_ms && ` • ${hydeInfo.generation_time_ms.toFixed(0)}ms`}
                </span>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Main results container */}
      <div className="bg-[var(--bg-secondary)] rounded-lg shadow-md border border-[var(--border)]">
        <div className="p-6 border-b border-[var(--border-subtle)]">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold text-[var(--text-primary)]">Search Results</h3>
              <p className="mt-1 text-sm text-[var(--text-secondary)]">
                Found {results.length} results across {Object.keys(groupedByCollection).length} collections
              </p>
            </div>
            {rerankingMetrics?.rerankingUsed && (
              <div className="text-right">
                <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-500/15 text-blue-400">
                  <svg className="mr-1 h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                  Reranked
                </span>
                {rerankingMetrics.rerankingTimeMs && (
                  <p className="mt-1 text-xs text-[var(--text-muted)]">
                    {rerankingMetrics.rerankingTimeMs.toFixed(0)}ms
                  </p>
                )}
              </div>
            )}
          </div>
        </div>

        <div className="divide-y divide-[var(--border-subtle)]">
          {Object.entries(groupedByCollection).map(([collectionId, collection]) => {
            const isCollectionExpanded = expandedCollections.has(collectionId);

            return (
              <div key={collectionId} className="bg-[var(--bg-secondary)]">
                {/* Collection Header */}
                <button
                  onClick={() => toggleCollection(collectionId)}
                  className="w-full flex items-center justify-between p-4 bg-[var(--bg-tertiary)] hover:bg-[var(--bg-primary)] transition-colors"
                >
                  <div className="flex items-center">
                    <ChevronRight
                      className={`h-5 w-5 text-[var(--text-muted)] transform transition-transform ${isCollectionExpanded ? 'rotate-90' : ''
                        }`}
                    />
                    <Layers className="ml-2 h-5 w-5 text-[var(--text-secondary)]" />
                    <h4 className="ml-3 text-sm font-semibold text-[var(--text-primary)]">
                      {collection.name}
                    </h4>
                  </div>
                  <div className="flex items-center space-x-4">
                    <span className="text-sm text-[var(--text-secondary)]">
                      {collection.totalResults} result{collection.totalResults !== 1 ? 's' : ''} in{' '}
                      {Object.keys(collection.documents).length} document{Object.keys(collection.documents).length !== 1 ? 's' : ''}
                    </span>
                  </div>
                </button>

                {/* Documents in Collection */}
                {isCollectionExpanded && (
                  <div className="divide-y divide-[var(--border-subtle)]">
                    {Object.entries(collection.documents).map(([docId, doc]) => {
                      const isDocExpanded = expandedDocs.has(docId);
                      const maxScoreChunk = doc.chunks.reduce((prev, current) =>
                        (prev.score > current.score) ? prev : current
                      );

                      return (
                        <div key={docId} className="hover:bg-[var(--bg-tertiary)] transition-colors">
                          {/* Document Header */}
                          <div
                            className="px-6 py-4 cursor-pointer"
                            onClick={() => toggleDocExpansion(docId)}
                          >
                            <div className="flex items-center justify-between">
                              <div className="flex items-center flex-1">
                                <ChevronRight
                                  className={`h-4 w-4 text-[var(--text-muted)] transform transition-transform ml-6 ${isDocExpanded ? 'rotate-90' : ''
                                    }`}
                                />
                                <FileText className="ml-2 h-4 w-4 text-[var(--text-muted)]" />
                                <div className="ml-3">
                                  <p className="text-sm font-medium text-[var(--text-primary)]">{doc.file_name}</p>
                                  <p className="text-xs text-[var(--text-muted)]">{doc.file_path}</p>
                                </div>
                              </div>
                              <div className="flex items-center space-x-4">
                                <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-500/15 text-blue-400">
                                  {doc.chunks.length} chunk{doc.chunks.length > 1 ? 's' : ''}
                                </span>
                                {maxScoreChunk && (
                                  <div className="flex items-center space-x-2 mt-2">
                                    <span className={`
                                      inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium
                                      ${maxScoreChunk.score > 0.7 ? 'bg-green-500/15 text-green-400' :
                                        maxScoreChunk.score > 0.5 ? 'bg-yellow-500/15 text-yellow-400' :
                                          'bg-[var(--bg-tertiary)] text-[var(--text-secondary)]'}
                                    `}>
                                      Score: {maxScoreChunk.score.toFixed(3)}
                                    </span>

                                    {maxScoreChunk.reranked_score !== undefined && maxScoreChunk.original_score !== undefined && (
                                      <span className="text-xs text-[var(--text-muted)]" title="Original score before reranking">
                                        (Original: {maxScoreChunk.original_score.toFixed(3)})
                                      </span>
                                    )}

                                    {maxScoreChunk.embedding_model && (
                                      <span className="text-xs text-[var(--text-muted)] border border-[var(--border-subtle)] px-2 py-0.5 rounded">
                                        {maxScoreChunk.embedding_model}
                                      </span>
                                    )}
                                  </div>
                                )}
                              </div>
                            </div>
                          </div>

                          {/* Chunks (Expandable) */}
                          {isDocExpanded && (
                            <div className="bg-[var(--bg-tertiary)] border-t border-[var(--border-subtle)]">
                              {doc.chunks.map((chunk, index) => (
                                <div
                                  key={chunk.chunk_id || `${docId}-${chunk.chunk_index}`}
                                  className={`px-4 py-3 bg-[var(--bg-tertiary)] cursor-pointer hover:bg-[var(--bg-primary)] transition-colors ${index !== doc.chunks.length - 1 ? 'border-b border-[var(--border-subtle)]' : ''
                                    }`}
                                  onClick={() => {/* TODO: Open chunk detail/context view */ }}
                                >
                                  <p className="text-sm text-[var(--text-secondary)] line-clamp-3">{chunk.content}</p>
                                  <div className="mt-2 flex items-center justify-between">
                                    <div className="flex items-center space-x-4 text-xs text-[var(--text-muted)]">
                                      <span>Chunk {chunk.chunk_index + 1} of {chunk.total_chunks}</span>
                                      <span className="font-medium">Score: {chunk.score.toFixed(3)}</span>
                                    </div>
                                    <button
                                      className="text-[var(--accent-primary)] hover:underline text-sm"
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        handleViewDocument(chunk.collection_id, docId, chunk.chunk_id);
                                      }}
                                    >
                                      View Document →
                                    </button>
                                  </div>
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {results.length === 0 && failedCollections.length === 0 && (
          <div className="p-6 text-center text-[var(--text-secondary)]">
            No results found
          </div>
        )}
      </div>
    </div>
  );
}

export default SearchResults;
