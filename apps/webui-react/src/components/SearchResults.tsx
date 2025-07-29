import { useState } from 'react';
import { useSearchStore } from '../stores/searchStore';
import { useUIStore } from '../stores/uiStore';
import { AlertTriangle, ChevronRight, FileText, Layers } from 'lucide-react';
import { GPUMemoryError } from './GPUMemoryError';

function SearchResults() {
  const { 
    results, 
    loading, 
    error, 
    rerankingMetrics,
    failedCollections,
    partialFailure
  } = useSearchStore();
  const setShowDocumentViewer = useUIStore((state) => state.setShowDocumentViewer);
  const [expandedDocs, setExpandedDocs] = useState<Set<string>>(new Set());
  const [expandedCollections, setExpandedCollections] = useState<Set<string>>(new Set());

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex justify-center items-center py-12">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          <span className="ml-3 text-gray-600">Searching...</span>
        </div>
      </div>
    );
  }

  if (error) {
    // Check if this is a GPU memory error
    if (error === 'GPU_MEMORY_ERROR' && (window as Window & { __gpuMemoryError?: { message: string; suggestion: string; currentModel: string } }).__gpuMemoryError) {
      const gpuError = (window as Window & { __gpuMemoryError?: { message: string; suggestion: string; currentModel: string } }).__gpuMemoryError;
      return (
        <div className="bg-white rounded-lg shadow-md p-6">
          <GPUMemoryError
            suggestion={gpuError.suggestion}
            currentModel={gpuError.currentModel}
            onSelectSmallerModel={(model) => {
              // This will be handled by the parent component
              // We need to pass this handler from SearchInterface
              if ((window as Window & { __handleSelectSmallerModel?: (model: string) => void }).__handleSelectSmallerModel) {
                (window as Window & { __handleSelectSmallerModel?: (model: string) => void }).__handleSelectSmallerModel(model);
              }
            }}
          />
        </div>
      );
    }
    
    // Regular error display
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="bg-red-50 border border-red-200 rounded-md p-4">
          <p className="text-sm text-red-600">{error}</p>
        </div>
      </div>
    );
  }

  if (results.length === 0 && failedCollections.length === 0) {
    return null;
  }

  // Group results first by collection, then by document
  // Note: Results without a collection_id are grouped under 'unknown' for consistency
  const groupedByCollection = results.reduce((acc, result) => {
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
      chunks: typeof results;
    }>;
    totalResults: number;
  }>);

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

  const toggleCollectionExpansion = (collectionId: string) => {
    const newExpanded = new Set(expandedCollections);
    if (newExpanded.has(collectionId)) {
      newExpanded.delete(collectionId);
    } else {
      newExpanded.add(collectionId);
    }
    setExpandedCollections(newExpanded);
  };

  // Auto-expand all collections by default if there are results
  if (expandedCollections.size === 0 && Object.keys(groupedByCollection).length > 0) {
    setExpandedCollections(new Set(Object.keys(groupedByCollection)));
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
                      <span className="font-medium">{failed.collection_name}</span>: {failed.error_message}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Main results container */}
      <div className="bg-white rounded-lg shadow-md">
        <div className="p-6 border-b border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold text-gray-900">Search Results</h3>
              <p className="mt-1 text-sm text-gray-600">
                Found {results.length} results across {Object.keys(groupedByCollection).length} collections
              </p>
            </div>
            {rerankingMetrics?.rerankingUsed && (
              <div className="text-right">
                <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                  <svg className="mr-1 h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                  Reranked
                </span>
                {rerankingMetrics.rerankingTimeMs && (
                  <p className="mt-1 text-xs text-gray-500">
                    {rerankingMetrics.rerankingTimeMs.toFixed(0)}ms
                  </p>
                )}
              </div>
            )}
          </div>
        </div>

        <div className="divide-y divide-gray-200">
          {Object.entries(groupedByCollection).map(([collectionId, collection]) => {
            const isCollectionExpanded = expandedCollections.has(collectionId);
            
            return (
              <div key={collectionId} className="bg-white">
                {/* Collection Header */}
                <div
                  className="px-6 py-4 bg-gray-50 cursor-pointer hover:bg-gray-100 transition-colors"
                  onClick={() => toggleCollectionExpansion(collectionId)}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center">
                      <ChevronRight 
                        className={`h-5 w-5 text-gray-400 transform transition-transform ${
                          isCollectionExpanded ? 'rotate-90' : ''
                        }`}
                      />
                      <Layers className="ml-2 h-5 w-5 text-gray-500" />
                      <h4 className="ml-3 text-sm font-semibold text-gray-900">
                        {collection.name}
                      </h4>
                    </div>
                    <div className="flex items-center space-x-4">
                      <span className="text-sm text-gray-500">
                        {collection.totalResults} result{collection.totalResults !== 1 ? 's' : ''} in{' '}
                        {Object.keys(collection.documents).length} document{Object.keys(collection.documents).length !== 1 ? 's' : ''}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Documents in Collection */}
                {isCollectionExpanded && (
                  <div className="divide-y divide-gray-200">
                    {Object.entries(collection.documents).map(([docId, doc]) => {
                      const isDocExpanded = expandedDocs.has(docId);
                      const maxScore = Math.max(...doc.chunks.map(c => c.score));
                      
                      return (
                        <div key={docId} className="hover:bg-gray-50 transition-colors">
                          {/* Document Header */}
                          <div
                            className="px-6 py-4 cursor-pointer"
                            onClick={() => toggleDocExpansion(docId)}
                          >
                            <div className="flex items-center justify-between">
                              <div className="flex items-center flex-1">
                                <ChevronRight
                                  className={`h-4 w-4 text-gray-400 transform transition-transform ml-6 ${
                                    isDocExpanded ? 'rotate-90' : ''
                                  }`}
                                />
                                <FileText className="ml-2 h-4 w-4 text-gray-400" />
                                <div className="ml-3">
                                  <p className="text-sm font-medium text-gray-900">{doc.file_name}</p>
                                  <p className="text-xs text-gray-500">{doc.file_path}</p>
                                </div>
                              </div>
                              <div className="flex items-center space-x-4">
                                <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                                  {doc.chunks.length} chunk{doc.chunks.length > 1 ? 's' : ''}
                                </span>
                                <span className="text-sm text-gray-500">
                                  Max score: {maxScore.toFixed(3)}
                                </span>
                              </div>
                            </div>
                          </div>

                          {/* Chunks (Expandable) */}
                          {isDocExpanded && (
                            <div className="bg-gray-50 border-t border-gray-200">
                              {doc.chunks.map((chunk, index) => (
                                <div
                                  key={chunk.chunk_id}
                                  className={`px-6 py-4 hover:bg-gray-100 cursor-pointer transition-colors ${
                                    index > 0 ? 'border-t border-gray-200' : ''
                                  }`}
                                  onClick={() => handleViewDocument(chunk.collection_id, docId, chunk.chunk_id)}
                                >
                                  <p className="text-sm text-gray-700 line-clamp-3">{chunk.content}</p>
                                  <div className="mt-2 flex items-center justify-between">
                                    <div className="flex items-center space-x-4 text-xs text-gray-500">
                                      <span>Chunk {chunk.chunk_index + 1} of {chunk.total_chunks}</span>
                                      <span className="font-medium">Score: {chunk.score.toFixed(3)}</span>
                                    </div>
                                    <button
                                      className="text-blue-600 hover:text-blue-800 text-sm"
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
          <div className="p-6 text-center text-gray-500">
            No results found
          </div>
        )}
      </div>
    </div>
  );
}

export default SearchResults;