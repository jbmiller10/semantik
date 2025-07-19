import { useState } from 'react';
import { useSearchStore } from '../stores/searchStore';
import { useUIStore } from '../stores/uiStore';

function SearchResults() {
  const { results, loading, error, rerankingMetrics } = useSearchStore();
  const setShowDocumentViewer = useUIStore((state) => state.setShowDocumentViewer);
  const [expandedDocs, setExpandedDocs] = useState<Set<string>>(new Set());

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
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="bg-red-50 border border-red-200 rounded-md p-4">
          <p className="text-sm text-red-600">{error}</p>
        </div>
      </div>
    );
  }

  if (results.length === 0) {
    return null;
  }

  // Group results by document
  const groupedResults = results.reduce((acc, result) => {
    if (!acc[result.doc_id]) {
      acc[result.doc_id] = {
        file_path: result.file_path,
        file_name: result.file_name,
        chunks: [],
      };
    }
    acc[result.doc_id].chunks.push(result);
    return acc;
  }, {} as Record<string, { file_path: string; file_name: string; chunks: typeof results }>);

  const handleViewDocument = (jobId: string, docId: string, chunkId?: string) => {
    setShowDocumentViewer({ jobId, docId, chunkId });
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

  // Find max score for each document
  const docMaxScores = Object.entries(groupedResults).reduce((acc, [docId, group]) => {
    acc[docId] = Math.max(...group.chunks.map(c => c.score));
    return acc;
  }, {} as Record<string, number>);

  return (
    <div className="bg-white rounded-lg shadow-md">
      <div className="p-6 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-gray-900">Search Results</h3>
            <p className="mt-1 text-sm text-gray-600">
              Found {results.length} results in {Object.keys(groupedResults).length} documents
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
        {Object.entries(groupedResults).map(([docId, group]) => {
          const isExpanded = expandedDocs.has(docId);
          const maxScore = docMaxScores[docId];
          
          return (
            <div key={docId} className="hover:bg-gray-50 transition-colors">
              {/* Document Header */}
              <div 
                className="px-6 py-4 cursor-pointer"
                onClick={() => toggleDocExpansion(docId)}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center flex-1">
                    <svg 
                      className={`h-5 w-5 text-gray-400 transform transition-transform ${isExpanded ? 'rotate-90' : ''}`}
                      fill="none" 
                      viewBox="0 0 24 24" 
                      stroke="currentColor"
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                    <svg className="ml-2 h-5 w-5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    <div className="ml-3">
                      <p className="text-sm font-medium text-gray-900">{group.file_name}</p>
                      <p className="text-xs text-gray-500">{group.file_path}</p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-4">
                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                      {group.chunks.length} chunk{group.chunks.length > 1 ? 's' : ''}
                    </span>
                    <span className="text-sm text-gray-500">
                      Max score: {maxScore.toFixed(3)}
                    </span>
                  </div>
                </div>
              </div>

              {/* Chunks (Expandable) */}
              {isExpanded && (
                <div className="bg-gray-50 border-t border-gray-200">
                  {group.chunks.map((chunk, index) => (
                    <div
                      key={chunk.chunk_id}
                      className={`px-6 py-4 hover:bg-gray-100 cursor-pointer transition-colors ${
                        index > 0 ? 'border-t border-gray-200' : ''
                      }`}
                      onClick={() => handleViewDocument(chunk.job_id || 'current', docId, chunk.chunk_id)}
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
                            handleViewDocument(chunk.job_id || 'current', docId, chunk.chunk_id);
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
      
      {Object.keys(groupedResults).length === 0 && (
        <div className="p-6 text-center text-gray-500">
          No results found
        </div>
      )}
    </div>
  );
}

export default SearchResults;