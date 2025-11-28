import { useEffect } from 'react';
import { useUIStore } from '../stores/uiStore';
import { useSearchStore } from '../stores/searchStore';
import DocumentViewer from './DocumentViewer';
import ErrorBoundary from './ErrorBoundary';

function DocumentViewerModal() {
  const { showDocumentViewer, setShowDocumentViewer } = useUIStore();
  const searchQuery = useSearchStore((state) => state.searchParams.query);

  useEffect(() => {
    // Prevent body scroll when modal is open
    if (showDocumentViewer) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = 'unset';
    }

    return () => {
      document.body.style.overflow = 'unset';
    };
  }, [showDocumentViewer]);

  if (!showDocumentViewer) return null;

  const { collectionId, docId, chunkId, chunkContent, startOffset, endOffset } = showDocumentViewer;

  return (
    <ErrorBoundary
      level="component"
      resetKeys={[collectionId, docId, chunkId ?? '']}
      fallback={(error, resetError) => (
        <div className="fixed inset-0 z-50 overflow-y-auto">
          <div className="flex items-center justify-center min-h-screen p-4">
            <div className="fixed inset-0 bg-black opacity-50" onClick={() => setShowDocumentViewer(null)} />
            <div className="relative bg-white rounded-lg max-w-md w-full p-6">
              <h3 className="text-lg font-medium text-gray-900 mb-2">Unable to load document</h3>
              <p className="text-sm text-gray-600 mb-4">{error.message}</p>
              <div className="flex space-x-3">
                <button
                  onClick={resetError}
                  className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
                >
                  Try Again
                </button>
                <button
                  onClick={() => setShowDocumentViewer(null)}
                  className="px-4 py-2 bg-gray-200 text-gray-700 rounded hover:bg-gray-300"
                >
                  Close
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    >
      <DocumentViewer
        collectionId={collectionId}
        docId={docId}
        chunkId={chunkId}
        chunkContent={chunkContent}
        startOffset={startOffset}
        endOffset={endOffset}
        query={searchQuery}
        onClose={() => setShowDocumentViewer(null)}
      />
    </ErrorBoundary>
  );
}

export default DocumentViewerModal;