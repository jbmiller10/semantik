import { useEffect, useState, useRef } from 'react';
// TODO: Document viewing endpoints need to be implemented in the v2 API
// The legacy endpoints (/api/documents/{collectionId}/{docId}/info and /api/documents/{collectionId}/{docId})
// have been removed from the backend. This component is temporarily disabled until the backend
// implements document viewing functionality in the v2 API.

// Declare global types for external libraries
declare global {
  interface Window {
    pdfjsLib: any;
    mammoth: any;
    marked: any;
    DOMPurify: any;
    Mark: any;
    emlformat: any;
  }
}

interface DocumentViewerProps {
  collectionId: string;
  docId: string;
  query?: string;
  onClose: () => void;
}

function DocumentViewer({ collectionId, docId, onClose }: DocumentViewerProps) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const contentRef = useRef<HTMLDivElement>(null);
  const markInstanceRef = useRef<any>(null);
  const pdfDocRef = useRef<any>(null);

  // Load document info
  useEffect(() => {
    // Document viewing endpoints not yet implemented in v2 API
    setError('Document viewing is temporarily unavailable. The backend v2 API does not yet support document content retrieval.');
    setLoading(false);
  }, [collectionId, docId]);

  // Apply highlights when content or query changes
  useEffect(() => {
    // Disabled until document content can be loaded
  }, []);

  const handleDownload = () => {
    // TODO: Implement document download in v2 API
    alert('Document download is temporarily unavailable. The backend v2 API does not yet support document retrieval.');
  };

  // Cleanup
  useEffect(() => {
    return () => {
      if (markInstanceRef.current) {
        markInstanceRef.current.unmark();
      }
      if (pdfDocRef.current) {
        pdfDocRef.current.destroy();
      }
    };
  }, []);

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      <div className="flex items-center justify-center min-h-screen p-4">
        <div className="fixed inset-0 bg-black opacity-50" onClick={onClose} />
        
        <div className="relative bg-white rounded-lg max-w-6xl w-full max-h-[90vh] flex flex-col">
          {/* Header */}
          <div className="flex items-center justify-between p-4 border-b">
            <h3 className="text-lg font-medium text-gray-900">
              Document Viewer
            </h3>
            
            <div className="flex items-center space-x-2">
              <button
                onClick={handleDownload}
                className="p-2 text-gray-600 hover:text-gray-900"
                title="Download"
              >
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                </svg>
              </button>
              
              <button
                onClick={onClose}
                className="p-2 text-gray-600 hover:text-gray-900"
              >
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          </div>

          {/* Content */}
          <div className="flex-1 overflow-auto p-4">
            {loading && (
              <div className="flex items-center justify-center h-64">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600" />
              </div>
            )}
            
            {error && (
              <div className="text-center py-8">
                <p className="text-red-600">{error}</p>
              </div>
            )}
            
            <div
              ref={contentRef}
              className="prose max-w-none"
              style={{ minHeight: '400px' }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

export default DocumentViewer;