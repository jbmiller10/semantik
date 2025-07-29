import { useEffect, useState, useRef } from 'react';
import { documentsV2Api } from '../services/api/v2';
import { getErrorMessage } from '../utils/errorUtils';

// Declare global types for external libraries
declare global {
  interface Window {
    pdfjsLib: {
      getDocument: (url: string) => { promise: Promise<PDFDocumentProxy> };
    };
    mammoth: {
      convertToHtml: (options: { arrayBuffer: ArrayBuffer }) => Promise<{ value: string }>;
    };
    marked: {
      parse: (markdown: string) => string;
    };
    DOMPurify: {
      sanitize: (dirty: string) => string;
    };
    Mark: new (element: HTMLElement) => {
      mark: (term: string, options?: Record<string, unknown>) => void;
      unmark: () => void;
    };
    emlformat: {
      parse: (emlContent: string) => { html: string; headers: Record<string, string> };
    };
  }
}

interface PDFDocumentProxy {
  numPages: number;
  getPage: (pageNumber: number) => Promise<PDFPageProxy>;
  destroy: () => void;
}

interface PDFPageProxy {
  getViewport: (options: { scale: number }) => { width: number; height: number };
  render: (params: { canvasContext: CanvasRenderingContext2D; viewport: unknown }) => { promise: Promise<void> };
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
  const markInstanceRef = useRef<InstanceType<typeof window.Mark> | null>(null);
  const pdfDocRef = useRef<PDFDocumentProxy | null>(null);

  // Load document content
  useEffect(() => {
    const loadDocument = async () => {
      try {
        setLoading(true);
        setError(null);

        // Get the document content URL
        const { url } = documentsV2Api.getContent(collectionId, docId);

        // For now, we'll display the document in an iframe
        // TODO: In the future, add specific handlers for different file types:
        // - PDF: Use pdf.js for better rendering and text selection
        // - DOCX: Use mammoth.js to convert to HTML
        // - TXT/Markdown: Fetch and display as text with syntax highlighting
        // - Images: Display directly with zoom controls
        // This will require fetching document metadata first to determine file type
        
        if (contentRef.current) {
          // Create iframe for document display
          const iframe = document.createElement('iframe');
          iframe.src = url;
          iframe.style.width = '100%';
          iframe.style.height = '100%';
          iframe.style.border = 'none';
          iframe.style.minHeight = '600px';
          
          // Set sandbox attributes for security
          iframe.setAttribute('sandbox', 'allow-same-origin allow-scripts');
          
          // Clear existing content and add iframe
          contentRef.current.innerHTML = '';
          contentRef.current.appendChild(iframe);
        }

        setLoading(false);
      } catch (err) {
        console.error('Failed to load document:', err);
        setError(getErrorMessage(err));
        setLoading(false);
      }
    };

    loadDocument();
  }, [collectionId, docId]);

  // Apply highlights when content or query changes
  useEffect(() => {
    // Disabled until document content can be loaded
  }, []);

  const handleDownload = () => {
    // Get the document content URL and headers
    const { url, headers } = documentsV2Api.getContent(collectionId, docId);
    
    // Create a temporary anchor element to trigger download
    const link = document.createElement('a');
    link.href = url;
    link.download = ''; // This will use the filename from the server response
    
    // If we have auth headers, we need to fetch the file first
    if (headers.Authorization) {
      fetch(url, { headers })
        .then(response => response.blob())
        .then(blob => {
          const blobUrl = URL.createObjectURL(blob);
          link.href = blobUrl;
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
          URL.revokeObjectURL(blobUrl);
        })
        .catch(err => {
          console.error('Download failed:', err);
          alert('Failed to download document. Please try again.');
        });
    } else {
      // No auth required, direct download
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

  // Cleanup
  useEffect(() => {
    // Copy ref values to local variables to avoid React hooks warnings
    const markInstance = markInstanceRef.current;
    const pdfDoc = pdfDocRef.current;
    
    return () => {
      if (markInstance) {
        markInstance.unmark();
      }
      if (pdfDoc) {
        pdfDoc.destroy();
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