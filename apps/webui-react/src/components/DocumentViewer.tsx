import { useEffect, useState, useRef } from 'react';
import { documentsV2Api } from '../services/api/v2';
import { getErrorMessage } from '../utils/errorUtils';
import PdfViewer from './PdfViewer';

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
    docx: {
      renderAsync: (data: Blob | ArrayBuffer, container: HTMLElement, styleContainer?: HTMLElement | null, options?: Record<string, unknown>) => Promise<void>;
      defaultOptions?: Record<string, unknown>;
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

// Script configurations with SRI hashes for security
const SCRIPT_CONFIGS = {
  jszip: {
    url: 'https://unpkg.com/jszip@3.10.1/dist/jszip.min.js',
    integrity: 'sha384-+mbV2IY1Zk/X1p/nWllGySJSUN8uMs+gUAN10Or95UBH0fpj6GfKgPmgC5EXieXG',
    crossOrigin: 'anonymous' as const
  },
  docxPreview: {
    url: 'https://unpkg.com/docx-preview@0.3.2/dist/docx-preview.min.js',
    integrity: 'sha384-WbeDqP/pDz1XLGS3CK6UwoSPLG1dRLX4FQqEEWWBMc4j8KM3s5eojZQGdW9Of0xV',
    crossOrigin: 'anonymous' as const
  }
};

// Helper function to dynamically load scripts with SRI
const loadScript = (config: typeof SCRIPT_CONFIGS[keyof typeof SCRIPT_CONFIGS]): Promise<void> => {
  return new Promise((resolve, reject) => {
    // Check if script is already loaded
    const existingScript = document.querySelector(`script[src="${config.url}"]`);
    if (existingScript) {
      resolve();
      return;
    }

    const script = document.createElement('script');
    script.src = config.url;
    script.integrity = config.integrity;
    script.crossOrigin = config.crossOrigin;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error(`Failed to load script: ${config.url}`));
    document.head.appendChild(script);
  });
};

// DOCX rendering options
const DOCX_RENDER_OPTIONS = {
  className: 'docx',
  inWrapper: true,
  ignoreWidth: false,
  ignoreHeight: false,
  ignoreFonts: false,
  breakPages: true,
  ignoreLastRenderedPageBreak: true,
  experimental: false,
  trimXmlDeclaration: true,
  useBase64URL: false,
  renderHeaders: true,
  renderFooters: true,
  renderFootnotes: true,
  renderEndnotes: true,
} as const;

function DocumentViewer({ collectionId, docId, onClose }: DocumentViewerProps) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [blobUrl, setBlobUrl] = useState<string | null>(null);
  const [isPdf, setIsPdf] = useState(false);

  const contentRef = useRef<HTMLDivElement>(null);
  const markInstanceRef = useRef<InstanceType<typeof window.Mark> | null>(null);

  // Load document content
  useEffect(() => {
    const loadDocument = async () => {
      try {
        setLoading(true);
        setError(null);
        setIsPdf(false);
        setBlobUrl(null);

        if (contentRef.current) {
          contentRef.current.innerHTML = '';
        }

        // Get the document content URL and headers
        const { url, headers } = documentsV2Api.getContent(collectionId, docId);

        // Fetch the document with authentication headers
        const response = await fetch(url, { 
          headers: headers.Authorization ? headers : undefined 
        });
        
        if (!response.ok) {
          if (response.status === 401) {
            throw new Error('Authentication required. Please log in again.');
          } else if (response.status === 403) {
            throw new Error('You do not have permission to view this document.');
          } else if (response.status === 404) {
            throw new Error('Document not found.');
          } else {
            const errorData = await response.json().catch(() => null);
            throw new Error(errorData?.detail || `Failed to load document (${response.status})`);
          }
        }

        // Get content type from response headers
        const contentType = response.headers.get('content-type') || 'application/octet-stream';

        // Handle different content types
        if (contentType.includes('text/') || contentType.includes('application/json')) {
          // Text-based content - read as text
          const text = await response.text();
          
          if (contentRef.current) {
            // Display text content directly
            if (contentType.includes('text/html')) {
              // Sanitize HTML content for security
              contentRef.current.innerHTML = window.DOMPurify ? 
                window.DOMPurify.sanitize(text) : text;
            } else if (contentType.includes('text/markdown')) {
              // Parse markdown if marked.js is available
              const html = window.marked ? window.marked.parse(text) : `<pre>${text}</pre>`;
              contentRef.current.innerHTML = window.DOMPurify ? 
                window.DOMPurify.sanitize(html) : html;
            } else {
              // Display plain text or JSON
              contentRef.current.innerHTML = `<pre style="white-space: pre-wrap; word-wrap: break-word;">${text}</pre>`;
            }
          }
        } else if (contentType.includes('image/')) {
          // Images - create blob URL
          const blob = await response.blob();
          const objectUrl = URL.createObjectURL(blob);
          setBlobUrl(objectUrl);
          
          if (contentRef.current) {
            contentRef.current.innerHTML = `
              <div style="text-align: center;">
                <img src="${objectUrl}" alt="Document" style="max-width: 100%; height: auto;" />
              </div>
            `;
          }
        } else if (contentType.includes('application/pdf')) {
          // PDFs - create blob URL for PDF.js or fallback display
          const blob = await response.blob();
          const objectUrl = URL.createObjectURL(blob);
          setBlobUrl(objectUrl);
          setIsPdf(true);
        } else if (
          contentType.includes('application/vnd.openxmlformats-officedocument.wordprocessingml.document') ||
          contentType.includes('application/msword')
        ) {
          // DOCX files - render using docx-preview library
          const blob = await response.blob();
          
          if (contentRef.current) {
            try {
              // Load required libraries with SRI for security
              await loadScript(SCRIPT_CONFIGS.jszip);
              await loadScript(SCRIPT_CONFIGS.docxPreview);
              
              // Clear container and create a wrapper div for docx content
              contentRef.current.innerHTML = '';
              const docxContainer = document.createElement('div');
              docxContainer.className = 'docx-wrapper';
              contentRef.current.appendChild(docxContainer);
              
              // Render DOCX
              if (window.docx) {
                await window.docx.renderAsync(blob, docxContainer, null, DOCX_RENDER_OPTIONS);
              } else {
                throw new Error('DOCX preview library failed to load');
              }
            } catch (docxError) {
              console.error('Failed to render DOCX:', docxError);
              // Fallback to download
              const objectUrl = URL.createObjectURL(blob);
              setBlobUrl(objectUrl);
              contentRef.current.innerHTML = `
                <div style="text-align: center; padding: 2rem;">
                  <p style="margin-bottom: 1rem;">Unable to preview this Word document.</p>
                  <a href="${objectUrl}" download class="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700">
                    Download Document
                  </a>
                </div>
              `;
            }
          }
        } else {
          // Other binary content - provide download link
          const blob = await response.blob();
          const objectUrl = URL.createObjectURL(blob);
          setBlobUrl(objectUrl);
          
          if (contentRef.current) {
            contentRef.current.innerHTML = `
              <div style="text-align: center; padding: 2rem;">
                <p style="margin-bottom: 1rem;">This file type cannot be displayed directly.</p>
                <a href="${objectUrl}" download class="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700">
                  Download File
                </a>
              </div>
            `;
          }
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

  const handleDownload = async () => {
    try {
      // Get the document content URL and headers
      const { url, headers } = documentsV2Api.getContent(collectionId, docId);
      
      // Always fetch with auth headers for consistency
      const response = await fetch(url, { 
        headers: headers.Authorization ? headers : undefined 
      });
      
      if (!response.ok) {
        throw new Error('Failed to download document');
      }
      
      // Get the filename from Content-Disposition header or use default
      const contentDisposition = response.headers.get('content-disposition');
      let filename = 'document';
      if (contentDisposition) {
        const match = contentDisposition.match(/filename="?([^"]+)"?/);
        if (match && match[1]) {
          filename = match[1];
        }
      }
      
      // Create blob and trigger download
      const blob = await response.blob();
      const blobUrl = URL.createObjectURL(blob);
      
      const link = document.createElement('a');
      link.href = blobUrl;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      // Clean up blob URL
      setTimeout(() => URL.revokeObjectURL(blobUrl), 100);
    } catch (err) {
      console.error('Download failed:', err);
      alert('Failed to download document. Please try again.');
    }
  };

  // Cleanup blob URLs
  useEffect(() => {
    return () => {
      if (blobUrl) {
        URL.revokeObjectURL(blobUrl);
      }
    };
  }, [blobUrl]);

  // Cleanup on unmount
  useEffect(() => {
    const markInstance = markInstanceRef.current;

    return () => {
      if (markInstance) {
        markInstance.unmark();
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
            
            {isPdf && blobUrl ? (
              <PdfViewer
                src={blobUrl}
                className="space-y-6"
                onError={(message) => setError(message)}
              />
            ) : (
              <div
                ref={contentRef}
                className="prose max-w-none"
                style={{ minHeight: '400px' }}
              />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default DocumentViewer;
