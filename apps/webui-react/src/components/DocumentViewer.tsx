import { useEffect, useState, useRef, useCallback } from 'react';
import api from '../services/api';

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

interface DocumentInfo {
  doc_id: string;
  filename: string;
  path: string;
  size: number;
  extension: string;
  modified: string;
  supported: boolean;
}

interface DocumentViewerProps {
  jobId: string;
  docId: string;
  query?: string;
  onClose: () => void;
}

const LIBRARY_URLS = {
  pdfjs: 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js',
  pdfjsWorker: 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js',
  mammoth: 'https://cdn.jsdelivr.net/npm/mammoth@1.6.0/mammoth.browser.min.js',
  marked: 'https://cdn.jsdelivr.net/npm/marked@9.1.6/marked.min.js',
  dompurify: 'https://cdn.jsdelivr.net/npm/dompurify@3.0.6/dist/purify.min.js',
  markjs: 'https://cdnjs.cloudflare.com/ajax/libs/mark.js/8.11.1/mark.min.js',
  emlformat: '/static/libs/eml-format.browser.min.js',
};

function DocumentViewer({ jobId, docId, query, onClose }: DocumentViewerProps) {

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [documentInfo, setDocumentInfo] = useState<DocumentInfo | null>(null);
  const [content, setContent] = useState<string>('');
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [highlights, setHighlights] = useState<Element[]>([]);
  const [currentHighlight, setCurrentHighlight] = useState(0);

  const contentRef = useRef<HTMLDivElement>(null);
  const markInstanceRef = useRef<any>(null);
  const pdfDocRef = useRef<any>(null);
  const pdfPageRef = useRef<any>(null);

  // Dynamic library loading
  const loadLibrary = useCallback(async (name: keyof typeof LIBRARY_URLS): Promise<void> => {
    const url = LIBRARY_URLS[name];
    const fallbackUrl = url.replace('https://cdnjs.cloudflare.com', '/static/libs')
      .replace('https://cdn.jsdelivr.net', '/static/libs');

    return new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = url;
      script.onload = () => resolve();
      script.onerror = () => {
        // Try fallback
        script.src = fallbackUrl;
        script.onload = () => resolve();
        script.onerror = () => reject(new Error(`Failed to load ${name}`));
      };
      document.head.appendChild(script);
    });
  }, []);

  // Load document info
  useEffect(() => {
    const loadDocumentInfo = async () => {
      try {
        const response = await api.get(`/api/documents/${jobId}/${docId}/info`);
        setDocumentInfo(response.data);
      } catch (err: any) {
        setError(err.response?.data?.detail || 'Failed to load document info');
        setLoading(false);
      }
    };

    loadDocumentInfo();
  }, [jobId, docId]);

  // Load and render document
  useEffect(() => {
    if (!documentInfo) return;

    const loadDocument = async () => {
      try {
        const fileExt = documentInfo.filename.split('.').pop()?.toLowerCase() || '';
        
        switch (fileExt) {
          case 'pdf':
            await loadPDF();
            break;
          case 'docx':
            await loadDOCX();
            break;
          case 'pptx':
            await loadPPTX();
            break;
          case 'txt':
            await loadTXT();
            break;
          case 'md':
            await loadMarkdown();
            break;
          case 'html':
          case 'htm':
            await loadHTML();
            break;
          case 'eml':
            await loadEML();
            break;
          case 'doc':
            setError('Legacy .doc format - download only');
            break;
          default:
            setError(`Unsupported file type: ${fileExt}`);
        }
      } catch (err: any) {
        setError(err.message || 'Failed to load document');
      } finally {
        setLoading(false);
      }
    };

    loadDocument();
  }, [documentInfo]);

  // Apply highlights when content or query changes
  useEffect(() => {
    if (!query || !content || !contentRef.current) return;

    const applyHighlights = async () => {
      if (!window.Mark) {
        await loadLibrary('markjs');
      }

      // Clear previous marks
      if (markInstanceRef.current) {
        markInstanceRef.current.unmark();
      }

      // Create new mark instance
      markInstanceRef.current = new window.Mark(contentRef.current);
      
      // Apply highlights
      markInstanceRef.current.mark(query, {
        separateWordSearch: false,
        done: () => {
          const marks = contentRef.current?.querySelectorAll('mark') || [];
          setHighlights(Array.from(marks));
          if (marks.length > 0) {
            marks[0].scrollIntoView({ behavior: 'smooth', block: 'center' });
          }
        },
      });
    };

    applyHighlights();
  }, [content, query, loadLibrary]);

  const loadPDF = async () => {
    await loadLibrary('pdfjs');
    
    if (!window.pdfjsLib.GlobalWorkerOptions.workerSrc) {
      window.pdfjsLib.GlobalWorkerOptions.workerSrc = LIBRARY_URLS.pdfjsWorker;
    }

    const response = await api.get(`/api/documents/${jobId}/${docId}`, {
      responseType: 'arraybuffer',
    });

    const loadingTask = window.pdfjsLib.getDocument({
      data: response.data,
      cMapUrl: 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/cmaps/',
      cMapPacked: true,
    });

    pdfDocRef.current = await loadingTask.promise;
    setTotalPages(pdfDocRef.current.numPages);
    await renderPDFPage(1);
  };

  const renderPDFPage = async (pageNum: number) => {
    if (!pdfDocRef.current || !contentRef.current) return;

    const page = await pdfDocRef.current.getPage(pageNum);
    pdfPageRef.current = page;

    const viewport = page.getViewport({ scale: 1.5 });
    
    // Clear previous content
    contentRef.current.innerHTML = '';

    // Create canvas
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    canvas.height = viewport.height;
    canvas.width = viewport.width;
    contentRef.current.appendChild(canvas);

    // Render PDF
    await page.render({
      canvasContext: context,
      viewport: viewport,
    }).promise;

    // Create text layer
    const textLayer = document.createElement('div');
    textLayer.className = 'textLayer';
    textLayer.style.position = 'absolute';
    textLayer.style.left = '0';
    textLayer.style.top = '0';
    textLayer.style.right = '0';
    textLayer.style.bottom = '0';
    contentRef.current.appendChild(textLayer);

    // Get text content
    const textContent = await page.getTextContent();
    
    // Render text layer
    window.pdfjsLib.renderTextLayer({
      textContent: textContent,
      container: textLayer,
      viewport: viewport,
      textDivs: [],
    });

    setContent(textContent.items.map((item: any) => item.str).join(' '));
  };

  const loadDOCX = async () => {
    await loadLibrary('mammoth');
    
    const response = await api.get(`/api/documents/${jobId}/${docId}`, {
      responseType: 'arraybuffer',
    });

    const result = await window.mammoth.convertToHtml({ arrayBuffer: response.data });
    setContent(result.value);
    if (contentRef.current) {
      contentRef.current.innerHTML = result.value;
    }
  };

  const loadPPTX = async () => {
    const response = await api.get(`/api/documents/${jobId}/${docId}`, {
      headers: {
        'Accept': 'text/markdown',
      },
    });

    if (response.headers['content-type']?.includes('markdown')) {
      await loadLibrary('marked');
      await loadLibrary('dompurify');
      
      const html = window.marked.parse(response.data);
      const clean = window.DOMPurify.sanitize(html);
      setContent(clean);
      if (contentRef.current) {
        contentRef.current.innerHTML = clean;
      }
    } else {
      setError('PPTX preview not available - download only');
    }
  };

  const loadTXT = async () => {
    const response = await api.get(`/api/documents/${jobId}/${docId}`, {
      responseType: 'text',
    });
    
    const text = response.data;
    setContent(text);
    if (contentRef.current) {
      contentRef.current.textContent = text;
    }
  };

  const loadMarkdown = async () => {
    await loadLibrary('marked');
    await loadLibrary('dompurify');
    
    const response = await api.get(`/api/documents/${jobId}/${docId}`, {
      responseType: 'text',
    });
    
    const html = window.marked.parse(response.data);
    const clean = window.DOMPurify.sanitize(html);
    setContent(clean);
    if (contentRef.current) {
      contentRef.current.innerHTML = clean;
    }
  };

  const loadHTML = async () => {
    await loadLibrary('dompurify');
    
    const response = await api.get(`/api/documents/${jobId}/${docId}`, {
      responseType: 'text',
    });
    
    const clean = window.DOMPurify.sanitize(response.data);
    setContent(clean);
    if (contentRef.current) {
      contentRef.current.innerHTML = clean;
    }
  };

  const loadEML = async () => {
    await loadLibrary('emlformat');
    
    const response = await api.get(`/api/documents/${jobId}/${docId}`, {
      responseType: 'text',
    });
    
    const email = window.emlformat.parse(response.data);
    const html = `
      <div class="email-content">
        <div class="email-headers">
          <p><strong>From:</strong> ${email.headers.from || ''}</p>
          <p><strong>To:</strong> ${email.headers.to || ''}</p>
          <p><strong>Subject:</strong> ${email.headers.subject || ''}</p>
          <p><strong>Date:</strong> ${email.headers.date || ''}</p>
        </div>
        <hr/>
        <div class="email-body">${email.body || ''}</div>
      </div>
    `;
    
    setContent(html);
    if (contentRef.current) {
      contentRef.current.innerHTML = html;
    }
  };

  const handlePageChange = (newPage: number) => {
    if (newPage >= 1 && newPage <= totalPages) {
      setCurrentPage(newPage);
      renderPDFPage(newPage);
    }
  };

  const handleHighlightNavigation = (direction: 'prev' | 'next') => {
    if (highlights.length === 0) return;

    let newIndex = currentHighlight;
    if (direction === 'next') {
      newIndex = (currentHighlight + 1) % highlights.length;
    } else {
      newIndex = currentHighlight === 0 ? highlights.length - 1 : currentHighlight - 1;
    }

    setCurrentHighlight(newIndex);
    highlights[newIndex].scrollIntoView({ behavior: 'smooth', block: 'center' });
  };

  const handleDownload = () => {
    const link = document.createElement('a');
    link.href = `/api/documents/${jobId}/${docId}`;
    link.download = documentInfo?.filename || 'document';
    link.click();
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
              {documentInfo?.filename || 'Document Viewer'}
            </h3>
            
            <div className="flex items-center space-x-2">
              {highlights.length > 0 && (
                <div className="flex items-center space-x-2">
                  <button
                    onClick={() => handleHighlightNavigation('prev')}
                    className="p-1 text-gray-600 hover:text-gray-900"
                  >
                    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                    </svg>
                  </button>
                  <span className="text-sm text-gray-600">
                    {currentHighlight + 1} / {highlights.length}
                  </span>
                  <button
                    onClick={() => handleHighlightNavigation('next')}
                    className="p-1 text-gray-600 hover:text-gray-900"
                  >
                    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </button>
                </div>
              )}
              
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

          {/* Footer - PDF Navigation */}
          {documentInfo?.filename.endsWith('.pdf') && totalPages > 1 && (
            <div className="flex items-center justify-center p-4 border-t">
              <button
                onClick={() => handlePageChange(currentPage - 1)}
                disabled={currentPage === 1}
                className="px-3 py-1 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Previous
              </button>
              <span className="mx-4 text-sm text-gray-700">
                Page {currentPage} of {totalPages}
              </span>
              <button
                onClick={() => handlePageChange(currentPage + 1)}
                disabled={currentPage === totalPages}
                className="px-3 py-1 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Next
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default DocumentViewer;