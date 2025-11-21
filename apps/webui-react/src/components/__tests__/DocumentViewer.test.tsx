import { render, screen, waitFor } from '@testing-library/react';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';
import userEvent from '@testing-library/user-event';
import DocumentViewer from '../DocumentViewer';
import { documentsV2Api } from '../../services/api/v2';

const { mockPdfViewer } = vi.hoisted(() => ({
  mockPdfViewer: vi.fn(() => <div data-testid="pdf-viewer" />),
}))

vi.mock('../PdfViewer', () => ({
  default: mockPdfViewer,
}));

// Mock the documents API
vi.mock('../../services/api/v2', () => ({
  documentsV2Api: {
    getContent: vi.fn()
  }
}));

// Mock global objects
const mockDOMPurify = {
  sanitize: vi.fn((html: string) => html)
};

const mockMarked = {
  parse: vi.fn((markdown: string) => `<p>${markdown}</p>`)
};

const mockDocx = {
  renderAsync: vi.fn().mockResolvedValue(undefined)
};

// Mock script loading
const mockScripts: Set<string> = new Set();

// Setup global mocks
beforeEach(() => {
  // Mock URL.createObjectURL and URL.revokeObjectURL
  global.URL.createObjectURL = vi.fn(() => 'blob:mock-url');
  global.URL.revokeObjectURL = vi.fn();
  
  // Mock window objects
  (window as Record<string, unknown>).DOMPurify = mockDOMPurify;
  (window as Record<string, unknown>).marked = mockMarked;
  (window as Record<string, unknown>).docx = null; // Reset docx
  
  // Mock document.querySelector for script loading
  const originalQuerySelector = document.querySelector;
  document.querySelector = vi.fn((selector: string) => {
    if (selector.startsWith('script[src=')) {
      const src = selector.match(/src="([^"]+)"/)?.[1];
      return mockScripts.has(src || '') ? {} : null;
    }
    return originalQuerySelector.call(document, selector);
  }) as typeof document.querySelector;

  // Mock document.createElement for script elements
  const originalCreateElement = document.createElement;
  document.createElement = vi.fn((tag: string) => {
    if (tag === 'script') {
      const script = {
        src: '',
        onload: null as (() => void) | null,
        onerror: null as (() => void) | null,
      };
      // Simulate script loading
      setTimeout(() => {
        if (script.src.includes('docx-preview')) {
          (window as Record<string, unknown>).docx = mockDocx;
        }
        if (script.onload) script.onload();
      }, 0);
      return script as unknown as HTMLScriptElement;
    }
    return originalCreateElement.call(document, tag);
  }) as typeof document.createElement;

  // Mock document.head.appendChild
  document.head.appendChild = vi.fn();
  
  // Clear mock scripts
  mockScripts.clear();
  
  // Reset all mocks
  vi.clearAllMocks();
  mockPdfViewer.mockClear();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('DocumentViewer', () => {
  const defaultProps = {
    collectionId: 'test-collection-id',
    docId: 'test-doc-id',
    onClose: vi.fn()
  };

  it('should fetch document with authentication headers', async () => {
    const mockUrl = 'http://api.test/document';
    const mockHeaders = { Authorization: 'Bearer test-token' };
    const mockContent = 'Test document content';
    
    vi.mocked(documentsV2Api.getContent).mockReturnValue({
      url: mockUrl,
      headers: mockHeaders
    });

    // Mock successful fetch response
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      headers: new Headers({ 'content-type': 'text/plain' }),
      text: () => Promise.resolve(mockContent)
    });

    render(<DocumentViewer {...defaultProps} />);

    // Wait for the document to load
    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(mockUrl, {
        headers: mockHeaders
      });
    });

    // Check that content is displayed
    await waitFor(() => {
      expect(screen.getByText(mockContent)).toBeInTheDocument();
    });
  });

  it('should handle authentication errors gracefully', async () => {
    const mockUrl = 'http://api.test/document';
    const mockHeaders = { Authorization: 'Bearer invalid-token' };
    
    vi.mocked(documentsV2Api.getContent).mockReturnValue({
      url: mockUrl,
      headers: mockHeaders
    });

    // Mock 401 response
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 401,
      json: () => Promise.resolve({ detail: 'Not authenticated' })
    });

    render(<DocumentViewer {...defaultProps} />);

    // Wait for error message
    await waitFor(() => {
      expect(screen.getByText('Authentication required. Please log in again.')).toBeInTheDocument();
    });
  });

  it('should handle different content types correctly', async () => {
    const mockUrl = 'http://api.test/document';
    const mockHeaders = { Authorization: 'Bearer test-token' };
    
    vi.mocked(documentsV2Api.getContent).mockReturnValue({
      url: mockUrl,
      headers: mockHeaders
    });

    // Test with HTML content
    const htmlContent = '<h1>Test HTML</h1>';
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      headers: new Headers({ 'content-type': 'text/html' }),
      text: () => Promise.resolve(htmlContent)
    });

    const { rerender } = render(<DocumentViewer {...defaultProps} />);

    await waitFor(() => {
      expect(mockDOMPurify.sanitize).toHaveBeenCalledWith(htmlContent);
    });

    // Test with markdown content
    const markdownContent = '# Test Markdown';
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      headers: new Headers({ 'content-type': 'text/markdown' }),
      text: () => Promise.resolve(markdownContent)
    });

    rerender(<DocumentViewer {...defaultProps} docId="test-doc-2" />);

    await waitFor(() => {
      expect(mockMarked.parse).toHaveBeenCalledWith(markdownContent);
    });
  });

  it('should handle binary content with blob URLs', async () => {
    const mockUrl = 'http://api.test/document';
    const mockHeaders = { Authorization: 'Bearer test-token' };
    
    vi.mocked(documentsV2Api.getContent).mockReturnValue({
      url: mockUrl,
      headers: mockHeaders
    });

    // Mock image response
    const mockBlob = new Blob(['fake image data'], { type: 'image/png' });
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      headers: new Headers({ 'content-type': 'image/png' }),
      blob: () => Promise.resolve(mockBlob)
    });

    render(<DocumentViewer {...defaultProps} />);

    await waitFor(() => {
      expect(global.URL.createObjectURL).toHaveBeenCalledWith(mockBlob);
      const img = screen.getByAltText('Document');
      expect(img).toHaveAttribute('src', 'blob:mock-url');
    });
  });

  it('should render PDFs using the PdfViewer component', async () => {
    const mockUrl = 'http://api.test/document';
    const mockHeaders = { Authorization: 'Bearer test-token' };

    vi.mocked(documentsV2Api.getContent).mockReturnValue({
      url: mockUrl,
      headers: mockHeaders
    });

    const mockBlob = new Blob(['fake pdf data'], { type: 'application/pdf' });
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      headers: new Headers({ 'content-type': 'application/pdf' }),
      blob: () => Promise.resolve(mockBlob)
    });

    render(<DocumentViewer {...defaultProps} />);

    await waitFor(() => {
      expect(mockPdfViewer).toHaveBeenCalled();
      const [props] = mockPdfViewer.mock.calls[0];
      expect(props.src).toBe('blob:mock-url');
      expect(screen.getByTestId('pdf-viewer')).toBeInTheDocument();
    });
  });

  it('should clean up blob URLs on unmount', async () => {
    const mockUrl = 'http://api.test/document';
    const mockHeaders = { Authorization: 'Bearer test-token' };
    
    vi.mocked(documentsV2Api.getContent).mockReturnValue({
      url: mockUrl,
      headers: mockHeaders
    });

    // Mock image response
    const mockBlob = new Blob(['fake image data'], { type: 'image/png' });
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      headers: new Headers({ 'content-type': 'image/png' }),
      blob: () => Promise.resolve(mockBlob)
    });

    const { unmount } = render(<DocumentViewer {...defaultProps} />);

    await waitFor(() => {
      expect(global.URL.createObjectURL).toHaveBeenCalled();
    });

    unmount();

    expect(global.URL.revokeObjectURL).toHaveBeenCalledWith('blob:mock-url');
  });

  it('should handle download functionality', async () => {
    const mockUrl = 'http://api.test/document';
    const mockHeaders = { Authorization: 'Bearer test-token' };
    const mockFilename = 'test-document.pdf';
    
    vi.mocked(documentsV2Api.getContent).mockReturnValue({
      url: mockUrl,
      headers: mockHeaders
    });

    // Mock PDF response
    const mockBlob = new Blob(['fake pdf data'], { type: 'application/pdf' });
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      headers: new Headers({ 
        'content-type': 'application/pdf',
        'content-disposition': `attachment; filename="${mockFilename}"`
      }),
      blob: () => Promise.resolve(mockBlob)
    });

    render(<DocumentViewer {...defaultProps} />);

    // Wait for document to load
    await waitFor(() => {
      expect(screen.getByTitle('Download')).toBeInTheDocument();
      expect(mockPdfViewer).toHaveBeenCalled();
    });

    // Mock createElement and click only when needed
    const mockLink = {
      href: '',
      download: '',
      click: vi.fn()
    };
    const originalCreateElement = document.createElement;
    document.createElement = vi.fn((tag) => {
      if (tag === 'a') return mockLink as unknown as HTMLAnchorElement;
      return originalCreateElement.call(document, tag);
    });
    const originalAppendChild = document.body.appendChild;
    const originalRemoveChild = document.body.removeChild;
    document.body.appendChild = vi.fn();
    document.body.removeChild = vi.fn();

    // Click download button
    const downloadButton = screen.getByTitle('Download');
    await userEvent.click(downloadButton);

    await waitFor(() => {
      expect(mockLink.download).toBe(mockFilename);
      expect(mockLink.click).toHaveBeenCalled();
    });

    // Restore original methods
    document.createElement = originalCreateElement;
    document.body.appendChild = originalAppendChild;
    document.body.removeChild = originalRemoveChild;
  });

  it('should handle requests without auth token', async () => {
    const mockUrl = 'http://api.test/document';
    const mockHeaders = {}; // No Authorization header
    const mockContent = 'Public document content';
    
    vi.mocked(documentsV2Api.getContent).mockReturnValue({
      url: mockUrl,
      headers: mockHeaders
    });

    // Mock successful fetch response
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      headers: new Headers({ 'content-type': 'text/plain' }),
      text: () => Promise.resolve(mockContent)
    });

    render(<DocumentViewer {...defaultProps} />);

    // Wait for the document to load
    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(mockUrl, {
        headers: undefined // Should pass undefined when no auth header
      });
    });

    // Check that content is displayed
    await waitFor(() => {
      expect(screen.getByText(mockContent)).toBeInTheDocument();
    });
  });

  it('should render DOCX files using docx-preview library', async () => {
    const mockUrl = 'http://api.test/document';
    const mockHeaders = { Authorization: 'Bearer test-token' };
    
    vi.mocked(documentsV2Api.getContent).mockReturnValue({
      url: mockUrl,
      headers: mockHeaders
    });

    // Mock DOCX response
    const mockBlob = new Blob(['fake docx data'], { 
      type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' 
    });
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      headers: new Headers({ 
        'content-type': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' 
      }),
      blob: () => Promise.resolve(mockBlob)
    });

    render(<DocumentViewer {...defaultProps} />);

    // Wait for scripts to load and docx to render
    await waitFor(() => {
      expect(document.head.appendChild).toHaveBeenCalledTimes(2); // jszip and docx-preview
    });

    await waitFor(() => {
      expect(mockDocx.renderAsync).toHaveBeenCalledWith(
        mockBlob,
        expect.any(HTMLElement),
        null,
        expect.objectContaining({
          className: 'docx',
          inWrapper: true,
          renderHeaders: true,
          renderFooters: true
        })
      );
    });
  });

  it('should handle legacy .doc files', async () => {
    const mockUrl = 'http://api.test/document';
    const mockHeaders = { Authorization: 'Bearer test-token' };
    
    vi.mocked(documentsV2Api.getContent).mockReturnValue({
      url: mockUrl,
      headers: mockHeaders
    });

    // Mock legacy DOC response
    const mockBlob = new Blob(['fake doc data'], { type: 'application/msword' });
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      headers: new Headers({ 'content-type': 'application/msword' }),
      blob: () => Promise.resolve(mockBlob)
    });

    render(<DocumentViewer {...defaultProps} />);

    // Should still attempt to load scripts and render
    await waitFor(() => {
      expect(document.head.appendChild).toHaveBeenCalledTimes(2);
    });

    await waitFor(() => {
      expect(mockDocx.renderAsync).toHaveBeenCalled();
    });
  });

  it('should fall back to download when DOCX rendering fails', async () => {
    const mockUrl = 'http://api.test/document';
    const mockHeaders = { Authorization: 'Bearer test-token' };
    
    vi.mocked(documentsV2Api.getContent).mockReturnValue({
      url: mockUrl,
      headers: mockHeaders
    });

    // Mock DOCX response
    const mockBlob = new Blob(['fake docx data'], { 
      type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' 
    });
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      headers: new Headers({ 
        'content-type': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' 
      }),
      blob: () => Promise.resolve(mockBlob)
    });

    // Make docx.renderAsync fail
    mockDocx.renderAsync.mockRejectedValueOnce(new Error('Render failed'));

    render(<DocumentViewer {...defaultProps} />);

    // Wait for fallback
    await waitFor(() => {
      expect(screen.getByText('Unable to preview this Word document.')).toBeInTheDocument();
      expect(screen.getByText('Download Document')).toBeInTheDocument();
    });
  });

  it('should not reload scripts if already present', async () => {
    const mockUrl = 'http://api.test/document';
    const mockHeaders = { Authorization: 'Bearer test-token' };
    
    // Mark scripts as already loaded
    mockScripts.add('https://unpkg.com/jszip@3.10.1/dist/jszip.min.js');
    mockScripts.add('https://unpkg.com/docx-preview@0.3.2/dist/docx-preview.min.js');
    (window as Record<string, unknown>).docx = mockDocx; // Pre-set docx

    vi.mocked(documentsV2Api.getContent).mockReturnValue({
      url: mockUrl,
      headers: mockHeaders
    });

    // Mock DOCX response
    const mockBlob = new Blob(['fake docx data'], { 
      type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' 
    });
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      headers: new Headers({ 
        'content-type': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' 
      }),
      blob: () => Promise.resolve(mockBlob)
    });

    render(<DocumentViewer {...defaultProps} />);

    // Scripts should not be loaded again
    await waitFor(() => {
      expect(document.head.appendChild).not.toHaveBeenCalled();
    });

    // But rendering should still happen
    await waitFor(() => {
      expect(mockDocx.renderAsync).toHaveBeenCalled();
    });
  });

  it('scrolls and highlights the matching chunk when chunkId is provided', async () => {
    const mockUrl = 'http://api.test/document';
    const mockHeaders = { Authorization: 'Bearer test-token' };

    vi.mocked(documentsV2Api.getContent).mockReturnValue({
      url: mockUrl,
      headers: mockHeaders,
    });

    const html = `
      <div>
        <p data-chunk-id="chunk-1">Alpha content</p>
        <div style="margin-top:1200px" data-chunk-id="chunk-2">Target chunk text</div>
        <p data-chunk-id="chunk-3">Omega content</p>
      </div>
    `;

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      headers: new Headers({ 'content-type': 'text/html' }),
      text: () => Promise.resolve(html),
    });

    const scrollSpy = vi.fn();
    Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', {
      configurable: true,
      value: scrollSpy,
    });

    render(
      <DocumentViewer
        collectionId="c-1"
        docId="d-1"
        chunkId="chunk-2"
        onClose={vi.fn()}
      />,
    );

    await waitFor(() => {
      const target = document.querySelector('[data-chunk-id="chunk-2"]') as HTMLElement;
      expect(target).toBeInTheDocument();
      expect(target.classList.contains('chunk-highlight')).toBe(true);
      expect(scrollSpy).toHaveBeenCalledWith({ behavior: 'smooth', block: 'center' });
    });
  });

  it('should find chunk by id attribute when data-chunk-id is not present', async () => {
    const mockUrl = 'http://api.test/document';
    const mockHeaders = { Authorization: 'Bearer test-token' };

    vi.mocked(documentsV2Api.getContent).mockReturnValue({
      url: mockUrl,
      headers: mockHeaders,
    });

    const html = `
      <div>
        <p id="chunk-100">Content with id attribute</p>
      </div>
    `;

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      headers: new Headers({ 'content-type': 'text/html' }),
      text: () => Promise.resolve(html),
    });

    const scrollSpy = vi.fn();
    Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', {
      configurable: true,
      value: scrollSpy,
    });

    render(
      <DocumentViewer
        collectionId="c-1"
        docId="d-1"
        chunkId="chunk-100"
        onClose={vi.fn()}
      />,
    );

    await waitFor(() => {
      const target = document.querySelector('#chunk-100') as HTMLElement;
      expect(target).toBeInTheDocument();
      expect(target.classList.contains('chunk-highlight')).toBe(true);
      expect(scrollSpy).toHaveBeenCalled();
    });
  });

  it('should not highlight or scroll when chunkId does not match any element', async () => {
    const mockUrl = 'http://api.test/document';
    const mockHeaders = { Authorization: 'Bearer test-token' };

    vi.mocked(documentsV2Api.getContent).mockReturnValue({
      url: mockUrl,
      headers: mockHeaders,
    });

    const html = `
      <div>
        <p data-chunk-id="chunk-1">Some content</p>
      </div>
    `;

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      headers: new Headers({ 'content-type': 'text/html' }),
      text: () => Promise.resolve(html),
    });

    const scrollSpy = vi.fn();
    Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', {
      configurable: true,
      value: scrollSpy,
    });

    render(
      <DocumentViewer
        collectionId="c-1"
        docId="d-1"
        chunkId="non-existent-chunk"
        onClose={vi.fn()}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText('Some content')).toBeInTheDocument();
    });

    // Should not call scrollIntoView when chunk not found
    expect(scrollSpy).not.toHaveBeenCalled();
  });

  it('should not highlight chunks in PDF documents', async () => {
    const mockUrl = 'http://api.test/document';
    const mockHeaders = { Authorization: 'Bearer test-token' };

    vi.mocked(documentsV2Api.getContent).mockReturnValue({
      url: mockUrl,
      headers: mockHeaders,
    });

    const mockBlob = new Blob(['fake pdf data'], { type: 'application/pdf' });
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      headers: new Headers({ 'content-type': 'application/pdf' }),
      blob: () => Promise.resolve(mockBlob),
    });

    const scrollSpy = vi.fn();
    Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', {
      configurable: true,
      value: scrollSpy,
    });

    render(
      <DocumentViewer
        collectionId="c-1"
        docId="d-1"
        chunkId="chunk-1"
        onClose={vi.fn()}
      />,
    );

    await waitFor(() => {
      expect(mockPdfViewer).toHaveBeenCalled();
    });

    // PDF documents should not attempt chunk highlighting
    expect(scrollSpy).not.toHaveBeenCalled();
  });

  it('should cleanup highlight class on unmount', async () => {
    const mockUrl = 'http://api.test/document';
    const mockHeaders = { Authorization: 'Bearer test-token' };

    vi.mocked(documentsV2Api.getContent).mockReturnValue({
      url: mockUrl,
      headers: mockHeaders,
    });

    const html = `
      <div>
        <p data-chunk-id="chunk-cleanup">Content to cleanup</p>
      </div>
    `;

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      headers: new Headers({ 'content-type': 'text/html' }),
      text: () => Promise.resolve(html),
    });

    Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', {
      configurable: true,
      value: vi.fn(),
    });

    const { unmount } = render(
      <DocumentViewer
        collectionId="c-1"
        docId="d-1"
        chunkId="chunk-cleanup"
        onClose={vi.fn()}
      />,
    );

    let target: HTMLElement;
    await waitFor(() => {
      target = document.querySelector('[data-chunk-id="chunk-cleanup"]') as HTMLElement;
      expect(target).toBeInTheDocument();
      expect(target.classList.contains('chunk-highlight')).toBe(true);
    });

    unmount();

    // Verify cleanup removed the highlight class
    expect(target!.classList.contains('chunk-highlight')).toBe(false);
  });
});
