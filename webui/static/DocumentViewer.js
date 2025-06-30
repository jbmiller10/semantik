/**
 * DocumentViewer.js - Multi-format document viewer with contextual highlighting
 * Supports PDF, DOCX, PPTX, TXT, MD, HTML, EML file formats
 */

class DocumentViewer {
    constructor() {
        this.currentDocument = null;
        this.currentPage = 1;
        this.totalPages = 1;
        this.searchQuery = '';
        this.highlights = [];
        this.currentHighlightIndex = 0;
        
        // Initialize the viewer modal
        this.initializeModal();
        
        // Load required libraries
        this.loadLibraries();
    }
    
    /**
     * Load required JavaScript libraries from CDN with local fallback
     */
    loadLibraries() {
        const libraries = [
            { name: 'pdfjsLib', check: () => window.pdfjsLib, cdn: 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js', local: '/static/libs/pdf.min.js', sri: 'sha256-W1eZ5vjGgGYyB6xbQu4U7tKkBvp69I9QwVTwwLFWaUY=' },
            { name: 'mammoth', check: () => window.mammoth, cdn: 'https://cdn.jsdelivr.net/npm/mammoth@1.6.0/mammoth.browser.min.js', local: '/static/libs/mammoth.browser.min.js', sri: 'sha384-nFoSjZIoH3CCp8W639jJyQkuPHinJ2NHe7on1xvlUA7SuGfJAfvMldrsoAVm6ECz' },
            { name: 'marked', check: () => window.marked, cdn: 'https://cdn.jsdelivr.net/npm/marked@9.1.6/marked.min.js', local: '/static/libs/marked.min.js', sri: 'sha384-odPBjvtXVM/5hOYIr3A1dB+flh0c3wAT3bSesIOqEGmyUA4JoKf/YTWy0XKOYAY7' },
            { name: 'DOMPurify', check: () => window.DOMPurify, cdn: 'https://cdn.jsdelivr.net/npm/dompurify@3.0.6/dist/purify.min.js', local: '/static/libs/purify.min.js', sri: 'sha384-cwS6YdhLI7XS60eoDiC+egV0qHp8zI+Cms46R0nbn8JrmoAzV9uFL60etMZhAnSu' },
            { name: 'Mark', check: () => window.Mark, cdn: 'https://cdn.jsdelivr.net/npm/mark.js@8.11.1/dist/mark.min.js', local: '/static/libs/mark.min.js', sri: 'sha384-t9DGTa+HJ3fETmsPZ37+56VRxK0NsOgFTQ1B4fxaxA+BM48rM5oXrg0/ncF/B3VX' },
            { name: 'emlFormat', check: () => window.emlFormat, cdn: 'https://cdn.jsdelivr.net/npm/eml-format@1.0.4/dist/eml-format.browser.min.js', local: '', sri: 'sha384-a81ubv0aZ39Ff3o+2Te5rLhImfB3b/3s6J2sK5xO3i2lJz1d2qD5fNmnJ2aB9gDE' },
        ];

        libraries.forEach(lib => {
            if (!lib.check()) {
                const script = document.createElement('script');
                script.src = lib.cdn;
                // Determine integrity type based on hash prefix
                if (lib.sri.startsWith('sha256-')) {
                    script.integrity = lib.sri;
                } else if (lib.sri.startsWith('sha384-')) {
                    script.integrity = lib.sri;
                } else if (lib.sri.startsWith('sha512-')) {
                    script.integrity = lib.sri;
                } else {
                    // Fallback if hash type is unknown or not provided
                    script.integrity = '';
                }
                script.crossOrigin = 'anonymous';
                script.onerror = () => {
                    if (lib.local) {
                        const fallbackScript = document.createElement('script');
                        fallbackScript.src = lib.local;
                        document.head.appendChild(fallbackScript);
                    }
                };
                document.head.appendChild(script);
            }
        });

        if (window.pdfjsLib) {
            window.pdfjsLib.GlobalWorkerOptions.workerSrc = 
                'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
        }
    }
    
    /**
     * Initialize the viewer modal
     */
    initializeModal() {
        // Check if modal already exists
        if (document.getElementById('document-viewer-modal')) {
            return;
        }
        
        const modalHtml = `
            <div id="document-viewer-modal" class="fixed inset-0 z-50 hidden">
                <!-- Backdrop -->
                <div class="fixed inset-0 bg-black bg-opacity-75" onclick="documentViewer.close()"></div>
                
                <!-- Modal Content -->
                <div class="fixed inset-4 bg-white rounded-lg shadow-2xl flex flex-col">
                    <!-- Header -->
                    <div class="flex items-center justify-between p-4 border-b">
                        <div class="flex items-center space-x-4">
                            <h2 id="viewer-title" class="text-xl font-semibold text-gray-800"></h2>
                            <span id="viewer-page-info" class="text-sm text-gray-600"></span>
                        </div>
                        <div class="flex items-center space-x-2">
                            <button onclick="documentViewer.previousHighlight()" 
                                    class="px-3 py-1 text-sm bg-yellow-100 hover:bg-yellow-200 rounded"
                                    title="Previous highlight">
                                <i class="fas fa-chevron-up"></i>
                            </button>
                            <span id="highlight-info" class="text-sm text-gray-600"></span>
                            <button onclick="documentViewer.nextHighlight()" 
                                    class="px-3 py-1 text-sm bg-yellow-100 hover:bg-yellow-200 rounded"
                                    title="Next highlight">
                                <i class="fas fa-chevron-down"></i>
                            </button>
                            <button onclick="documentViewer.downloadDocument()" 
                                    class="px-3 py-1 text-sm bg-blue-500 hover:bg-blue-600 text-white rounded"
                                    title="Download">
                                <i class="fas fa-download"></i>
                            </button>
                            <button onclick="documentViewer.close()" 
                                    class="px-3 py-1 text-sm bg-gray-500 hover:bg-gray-600 text-white rounded"
                                    title="Close">
                                <i class="fas fa-times"></i>
                            </button>
                        </div>
                    </div>
                    
                    <!-- Content Area -->
                    <div id="viewer-content" class="flex-1 overflow-auto p-4 bg-gray-50">
                        <!-- Content will be rendered here -->
                    </div>
                    
                    <!-- Footer with page navigation -->
                    <div id="viewer-footer" class="flex items-center justify-center p-4 border-t space-x-4">
                        <button onclick="documentViewer.previousPage()" 
                                id="prev-page-btn"
                                class="px-3 py-1 text-sm bg-gray-200 hover:bg-gray-300 rounded disabled:opacity-50"
                                disabled>
                            <i class="fas fa-chevron-left"></i> Previous
                        </button>
                        <input type="number" 
                               id="page-input" 
                               min="1" 
                               value="1"
                               class="w-16 px-2 py-1 text-center border rounded"
                               onchange="documentViewer.goToPage(this.value)">
                        <span class="text-sm text-gray-600">of <span id="total-pages">1</span></span>
                        <button onclick="documentViewer.nextPage()" 
                                id="next-page-btn"
                                class="px-3 py-1 text-sm bg-gray-200 hover:bg-gray-300 rounded disabled:opacity-50"
                                disabled>
                            Next <i class="fas fa-chevron-right"></i>
                        </button>
                    </div>
                </div>
            </div>
        `;
        
        document.body.insertAdjacentHTML('beforeend', modalHtml);
    }
    
    /**
     * Open the document viewer
     * @param {string} jobId - The job ID
     * @param {string} docId - The document ID
     * @param {string} filename - The filename for display
     * @param {Object} options - Additional options (searchQuery, pageNumber, etc.)
     */
    async open(jobId, docId, filename, options = {}) {
        this.currentDocument = { jobId, docId, filename };
        this.searchQuery = options.searchQuery || '';
        
        // Show modal
        document.getElementById('document-viewer-modal').classList.remove('hidden');
        document.getElementById('viewer-title').textContent = filename;
        
        // Show loading state
        this.showLoading();
        
        try {
            // Get document info first
            const infoResponse = await fetch(`/api/documents/${jobId}/${docId}/info`, {
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('access_token')}`
                }
            });
            
            if (!infoResponse.ok) {
                throw new Error('Failed to get document info');
            }
            
            const info = await infoResponse.json();
            
            // Render based on file type
            const extension = info.extension.toLowerCase();
            
            if (!info.supported) {
                this.showUnsupportedMessage(extension);
                return;
            }
            
            // Load and render the document
            await this.renderDocument(extension, options.pageNumber);
            
            // Apply highlights if search query exists
            if (this.searchQuery) {
                this.applyHighlights();
            }
            
            // Navigate to specific page if provided
            if (options.pageNumber && this.totalPages > 1) {
                this.goToPage(options.pageNumber);
            }
            
        } catch (error) {
            console.error('Error opening document:', error);
            this.showError(error.message);
        }
    }
    
    /**
     * Render the document based on its type
     */
    async renderDocument(extension, targetPage = 1) {
        const contentDiv = document.getElementById('viewer-content');
        
        switch (extension) {
            case '.pdf':
                await this.renderPDF(targetPage);
                break;
            case '.docx':
                await this.renderDOCX();
                break;
            case '.txt':
            case '.text':
                await this.renderText();
                break;
            case '.md':
                await this.renderMarkdown();
                break;
            case '.html':
                await this.renderHTML();
                break;
            case '.pptx':
                await this.renderPPTX();
                break;
            case '.eml':
                await this.renderEML();
                break;
            case '.doc':
                this.showLegacyWordMessage();
                break;
            default:
                this.showUnsupportedMessage(extension);
        }
    }
    
    /**
     * Render PDF using PDF.js
     */
    async renderPDF(targetPage = 1) {
        // Wait for PDF.js to load
        while (!window.pdfjsLib) {
            await new Promise(resolve => setTimeout(resolve, 100));
        }
        
        const { jobId, docId } = this.currentDocument;
        const url = `/api/documents/${jobId}/${docId}`;
        
        // Load the PDF
        const loadingTask = pdfjsLib.getDocument({
            url: url,
            httpHeaders: {
                'Authorization': `Bearer ${localStorage.getItem('access_token')}`
            }
        });
        
        const pdf = await loadingTask.promise;
        this.totalPages = pdf.numPages;
        this.currentPage = Math.min(targetPage, this.totalPages);
        
        // Update page controls
        this.updatePageControls();
        
        // Render the current page
        await this.renderPDFPage(pdf, this.currentPage);
        
        // Store PDF object for navigation
        this.pdfDocument = pdf;
    }
    
    /**
     * Render a specific PDF page
     */
    async renderPDFPage(pdf, pageNumber) {
        const page = await pdf.getPage(pageNumber);
        const viewport = page.getViewport({ scale: 1.5 });
        
        const container = document.getElementById('viewer-content');
        container.innerHTML = '';
        
        // Create canvas
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        canvas.height = viewport.height;
        canvas.width = viewport.width;
        canvas.className = 'mx-auto shadow-lg';
        
        // Render PDF page
        await page.render({
            canvasContext: context,
            viewport: viewport
        }).promise;
        
        container.appendChild(canvas);
        
        // Create text layer for selection and highlighting
        const textLayerDiv = document.createElement('div');
        textLayerDiv.className = 'textLayer';
        textLayerDiv.style.position = 'absolute';
        textLayerDiv.style.left = '50%';
        textLayerDiv.style.transform = 'translateX(-50%)';
        textLayerDiv.style.width = `${viewport.width}px`;
        textLayerDiv.style.height = `${viewport.height}px`;
        container.appendChild(textLayerDiv);
        
        // Get text content
        const textContent = await page.getTextContent();
        
        // Render text layer
        pdfjsLib.renderTextLayer({
            textContent: textContent,
            container: textLayerDiv,
            viewport: viewport,
            textDivs: []
        });
        
        // Apply highlights after text layer is rendered
        setTimeout(() => this.applyHighlights(), 100);
    }
    
    /**
     * Render DOCX using Mammoth.js
     */
    async renderDOCX() {
        // Wait for Mammoth to load
        while (!window.mammoth) {
            await new Promise(resolve => setTimeout(resolve, 100));
        }
        
        const { jobId, docId } = this.currentDocument;
        const response = await fetch(`/api/documents/${jobId}/${docId}`, {
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('access_token')}`
            }
        });
        
        const arrayBuffer = await response.arrayBuffer();
        const result = await mammoth.convertToHtml({ arrayBuffer: arrayBuffer });
        
        const container = document.getElementById('viewer-content');
        container.innerHTML = `
            <div class="max-w-4xl mx-auto bg-white p-8 shadow-lg">
                ${DOMPurify.sanitize(result.value)}
            </div>
        `;
        
        // Hide page navigation for single-page documents
        document.getElementById('viewer-footer').style.display = 'none';
        
        // Apply highlights
        this.applyHighlights();
    }
    
    /**
     * Render plain text files
     */
    async renderText() {
        const { jobId, docId } = this.currentDocument;
        const response = await fetch(`/api/documents/${jobId}/${docId}`, {
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('access_token')}`
            }
        });
        
        const text = await response.text();
        
        const container = document.getElementById('viewer-content');
        container.innerHTML = `
            <div class="max-w-4xl mx-auto bg-white p-8 shadow-lg">
                <pre class="whitespace-pre-wrap font-mono text-sm">${this.escapeHtml(text)}</pre>
            </div>
        `;
        
        // Hide page navigation
        document.getElementById('viewer-footer').style.display = 'none';
        
        // Apply highlights
        this.applyHighlights();
    }
    
    /**
     * Render Markdown files
     */
    async renderMarkdown() {
        // Wait for Marked to load
        while (!window.marked) {
            await new Promise(resolve => setTimeout(resolve, 100));
        }
        
        const { jobId, docId } = this.currentDocument;
        const response = await fetch(`/api/documents/${jobId}/${docId}`, {
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('access_token')}`
            }
        });
        
        const markdown = await response.text();
        const html = marked.parse(markdown);
        
        const container = document.getElementById('viewer-content');
        container.innerHTML = `
            <div class="max-w-4xl mx-auto bg-white p-8 shadow-lg prose prose-sm max-w-none">
                ${DOMPurify.sanitize(html)}
            </div>
        `;
        
        // Hide page navigation
        document.getElementById('viewer-footer').style.display = 'none';
        
        // Apply highlights
        this.applyHighlights();
    }
    
    /**
     * Render HTML files
     */
    async renderHTML() {
        const { jobId, docId } = this.currentDocument;
        const response = await fetch(`/api/documents/${jobId}/${docId}`, {
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('access_token')}`
            }
        });
        
        const html = await response.text();
        
        const container = document.getElementById('viewer-content');
        container.innerHTML = `
            <div class="max-w-4xl mx-auto bg-white p-8 shadow-lg">
                ${DOMPurify.sanitize(html)}
            </div>
        `;
        
        // Hide page navigation
        document.getElementById('viewer-footer').style.display = 'none';
        
        // Apply highlights
        this.applyHighlights();
    }
    
    /**
     * Render PowerPoint presentations by converting to Markdown server-side
     */
    async renderPPTX() {
        // Wait for Marked to load (we'll use it to render the converted markdown)
        while (!window.marked) {
            await new Promise(resolve => setTimeout(resolve, 100));
        }
        
        const { jobId, docId, filename } = this.currentDocument;
        const container = document.getElementById('viewer-content');
        
        // Show loading state
        container.innerHTML = `
            <div class="text-center p-8">
                <i class="fas fa-spinner fa-spin text-4xl text-blue-500 mb-4"></i>
                <p class="text-gray-600">Converting presentation...</p>
            </div>
        `;
        
        try {
            const response = await fetch(`/api/documents/${jobId}/${docId}`, {
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('access_token')}`
                }
            });
            
            if (!response.ok) {
                throw new Error('Failed to load presentation');
            }
            
            // Check if the server converted it to markdown
            const contentType = response.headers.get('content-type');
            const isConverted = response.headers.get('X-Converted-From') === 'pptx';
            const imageSessionId = response.headers.get('X-Image-Session-Id');
            
            
            if (contentType && contentType.includes('text/markdown') && isConverted) {
                // Server converted to markdown, render it
                const markdown = await response.text();
                const html = marked.parse(markdown);
                
                container.innerHTML = `
                    <div class="max-w-4xl mx-auto bg-white p-8 shadow-lg">
                        <div class="bg-orange-50 border-l-4 border-orange-400 p-4 mb-6">
                            <p class="text-sm text-orange-700">
                                <i class="fas fa-info-circle mr-2"></i>
                                This is a converted preview of the PowerPoint presentation. 
                                <button onclick="documentViewer.downloadDocument()" 
                                        class="ml-2 underline hover:text-orange-900">
                                    Download original
                                </button>
                            </p>
                        </div>
                        <div class="prose prose-sm max-w-none">
                            ${DOMPurify.sanitize(html)}
                        </div>
                    </div>
                `;
                
                // Apply highlights
                this.applyHighlights();
            } else {
                // Fallback: Show download option if conversion failed
                container.innerHTML = `
                    <div class="max-w-4xl mx-auto bg-white p-8 shadow-lg">
                        <div class="text-center py-8">
                            <i class="fas fa-file-powerpoint text-6xl text-orange-500 mb-4"></i>
                            <h3 class="text-lg font-semibold mb-2">PowerPoint Presentation</h3>
                            <p class="text-gray-600 mb-4">
                                Preview is not available. Please download the file to view it.
                            </p>
                            <button onclick="documentViewer.downloadDocument()" 
                                    class="px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded">
                                <i class="fas fa-download mr-2"></i>Download File
                            </button>
                        </div>
                    </div>
                `;
            }
        } catch (error) {
            console.error('Error loading PPTX:', error);
            container.innerHTML = `
                <div class="max-w-4xl mx-auto bg-white p-8 shadow-lg">
                    <div class="text-center py-8">
                        <i class="fas fa-exclamation-triangle text-4xl text-red-500 mb-4"></i>
                        <h3 class="text-lg font-semibold mb-2">Error Loading Presentation</h3>
                        <p class="text-gray-600 mb-4">${this.escapeHtml(error.message)}</p>
                        <button onclick="documentViewer.downloadDocument()" 
                                class="px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded">
                            <i class="fas fa-download mr-2"></i>Download File
                        </button>
                    </div>
                </div>
            `;
        }
        
        // Hide page navigation
        document.getElementById('viewer-footer').style.display = 'none';
    }

    /**
     * Render email files
     */
    async renderEML() {
        const { jobId, docId } = this.currentDocument;
        const response = await fetch(`/api/documents/${jobId}/${docId}`, {
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('access_token')}`
            }
        });
        const emlText = await response.text();

        const container = document.getElementById('viewer-content');
        const email = emlFormat.parse(emlText);

        container.innerHTML = `
            <div class="max-w-4xl mx-auto bg-white p-8 shadow-lg">
                <div class="mb-4 pb-4 border-b">
                    <h3 class="text-xl font-semibold mb-2">${email.subject || 'No Subject'}</h3>
                    <p><strong>From:</strong> ${this.escapeHtml(email.from.text)}</p>
                    <p><strong>To:</strong> ${this.escapeHtml(email.to.text)}</p>
                    <p><strong>Date:</strong> ${email.date}</p>
                </div>
                <div class="prose max-w-none">${DOMPurify.sanitize(email.html || `<pre>${this.escapeHtml(email.text)}</pre>`)}</div>
            </div>
        `;

        document.getElementById('viewer-footer').style.display = 'none';
        this.applyHighlights();
    }
    
    /**
     * Show message for legacy Word documents
     */
    showLegacyWordMessage() {
        const container = document.getElementById('viewer-content');
        container.innerHTML = `
            <div class="max-w-4xl mx-auto bg-white p-8 shadow-lg">
                <div class="text-center py-8">
                    <i class="fas fa-file-word text-6xl text-blue-500 mb-4"></i>
                    <h3 class="text-lg font-semibold mb-2">Legacy Word Document</h3>
                    <p class="text-gray-600 mb-4">
                        Live preview is not supported for .doc files.
                        Please download the file to view it in Microsoft Word.
                    </p>
                    <button onclick="documentViewer.downloadDocument()" 
                            class="px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded">
                        <i class="fas fa-download mr-2"></i>Download File
                    </button>
                </div>
            </div>
        `;
        
        // Hide page navigation
        document.getElementById('viewer-footer').style.display = 'none';
    }
    
    /**
     * Show unsupported file message
     */
    showUnsupportedMessage(extension) {
        const container = document.getElementById('viewer-content');
        container.innerHTML = `
            <div class="max-w-4xl mx-auto bg-white p-8 shadow-lg">
                <div class="text-center py-8">
                    <i class="fas fa-file text-6xl text-gray-400 mb-4"></i>
                    <h3 class="text-lg font-semibold mb-2">Unsupported File Type</h3>
                    <p class="text-gray-600 mb-4">
                        Preview is not available for ${extension} files.
                    </p>
                    <button onclick="documentViewer.downloadDocument()" 
                            class="px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded">
                        <i class="fas fa-download mr-2"></i>Download File
                    </button>
                </div>
            </div>
        `;
        
        // Hide page navigation
        document.getElementById('viewer-footer').style.display = 'none';
    }
    
    /**
     * Apply search highlights to the document
     */
    applyHighlights() {
        if (!this.searchQuery || !window.Mark) return;
        
        const container = document.getElementById('viewer-content');
        const marker = new Mark(container);
        
        // Clear previous highlights
        marker.unmark();
        
        // Split search query into words (safer regex)
        const words = this.searchQuery.split(/[\s,.-;:]+/).filter(word => word.length > 0);
        
        // Apply highlights
        words.forEach(word => {
            marker.mark(word, {
                className: 'bg-yellow-300',
                caseSensitive: false,
                separateWordSearch: false,
                done: () => {
                    // Update highlight count
                    this.highlights = container.querySelectorAll('mark');
                    this.updateHighlightInfo();
                }
            });
        });
    }
    
    /**
     * Navigate to next highlight
     */
    nextHighlight() {
        if (this.highlights.length === 0) return;
        
        this.currentHighlightIndex = (this.currentHighlightIndex + 1) % this.highlights.length;
        this.scrollToHighlight();
    }
    
    /**
     * Navigate to previous highlight
     */
    previousHighlight() {
        if (this.highlights.length === 0) return;
        
        this.currentHighlightIndex = this.currentHighlightIndex - 1;
        if (this.currentHighlightIndex < 0) {
            this.currentHighlightIndex = this.highlights.length - 1;
        }
        this.scrollToHighlight();
    }
    
    /**
     * Scroll to current highlight
     */
    scrollToHighlight() {
        const highlight = this.highlights[this.currentHighlightIndex];
        if (highlight) {
            highlight.scrollIntoView({ behavior: 'smooth', block: 'center' });
            
            // Flash the highlight
            highlight.style.backgroundColor = '#FDE047';
            setTimeout(() => {
                highlight.style.backgroundColor = '';
            }, 500);
        }
        
        this.updateHighlightInfo();
    }
    
    /**
     * Update highlight navigation info
     */
    updateHighlightInfo() {
        const infoElement = document.getElementById('highlight-info');
        if (this.highlights.length > 0) {
            infoElement.textContent = `${this.currentHighlightIndex + 1} / ${this.highlights.length}`;
        } else {
            infoElement.textContent = '';
        }
    }
    
    /**
     * Navigate to next page
     */
    async nextPage() {
        if (this.currentPage < this.totalPages) {
            this.currentPage++;
            await this.renderPDFPage(this.pdfDocument, this.currentPage);
            this.updatePageControls();
        }
    }
    
    /**
     * Navigate to previous page
     */
    async previousPage() {
        if (this.currentPage > 1) {
            this.currentPage--;
            await this.renderPDFPage(this.pdfDocument, this.currentPage);
            this.updatePageControls();
        }
    }
    
    /**
     * Go to specific page
     */
    async goToPage(pageNumber) {
        pageNumber = parseInt(pageNumber);
        if (pageNumber > 0 && pageNumber <= this.totalPages) {
            this.currentPage = pageNumber;
            await this.renderPDFPage(this.pdfDocument, this.currentPage);
            this.updatePageControls();
        }
    }
    
    /**
     * Update page navigation controls
     */
    updatePageControls() {
        document.getElementById('page-input').value = this.currentPage;
        document.getElementById('page-input').max = this.totalPages;
        document.getElementById('total-pages').textContent = this.totalPages;
        document.getElementById('viewer-page-info').textContent = `Page ${this.currentPage} of ${this.totalPages}`;
        
        // Enable/disable navigation buttons
        document.getElementById('prev-page-btn').disabled = (this.currentPage === 1);
        document.getElementById('next-page-btn').disabled = (this.currentPage === this.totalPages);
        
        // Show/hide footer based on page count
        document.getElementById('viewer-footer').style.display = 
            this.totalPages > 1 ? 'flex' : 'none';
    }
    
    /**
     * Download the current document
     */
    downloadDocument() {
        const { jobId, docId, filename } = this.currentDocument;
        const url = `/api/documents/${jobId}/${docId}`;
        
        // Create a temporary anchor element
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.target = '_blank';
        
        // Add auth header using fetch
        fetch(url, {
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('access_token')}`
            }
        })
        .then(response => response.blob())
        .then(blob => {
            const url = window.URL.createObjectURL(blob);
            a.href = url;
            a.click();
            window.URL.revokeObjectURL(url);
        });
    }
    
    /**
     * Show loading state
     */
    showLoading() {
        const container = document.getElementById('viewer-content');
        container.innerHTML = `
            <div class="flex items-center justify-center h-full">
                <div class="text-center">
                    <i class="fas fa-spinner fa-spin text-4xl text-blue-500 mb-4"></i>
                    <p class="text-gray-600">Loading document...</p>
                </div>
            </div>
        `;
    }
    
    /**
     * Show error message
     */
    showError(message) {
        const container = document.getElementById('viewer-content');
        container.innerHTML = `
            <div class="flex items-center justify-center h-full">
                <div class="text-center">
                    <i class="fas fa-exclamation-triangle text-4xl text-red-500 mb-4"></i>
                    <p class="text-gray-800 font-semibold mb-2">Error Loading Document</p>
                    <p class="text-gray-600">${this.escapeHtml(message)}</p>
                </div>
            </div>
        `;
    }
    
    /**
     * Close the viewer
     */
    close() {
        // Hide the modal
        document.getElementById('document-viewer-modal').classList.add('hidden');
        
        // Clean up PDF resources
        if (this.pdfDocument) {
            this.pdfDocument.cleanup();
            this.pdfDocument.destroy();
            this.pdfDocument = null;
        }
        
        
        // Clear content from DOM
        const contentElement = document.getElementById('viewer-content');
        if (contentElement) {
            contentElement.innerHTML = '';
        }
        
        // Reset all properties
        this.currentDocument = null;
        this.currentPage = 1;
        this.totalPages = 1;
        this.searchQuery = '';
        this.highlights = [];
        this.currentHighlightIndex = 0;
        
        // Force garbage collection hint (browser may ignore)
        if (window.gc) {
            window.gc();
        }
    }
    
    /**
     * Escape HTML for safe display
     */
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Create global instance
window.documentViewer = new DocumentViewer();