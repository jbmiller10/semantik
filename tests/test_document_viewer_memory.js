"""
Memory leak tests for DocumentViewer
These tests verify proper cleanup of resources
"""

describe('DocumentViewer Memory Management', () => {
    let documentViewer;
    let originalCreateElement;
    let createdElements = [];
    
    beforeEach(() => {
        // Track all created elements
        originalCreateElement = document.createElement;
        document.createElement = jest.fn((tag) => {
            const element = originalCreateElement.call(document, tag);
            createdElements.push(element);
            return element;
        });
        
        // Create viewer instance
        documentViewer = new DocumentViewer();
    });
    
    afterEach(() => {
        document.createElement = originalCreateElement;
        createdElements = [];
    });
    
    test('should clean up PDF.js resources on close', async () => {
        // Mock PDF document
        const mockPdfDoc = {
            cleanup: jest.fn().mockResolvedValue(),
            destroy: jest.fn().mockResolvedValue(),
            numPages: 10
        };
        
        // Simulate PDF loading
        documentViewer.pdfDocument = mockPdfDoc;
        
        // Close viewer
        await documentViewer.close();
        
        // Verify cleanup
        expect(mockPdfDoc.cleanup).toHaveBeenCalled();
        expect(mockPdfDoc.destroy).toHaveBeenCalled();
        expect(documentViewer.pdfDocument).toBeNull();
    });
    
    test('should remove all event listeners on close', () => {
        const contentElement = document.getElementById('viewer-content');
        
        // Add some test content with event listeners
        const testButton = document.createElement('button');
        const clickHandler = jest.fn();
        testButton.addEventListener('click', clickHandler);
        contentElement.appendChild(testButton);
        
        // Close viewer
        documentViewer.close();
        
        // Verify content is cleared
        expect(contentElement.innerHTML).toBe('');
        expect(contentElement.firstChild).toBeNull();
    });
    
    test('should clean up Mark.js instances', () => {
        // Create mock Mark instance
        const mockMarkInstance = {
            unmark: jest.fn(),
            mark: jest.fn()
        };
        
        documentViewer.markInstance = mockMarkInstance;
        
        // Close viewer
        documentViewer.close();
        
        // Verify cleanup
        expect(mockMarkInstance.unmark).toHaveBeenCalled();
        expect(documentViewer.markInstance).toBeNull();
    });
    
    test('should revoke blob URLs', () => {
        // Mock URL.revokeObjectURL
        const originalRevoke = URL.revokeObjectURL;
        URL.revokeObjectURL = jest.fn();
        
        // Set a blob URL
        documentViewer.currentBlobUrl = 'blob:http://localhost/test-blob';
        
        // Close viewer
        documentViewer.close();
        
        // Verify cleanup
        expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:http://localhost/test-blob');
        expect(documentViewer.currentBlobUrl).toBeNull();
        
        // Restore
        URL.revokeObjectURL = originalRevoke;
    });
    
    test('should reset all properties to initial state', () => {
        // Set various properties
        documentViewer.currentDocument = { jobId: 'test', docId: '123' };
        documentViewer.currentPage = 5;
        documentViewer.totalPages = 10;
        documentViewer.searchQuery = 'test search';
        documentViewer.highlights = ['h1', 'h2', 'h3'];
        documentViewer.currentHighlightIndex = 2;
        
        // Close viewer
        documentViewer.close();
        
        // Verify all properties are reset
        expect(documentViewer.currentDocument).toBeNull();
        expect(documentViewer.currentPage).toBe(1);
        expect(documentViewer.totalPages).toBe(1);
        expect(documentViewer.searchQuery).toBe('');
        expect(documentViewer.highlights).toEqual([]);
        expect(documentViewer.currentHighlightIndex).toBe(0);
    });
    
    test('should not leak memory with repeated open/close cycles', async () => {
        // Simulate multiple open/close cycles
        for (let i = 0; i < 10; i++) {
            // Open document
            await documentViewer.openDocument({
                jobId: 'test-job',
                docId: `doc-${i}`,
                filename: `test-${i}.pdf`,
                query: 'search term'
            });
            
            // Simulate some activity
            documentViewer.currentPage = 5;
            documentViewer.highlights = new Array(100).fill('highlight');
            
            // Close document
            documentViewer.close();
        }
        
        // Verify final state is clean
        expect(documentViewer.currentDocument).toBeNull();
        expect(documentViewer.highlights).toEqual([]);
        expect(documentViewer.pdfDocument).toBeNull();
    });
});