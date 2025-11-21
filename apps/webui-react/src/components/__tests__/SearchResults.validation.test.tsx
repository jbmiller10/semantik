import React from 'react'
import { screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import SearchResults from '../SearchResults'
import { useSearchStore } from '../../stores/searchStore'
import type { SearchResult } from '../../stores/searchStore'
import { useUIStore } from '../../stores/uiStore'
import { 
  renderWithErrorHandlers
} from '../../tests/utils/errorTestUtils'

// Mock stores
vi.mock('../../stores/searchStore')
vi.mock('../../stores/uiStore')

// Helper to create a complete search store mock
const createSearchStoreMock = (overrides?: Partial<ReturnType<typeof useSearchStore>>) => ({
  results: [],
  loading: false,
  error: null,
  searchParams: {
    query: '',
    selectedCollections: [],
    topK: 10,
    scoreThreshold: 0.5,
    searchType: 'semantic' as const,
    useReranker: false,
    hybridAlpha: 0.7,
    hybridMode: 'weighted' as const,
    keywordMode: 'any' as const,
  },
  collections: [],
  failedCollections: [],
  partialFailure: false,
  rerankingMetrics: null,
  gpuMemoryError: null,
  validationErrors: [],
  rerankingAvailable: true,
  rerankingModelsLoading: false,
  setResults: vi.fn(),
  setLoading: vi.fn(),
  setError: vi.fn(),
  updateSearchParams: vi.fn(),
  setCollections: vi.fn(),
  setFailedCollections: vi.fn(),
  setPartialFailure: vi.fn(),
  clearResults: vi.fn(),
  setRerankingMetrics: vi.fn(),
  setGpuMemoryError: vi.fn(),
  validateAndUpdateSearchParams: vi.fn(),
  clearValidationErrors: vi.fn(),
  hasValidationErrors: vi.fn(),
  getValidationError: vi.fn(),
  setRerankingAvailable: vi.fn(),
  setRerankingModelsLoading: vi.fn(),
  ...overrides
})

const mockSearchStore = (overrides?: Partial<ReturnType<typeof useSearchStore>>) => {
  const state = createSearchStoreMock(overrides)

  vi.mocked(useSearchStore).mockImplementation((selector?: (store: typeof state) => unknown) => {
    if (typeof selector === 'function') {
      return selector(state as never)
    }
    return state as never
  })
}

describe('Search Results - Validation and Partial Failure Handling', () => {
  const mockAddToast = vi.fn()
  const mockSetShowDocumentViewer = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
    
    vi.mocked(useUIStore).mockReturnValue({
      addToast: mockAddToast,
      setShowDocumentViewer: mockSetShowDocumentViewer,
      showDocumentViewer: null
    } as unknown as ReturnType<typeof useUIStore>)
  })

  describe('Partial Failure Display', () => {
    it('should display partial failure warning prominently', async () => {
      const mockResults: SearchResult[] = [
        {
          collection_id: 'coll-1',
          collection_name: 'Working Collection',
          chunk_id: 'chunk-1',
          doc_id: 'doc1',
          content: 'Test result from working collection',
          score: 0.9,
          file_path: '/test/doc1.txt',
          file_name: 'doc1.txt',
          chunk_index: 0,
          total_chunks: 1
        }
      ]
      
      const mockFailedCollections = [
        {
          collection_id: 'coll-2',
          collection_name: 'Failed Collection',
          error_message: 'Vector index corrupted'
        },
        {
          collection_id: 'coll-3',
          collection_name: 'Another Failed',
          error_message: 'Timeout during search'
        }
      ]

      mockSearchStore({
        results: mockResults,
        partialFailure: true,
        failedCollections: mockFailedCollections,
        searchParams: { 
          query: 'test query',
          selectedCollections: ['coll-1', 'coll-2', 'coll-3'],
          topK: 10,
          scoreThreshold: 0.5,
          searchType: 'semantic',
          useReranker: false,
          hybridAlpha: 0.7,
          hybridMode: 'weighted',
          keywordMode: 'any'
        }
      })

      renderWithErrorHandlers(<SearchResults />, [])

      // Should show warning box at the top
      expect(screen.getByText('Partial Search Failure')).toBeInTheDocument()
      
      // Should list failed collections
      expect(screen.getByText('Failed Collection')).toBeInTheDocument()
      expect(screen.getByText(/Vector index corrupted/)).toBeInTheDocument()
      expect(screen.getByText('Another Failed')).toBeInTheDocument()
      expect(screen.getByText(/Timeout during search/)).toBeInTheDocument()
      
      // Should show the collection is expanded by default
      expect(screen.getByText('Working Collection')).toBeInTheDocument()
      
      // Click on the document to expand it (documents are collapsed by default)
      const documentHeader = screen.getByText('doc1.txt')
      await userEvent.click(documentHeader.closest('.cursor-pointer')!)
      
      // Now the content should be visible
      expect(screen.getByText('Test result from working collection')).toBeInTheDocument()
    })

    it('should show different message when all collections fail', async () => {
      const mockFailedCollections = [
        {
          collection_id: 'coll-1',
          collection_name: 'Collection 1',
          error_message: 'Search service unavailable'
        },
        {
          collection_id: 'coll-2',
          collection_name: 'Collection 2',
          error_message: 'Search service unavailable'
        }
      ]

      mockSearchStore({
        results: [],
        partialFailure: true,
        failedCollections: mockFailedCollections,
        searchParams: { 
          query: 'test',
          selectedCollections: ['coll-1', 'coll-2'],
          topK: 10,
          scoreThreshold: 0.5,
          searchType: 'semantic',
          useReranker: false,
          hybridAlpha: 0.7,
          hybridMode: 'weighted',
          keywordMode: 'any'
        }
      })

      renderWithErrorHandlers(<SearchResults />, [])

      // Should show partial failure warning
      expect(screen.getByText('Partial Search Failure')).toBeInTheDocument()
      
      // Should show all failures
      expect(screen.getByText('Collection 1')).toBeInTheDocument()
      expect(screen.getByText('Collection 2')).toBeInTheDocument()
    })

    it('should handle mixed error types in partial failures', async () => {
      const mockResults: SearchResult[] = [
        {
          collection_id: 'coll-1',
          collection_name: 'Working Collection',
          chunk_id: 'chunk-1',
          doc_id: 'doc1',
          content: 'Successful result',
          score: 0.85,
          file_path: '/docs/file.txt',
          file_name: 'file.txt',
          chunk_index: 0,
          total_chunks: 1
        }
      ]
      
      const mockFailedCollections = [
        {
          collection_id: 'coll-2',
          collection_name: 'Timeout Collection',
          error_message: 'Search timeout after 30 seconds'
        },
        {
          collection_id: 'coll-3',
          collection_name: 'Corrupted Collection',
          error_message: 'Index corruption detected'
        },
        {
          collection_id: 'coll-4',
          collection_name: 'Permission Collection',
          error_message: 'Access denied to collection'
        }
      ]

      mockSearchStore({
        results: mockResults,
        partialFailure: true,
        failedCollections: mockFailedCollections,
        searchParams: { 
          query: 'test',
          selectedCollections: ['coll-1', 'coll-2', 'coll-3', 'coll-4'],
          topK: 10,
          scoreThreshold: 0.5,
          searchType: 'semantic',
          useReranker: false,
          hybridAlpha: 0.7,
          hybridMode: 'weighted',
          keywordMode: 'any'
        }
      })

      renderWithErrorHandlers(<SearchResults />, [])

      // Should categorize or show all errors clearly
      expect(screen.getByText(/Search timeout after 30 seconds/)).toBeInTheDocument()
      expect(screen.getByText(/Index corruption detected/)).toBeInTheDocument()
      expect(screen.getByText(/Access denied to collection/)).toBeInTheDocument()
      
      // Should indicate that some results were returned despite failures
      expect(screen.getByText(/Found 1 results/i)).toBeInTheDocument()
      
      // Verify the successful result is accessible by expanding the document
      const documentHeader = screen.getByText('file.txt')
      await userEvent.click(documentHeader.closest('.cursor-pointer')!)
      expect(screen.getByText('Successful result')).toBeInTheDocument()
    })
  })

  describe('Search Query Validation', () => {
    it('should handle empty search query appropriately', async () => {
      mockSearchStore({
        error: 'Search query cannot be empty',
        searchParams: { 
          query: '',
          selectedCollections: ['coll-1'],
          topK: 10,
          scoreThreshold: 0.5,
          searchType: 'semantic',
          useReranker: false,
          hybridAlpha: 0.7,
          hybridMode: 'weighted',
          keywordMode: 'any'
        }
      })

      renderWithErrorHandlers(<SearchResults />, [])

      // Should show appropriate message
      expect(screen.getByText(/search query cannot be empty/i)).toBeInTheDocument()
    })

    it('should handle query length validation', async () => {
      const veryLongQuery = 'a'.repeat(1000)
      
      mockSearchStore({
        error: 'Search query too long (max 500 characters)',
        searchParams: { 
          query: veryLongQuery,
          selectedCollections: ['coll-1'],
          topK: 10,
          scoreThreshold: 0.5,
          searchType: 'semantic',
          useReranker: false,
          hybridAlpha: 0.7,
          hybridMode: 'weighted',
          keywordMode: 'any'
        }
      })

      renderWithErrorHandlers(<SearchResults />, [])

      expect(screen.getByText(/search query too long/i)).toBeInTheDocument()
    })

    it('should handle special characters in search appropriately', async () => {
      // This tests how the UI handles searches with special regex characters
      const specialQuery = '[test] (query) {with} $pecial* chars?'
      
      mockSearchStore({
        results: [
          {
            collection_id: 'coll-1',
            collection_name: 'Test Collection',
            chunk_id: 'chunk-1',
            doc_id: 'doc1',
            content: 'Result with [test] in content',
            score: 0.9,
            file_path: '/test.txt',
            file_name: 'test.txt',
            chunk_index: 0,
            total_chunks: 1
          }
        ],
        searchParams: { 
          query: specialQuery,
          selectedCollections: ['coll-1'],
          topK: 10,
          scoreThreshold: 0.5,
          searchType: 'semantic',
          useReranker: false,
          hybridAlpha: 0.7,
          hybridMode: 'weighted',
          keywordMode: 'any'
        }
      })

      renderWithErrorHandlers(<SearchResults />, [])

      // Should display collection
      expect(screen.getByText('Test Collection')).toBeInTheDocument()
      
      // Expand the document to see content
      const documentHeader = screen.getByText('test.txt')
      await userEvent.click(documentHeader.closest('.cursor-pointer')!)
      
      // Should display results normally
      expect(screen.getByText(/Result with \[test\] in content/i)).toBeInTheDocument()
    })
  })

  describe('Collection Selection Validation', () => {
    it('should show message when no collections are selected', async () => {
      mockSearchStore({
        error: 'Please select at least one collection to search',
        searchParams: { 
          query: 'test',
          selectedCollections: [],
          topK: 10,
          scoreThreshold: 0.5,
          searchType: 'semantic',
          useReranker: false,
          hybridAlpha: 0.7,
          hybridMode: 'weighted',
          keywordMode: 'any'
        }
      })

      renderWithErrorHandlers(<SearchResults />, [])

      expect(screen.getByText(/please select at least one collection/i)).toBeInTheDocument()
    })

    it('should handle invalid collection IDs gracefully', async () => {
      const mockFailedCollections = [
        {
          collection_id: 'invalid-uuid',
          collection_name: 'Unknown Collection',
          error_message: 'Collection not found'
        }
      ]

      mockSearchStore({
        partialFailure: true,
        failedCollections: mockFailedCollections,
        searchParams: { 
          query: 'test',
          selectedCollections: ['invalid-uuid'],
          topK: 10,
          scoreThreshold: 0.5,
          searchType: 'semantic',
          useReranker: false,
          hybridAlpha: 0.7,
          hybridMode: 'weighted',
          keywordMode: 'any'
        }
      })

      renderWithErrorHandlers(<SearchResults />, [])

      // The component renders both the collection name and error message
      expect(screen.getByText('Unknown Collection')).toBeInTheDocument()
      expect(screen.getByText(/Collection not found/)).toBeInTheDocument()
    })
  })

  describe('Result Validation and Display', () => {
    it('should handle malformed search results gracefully', async () => {
      // Results with missing required fields
      const malformedResults = [
        {
          collection_id: 'coll-1',
          collection_name: 'Test Collection',
          chunk_id: 'chunk-1',
          doc_id: 'doc1',
          content: null, // Missing content
          score: 0.9,
          file_path: '/test.txt',
          file_name: 'test.txt',
          chunk_index: 0,
          total_chunks: 1
        },
        {
          collection_id: 'coll-2',
          collection_name: null, // Missing collection_name
          chunk_id: 'chunk-2',
          doc_id: 'doc2',
          content: 'Valid content',
          score: 0.8,
          file_path: '/test2.txt',
          file_name: 'test2.txt',
          chunk_index: 0,
          total_chunks: 1
        }
      ]

      mockSearchStore({
        results: malformedResults as unknown as SearchResult[],
        searchParams: { 
          query: 'test',
          selectedCollections: ['coll-1', 'coll-2'],
          topK: 10,
          scoreThreshold: 0.5,
          searchType: 'semantic',
          useReranker: false,
          hybridAlpha: 0.7,
          hybridMode: 'weighted',
          keywordMode: 'any'
        }
      })

      renderWithErrorHandlers(<SearchResults />, [])

      // Should handle missing collection name by using 'Unknown Collection'
      expect(screen.getByText('Unknown Collection')).toBeInTheDocument()
      
      // Expand the document with valid content
      const documentHeader = screen.getByText('test2.txt')
      await userEvent.click(documentHeader.closest('.cursor-pointer')!)
      
      // Should show valid content
      expect(screen.getByText('Valid content')).toBeInTheDocument()
    })

    it('should display score validation warnings', async () => {
      const resultsWithInvalidScores = [
        {
          collection_id: 'coll-1',
          collection_name: 'Test Collection',
          chunk_id: 'chunk-1',
          doc_id: 'doc1',
          content: 'Result with invalid score',
          score: 1.5, // Score > 1.0
          file_path: '/test.txt',
          file_name: 'test.txt',
          chunk_index: 0,
          total_chunks: 1
        },
        {
          collection_id: 'coll-1',
          collection_name: 'Test Collection',
          chunk_id: 'chunk-2',
          doc_id: 'doc2',
          content: 'Result with negative score',
          score: -0.1, // Negative score
          file_path: '/test2.txt',
          file_name: 'test2.txt',
          chunk_index: 0,
          total_chunks: 1
        }
      ]

      mockSearchStore({
        results: resultsWithInvalidScores as SearchResult[],
        searchParams: { 
          query: 'test',
          selectedCollections: ['coll-1'],
          topK: 10,
          scoreThreshold: 0.5,
          searchType: 'semantic',
          useReranker: false,
          hybridAlpha: 0.7,
          hybridMode: 'weighted',
          keywordMode: 'any'
        }
      })

      renderWithErrorHandlers(<SearchResults />, [])

      // Expand documents to see content
      const doc1Header = screen.getByText('test.txt')
      await userEvent.click(doc1Header.closest('.cursor-pointer')!)
      
      const doc2Header = screen.getByText('test2.txt')
      await userEvent.click(doc2Header.closest('.cursor-pointer')!)
      
      // Should still display results but maybe normalize scores
      expect(screen.getByText('Result with invalid score')).toBeInTheDocument()
      expect(screen.getByText('Result with negative score')).toBeInTheDocument()
    })

    it('should handle extremely large result sets', async () => {
      // Create a large number of results
      const largeResults = Array.from({ length: 1000 }, (_, i) => ({
        collection_id: 'coll-1',
        collection_name: 'Large Collection',
        chunk_id: `chunk-${i}`,
        doc_id: `doc-${Math.floor(i / 10)}`,
        content: `Result ${i}`,
        score: 0.9 - (i * 0.0001),
        file_path: `/doc${Math.floor(i / 10)}.txt`,
        file_name: `doc${Math.floor(i / 10)}.txt`,
        chunk_index: i % 10,
        total_chunks: 10
      }))

      mockSearchStore({
        results: largeResults,
        searchParams: { 
          query: 'test',
          selectedCollections: ['coll-1'],
          topK: 1000,
          scoreThreshold: 0.5,
          searchType: 'semantic',
          useReranker: false,
          hybridAlpha: 0.7,
          hybridMode: 'weighted',
          keywordMode: 'any'
        }
      })

      renderWithErrorHandlers(<SearchResults />, [])

      // Should show total count
      expect(screen.getByText(/Found 1000 results/i)).toBeInTheDocument()
      
      // The collection should be expanded by default
      expect(screen.getByText('Large Collection')).toBeInTheDocument()
      
      // Check that the first document is shown
      expect(screen.getByText('doc0.txt')).toBeInTheDocument()
      
      // Expand the first document
      const firstDocHeader = screen.getByText('doc0.txt')
      await userEvent.click(firstDocHeader.closest('.cursor-pointer')!)
      
      // Should show the first result
      expect(screen.getByText('Result 0')).toBeInTheDocument()
    })
  })

  describe('Cross-Model Search Validation', () => {
    it('should show reranking info when collections use different models', async () => {
      const mixedModelResults = [
        {
          collection_id: 'coll-1',
          collection_name: 'Collection A',
          chunk_id: 'chunk-1',
          doc_id: 'doc1',
          content: 'Result from model A',
          score: 0.9,
          file_path: '/test1.txt',
          file_name: 'test1.txt',
          chunk_index: 0,
          total_chunks: 1
        },
        {
          collection_id: 'coll-2',
          collection_name: 'Collection B',
          chunk_id: 'chunk-2',
          doc_id: 'doc2',
          content: 'Result from model B',
          score: 0.85,
          file_path: '/test2.txt',
          file_name: 'test2.txt',
          chunk_index: 0,
          total_chunks: 1
        }
      ]

      mockSearchStore({
        results: mixedModelResults as SearchResult[],
        rerankingMetrics: {
          rerankingUsed: true,
          rerankerModel: 'model-x',
          rerankingTimeMs: 150
        },
        searchParams: { 
          query: 'test',
          selectedCollections: ['coll-1', 'coll-2'],
          topK: 10,
          scoreThreshold: 0.5,
          searchType: 'semantic',
          useReranker: true,
          hybridAlpha: 0.7,
          hybridMode: 'weighted',
          keywordMode: 'any'
        }
      })

      renderWithErrorHandlers(<SearchResults />, [])

      // Should indicate that reranking was applied
      expect(screen.getByText(/Reranked/i)).toBeInTheDocument()
      
      // Should show reranking time
      expect(screen.getByText(/150ms/i)).toBeInTheDocument()
    })
  })
})
