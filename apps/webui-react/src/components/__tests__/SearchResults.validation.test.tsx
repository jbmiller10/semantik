import React from 'react'
import { screen } from '@testing-library/react'
import { SearchResults } from '../SearchResults'
import { useSearchStore } from '../../stores/searchStore'
import { useUIStore } from '../../stores/uiStore'
import { 
  renderWithErrorHandlers
} from '../../tests/utils/errorTestUtils'

// Mock stores
vi.mock('../../stores/searchStore')
vi.mock('../../stores/collectionStore')
vi.mock('../../stores/uiStore')

describe('Search Results - Validation and Partial Failure Handling', () => {
  const mockAddToast = vi.fn()
  const mockSetShowDocumentViewer = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
    
    vi.mocked(useUIStore).mockReturnValue({
      addToast: mockAddToast,
      setShowDocumentViewer: mockSetShowDocumentViewer,
      showDocumentViewer: null
    } as any)
  })

  describe('Partial Failure Display', () => {
    it('should display partial failure warning prominently', async () => {
      const mockResults = [
        {
          collection_id: 'coll-1',
          collection_name: 'Working Collection',
          chunk_id: 1,
          content: 'Test result from working collection',
          score: 0.9,
          file_path: '/test/doc1.txt',
          file_name: 'doc1.txt'
        }
      ]
      
      const mockFailedCollections = [
        {
          collection_id: 'coll-2',
          collection_name: 'Failed Collection',
          error: 'Vector index corrupted'
        },
        {
          collection_id: 'coll-3',
          collection_name: 'Another Failed',
          error: 'Timeout during search'
        }
      ]

      vi.mocked(useSearchStore).mockReturnValue({
        results: mockResults,
        totalResults: 1,
        isSearching: false,
        error: null,
        searchTime: 2.5,
        partialFailure: true,
        failedCollections: mockFailedCollections,
        searchParams: { query: 'test query' }
      } as any)

      renderWithErrorHandlers(<SearchResults />, [])

      // Should show warning box at the top
      const warningBox = screen.getByRole('alert')
      expect(warningBox).toBeInTheDocument()
      expect(warningBox).toHaveClass('bg-yellow-50') // Yellow background for warning
      
      // Should list failed collections
      expect(screen.getByText('Failed Collection')).toBeInTheDocument()
      expect(screen.getByText('Vector index corrupted')).toBeInTheDocument()
      expect(screen.getByText('Another Failed')).toBeInTheDocument()
      expect(screen.getByText('Timeout during search')).toBeInTheDocument()
      
      // Should still show successful results
      expect(screen.getByText('Test result from working collection')).toBeInTheDocument()
    })

    it('should show different message when all collections fail', async () => {
      const mockFailedCollections = [
        {
          collection_id: 'coll-1',
          collection_name: 'Collection 1',
          error: 'Search service unavailable'
        },
        {
          collection_id: 'coll-2',
          collection_name: 'Collection 2',
          error: 'Search service unavailable'
        }
      ]

      vi.mocked(useSearchStore).mockReturnValue({
        results: [],
        totalResults: 0,
        isSearching: false,
        error: null,
        searchTime: 0.5,
        partialFailure: true,
        failedCollections: mockFailedCollections,
        searchParams: { query: 'test' }
      } as any)

      renderWithErrorHandlers(<SearchResults />, [])

      // Should show error state
      expect(screen.getByText(/no results found/i)).toBeInTheDocument()
      
      // Should show all failures
      const warningBox = screen.getByRole('alert')
      expect(warningBox).toHaveTextContent('All selected collections failed to return results')
    })

    it('should handle mixed error types in partial failures', async () => {
      const mockResults = [
        {
          collection_id: 'coll-1',
          collection_name: 'Working Collection',
          chunk_id: 1,
          content: 'Successful result',
          score: 0.85,
          file_path: '/docs/file.txt'
        }
      ]
      
      const mockFailedCollections = [
        {
          collection_id: 'coll-2',
          collection_name: 'Timeout Collection',
          error: 'Search timeout after 30 seconds'
        },
        {
          collection_id: 'coll-3',
          collection_name: 'Corrupted Collection',
          error: 'Index corruption detected'
        },
        {
          collection_id: 'coll-4',
          collection_name: 'Permission Collection',
          error: 'Access denied to collection'
        }
      ]

      vi.mocked(useSearchStore).mockReturnValue({
        results: mockResults,
        totalResults: 1,
        isSearching: false,
        error: null,
        searchTime: 30.5,
        partialFailure: true,
        failedCollections: mockFailedCollections,
        searchParams: { query: 'test' }
      } as any)

      renderWithErrorHandlers(<SearchResults />, [])

      // Should categorize or show all errors clearly
      expect(screen.getByText('Search timeout after 30 seconds')).toBeInTheDocument()
      expect(screen.getByText('Index corruption detected')).toBeInTheDocument()
      expect(screen.getByText('Access denied to collection')).toBeInTheDocument()
      
      // Should indicate that some results were returned despite failures
      expect(screen.getByText(/1 result/i)).toBeInTheDocument()
    })
  })

  describe('Search Query Validation', () => {
    it('should handle empty search query appropriately', async () => {
      vi.mocked(useSearchStore).mockReturnValue({
        results: [],
        totalResults: 0,
        isSearching: false,
        error: 'Search query cannot be empty',
        searchTime: 0,
        partialFailure: false,
        failedCollections: [],
        searchParams: { query: '' }
      } as any)

      renderWithErrorHandlers(<SearchResults />, [])

      // Should show appropriate message
      expect(screen.getByText(/search query cannot be empty/i)).toBeInTheDocument()
    })

    it('should handle query length validation', async () => {
      const veryLongQuery = 'a'.repeat(1000)
      
      vi.mocked(useSearchStore).mockReturnValue({
        results: [],
        totalResults: 0,
        isSearching: false,
        error: 'Search query too long (max 500 characters)',
        searchTime: 0,
        partialFailure: false,
        failedCollections: [],
        searchParams: { query: veryLongQuery }
      } as any)

      renderWithErrorHandlers(<SearchResults />, [])

      expect(screen.getByText(/search query too long/i)).toBeInTheDocument()
    })

    it('should handle special characters in search appropriately', async () => {
      // This tests how the UI handles searches with special regex characters
      const specialQuery = '[test] (query) {with} $pecial* chars?'
      
      vi.mocked(useSearchStore).mockReturnValue({
        results: [
          {
            collection_id: 'coll-1',
            collection_name: 'Test Collection',
            chunk_id: 1,
            content: 'Result with [test] in content',
            score: 0.9,
            file_path: '/test.txt'
          }
        ],
        totalResults: 1,
        isSearching: false,
        error: null,
        searchTime: 0.5,
        partialFailure: false,
        failedCollections: [],
        searchParams: { query: specialQuery }
      } as any)

      renderWithErrorHandlers(<SearchResults />, [])

      // Should display results normally
      expect(screen.getByText(/Result with \[test\] in content/i)).toBeInTheDocument()
    })
  })

  describe('Collection Selection Validation', () => {
    it('should show message when no collections are selected', async () => {
      vi.mocked(useSearchStore).mockReturnValue({
        results: [],
        totalResults: 0,
        isSearching: false,
        error: 'Please select at least one collection to search',
        searchTime: 0,
        partialFailure: false,
        failedCollections: [],
        searchParams: { 
          query: 'test',
          selectedCollections: []
        }
      } as any)

      renderWithErrorHandlers(<SearchResults />, [])

      expect(screen.getByText(/please select at least one collection/i)).toBeInTheDocument()
    })

    it('should handle invalid collection IDs gracefully', async () => {
      const mockFailedCollections = [
        {
          collection_id: 'invalid-uuid',
          collection_name: 'Unknown Collection',
          error: 'Collection not found'
        }
      ]

      vi.mocked(useSearchStore).mockReturnValue({
        results: [],
        totalResults: 0,
        isSearching: false,
        error: null,
        searchTime: 0.5,
        partialFailure: true,
        failedCollections: mockFailedCollections,
        searchParams: { 
          query: 'test',
          selectedCollections: ['invalid-uuid']
        }
      } as any)

      renderWithErrorHandlers(<SearchResults />, [])

      expect(screen.getByText('Collection not found')).toBeInTheDocument()
    })
  })

  describe('Result Validation and Display', () => {
    it('should handle malformed search results gracefully', async () => {
      // Results with missing required fields
      const malformedResults = [
        {
          collection_id: 'coll-1',
          collection_name: 'Test Collection',
          chunk_id: 1,
          content: null, // Missing content
          score: 0.9,
          file_path: '/test.txt'
        },
        {
          collection_id: 'coll-2',
          // Missing collection_name
          chunk_id: 2,
          content: 'Valid content',
          score: 0.8,
          file_path: '/test2.txt'
        }
      ]

      vi.mocked(useSearchStore).mockReturnValue({
        results: malformedResults as any,
        totalResults: 2,
        isSearching: false,
        error: null,
        searchTime: 1.0,
        partialFailure: false,
        failedCollections: [],
        searchParams: { query: 'test' }
      } as any)

      renderWithErrorHandlers(<SearchResults />, [])

      // Should handle missing content gracefully
      expect(screen.queryByText('null')).not.toBeInTheDocument()
      
      // Should show valid content
      expect(screen.getByText('Valid content')).toBeInTheDocument()
      
      // Should handle missing collection name
      expect(screen.getByText(/unknown/i)).toBeInTheDocument()
    })

    it('should display score validation warnings', async () => {
      const resultsWithInvalidScores = [
        {
          collection_id: 'coll-1',
          collection_name: 'Test Collection',
          chunk_id: 1,
          content: 'Result with invalid score',
          score: 1.5, // Score > 1.0
          file_path: '/test.txt'
        },
        {
          collection_id: 'coll-1',
          collection_name: 'Test Collection',
          chunk_id: 2,
          content: 'Result with negative score',
          score: -0.1, // Negative score
          file_path: '/test2.txt'
        }
      ]

      vi.mocked(useSearchStore).mockReturnValue({
        results: resultsWithInvalidScores as any,
        totalResults: 2,
        isSearching: false,
        error: null,
        searchTime: 1.0,
        partialFailure: false,
        failedCollections: [],
        searchParams: { query: 'test' }
      } as any)

      renderWithErrorHandlers(<SearchResults />, [])

      // Should still display results but maybe normalize scores
      expect(screen.getByText('Result with invalid score')).toBeInTheDocument()
      expect(screen.getByText('Result with negative score')).toBeInTheDocument()
    })

    it('should handle extremely large result sets', async () => {
      // Create a large number of results
      const largeResults = Array.from({ length: 1000 }, (_, i) => ({
        collection_id: 'coll-1',
        collection_name: 'Large Collection',
        chunk_id: i,
        content: `Result ${i}`,
        score: 0.9 - (i * 0.0001),
        file_path: `/doc${i}.txt`
      }))

      vi.mocked(useSearchStore).mockReturnValue({
        results: largeResults,
        totalResults: 1000,
        isSearching: false,
        error: null,
        searchTime: 5.5,
        partialFailure: false,
        failedCollections: [],
        searchParams: { query: 'test' }
      } as any)

      renderWithErrorHandlers(<SearchResults />, [])

      // Should show total count
      expect(screen.getByText(/1000 results/i)).toBeInTheDocument()
      
      // Should implement pagination or virtualization (implementation specific)
      // For now, just check that it doesn't crash
      expect(screen.getByText('Result 0')).toBeInTheDocument()
    })
  })

  describe('Cross-Model Search Validation', () => {
    it('should show reranking info when collections use different models', async () => {
      const mixedModelResults = [
        {
          collection_id: 'coll-1',
          collection_name: 'Collection A',
          chunk_id: 1,
          content: 'Result from model A',
          score: 0.9,
          original_score: 0.7,
          reranked_score: 0.9,
          embedding_model: 'model-a',
          file_path: '/test1.txt'
        },
        {
          collection_id: 'coll-2',
          collection_name: 'Collection B',
          chunk_id: 2,
          content: 'Result from model B',
          score: 0.85,
          original_score: 0.95,
          reranked_score: 0.85,
          embedding_model: 'model-b',
          file_path: '/test2.txt'
        }
      ]

      vi.mocked(useSearchStore).mockReturnValue({
        results: mixedModelResults as any,
        totalResults: 2,
        isSearching: false,
        error: null,
        searchTime: 3.0,
        partialFailure: false,
        failedCollections: [],
        searchParams: { query: 'test' }
      } as any)

      renderWithErrorHandlers(<SearchResults />, [])

      // Should indicate that reranking was applied
      expect(screen.getByText(/reranked/i)).toBeInTheDocument()
      
      // Should show both original and reranked scores
      expect(screen.getByText(/0.7/)).toBeInTheDocument() // Original score
      expect(screen.getByText(/0.9/)).toBeInTheDocument() // Reranked score
    })
  })
})