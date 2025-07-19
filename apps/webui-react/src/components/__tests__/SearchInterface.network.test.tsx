import React from 'react'
import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { SearchInterface } from '../SearchInterface'
import { useSearchStore } from '../../stores/searchStore'
import { useCollectionStore } from '../../stores/collectionStore'
import { useUIStore } from '../../stores/uiStore'
import { 
  renderWithErrorHandlers, 
  waitForError,
  waitForToast,
  simulateOffline,
  simulateOnline
} from '../../tests/utils/errorTestUtils'
import { searchErrorHandlers } from '../../tests/mocks/errorHandlers'
import { server } from '../../tests/mocks/server'
import { handlers } from '../../tests/mocks/handlers'

// Mock stores
vi.mock('../../stores/searchStore')
vi.mock('../../stores/collectionStore')
vi.mock('../../stores/uiStore')

describe('SearchInterface - Network Error Handling', () => {
  const mockSearch = vi.fn()
  const mockSetError = vi.fn()
  const mockAddToast = vi.fn()
  const mockCollections = [
    {
      uuid: 'coll-1',
      name: 'Test Collection 1',
      status: 'ready',
      document_count: 100,
      vector_count: 1000,
      embedding_model: 'test-model',
      quantization: 'float16'
    },
    {
      uuid: 'coll-2', 
      name: 'Test Collection 2',
      status: 'ready',
      document_count: 200,
      vector_count: 2000,
      embedding_model: 'test-model'
    }
  ]

  beforeEach(() => {
    vi.clearAllMocks()
    
    vi.mocked(useSearchStore).mockReturnValue({
      searchParams: {
        query: '',
        selectedCollections: [],
        limit: 10,
        searchType: 'hybrid',
        includeContent: true,
        weightKeyword: 0.5,
        weightSemantic: 0.5,
        rerankModel: null
      },
      isSearching: false,
      error: null,
      setSearchParams: vi.fn(),
      performSearch: mockSearch,
      setError: mockSetError,
      setResults: vi.fn(),
      clearResults: vi.fn(),
      results: [],
      totalResults: 0,
      searchTime: 0,
      failedCollections: [],
      partialFailure: false
    } as any)
    
    vi.mocked(useCollectionStore).mockReturnValue({
      collections: mockCollections,
      fetchCollections: vi.fn(),
      loading: false,
      error: null
    } as any)
    
    vi.mocked(useUIStore).mockReturnValue({
      addToast: mockAddToast
    } as any)
  })

  describe('Search Request Network Failures', () => {
    it('should show error when search fails due to network error', async () => {
      renderWithErrorHandlers(
        <SearchInterface />,
        searchErrorHandlers.networkError()
      )

      // Select collections
      const multiSelect = screen.getByText(/select collections/i)
      await userEvent.click(multiSelect)
      
      const collection1 = await screen.findByText('Test Collection 1')
      await userEvent.click(collection1)
      
      // Enter search query
      const searchInput = screen.getByPlaceholderText(/search across your collections/i)
      await userEvent.type(searchInput, 'test query')
      
      // Mock search to reject
      mockSearch.mockRejectedValue(new Error('Network error'))
      
      // Submit search
      const searchButton = screen.getByRole('button', { name: /search/i })
      await userEvent.click(searchButton)
      
      // Should show error
      await waitFor(() => {
        expect(mockSetError).toHaveBeenCalledWith('Network error')
      })
    })

    it('should preserve search parameters after network error', async () => {
      renderWithErrorHandlers(
        <SearchInterface />,
        searchErrorHandlers.networkError()
      )

      // Set up search parameters
      const searchInput = screen.getByPlaceholderText(/search across your collections/i)
      await userEvent.type(searchInput, 'important documents')
      
      // Select collections
      const multiSelect = screen.getByText(/select collections/i)
      await userEvent.click(multiSelect)
      await userEvent.click(await screen.findByText('Test Collection 1'))
      await userEvent.click(await screen.findByText('Test Collection 2'))
      
      mockSearch.mockRejectedValue(new Error('Network error'))
      
      // Search
      await userEvent.click(screen.getByRole('button', { name: /search/i }))
      
      await waitFor(() => {
        expect(mockSetError).toHaveBeenCalled()
      })
      
      // All parameters should be preserved
      expect(searchInput).toHaveValue('important documents')
      // Collections should still be selected (would need to check the component state)
    })

    it('should handle offline to online transition', async () => {
      renderWithErrorHandlers(
        <SearchInterface />,
        []
      )

      const searchInput = screen.getByPlaceholderText(/search across your collections/i)
      await userEvent.type(searchInput, 'test')
      
      // Go offline
      simulateOffline()
      mockSearch.mockRejectedValue(new Error('Network error'))
      
      await userEvent.click(screen.getByRole('button', { name: /search/i }))
      
      await waitFor(() => {
        expect(mockSetError).toHaveBeenCalledWith('Network error')
      })
      
      // Go back online
      simulateOnline()
      mockSearch.mockResolvedValue(undefined)
      mockSetError.mockClear()
      
      // Try again
      await userEvent.click(screen.getByRole('button', { name: /search/i }))
      
      // Should work now
      await waitFor(() => {
        expect(mockSearch).toHaveBeenCalledWith(expect.objectContaining({
          query: 'test'
        }))
      })
      expect(mockSetError).not.toHaveBeenCalled()
    })

    it('should disable search button during network request', async () => {
      // Mock slow request
      mockSearch.mockImplementation(() => 
        new Promise(resolve => setTimeout(resolve, 1000))
      )
      
      renderWithErrorHandlers(
        <SearchInterface />,
        []
      )

      await userEvent.type(screen.getByPlaceholderText(/search/i), 'test')
      
      const searchButton = screen.getByRole('button', { name: /search/i })
      
      // Update the mock to show searching state
      vi.mocked(useSearchStore).mockReturnValue({
        ...vi.mocked(useSearchStore).mock.results[0].value,
        isSearching: true
      } as any)
      
      await userEvent.click(searchButton)
      
      // Rerender to see the updated state
      renderWithErrorHandlers(
        <SearchInterface />,
        []
      )
      
      // Button should show searching state
      expect(screen.getByRole('button', { name: /searching/i })).toBeDisabled()
    })
  })

  describe('Partial Failure Handling', () => {
    it('should display results and warnings for partial failures', async () => {
      renderWithErrorHandlers(
        <SearchInterface />,
        searchErrorHandlers.partialFailure()
      )

      // Select all collections
      const multiSelect = screen.getByText(/select collections/i)
      await userEvent.click(multiSelect)
      await userEvent.click(screen.getByText(/select all/i))
      
      await userEvent.type(screen.getByPlaceholderText(/search/i), 'test')
      
      // Mock the search to return partial results
      mockSearch.mockImplementation(async () => {
        // Update the store to reflect partial failure
        vi.mocked(useSearchStore).mockReturnValue({
          ...vi.mocked(useSearchStore).mock.results[0].value,
          partialFailure: true,
          failedCollections: [
            {
              collection_id: 'coll-2',
              collection_name: 'Failed Collection',
              error: 'Vector index corrupted'
            }
          ],
          results: [
            {
              collection_id: 'coll-1',
              collection_name: 'Working Collection',
              chunk_id: 1,
              content: 'Test result',
              score: 0.9
            }
          ],
          totalResults: 1
        } as any)
      })
      
      await userEvent.click(screen.getByRole('button', { name: /search/i }))
      
      // Should show warning toast about partial failure
      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith(
          expect.stringContaining('Search completed with errors'),
          'warning'
        )
      })
    })

    it('should handle all collections failing gracefully', async () => {
      renderWithErrorHandlers(
        <SearchInterface />,
        []
      )

      // Mock all collections failing
      mockSearch.mockImplementation(async () => {
        vi.mocked(useSearchStore).mockReturnValue({
          ...vi.mocked(useSearchStore).mock.results[0].value,
          partialFailure: true,
          failedCollections: mockCollections.map(c => ({
            collection_id: c.uuid,
            collection_name: c.name,
            error: 'Search timeout'
          })),
          results: [],
          totalResults: 0
        } as any)
      })
      
      await userEvent.type(screen.getByPlaceholderText(/search/i), 'test')
      await userEvent.click(screen.getByRole('button', { name: /search/i }))
      
      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith(
          expect.stringContaining('All collections failed'),
          'error'
        )
      })
    })
  })

  describe('Timeout Handling', () => {
    it('should handle search timeout appropriately', async () => {
      renderWithErrorHandlers(
        <SearchInterface />,
        []
      )

      // Mock timeout
      mockSearch.mockRejectedValue(new Error('Request timeout after 30s'))
      
      await userEvent.type(screen.getByPlaceholderText(/search/i), 'complex query')
      await userEvent.click(screen.getByRole('button', { name: /search/i }))
      
      await waitFor(() => {
        expect(mockSetError).toHaveBeenCalledWith('Request timeout after 30s')
      })
      
      // Should suggest trying with fewer collections or simpler query
      // (This would be in the error message displayed to the user)
    })

    it('should handle rapid search requests', async () => {
      renderWithErrorHandlers(
        <SearchInterface />,
        []
      )

      const searchInput = screen.getByPlaceholderText(/search/i)
      const searchButton = screen.getByRole('button', { name: /search/i })
      
      // Type and search rapidly
      await userEvent.type(searchInput, 'a')
      await userEvent.click(searchButton)
      
      await userEvent.clear(searchInput)
      await userEvent.type(searchInput, 'ab')
      await userEvent.click(searchButton)
      
      await userEvent.clear(searchInput)
      await userEvent.type(searchInput, 'abc')
      await userEvent.click(searchButton)
      
      // Should handle all requests without errors
      await waitFor(() => {
        expect(mockSearch).toHaveBeenCalledTimes(3)
      })
    })
  })

  describe('Collection Loading Errors During Search', () => {
    it('should handle collection list refresh failure gracefully', async () => {
      const mockFetchCollections = vi.fn()
      vi.mocked(useCollectionStore).mockReturnValue({
        ...vi.mocked(useCollectionStore).mock.results[0].value,
        collections: [],
        fetchCollections: mockFetchCollections,
        error: 'Failed to load collections'
      } as any)
      
      renderWithErrorHandlers(
        <SearchInterface />,
        collectionErrorHandlers.networkError()
      )

      // Should show message about no collections
      expect(screen.getByText(/loading collections/i)).toBeInTheDocument()
      
      // Search should be disabled when no collections
      const searchButton = screen.getByRole('button', { name: /search/i })
      expect(searchButton).toBeDisabled()
    })
  })
})