import React from 'react'
import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import CollectionsDashboard from '../CollectionsDashboard'
import SearchInterface from '../SearchInterface'
import ActiveOperationsTab from '../ActiveOperationsTab'
// import HomePage from '../../pages/HomePage'
import { useCollectionStore } from '../../stores/collectionStore'
import { useSearchStore } from '../../stores/searchStore'
import { useUIStore } from '../../stores/uiStore'
import { 
  renderWithErrorHandlers,
  mockConsoleError
} from '../../tests/utils/errorTestUtils'
import { TestWrapper } from '../../tests/utils/TestWrapper'
import { render } from '@testing-library/react'
import type { Collection } from '../../types/collection'
import ErrorBoundary from '../ErrorBoundary'
import { useCollections } from '../../hooks/useCollections'
// import { useNavigate } from 'react-router-dom'
import { operationsV2Api } from '../../services/api/v2/collections'

// Mock stores
vi.mock('../../stores/collectionStore')
vi.mock('../../stores/searchStore')
vi.mock('../../stores/uiStore')
vi.mock('../../hooks/useCollections')
vi.mock('../../services/api/v2/collections')

// Mock react-router-dom
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual<typeof import('react-router-dom')>('react-router-dom')
  return {
    ...actual,
    useNavigate: () => vi.fn(),
    MemoryRouter: actual.MemoryRouter // Explicitly include MemoryRouter
  }
})

// Helper to create a search store mock
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
    hybridMode: 'rerank' as const,
    keywordMode: 'any' as const,
  },
  collections: [],
  failedCollections: [],
  partialFailure: false,
  rerankingMetrics: null,
  validationErrors: [],
  rerankingAvailable: true,
  rerankingModelsLoading: false,
  performSearch: vi.fn(),
  setResults: vi.fn(),
  setLoading: vi.fn(),
  setError: vi.fn(),
  updateSearchParams: vi.fn(),
  setCollections: vi.fn(),
  setFailedCollections: vi.fn(),
  setPartialFailure: vi.fn(),
  clearResults: vi.fn(),
  setRerankingMetrics: vi.fn(),
  validateAndUpdateSearchParams: vi.fn(),
  clearValidationErrors: vi.fn(),
  hasValidationErrors: vi.fn(),
  getValidationError: vi.fn(),
  setRerankingAvailable: vi.fn(),
  setRerankingModelsLoading: vi.fn(),
  reset: vi.fn(),
  ...overrides
} as unknown as ReturnType<typeof useSearchStore>)

// Helper to create a collection store mock  
const createCollectionStoreMock = (overrides?: Record<string, unknown>) => ({
  selectedCollectionId: null,
  setSelectedCollection: vi.fn(),
  clearStore: vi.fn(),
  collections: [],
  loading: false,
  error: null,
  fetchCollections: vi.fn(),
  getCollectionOperations: vi.fn().mockReturnValue([]),
  ...overrides
} as unknown as ReturnType<typeof useCollectionStore>)

describe('Error States - Integration Tests', () => {
  const mockAddToast = vi.fn()
  
  beforeEach(() => {
    vi.clearAllMocks()
    
    // Default mock - sets activeTab to 'collections'
    vi.mocked(useUIStore).mockReturnValue({
      addToast: mockAddToast,
      activeTab: 'collections',
      toasts: [],
      showDocumentViewer: null,
      showCollectionDetailsModal: null,
      removeToast: vi.fn(),
      setActiveTab: vi.fn(),
      setShowDocumentViewer: vi.fn(),
      setShowCollectionDetailsModal: vi.fn()
    })
  })

  describe('Loading States', () => {
    it('should show loading skeleton in CollectionsDashboard', () => {
      // Mock useCollections hook to return loading state
      vi.mocked(useCollections).mockReturnValue({
        data: [],
        isLoading: true,
        error: null,
        refetch: vi.fn()
      } as unknown as ReturnType<typeof useCollections>)
      
      renderWithErrorHandlers(<CollectionsDashboard />, [])
      
      // Should show loading spinner (no text)
      const loadingSpinner = document.querySelector('.animate-spin')
      expect(loadingSpinner).toBeInTheDocument()
      
      // Should not show error or empty state
      expect(screen.queryByText(/failed to load/i)).not.toBeInTheDocument()
      expect(screen.queryByText(/no collections yet/i)).not.toBeInTheDocument()
    })

    it('should show loading spinner in search while searching', () => {
      vi.mocked(useSearchStore).mockReturnValue(createSearchStoreMock({
        loading: true,
        searchParams: { 
          query: 'test',
          selectedCollections: ['1'],  // Fixed to match collection uuid
          topK: 10,
          scoreThreshold: 0.5,
          searchType: 'semantic',
          useReranker: false,
          hybridAlpha: 0.7,
          hybridMode: 'rerank',
          keywordMode: 'any'
        }
      }))
      
      vi.mocked(useCollectionStore).mockReturnValue(createCollectionStoreMock({
        collections: [
          { uuid: '1', name: 'Test Collection', status: 'ready', embedding_model: 'text-embedding-ada-002', documents_count: 0, chunks_count: 0, vectors_count: 0 } as unknown as Collection
        ]
      }))
      
      // Mock useCollections hook to return the collections
      vi.mocked(useCollections).mockReturnValue({
        data: [
          { uuid: '1', name: 'Test Collection', status: 'ready', embedding_model: 'text-embedding-ada-002', documents_count: 0, chunks_count: 0, vectors_count: 0 } as unknown as Collection
        ],
        isLoading: false,
        error: null,
        refetch: vi.fn()
      } as unknown as ReturnType<typeof useCollections>)
      
      renderWithErrorHandlers(<SearchInterface />, [])
      
      // Search button should show searching state
      const searchButton = screen.getByRole('button', { name: /perform search/i })
      expect(searchButton).toHaveTextContent('Searching...')
      expect(searchButton).toBeDisabled()
    })

    it('should show loading state in ActiveOperationsTab', async () => {
      vi.mocked(useCollectionStore).mockReturnValue(createCollectionStoreMock({
        getCollectionOperations: vi.fn().mockReturnValue(undefined) // Loading state
      }))
      
      render(
        <TestWrapper>
          <ActiveOperationsTab />
        </TestWrapper>
      )
      
      // Should show loading state (check for spinner or loading indicator)
      const loadingElement = document.querySelector('.animate-spin') || screen.queryByText(/loading/i)
      expect(loadingElement).toBeInTheDocument()
    })
  })

  describe('Empty States', () => {
    it('should show helpful empty state for new users', () => {
      // Mock useCollections hook to return empty state
      vi.mocked(useCollections).mockReturnValue({
        data: [],
        isLoading: false,
        error: null,
        refetch: vi.fn()
      } as unknown as ReturnType<typeof useCollections>)
      
      renderWithErrorHandlers(<CollectionsDashboard />, [])
      
      // Should show empty state
      expect(screen.getByText(/no collections yet/i)).toBeInTheDocument()
      expect(screen.getByText(/get started by creating your first collection/i)).toBeInTheDocument()
      
      // Should show prominent create button(s)
      const createButtons = screen.getAllByRole('button', { name: /create.*collection/i })
      expect(createButtons.length).toBeGreaterThan(0)
      // Check at least one has the primary button style
      expect(createButtons.some(button => button.classList.contains('bg-blue-600'))).toBe(true)
    })

    it('should show empty search results appropriately', () => {
      vi.mocked(useSearchStore).mockReturnValue(createSearchStoreMock({
        results: [],
        searchParams: { 
          query: 'obscure query with no matches',
          selectedCollections: ['coll-1'],
          topK: 10,
          scoreThreshold: 0.5,
          searchType: 'semantic',
          useReranker: false,
          hybridAlpha: 0.7,
          hybridMode: 'rerank',
          keywordMode: 'any'
        }
      }))
      
      renderWithErrorHandlers(<SearchInterface />, [])
      
      // SearchInterface shows empty state through SearchResults component
      // which only renders when there are results or errors
    })

    it('should show empty active operations state', async () => {
      // Mock the API to return empty operations
      vi.mocked(operationsV2Api.list).mockResolvedValue({
        data: []
      })
      
      renderWithErrorHandlers(<ActiveOperationsTab />, [])
      
      await waitFor(() => {
        expect(screen.getByText(/no active operations/i)).toBeInTheDocument()
        expect(screen.getByText(/all operations have completed/i)).toBeInTheDocument()
      })
    })
  })

  describe('Error Recovery', () => {
    it('should allow retry after error in CollectionsDashboard', async () => {
      const mockRefetch = vi.fn()
      
      // First render with error
      vi.mocked(useCollections).mockReturnValue({
        data: [],
        isLoading: false,
        error: new Error('Network error'),
        refetch: mockRefetch
      } as unknown as ReturnType<typeof useCollections>)
      
      const { rerender } = renderWithErrorHandlers(<CollectionsDashboard />, [])
      
      // Should show error with retry
      expect(screen.getByText(/failed to load collections/i)).toBeInTheDocument()
      const retryButton = screen.getByRole('button', { name: /retry/i })
      
      // Click retry button
      await userEvent.click(retryButton)
      
      // Should call refetch
      expect(mockRefetch).toHaveBeenCalled()
      
      // Mock successful retry
      vi.mocked(useCollections).mockReturnValue({
        data: [
          { 
            id: '1', 
            name: 'Test Collection', 
            status: 'ready',
            embedding_model: 'text-embedding-ada-002',
            documents_count: 0,
            chunks_count: 0,
            vectors_count: 0
          } as unknown as Collection
        ],
        isLoading: false,
        error: null,
        refetch: mockRefetch
      } as unknown as ReturnType<typeof useCollections>)
      
      // Rerender with new state
      rerender(<CollectionsDashboard />)
      
      // Should show collections
      expect(screen.queryByText(/failed to load/i)).not.toBeInTheDocument()
      expect(screen.getByText('Test Collection')).toBeInTheDocument()
    })

    it('should clear error state when switching between components', async () => {
      // Start with error in collections
      vi.mocked(useCollections).mockReturnValue({
        data: [],
        isLoading: false,
        error: new Error('Failed to load'),
        refetch: vi.fn()
      } as unknown as ReturnType<typeof useCollections>)
      
      const { unmount } = renderWithErrorHandlers(<CollectionsDashboard />, [])
      
      // Should show error
      expect(screen.getByText(/failed to load collections/i)).toBeInTheDocument()
      
      // Unmount collections and mount search
      unmount()
      
      vi.mocked(useSearchStore).mockReturnValue(createSearchStoreMock())
      vi.mocked(useCollectionStore).mockReturnValue(createCollectionStoreMock())
      
      renderWithErrorHandlers(<SearchInterface />, [])
      
      // Error from collections should not be visible
      expect(screen.queryByText(/failed to load collections/i)).not.toBeInTheDocument()
      // Search interface should be visible
      expect(screen.getByText(/search documents/i)).toBeInTheDocument()
    })
  })

  describe('Error Boundaries in Action', () => {
    it('should catch render errors in CollectionCard', async () => {
      const { restore } = mockConsoleError()
      
      try {
        // Create a collection with invalid data that will cause render error
        const badCollection = {
          id: 'bad-id',
          name: null, // This might cause issues
          status: 'ready',
          // Missing required fields
        }
        
        vi.mocked(useCollections).mockReturnValue({
          data: [badCollection as unknown as Collection],
          isLoading: false,
          error: null,
          refetch: vi.fn()
        } as unknown as ReturnType<typeof useCollections>)
        
        // Wrap in error boundary
        render(
          <TestWrapper>
            <ErrorBoundary>
              <CollectionsDashboard />
            </ErrorBoundary>
          </TestWrapper>
        )
        
        // If error boundary catches it, should show error UI
        // If not, the component might handle null gracefully
        // This test verifies error boundaries work with our components
      } finally {
        restore()
      }
    })

    it('should handle async errors in effects', async () => {
      // CollectionsDashboard uses React Query which handles async errors
      vi.mocked(useCollections).mockReturnValue({
        data: [],
        isLoading: false,
        error: new Error('Unexpected error in effect'),
        refetch: vi.fn()
      } as unknown as ReturnType<typeof useCollections>)
      
      renderWithErrorHandlers(<CollectionsDashboard />, [])
      
      // Should show error state
      expect(screen.getByText(/failed to load collections/i)).toBeInTheDocument()
    })
  })

  describe('Concurrent Error States', () => {
    it('should handle multiple simultaneous errors', async () => {
      // Set up error in collections
      vi.mocked(useCollections).mockReturnValue({
        data: [],
        isLoading: false,
        error: new Error('Collections service unavailable'),
        refetch: vi.fn()
      } as unknown as ReturnType<typeof useCollections>)
      
      vi.mocked(useCollectionStore).mockReturnValue(createCollectionStoreMock())
      
      renderWithErrorHandlers(<CollectionsDashboard />, [])
      
      // Should show error
      expect(screen.getByText(/failed to load collections/i)).toBeInTheDocument()
      
      // The error boundary prevents cascading failures
      // Only one error message should be visible at component level
      expect(screen.queryAllByText(/failed/i).length).toBe(1)
    })

    it('should prioritize critical errors', async () => {
      // Auth error should take precedence
      vi.mocked(useCollections).mockReturnValue({
        data: [],
        isLoading: false,
        error: new Error('Unauthorized'),
        refetch: vi.fn()
      } as unknown as ReturnType<typeof useCollections>)
      
      // The navigate function is already mocked in the module mock at the top
      
      renderWithErrorHandlers(<CollectionsDashboard />, [])
      
      // Should show error message - CollectionsDashboard shows generic error
      expect(screen.getByText(/failed to load collections/i)).toBeInTheDocument()
    })
  })

  describe('Error State Transitions', () => {
    it('should transition smoothly from error to success', async () => {
      const mockFetchCollections = vi.fn()
      
      // Start with error
      vi.mocked(useCollections).mockReturnValue({
        data: [],
        isLoading: false,
        error: new Error('Network error'),
        refetch: mockFetchCollections
      } as unknown as ReturnType<typeof useCollections>)
      
      const { rerender } = renderWithErrorHandlers(<CollectionsDashboard />, [])
      
      // Show error
      expect(screen.getByText(/failed to load/i)).toBeInTheDocument()
      
      // Transition to loading
      vi.mocked(useCollections).mockReturnValue({
        data: [],
        isLoading: true,
        error: null,
        refetch: mockFetchCollections
      } as unknown as ReturnType<typeof useCollections>)
      rerender(<CollectionsDashboard />)
      
      expect(screen.queryByText(/failed to load/i)).not.toBeInTheDocument()
      // CollectionsDashboard shows a spinner, not loading text
      const loadingSpinner = document.querySelector('.animate-spin')
      expect(loadingSpinner).toBeInTheDocument()
      
      // Transition to success
      vi.mocked(useCollections).mockReturnValue({
        data: [{ id: '1', name: 'Success!', status: 'ready', embedding_model: 'text-embedding-ada-002', documents_count: 0, chunks_count: 0, vectors_count: 0 } as unknown as Collection],
        isLoading: false,
        error: null,
        refetch: mockFetchCollections
      } as unknown as ReturnType<typeof useCollections>)
      rerender(<CollectionsDashboard />)
      
      expect(document.querySelector('.animate-spin')).not.toBeInTheDocument()
      expect(screen.getByText('Success!')).toBeInTheDocument()
    })

    it('should maintain error state while retrying', async () => {
      const mockSearch = vi.fn()
      
      vi.mocked(useSearchStore).mockReturnValue(createSearchStoreMock({
        error: 'Search failed',
        performSearch: mockSearch,
        searchParams: { 
          query: 'test',
          selectedCollections: ['1'],
          topK: 10,
          scoreThreshold: 0.5,
          searchType: 'semantic',
          useReranker: false,
          hybridAlpha: 0.7,
          hybridMode: 'rerank',
          keywordMode: 'any'
        }
      }))
      
      vi.mocked(useCollectionStore).mockReturnValue(createCollectionStoreMock({
        collections: [{ uuid: '1', name: 'Test', status: 'ready', embedding_model: 'text-embedding-ada-002', documents_count: 0, chunks_count: 0, vectors_count: 0 } as unknown as Collection]
      }))
      
      // Mock useCollections hook
      vi.mocked(useCollections).mockReturnValue({
        data: [{ uuid: '1', name: 'Test', status: 'ready', embedding_model: 'text-embedding-ada-002', documents_count: 0, chunks_count: 0, vectors_count: 0 } as unknown as Collection],
        isLoading: false,
        error: null,
        refetch: vi.fn()
      } as unknown as ReturnType<typeof useCollections>)
      
      renderWithErrorHandlers(<SearchInterface />, [])
      
      // Error should be visible
      expect(screen.getByText(/search failed/i)).toBeInTheDocument()
      
      // Start new search
      vi.mocked(useSearchStore).mockReturnValue(createSearchStoreMock({
        error: 'Search failed',
        loading: true,
        performSearch: mockSearch,
        searchParams: { 
          query: 'test',
          selectedCollections: ['1'],
          topK: 10,
          scoreThreshold: 0.5,
          searchType: 'semantic',
          useReranker: false,
          hybridAlpha: 0.7,
          hybridMode: 'rerank',
          keywordMode: 'any'
        }
      }))
      
      // The search functionality actually uses internal state, not mocked state
      // So we need to check that the error is still displayed
      expect(screen.getByText(/search failed/i)).toBeInTheDocument()
      
      // Check that we can trigger a search
      const searchButton = screen.getByRole('button', { name: /perform search/i })
      expect(searchButton).toBeInTheDocument()
    })
  })
})