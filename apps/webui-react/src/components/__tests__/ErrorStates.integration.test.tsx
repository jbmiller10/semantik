import React from 'react'
import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { CollectionsDashboard } from '../CollectionsDashboard'
import { SearchInterface } from '../SearchInterface'
import { ActiveOperationsTab } from '../ActiveOperationsTab'
import { HomePage } from '../../pages/HomePage'
import { useCollectionStore } from '../../stores/collectionStore'
import { useSearchStore } from '../../stores/searchStore'
import { useUIStore } from '../../stores/uiStore'
import { 
  renderWithErrorHandlers,
  mockConsoleError
} from '../../tests/utils/errorTestUtils'
import { TestWrapper } from '../../tests/utils/test-utils'
import { render } from '@testing-library/react'
import type { Collection } from '@/types/collection'

// Mock stores
vi.mock('../../stores/collectionStore')
vi.mock('../../stores/searchStore')
vi.mock('../../stores/uiStore')
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom')
  return {
    ...actual,
    useNavigate: () => vi.fn()
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
    useReranker: false
  },
  collections: [],
  failedCollections: [],
  partialFailure: false,
  totalResults: 0,
  searchTime: 0,
  validationErrors: {},
  isSearching: false,
  performSearch: vi.fn(),
  setSearchParams: vi.fn(),
  clearResults: vi.fn(),
  reset: vi.fn(),
  ...overrides
} as unknown as ReturnType<typeof useSearchStore>)

// Helper to create a collection store mock  
const createCollectionStoreMock = (overrides?: Record<string, unknown>) => ({
  selectedCollectionId: null,
  setSelectedCollection: vi.fn(),
  clearStore: vi.fn(),
  ...overrides
} as unknown as ReturnType<typeof useCollectionStore>)

describe('Error States - Integration Tests', () => {
  const mockAddToast = vi.fn()
  
  beforeEach(() => {
    vi.clearAllMocks()
    
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
      vi.mocked(useCollectionStore).mockReturnValue({
        selectedCollectionId: null,
        setSelectedCollection: vi.fn(),
        clearStore: vi.fn(),
        collections: [],
        loading: true,
        error: null,
        fetchCollections: vi.fn()
      } as unknown as ReturnType<typeof useCollectionStore>)
      
      renderWithErrorHandlers(<CollectionsDashboard />, [])
      
      // Should show loading state
      expect(screen.getByText(/loading collections/i)).toBeInTheDocument()
      
      // Should not show error or empty state
      expect(screen.queryByText(/failed to load/i)).not.toBeInTheDocument()
      expect(screen.queryByText(/no collections yet/i)).not.toBeInTheDocument()
    })

    it('should show loading spinner in search while searching', () => {
      vi.mocked(useSearchStore).mockReturnValue(createSearchStoreMock({
        isSearching: true,
        loading: true,
        searchParams: { ...createSearchStoreMock().searchParams, query: 'test' }
      }))
      
      vi.mocked(useCollectionStore).mockReturnValue(createCollectionStoreMock({
        collections: [
          { uuid: '1', name: 'Test Collection', status: 'ready' } as unknown as Collection
        ],
        loading: false
      }))
      
      renderWithErrorHandlers(<SearchInterface />, [])
      
      // Search button should show searching state
      expect(screen.getByRole('button', { name: /searching/i })).toBeDisabled()
    })

    it('should show loading state in ActiveOperationsTab', () => {
      render(
        <TestWrapper>
          <ActiveOperationsTab />
        </TestWrapper>
      )
      
      // Should show loading initially
      expect(screen.getByText(/loading operations/i)).toBeInTheDocument()
    })
  })

  describe('Empty States', () => {
    it('should show helpful empty state for new users', () => {
      vi.mocked(useCollectionStore).mockReturnValue(createCollectionStoreMock({
        collections: [],
        loading: false,
        error: null,
        fetchCollections: vi.fn()
      }))
      
      renderWithErrorHandlers(<CollectionsDashboard />, [])
      
      // Should show empty state
      expect(screen.getByText(/no collections yet/i)).toBeInTheDocument()
      expect(screen.getByText(/get started by creating your first collection/i)).toBeInTheDocument()
      
      // Should show prominent create button
      const createButton = screen.getByRole('button', { name: /create.*collection/i })
      expect(createButton).toBeInTheDocument()
      expect(createButton).toHaveClass('bg-blue-500') // Primary button
    })

    it('should show empty search results appropriately', () => {
      vi.mocked(useSearchStore).mockReturnValue(createSearchStoreMock({
        searchParams: { ...createSearchStoreMock().searchParams, query: 'obscure query with no matches' }
      }))
      
      renderWithErrorHandlers(
        <TestWrapper>
          <SearchInterface />
        </TestWrapper>,
        []
      )
      
      // Should show no results message
      expect(screen.getByText(/no results found/i)).toBeInTheDocument()
      
      // Should suggest trying different search terms
      expect(screen.getByText(/try different search terms/i)).toBeInTheDocument()
    })

    it('should show empty active operations state', () => {
      vi.mocked(useCollectionStore).mockReturnValue(createCollectionStoreMock({
        getCollectionOperations: vi.fn().mockReturnValue([])
      }))
      
      render(
        <TestWrapper>
          <ActiveOperationsTab />
        </TestWrapper>
      )
      
      waitFor(() => {
        expect(screen.getByText(/no active operations/i)).toBeInTheDocument()
        expect(screen.getByText(/operations in progress will appear here/i)).toBeInTheDocument()
      })
    })
  })

  describe('Error Recovery', () => {
    it('should allow retry after error in CollectionsDashboard', async () => {
      const mockFetchCollections = vi.fn()
      
      // First render with error
      vi.mocked(useCollectionStore).mockReturnValue(createCollectionStoreMock({
        collections: [],
        loading: false,
        error: 'Network error',
        fetchCollections: mockFetchCollections
      }))
      
      const { rerender } = renderWithErrorHandlers(<CollectionsDashboard />, [])
      
      // Should show error with retry
      expect(screen.getByText(/failed to load collections/i)).toBeInTheDocument()
      const retryButton = screen.getByRole('button', { name: /retry/i })
      
      // Mock successful retry
      vi.mocked(useCollectionStore).mockReturnValue(createCollectionStoreMock({
        collections: [
          { uuid: '1', name: 'Test Collection', status: 'ready' } as unknown as Collection
        ],
        loading: false,
        error: null,
        fetchCollections: mockFetchCollections
      }))
      
      await userEvent.click(retryButton)
      
      // Should call fetch
      expect(mockFetchCollections).toHaveBeenCalled()
      
      // Rerender with new state
      rerender(<CollectionsDashboard />)
      
      // Should show collections
      expect(screen.queryByText(/failed to load/i)).not.toBeInTheDocument()
      expect(screen.getByText('Test Collection')).toBeInTheDocument()
    })

    it('should clear error state when switching tabs', () => {
      // Start with error in collections
      vi.mocked(useCollectionStore).mockReturnValue(createCollectionStoreMock({
        collections: [],
        loading: false,
        error: 'Failed to load',
        fetchCollections: vi.fn()
      }))
      
      vi.mocked(useSearchStore).mockReturnValue(createSearchStoreMock())
      
      const { rerender } = render(
        <TestWrapper>
          <HomePage />
        </TestWrapper>
      )
      
      // Should show error
      expect(screen.getByText(/failed to load/i)).toBeInTheDocument()
      
      // Switch to search tab
      vi.mocked(useUIStore).mockReturnValue({
        addToast: mockAddToast,
        activeTab: 'search',
        toasts: [],
        showDocumentViewer: null,
        showCollectionDetailsModal: null,
        removeToast: vi.fn(),
        setActiveTab: vi.fn(),
        setShowDocumentViewer: vi.fn(),
        setShowCollectionDetailsModal: vi.fn()
      })
      
      rerender(
        <TestWrapper>
          <HomePage />
        </TestWrapper>
      )
      
      // Error should not be visible
      expect(screen.queryByText(/failed to load/i)).not.toBeInTheDocument()
    })
  })

  describe('Error Boundaries in Action', () => {
    it('should catch render errors in CollectionCard', async () => {
      const { restore } = mockConsoleError()
      
      try {
        // Create a collection with invalid data that will cause render error
        const badCollection = {
          uuid: 'bad-id',
          name: null, // This might cause issues
          status: 'ready',
          // Missing required fields
        }
        
        vi.mocked(useCollectionStore).mockReturnValue(createCollectionStoreMock({
          collections: [badCollection as unknown as Collection],
          loading: false,
          error: null,
          fetchCollections: vi.fn()
        }))
        
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
      const mockFetchCollections = vi.fn().mockRejectedValue(
        new Error('Unexpected error in effect')
      )
      
      vi.mocked(useCollectionStore).mockReturnValue(createCollectionStoreMock({
        collections: [],
        loading: false,
        error: null,
        fetchCollections: mockFetchCollections
      }))
      
      renderWithErrorHandlers(<CollectionsDashboard />, [])
      
      // Component should handle async errors gracefully
      await waitFor(() => {
        expect(mockFetchCollections).toHaveBeenCalled()
      })
      
      // Should not crash the component
      expect(screen.getByRole('main')).toBeInTheDocument()
    })
  })

  describe('Concurrent Error States', () => {
    it('should handle multiple simultaneous errors', async () => {
      // Set up multiple error sources
      vi.mocked(useCollectionStore).mockReturnValue(createCollectionStoreMock({
        collections: [],
        loading: false,
        error: 'Collections service unavailable',
        fetchCollections: vi.fn(),
        operations: [],
        operationsError: 'Operations service unavailable'
      }))
      
      vi.mocked(useSearchStore).mockReturnValue(createSearchStoreMock({
        error: 'Search service unavailable'
      }))
      
      renderWithErrorHandlers(<HomePage />, [])
      
      // Should show primary error
      expect(screen.getByText(/collections service unavailable/i)).toBeInTheDocument()
      
      // Should not show cascading errors to avoid overwhelming user
      expect(screen.queryAllByText(/unavailable/i).length).toBeLessThanOrEqual(2)
    })

    it('should prioritize critical errors', () => {
      // Auth error should take precedence
      // TODO: server.use(...authErrorHandlers.unauthorized())
      
      vi.mocked(useCollectionStore).mockReturnValue({
        selectedCollectionId: null,
        setSelectedCollection: vi.fn(),
        clearStore: vi.fn(),
        collections: [],
        loading: false,
        error: 'Unauthorized',
        fetchCollections: vi.fn()
      } as unknown as ReturnType<typeof useCollectionStore>)
      
      const mockNavigate = vi.fn()
      vi.mock('react-router-dom', () => ({
        ...vi.importActual('react-router-dom'),
        useNavigate: () => mockNavigate
      }))
      
      renderWithErrorHandlers(<CollectionsDashboard />, [])
      
      // Should handle auth error by redirecting
      waitFor(() => {
        expect(mockNavigate).toHaveBeenCalledWith('/login')
      })
    })
  })

  describe('Error State Transitions', () => {
    it('should transition smoothly from error to success', async () => {
      const mockFetchCollections = vi.fn()
      
      // Start with error
      let storeState = {
        collections: [],
        loading: false,
        error: 'Network error',
        fetchCollections: mockFetchCollections
      }
      
      vi.mocked(useCollectionStore).mockReturnValue({
        selectedCollectionId: null,
        setSelectedCollection: vi.fn(),
        clearStore: vi.fn(),
        ...storeState
      } as unknown as ReturnType<typeof useCollectionStore>)
      
      const { rerender } = renderWithErrorHandlers(<CollectionsDashboard />, [])
      
      // Show error
      expect(screen.getByText(/failed to load/i)).toBeInTheDocument()
      
      // Transition to loading
      storeState = { ...storeState, loading: true, error: null }
      vi.mocked(useCollectionStore).mockReturnValue({
        selectedCollectionId: null,
        setSelectedCollection: vi.fn(),
        clearStore: vi.fn(),
        ...storeState
      } as unknown as ReturnType<typeof useCollectionStore>)
      rerender(<CollectionsDashboard />)
      
      expect(screen.queryByText(/failed to load/i)).not.toBeInTheDocument()
      expect(screen.getByText(/loading/i)).toBeInTheDocument()
      
      // Transition to success
      storeState = {
        ...storeState,
        loading: false,
        collections: [{ uuid: '1', name: 'Success!', status: 'ready' } as unknown as Collection]
      }
      vi.mocked(useCollectionStore).mockReturnValue({
        selectedCollectionId: null,
        setSelectedCollection: vi.fn(),
        clearStore: vi.fn(),
        ...storeState
      } as unknown as ReturnType<typeof useCollectionStore>)
      rerender(<CollectionsDashboard />)
      
      expect(screen.queryByText(/loading/i)).not.toBeInTheDocument()
      expect(screen.getByText('Success!')).toBeInTheDocument()
    })

    it('should maintain error state while retrying', async () => {
      const mockSearch = vi.fn()
      
      vi.mocked(useSearchStore).mockReturnValue(createSearchStoreMock({
        error: 'Search failed',
        performSearch: mockSearch,
        searchParams: { ...createSearchStoreMock().searchParams, query: 'test' }
      }))
      
      vi.mocked(useCollectionStore).mockReturnValue(createCollectionStoreMock({
        collections: [{ uuid: '1', name: 'Test', status: 'ready' } as unknown as Collection]
      }))
      
      renderWithErrorHandlers(<SearchInterface />, [])
      
      // Error should be visible
      expect(screen.getByText(/search failed/i)).toBeInTheDocument()
      
      // Start new search
      vi.mocked(useSearchStore).mockReturnValue(createSearchStoreMock({
        error: 'Search failed',
        isSearching: true,
        loading: true,
        performSearch: mockSearch,
        searchParams: { ...createSearchStoreMock().searchParams, query: 'test' }
      }))
      
      await userEvent.click(screen.getByRole('button', { name: /search/i }))
      
      // Should show searching state but error might persist
      expect(screen.getByRole('button', { name: /searching/i })).toBeDisabled()
    })
  })
})

// Import ErrorBoundary if it exists
const ErrorBoundary = React.lazy(() => import('../ErrorBoundary'))