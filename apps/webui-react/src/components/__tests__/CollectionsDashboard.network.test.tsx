import React from 'react'
import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import CollectionsDashboard from '../CollectionsDashboard'
import { 
  renderWithErrorHandlers, 
  waitForError, 
  simulateOffline,
  simulateOnline,
  // testRetryFunctionality
} from '../../tests/utils/errorTestUtils'
import { collectionErrorHandlers } from '../../tests/mocks/errorHandlers'
// import { createTimeoutHandler } from '../../tests/mocks/errorHandlers'
// import { server } from '../../tests/mocks/server'
// import { handlers } from '../../tests/mocks/handlers'

// Mock hooks and stores
const mockCollectionsQuery = {
  data: [],
  isLoading: false,
  error: null,
  refetch: vi.fn(),
};

const mockCreateCollectionMutation = {
  mutateAsync: vi.fn(),
  isError: false,
  isPending: false,
};

const mockAddToast = vi.fn();

vi.mock('../../hooks/useCollections', () => ({
  useCollections: () => mockCollectionsQuery,
  useCreateCollection: () => mockCreateCollectionMutation,
}));

// Mock useAddSource from useCollectionOperations
vi.mock('../../hooks/useCollectionOperations', () => ({
  useAddSource: () => ({
    mutateAsync: vi.fn(),
    isPending: false,
    isError: false,
  }),
}));

// Mock useConnectorCatalog
vi.mock('../../hooks/useConnectors', () => ({
  useConnectorCatalog: () => ({
    data: {
      directory: {
        display_name: 'Directory',
        description: 'Local directory',
        fields: [],
        supports_preview: false,
      },
    },
    isLoading: false,
  }),
}));

// Mock usePreferences (needed by AnimationContext)
vi.mock('../../hooks/usePreferences', () => ({
  usePreferences: () => ({
    data: { interface: { animation_enabled: true } },
    isLoading: false,
  }),
}));

vi.mock('../../stores/uiStore', () => ({
  useUIStore: () => ({
    addToast: mockAddToast,
  }),
}));

// Mock directory scan
vi.mock('../../hooks/useDirectoryScan', () => ({
  useDirectoryScan: () => ({
    scanning: false,
    scanResult: null,
    error: null,
    startScan: vi.fn(),
    reset: vi.fn(),
  }),
}));

// Mock operation progress
vi.mock('../../hooks/useOperationProgress', () => ({
  useOperationProgress: () => ({
    sendMessage: vi.fn(),
    readyState: WebSocket.CLOSED,
    isConnected: false,
  }),
}));

// Mock navigate
const mockNavigate = vi.fn();
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom');
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  };
});

describe('CollectionsDashboard - Network Error Handling', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Reset mock states
    mockCollectionsQuery.data = [];
    mockCollectionsQuery.isLoading = false;
    mockCollectionsQuery.error = null;
  });
  describe('Collection Loading Failures', () => {
    it('should show error message when collections fail to load due to network error', async () => {
      // Set mock to return error state
      mockCollectionsQuery.error = new Error('Network error')
      mockCollectionsQuery.isLoading = false
      
      renderWithErrorHandlers(
        <CollectionsDashboard />,
        collectionErrorHandlers.networkError()
      )

      await waitForError('Failed to load collections')
      
      // Should show retry button
      const retryButton = screen.getByRole('button', { name: /retry/i })
      expect(retryButton).toBeInTheDocument()
      
      // Should not show any collections
      expect(screen.queryByTestId('collection-card')).not.toBeInTheDocument()
    })

    it('should retry loading collections when retry button is clicked', async () => {
      // Set mock to return error state
      mockCollectionsQuery.error = new Error('Network error')
      mockCollectionsQuery.isLoading = false
      mockCollectionsQuery.refetch = vi.fn()
      
      renderWithErrorHandlers(
        <CollectionsDashboard />,
        collectionErrorHandlers.networkError()
      )

      await waitForError('Failed to load collections')
      
      const retryButton = screen.getByRole('button', { name: /retry/i })
      await userEvent.click(retryButton)
      
      // Should trigger refetch
      await waitFor(() => {
        expect(mockCollectionsQuery.refetch).toHaveBeenCalled()
      })
    })

    it('should handle offline to online transition', async () => {
      simulateOffline()
      
      // Set mock to return error state initially
      mockCollectionsQuery.error = new Error('Network error')
      mockCollectionsQuery.isLoading = false
      mockCollectionsQuery.refetch = vi.fn()
      
      renderWithErrorHandlers(
        <CollectionsDashboard />,
        collectionErrorHandlers.networkError()
      )

      await waitForError('Failed to load collections')
      
      // Go back online
      simulateOnline()
      
      // Click retry
      const retryButton = screen.getByRole('button', { name: /retry/i })
      await userEvent.click(retryButton)
      
      // Should trigger refetch
      await waitFor(() => {
        expect(mockCollectionsQuery.refetch).toHaveBeenCalled()
      })
    })

    it('should persist error state when retry also fails', async () => {
      // Set mock to return error state
      mockCollectionsQuery.error = new Error('Network error')
      mockCollectionsQuery.isLoading = false
      mockCollectionsQuery.refetch = vi.fn().mockImplementation(() => {
        // Keep error state on retry
        mockCollectionsQuery.error = new Error('Network error')
      })
      
      renderWithErrorHandlers(
        <CollectionsDashboard />,
        collectionErrorHandlers.networkError()
      )

      await waitForError('Failed to load collections')
      
      const retryButton = screen.getByRole('button', { name: /retry/i })
      await userEvent.click(retryButton)
      
      // Error should persist
      await waitForError('Failed to load collections')
      expect(retryButton).toBeInTheDocument()
    })
  })

  describe('Collection Creation Network Failures', () => {
    beforeEach(async () => {
      // Reset the mock state to show no collections
      mockCollectionsQuery.data = []
      mockCollectionsQuery.isLoading = false
      mockCollectionsQuery.error = null
      
      // Reset mutation mock
      mockCreateCollectionMutation.mutateAsync = vi.fn()
      mockCreateCollectionMutation.isError = false
      mockCreateCollectionMutation.isPending = false
      
      // Render the component
      renderWithErrorHandlers(<CollectionsDashboard />, [])
      
      // Wait for component to render
      await waitFor(() => {
        expect(screen.getByText('Collections')).toBeInTheDocument()
      })
      
      // Open create modal - when empty state, there are two buttons, click the header one
      const header = screen.getByRole('heading', { name: 'Collections' }).parentElement?.parentElement
      const headerButton = within(header!).getByRole('button', { name: /new collection/i })
      await userEvent.click(headerButton)
      
      // Wait for modal to open and be fully rendered
      await waitFor(() => {
        expect(screen.getByRole('dialog')).toBeInTheDocument()
        // Look for the wizard title specifically (h2 with id="wizard-title")
        expect(screen.getByRole('heading', { name: /create collection/i, level: 2 })).toBeInTheDocument()
        expect(screen.getByLabelText(/collection name/i)).toBeInTheDocument()
      })
    })

    it('should show error toast when collection creation fails due to network error', async () => {
      // Set up mutation to fail
      mockCreateCollectionMutation.mutateAsync = vi.fn().mockRejectedValue(new Error('Network error'))

      // Fill form - Step 1: Basics
      const nameInput = screen.getByLabelText(/collection name/i)
      await userEvent.type(nameInput, 'Test Collection')

      // Navigate to Step 2: Mode Selection
      const modal = screen.getByRole('dialog')
      let nextButton = within(modal).getByRole('button', { name: /next/i })
      await userEvent.click(nextButton)

      // Wait for step 2
      await waitFor(() => {
        expect(screen.getByText(/manual/i)).toBeInTheDocument()
      })

      // Navigate to Step 3: Configure (manual flow)
      nextButton = within(modal).getByRole('button', { name: /next/i })
      await userEvent.click(nextButton)

      // Wait for step 3 with Create Collection button
      await waitFor(() => {
        expect(within(modal).getByRole('button', { name: /create collection/i })).toBeInTheDocument()
      })

      // Click Create Collection
      const createButton = within(modal).getByRole('button', { name: /create collection/i })
      await userEvent.click(createButton)

      // Should call mutateAsync
      await waitFor(() => {
        expect(mockCreateCollectionMutation.mutateAsync).toHaveBeenCalled()
      })

      // Should show error toast
      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith({
          message: expect.stringContaining('Network error'),
          type: 'error'
        })
      })

      // Modal should remain open (wizard stays on current step on error)
      expect(screen.getByRole('dialog')).toBeInTheDocument()
    })

    it('should preserve form data when network error occurs', async () => {
      // Set up mutation to fail - override the mock again to ensure it's set
      mockCreateCollectionMutation.mutateAsync = vi.fn().mockRejectedValue(new Error('Network error'))
      mockCreateCollectionMutation.isError = false
      mockCreateCollectionMutation.isPending = false

      // Wait for modal to be fully rendered
      await waitFor(() => {
        expect(screen.getByRole('dialog')).toBeInTheDocument()
      })

      // Fill out form - Step 1: Name
      await userEvent.type(screen.getByLabelText(/collection name/i), 'My Collection')

      const modal = screen.getByRole('dialog')

      // Navigate through steps to reach Create Collection
      let nextButton = within(modal).getByRole('button', { name: /next/i })
      await userEvent.click(nextButton)

      // Wait for step 2
      await waitFor(() => {
        expect(screen.getByText(/manual/i)).toBeInTheDocument()
      })

      // Navigate to Step 3
      nextButton = within(modal).getByRole('button', { name: /next/i })
      await userEvent.click(nextButton)

      // Wait for step 3 with Create Collection button
      await waitFor(() => {
        expect(within(modal).getByRole('button', { name: /create collection/i })).toBeInTheDocument()
      })

      // Click Create Collection
      const createButton = within(modal).getByRole('button', { name: /create collection/i })
      await userEvent.click(createButton)
      
      // Should show error toast
      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith({
          message: expect.stringContaining('Network error'),
          type: 'error'
        })
      })

      // Modal should still be open after error
      expect(screen.getByRole('dialog')).toBeInTheDocument()

      // Can navigate back to step 1 to verify form data preserved
      const backButton = within(modal).getByRole('button', { name: /back/i })
      await userEvent.click(backButton) // Go to step 2
      await userEvent.click(within(modal).getByRole('button', { name: /back/i })) // Go to step 1

      // Form data should be preserved
      await waitFor(() => {
        expect(screen.getByLabelText(/collection name/i)).toHaveValue('My Collection')
      })
    })

    it('should allow retry after network error with preserved data', async () => {
      // First attempt fails
      mockCreateCollectionMutation.mutateAsync = vi.fn()
        .mockRejectedValueOnce(new Error('Network error'))
        .mockResolvedValueOnce({
          id: '123',
          name: 'Test Collection',
          initial_operation_id: 'op-123'
        })

      // Step 1: Fill form
      await userEvent.type(screen.getByLabelText(/collection name/i), 'Test Collection')

      const modal = screen.getByRole('dialog')

      // Navigate to step 3
      let nextButton = within(modal).getByRole('button', { name: /next/i })
      await userEvent.click(nextButton)
      await waitFor(() => expect(screen.getByText(/manual/i)).toBeInTheDocument())

      nextButton = within(modal).getByRole('button', { name: /next/i })
      await userEvent.click(nextButton)

      await waitFor(() => {
        expect(within(modal).getByRole('button', { name: /create collection/i })).toBeInTheDocument()
      })

      // First attempt - should fail
      const createButton = within(modal).getByRole('button', { name: /create collection/i })
      await userEvent.click(createButton)

      // Should show error toast
      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith({
          message: expect.stringContaining('Network error'),
          type: 'error'
        })
      })

      // Clear previous calls
      mockAddToast.mockClear()

      // Try again - should succeed
      await userEvent.click(createButton)

      // Should succeed this time
      await waitFor(() => {
        expect(mockCreateCollectionMutation.mutateAsync).toHaveBeenCalledTimes(2)
      })

      // Should show success toast
      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith({
          message: 'Collection created successfully!',
          type: 'success'
        })
      })
    })
  })

  describe('Auto-refresh Network Handling', () => {
    it('should handle network errors during auto-refresh gracefully', async () => {
      // Start with successful load - mock returns data
      mockCollectionsQuery.data = [
        {
          id: '1',
          name: 'Test Collection',
          description: 'Test description',
          status: 'ready',
          document_count: 10,
          vector_count: 100,
          owner_id: 1,
          vector_store_name: 'test-store',
          embedding_model: 'test-model',
          quantization: 'float16',
          chunk_size: 1000,
          chunk_overlap: 200,
          is_public: false,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString()
        }
      ]
      mockCollectionsQuery.isLoading = false
      mockCollectionsQuery.error = null
      
      renderWithErrorHandlers(<CollectionsDashboard />, [])
      
      // Wait for initial load - should show the collection
      await waitFor(() => {
        expect(screen.getByText('Test Collection')).toBeInTheDocument()
      })
      
      // Now simulate error during refresh - but keep the data (stale while revalidating)
      mockCollectionsQuery.error = new Error('Network error')
      
      // Collections should remain visible (stale data)
      // Error might show as a toast but shouldn't replace the UI
      expect(screen.getByText('Test Collection')).toBeInTheDocument()
      expect(screen.queryByText(/failed to load collections/i)).not.toBeInTheDocument()
    })
  })

  describe('Search with Network Errors', () => {
    it('should still allow searching in already loaded collections when network fails', async () => {
      // Load successfully first - mock returns data
      mockCollectionsQuery.data = [
        {
          id: '1',
          name: 'Test Collection',
          description: 'Test description',
          status: 'ready',
          document_count: 10,
          vector_count: 100,
          owner_id: 1,
          vector_store_name: 'test-store',
          embedding_model: 'test-model',
          quantization: 'float16',
          chunk_size: 1000,
          chunk_overlap: 200,
          is_public: false,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString()
        },
        {
          id: '2',
          name: 'Another Collection',
          description: 'Another description',
          status: 'ready',
          document_count: 5,
          vector_count: 50,
          owner_id: 1,
          vector_store_name: 'test-store-2',
          embedding_model: 'test-model',
          quantization: 'float16',
          chunk_size: 1000,
          chunk_overlap: 200,
          is_public: false,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString()
        }
      ]
      mockCollectionsQuery.isLoading = false
      mockCollectionsQuery.error = null
      
      renderWithErrorHandlers(<CollectionsDashboard />, [])
      
      await waitFor(() => {
        expect(screen.getByText('Test Collection')).toBeInTheDocument()
        expect(screen.getByText('Another Collection')).toBeInTheDocument()
      })
      
      // Now go offline
      simulateOffline()
      
      // Search should still work on cached data
      const searchInput = screen.getByPlaceholderText(/search collections/i)
      await userEvent.type(searchInput, 'test')
      
      // Should filter existing collections
      // (This assumes the search is client-side on already loaded data)
      expect(searchInput).toHaveValue('test')
      expect(screen.getByText('Test Collection')).toBeInTheDocument()
      expect(screen.queryByText('Another Collection')).not.toBeInTheDocument()
    })
  })

  describe('Timeout Handling', () => {
    it('should handle request timeouts appropriately', async () => {
      // Set initial loading state
      mockCollectionsQuery.isLoading = true
      mockCollectionsQuery.error = null
      mockCollectionsQuery.data = []
      
      const { rerender } = renderWithErrorHandlers(<CollectionsDashboard />, [])
      
      // Should show loading initially
      expect(screen.getByRole('status')).toBeInTheDocument() // Loading spinner
      
      // Simulate timeout - update state and rerender
      mockCollectionsQuery.isLoading = false
      mockCollectionsQuery.error = new Error('Request timeout')
      rerender(<CollectionsDashboard />)
      
      // Should show error
      await waitFor(() => {
        expect(screen.getByText(/failed to load collections/i)).toBeInTheDocument()
      })
      
      // Should offer retry
      expect(screen.getByRole('button', { name: /retry/i })).toBeInTheDocument()
    })
  })
})