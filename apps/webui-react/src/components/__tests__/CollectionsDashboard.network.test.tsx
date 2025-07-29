import React from 'react'
import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import CollectionsDashboard from '../CollectionsDashboard'
import { 
  renderWithErrorHandlers, 
  waitForError, 
  simulateOffline,
  simulateOnline,
  testRetryFunctionality
} from '../../tests/utils/errorTestUtils'
import { collectionErrorHandlers, createTimeoutHandler } from '../../tests/mocks/errorHandlers'
import { server } from '../../tests/mocks/server'
import { handlers } from '../../tests/mocks/handlers'

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
      // First render with success to see the UI
      server.use(...handlers)
      renderWithErrorHandlers(<CollectionsDashboard />, [])
      
      // Open create modal
      const createButton = await screen.findByRole('button', { name: /create.*collection/i })
      await userEvent.click(createButton)
      
      // Wait for modal to open
      await waitFor(() => {
        expect(screen.getByText(/create new collection/i)).toBeInTheDocument()
      })
    })

    it('should show error toast when collection creation fails due to network error', async () => {
      // Set up mutation to fail
      mockCreateCollectionMutation.mutateAsync = vi.fn().mockRejectedValue(new Error('Network error'))
      
      // Fill form
      const nameInput = screen.getByLabelText(/collection name/i)
      await userEvent.type(nameInput, 'Test Collection')
      
      // Submit
      // Submit the form - find the submit button by querying inside the form
      const submitButtons = await screen.findAllByText('Create Collection')
      // The last one should be the submit button in the modal
      const submitButton = submitButtons[submitButtons.length - 1]
      await userEvent.click(submitButton)
      
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
      
      // Modal should remain open with form data intact
      expect(screen.getByLabelText(/collection name/i)).toHaveValue('Test Collection')
      expect(screen.getByText(/create new collection/i)).toBeInTheDocument()
    })

    it('should preserve form data when network error occurs', async () => {
      // Set up mutation to fail
      mockCreateCollectionMutation.mutateAsync = vi.fn().mockRejectedValue(new Error('Network error'))
      
      // Fill out complete form
      await userEvent.type(screen.getByLabelText(/collection name/i), 'My Collection')
      await userEvent.type(screen.getByLabelText(/description/i), 'Test description')
      
      // Expand advanced settings
      const advancedButton = screen.getByText(/advanced settings/i)
      await userEvent.click(advancedButton)
      
      await userEvent.clear(screen.getByLabelText(/chunk size/i))
      await userEvent.type(screen.getByLabelText(/chunk size/i), '1024')
      
      // Submit
      // Submit the form - find the submit button by querying inside the form
      const submitButtons = await screen.findAllByText('Create Collection')
      // The last one should be the submit button in the modal
      const submitButton = submitButtons[submitButtons.length - 1]
      await userEvent.click(submitButton)
      
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
      
      // All form data should be preserved
      expect(screen.getByLabelText(/collection name/i)).toHaveValue('My Collection')
      expect(screen.getByLabelText(/description/i)).toHaveValue('Test description')
      expect(screen.getByLabelText(/chunk size/i)).toHaveValue('1024')
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
      
      await userEvent.type(screen.getByLabelText(/collection name/i), 'Test Collection')
      
      // Submit the form - find the submit button by querying inside the form
      const submitButtons = await screen.findAllByText('Create Collection')
      // The last one should be the submit button in the modal
      const submitButton = submitButtons[submitButtons.length - 1]
      await userEvent.click(submitButton)
      
      // Should show error toast
      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith({
          message: expect.stringContaining('Network error'),
          type: 'error'
        })
      })
      
      // Clear previous calls
      mockAddToast.mockClear()
      
      // Try again - data should still be there
      await userEvent.click(submitButton)
      
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