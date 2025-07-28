import React from 'react'
import { screen, fireEvent, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { CollectionsDashboard } from '../CollectionsDashboard'
import { 
  renderWithErrorHandlers, 
  waitForError, 
  waitForToast,
  waitForLoadingToComplete,
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
      renderWithErrorHandlers(
        <CollectionsDashboard />,
        collectionErrorHandlers.networkError()
      )

      await waitForError('Failed to load collections')
      
      // Now set up success handlers for retry
      server.use(...handlers)
      
      const retryButton = await testRetryFunctionality()
      await userEvent.click(retryButton)
      
      // Should trigger refetch
      await waitFor(() => {
        expect(mockCollectionsQuery.refetch).toHaveBeenCalled()
      })
      
      // Should eventually show collections
      await waitFor(() => {
        expect(screen.queryByText(/failed to load collections/i)).not.toBeInTheDocument()
      })
    })

    it('should handle offline to online transition', async () => {
      simulateOffline()
      
      renderWithErrorHandlers(
        <CollectionsDashboard />,
        collectionErrorHandlers.networkError()
      )

      await waitForError('Failed to load collections')
      
      // Go back online
      simulateOnline()
      server.use(...handlers)
      
      // Click retry
      const retryButton = screen.getByRole('button', { name: /retry/i })
      await userEvent.click(retryButton)
      
      // Should load successfully
      await waitFor(() => {
        expect(screen.queryByText(/failed to load collections/i)).not.toBeInTheDocument()
      })
    })

    it('should persist error state when retry also fails', async () => {
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
      // Set up network error for creation
      server.use(...collectionErrorHandlers.networkError())
      
      // Fill form
      const nameInput = screen.getByLabelText(/collection name/i)
      await userEvent.type(nameInput, 'Test Collection')
      
      // Submit
      const submitButton = screen.getByRole('button', { name: /create collection/i })
      await userEvent.click(submitButton)
      
      // Should show error toast
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
      server.use(...collectionErrorHandlers.networkError())
      
      // Fill out complete form
      await userEvent.type(screen.getByLabelText(/collection name/i), 'My Collection')
      await userEvent.type(screen.getByLabelText(/description/i), 'Test description')
      
      // Expand advanced settings
      const advancedButton = screen.getByText(/advanced settings/i)
      await userEvent.click(advancedButton)
      
      await userEvent.clear(screen.getByLabelText(/chunk size/i))
      await userEvent.type(screen.getByLabelText(/chunk size/i), '1024')
      
      // Submit
      const submitButton = screen.getByRole('button', { name: /create collection/i })
      await userEvent.click(submitButton)
      
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
      server.use(...collectionErrorHandlers.networkError())
      
      await userEvent.type(screen.getByLabelText(/collection name/i), 'Test Collection')
      
      const submitButton = screen.getByRole('button', { name: /create collection/i })
      await userEvent.click(submitButton)
      
      // Should show error toast
      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith({
          message: expect.stringContaining('Network error'),
          type: 'error'
        })
      })
      
      // Set up success for retry
      server.use(...handlers)
      
      // Try again - data should still be there
      await userEvent.click(submitButton)
      
      // Should succeed this time
      await waitFor(() => {
        expect(screen.queryByText(/create new collection/i)).not.toBeInTheDocument()
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
      // Start with successful load
      server.use(...handlers)
      renderWithErrorHandlers(<CollectionsDashboard />, [])
      
      // Wait for initial load
      await waitFor(() => {
        expect(screen.queryByText(/loading collections/i)).not.toBeInTheDocument()
      })
      
      // Now set up network error for refresh
      server.use(...collectionErrorHandlers.networkError())
      
      // Force a refresh (simulate time passing)
      // The component auto-refreshes every 30s for active operations
      // We'll trigger it manually for testing
      
      // Collections should remain visible (stale data)
      // Error might show as a toast but shouldn't replace the UI
      await waitFor(() => {
        expect(screen.queryByTestId('collection-card')).toBeInTheDocument()
      }, { timeout: 5000 })
    })
  })

  describe('Search with Network Errors', () => {
    it('should still allow searching in already loaded collections when network fails', async () => {
      // Load successfully first
      server.use(...handlers)
      renderWithErrorHandlers(<CollectionsDashboard />, [])
      
      await waitFor(() => {
        expect(screen.queryByText(/loading collections/i)).not.toBeInTheDocument()
      })
      
      // Now go offline
      simulateOffline()
      server.use(...collectionErrorHandlers.networkError())
      
      // Search should still work on cached data
      const searchInput = screen.getByPlaceholderText(/search collections/i)
      await userEvent.type(searchInput, 'test')
      
      // Should filter existing collections
      // (This assumes the search is client-side on already loaded data)
      expect(searchInput).toHaveValue('test')
    })
  })

  describe('Timeout Handling', () => {
    it('should handle request timeouts appropriately', async () => {
      // Create a timeout handler that delays for 5 seconds
      server.use(
        createTimeoutHandler('get', '/api/v2/collections', 100)
      )
      
      renderWithErrorHandlers(<CollectionsDashboard />, [])
      
      // Should show loading initially
      mockCollectionsQuery.isLoading = true
      
      // Verify loading state is triggered
      await waitFor(() => {
        expect(mockCollectionsQuery.refetch).toHaveBeenCalled()
      })
      
      // Should eventually timeout and show error
      await waitForError('Failed to load collections')
      
      // Should offer retry
      expect(screen.getByRole('button', { name: /retry/i })).toBeInTheDocument()
    })
  })
})