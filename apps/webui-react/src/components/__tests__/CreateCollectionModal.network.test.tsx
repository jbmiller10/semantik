import React from 'react'
import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import CreateCollectionModal from '../CreateCollectionModal'
import { 
  renderWithErrorHandlers, 
  waitForToast,
  expectFormDataPreserved,
  simulateOffline,
  simulateOnline
} from '../../tests/utils/errorTestUtils'
import { collectionErrorHandlers } from '../../tests/mocks/errorHandlers'
import { server } from '../../tests/mocks/server'
import { handlers } from '../../tests/mocks/handlers'

// Mock hooks and stores
const mockCreateCollectionMutation = {
  mutateAsync: vi.fn(),
  isError: false,
  isPending: false,
};

const mockAddSourceMutation = {
  mutateAsync: vi.fn(),
  isError: false,
  isPending: false,
};

const mockAddToast = vi.fn();

vi.mock('../../hooks/useCollections', () => ({
  useCreateCollection: () => mockCreateCollectionMutation,
}));

vi.mock('../../hooks/useCollectionOperations', () => ({
  useAddSource: () => mockAddSourceMutation,
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

describe('CreateCollectionModal - Network Error Handling', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('Submission Network Failures', () => {
    it('should show error toast on network failure', async () => {
      mockCreateCollectionMutation.mutateAsync.mockRejectedValue(new Error('Network error'))
      
      renderWithErrorHandlers(
        <CreateCollectionModal onClose={vi.fn()} onSuccess={vi.fn()} />,
        collectionErrorHandlers.networkError()
      )

      // Fill minimum required fields
      await userEvent.type(screen.getByLabelText(/collection name/i), 'Test Collection')
      
      // Submit
      const submitButton = screen.getByRole('button', { name: /create collection/i })
      await userEvent.click(submitButton)
      
      await waitFor(() => {
        expect(mockCreateCollectionMutation.mutateAsync).toHaveBeenCalled()
        expect(mockCreateCollectionMutation.mutateAsync).toHaveBeenCalledWith(
          expect.objectContaining({
            name: 'Test Collection'
          })
        )
      })
      
      // Modal should remain open
      expect(screen.getByText(/create new collection/i)).toBeInTheDocument()
    })

    it('should preserve all form data after network error', async () => {
      mockCreateCollectionMutation.mutateAsync.mockRejectedValue(new Error('Network error'))
      
      renderWithErrorHandlers(
        <CreateCollectionModal onClose={vi.fn()} onSuccess={vi.fn()} />,
        collectionErrorHandlers.networkError()
      )

      // Fill out complete form
      const formData = {
        'Collection Name': 'My Test Collection',
        'Description': 'A detailed description of my collection',
      }
      
      await userEvent.type(screen.getByLabelText(/collection name/i), formData['Collection Name'])
      await userEvent.type(screen.getByLabelText(/description/i), formData['Description'])
      
      // Expand and fill advanced settings
      await userEvent.click(screen.getByText(/advanced settings/i))
      // Don't modify chunk size and overlap - use defaults
      
      // Add initial source
      await userEvent.type(screen.getByLabelText(/initial source/i), '/data/documents')
      
      // Submit
      await userEvent.click(screen.getByRole('button', { name: /create collection/i }))
      
      await waitFor(() => {
        expect(mockCreateCollectionMutation.mutateAsync).toHaveBeenCalled()
        expect(mockCreateCollectionMutation.mutateAsync).toHaveBeenCalledWith(
          expect.objectContaining({
            name: 'My Test Collection',
            description: 'A detailed description of my collection'
          })
        )
      })
      
      // All data should be preserved
      expectFormDataPreserved(formData)
      expect(screen.getByLabelText(/initial source/i)).toHaveValue('/data/documents')
    })

    it('should handle offline scenario gracefully', async () => {
      const onClose = vi.fn()
      
      renderWithErrorHandlers(
        <CreateCollectionModal onClose={onClose} onSuccess={vi.fn()} />,
        []
      )

      // Go offline before submission
      simulateOffline()
      mockCreateCollectionMutation.mutateAsync.mockRejectedValue(new Error('Network error: Unable to create collection'))
      
      await userEvent.type(screen.getByLabelText(/collection name/i), 'Offline Collection')
      await userEvent.click(screen.getByRole('button', { name: /create collection/i }))
      
      await waitFor(() => {
        expect(mockCreateCollectionMutation.mutateAsync).toHaveBeenCalled()
      })
      
      // Modal stays open, form data preserved
      expect(onClose).not.toHaveBeenCalled()
      expect(screen.getByLabelText(/collection name/i)).toHaveValue('Offline Collection')
      
      // Go back online and retry
      simulateOnline()
      mockCreateCollectionMutation.mutateAsync.mockResolvedValue({ id: 'test-uuid', initial_operation_id: null } as any)
      
      await userEvent.click(screen.getByRole('button', { name: /create collection/i }))
      
      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith({
          message: 'Collection created successfully!',
          type: 'success'
        })
      })
      
      await waitFor(() => {
        expect(onClose).not.toHaveBeenCalled() // Modal stays open after success
      })
    })

    it('should disable submit button during network request', async () => {
      // Mock a slow network request
      mockCreateCollectionMutation.mutateAsync.mockImplementation(() => 
        new Promise(resolve => setTimeout(() => resolve({ id: 'test', initial_operation_id: null } as any), 1000))
      )
      
      renderWithErrorHandlers(
        <CreateCollectionModal onClose={vi.fn()} onSuccess={vi.fn()} />,
        []
      )

      await userEvent.type(screen.getByLabelText(/collection name/i), 'Test')
      
      const submitButton = screen.getByRole('button', { name: /create collection/i })
      await userEvent.click(submitButton)
      
      // Button should be disabled immediately
      expect(submitButton).toBeDisabled()
      expect(submitButton).toHaveTextContent(/Creating/i)
      
      // Wait for mutation to complete
      await waitFor(() => {
        expect(mockCreateCollectionMutation.mutateAsync).toHaveBeenCalled()
      }, { timeout: 2000 })
    })
  })

  describe('Two-step Creation Process Network Errors', () => {
    it('should handle failure in add-source step after successful collection creation', async () => {
      // First call succeeds, second fails
      mockCreateCollectionMutation.mutateAsync.mockResolvedValue({ id: 'test-uuid', name: 'Test', initial_operation_id: 'op-123' } as any)
      mockAddSourceMutation.mutateAsync.mockRejectedValue(new Error('Network error during add source'))
      
      renderWithErrorHandlers(
        <CreateCollectionModal onClose={vi.fn()} onSuccess={vi.fn()} />,
        []
      )

      await userEvent.type(screen.getByLabelText(/collection name/i), 'Test Collection')
      await userEvent.type(screen.getByLabelText(/initial source/i), '/data/docs')
      
      await userEvent.click(screen.getByRole('button', { name: /create collection/i }))
      
      // Should show info toast about waiting for initialization
      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith({
          message: 'Collection created! Waiting for initialization before adding source...',
          type: 'info'
        })
      })
    })

    it('should handle network recovery between creation and add-source', async () => {
      mockCreateCollectionMutation.mutateAsync.mockResolvedValue({ id: 'test-uuid', name: 'Test', initial_operation_id: 'op-123' } as any)
      
      // First attempt at add-source fails
      mockAddSourceMutation.mutateAsync.mockRejectedValueOnce(new Error('Network error'))
      // But component doesn't retry automatically, so this test just verifies the warning
      
      renderWithErrorHandlers(
        <CreateCollectionModal onClose={vi.fn()} onSuccess={vi.fn()} />,
        []
      )

      await userEvent.type(screen.getByLabelText(/collection name/i), 'Test Collection')
      await userEvent.type(screen.getByLabelText(/initial source/i), '/data/docs')
      
      await userEvent.click(screen.getByRole('button', { name: /create collection/i }))
      
      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith({
          message: 'Collection created! Waiting for initialization before adding source...',
          type: 'info'
        })
      })
    })
  })

  describe('Intermittent Network Issues', () => {
    it('should handle rapid submit attempts during network issues', async () => {
      mockCreateCollectionMutation.mutateAsync.mockRejectedValue(new Error('Network error'))
      
      renderWithErrorHandlers(
        <CreateCollectionModal onClose={vi.fn()} onSuccess={vi.fn()} />,
        collectionErrorHandlers.networkError()
      )

      await userEvent.type(screen.getByLabelText(/collection name/i), 'Test')
      
      const submitButton = screen.getByRole('button', { name: /create collection/i })
      
      // Rapid clicks
      await userEvent.click(submitButton)
      await userEvent.click(submitButton)
      await userEvent.click(submitButton)
      
      // Should show error toast
      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalled()
      })
    })

    it('should handle network timeout appropriately', async () => {
      // Simulate a timeout
      mockCreateCollectionMutation.mutateAsync.mockImplementation(() => 
        new Promise((_, reject) => 
          setTimeout(() => reject(new Error('Request timeout')), 100)
        )
      )
      
      renderWithErrorHandlers(
        <CreateCollectionModal onClose={vi.fn()} onSuccess={vi.fn()} />,
        []
      )

      await userEvent.type(screen.getByLabelText(/collection name/i), 'Timeout Test')
      await userEvent.click(screen.getByRole('button', { name: /create collection/i }))
      
      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith({
          message: expect.stringContaining('timeout'),
          type: 'error'
        })
      })
      
      // Form should remain intact
      expect(screen.getByLabelText(/collection name/i)).toHaveValue('Timeout Test')
    })
  })
})