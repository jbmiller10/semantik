import React from 'react'
import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { CreateCollectionModal } from '../CreateCollectionModal'
import { useCollectionStore } from '../../stores/collectionStore'
import { useUIStore } from '../../stores/uiStore'
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

// Mock the stores
vi.mock('../../stores/collectionStore')
vi.mock('../../stores/uiStore')

describe('CreateCollectionModal - Network Error Handling', () => {
  const mockCreateCollection = vi.fn()
  const mockAddToast = vi.fn()
  
  beforeEach(() => {
    vi.clearAllMocks()
    
    vi.mocked(useCollectionStore).mockReturnValue({
      createCollection: mockCreateCollection,
    } as any)
    
    vi.mocked(useUIStore).mockReturnValue({
      addToast: mockAddToast,
    } as any)
  })

  describe('Submission Network Failures', () => {
    it('should show error toast on network failure', async () => {
      mockCreateCollection.mockRejectedValue(new Error('Network error'))
      
      renderWithErrorHandlers(
        <CreateCollectionModal isOpen={true} onClose={vi.fn()} />,
        collectionErrorHandlers.networkError()
      )

      // Fill minimum required fields
      await userEvent.type(screen.getByLabelText(/collection name/i), 'Test Collection')
      
      // Submit
      const submitButton = screen.getByRole('button', { name: /create$/i })
      await userEvent.click(submitButton)
      
      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith(
          expect.stringContaining('Network error'),
          'error'
        )
      })
      
      // Modal should remain open
      expect(screen.getByRole('dialog')).toBeInTheDocument()
    })

    it('should preserve all form data after network error', async () => {
      mockCreateCollection.mockRejectedValue(new Error('Network error'))
      
      renderWithErrorHandlers(
        <CreateCollectionModal isOpen={true} onClose={vi.fn()} />,
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
      await userEvent.clear(screen.getByLabelText(/chunk size/i))
      await userEvent.type(screen.getByLabelText(/chunk size/i), '1024')
      await userEvent.clear(screen.getByLabelText(/chunk overlap/i))
      await userEvent.type(screen.getByLabelText(/chunk overlap/i), '100')
      
      // Add initial source
      await userEvent.type(screen.getByLabelText(/initial source/i), '/data/documents')
      
      // Submit
      await userEvent.click(screen.getByRole('button', { name: /create$/i }))
      
      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith(
          expect.stringContaining('Network error'),
          'error'
        )
      })
      
      // All data should be preserved
      expectFormDataPreserved(formData)
      expect(screen.getByLabelText(/chunk size/i)).toHaveValue('1024')
      expect(screen.getByLabelText(/chunk overlap/i)).toHaveValue('100')
      expect(screen.getByLabelText(/initial source/i)).toHaveValue('/data/documents')
    })

    it('should handle offline scenario gracefully', async () => {
      const onClose = vi.fn()
      
      renderWithErrorHandlers(
        <CreateCollectionModal isOpen={true} onClose={onClose} />,
        []
      )

      // Go offline before submission
      simulateOffline()
      mockCreateCollection.mockRejectedValue(new Error('Network error: Unable to create collection'))
      
      await userEvent.type(screen.getByLabelText(/collection name/i), 'Offline Collection')
      await userEvent.click(screen.getByRole('button', { name: /create$/i }))
      
      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith(
          expect.stringContaining('Network error'),
          'error'
        )
      })
      
      // Modal stays open, form data preserved
      expect(onClose).not.toHaveBeenCalled()
      expect(screen.getByLabelText(/collection name/i)).toHaveValue('Offline Collection')
      
      // Go back online and retry
      simulateOnline()
      mockCreateCollection.mockResolvedValue({ uuid: 'test-uuid' } as any)
      
      await userEvent.click(screen.getByRole('button', { name: /create$/i }))
      
      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith(
          'Collection created successfully',
          'success'
        )
      })
      
      expect(onClose).toHaveBeenCalled()
    })

    it('should disable submit button during network request', async () => {
      // Mock a slow network request
      mockCreateCollection.mockImplementation(() => 
        new Promise(resolve => setTimeout(() => resolve({ uuid: 'test' } as any), 1000))
      )
      
      renderWithErrorHandlers(
        <CreateCollectionModal isOpen={true} onClose={vi.fn()} />,
        []
      )

      await userEvent.type(screen.getByLabelText(/collection name/i), 'Test')
      
      const submitButton = screen.getByRole('button', { name: /create$/i })
      await userEvent.click(submitButton)
      
      // Button should be disabled immediately
      expect(submitButton).toBeDisabled()
      expect(submitButton).toHaveTextContent(/creating/i)
      
      // Wait for completion
      await waitFor(() => {
        expect(submitButton).not.toBeDisabled()
      })
    })
  })

  describe('Two-step Creation Process Network Errors', () => {
    it('should handle failure in add-source step after successful collection creation', async () => {
      const mockAddSource = vi.fn()
      vi.mocked(useCollectionStore).mockReturnValue({
        createCollection: mockCreateCollection,
        addSource: mockAddSource,
      } as any)
      
      // First call succeeds, second fails
      mockCreateCollection.mockResolvedValue({ uuid: 'test-uuid', name: 'Test' } as any)
      mockAddSource.mockRejectedValue(new Error('Network error during add source'))
      
      renderWithErrorHandlers(
        <CreateCollectionModal isOpen={true} onClose={vi.fn()} />,
        []
      )

      await userEvent.type(screen.getByLabelText(/collection name/i), 'Test Collection')
      await userEvent.type(screen.getByLabelText(/initial source/i), '/data/docs')
      
      await userEvent.click(screen.getByRole('button', { name: /create$/i }))
      
      // Should show both toasts
      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith(
          'Collection created successfully',
          'success'
        )
        expect(mockAddToast).toHaveBeenCalledWith(
          expect.stringContaining('created but failed to add initial source'),
          'warning'
        )
      })
    })

    it('should handle network recovery between creation and add-source', async () => {
      const mockAddSource = vi.fn()
      vi.mocked(useCollectionStore).mockReturnValue({
        createCollection: mockCreateCollection,
        addSource: mockAddSource,
      } as any)
      
      mockCreateCollection.mockResolvedValue({ uuid: 'test-uuid', name: 'Test' } as any)
      
      // First attempt at add-source fails
      mockAddSource.mockRejectedValueOnce(new Error('Network error'))
      // But component doesn't retry automatically, so this test just verifies the warning
      
      renderWithErrorHandlers(
        <CreateCollectionModal isOpen={true} onClose={vi.fn()} />,
        []
      )

      await userEvent.type(screen.getByLabelText(/collection name/i), 'Test Collection')
      await userEvent.type(screen.getByLabelText(/initial source/i), '/data/docs')
      
      await userEvent.click(screen.getByRole('button', { name: /create$/i }))
      
      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith(
          expect.stringContaining('created but failed to add initial source'),
          'warning'
        )
      })
    })
  })

  describe('Intermittent Network Issues', () => {
    it('should handle rapid submit attempts during network issues', async () => {
      mockCreateCollection.mockRejectedValue(new Error('Network error'))
      
      renderWithErrorHandlers(
        <CreateCollectionModal isOpen={true} onClose={vi.fn()} />,
        collectionErrorHandlers.networkError()
      )

      await userEvent.type(screen.getByLabelText(/collection name/i), 'Test')
      
      const submitButton = screen.getByRole('button', { name: /create$/i })
      
      // Rapid clicks
      await userEvent.click(submitButton)
      await userEvent.click(submitButton)
      await userEvent.click(submitButton)
      
      // Should only show one error toast, not multiple
      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledTimes(1)
      })
    })

    it('should handle network timeout appropriately', async () => {
      // Simulate a timeout
      mockCreateCollection.mockImplementation(() => 
        new Promise((_, reject) => 
          setTimeout(() => reject(new Error('Request timeout')), 100)
        )
      )
      
      renderWithErrorHandlers(
        <CreateCollectionModal isOpen={true} onClose={vi.fn()} />,
        []
      )

      await userEvent.type(screen.getByLabelText(/collection name/i), 'Timeout Test')
      await userEvent.click(screen.getByRole('button', { name: /create$/i }))
      
      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith(
          expect.stringContaining('timeout'),
          'error'
        )
      })
      
      // Form should remain intact
      expect(screen.getByLabelText(/collection name/i)).toHaveValue('Timeout Test')
    })
  })
})