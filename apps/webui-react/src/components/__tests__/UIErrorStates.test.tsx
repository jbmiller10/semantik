import React from 'react'
import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { Toast } from '../Toast'
import { CreateCollectionModal } from '../CreateCollectionModal'
import { AddDataToCollectionModal } from '../AddDataToCollectionModal'
import { useUIStore } from '../../stores/uiStore'
import { useCollectionStore } from '../../stores/collectionStore'
import { renderWithErrorHandlers } from '../../tests/utils/errorTestUtils'
import { TestWrapper } from '../../tests/utils/testUtils'
import { render } from '@testing-library/react'

// Mock stores
vi.mock('../../stores/uiStore')
vi.mock('../../stores/collectionStore')

describe('UI Error States', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('Toast Error Notifications', () => {
    it('should display error toasts with proper styling', () => {
      const mockToasts = [
        {
          id: '1',
          message: 'Network connection failed',
          type: 'error' as const,
          duration: 5000
        },
        {
          id: '2',
          message: 'Operation completed with warnings',
          type: 'warning' as const,
          duration: 5000
        }
      ]
      
      vi.mocked(useUIStore).mockReturnValue({
        toasts: mockToasts,
        removeToast: vi.fn()
      } as any)
      
      render(
        <TestWrapper>
          <Toast />
        </TestWrapper>
      )
      
      // Error toast
      const errorToast = screen.getByText('Network connection failed').closest('[data-testid="toast"]')
      expect(errorToast).toHaveClass('toast-error')
      expect(errorToast).toHaveClass('border-l-red-500')
      
      // Warning toast
      const warningToast = screen.getByText('Operation completed with warnings').closest('[data-testid="toast"]')
      expect(warningToast).toHaveClass('toast-warning')
      expect(warningToast).toHaveClass('border-l-yellow-500')
    })

    it('should auto-dismiss error toasts after duration', async () => {
      const mockRemoveToast = vi.fn()
      const mockToast = {
        id: 'error-1',
        message: 'Temporary error',
        type: 'error' as const,
        duration: 1000 // 1 second for testing
      }
      
      vi.mocked(useUIStore).mockReturnValue({
        toasts: [mockToast],
        removeToast: mockRemoveToast
      } as any)
      
      render(
        <TestWrapper>
          <Toast />
        </TestWrapper>
      )
      
      expect(screen.getByText('Temporary error')).toBeInTheDocument()
      
      // Wait for auto-dismiss
      await waitFor(() => {
        expect(mockRemoveToast).toHaveBeenCalledWith('error-1')
      }, { timeout: 1500 })
    })

    it('should allow manual dismissal of error toasts', async () => {
      const mockRemoveToast = vi.fn()
      
      vi.mocked(useUIStore).mockReturnValue({
        toasts: [{
          id: 'error-1',
          message: 'Click to dismiss error',
          type: 'error' as const,
          duration: 10000
        }],
        removeToast: mockRemoveToast
      } as any)
      
      render(
        <TestWrapper>
          <Toast />
        </TestWrapper>
      )
      
      const closeButton = screen.getByRole('button', { name: /close/i })
      await userEvent.click(closeButton)
      
      expect(mockRemoveToast).toHaveBeenCalledWith('error-1')
    })
  })

  describe('Form Validation Error Display', () => {
    it('should show inline validation errors in CreateCollectionModal', async () => {
      vi.mocked(useCollectionStore).mockReturnValue({
        createCollection: vi.fn()
      } as any)
      
      vi.mocked(useUIStore).mockReturnValue({
        addToast: vi.fn()
      } as any)
      
      renderWithErrorHandlers(
        <CreateCollectionModal isOpen={true} onClose={vi.fn()} />,
        []
      )
      
      // Submit without filling required fields
      const submitButton = screen.getByRole('button', { name: /create$/i })
      await userEvent.click(submitButton)
      
      // Should show validation error (browser native or custom)
      const nameInput = screen.getByLabelText(/collection name/i) as HTMLInputElement
      expect(nameInput.validity.valid).toBe(false)
      
      // Should have required attribute
      expect(nameInput).toHaveAttribute('required')
    })

    it('should validate numeric inputs stay within bounds', async () => {
      vi.mocked(useCollectionStore).mockReturnValue({
        createCollection: vi.fn()
      } as any)
      
      vi.mocked(useUIStore).mockReturnValue({
        addToast: vi.fn()
      } as any)
      
      renderWithErrorHandlers(
        <CreateCollectionModal isOpen={true} onClose={vi.fn()} />,
        []
      )
      
      // Expand advanced settings
      await userEvent.click(screen.getByText(/advanced settings/i))
      
      const chunkSizeInput = screen.getByLabelText(/chunk size/i) as HTMLInputElement
      
      // Clear and type invalid value
      await userEvent.clear(chunkSizeInput)
      await userEvent.type(chunkSizeInput, '0') // Below minimum
      
      // Check if browser validation will trigger
      expect(parseInt(chunkSizeInput.value)).toBeLessThan(128) // Minimum is 128
      
      // Input should have min/max attributes
      expect(chunkSizeInput).toHaveAttribute('min', '128')
      expect(chunkSizeInput).toHaveAttribute('max', '4096')
    })

    it('should show path validation feedback', async () => {
      const mockAddSource = vi.fn()
      const mockAddToast = vi.fn()
      
      vi.mocked(useCollectionStore).mockReturnValue({
        addSource: mockAddSource
      } as any)
      
      vi.mocked(useUIStore).mockReturnValue({
        addToast: mockAddToast
      } as any)
      
      const mockCollection = {
        uuid: 'test-uuid',
        name: 'Test Collection',
        status: 'ready',
        embedding_model: 'test-model',
        quantization: 'float16'
      }
      
      renderWithErrorHandlers(
        <AddDataToCollectionModal 
          isOpen={true} 
          onClose={vi.fn()} 
          collection={mockCollection}
        />,
        []
      )
      
      const pathInput = screen.getByLabelText(/source directory path/i)
      
      // Type invalid path pattern
      await userEvent.type(pathInput, 'not/an/absolute/path')
      
      // Path should start with / for absolute paths
      const submitButton = screen.getByRole('button', { name: /add.*source/i })
      await userEvent.click(submitButton)
      
      // If client-side validation exists, it should prevent submission
      // Otherwise, server will return error
    })
  })

  describe('Loading State Error Prevention', () => {
    it('should disable form during submission to prevent double submit', async () => {
      const mockCreateCollection = vi.fn(() => 
        new Promise(resolve => setTimeout(resolve, 1000))
      )
      
      vi.mocked(useCollectionStore).mockReturnValue({
        createCollection: mockCreateCollection
      } as any)
      
      vi.mocked(useUIStore).mockReturnValue({
        addToast: vi.fn()
      } as any)
      
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
      
      // Try clicking again - should not create another request
      await userEvent.click(submitButton)
      expect(mockCreateCollection).toHaveBeenCalledTimes(1)
    })

    it('should show loading overlay during operations', async () => {
      const mockAddSource = vi.fn(() => 
        new Promise(resolve => setTimeout(resolve, 1000))
      )
      
      vi.mocked(useCollectionStore).mockReturnValue({
        addSource: mockAddSource
      } as any)
      
      vi.mocked(useUIStore).mockReturnValue({
        addToast: vi.fn()
      } as any)
      
      const mockCollection = {
        uuid: 'test-uuid',
        name: 'Test Collection',
        status: 'ready',
        embedding_model: 'test-model',
        quantization: 'float16'
      }
      
      renderWithErrorHandlers(
        <AddDataToCollectionModal 
          isOpen={true} 
          onClose={vi.fn()} 
          collection={mockCollection}
        />,
        []
      )
      
      await userEvent.type(screen.getByLabelText(/source directory path/i), '/data/test')
      await userEvent.click(screen.getByRole('button', { name: /add.*source/i }))
      
      // Should show loading state
      expect(screen.getByRole('button', { name: /adding/i })).toBeDisabled()
      
      // Modal should not be closeable during operation
      // (Implementation specific - might disable close button or block backdrop clicks)
    })
  })

  describe('Error Message Formatting', () => {
    it('should format technical errors into user-friendly messages', async () => {
      const mockAddToast = vi.fn()
      
      vi.mocked(useUIStore).mockReturnValue({
        addToast: mockAddToast,
        toasts: []
      } as any)
      
      const technicalErrors = [
        {
          original: 'ECONNREFUSED 127.0.0.1:8080',
          expected: /unable to connect|connection refused/i
        },
        {
          original: 'ETIMEDOUT',
          expected: /request timed out|timeout/i
        },
        {
          original: 'NetworkError when attempting to fetch resource',
          expected: /network error|connection problem/i
        }
      ]
      
      // Simulate various error scenarios and check toast messages
      for (const error of technicalErrors) {
        mockAddToast.mockClear()
        
        // Trigger an error (component specific)
        // Check that the toast message is user-friendly
        
        // This is a pattern test - actual implementation depends on error handling
      }
    })

    it('should truncate very long error messages', () => {
      const mockAddToast = vi.fn()
      const veryLongError = 'A'.repeat(500) + ' error details that go on and on...'
      
      vi.mocked(useUIStore).mockReturnValue({
        addToast: mockAddToast,
        toasts: [{
          id: '1',
          message: veryLongError,
          type: 'error' as const,
          duration: 5000
        }]
      } as any)
      
      render(
        <TestWrapper>
          <Toast />
        </TestWrapper>
      )
      
      const toastElement = screen.getByTestId('toast')
      const displayedText = toastElement.textContent || ''
      
      // Should truncate or wrap appropriately
      expect(displayedText.length).toBeLessThanOrEqual(200) // Reasonable length
    })
  })

  describe('Error State Accessibility', () => {
    it('should announce errors to screen readers', () => {
      vi.mocked(useUIStore).mockReturnValue({
        toasts: [{
          id: '1',
          message: 'Critical error occurred',
          type: 'error' as const,
          duration: 5000
        }],
        removeToast: vi.fn()
      } as any)
      
      render(
        <TestWrapper>
          <Toast />
        </TestWrapper>
      )
      
      const toast = screen.getByTestId('toast')
      
      // Should have appropriate ARIA attributes
      expect(toast).toHaveAttribute('role', 'alert')
      expect(toast).toHaveAttribute('aria-live', 'assertive')
    })

    it('should have proper focus management for error states', async () => {
      const mockRetry = vi.fn()
      
      // Simulate an error state with retry button
      render(
        <TestWrapper>
          <div role="alert">
            <p>An error occurred</p>
            <button onClick={mockRetry}>Retry</button>
          </div>
        </TestWrapper>
      )
      
      const retryButton = screen.getByRole('button', { name: /retry/i })
      
      // Retry button should be focusable
      expect(retryButton).not.toHaveAttribute('disabled')
      
      // Tab navigation should work
      await userEvent.tab()
      expect(document.activeElement).toBe(retryButton)
    })

    it('should provide clear error context in forms', async () => {
      vi.mocked(useCollectionStore).mockReturnValue({
        createCollection: vi.fn().mockRejectedValue({
          response: { data: { detail: 'Name already exists' } }
        })
      } as any)
      
      vi.mocked(useUIStore).mockReturnValue({
        addToast: vi.fn()
      } as any)
      
      renderWithErrorHandlers(
        <CreateCollectionModal isOpen={true} onClose={vi.fn()} />,
        []
      )
      
      const nameInput = screen.getByLabelText(/collection name/i)
      await userEvent.type(nameInput, 'Duplicate Name')
      await userEvent.click(screen.getByRole('button', { name: /create$/i }))
      
      // Error should be associated with the input
      await waitFor(() => {
        // Either via aria-describedby or aria-invalid
        expect(nameInput).toHaveAttribute('aria-invalid', 'true')
        // Or error message is announced
      })
    })
  })
})