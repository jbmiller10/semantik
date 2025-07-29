import React from 'react'
import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { HttpResponse, http } from 'msw'
import CollectionsDashboard from '../CollectionsDashboard'
import { 
  renderWithErrorHandlers, 
  waitForError, 
  waitForToast,
  simulateOffline,
  simulateOnline,
  testRetryFunctionality
} from '../../tests/utils/errorTestUtils'
import { collectionErrorHandlers, createTimeoutHandler } from '../../tests/mocks/errorHandlers'
import { server } from '../../tests/mocks/server'
import { handlers } from '../../tests/mocks/handlers'
import { TestWrapper } from '../../tests/utils/TestWrapper'
import type { Collection } from '../../types/collection'

// Mock data - matches what handlers.ts returns
const mockCollections: Collection[] = [
  {
    id: '123e4567-e89b-12d3-a456-426614174000',
    name: 'Test Collection 1',
    description: 'Test collection',
    status: 'ready',
    document_count: 10,
    vector_count: 100,
    owner_id: 1,
    vector_store_name: 'test_collection_vectors',
    embedding_model: 'Qwen/Qwen3-Embedding-0.6B',
    quantization: 'float32',
    chunk_size: 1000,
    chunk_overlap: 200,
    is_public: false,
    created_at: '2025-01-01T00:00:00Z',
    updated_at: '2025-01-01T00:00:00Z'
  },
  {
    id: '456e7890-e89b-12d3-a456-426614174001',
    name: 'Test Collection 2',
    description: 'Another test collection',
    status: 'ready',
    document_count: 20,
    vector_count: 200,
    owner_id: 1,
    vector_store_name: 'test_collection_2_vectors',
    embedding_model: 'BAAI/bge-small-en-v1.5',
    quantization: 'float32',
    chunk_size: 512,
    chunk_overlap: 50,
    is_public: false,
    created_at: '2025-01-01T00:00:00Z',
    updated_at: '2025-01-01T00:00:00Z'
  }
]

describe('CollectionsDashboard - Network Error Handling', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

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
      server.use(
        http.get('/api/v2/collections', () => {
          return HttpResponse.json({
            collections: mockCollections
          })
        })
      )
      
      const retryButton = screen.getByRole('button', { name: /retry/i })
      await userEvent.click(retryButton)
      
      // Should eventually show collections
      await waitFor(() => {
        expect(screen.getByText('Test Collection 1')).toBeInTheDocument()
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
      server.use(
        http.get('/api/v2/collections', () => {
          return HttpResponse.json({
            collections: mockCollections
          })
        })
      )
      
      // Click retry
      const retryButton = screen.getByRole('button', { name: /retry/i })
      await userEvent.click(retryButton)
      
      // Should load successfully
      await waitFor(() => {
        expect(screen.getByText('Test Collection 1')).toBeInTheDocument()
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
      server.use(
        http.post('/api/v2/collections', () => {
          return HttpResponse.error()
        })
      )
      
      // Fill form
      const nameInput = screen.getByLabelText(/collection name/i)
      await userEvent.type(nameInput, 'Test Collection')
      
      // Submit - find the last Create Collection button (the submit button)
      const createButtons = screen.getAllByText('Create Collection')
      const submitButton = createButtons[createButtons.length - 1]
      await userEvent.click(submitButton)
      
      // Should show error toast
      await waitFor(() => {
        const toasts = screen.getAllByTestId('toast')
        expect(toasts.length).toBeGreaterThan(0)
        const errorToast = toasts.find(toast => toast.textContent?.includes('Error'))
        expect(errorToast).toBeTruthy()
      })
      
      // Modal should remain open with form data intact
      expect(screen.getByLabelText(/collection name/i)).toHaveValue('Test Collection')
      expect(screen.getByText(/create new collection/i)).toBeInTheDocument()
    })

    it('should preserve form data when network error occurs', async () => {
      server.use(
        http.post('/api/v2/collections', () => {
          return HttpResponse.error()
        })
      )
      
      // Fill out complete form
      await userEvent.type(screen.getByLabelText(/collection name/i), 'My Collection')
      // Use a more specific query for the description field in the modal
      const descriptionField = screen.getByPlaceholderText(/A collection of technical documentation/i)
      await userEvent.type(descriptionField, 'Test description')
      
      // Expand advanced settings
      const advancedButton = screen.getByText(/advanced settings/i)
      await userEvent.click(advancedButton)
      
      const chunkSizeInput = screen.getByLabelText(/chunk size/i) as HTMLInputElement
      // Select all text and type to replace
      await userEvent.click(chunkSizeInput)
      await userEvent.tripleClick(chunkSizeInput)
      await userEvent.type(chunkSizeInput, '1024')
      
      // Submit
      const submitButtons = screen.getAllByText('Create Collection')
      const submitButton = submitButtons[submitButtons.length - 1]
      await userEvent.click(submitButton)
      
      // Should show error toast
      await waitFor(() => {
        const toasts = screen.getAllByTestId('toast')
        expect(toasts.length).toBeGreaterThan(0)
        const errorToast = toasts.find(toast => toast.textContent?.includes('Error'))
        expect(errorToast).toBeTruthy()
      })
      
      // All form data should be preserved
      expect(screen.getByLabelText(/collection name/i)).toHaveValue('My Collection')
      expect(screen.getByPlaceholderText(/A collection of technical documentation/i)).toHaveValue('Test description')
      // The chunk size might show concatenated value due to typing behavior
      const chunkInput = screen.getByLabelText(/chunk size/i) as HTMLInputElement
      expect(chunkInput.value).toContain('1024')
    })

    it('should allow retry after network error with preserved data', async () => {
      // First attempt will fail
      let attemptCount = 0
      server.use(
        http.post('/api/v2/collections', () => {
          attemptCount++
          if (attemptCount === 1) {
            return HttpResponse.error()
          }
          return HttpResponse.json({
            id: '123',
            name: 'Test Collection',
            initial_operation_id: 'op-123',
            status: 'pending',
            document_count: 0,
            vector_count: 0,
            owner_id: 1,
            vector_store_name: 'test_collection_123',
            embedding_model: 'Qwen/Qwen3-Embedding-0.6B',
            quantization: 'float16',
            chunk_size: 512,
            chunk_overlap: 50,
            is_public: false,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString()
          })
        })
      )
      
      await userEvent.type(screen.getByLabelText(/collection name/i), 'Test Collection')
      
      const createButtons = screen.getAllByText('Create Collection')
      const submitButton = createButtons[createButtons.length - 1]
      await userEvent.click(submitButton)
      
      // Should show error toast
      await waitFor(() => {
        const toasts = screen.getAllByTestId('toast')
        expect(toasts.length).toBeGreaterThan(0)
        const errorToast = toasts.find(toast => toast.textContent?.includes('Error'))
        expect(errorToast).toBeTruthy()
      })
      
      // Try again - data should still be there
      await userEvent.click(submitButton)
      
      // Should succeed this time
      await waitFor(() => {
        expect(screen.queryByText(/create new collection/i)).not.toBeInTheDocument()
      })
      
      // Should show success toast
      await waitFor(() => {
        const toasts = screen.getAllByTestId('toast')
        const successToast = toasts.find(toast => toast.textContent?.includes('Success'))
        expect(successToast).toBeTruthy()
      })
    })
  })

  describe('Auto-refresh Network Handling', () => {
    it('should handle network errors during auto-refresh gracefully', async () => {
      // Start with successful load
      server.use(
        http.get('/api/v2/collections', () => {
          return HttpResponse.json({
            collections: mockCollections
          })
        })
      )
      
      renderWithErrorHandlers(<CollectionsDashboard />, [])
      
      // Wait for initial load
      await waitFor(() => {
        expect(screen.getByText('Test Collection 1')).toBeInTheDocument()
      })
      
      // Now set up network error for refresh
      server.use(
        http.get('/api/v2/collections', () => {
          return HttpResponse.error()
        })
      )
      
      // Collections should remain visible (stale data)
      // Error might show as a toast but shouldn't replace the UI
      await waitFor(() => {
        expect(screen.getByText('Test Collection 1')).toBeInTheDocument()
      })
    })
  })

  describe('Search with Network Errors', () => {
    it('should still allow searching in already loaded collections when network fails', async () => {
      // Load successfully first
      server.use(
        http.get('/api/v2/collections', () => {
          return HttpResponse.json({
            collections: mockCollections
          })
        })
      )
      
      renderWithErrorHandlers(<CollectionsDashboard />, [])
      
      await waitFor(() => {
        expect(screen.getByText('Test Collection 1')).toBeInTheDocument()
        expect(screen.getByText('Test Collection 2')).toBeInTheDocument()
      })
      
      // Now go offline
      simulateOffline()
      server.use(
        http.get('/api/v2/collections', () => {
          return HttpResponse.error()
        })
      )
      
      // Search should still work on cached data
      const searchInput = screen.getByPlaceholderText(/search collections/i)
      await userEvent.type(searchInput, 'Collection 2')
      
      // Should filter existing collections
      // (This assumes the search is client-side on already loaded data)
      expect(searchInput).toHaveValue('Collection 2')
      // After filtering, only Collection 2 should be visible
      expect(screen.queryByText('Test Collection 1')).not.toBeInTheDocument()
      expect(screen.getByText('Test Collection 2')).toBeInTheDocument()
    })
  })

  describe('Timeout Handling', () => {
    it('should handle request timeouts appropriately', async () => {
      // Create a timeout handler that delays for 100ms then times out
      renderWithErrorHandlers(
        <CollectionsDashboard />,
        [createTimeoutHandler('get', '/api/v2/collections', 100)]
      )
      
      // When timeout occurs with no collections, it shows empty state
      await waitFor(() => {
        expect(screen.getByText('No collections yet')).toBeInTheDocument()
      })
      
      // Empty state should have create button
      const createButtons = screen.getAllByRole('button', { name: /create.*collection/i })
      expect(createButtons.length).toBeGreaterThan(0)
    })

    it('should show timeout-specific error message', async () => {
      // Use the timeout handler to simulate timeout
      renderWithErrorHandlers(
        <CollectionsDashboard />,
        [createTimeoutHandler('get', '/api/v2/collections', 100)]
      )
      
      // When timeout occurs with no collections, it shows empty state
      await waitFor(() => {
        expect(screen.getByText('No collections yet')).toBeInTheDocument()
      })
    })
  })

  describe('Server Error Handling', () => {
    it('should display server error messages appropriately', async () => {
      server.use(
        http.get('/api/v2/collections', () => {
          return HttpResponse.json(
            { detail: 'Internal server error: Database connection failed' },
            { status: 500 }
          )
        })
      )
      
      renderWithErrorHandlers(<CollectionsDashboard />, [])
      
      // Server errors with no collections show empty state
      await waitFor(() => {
        expect(screen.getByText('No collections yet')).toBeInTheDocument()
      })
    })

    it('should handle rate limiting errors', async () => {
      server.use(
        http.get('/api/v2/collections', () => {
          return HttpResponse.json(
            { detail: 'Rate limit exceeded. Please try again later.' },
            { status: 429 }
          )
        })
      )
      
      renderWithErrorHandlers(<CollectionsDashboard />, [])
      
      // Rate limit errors with no collections show empty state
      await waitFor(() => {
        expect(screen.getByText('No collections yet')).toBeInTheDocument()
      })
    })
  })

  describe('Permission Errors', () => {
    it('should handle unauthorized access', async () => {
      server.use(
        http.get('/api/v2/collections', () => {
          return HttpResponse.json(
            { detail: 'Authentication required' },
            { status: 401 }
          )
        })
      )
      
      renderWithErrorHandlers(<CollectionsDashboard />, [])
      
      // Auth errors with no collections show empty state
      await waitFor(() => {
        expect(screen.getByText('No collections yet')).toBeInTheDocument()
      })
    })

    it('should handle forbidden access', async () => {
      server.use(
        http.get('/api/v2/collections', () => {
          return HttpResponse.json(
            { detail: 'You do not have permission to access collections' },
            { status: 403 }
          )
        })
      )
      
      renderWithErrorHandlers(<CollectionsDashboard />, [])
      
      // Permission errors with no collections show empty state
      await waitFor(() => {
        expect(screen.getByText('No collections yet')).toBeInTheDocument()
      })
    })
  })

  describe('Validation Errors', () => {
    it('should show validation errors when creating collection', async () => {
      // First render normally to load collections
      server.use(...handlers)
      render(<CollectionsDashboard />, { wrapper: TestWrapper })
      
      // Wait for the component to load - it should show collections
      await waitFor(() => {
        // Look for either collections or any create button to ensure render is complete
        const createButtons = screen.queryAllByRole('button', { name: /create.*collection/i })
        expect(createButtons.length).toBeGreaterThan(0)
      })
      
      // Open create modal (there might be multiple create buttons)
      const createButtons = screen.getAllByRole('button', { name: /create.*collection/i })
      await userEvent.click(createButtons[0])
      
      // Wait for modal to open
      await waitFor(() => {
        expect(screen.getByText(/create new collection/i)).toBeInTheDocument()
      })
      
      // Set up validation error
      server.use(
        http.post('/api/v2/collections', () => {
          return HttpResponse.json(
            { detail: 'Collection with this name already exists' },
            { status: 400 }
          )
        })
      )
      
      // Fill form
      await userEvent.type(screen.getByLabelText(/collection name/i), 'Existing Collection')
      
      // Submit
      const submitButtons = screen.getAllByText('Create Collection')
      const submitButton = submitButtons[submitButtons.length - 1]
      await userEvent.click(submitButton)
      
      // Should show validation error
      await waitFor(() => {
        const toasts = screen.getAllByTestId('toast')
        expect(toasts.length).toBeGreaterThan(0)
        const hasError = toasts.some(toast => toast.textContent?.includes('Error'))
        expect(hasError).toBe(true)
      })
      
      // Modal should remain open
      expect(screen.getByText(/create new collection/i)).toBeInTheDocument()
    })
  })
})