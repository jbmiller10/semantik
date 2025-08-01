import React from 'react'
import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { App } from '../../App'
import { server } from '../mocks/server'
import { 
  collectionErrorHandlers,
  searchErrorHandlers,
  authErrorHandlers,
  createSlowResponseHandler
} from '../mocks/errorHandlers'
import { handlers } from '../mocks/handlers'
import { 
  simulateOffline,
  simulateOnline,
  mockWebSocket
} from '../utils/errorTestUtils'
import { render } from '@testing-library/react'

describe('E2E Error Flows', () => {
  let mockWs: { restore: () => void }

  beforeEach(() => {
    // Clear all storage
    localStorage.clear()
    sessionStorage.clear()
    
    // Mock WebSocket
    mockWs = mockWebSocket()
    
    // Set up default handlers
    server.use(...handlers)
  })

  afterEach(() => {
    mockWs.restore()
  })

  describe('Complete Network Failure Recovery Flow', () => {
    it('should handle network failure during collection creation and recovery', async () => {
      const user = userEvent.setup()
      
      // Render app
      render(<App />)
      
      // Wait for initial load
      await waitFor(() => {
        expect(screen.getByText(/collections/i)).toBeInTheDocument()
      })
      
      // Open create collection modal
      const createButton = await screen.findByRole('button', { name: /create.*collection/i })
      await user.click(createButton)
      
      // Fill form
      const modal = screen.getByRole('dialog')
      const nameInput = within(modal).getByLabelText(/collection name/i)
      await user.type(nameInput, 'Network Test Collection')
      
      const descInput = within(modal).getByLabelText(/description/i)
      await user.type(descInput, 'Testing network failure recovery')
      
      // Simulate network going offline
      simulateOffline()
      server.use(...collectionErrorHandlers.networkError())
      
      // Try to create
      const submitButton = within(modal).getByRole('button', { name: /create$/i })
      await user.click(submitButton)
      
      // Should show network error
      await waitFor(() => {
        expect(screen.getByText(/network error/i)).toBeInTheDocument()
      })
      
      // Form data should be preserved
      expect(nameInput).toHaveValue('Network Test Collection')
      expect(descInput).toHaveValue('Testing network failure recovery')
      
      // Go back online
      simulateOnline()
      server.use(...handlers)
      
      // Retry submission
      await user.click(submitButton)
      
      // Should succeed
      await waitFor(() => {
        expect(screen.getByText(/collection created successfully/i)).toBeInTheDocument()
      })
      
      // Modal should close
      expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
      
      // New collection should appear
      expect(screen.getByText('Network Test Collection')).toBeInTheDocument()
    })

    it('should handle search with partial collection failures', async () => {
      const user = userEvent.setup()
      
      render(<App />)
      
      // Navigate to search
      const searchTab = await screen.findByRole('tab', { name: /search/i })
      await user.click(searchTab)
      
      // Wait for collections to load
      await waitFor(() => {
        expect(screen.getByText(/select collections/i)).toBeInTheDocument()
      })
      
      // Select collections
      const multiSelect = screen.getByText(/select collections/i)
      await user.click(multiSelect)
      
      const selectAll = await screen.findByRole('button', { name: /select all/i })
      await user.click(selectAll)
      
      // Click outside to close dropdown
      await user.click(document.body)
      
      // Enter search query
      const searchInput = screen.getByPlaceholderText(/search across your collections/i)
      await user.type(searchInput, 'important documents')
      
      // Set up partial failure
      server.use(...searchErrorHandlers.partialFailure())
      
      // Search
      const searchButton = screen.getByRole('button', { name: /search/i })
      await user.click(searchButton)
      
      // Should show partial failure warning
      await waitFor(() => {
        const alert = screen.getByRole('alert')
        expect(alert).toHaveTextContent(/search completed with errors/i)
        expect(alert).toHaveTextContent(/Failed Collection/)
        expect(alert).toHaveTextContent(/Vector index corrupted/)
      })
      
      // But should still show successful results
      expect(screen.getByText('Result from model A')).toBeInTheDocument()
      
      // User can acknowledge and continue
      // Search functionality remains available
      expect(searchInput).not.toBeDisabled()
      expect(searchButton).not.toBeDisabled()
    })
  })

  describe('Authentication Error Recovery Flow', () => {
    it('should handle token expiry during active session', async () => {
      const user = userEvent.setup()
      
      // Set up initial auth
      localStorage.setItem('access_token', 'valid-token')
      localStorage.setItem('refresh_token', 'valid-refresh')
      
      render(<App />)
      
      // User is working normally
      await waitFor(() => {
        expect(screen.getByText(/collections/i)).toBeInTheDocument()
      })
      
      // Open create modal
      const createButton = await screen.findByRole('button', { name: /create.*collection/i })
      await user.click(createButton)
      
      // Fill form
      await user.type(screen.getByLabelText(/collection name/i), 'Auth Test')
      
      // Simulate token expiry
      server.use(...authErrorHandlers.unauthorized())
      
      // Try to create
      await user.click(screen.getByRole('button', { name: /create$/i }))
      
      // Should redirect to login
      await waitFor(() => {
        expect(window.location.pathname).toBe('/login')
      })
      
      // Tokens should be cleared
      expect(localStorage.getItem('access_token')).toBeNull()
      expect(localStorage.getItem('refresh_token')).toBeNull()
      
      // After re-login, user should return to collections
      // (This would be tested with actual login flow)
    })

    it('should handle permission denied gracefully', async () => {
      const user = userEvent.setup()
      
      render(<App />)
      
      // User has some collections
      await waitFor(() => {
        expect(screen.getByText(/collections/i)).toBeInTheDocument()
      })
      
      // Try to access another user's collection via URL
      // (In real app, this would be navigation)
      server.use(...collectionErrorHandlers.permissionError())
      
      // Simulate clicking on a shared collection that user lost access to
      const restrictedCollection = await screen.findByText('Restricted Collection')
      await user.click(restrictedCollection)
      
      // Should show permission error
      await waitFor(() => {
        expect(screen.getByText(/do not have permission/i)).toBeInTheDocument()
      })
      
      // User should be able to go back to their collections
      const backButton = screen.getByRole('button', { name: /back/i })
      await user.click(backButton)
      
      // Should be back at collections list
      expect(screen.getByText(/collections/i)).toBeInTheDocument()
    })
  })

  describe('Operation Error and Recovery Flow', () => {
    it('should handle operation failure with retry', async () => {
      const user = userEvent.setup()
      
      render(<App />)
      
      // Create a collection first
      await waitFor(() => {
        expect(screen.getByText(/collections/i)).toBeInTheDocument()
      })
      
      // Open add data modal for existing collection
      const collection = await screen.findByText('Test Collection')
      await user.click(collection)
      
      // In details modal, click add source
      const addSourceButton = await screen.findByRole('button', { name: /add.*source/i })
      await user.click(addSourceButton)
      
      // Fill path
      await user.type(screen.getByLabelText(/source directory/i), '/data/large-dataset')
      
      // First attempt fails
      server.use(
        collectionErrorHandlers.createErrorHandler('post', '/api/v2/collections/:uuid/add-source', 503, {
          detail: 'Insufficient resources available. Please try again later.'
        })
      )
      
      await user.click(screen.getByRole('button', { name: /add.*source/i }))
      
      // Should show error
      await waitFor(() => {
        expect(screen.getByText(/insufficient resources/i)).toBeInTheDocument()
      })
      
      // Wait a moment and retry
      await new Promise(resolve => setTimeout(resolve, 1000))
      
      // Now resources are available
      server.use(...handlers)
      
      await user.click(screen.getByRole('button', { name: /add.*source/i }))
      
      // Should succeed
      await waitFor(() => {
        expect(screen.getByText(/source added successfully/i)).toBeInTheDocument()
      })
      
      // Should navigate to show operation progress
      expect(screen.getByText(/processing documents/i)).toBeInTheDocument()
    })

    it('should handle WebSocket disconnection during operation monitoring', async () => {
      const user = userEvent.setup()
      
      render(<App />)
      
      // Navigate to active operations
      const operationsTab = await screen.findByRole('tab', { name: /active operations/i })
      await user.click(operationsTab)
      
      // Should show active operations with live indicators
      await waitFor(() => {
        expect(screen.getByText(/processing documents/i)).toBeInTheDocument()
        expect(screen.getByText(/live/i)).toBeInTheDocument()
      })
      
      // Simulate WebSocket disconnection
      // (Mock WebSocket will handle this)
      
      // Live indicator should disappear but operation info remains
      await waitFor(() => {
        expect(screen.queryByText(/live/i)).not.toBeInTheDocument()
      })
      
      // Progress should still update via polling
      expect(screen.getByText(/processing documents/i)).toBeInTheDocument()
      
      // Operation completes successfully despite WebSocket issues
      await waitFor(() => {
        expect(screen.queryByText(/processing documents/i)).not.toBeInTheDocument()
        expect(screen.getByText(/no active operations/i)).toBeInTheDocument()
      }, { timeout: 10000 })
    })
  })

  describe('Cascading Error Recovery', () => {
    it('should handle multiple simultaneous errors gracefully', async () => {
      const user = userEvent.setup()
      
      render(<App />)
      
      // Set up multiple error conditions
      server.use(
        // Collections endpoint is slow
        createSlowResponseHandler('get', '/api/v2/collections', 2000, { collections: [] }),
        // Search endpoint fails
        searchErrorHandlers.serverError()[0],
        // Operations endpoint fails
        collectionErrorHandlers.createErrorHandler('get', '/api/v2/operations', 500)
      )
      
      // App should load with degraded functionality
      await waitFor(() => {
        expect(screen.getByText(/loading collections/i)).toBeInTheDocument()
      })
      
      // Try to navigate to search while collections still loading
      const searchTab = screen.getByRole('tab', { name: /search/i })
      await user.click(searchTab)
      
      // Search should show appropriate state
      expect(screen.getByText(/loading collections/i)).toBeInTheDocument()
      
      // Collections eventually load
      await waitFor(() => {
        expect(screen.queryByText(/loading collections/i)).not.toBeInTheDocument()
      }, { timeout: 3000 })
      
      // Try to search - should fail
      await user.type(screen.getByPlaceholderText(/search/i), 'test')
      await user.click(screen.getByRole('button', { name: /search/i }))
      
      await waitFor(() => {
        expect(screen.getByText(/search service unavailable/i)).toBeInTheDocument()
      })
      
      // Navigate to operations - should show error
      const operationsTab = screen.getByRole('tab', { name: /active operations/i })
      await user.click(operationsTab)
      
      await waitFor(() => {
        expect(screen.getByText(/failed to load operations/i)).toBeInTheDocument()
      })
      
      // Fix services one by one
      server.use(...handlers)
      
      // Retry operations
      await user.click(screen.getByRole('button', { name: /retry/i }))
      
      await waitFor(() => {
        expect(screen.queryByText(/failed to load operations/i)).not.toBeInTheDocument()
      })
      
      // App recovers to full functionality
    })

    it('should maintain data consistency during error recovery', async () => {
      const user = userEvent.setup()
      
      render(<App />)
      
      // User is creating a collection with initial source
      const createButton = await screen.findByRole('button', { name: /create.*collection/i })
      await user.click(createButton)
      
      await user.type(screen.getByLabelText(/collection name/i), 'Consistency Test')
      await user.type(screen.getByLabelText(/initial source/i), '/data/documents')
      
      // First step succeeds (create collection)
      // Second step fails (add source)
      let callCount = 0
      server.use(
        collectionErrorHandlers.createErrorHandler('post', '/api/v2/collections', 201, {
          uuid: 'new-collection-id',
          name: 'Consistency Test'
        }),
        {
          handler: collectionErrorHandlers.createNetworkErrorHandler('post', '/api/v2/collections/:uuid/add-source'),
          predicate: () => ++callCount === 1
        }
      )
      
      await user.click(screen.getByRole('button', { name: /create$/i }))
      
      // Should show partial success
      await waitFor(() => {
        expect(screen.getByText(/collection created successfully/i)).toBeInTheDocument()
        expect(screen.getByText(/failed to add initial source/i)).toBeInTheDocument()
      })
      
      // Collection should exist but without source
      expect(screen.getByText('Consistency Test')).toBeInTheDocument()
      
      // User can manually add source later
      await user.click(screen.getByText('Consistency Test'))
      
      const addSourceButton = await screen.findByRole('button', { name: /add.*source/i })
      expect(addSourceButton).toBeInTheDocument()
    })
  })

  describe('Performance Under Error Conditions', () => {
    it('should remain responsive during multiple retry attempts', async () => {
      const user = userEvent.setup()
      
      render(<App />)
      
      // Set up persistent failures
      server.use(...collectionErrorHandlers.networkError())
      
      // Multiple rapid retry attempts
      for (let i = 0; i < 5; i++) {
        const retryButton = await screen.findByRole('button', { name: /retry/i })
        await user.click(retryButton)
        
        // UI should remain responsive
        expect(screen.getByText(/loading/i)).toBeInTheDocument()
        
        await waitFor(() => {
          expect(screen.getByText(/failed to load/i)).toBeInTheDocument()
        })
      }
      
      // App should not crash or become unresponsive
      expect(screen.getByRole('button', { name: /retry/i })).toBeInTheDocument()
    })
  })
})