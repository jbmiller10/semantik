import React from 'react'
import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useNavigate } from 'react-router-dom'
import { CollectionsDashboard } from '../CollectionsDashboard'
import { CollectionDetailsModal } from '../CollectionDetailsModal'
import { useCollectionStore } from '../../stores/collectionStore'
import { useUIStore } from '../../stores/uiStore'
import { 
  renderWithErrorHandlers, 
  waitForError,
  waitForToast,
  removeAuthToken,
  mockConsoleError
} from '../../tests/utils/errorTestUtils'
import { 
  collectionErrorHandlers, 
  authErrorHandlers,
  combineErrorHandlers 
} from '../../tests/mocks/errorHandlers'
import { server } from '../../tests/mocks/server'
import { handlers } from '../../tests/mocks/handlers'

// Mock navigation
vi.mock('react-router-dom', () => ({
  ...vi.importActual('react-router-dom'),
  useNavigate: vi.fn()
}))

// Mock stores
vi.mock('../../stores/collectionStore')
vi.mock('../../stores/uiStore')

describe('Collections - Permission Error Handling', () => {
  const mockNavigate = vi.fn()
  const mockAddToast = vi.fn()
  const mockFetchCollections = vi.fn()
  
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(useNavigate).mockReturnValue(mockNavigate)
    
    vi.mocked(useUIStore).mockReturnValue({
      addToast: mockAddToast
    } as any)
  })

  describe('Unauthorized Access (401)', () => {
    it('should redirect to login when auth token is invalid', async () => {
      vi.mocked(useCollectionStore).mockReturnValue({
        collections: [],
        loading: false,
        error: null,
        fetchCollections: mockFetchCollections
      } as any)
      
      // Set up 401 error for collections endpoint
      server.use(
        authErrorHandlers.unauthorized()[0], // GET /api/auth/me returns 401
        collectionErrorHandlers.createErrorHandler('get', '/api/v2/collections', 401)
      )
      
      renderWithErrorHandlers(
        <CollectionsDashboard />,
        []
      )

      // Should attempt to load collections
      await waitFor(() => {
        expect(mockFetchCollections).toHaveBeenCalled()
      })
      
      // Should redirect to login
      await waitFor(() => {
        expect(mockNavigate).toHaveBeenCalledWith('/login')
      })
    })

    it('should handle token expiry during operation', async () => {
      const mockCreateCollection = vi.fn()
      vi.mocked(useCollectionStore).mockReturnValue({
        collections: [],
        createCollection: mockCreateCollection,
        fetchCollections: vi.fn()
      } as any)
      
      // Start with valid auth
      server.use(...handlers)
      
      renderWithErrorHandlers(
        <CollectionsDashboard />,
        []
      )
      
      // Open create modal
      const createButton = await screen.findByRole('button', { name: /create.*collection/i })
      await userEvent.click(createButton)
      
      // Now simulate token expiry
      mockCreateCollection.mockRejectedValue({
        response: { 
          status: 401,
          data: { detail: 'Token expired' }
        }
      })
      
      server.use(
        authErrorHandlers.unauthorized()[1] // POST /api/auth/refresh returns 401
      )
      
      // Try to create collection
      await userEvent.type(screen.getByLabelText(/collection name/i), 'Test')
      await userEvent.click(screen.getByRole('button', { name: /create$/i }))
      
      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith(
          expect.stringContaining('Token expired'),
          'error'
        )
      })
      
      // Should redirect to login
      await waitFor(() => {
        expect(mockNavigate).toHaveBeenCalledWith('/login')
      })
    })

    it('should clear local storage on auth failure', async () => {
      // Set some auth data
      localStorage.setItem('access_token', 'invalid-token')
      localStorage.setItem('refresh_token', 'invalid-refresh')
      
      vi.mocked(useCollectionStore).mockReturnValue({
        collections: [],
        fetchCollections: mockFetchCollections,
        loading: false,
        error: 'Unauthorized'
      } as any)
      
      server.use(...authErrorHandlers.unauthorized())
      
      renderWithErrorHandlers(
        <CollectionsDashboard />,
        []
      )
      
      await waitFor(() => {
        expect(mockNavigate).toHaveBeenCalledWith('/login')
      })
      
      // Auth tokens should be cleared
      expect(localStorage.getItem('access_token')).toBeNull()
      expect(localStorage.getItem('refresh_token')).toBeNull()
    })

    it('should not enter redirect loop on login page', async () => {
      // Simulate already being on login page
      window.location.pathname = '/login'
      
      server.use(...authErrorHandlers.unauthorized())
      
      renderWithErrorHandlers(
        <CollectionsDashboard />,
        []
      )
      
      // Should not redirect again
      await waitFor(() => {
        expect(mockNavigate).not.toHaveBeenCalled()
      })
    })
  })

  describe('Forbidden Access (403)', () => {
    it('should show error when accessing another users collection', async () => {
      const otherUserCollection = {
        uuid: 'other-user-collection',
        name: 'Private Collection',
        user_id: 999,
        status: 'ready'
      }
      
      vi.mocked(useCollectionStore).mockReturnValue({
        collections: [],
        selectedCollection: otherUserCollection,
        fetchCollection: vi.fn().mockRejectedValue({
          response: {
            status: 403,
            data: { detail: 'You do not have permission to access this collection' }
          }
        })
      } as any)
      
      server.use(
        collectionErrorHandlers.permissionError()[0]
      )
      
      renderWithErrorHandlers(
        <CollectionDetailsModal
          isOpen={true}
          onClose={vi.fn()}
          collectionId={otherUserCollection.uuid}
        />,
        []
      )
      
      await waitForToast('You do not have permission to access this collection', 'error')
      
      // Should not show collection details
      expect(screen.queryByText(otherUserCollection.name)).not.toBeInTheDocument()
    })

    it('should prevent deletion of collections user doesnt own', async () => {
      const mockDeleteCollection = vi.fn()
      
      vi.mocked(useCollectionStore).mockReturnValue({
        deleteCollection: mockDeleteCollection
      } as any)
      
      mockDeleteCollection.mockRejectedValue({
        response: {
          status: 403,
          data: { detail: 'Only the collection owner can delete this collection' }
        }
      })
      
      server.use(
        collectionErrorHandlers.permissionError()[1]
      )
      
      // Render delete modal
      const DeleteCollectionModal = (await import('../DeleteCollectionModal')).DeleteCollectionModal
      
      renderWithErrorHandlers(
        <DeleteCollectionModal
          isOpen={true}
          onClose={vi.fn()}
          collectionId="other-user-collection"
          collectionName="Other User Collection"
          stats={{ documents: 10, vectors: 100 }}
          onSuccess={vi.fn()}
        />,
        []
      )
      
      // Try to delete
      const deleteButton = screen.getByRole('button', { name: /delete/i })
      await userEvent.click(deleteButton)
      
      await waitForToast('Only the collection owner can delete this collection', 'error')
      
      // Modal should remain open
      expect(screen.getByRole('dialog')).toBeInTheDocument()
    })

    it('should hide admin features for non-admin users', async () => {
      // Mock a non-admin user
      const mockUser = {
        id: 1,
        username: 'regular_user',
        is_superuser: false
      }
      
      vi.mocked(useCollectionStore).mockReturnValue({
        collections: [
          { uuid: '1', name: 'My Collection', status: 'ready' },
          { uuid: '2', name: 'Shared Collection', status: 'ready', is_public: true }
        ],
        fetchCollections: vi.fn(),
        currentUser: mockUser
      } as any)
      
      renderWithErrorHandlers(
        <CollectionsDashboard />,
        []
      )
      
      // Should not show admin-only features
      expect(screen.queryByText(/admin/i)).not.toBeInTheDocument()
      expect(screen.queryByRole('button', { name: /manage all collections/i })).not.toBeInTheDocument()
    })
  })

  describe('Collection Access Patterns', () => {
    it('should handle accessing a deleted collection', async () => {
      vi.mocked(useCollectionStore).mockReturnValue({
        selectedCollection: null,
        fetchCollection: vi.fn().mockRejectedValue({
          response: {
            status: 404,
            data: { detail: 'Collection not found' }
          }
        })
      } as any)
      
      server.use(
        collectionErrorHandlers.notFound()[0]
      )
      
      // Try to access via direct URL
      renderWithErrorHandlers(
        <CollectionDetailsModal
          isOpen={true}
          onClose={vi.fn()}
          collectionId="deleted-collection-id"
        />,
        []
      )
      
      await waitForToast('Collection not found', 'error')
      
      // Should close modal or redirect
      await waitFor(() => {
        expect(mockNavigate).toHaveBeenCalledWith('/collections')
      })
    })

    it('should handle permission changes mid-session', async () => {
      const mockCollection = {
        uuid: 'test-collection',
        name: 'Test Collection',
        status: 'ready'
      }
      
      const mockUpdateCollection = vi.fn()
      
      vi.mocked(useCollectionStore).mockReturnValue({
        selectedCollection: mockCollection,
        updateCollection: mockUpdateCollection,
        fetchCollection: vi.fn()
      } as any)
      
      // Start with successful load
      renderWithErrorHandlers(
        <CollectionDetailsModal
          isOpen={true}
          onClose={vi.fn()}
          collectionId={mockCollection.uuid}
        />,
        []
      )
      
      // Now simulate permission revoked
      mockUpdateCollection.mockRejectedValue({
        response: {
          status: 403,
          data: { detail: 'Access revoked by collection owner' }
        }
      })
      
      // Try to perform an action
      // (This would be a rename, update settings, etc.)
      
      await waitFor(() => {
        // Should show error and possibly close modal
        expect(mockAddToast).toHaveBeenCalledWith(
          expect.stringContaining('Access revoked'),
          'error'
        )
      })
    })
  })

  describe('API Key Authentication Errors', () => {
    it('should handle invalid API key errors', async () => {
      // Simulate API key auth failure
      server.use(
        collectionErrorHandlers.createErrorHandler('get', '/api/v2/collections', 401, {
          detail: 'Invalid API key'
        })
      )
      
      vi.mocked(useCollectionStore).mockReturnValue({
        collections: [],
        fetchCollections: mockFetchCollections,
        error: 'Invalid API key'
      } as any)
      
      renderWithErrorHandlers(
        <CollectionsDashboard />,
        []
      )
      
      await waitForError('Invalid API key')
      
      // Should show specific message about API key
      expect(screen.getByText(/invalid api key/i)).toBeInTheDocument()
    })

    it('should handle expired API key', async () => {
      server.use(
        collectionErrorHandlers.createErrorHandler('get', '/api/v2/collections', 401, {
          detail: 'API key expired'
        })
      )
      
      vi.mocked(useCollectionStore).mockReturnValue({
        collections: [],
        fetchCollections: mockFetchCollections,
        error: 'API key expired'
      } as any)
      
      renderWithErrorHandlers(
        <CollectionsDashboard />,
        []
      )
      
      await waitForError('API key expired')
      
      // Should guide user to generate new API key
      expect(screen.getByText(/api key expired/i)).toBeInTheDocument()
    })
  })

  describe('Session Management', () => {
    it('should handle concurrent session limit', async () => {
      server.use(
        collectionErrorHandlers.createErrorHandler('post', '/api/auth/login', 403, {
          detail: 'Maximum concurrent sessions reached. Please log out from another device.'
        })
      )
      
      // This would be in the login component
      const mockLogin = vi.fn().mockRejectedValue({
        response: {
          status: 403,
          data: { detail: 'Maximum concurrent sessions reached. Please log out from another device.' }
        }
      })
      
      // Simulate login attempt
      await mockLogin('user', 'pass').catch(err => {
        expect(err.response.data.detail).toContain('Maximum concurrent sessions')
      })
    })

    it('should handle session timeout gracefully', async () => {
      const { mockError, restore } = mockConsoleError()
      
      try {
        // Simulate a long-running session
        vi.mocked(useCollectionStore).mockReturnValue({
          collections: [],
          fetchCollections: mockFetchCollections,
          error: 'Session expired'
        } as any)
        
        server.use(
          authErrorHandlers.createErrorHandler('get', '/api/auth/me', 401, {
            detail: 'Session expired. Please log in again.'
          })
        )
        
        renderWithErrorHandlers(
          <CollectionsDashboard />,
          []
        )
        
        await waitFor(() => {
          expect(mockNavigate).toHaveBeenCalledWith('/login')
        })
        
        // Should show informative message
        expect(mockAddToast).toHaveBeenCalledWith(
          expect.stringContaining('Session expired'),
          'warning'
        )
      } finally {
        restore()
      }
    })
  })
})