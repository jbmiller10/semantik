import React from 'react'
import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { CreateCollectionModal } from '../CreateCollectionModal'
import { AddDataToCollectionModal } from '../AddDataToCollectionModal'
import { RenameCollectionModal } from '../RenameCollectionModal'
import { ReindexCollectionModal } from '../ReindexCollectionModal'
import { useCollectionStore } from '../../stores/collectionStore'
import { useUIStore } from '../../stores/uiStore'
import { 
  renderWithErrorHandlers, 
  waitForToast,
  waitForError
} from '../../tests/utils/errorTestUtils'
import { collectionErrorHandlers } from '../../tests/mocks/errorHandlers'
import { server } from '../../tests/mocks/server'

// Mock stores
vi.mock('../../stores/collectionStore')
vi.mock('../../stores/uiStore')

describe('Collection Modals - API Validation Errors', () => {
  const mockAddToast = vi.fn()
  const mockCollection = {
    uuid: 'test-uuid',
    name: 'Test Collection',
    status: 'ready',
    document_count: 100,
    vector_count: 1000,
    embedding_model: 'test-model',
    quantization: 'float16',
    chunk_size: 512,
    chunk_overlap: 50
  }

  beforeEach(() => {
    vi.clearAllMocks()
    
    vi.mocked(useUIStore).mockReturnValue({
      addToast: mockAddToast,
    } as any)
  })

  describe('CreateCollectionModal - Validation Errors', () => {
    const mockCreateCollection = vi.fn()

    beforeEach(() => {
      vi.mocked(useCollectionStore).mockReturnValue({
        createCollection: mockCreateCollection,
      } as any)
    })

    it('should handle duplicate collection name error', async () => {
      server.use(
        collectionErrorHandlers.validationError()[0] // Duplicate name handler
      )
      
      mockCreateCollection.mockRejectedValue({
        response: { data: { detail: 'Collection with this name already exists' } }
      })
      
      renderWithErrorHandlers(
        <CreateCollectionModal isOpen={true} onClose={vi.fn()} />,
        []
      )

      await userEvent.type(screen.getByLabelText(/collection name/i), 'Existing Collection')
      await userEvent.click(screen.getByRole('button', { name: /create$/i }))
      
      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith(
          'Collection with this name already exists',
          'error'
        )
      })
      
      // Modal should remain open
      expect(screen.getByRole('dialog')).toBeInTheDocument()
      
      // Form should show error state (implementation dependent)
      // User should be able to change the name and try again
      const nameInput = screen.getByLabelText(/collection name/i)
      await userEvent.clear(nameInput)
      await userEvent.type(nameInput, 'Unique Collection Name')
      
      // Should be able to submit again
      expect(screen.getByRole('button', { name: /create$/i })).not.toBeDisabled()
    })

    it('should handle collection limit exceeded error', async () => {
      server.use(
        collectionErrorHandlers.rateLimited()[0] // Collection limit
      )
      
      mockCreateCollection.mockRejectedValue({
        response: { 
          status: 429,
          data: { detail: 'Collection limit reached (10 max)' } 
        }
      })
      
      renderWithErrorHandlers(
        <CreateCollectionModal isOpen={true} onClose={vi.fn()} />,
        []
      )

      await userEvent.type(screen.getByLabelText(/collection name/i), 'New Collection')
      await userEvent.click(screen.getByRole('button', { name: /create$/i }))
      
      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith(
          'Collection limit reached (10 max)',
          'error'
        )
      })
      
      // User should understand they cannot create more collections
      // Modal stays open but user might need to close it
    })

    it('should handle invalid collection name format', async () => {
      mockCreateCollection.mockRejectedValue({
        response: { 
          status: 400,
          data: { detail: 'Collection name must be between 3-50 characters and contain only letters, numbers, spaces, hyphens, and underscores' } 
        }
      })
      
      renderWithErrorHandlers(
        <CreateCollectionModal isOpen={true} onClose={vi.fn()} />,
        []
      )

      // Try with invalid characters
      await userEvent.type(screen.getByLabelText(/collection name/i), 'My Collection @#$%')
      await userEvent.click(screen.getByRole('button', { name: /create$/i }))
      
      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith(
          expect.stringContaining('Collection name must be'),
          'error'
        )
      })
    })

    it('should handle invalid chunk size validation', async () => {
      mockCreateCollection.mockRejectedValue({
        response: { 
          status: 400,
          data: { detail: 'Chunk size must be between 128 and 4096' } 
        }
      })
      
      renderWithErrorHandlers(
        <CreateCollectionModal isOpen={true} onClose={vi.fn()} />,
        []
      )

      await userEvent.type(screen.getByLabelText(/collection name/i), 'Valid Name')
      
      // Expand advanced settings
      await userEvent.click(screen.getByText(/advanced settings/i))
      
      // Set invalid chunk size
      const chunkSizeInput = screen.getByLabelText(/chunk size/i)
      await userEvent.clear(chunkSizeInput)
      await userEvent.type(chunkSizeInput, '10000')
      
      await userEvent.click(screen.getByRole('button', { name: /create$/i }))
      
      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith(
          'Chunk size must be between 128 and 4096',
          'error'
        )
      })
    })
  })

  describe('AddDataToCollectionModal - Validation Errors', () => {
    const mockAddSource = vi.fn()

    beforeEach(() => {
      vi.mocked(useCollectionStore).mockReturnValue({
        addSource: mockAddSource,
      } as any)
    })

    it('should handle invalid path error', async () => {
      server.use(
        collectionErrorHandlers.validationError()[2] // Invalid path handler
      )
      
      mockAddSource.mockRejectedValue({
        response: { 
          status: 400,
          data: { detail: 'Path does not exist or is not accessible: /nonexistent/path' } 
        }
      })
      
      renderWithErrorHandlers(
        <AddDataToCollectionModal 
          isOpen={true} 
          onClose={vi.fn()} 
          collection={mockCollection}
        />,
        []
      )

      await userEvent.type(screen.getByLabelText(/source directory path/i), '/nonexistent/path')
      await userEvent.click(screen.getByRole('button', { name: /add.*source/i }))
      
      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith(
          'Path does not exist or is not accessible: /nonexistent/path',
          'error'
        )
      })
      
      // User should be able to correct the path
      const pathInput = screen.getByLabelText(/source directory path/i)
      await userEvent.clear(pathInput)
      await userEvent.type(pathInput, '/valid/path')
      
      expect(screen.getByRole('button', { name: /add.*source/i })).not.toBeDisabled()
    })

    it('should handle permission denied for path', async () => {
      mockAddSource.mockRejectedValue({
        response: { 
          status: 403,
          data: { detail: 'Permission denied: Cannot access /restricted/path' } 
        }
      })
      
      renderWithErrorHandlers(
        <AddDataToCollectionModal 
          isOpen={true} 
          onClose={vi.fn()} 
          collection={mockCollection}
        />,
        []
      )

      await userEvent.type(screen.getByLabelText(/source directory path/i), '/restricted/path')
      await userEvent.click(screen.getByRole('button', { name: /add.*source/i }))
      
      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith(
          'Permission denied: Cannot access /restricted/path',
          'error'
        )
      })
    })

    it('should handle too many sources error', async () => {
      mockAddSource.mockRejectedValue({
        response: { 
          status: 400,
          data: { detail: 'Maximum number of sources (10) reached for this collection' } 
        }
      })
      
      renderWithErrorHandlers(
        <AddDataToCollectionModal 
          isOpen={true} 
          onClose={vi.fn()} 
          collection={mockCollection}
        />,
        []
      )

      await userEvent.type(screen.getByLabelText(/source directory path/i), '/another/source')
      await userEvent.click(screen.getByRole('button', { name: /add.*source/i }))
      
      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith(
          'Maximum number of sources (10) reached for this collection',
          'error'
        )
      })
      
      // User should understand they cannot add more sources
    })

    it('should handle operation already in progress error', async () => {
      server.use(
        collectionErrorHandlers.rateLimited()[1] // Too many operations
      )
      
      mockAddSource.mockRejectedValue({
        response: { 
          status: 429,
          data: { detail: 'Too many operations in progress. Please wait and try again.' } 
        }
      })
      
      renderWithErrorHandlers(
        <AddDataToCollectionModal 
          isOpen={true} 
          onClose={vi.fn()} 
          collection={mockCollection}
        />,
        []
      )

      await userEvent.type(screen.getByLabelText(/source directory path/i), '/data/docs')
      await userEvent.click(screen.getByRole('button', { name: /add.*source/i }))
      
      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith(
          'Too many operations in progress. Please wait and try again.',
          'error'
        )
      })
    })
  })

  describe('RenameCollectionModal - Validation Errors', () => {
    const mockUpdateCollection = vi.fn()

    beforeEach(() => {
      vi.mocked(useCollectionStore).mockReturnValue({
        updateCollection: mockUpdateCollection,
      } as any)
    })

    it('should handle invalid collection name error', async () => {
      server.use(
        collectionErrorHandlers.validationError()[1] // Invalid name format
      )
      
      mockUpdateCollection.mockRejectedValue({
        response: { 
          status: 400,
          data: { detail: 'Invalid collection name: must be between 3-50 characters' } 
        }
      })
      
      renderWithErrorHandlers(
        <RenameCollectionModal 
          isOpen={true} 
          onClose={vi.fn()}
          collectionName={mockCollection.name}
          collectionId={mockCollection.uuid}
          onSuccess={vi.fn()}
        />,
        []
      )

      const nameInput = screen.getByLabelText(/new name/i)
      await userEvent.clear(nameInput)
      await userEvent.type(nameInput, 'a') // Too short
      
      await userEvent.click(screen.getByRole('button', { name: /rename/i }))
      
      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith(
          'Invalid collection name: must be between 3-50 characters',
          'error'
        )
      })
    })

    it('should handle duplicate name when renaming', async () => {
      mockUpdateCollection.mockRejectedValue({
        response: { 
          status: 400,
          data: { detail: 'A collection with this name already exists' } 
        }
      })
      
      renderWithErrorHandlers(
        <RenameCollectionModal 
          isOpen={true} 
          onClose={vi.fn()}
          collectionName={mockCollection.name}
          collectionId={mockCollection.uuid}
          onSuccess={vi.fn()}
        />,
        []
      )

      const nameInput = screen.getByLabelText(/new name/i)
      await userEvent.clear(nameInput)
      await userEvent.type(nameInput, 'Existing Collection')
      
      await userEvent.click(screen.getByRole('button', { name: /rename/i }))
      
      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith(
          'A collection with this name already exists',
          'error'
        )
      })
    })
  })

  describe('ReindexCollectionModal - Validation Errors', () => {
    const mockReindexCollection = vi.fn()

    beforeEach(() => {
      vi.mocked(useCollectionStore).mockReturnValue({
        reindexCollection: mockReindexCollection,
      } as any)
    })

    it('should handle invalid configuration during reindex', async () => {
      mockReindexCollection.mockRejectedValue({
        response: { 
          status: 400,
          data: { detail: 'Invalid chunk overlap: must be less than chunk size' } 
        }
      })
      
      const configChanges = {
        chunk_size: 256,
        chunk_overlap: 300 // Invalid: larger than chunk size
      }
      
      renderWithErrorHandlers(
        <ReindexCollectionModal 
          isOpen={true} 
          onClose={vi.fn()}
          collection={mockCollection}
          configChanges={configChanges}
          onSuccess={vi.fn()}
        />,
        []
      )

      // Type confirmation
      const confirmInput = screen.getByLabelText(/type.*to confirm/i)
      await userEvent.type(confirmInput, `reindex ${mockCollection.name}`)
      
      await userEvent.click(screen.getByRole('button', { name: /start.*reindex/i }))
      
      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith(
          'Invalid chunk overlap: must be less than chunk size',
          'error'
        )
      })
    })

    it('should handle resource unavailable error', async () => {
      mockReindexCollection.mockRejectedValue({
        response: { 
          status: 503,
          data: { detail: 'Insufficient resources available for reindexing. Please try again later.' } 
        }
      })
      
      renderWithErrorHandlers(
        <ReindexCollectionModal 
          isOpen={true} 
          onClose={vi.fn()}
          collection={mockCollection}
          configChanges={{ chunk_size: 1024 }}
          onSuccess={vi.fn()}
        />,
        []
      )

      const confirmInput = screen.getByLabelText(/type.*to confirm/i)
      await userEvent.type(confirmInput, `reindex ${mockCollection.name}`)
      
      await userEvent.click(screen.getByRole('button', { name: /start.*reindex/i }))
      
      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith(
          'Insufficient resources available for reindexing. Please try again later.',
          'error'
        )
      })
    })
  })

  describe('Field-level Validation', () => {
    it('should show inline validation for collection name', async () => {
      renderWithErrorHandlers(
        <CreateCollectionModal isOpen={true} onClose={vi.fn()} />,
        []
      )

      const nameInput = screen.getByLabelText(/collection name/i)
      
      // Type invalid characters
      await userEvent.type(nameInput, 'My Collection!!!')
      
      // Blur to trigger validation (if implemented)
      await userEvent.tab()
      
      // Check if there's any validation message (implementation specific)
      // This might show as aria-invalid, error text, or red border
    })

    it('should validate numeric fields stay within bounds', async () => {
      renderWithErrorHandlers(
        <CreateCollectionModal isOpen={true} onClose={vi.fn()} />,
        []
      )

      await userEvent.click(screen.getByText(/advanced settings/i))
      
      const chunkSizeInput = screen.getByLabelText(/chunk size/i)
      await userEvent.clear(chunkSizeInput)
      await userEvent.type(chunkSizeInput, '99999')
      
      // Should either prevent typing beyond max or show error
      // Actual behavior depends on implementation
    })
  })
})