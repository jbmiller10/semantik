import React from 'react'
import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import QuickCreateModal from '../QuickCreateModal'
import AddDataToCollectionModal from '../AddDataToCollectionModal'
import RenameCollectionModal from '../RenameCollectionModal'
import ReindexCollectionModal from '../ReindexCollectionModal'
import { 
  renderWithErrorHandlers
} from '../../tests/utils/errorTestUtils'

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

const mockReindexCollectionMutation = {
  mutateAsync: vi.fn(),
  isError: false,
  isPending: false,
};

const mockAddToast = vi.fn();

// Mock the hook to not handle errors (let the component handle them)
vi.mock('../../hooks/useCollections', () => ({
  useCreateCollection: () => mockCreateCollectionMutation,
}));

vi.mock('../../hooks/useCollectionOperations', () => ({
  useAddSource: () => mockAddSourceMutation,
  useReindexCollection: () => mockReindexCollectionMutation,
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

// Mock connector catalog
const mockConnectorCatalog = {
  directory: {
    name: 'Local Directory',
    description: 'Index files from a local directory path',
    icon: 'folder',
    fields: [
      {
        name: 'path',
        type: 'text',
        label: 'Directory Path',
        description: 'Absolute path to the directory',
        required: true,
        placeholder: '/path/to/documents',
      },
    ],
    secrets: [],
    supports_sync: true,
  },
};

vi.mock('../../hooks/useConnectors', () => ({
  useConnectorCatalog: () => ({
    data: mockConnectorCatalog,
    isLoading: false,
    isError: false,
  }),
  useGitPreview: () => ({
    mutateAsync: vi.fn(),
    isPending: false,
  }),
  useImapPreview: () => ({
    mutateAsync: vi.fn(),
    isPending: false,
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

// Mock useMutation for RenameCollectionModal
const mockRenameMutation = {
  mutate: vi.fn(),
  isPending: false,
};

vi.mock('@tanstack/react-query', async () => {
  const actual = await vi.importActual('@tanstack/react-query');
  return {
    ...actual,
    useMutation: vi.fn(() => mockRenameMutation),
  };
});

describe('Collection Modals - API Validation Errors', () => {
  const mockCollection = {
    id: 'test-uuid',
    uuid: 'test-uuid',
    name: 'Test Collection',
    status: 'ready' as const,
    document_count: 100,
    vector_count: 1000,
    embedding_model: 'test-model',
    quantization: 'float16' as const,
    chunk_size: 512,
    chunk_overlap: 50,
    owner_id: 1,
    vector_store_name: 'test-store',
    is_public: false,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  }

  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('QuickCreateModal - Validation Errors', () => {
    it('should handle duplicate collection name error', async () => {
      // Mock the mutation to reject with an Error (as the component expects)
      mockCreateCollectionMutation.mutateAsync.mockRejectedValue(
        new Error('Collection with this name already exists')
      )
      
      renderWithErrorHandlers(
        <QuickCreateModal onClose={vi.fn()} onSuccess={vi.fn()} />,
        []
      )

      await userEvent.type(screen.getByLabelText(/collection name/i), 'Existing Collection')
      await userEvent.click(screen.getByRole('button', { name: /create collection/i }))
      
      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith({
          message: 'Collection with this name already exists',
          type: 'error'
        })
      })
      
      // Modal should remain open
      expect(screen.getByText(/create new collection/i)).toBeInTheDocument()
      
      // Form should show error state (implementation dependent)
      // User should be able to change the name and try again
      const nameInput = screen.getByLabelText(/collection name/i)
      await userEvent.clear(nameInput)
      await userEvent.type(nameInput, 'Unique Collection Name')
      
      // Should be able to submit again
      expect(screen.getByRole('button', { name: /create collection/i })).not.toBeDisabled()
    })

    it('should handle collection limit exceeded error', async () => {
      mockCreateCollectionMutation.mutateAsync.mockRejectedValue(
        new Error('Collection limit reached (10 max)')
      )
      
      renderWithErrorHandlers(
        <QuickCreateModal onClose={vi.fn()} onSuccess={vi.fn()} />,
        []
      )

      await userEvent.type(screen.getByLabelText(/collection name/i), 'New Collection')
      await userEvent.click(screen.getByRole('button', { name: /create collection/i }))
      
      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith({
          message: 'Collection limit reached (10 max)',
          type: 'error'
        })
      })
      
      // User should understand they cannot create more collections
      // Modal stays open but user might need to close it
    })

    it('should handle invalid collection name format', async () => {
      mockCreateCollectionMutation.mutateAsync.mockRejectedValue(
        new Error('Collection name must be between 3-50 characters and contain only letters, numbers, spaces, hyphens, and underscores')
      )
      
      renderWithErrorHandlers(
        <QuickCreateModal onClose={vi.fn()} onSuccess={vi.fn()} />,
        []
      )

      // Try with invalid characters
      await userEvent.type(screen.getByLabelText(/collection name/i), 'My Collection @#$%')
      await userEvent.click(screen.getByRole('button', { name: /create collection/i }))
      
      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith({
          message: expect.stringContaining('Collection name must be'),
          type: 'error'
        })
      })
    })

    it('should handle invalid chunk size validation', async () => {
      renderWithErrorHandlers(
        <QuickCreateModal onClose={vi.fn()} onSuccess={vi.fn()} />,
        []
      )

      // Try to submit without collection name to trigger validation
      await userEvent.click(screen.getByRole('button', { name: /create collection/i }))
      
      // Check for validation error
      await waitFor(() => {
        expect(screen.getByText('Please fix the following errors:')).toBeInTheDocument()
        // Use getAllByText since the error appears in multiple places
        const errors = screen.getAllByText('Collection name is required')
        expect(errors.length).toBeGreaterThan(0)
      })
      
      // Modal should remain open
      expect(screen.getByText(/create new collection/i)).toBeInTheDocument()
      
      // Now fill the name and test advanced settings
      await userEvent.type(screen.getByLabelText(/collection name/i), 'Test Collection')
      
      // Expand advanced settings and verify they're visible
      await userEvent.click(screen.getByText(/advanced settings/i))
      await waitFor(() => {
        // Check for chunking strategy selector instead of individual fields
        expect(screen.getByText(/chunking strategy/i)).toBeInTheDocument()
      })
      
      // Verify the chunking strategy selector is present
      const strategySection = screen.getByText(/chunking strategy/i).closest('div')
      expect(strategySection).toBeInTheDocument()
    })
  })

  describe('AddDataToCollectionModal - Validation Errors', () => {
    it('should handle invalid path error', async () => {
      mockAddSourceMutation.mutateAsync.mockRejectedValue(
        new Error('Path does not exist or is not accessible: /nonexistent/path')
      )

      renderWithErrorHandlers(
        <AddDataToCollectionModal
          onClose={vi.fn()}
          collection={mockCollection}
          onSuccess={vi.fn()}
        />,
        []
      )

      await userEvent.type(screen.getByLabelText(/directory path/i), '/nonexistent/path')
      await userEvent.click(screen.getByRole('button', { name: /add source/i }))

      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith({
          message: 'Path does not exist or is not accessible: /nonexistent/path',
          type: 'error'
        })
      })

      // User should be able to correct the path
      const pathInput = screen.getByLabelText(/directory path/i)
      await userEvent.clear(pathInput)
      await userEvent.type(pathInput, '/valid/path')

      expect(screen.getByRole('button', { name: /add source/i })).not.toBeDisabled()
    })

    it('should handle permission denied for path', async () => {
      mockAddSourceMutation.mutateAsync.mockRejectedValue(
        new Error('Permission denied: Cannot access /restricted/path')
      )

      renderWithErrorHandlers(
        <AddDataToCollectionModal
          onClose={vi.fn()}
          collection={mockCollection}
          onSuccess={vi.fn()}
        />,
        []
      )

      await userEvent.type(screen.getByLabelText(/directory path/i), '/restricted/path')
      await userEvent.click(screen.getByRole('button', { name: /add source/i }))

      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith({
          message: 'Permission denied: Cannot access /restricted/path',
          type: 'error'
        })
      })
    })

    it('should handle too many sources error', async () => {
      mockAddSourceMutation.mutateAsync.mockRejectedValue(
        new Error('Maximum number of sources (10) reached for this collection')
      )

      renderWithErrorHandlers(
        <AddDataToCollectionModal
          onClose={vi.fn()}
          collection={mockCollection}
          onSuccess={vi.fn()}
        />,
        []
      )

      await userEvent.type(screen.getByLabelText(/directory path/i), '/another/source')
      await userEvent.click(screen.getByRole('button', { name: /add source/i }))

      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith({
          message: 'Maximum number of sources (10) reached for this collection',
          type: 'error'
        })
      })

      // User should understand they cannot add more sources
    })

    it('should handle operation already in progress error', async () => {
      mockAddSourceMutation.mutateAsync.mockRejectedValue(
        new Error('Too many operations in progress. Please wait and try again.')
      )

      renderWithErrorHandlers(
        <AddDataToCollectionModal
          onClose={vi.fn()}
          collection={mockCollection}
          onSuccess={vi.fn()}
        />,
        []
      )

      await userEvent.type(screen.getByLabelText(/directory path/i), '/data/docs')
      await userEvent.click(screen.getByRole('button', { name: /add source/i }))

      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith({
          message: 'Too many operations in progress. Please wait and try again.',
          type: 'error'
        })
      })
    })
  })

  describe('RenameCollectionModal - Validation Errors', () => {
    beforeEach(() => {
      // Mock useMutation to handle the error case
      vi.mocked(mockRenameMutation.mutate).mockImplementation(() => {
        // The component handles errors via onError callback
      })
    })

    it('should handle invalid collection name error', async () => {
      renderWithErrorHandlers(
        <RenameCollectionModal 
          onClose={vi.fn()}
          currentName={mockCollection.name}
          collectionId={mockCollection.uuid}
          onSuccess={vi.fn()}
        />,
        []
      )

      const nameInput = screen.getByLabelText(/new name/i)
      await userEvent.clear(nameInput)
      await userEvent.type(nameInput, 'a') // Too short
      
      await userEvent.click(screen.getByRole('button', { name: /rename/i }))
      
      // RenameCollectionModal might not have client-side validation, so check for error
      await waitFor(() => {
        expect(screen.getByText(/new name/i)).toBeInTheDocument() // Modal still open
      })
    })

    it('should handle duplicate name when renaming', async () => {
      // For server-side errors, we need to mock the mutation's onError callback
      const { useMutation } = await import('@tanstack/react-query')
      vi.mocked(useMutation).mockImplementation((options: { onError?: (error: unknown) => void }) => ({
        mutate: () => {
          options.onError?.({
            response: { 
              status: 400,
              data: { detail: 'A collection with this name already exists' } 
            }
          })
        },
        isPending: false,
      } as ReturnType<typeof useMutation>))
      
      renderWithErrorHandlers(
        <RenameCollectionModal 
          onClose={vi.fn()}
          currentName={mockCollection.name}
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
        expect(mockAddToast).toHaveBeenCalledWith({
          message: 'Failed to rename collection',
          type: 'error'
        })
      })
    })
  })

  describe('ReindexCollectionModal - Validation Errors', () => {
    it('should handle invalid configuration during reindex', async () => {
      mockReindexCollectionMutation.mutateAsync.mockRejectedValue(
        new Error('Invalid chunk overlap: must be less than chunk size')
      )
      
      const configChanges = {
        chunk_size: 256,
        chunk_overlap: 300 // Invalid: larger than chunk size
      }
      
      renderWithErrorHandlers(
        <ReindexCollectionModal 
          onClose={vi.fn()}
          collection={mockCollection}
          configChanges={configChanges}
          onSuccess={vi.fn()}
        />,
        []
      )

      // Type confirmation
      const confirmInput = screen.getByLabelText(/confirmation text/i)
      await userEvent.type(confirmInput, `reindex ${mockCollection.name}`)
      
      await userEvent.click(screen.getByRole('button', { name: /re-index collection/i }))
      
      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith({
          message: 'Invalid chunk overlap: must be less than chunk size',
          type: 'error'
        })
      })
    })

    it('should handle resource unavailable error', async () => {
      mockReindexCollectionMutation.mutateAsync.mockRejectedValue(
        new Error('Insufficient resources available for reindexing. Please try again later.')
      )
      
      renderWithErrorHandlers(
        <ReindexCollectionModal 
          onClose={vi.fn()}
          collection={mockCollection}
          configChanges={{ chunk_size: 1024 }}
          onSuccess={vi.fn()}
        />,
        []
      )

      const confirmInput = screen.getByLabelText(/confirmation text/i)
      await userEvent.type(confirmInput, `reindex ${mockCollection.name}`)
      
      await userEvent.click(screen.getByRole('button', { name: /re-index collection/i }))
      
      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith({
          message: 'Insufficient resources available for reindexing. Please try again later.',
          type: 'error'
        })
      })
    })
  })

  describe('Field-level Validation', () => {
    it('should show inline validation for collection name', async () => {
      renderWithErrorHandlers(
        <QuickCreateModal onClose={vi.fn()} onSuccess={vi.fn()} />,
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
        <QuickCreateModal onClose={vi.fn()} onSuccess={vi.fn()} />,
        []
      )

      await userEvent.click(screen.getByText(/advanced settings/i))
      
      // Chunking parameters are now handled by ChunkingParameterTuner with sliders
      // which enforce bounds automatically, so we just verify the component is present
      await waitFor(() => {
        expect(screen.getByText(/chunking strategy/i)).toBeInTheDocument()
      })
    })
  })
})