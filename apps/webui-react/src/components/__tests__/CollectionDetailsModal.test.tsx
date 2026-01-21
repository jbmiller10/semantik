import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import type { Collection, Operation } from '../../types/collection';
import type { DocumentResponse } from '../../services/api/v2/types';

// Use vi.hoisted to define mocks that will be used in hoisted vi.mock calls
const {
  mockShowCollectionDetailsModal,
  mockSetShowCollectionDetailsModal,
  mockAddToast,
  mockInvalidateQueries,
} = vi.hoisted(() => ({
  mockShowCollectionDetailsModal: vi.fn(),
  mockSetShowCollectionDetailsModal: vi.fn(),
  mockAddToast: vi.fn(),
  mockInvalidateQueries: vi.fn(),
}));

vi.mock('../../stores/uiStore', () => ({
  useUIStore: () => ({
    showCollectionDetailsModal: mockShowCollectionDetailsModal(),
    setShowCollectionDetailsModal: mockSetShowCollectionDetailsModal,
    addToast: mockAddToast,
  }),
}));

// Mock API
vi.mock('../../services/api/v2/collections', () => ({
  collectionsV2Api: {
    get: vi.fn(),
    listOperations: vi.fn(),
    listDocuments: vi.fn(),
    listSources: vi.fn(),
  },
}));

vi.mock('../../services/api/v2/documents', () => ({
  documentsV2Api: {
    retry: vi.fn(),
    retryFailed: vi.fn(),
    getFailedCount: vi.fn(),
  },
}));

// Mock collectionKeys - must match actual factory in useCollections.ts
vi.mock('../../hooks/useCollections', () => ({
  collectionKeys: {
    all: ['collections'],
    lists: () => ['collections', 'list'],
    list: (filters: unknown) => ['collections', 'list', filters],
    details: () => ['collections', 'detail'],
    detail: (id: string) => ['collections', 'detail', id],
  },
}));

// Mock operationKeys - must match actual factory in useCollectionOperations.ts
vi.mock('../../hooks/useCollectionOperations', () => ({
  operationKeys: {
    all: ['operations'],
    lists: () => ['operations', 'list'],
    list: (collectionId: string) => ['operations', 'list', collectionId],
  },
}));

// Mock React Query
vi.mock('@tanstack/react-query', async () => {
  const actual = await vi.importActual('@tanstack/react-query');
  return {
    ...actual,
    useQueryClient: () => ({
      invalidateQueries: mockInvalidateQueries,
    }),
  };
});

// Mock child modals with inline implementations (no hoisting needed)
vi.mock('../AddDataToCollectionModal', () => ({
  default: ({ onSuccess, onClose }: { onSuccess: () => void; onClose: () => void }) => (
    <div data-testid="add-data-modal">
      <button onClick={onSuccess}>Add Data Success</button>
      <button onClick={onClose}>Close Add Data</button>
    </div>
  ),
}));

vi.mock('../RenameCollectionModal', () => ({
  default: ({ onSuccess, onClose }: { onSuccess: () => void; onClose: () => void }) => (
    <div data-testid="rename-modal">
      <button onClick={onSuccess}>Rename Success</button>
      <button onClick={onClose}>Close Rename</button>
    </div>
  ),
}));

vi.mock('../DeleteCollectionModal', () => ({
  default: ({ onSuccess, onClose }: { onSuccess: () => void; onClose: () => void }) => (
    <div data-testid="delete-modal">
      <button onClick={onSuccess}>Delete Success</button>
      <button onClick={onClose}>Close Delete</button>
    </div>
  ),
}));

vi.mock('../ReindexCollectionModal', () => ({
  default: ({ onSuccess, onClose }: { onSuccess: () => void; onClose: () => void }) => (
    <div data-testid="reindex-modal">
      <button onClick={onSuccess}>Reindex Success</button>
      <button onClick={onClose}>Close Reindex</button>
    </div>
  ),
}));

vi.mock('../EmbeddingVisualizationTab', () => ({
  default: ({ collectionId }: { collectionId: string }) => (
    <div data-testid="embedding-visualization" data-collection-id={collectionId} />
  ),
}));

vi.mock('../collection/SparseIndexPanel', () => ({
  SparseIndexPanel: () => <div data-testid="sparse-index-panel" />,
}));

// Import components after mocks
import CollectionDetailsModal from '../CollectionDetailsModal';
import { TestWrapper } from '../../tests/utils/TestWrapper';
import { collectionsV2Api } from '../../services/api/v2/collections';
import { documentsV2Api } from '../../services/api/v2/documents';

// Get mocked functions (cast through unknown for type safety)
const mockCollectionsApi = collectionsV2Api as unknown as {
  get: ReturnType<typeof vi.fn>;
  listOperations: ReturnType<typeof vi.fn>;
  listDocuments: ReturnType<typeof vi.fn>;
  listSources: ReturnType<typeof vi.fn>;
};

const mockDocumentsApi = documentsV2Api as unknown as {
  retry: ReturnType<typeof vi.fn>;
  retryFailed: ReturnType<typeof vi.fn>;
  getFailedCount: ReturnType<typeof vi.fn>;
};

// Test data
const mockCollection: Collection = {
  id: 'test-collection-id',
  name: 'Test Collection',
  description: 'Test description',
  owner_id: 1,
  vector_store_name: 'test_vector_store',
  embedding_model: 'sentence-transformers/all-MiniLM-L6-v2',
  quantization: 'float32',
  chunk_size: 512,
  chunk_overlap: 50,
  is_public: false,
  status: 'ready',
  document_count: 100,
  vector_count: 500,
  total_size_bytes: 1048576,
  sync_mode: 'one_time',
  created_at: '2024-01-01T00:00:00Z',
  updated_at: '2024-01-02T00:00:00Z',
};

const mockCollectionWithChunkingStrategy: Collection = {
  ...mockCollection,
  chunking_strategy: 'recursive',
  chunking_config: {
    chunk_size: 600,
    chunk_overlap: 100,
    preserve_sentences: true,
  },
};

const mockCollectionWithLegacyChunking: Collection = {
  ...mockCollection,
  chunking_strategy: undefined,
  chunking_config: undefined,
  chunk_size: 512,
  chunk_overlap: 50,
};

const mockOperations: Operation[] = [
  {
    id: 'op-1',
    collection_id: 'test-collection-id',
    type: 'index',
    status: 'completed',
    config: {},
    created_at: '2024-01-01T00:00:00Z',
    started_at: '2024-01-01T00:01:00Z',
    completed_at: '2024-01-01T00:05:00Z',
  },
  {
    id: 'op-2',
    collection_id: 'test-collection-id',
    type: 'append',
    status: 'processing',
    config: {},
    created_at: '2024-01-02T00:00:00Z',
    started_at: '2024-01-02T00:01:00Z',
  },
];

const mockDocuments: DocumentResponse[] = [
  {
    id: 'doc-1',
    collection_id: 'test-collection-id',
    file_name: 'file1.txt',
    file_path: '/data/source1/file1.txt',
    file_size: 1024,
    mime_type: 'text/plain',
    content_hash: 'abc123',
    status: 'completed',
    error_message: null,
    chunk_count: 10,
    retry_count: 0,
    last_retry_at: null,
    error_category: null,
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z',
  },
  {
    id: 'doc-2',
    collection_id: 'test-collection-id',
    file_name: 'file2.txt',
    file_path: '/data/source2/file2.txt',
    file_size: 2048,
    mime_type: 'text/plain',
    content_hash: 'def456',
    status: 'completed',
    error_message: null,
    chunk_count: 20,
    retry_count: 0,
    last_retry_at: null,
    error_category: null,
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z',
  },
];

describe('CollectionDetailsModal', () => {
  const user = userEvent.setup();

  beforeEach(() => {
    vi.clearAllMocks();
    // Reset the mocks to their default resolved values
    mockCollectionsApi.get.mockResolvedValue({ data: mockCollection });
    mockCollectionsApi.listOperations.mockResolvedValue({ data: mockOperations });
    mockCollectionsApi.listDocuments.mockResolvedValue({
      data: {
        documents: mockDocuments,
        total: 2,
        page: 1,
        per_page: 50,
      },
    });
    mockCollectionsApi.listSources.mockResolvedValue({
      data: {
        items: [],
        total: 0,
        offset: 0,
        limit: 50,
      },
    });
    // Mock documents API for retry functionality
    mockDocumentsApi.getFailedCount.mockResolvedValue({
      data: { transient: 0, permanent: 0, unknown: 0, total: 0 },
    });
    mockDocumentsApi.retry.mockResolvedValue({ data: mockDocuments[0] });
    mockDocumentsApi.retryFailed.mockResolvedValue({
      data: { reset_count: 0, message: 'No documents to retry' },
    });
  });

  afterEach(() => {
    // Clean up any hanging promises
    vi.clearAllMocks();
  });

  describe('Modal Rendering', () => {
    it('should not render when showCollectionDetailsModal is null', () => {
      mockShowCollectionDetailsModal.mockReturnValue(null);
      const { container } = render(
        <TestWrapper>
          <CollectionDetailsModal />
        </TestWrapper>
      );
      expect(container.firstChild).toBeNull();
    });

    it('should render modal with collection data', async () => {
      mockShowCollectionDetailsModal.mockReturnValue('test-collection-id');
      render(
        <TestWrapper>
          <CollectionDetailsModal />
        </TestWrapper>
      );

      await waitFor(() => {
        expect(screen.getByText('Test Collection')).toBeInTheDocument();
      });

      // Stats are now displayed in a grid with separate stats
      expect(screen.getByText('Documents')).toBeInTheDocument();
      expect(screen.getByText('Vectors')).toBeInTheDocument();
      expect(screen.getByText('Operations')).toBeInTheDocument();
      // Check actual values
      expect(screen.getByText('100')).toBeInTheDocument(); // document_count
      expect(screen.getByText('500')).toBeInTheDocument(); // vector_count
    });

    it('should show loading state while fetching data', async () => {
      mockShowCollectionDetailsModal.mockReturnValue('test-collection-id');
      // Create a never-resolving promise to keep loading state
      mockCollectionsApi.get.mockImplementation(() => new Promise(() => {}));

      render(
        <TestWrapper>
          <CollectionDetailsModal />
        </TestWrapper>
      );

      // Wait for the loading spinner to appear
      await waitFor(() => {
        const spinner = document.querySelector('.animate-spin');
        expect(spinner).toBeInTheDocument();
      });

      // The component shows "Loading..." in the header
      const heading = screen.getByRole('heading', { level: 2 });
      expect(heading).toHaveTextContent('Loading...');
    });
  });

  describe('Tab Navigation', () => {
    beforeEach(() => {
      mockShowCollectionDetailsModal.mockReturnValue('test-collection-id');
    });

    it('should render all tabs', async () => {
      render(
        <TestWrapper>
          <CollectionDetailsModal />
        </TestWrapper>
      );

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /overview/i })).toBeInTheDocument();
      });

      expect(screen.getByRole('button', { name: /jobs/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /files/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /settings/i })).toBeInTheDocument();
    });

    it('should show overview tab by default', async () => {
      render(
        <TestWrapper>
          <CollectionDetailsModal />
        </TestWrapper>
      );

      await waitFor(() => {
        expect(screen.getByText('Statistics')).toBeInTheDocument();
      });

      // Overview tab shows Statistics with Documents, Vectors, etc.
      expect(screen.getByText('Documents')).toBeInTheDocument();
      expect(screen.getByText('Vectors')).toBeInTheDocument();
    });

    it('should switch to jobs tab and show operations', async () => {
      render(
        <TestWrapper>
          <CollectionDetailsModal />
        </TestWrapper>
      );

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /jobs/i })).toBeInTheDocument();
      });

      await user.click(screen.getByRole('button', { name: /jobs/i }));

      // Jobs tab shows a table with operations
      await waitFor(() => {
        expect(screen.getByText('Type')).toBeInTheDocument();
      });

      // Check table headers
      expect(screen.getByText('Status')).toBeInTheDocument();
      expect(screen.getByText('Started')).toBeInTheDocument();
      expect(screen.getByText('Duration')).toBeInTheDocument();
      // Check operation statuses
      expect(screen.getByText('completed')).toBeInTheDocument();
      expect(screen.getByText('processing')).toBeInTheDocument();
    });

    it('should switch to files tab and show documents', async () => {
      render(
        <TestWrapper>
          <CollectionDetailsModal />
        </TestWrapper>
      );

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /files/i })).toBeInTheDocument();
      });

      await user.click(screen.getByRole('button', { name: /files/i }));

      await waitFor(() => {
        expect(screen.getByText('Files (2)')).toBeInTheDocument();
      });

      // Documents display file_name
      expect(screen.getByText('file1.txt')).toBeInTheDocument();
      expect(screen.getByText('file2.txt')).toBeInTheDocument();
    });

    it('should switch to settings tab', async () => {
      render(
        <TestWrapper>
          <CollectionDetailsModal />
        </TestWrapper>
      );

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /settings/i })).toBeInTheDocument();
      });

      await user.click(screen.getByRole('button', { name: /settings/i }));

      await waitFor(() => {
        expect(screen.getByText('Configuration')).toBeInTheDocument();
      });

      // Settings shows read-only configuration and reindex section
      expect(screen.getByText('Embedding Model')).toBeInTheDocument();
      // Use role to find the re-index button specifically
      expect(screen.getByRole('button', { name: /re-index collection/i })).toBeInTheDocument();
    });
  });

  describe('Action Buttons', () => {
    beforeEach(() => {
      mockShowCollectionDetailsModal.mockReturnValue('test-collection-id');
    });

    it('should open add data modal when clicking Add Data', async () => {
      render(
        <TestWrapper>
          <CollectionDetailsModal />
        </TestWrapper>
      );

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /add data/i })).toBeInTheDocument();
      });

      await user.click(screen.getByRole('button', { name: /add data/i }));

      expect(screen.getByTestId('add-data-modal')).toBeInTheDocument();
    });

    it('should open rename modal when clicking Rename', async () => {
      render(
        <TestWrapper>
          <CollectionDetailsModal />
        </TestWrapper>
      );

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /rename/i })).toBeInTheDocument();
      });

      await user.click(screen.getByRole('button', { name: /rename/i }));

      expect(screen.getByTestId('rename-modal')).toBeInTheDocument();
    });

    it('should open delete modal when clicking Delete', async () => {
      render(
        <TestWrapper>
          <CollectionDetailsModal />
        </TestWrapper>
      );

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /delete/i })).toBeInTheDocument();
      });

      await user.click(screen.getByRole('button', { name: /delete/i }));

      expect(screen.getByTestId('delete-modal')).toBeInTheDocument();
    });

    it('should disable action buttons when collection is not loaded', () => {
      mockShowCollectionDetailsModal.mockReturnValue('test-collection-id');
      mockCollectionsApi.get.mockImplementation(() => new Promise(() => {})); // Never resolves

      render(
        <TestWrapper>
          <CollectionDetailsModal />
        </TestWrapper>
      );

      expect(screen.getByRole('button', { name: /add data/i })).toBeDisabled();
      expect(screen.getByRole('button', { name: /rename/i })).toBeDisabled();
      expect(screen.getByRole('button', { name: /delete/i })).toBeDisabled();
    });
  });

  describe('Modal Operations', () => {
    beforeEach(() => {
      mockShowCollectionDetailsModal.mockReturnValue('test-collection-id');
    });

    it('should handle successful add data operation', async () => {
      render(
        <TestWrapper>
          <CollectionDetailsModal />
        </TestWrapper>
      );

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /add data/i })).toBeInTheDocument();
      });

      await user.click(screen.getByRole('button', { name: /add data/i }));
      await user.click(screen.getByText('Add Data Success'));

      expect(mockInvalidateQueries).toHaveBeenCalledWith({ queryKey: ['collections', 'detail', 'test-collection-id'] });
      expect(mockAddToast).toHaveBeenCalledWith({
        type: 'success',
        message: 'Files uploaded successfully',
      });
    });

    it('should handle successful rename operation', async () => {
      render(
        <TestWrapper>
          <CollectionDetailsModal />
        </TestWrapper>
      );

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /rename/i })).toBeInTheDocument();
      });

      await user.click(screen.getByRole('button', { name: /rename/i }));
      await user.click(screen.getByText('Rename Success'));

      expect(mockInvalidateQueries).toHaveBeenCalledWith({ queryKey: ['collections'] });
      expect(mockAddToast).toHaveBeenCalledWith({
        type: 'success',
        message: 'Collection renamed successfully',
      });
    });

    it('should handle successful delete operation', async () => {
      render(
        <TestWrapper>
          <CollectionDetailsModal />
        </TestWrapper>
      );

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /delete/i })).toBeInTheDocument();
      });

      await user.click(screen.getByRole('button', { name: /delete/i }));
      await user.click(screen.getByText('Delete Success'));

      expect(mockSetShowCollectionDetailsModal).toHaveBeenCalledWith(null);
      expect(mockInvalidateQueries).toHaveBeenCalledWith({ queryKey: ['collections'] });
      expect(mockAddToast).toHaveBeenCalledWith({
        type: 'success',
        message: 'Collection deleted successfully',
      });
    });
  });

  describe('Close Functionality', () => {
    beforeEach(() => {
      mockShowCollectionDetailsModal.mockReturnValue('test-collection-id');
    });

    it('should close modal when clicking close button', async () => {
      render(
        <TestWrapper>
          <CollectionDetailsModal />
        </TestWrapper>
      );

      await waitFor(() => {
        expect(screen.getByText('Test Collection')).toBeInTheDocument();
      });

      // Find the close button - it contains the XCircle icon (lucide-circle-x class)
      const closeButton = document.querySelector('button .lucide-circle-x')?.parentElement as HTMLButtonElement;
      expect(closeButton).toBeTruthy();
      await user.click(closeButton);

      expect(mockSetShowCollectionDetailsModal).toHaveBeenCalledWith(null);
    });

    it('should close modal when clicking backdrop', async () => {
      render(
        <TestWrapper>
          <CollectionDetailsModal />
        </TestWrapper>
      );

      await waitFor(() => {
        expect(screen.getByText('Test Collection')).toBeInTheDocument();
      });

      // Find the backdrop by its class (using CSS variables now)
      const backdrop = document.querySelector('.fixed.inset-0.backdrop-blur-sm') as HTMLDivElement;
      expect(backdrop).toBeTruthy();
      await user.click(backdrop);

      expect(mockSetShowCollectionDetailsModal).toHaveBeenCalledWith(null);
    });
  });

  describe('Settings Tab Functionality', () => {
    beforeEach(() => {
      mockShowCollectionDetailsModal.mockReturnValue('test-collection-id');
    });

    it('should display current configuration values', async () => {
      render(
        <TestWrapper>
          <CollectionDetailsModal />
        </TestWrapper>
      );

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /settings/i })).toBeInTheDocument();
      });

      await user.click(screen.getByRole('button', { name: /settings/i }));

      // Settings tab shows Configuration heading and embedding model
      await waitFor(() => {
        expect(screen.getByText('Configuration')).toBeInTheDocument();
      });
      expect(screen.getByText('Embedding Model')).toBeInTheDocument();
      expect(screen.getByText('sentence-transformers/all-MiniLM-L6-v2')).toBeInTheDocument();
    });

    it('should show reindex button', async () => {
      render(
        <TestWrapper>
          <CollectionDetailsModal />
        </TestWrapper>
      );

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /settings/i })).toBeInTheDocument();
      });

      await user.click(screen.getByRole('button', { name: /settings/i }));

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /re-index collection/i })).toBeInTheDocument();
      });
    });

    it('should show reindex warning', async () => {
      render(
        <TestWrapper>
          <CollectionDetailsModal />
        </TestWrapper>
      );

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /settings/i })).toBeInTheDocument();
      });

      await user.click(screen.getByRole('button', { name: /settings/i }));

      // Check that the warning message is shown (now in an amber banner)
      await waitFor(() => {
        expect(screen.getByText('Action Required')).toBeInTheDocument();
      });
      expect(screen.getByText(/Re-indexing will delete all vectors/)).toBeInTheDocument();
    });

    it('should open reindex modal when clicking re-index button', async () => {
      render(
        <TestWrapper>
          <CollectionDetailsModal />
        </TestWrapper>
      );

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /settings/i })).toBeInTheDocument();
      });

      await user.click(screen.getByRole('button', { name: /settings/i }));

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /re-index collection/i })).toBeInTheDocument();
      });

      await user.click(screen.getByRole('button', { name: /re-index collection/i }));

      expect(screen.getByTestId('reindex-modal')).toBeInTheDocument();
    });
  });

  describe('Documents Pagination', () => {
    beforeEach(() => {
      mockShowCollectionDetailsModal.mockReturnValue('test-collection-id');
      mockCollectionsApi.listDocuments.mockResolvedValue({
        data: {
          documents: mockDocuments,
          total: 100,
          page: 1,
          per_page: 50,
        },
      });
    });

    it('should display pagination controls when documents exceed per_page', async () => {
      render(
        <TestWrapper>
          <CollectionDetailsModal />
        </TestWrapper>
      );

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /files/i })).toBeInTheDocument();
      });

      await user.click(screen.getByRole('button', { name: /files/i }));

      await waitFor(() => {
        expect(screen.getByText('Page 1')).toBeInTheDocument();
      });

      expect(screen.getByRole('button', { name: /previous/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /next/i })).toBeInTheDocument();
    });

    it('should disable previous button on first page', async () => {
      render(
        <TestWrapper>
          <CollectionDetailsModal />
        </TestWrapper>
      );

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /files/i })).toBeInTheDocument();
      });

      await user.click(screen.getByRole('button', { name: /files/i }));

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /previous/i })).toBeDisabled();
      });
    });

    it('should navigate to next page', async () => {
      render(
        <TestWrapper>
          <CollectionDetailsModal />
        </TestWrapper>
      );

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /files/i })).toBeInTheDocument();
      });

      await user.click(screen.getByRole('button', { name: /files/i }));

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /next/i })).toBeInTheDocument();
      });

      // Click next
      await user.click(screen.getByRole('button', { name: /next/i }));

      expect(mockCollectionsApi.listDocuments).toHaveBeenCalledWith('test-collection-id', {
        page: 2,
        limit: 50,
      });
    });

    it('should not display pagination when documents fit on one page', async () => {
      mockCollectionsApi.listDocuments.mockResolvedValue({
        data: {
          documents: mockDocuments,
          total: 2,
          page: 1,
          per_page: 50,
        },
      });

      render(
        <TestWrapper>
          <CollectionDetailsModal />
        </TestWrapper>
      );

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /files/i })).toBeInTheDocument();
      });

      await user.click(screen.getByRole('button', { name: /files/i }));

      await waitFor(() => {
        expect(screen.getByText('Files (2)')).toBeInTheDocument();
      });

      expect(screen.queryByText(/Page 1/)).not.toBeInTheDocument();
      expect(screen.queryByRole('button', { name: /previous/i })).not.toBeInTheDocument();
      expect(screen.queryByRole('button', { name: /next/i })).not.toBeInTheDocument();
    });
  });

  describe('Document Status and Retry Actions', () => {
    beforeEach(() => {
      mockShowCollectionDetailsModal.mockReturnValue('test-collection-id');
    });

    it('should display failed document status and show retry button', async () => {
      const failedDoc = {
        ...mockDocuments[0],
        id: 'failed-doc',
        status: 'failed' as const,
        error_message: 'Extraction error',
        error_category: 'transient' as const,
        retry_count: 1,
        chunk_count: 0,
      };

      mockCollectionsApi.listDocuments.mockResolvedValue({
        data: {
          documents: [failedDoc],
          total: 1,
          page: 1,
          per_page: 50,
        },
      });
      mockDocumentsApi.getFailedCount.mockResolvedValue({
        data: { transient: 1, permanent: 0, unknown: 0, total: 1 },
      });

      render(
        <TestWrapper>
          <CollectionDetailsModal />
        </TestWrapper>
      );

      await user.click(await screen.findByRole('button', { name: /files/i }));

      // Status shows as "failed"
      await waitFor(() => {
        expect(screen.getByText('failed')).toBeInTheDocument();
      });
    });

    it('should show retry all failed button when there are retryable documents', async () => {
      const failedDoc = {
        ...mockDocuments[0],
        id: 'failed-doc',
        status: 'failed' as const,
        error_category: 'transient' as const,
      };

      mockCollectionsApi.listDocuments.mockResolvedValue({
        data: {
          documents: [failedDoc],
          total: 1,
          page: 1,
          per_page: 50,
        },
      });
      mockDocumentsApi.getFailedCount.mockResolvedValue({
        data: { transient: 1, permanent: 0, unknown: 0, total: 1 },
      });
      mockDocumentsApi.retryFailed.mockResolvedValue({
        data: { reset_count: 1, message: 'Reset 1 failed document(s) for retry' },
      });

      render(
        <TestWrapper>
          <CollectionDetailsModal />
        </TestWrapper>
      );

      await user.click(await screen.findByRole('button', { name: /files/i }));

      const retryAllButton = await screen.findByRole('button', { name: /retry all failed/i });
      expect(retryAllButton).toBeInTheDocument();
      await user.click(retryAllButton);

      await waitFor(() => {
        expect(mockDocumentsApi.retryFailed).toHaveBeenCalledWith('test-collection-id');
      });
    });

    it('should display document status correctly', async () => {
      const completedDoc = { ...mockDocuments[0], id: 'completed-doc', status: 'completed' as const };

      mockCollectionsApi.listDocuments.mockResolvedValue({
        data: {
          documents: [completedDoc],
          total: 1,
          page: 1,
          per_page: 50,
        },
      });

      render(
        <TestWrapper>
          <CollectionDetailsModal />
        </TestWrapper>
      );

      await user.click(await screen.findByRole('button', { name: /files/i }));

      await waitFor(() => {
        expect(screen.getByText('completed')).toBeInTheDocument();
      });
    });
  });

  describe('Visualize Tab', () => {
    beforeEach(() => {
      mockShowCollectionDetailsModal.mockReturnValue('test-collection-id');
    });

    it('should render EmbeddingVisualizationTab with collection props', async () => {
      render(
        <TestWrapper>
          <CollectionDetailsModal />
        </TestWrapper>
      );

      await user.click(await screen.findByRole('button', { name: /visualize/i }));

      const viz = await screen.findByTestId('embedding-visualization');
      expect(viz).toBeInTheDocument();
      // Verify the collection ID is passed via data attribute from our mock
      expect(viz).toHaveAttribute('data-collection-id', 'test-collection-id');
    });
  });

  describe('Statistics Display', () => {
    beforeEach(() => {
      mockShowCollectionDetailsModal.mockReturnValue('test-collection-id');
    });

    it('should format numbers correctly', async () => {
      const largeCollection = {
        ...mockCollection,
        document_count: 1234567,
        vector_count: 9876543,
        total_size_bytes: 1073741824, // 1 GB
      };
      mockCollectionsApi.get.mockResolvedValue({ data: largeCollection });

      render(
        <TestWrapper>
          <CollectionDetailsModal />
        </TestWrapper>
      );

      await waitFor(() => {
        expect(screen.getByText('1,234,567')).toBeInTheDocument(); // Documents
      });

      expect(screen.getByText('9,876,543')).toBeInTheDocument(); // Vectors
      expect(screen.getByText('1 GB')).toBeInTheDocument(); // Total Size
    });

    it('should handle zero values gracefully', async () => {
      const emptyCollection = {
        ...mockCollection,
        document_count: 0,
        vector_count: 0,
        total_size_bytes: 0,
      };

      mockCollectionsApi.get.mockResolvedValue({ data: emptyCollection });

      render(
        <TestWrapper>
          <CollectionDetailsModal />
        </TestWrapper>
      );

      await waitFor(() => {
        expect(screen.getByText('Statistics')).toBeInTheDocument();
      });

      // Check for 0 Bytes in the statistics section
      expect(screen.getByText('0 Bytes')).toBeInTheDocument();
    });
  });

  describe('Chunking Strategy Display', () => {
    beforeEach(() => {
      mockShowCollectionDetailsModal.mockReturnValue('test-collection-id');
    });

    it('should display modern chunking strategy in settings tab', async () => {
      mockCollectionsApi.get.mockResolvedValue({ data: mockCollectionWithChunkingStrategy });

      render(
        <TestWrapper>
          <CollectionDetailsModal />
        </TestWrapper>
      );

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /settings/i })).toBeInTheDocument();
      });

      await user.click(screen.getByRole('button', { name: /settings/i }));

      await waitFor(() => {
        expect(screen.getByText('Configuration')).toBeInTheDocument();
      });

      // Should show strategy name (Recursive is the display name)
      expect(screen.getByText('Recursive')).toBeInTheDocument();
    });

    it('should display legacy chunking warning in settings tab', async () => {
      mockCollectionsApi.get.mockResolvedValue({ data: mockCollectionWithLegacyChunking });

      render(
        <TestWrapper>
          <CollectionDetailsModal />
        </TestWrapper>
      );

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /settings/i })).toBeInTheDocument();
      });

      await user.click(screen.getByRole('button', { name: /settings/i }));

      await waitFor(() => {
        expect(screen.getByText('Configuration')).toBeInTheDocument();
      });

      // Should show legacy warning
      expect(screen.getByText('Legacy Chunking')).toBeInTheDocument();
    });
  });

  describe('Reindex Success Handler', () => {
    beforeEach(() => {
      mockShowCollectionDetailsModal.mockReturnValue('test-collection-id');
    });

    it('should handle successful reindex operation', async () => {
      render(
        <TestWrapper>
          <CollectionDetailsModal />
        </TestWrapper>
      );

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /settings/i })).toBeInTheDocument();
      });

      await user.click(screen.getByRole('button', { name: /settings/i }));

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /re-index collection/i })).toBeInTheDocument();
      });

      await user.click(screen.getByRole('button', { name: /re-index collection/i }));
      await user.click(screen.getByText('Reindex Success'));

      expect(mockInvalidateQueries).toHaveBeenCalledWith({ queryKey: ['collections', 'detail', 'test-collection-id'] });
      expect(mockInvalidateQueries).toHaveBeenCalledWith({ queryKey: ['operations', 'list', 'test-collection-id'] });
      expect(mockAddToast).toHaveBeenCalledWith({
        type: 'success',
        message: 'Re-indexing started successfully',
      });

      // Should not show the reindex modal anymore
      expect(screen.queryByTestId('reindex-modal')).not.toBeInTheDocument();
    });
  });

  describe('Nested Modal Interactions', () => {
    beforeEach(() => {
      mockShowCollectionDetailsModal.mockReturnValue('test-collection-id');
    });

    it('should properly handle closing nested modals', async () => {
      render(
        <TestWrapper>
          <CollectionDetailsModal />
        </TestWrapper>
      );

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /add data/i })).toBeInTheDocument();
      });

      // Open add data modal
      await user.click(screen.getByRole('button', { name: /add data/i }));
      expect(screen.getByTestId('add-data-modal')).toBeInTheDocument();

      // Close add data modal
      await user.click(screen.getByText('Close Add Data'));
      expect(screen.queryByTestId('add-data-modal')).not.toBeInTheDocument();

      // Main modal should still be open
      expect(screen.getByText('Test Collection')).toBeInTheDocument();
    });

    it('should handle multiple modals opening and closing sequentially', async () => {
      render(
        <TestWrapper>
          <CollectionDetailsModal />
        </TestWrapper>
      );

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /add data/i })).toBeInTheDocument();
      });

      // Open and close add data modal
      await user.click(screen.getByRole('button', { name: /add data/i }));
      expect(screen.getByTestId('add-data-modal')).toBeInTheDocument();
      await user.click(screen.getByText('Close Add Data'));

      // Open and close rename modal
      await user.click(screen.getByRole('button', { name: /rename/i }));
      expect(screen.getByTestId('rename-modal')).toBeInTheDocument();
      await user.click(screen.getByText('Close Rename'));

      // Open and close delete modal
      await user.click(screen.getByRole('button', { name: /delete/i }));
      expect(screen.getByTestId('delete-modal')).toBeInTheDocument();
      await user.click(screen.getByText('Close Delete'));

      // Main modal should still be open
      expect(screen.getByText('Test Collection')).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    beforeEach(() => {
      mockShowCollectionDetailsModal.mockReturnValue('test-collection-id');
    });

    it('should have proper navigation element with Tabs label', async () => {
      render(
        <TestWrapper>
          <CollectionDetailsModal />
        </TestWrapper>
      );

      await waitFor(() => {
        expect(screen.getByText('Test Collection')).toBeInTheDocument();
      });

      // Check for navigation
      expect(screen.getByLabelText('Tabs')).toBeInTheDocument();
    });

    it('should contain all interactive elements within the modal', async () => {
      render(
        <TestWrapper>
          <CollectionDetailsModal />
        </TestWrapper>
      );

      await waitFor(() => {
        expect(screen.getByText('Test Collection')).toBeInTheDocument();
      });

      // Test that all interactive elements are within the modal
      const modal = document.querySelector('.fixed.inset-4');
      expect(modal).toBeTruthy();
      const buttons = screen.getAllByRole('button');

      buttons.forEach((button) => {
        expect(modal).toContainElement(button);
      });
    });
  });
});
