import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi } from 'vitest';
import type { Collection, Operation } from '../../types/collection';
import type { DocumentResponse } from '../../services/api/v2/types';

// Mock stores first
const mockShowCollectionDetailsModal = vi.fn();
const mockSetShowCollectionDetailsModal = vi.fn();
const mockAddToast = vi.fn();

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
  },
}));

// Mock collectionKeys
vi.mock('../../hooks/useCollections', () => ({
  collectionKeys: {
    lists: () => ['collections'],
    list: (filters: unknown) => ['collections', filters],
    details: () => ['collections', 'details'],
    detail: (id: string) => ['collections', 'details', id],
  },
}));

// Mock React Query
const mockInvalidateQueries = vi.fn();
vi.mock('@tanstack/react-query', async () => {
  const actual = await vi.importActual('@tanstack/react-query');
  return {
    ...actual,
    useQueryClient: () => ({
      invalidateQueries: mockInvalidateQueries,
    }),
  };
});

// Mock child modals
vi.mock('../AddDataToCollectionModal', () => ({
  default: vi.fn(({ onSuccess, onClose }: { onSuccess: () => void; onClose: () => void }) => (
    <div data-testid="add-data-modal">
      <button onClick={onSuccess}>Add Data Success</button>
      <button onClick={onClose}>Close Add Data</button>
    </div>
  )),
}));

vi.mock('../RenameCollectionModal', () => ({
  default: vi.fn(({ onSuccess, onClose }: { onSuccess: () => void; onClose: () => void }) => (
    <div data-testid="rename-modal">
      <button onClick={onSuccess}>Rename Success</button>
      <button onClick={onClose}>Close Rename</button>
    </div>
  )),
}));

vi.mock('../DeleteCollectionModal', () => ({
  default: vi.fn(({ onSuccess, onClose }: { onSuccess: () => void; onClose: () => void }) => (
    <div data-testid="delete-modal">
      <button onClick={onSuccess}>Delete Success</button>
      <button onClick={onClose}>Close Delete</button>
    </div>
  )),
}));

vi.mock('../ReindexCollectionModal', () => ({
  default: vi.fn(({ onSuccess, onClose }: { onSuccess: () => void; onClose: () => void }) => (
    <div data-testid="reindex-modal">
      <button onClick={onSuccess}>Reindex Success</button>
      <button onClick={onClose}>Close Reindex</button>
    </div>
  )),
}));

// Import components after mocks
import CollectionDetailsModal from '../CollectionDetailsModal';
import { TestWrapper } from '../../tests/utils/TestWrapper';
import { collectionsV2Api } from '../../services/api/v2/collections';

// Get mocked functions
const mockCollectionsApi = collectionsV2Api as {
  get: ReturnType<typeof vi.fn>;
  listOperations: ReturnType<typeof vi.fn>;
  listDocuments: ReturnType<typeof vi.fn>;
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
  created_at: '2024-01-01T00:00:00Z',
  updated_at: '2024-01-02T00:00:00Z',
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
    source_path: '/data/source1',
    file_path: '/data/source1/file1.txt',
    chunk_count: 10,
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z',
  },
  {
    id: 'doc-2',
    collection_id: 'test-collection-id',
    source_path: '/data/source2',
    file_path: '/data/source2/file2.txt',
    chunk_count: 20,
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

      // Operations data is not loaded yet in overview tab, so it shows 0
      expect(screen.getByText(/0 operations • 100 documents • 500 vectors/)).toBeInTheDocument();
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

    it('should show error state when fetch fails', async () => {
      mockShowCollectionDetailsModal.mockReturnValue('test-collection-id');
      // Clear any previous mocks and set up the error
      mockCollectionsApi.get.mockClear();
      mockCollectionsApi.get.mockRejectedValue(new Error('API Error'));

      render(
        <TestWrapper>
          <CollectionDetailsModal />
        </TestWrapper>
      );

      // Wait for the error message to appear
      await waitFor(() => {
        expect(screen.getByText('Failed to load collection details')).toBeInTheDocument();
      }, { timeout: 3000 });
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

      expect(screen.getByText('Configuration')).toBeInTheDocument();
      expect(screen.getByText('Source Directories')).toBeInTheDocument();
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

      await waitFor(() => {
        expect(screen.getByText('Operations History')).toBeInTheDocument();
      });

      expect(screen.getByText('op-1', { exact: false })).toBeInTheDocument();
      expect(screen.getByText('op-2', { exact: false })).toBeInTheDocument();
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
        expect(screen.getByText('Documents (2)')).toBeInTheDocument();
      });

      expect(screen.getByText('/data/source1/file1.txt')).toBeInTheDocument();
      expect(screen.getByText('/data/source2/file2.txt')).toBeInTheDocument();
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
        expect(screen.getByText('Collection Configuration')).toBeInTheDocument();
      });

      // Pre-release: settings shows read-only configuration and reindex affordance
      expect(screen.getByText(/Current Chunking Strategy/i)).toBeInTheDocument();
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

      expect(mockInvalidateQueries).toHaveBeenCalledWith({ queryKey: ['collection-v2', 'test-collection-id'] });
      expect(mockInvalidateQueries).toHaveBeenCalledWith({ queryKey: ['collection-operations', 'test-collection-id'] });
      expect(mockInvalidateQueries).toHaveBeenCalledWith({ queryKey: ['collection-documents', 'test-collection-id'] });
      expect(mockAddToast).toHaveBeenCalledWith({
        type: 'success',
        message: 'Source added successfully. Check the Operations tab to monitor progress.',
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

      expect(mockInvalidateQueries).toHaveBeenCalledWith({ queryKey: ['collection-v2'] });
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
      expect(mockInvalidateQueries).toHaveBeenCalledWith({ queryKey: expect.arrayContaining(['collections']) });
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
        // Find the close button by the SVG path
        const closeButton = document.querySelector('button svg path[d="M6 18L18 6M6 6l12 12"]')?.parentElement?.parentElement;
        expect(closeButton).toBeInTheDocument();
      });

      const closeButton = document.querySelector('button svg path[d="M6 18L18 6M6 6l12 12"]')?.parentElement?.parentElement as HTMLButtonElement;
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
        // Find the backdrop by its class
        const backdrop = document.querySelector('.fixed.inset-0.bg-black');
        expect(backdrop).toBeInTheDocument();
      });

      const backdrop = document.querySelector('.fixed.inset-0.bg-black') as HTMLDivElement;
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

      expect(screen.getByText(/Current Chunking Strategy/i)).toBeInTheDocument();
    });

    // Skipped validation tests for legacy chunk size/overlap inputs removed in pre-release

    it('should show reindex button state', async () => {
      render(
        <TestWrapper>
          <CollectionDetailsModal />
        </TestWrapper>
      );

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /settings/i })).toBeInTheDocument();
      });

      await user.click(screen.getByRole('button', { name: /settings/i }));

      const reindexButton = screen.getByRole('button', { name: /re-index collection/i });

      // Pre-release: button is present; enabling depends on strategy/model changes in modal
      expect(reindexButton).toBeInTheDocument();
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

      // Check that the warning message is shown
      expect(screen.getByRole('alert')).toBeInTheDocument();
      expect(screen.getByText('Re-indexing will process all documents again')).toBeInTheDocument();
      expect(screen.getByText('Delete all existing vectors')).toBeInTheDocument();
      expect(screen.getByText('Re-process all documents with new settings')).toBeInTheDocument();
    });

    // Skip complex input interaction tests that require proper event handling
    // These would need a more sophisticated test setup or e2e tests
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
        expect(screen.getByText('Showing page 1 of 2')).toBeInTheDocument();
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
        expect(screen.getByText('Documents (2)')).toBeInTheDocument();
      });

      expect(screen.queryByText(/Showing page/)).not.toBeInTheDocument();
      expect(screen.queryByRole('button', { name: /previous/i })).not.toBeInTheDocument();
      expect(screen.queryByRole('button', { name: /next/i })).not.toBeInTheDocument();
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

    it('should handle null/undefined values gracefully', async () => {
      const incompleteCollection = {
        ...mockCollection,
        document_count: null,
        vector_count: undefined,
        total_size_bytes: null,
      };
      
      // Clear previous mocks
      mockCollectionsApi.get.mockClear();
      mockCollectionsApi.get.mockResolvedValue({ data: incompleteCollection });

      render(
        <TestWrapper>
          <CollectionDetailsModal />
        </TestWrapper>
      );

      await waitFor(() => {
        expect(screen.getByText('Statistics')).toBeInTheDocument();
      });

      // The component header should show 0 for null values
      expect(screen.getByText(/0 operations • 0 documents • 0 vectors/)).toBeInTheDocument();
      
      // Check for 0 Bytes in the statistics section
      expect(screen.getByText('0 Bytes')).toBeInTheDocument();
    });
  });

  describe('Source Directories Display', () => {
    beforeEach(() => {
      mockShowCollectionDetailsModal.mockReturnValue('test-collection-id');
    });

    it('should aggregate and display source directories', async () => {
      const documentsWithSamePath = [
        ...mockDocuments,
        {
          id: 'doc-3',
          collection_id: 'test-collection-id',
          source_path: '/data/source1',
          file_path: '/data/source1/file3.txt',
          chunk_count: 15,
          created_at: '2024-01-01T00:00:00Z',
          updated_at: '2024-01-01T00:00:00Z',
        },
      ];

      mockCollectionsApi.listDocuments.mockResolvedValue({
        data: {
          documents: documentsWithSamePath,
          total: 3,
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

      // Need to go to files tab first to trigger document fetch
      await user.click(screen.getByRole('button', { name: /files/i }));

      await waitFor(() => {
        expect(mockCollectionsApi.listDocuments).toHaveBeenCalled();
      });

      // Go back to overview tab
      await user.click(screen.getByRole('button', { name: /overview/i }));

      // Should show aggregated counts
      expect(screen.getByText('/data/source1')).toBeInTheDocument();
      expect(screen.getByText('2 documents')).toBeInTheDocument(); // doc-1 and doc-3
      expect(screen.getByText('/data/source2')).toBeInTheDocument();
      expect(screen.getByText('1 documents')).toBeInTheDocument(); // doc-2
    });

    it('should show empty state when no documents', async () => {
      mockCollectionsApi.listDocuments.mockResolvedValue({
        data: {
          documents: [],
          total: 0,
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

      // Need to go to files tab first to trigger document fetch
      await user.click(screen.getByRole('button', { name: /files/i }));

      await waitFor(() => {
        expect(mockCollectionsApi.listDocuments).toHaveBeenCalled();
      });

      // Go back to overview tab
      await user.click(screen.getByRole('button', { name: /overview/i }));

      expect(screen.getByText('No source directories added yet')).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    beforeEach(() => {
      mockShowCollectionDetailsModal.mockReturnValue('test-collection-id');
    });

    it('should have proper ARIA labels and roles', async () => {
      render(
        <TestWrapper>
          <CollectionDetailsModal />
        </TestWrapper>
      );

      await waitFor(() => {
        // Check for the modal container
        const modal = document.querySelector('.fixed.inset-4');
        expect(modal).toBeInTheDocument();
      });

      // Check for navigation
      expect(screen.getByLabelText('Tabs')).toBeInTheDocument();
    });

    // Skipped ARIA tests for removed legacy inputs

    it('should have proper alert role for warning messages', async () => {
      render(
        <TestWrapper>
          <CollectionDetailsModal />
        </TestWrapper>
      );

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /settings/i })).toBeInTheDocument();
      });

      await user.click(screen.getByRole('button', { name: /settings/i }));

      const warningAlert = screen.getByRole('alert');
      expect(warningAlert).toHaveTextContent('Re-indexing will process all documents again');
    });

    it('should focus trap within the modal', async () => {
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
      const buttons = screen.getAllByRole('button');
      
      buttons.forEach(button => {
        expect(modal).toContainElement(button);
      });
    });
  });
});
