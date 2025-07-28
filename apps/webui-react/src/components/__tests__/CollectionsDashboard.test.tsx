import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { vi } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import CollectionsDashboard from '../CollectionsDashboard';
import type { Collection } from '../../types/collection';

// Mock the useCollections hook
vi.mock('../../hooks/useCollections', () => ({
  useCollections: vi.fn(),
}));

// Mock the CreateCollectionModal component
vi.mock('../CreateCollectionModal', () => ({
  default: ({ onClose, onSuccess }: { onClose: () => void; onSuccess: () => void }) => (
    <div data-testid="create-collection-modal">
      <button onClick={onClose}>Close Modal</button>
      <button onClick={onSuccess}>Create Success</button>
    </div>
  ),
}));

// Mock the CollectionCard component
vi.mock('../CollectionCard', () => ({
  default: ({ collection }: { collection: Collection }) => (
    <div data-testid={`collection-card-${collection.id}`}>
      <h3>{collection.name}</h3>
      <p>{collection.description}</p>
      <span>{collection.status}</span>
    </div>
  ),
}));

import { useCollections } from '../../hooks/useCollections';

// Helper function to create a test collection
const createTestCollection = (overrides?: Partial<Collection>): Collection => ({
  id: 'test-id-1',
  name: 'Test Collection',
  description: 'Test description',
  owner_id: 1,
  vector_store_name: 'test_store',
  embedding_model: 'test-model',
  quantization: 'float16',
  chunk_size: 1000,
  chunk_overlap: 200,
  is_public: false,
  status: 'ready',
  document_count: 10,
  vector_count: 100,
  created_at: '2024-01-01T00:00:00Z',
  updated_at: '2024-01-01T12:00:00Z',
  ...overrides,
});

const renderWithQueryClient = (component: React.ReactElement) => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      {component}
    </QueryClientProvider>
  );
};

describe('CollectionsDashboard', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Initial Rendering', () => {
    it('should render collections grid with multiple collections', () => {
      const mockCollections = [
        createTestCollection({ id: 'col-1', name: 'Collection 1' }),
        createTestCollection({ id: 'col-2', name: 'Collection 2' }),
        createTestCollection({ id: 'col-3', name: 'Collection 3' }),
      ];

      vi.mocked(useCollections).mockReturnValue({
        data: mockCollections,
        isLoading: false,
        error: null,
        refetch: vi.fn(),
      } as any);

      renderWithQueryClient(<CollectionsDashboard />);

      expect(screen.getByText('Collections')).toBeInTheDocument();
      expect(screen.getByText('Manage your document collections and knowledge bases')).toBeInTheDocument();
      expect(screen.getByTestId('collection-card-col-1')).toBeInTheDocument();
      expect(screen.getByTestId('collection-card-col-2')).toBeInTheDocument();
      expect(screen.getByTestId('collection-card-col-3')).toBeInTheDocument();
    });

    it('should sort collections by updated_at (most recent first)', () => {
      const mockCollections = [
        createTestCollection({ 
          id: 'col-1', 
          name: 'Oldest Collection',
          updated_at: '2024-01-01T00:00:00Z' 
        }),
        createTestCollection({ 
          id: 'col-2', 
          name: 'Newest Collection',
          updated_at: '2024-01-03T00:00:00Z' 
        }),
        createTestCollection({ 
          id: 'col-3', 
          name: 'Middle Collection',
          updated_at: '2024-01-02T00:00:00Z' 
        }),
      ];

      vi.mocked(useCollections).mockReturnValue({
        data: mockCollections,
        isLoading: false,
        error: null,
        refetch: vi.fn(),
      } as any);

      renderWithQueryClient(<CollectionsDashboard />);

      const collectionCards = screen.getAllByTestId(/^collection-card-/);
      expect(collectionCards[0]).toHaveAttribute('data-testid', 'collection-card-col-2');
      expect(collectionCards[1]).toHaveAttribute('data-testid', 'collection-card-col-3');
      expect(collectionCards[2]).toHaveAttribute('data-testid', 'collection-card-col-1');
    });
  });

  describe('Search Functionality', () => {
    it('should filter collections by name', () => {
      const mockCollections = [
        createTestCollection({ id: 'col-1', name: 'Machine Learning Dataset' }),
        createTestCollection({ id: 'col-2', name: 'Documentation Archive' }),
        createTestCollection({ id: 'col-3', name: 'Research Papers' }),
      ];

      vi.mocked(useCollections).mockReturnValue({
        data: mockCollections,
        isLoading: false,
        error: null,
        refetch: vi.fn(),
      } as any);

      renderWithQueryClient(<CollectionsDashboard />);

      const searchInput = screen.getByPlaceholderText('Search collections...');
      fireEvent.change(searchInput, { target: { value: 'machine' } });

      expect(screen.getByTestId('collection-card-col-1')).toBeInTheDocument();
      expect(screen.queryByTestId('collection-card-col-2')).not.toBeInTheDocument();
      expect(screen.queryByTestId('collection-card-col-3')).not.toBeInTheDocument();
    });

    it('should filter collections by description', () => {
      const mockCollections = [
        createTestCollection({ id: 'col-1', description: 'Contains ML datasets' }),
        createTestCollection({ id: 'col-2', description: 'Product documentation' }),
        createTestCollection({ id: 'col-3', description: 'Academic papers' }),
      ];

      vi.mocked(useCollections).mockReturnValue({
        data: mockCollections,
        isLoading: false,
        error: null,
        refetch: vi.fn(),
      } as any);

      renderWithQueryClient(<CollectionsDashboard />);

      const searchInput = screen.getByPlaceholderText('Search collections...');
      fireEvent.change(searchInput, { target: { value: 'documentation' } });

      expect(screen.queryByTestId('collection-card-col-1')).not.toBeInTheDocument();
      expect(screen.getByTestId('collection-card-col-2')).toBeInTheDocument();
      expect(screen.queryByTestId('collection-card-col-3')).not.toBeInTheDocument();
    });

    it('should perform case-insensitive search', () => {
      const mockCollections = [
        createTestCollection({ id: 'col-1', name: 'Machine Learning' }),
        createTestCollection({ id: 'col-2', name: 'machine learning' }),
        createTestCollection({ id: 'col-3', name: 'MACHINE LEARNING' }),
      ];

      vi.mocked(useCollections).mockReturnValue({
        data: mockCollections,
        isLoading: false,
        error: null,
        refetch: vi.fn(),
      } as any);

      renderWithQueryClient(<CollectionsDashboard />);

      const searchInput = screen.getByPlaceholderText('Search collections...');
      fireEvent.change(searchInput, { target: { value: 'MaChInE' } });

      expect(screen.getByTestId('collection-card-col-1')).toBeInTheDocument();
      expect(screen.getByTestId('collection-card-col-2')).toBeInTheDocument();
      expect(screen.getByTestId('collection-card-col-3')).toBeInTheDocument();
    });

    it('should show results count when searching', () => {
      const mockCollections = [
        createTestCollection({ id: 'col-1', name: 'Test Collection 1' }),
        createTestCollection({ id: 'col-2', name: 'Test Collection 2' }),
        createTestCollection({ id: 'col-3', name: 'Different Name' }),
      ];

      vi.mocked(useCollections).mockReturnValue({
        data: mockCollections,
        isLoading: false,
        error: null,
        refetch: vi.fn(),
      } as any);

      renderWithQueryClient(<CollectionsDashboard />);

      const searchInput = screen.getByPlaceholderText('Search collections...');
      fireEvent.change(searchInput, { target: { value: 'Test Collection' } });

      expect(screen.getByText('Found 2 collections')).toBeInTheDocument();
    });

    it('should show singular result count for one match', () => {
      const mockCollections = [
        createTestCollection({ id: 'col-1', name: 'Unique Name' }),
        createTestCollection({ id: 'col-2', name: 'Different' }),
      ];

      vi.mocked(useCollections).mockReturnValue({
        data: mockCollections,
        isLoading: false,
        error: null,
        refetch: vi.fn(),
      } as any);

      renderWithQueryClient(<CollectionsDashboard />);

      const searchInput = screen.getByPlaceholderText('Search collections...');
      fireEvent.change(searchInput, { target: { value: 'Unique' } });

      expect(screen.getByText('Found 1 collection')).toBeInTheDocument();
    });
  });

  describe('Status Filtering', () => {
    it('should filter collections by status', () => {
      const mockCollections = [
        createTestCollection({ id: 'col-1', status: 'ready' }),
        createTestCollection({ id: 'col-2', status: 'processing' }),
        createTestCollection({ id: 'col-3', status: 'error' }),
        createTestCollection({ id: 'col-4', status: 'degraded' }),
      ];

      vi.mocked(useCollections).mockReturnValue({
        data: mockCollections,
        isLoading: false,
        error: null,
        refetch: vi.fn(),
      } as any);

      renderWithQueryClient(<CollectionsDashboard />);

      const statusFilter = screen.getByLabelText('Filter collections by status');
      
      // Filter by ready status
      fireEvent.change(statusFilter, { target: { value: 'ready' } });
      expect(screen.getByTestId('collection-card-col-1')).toBeInTheDocument();
      expect(screen.queryByTestId('collection-card-col-2')).not.toBeInTheDocument();
      expect(screen.queryByTestId('collection-card-col-3')).not.toBeInTheDocument();
      expect(screen.queryByTestId('collection-card-col-4')).not.toBeInTheDocument();

      // Filter by processing status
      fireEvent.change(statusFilter, { target: { value: 'processing' } });
      expect(screen.queryByTestId('collection-card-col-1')).not.toBeInTheDocument();
      expect(screen.getByTestId('collection-card-col-2')).toBeInTheDocument();
      expect(screen.queryByTestId('collection-card-col-3')).not.toBeInTheDocument();
      expect(screen.queryByTestId('collection-card-col-4')).not.toBeInTheDocument();
    });

    it('should show all collections when "All Status" is selected', () => {
      const mockCollections = [
        createTestCollection({ id: 'col-1', status: 'ready' }),
        createTestCollection({ id: 'col-2', status: 'processing' }),
        createTestCollection({ id: 'col-3', status: 'error' }),
      ];

      vi.mocked(useCollections).mockReturnValue({
        data: mockCollections,
        isLoading: false,
        error: null,
        refetch: vi.fn(),
      } as any);

      renderWithQueryClient(<CollectionsDashboard />);

      const statusFilter = screen.getByLabelText('Filter collections by status');
      fireEvent.change(statusFilter, { target: { value: 'all' } });

      expect(screen.getByTestId('collection-card-col-1')).toBeInTheDocument();
      expect(screen.getByTestId('collection-card-col-2')).toBeInTheDocument();
      expect(screen.getByTestId('collection-card-col-3')).toBeInTheDocument();
    });

    it('should show results count when filtering by status', () => {
      const mockCollections = [
        createTestCollection({ id: 'col-1', status: 'ready' }),
        createTestCollection({ id: 'col-2', status: 'ready' }),
        createTestCollection({ id: 'col-3', status: 'error' }),
      ];

      vi.mocked(useCollections).mockReturnValue({
        data: mockCollections,
        isLoading: false,
        error: null,
        refetch: vi.fn(),
      } as any);

      renderWithQueryClient(<CollectionsDashboard />);

      const statusFilter = screen.getByLabelText('Filter collections by status');
      fireEvent.change(statusFilter, { target: { value: 'ready' } });

      expect(screen.getByText('Found 2 collections')).toBeInTheDocument();
    });
  });

  describe('Combined Filtering', () => {
    it('should apply both search and status filters', () => {
      const mockCollections = [
        createTestCollection({ id: 'col-1', name: 'ML Dataset', status: 'ready' }),
        createTestCollection({ id: 'col-2', name: 'ML Papers', status: 'processing' }),
        createTestCollection({ id: 'col-3', name: 'Documentation', status: 'ready' }),
      ];

      vi.mocked(useCollections).mockReturnValue({
        data: mockCollections,
        isLoading: false,
        error: null,
        refetch: vi.fn(),
      } as any);

      renderWithQueryClient(<CollectionsDashboard />);

      const searchInput = screen.getByPlaceholderText('Search collections...');
      const statusFilter = screen.getByLabelText('Filter collections by status');

      fireEvent.change(searchInput, { target: { value: 'ML' } });
      fireEvent.change(statusFilter, { target: { value: 'ready' } });

      expect(screen.getByTestId('collection-card-col-1')).toBeInTheDocument();
      expect(screen.queryByTestId('collection-card-col-2')).not.toBeInTheDocument();
      expect(screen.queryByTestId('collection-card-col-3')).not.toBeInTheDocument();
      expect(screen.getByText('Found 1 collection')).toBeInTheDocument();
    });
  });

  describe('Empty States', () => {
    it('should show empty state when no collections exist', () => {
      vi.mocked(useCollections).mockReturnValue({
        data: [],
        isLoading: false,
        error: null,
        refetch: vi.fn(),
      } as any);

      renderWithQueryClient(<CollectionsDashboard />);

      expect(screen.getByText('No collections yet')).toBeInTheDocument();
      expect(screen.getByText('Get started by creating your first collection.')).toBeInTheDocument();
      
      // Should show the create button in empty state
      const createButtons = screen.getAllByText('Create Collection');
      expect(createButtons.length).toBeGreaterThan(1); // Header and empty state
    });

    it('should show no results message when search returns no matches', () => {
      const mockCollections = [
        createTestCollection({ id: 'col-1', name: 'Collection 1' }),
        createTestCollection({ id: 'col-2', name: 'Collection 2' }),
      ];

      vi.mocked(useCollections).mockReturnValue({
        data: mockCollections,
        isLoading: false,
        error: null,
        refetch: vi.fn(),
      } as any);

      renderWithQueryClient(<CollectionsDashboard />);

      const searchInput = screen.getByPlaceholderText('Search collections...');
      fireEvent.change(searchInput, { target: { value: 'nonexistent' } });

      expect(screen.getByText('No collections match your search criteria.')).toBeInTheDocument();
    });

    it('should show no results message when status filter returns no matches', () => {
      const mockCollections = [
        createTestCollection({ id: 'col-1', status: 'ready' }),
        createTestCollection({ id: 'col-2', status: 'ready' }),
      ];

      vi.mocked(useCollections).mockReturnValue({
        data: mockCollections,
        isLoading: false,
        error: null,
        refetch: vi.fn(),
      } as any);

      renderWithQueryClient(<CollectionsDashboard />);

      const statusFilter = screen.getByLabelText('Filter collections by status');
      fireEvent.change(statusFilter, { target: { value: 'error' } });

      expect(screen.getByText('No collections match your search criteria.')).toBeInTheDocument();
    });
  });

  describe('Loading State', () => {
    it('should show loading spinner when loading initial data', () => {
      vi.mocked(useCollections).mockReturnValue({
        data: [],
        isLoading: true,
        error: null,
        refetch: vi.fn(),
      } as any);

      renderWithQueryClient(<CollectionsDashboard />);

      // Check for the spinner by its classes instead of role
      const spinner = document.querySelector('.animate-spin');
      expect(spinner).toBeInTheDocument();
      expect(spinner).toHaveClass('rounded-full', 'h-8', 'w-8', 'border-b-2', 'border-blue-600');
    });

    it('should not show loading spinner when data exists and reloading', () => {
      const mockCollections = [
        createTestCollection({ id: 'col-1', name: 'Collection 1' }),
      ];

      vi.mocked(useCollections).mockReturnValue({
        data: mockCollections,
        isLoading: true, // Loading but data exists
        error: null,
        refetch: vi.fn(),
      } as any);

      renderWithQueryClient(<CollectionsDashboard />);

      // Should show collections, not spinner
      expect(screen.getByTestId('collection-card-col-1')).toBeInTheDocument();
      const spinner = document.querySelector('.animate-spin');
      expect(spinner).not.toBeInTheDocument();
    });
  });

  describe('Error State', () => {
    it('should show error state with retry button when error occurs and no data', () => {
      const mockRefetch = vi.fn();
      const mockError = new Error('Network error');

      vi.mocked(useCollections).mockReturnValue({
        data: [],
        isLoading: false,
        error: mockError,
        refetch: mockRefetch,
      } as any);

      renderWithQueryClient(<CollectionsDashboard />);

      expect(screen.getByText('Failed to load collections')).toBeInTheDocument();
      
      const retryButton = screen.getByText('Retry');
      expect(retryButton).toBeInTheDocument();
      
      fireEvent.click(retryButton);
      expect(mockRefetch).toHaveBeenCalledTimes(1);
    });

    it('should show collections even if error exists when data is present', () => {
      const mockCollections = [
        createTestCollection({ id: 'col-1', name: 'Collection 1' }),
      ];
      const mockError = new Error('Network error');

      vi.mocked(useCollections).mockReturnValue({
        data: mockCollections,
        isLoading: false,
        error: mockError,
        refetch: vi.fn(),
      } as any);

      renderWithQueryClient(<CollectionsDashboard />);

      // Should show collections, not error
      expect(screen.getByTestId('collection-card-col-1')).toBeInTheDocument();
      expect(screen.queryByText('Failed to load collections')).not.toBeInTheDocument();
    });
  });

  describe('Create Collection Modal', () => {
    it('should open create collection modal when header button is clicked', () => {
      vi.mocked(useCollections).mockReturnValue({
        data: [],
        isLoading: false,
        error: null,
        refetch: vi.fn(),
      } as any);

      renderWithQueryClient(<CollectionsDashboard />);

      const createButton = screen.getAllByText('Create Collection')[0]; // Header button
      fireEvent.click(createButton);

      expect(screen.getByTestId('create-collection-modal')).toBeInTheDocument();
    });

    it('should open create collection modal from empty state', () => {
      vi.mocked(useCollections).mockReturnValue({
        data: [],
        isLoading: false,
        error: null,
        refetch: vi.fn(),
      } as any);

      renderWithQueryClient(<CollectionsDashboard />);

      const createButtons = screen.getAllByText('Create Collection');
      const emptyStateButton = createButtons[createButtons.length - 1]; // Last button is in empty state
      fireEvent.click(emptyStateButton);

      expect(screen.getByTestId('create-collection-modal')).toBeInTheDocument();
    });

    it('should close modal when close is triggered', async () => {
      vi.mocked(useCollections).mockReturnValue({
        data: [],
        isLoading: false,
        error: null,
        refetch: vi.fn(),
      } as any);

      renderWithQueryClient(<CollectionsDashboard />);

      const createButton = screen.getAllByText('Create Collection')[0];
      fireEvent.click(createButton);

      const closeButton = screen.getByText('Close Modal');
      fireEvent.click(closeButton);

      await waitFor(() => {
        expect(screen.queryByTestId('create-collection-modal')).not.toBeInTheDocument();
      });
    });

    it('should close modal when success is triggered', async () => {
      vi.mocked(useCollections).mockReturnValue({
        data: [],
        isLoading: false,
        error: null,
        refetch: vi.fn(),
      } as any);

      renderWithQueryClient(<CollectionsDashboard />);

      const createButton = screen.getAllByText('Create Collection')[0];
      fireEvent.click(createButton);

      const successButton = screen.getByText('Create Success');
      fireEvent.click(successButton);

      await waitFor(() => {
        expect(screen.queryByTestId('create-collection-modal')).not.toBeInTheDocument();
      });
    });
  });

  describe('Accessibility', () => {
    it('should have proper aria labels for search input', () => {
      vi.mocked(useCollections).mockReturnValue({
        data: [],
        isLoading: false,
        error: null,
        refetch: vi.fn(),
      } as any);

      renderWithQueryClient(<CollectionsDashboard />);

      const searchInput = screen.getByLabelText('Search collections by name or description');
      expect(searchInput).toBeInTheDocument();
    });

    it('should have proper aria labels for status filter', () => {
      vi.mocked(useCollections).mockReturnValue({
        data: [],
        isLoading: false,
        error: null,
        refetch: vi.fn(),
      } as any);

      renderWithQueryClient(<CollectionsDashboard />);

      const statusFilter = screen.getByLabelText('Filter collections by status');
      expect(statusFilter).toBeInTheDocument();
    });

    it('should have screen reader only labels', () => {
      vi.mocked(useCollections).mockReturnValue({
        data: [],
        isLoading: false,
        error: null,
        refetch: vi.fn(),
      } as any);

      renderWithQueryClient(<CollectionsDashboard />);

      const srOnlyLabels = screen.getAllByText('Search collections').filter(
        element => element.classList.contains('sr-only')
      );
      expect(srOnlyLabels.length).toBeGreaterThan(0);
    });
  });
});