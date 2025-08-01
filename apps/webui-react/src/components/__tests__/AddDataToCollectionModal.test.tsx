import { describe, it, expect, vi, beforeEach, type Mock } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import AddDataToCollectionModal from '../AddDataToCollectionModal';
import { useAddSource } from '../../hooks/useCollectionOperations';
import { useUIStore } from '../../stores/uiStore';
import type { Collection } from '../../types/collection';

// Mock dependencies
vi.mock('react-router-dom', () => ({
  useNavigate: vi.fn(),
}));

vi.mock('../../hooks/useCollectionOperations', () => ({
  useAddSource: vi.fn(),
}));

vi.mock('../../stores/uiStore', () => ({
  useUIStore: vi.fn(),
}));

// Mock collection data
const mockCollection: Collection = {
  id: 'test-collection-id',
  name: 'Test Collection',
  description: 'Test description',
  owner_id: 1,
  vector_store_name: 'test_vector_store',
  embedding_model: 'text-embedding-ada-002',
  quantization: 'float32',
  chunk_size: 1000,
  chunk_overlap: 200,
  is_public: false,
  status: 'ready',
  document_count: 42,
  vector_count: 100,
  total_size_bytes: 1024000,
  created_at: '2024-01-01T00:00:00Z',
  updated_at: '2024-01-01T00:00:00Z',
};

describe('AddDataToCollectionModal', () => {
  let queryClient: QueryClient;
  let mockNavigate: Mock;
  let mockAddToast: Mock;
  let mockMutateAsync: Mock;
  let mockOnClose: Mock;
  let mockOnSuccess: Mock;

  beforeEach(() => {
    // Reset all mocks
    vi.clearAllMocks();

    // Create a new QueryClient for each test
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    });

    // Setup mock functions
    mockNavigate = vi.fn();
    mockAddToast = vi.fn();
    mockMutateAsync = vi.fn();
    mockOnClose = vi.fn();
    mockOnSuccess = vi.fn();

    // Mock useNavigate
    (useNavigate as Mock).mockReturnValue(mockNavigate);

    // Mock useUIStore
    (useUIStore as unknown as Mock).mockReturnValue({
      addToast: mockAddToast,
    });

    // Mock useAddSource
    (useAddSource as Mock).mockReturnValue({
      mutateAsync: mockMutateAsync,
      isPending: false,
      isError: false,
    });
  });

  const renderComponent = (props = {}) => {
    return render(
      <QueryClientProvider client={queryClient}>
        <AddDataToCollectionModal
          collection={mockCollection}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
          {...props}
        />
      </QueryClientProvider>
    );
  };

  describe('Modal Rendering', () => {
    it('should render the modal with collection data', () => {
      renderComponent();

      // Check modal title
      expect(screen.getByText('Add Data to Collection')).toBeInTheDocument();
      
      // Check collection name in subtitle
      expect(screen.getByText(/Add new documents to "Test Collection"/)).toBeInTheDocument();

      // Check form elements
      expect(screen.getByLabelText('Source Directory Path')).toBeInTheDocument();
      expect(screen.getByPlaceholderText('/path/to/documents')).toBeInTheDocument();

      // Check collection settings display
      expect(screen.getByText('Collection Settings')).toBeInTheDocument();
      expect(screen.getByText('text-embedding-ada-002')).toBeInTheDocument();
      expect(screen.getByText('1000 characters')).toBeInTheDocument();
      expect(screen.getByText('200 characters')).toBeInTheDocument();
      expect(screen.getByText('ready')).toBeInTheDocument();
      expect(screen.getByText('42')).toBeInTheDocument();

      // Check buttons
      expect(screen.getByRole('button', { name: 'Cancel' })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Add Data' })).toBeInTheDocument();
    });

    it('should display the info message about duplicate files', () => {
      renderComponent();
      
      expect(screen.getByText(/Duplicate files will be automatically skipped/)).toBeInTheDocument();
    });

    it('should have correct accessibility attributes', () => {
      renderComponent();

      const input = screen.getByLabelText('Source Directory Path');
      expect(input).toHaveAttribute('type', 'text');
      expect(input).toHaveAttribute('id', 'sourcePath');
      expect(input).toHaveAttribute('required');
      // Note: autoFocus as a React prop doesn't translate to an HTML attribute
      // Instead, the element should receive focus
      expect(document.activeElement).toBe(input);
    });
  });

  describe('Path Input Validation', () => {
    it('should show error toast when submitting empty path', async () => {
      renderComponent();

      // Submit the form directly to bypass HTML5 validation
      const form = screen.getByLabelText('Source Directory Path').closest('form')!;
      fireEvent.submit(form);

      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith({
          type: 'error',
          message: 'Please enter a directory path',
        });
      });
      expect(mockMutateAsync).not.toHaveBeenCalled();
    });

    it('should show error toast when submitting whitespace-only path', async () => {
      renderComponent();

      const input = screen.getByLabelText('Source Directory Path');
      await userEvent.type(input, '   ');

      const submitButton = screen.getByRole('button', { name: 'Add Data' });
      await userEvent.click(submitButton);

      expect(mockAddToast).toHaveBeenCalledWith({
        type: 'error',
        message: 'Please enter a directory path',
      });
      expect(mockMutateAsync).not.toHaveBeenCalled();
    });

    it('should trim whitespace from valid paths', async () => {
      renderComponent();
      mockMutateAsync.mockResolvedValue({ data: { id: 'operation-id' } });

      const input = screen.getByLabelText('Source Directory Path');
      await userEvent.type(input, '  /path/to/documents  ');

      const submitButton = screen.getByRole('button', { name: 'Add Data' });
      await userEvent.click(submitButton);

      expect(mockMutateAsync).toHaveBeenCalledWith({
        collectionId: 'test-collection-id',
        sourcePath: '/path/to/documents',
        config: {
          chunk_size: 1000,
          chunk_overlap: 200,
        },
      });
    });
  });

  describe('Source Addition Flow', () => {
    it('should successfully add a source and navigate to collection detail', async () => {
      renderComponent();
      mockMutateAsync.mockResolvedValue({ data: { id: 'operation-id' } });

      const input = screen.getByLabelText('Source Directory Path');
      await userEvent.type(input, '/valid/path');

      const submitButton = screen.getByRole('button', { name: 'Add Data' });
      await userEvent.click(submitButton);

      await waitFor(() => {
        expect(mockMutateAsync).toHaveBeenCalledWith({
          collectionId: 'test-collection-id',
          sourcePath: '/valid/path',
          config: {
            chunk_size: 1000,
            chunk_overlap: 200,
          },
        });
      });

      expect(mockNavigate).toHaveBeenCalledWith('/collections/test-collection-id');
      expect(mockOnSuccess).toHaveBeenCalled();
    });

    it('should handle API errors from mutation', async () => {
      renderComponent();
      const error = new Error('Failed to add source');
      mockMutateAsync.mockRejectedValue(error);

      const input = screen.getByLabelText('Source Directory Path');
      await userEvent.type(input, '/invalid/path');

      const submitButton = screen.getByRole('button', { name: 'Add Data' });
      await userEvent.click(submitButton);

      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith({
          message: 'Failed to add source',
          type: 'error',
        });
      });

      expect(mockNavigate).not.toHaveBeenCalled();
      expect(mockOnSuccess).not.toHaveBeenCalled();
    });

    it('should handle non-Error exceptions', async () => {
      renderComponent();
      mockMutateAsync.mockRejectedValue('Unknown error');

      const input = screen.getByLabelText('Source Directory Path');
      await userEvent.type(input, '/path');

      const submitButton = screen.getByRole('button', { name: 'Add Data' });
      await userEvent.click(submitButton);

      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith({
          message: 'Failed to add data source',
          type: 'error',
        });
      });
    });

    it('should not show duplicate error toasts when mutation handles error', async () => {
      // Mock mutation to indicate it already handled the error
      (useAddSource as Mock).mockReturnValue({
        mutateAsync: mockMutateAsync,
        isPending: false,
        isError: true, // Mutation already handled error
      });

      renderComponent();
      mockMutateAsync.mockRejectedValue(new Error('API Error'));

      const input = screen.getByLabelText('Source Directory Path');
      await userEvent.type(input, '/path');

      const submitButton = screen.getByRole('button', { name: 'Add Data' });
      await userEvent.click(submitButton);

      await waitFor(() => {
        expect(mockMutateAsync).toHaveBeenCalled();
      });

      // Should not call addToast since mutation already handled the error
      expect(mockAddToast).not.toHaveBeenCalled();
    });
  });

  describe('Loading States', () => {
    it('should disable form elements while submitting', async () => {
      renderComponent();
      mockMutateAsync.mockImplementation(() => new Promise(() => {})); // Never resolves

      const input = screen.getByLabelText('Source Directory Path');
      await userEvent.type(input, '/path');

      const submitButton = screen.getByRole('button', { name: 'Add Data' });
      const cancelButton = screen.getByRole('button', { name: 'Cancel' });

      await userEvent.click(submitButton);

      await waitFor(() => {
        expect(submitButton).toBeDisabled();
        expect(submitButton).toHaveTextContent('Adding Source...');
        expect(cancelButton).toBeDisabled();
      });
    });

    it('should show loading state when mutation is pending', () => {
      (useAddSource as Mock).mockReturnValue({
        mutateAsync: mockMutateAsync,
        isPending: true,
        isError: false,
      });

      renderComponent();

      const submitButton = screen.getByRole('button', { name: 'Adding Source...' });
      const cancelButton = screen.getByRole('button', { name: 'Cancel' });

      expect(submitButton).toBeDisabled();
      expect(cancelButton).toBeDisabled();
    });

    it('should re-enable form after successful submission', async () => {
      renderComponent();
      mockMutateAsync.mockResolvedValue({ data: { id: 'operation-id' } });

      const input = screen.getByLabelText('Source Directory Path');
      await userEvent.type(input, '/path');

      const submitButton = screen.getByRole('button', { name: 'Add Data' });
      await userEvent.click(submitButton);

      await waitFor(() => {
        expect(mockOnSuccess).toHaveBeenCalled();
      });

      // Form should be re-enabled after success (though modal would typically close)
      expect(submitButton).not.toBeDisabled();
      expect(submitButton).toHaveTextContent('Add Data');
    });

    it('should re-enable form after error', async () => {
      renderComponent();
      mockMutateAsync.mockRejectedValue(new Error('Failed'));

      const input = screen.getByLabelText('Source Directory Path');
      await userEvent.type(input, '/path');

      const submitButton = screen.getByRole('button', { name: 'Add Data' });
      await userEvent.click(submitButton);

      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalled();
      });

      expect(submitButton).not.toBeDisabled();
      expect(submitButton).toHaveTextContent('Add Data');
    });
  });

  describe('Modal Close Functionality', () => {
    it('should call onClose when clicking Cancel button', async () => {
      renderComponent();

      const cancelButton = screen.getByRole('button', { name: 'Cancel' });
      await userEvent.click(cancelButton);

      expect(mockOnClose).toHaveBeenCalledTimes(1);
    });

    it('should call onClose when clicking backdrop', async () => {
      renderComponent();

      // Find the backdrop (first div with fixed positioning and bg-opacity)
      const backdrop = document.querySelector('.fixed.inset-0.bg-black.bg-opacity-50');
      expect(backdrop).toBeInTheDocument();

      fireEvent.click(backdrop!);

      expect(mockOnClose).toHaveBeenCalledTimes(1);
    });

    it('should not close modal when clicking inside modal content', async () => {
      renderComponent();

      const modalContent = screen.getByText('Add Data to Collection').closest('div');
      fireEvent.click(modalContent!);

      expect(mockOnClose).not.toHaveBeenCalled();
    });
  });

  describe('Form Submission', () => {
    it('should prevent default form submission', async () => {
      renderComponent();

      const form = screen.getByLabelText('Source Directory Path').closest('form');
      expect(form).toBeInTheDocument();

      const submitEvent = new Event('submit', { bubbles: true, cancelable: true });
      const preventDefaultSpy = vi.spyOn(submitEvent, 'preventDefault');

      fireEvent(form!, submitEvent);

      expect(preventDefaultSpy).toHaveBeenCalled();
    });

    it('should submit form on Enter key in input field', async () => {
      renderComponent();
      mockMutateAsync.mockResolvedValue({ data: { id: 'operation-id' } });

      const input = screen.getByLabelText('Source Directory Path');
      await userEvent.type(input, '/path/to/docs{Enter}');

      await waitFor(() => {
        expect(mockMutateAsync).toHaveBeenCalled();
      });
    });
  });

  describe('Collection Status Variations', () => {
    it('should display different collection statuses correctly', () => {
      const processingCollection = {
        ...mockCollection,
        status: 'processing' as const,
      };

      renderComponent({ collection: processingCollection });

      expect(screen.getByText('processing')).toBeInTheDocument();
    });

    it('should handle collections with missing optional fields', () => {
      const minimalCollection = {
        ...mockCollection,
        description: undefined,
        status_message: undefined,
        metadata: undefined,
        total_size_bytes: undefined,
      };

      renderComponent({ collection: minimalCollection });

      // Should still render without errors
      expect(screen.getByText('Add Data to Collection')).toBeInTheDocument();
    });
  });

  describe('Input Field Behavior', () => {
    it('should focus the input field on mount', () => {
      renderComponent();
      
      const input = screen.getByLabelText('Source Directory Path');
      expect(document.activeElement).toBe(input);
    });

    it('should update input value as user types', async () => {
      renderComponent();
      
      const input = screen.getByLabelText('Source Directory Path') as HTMLInputElement;
      const testPath = '/home/user/documents';
      
      await userEvent.type(input, testPath);
      
      expect(input.value).toBe(testPath);
    });

  });

  describe('Configuration Display', () => {
    it('should display custom chunk configuration', () => {
      const customCollection = {
        ...mockCollection,
        chunk_size: 2000,
        chunk_overlap: 500,
      };
      
      renderComponent({ collection: customCollection });
      
      expect(screen.getByText('2000 characters')).toBeInTheDocument();
      expect(screen.getByText('500 characters')).toBeInTheDocument();
    });

    it('should display correct embedding model configuration', () => {
      const customCollection = {
        ...mockCollection,
        embedding_model: 'text-embedding-3-small',
      };
      
      renderComponent({ collection: customCollection });
      
      expect(screen.getByText('text-embedding-3-small')).toBeInTheDocument();
    });
  });

  describe('Error Recovery', () => {
    it('should allow retry after failed submission', async () => {
      renderComponent();
      
      // First attempt fails
      mockMutateAsync.mockRejectedValueOnce(new Error('Network error'));
      
      const input = screen.getByLabelText('Source Directory Path');
      await userEvent.type(input, '/path/to/docs');
      
      const submitButton = screen.getByRole('button', { name: 'Add Data' });
      await userEvent.click(submitButton);
      
      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith({
          message: 'Network error',
          type: 'error',
        });
      });
      
      // Second attempt succeeds
      mockMutateAsync.mockResolvedValueOnce({ data: { id: 'operation-id' } });
      
      await userEvent.click(submitButton);
      
      await waitFor(() => {
        expect(mockNavigate).toHaveBeenCalledWith('/collections/test-collection-id');
        expect(mockOnSuccess).toHaveBeenCalled();
      });
    });
  });

  describe('Edge Cases', () => {
    it('should handle very long collection names gracefully', () => {
      const longNameCollection = {
        ...mockCollection,
        name: 'A'.repeat(100),
      };

      renderComponent({ collection: longNameCollection });

      // Check that the long name is displayed (might be truncated with CSS)
      expect(screen.getByText(new RegExp(`Add new documents to "${longNameCollection.name}"`)))
        .toBeInTheDocument();
    });

    it('should handle special characters in paths', async () => {
      renderComponent();
      mockMutateAsync.mockResolvedValue({ data: { id: 'operation-id' } });

      const input = screen.getByLabelText('Source Directory Path');
      const specialPath = '/path/with spaces/and-special@chars#test';
      await userEvent.type(input, specialPath);

      const submitButton = screen.getByRole('button', { name: 'Add Data' });
      await userEvent.click(submitButton);

      expect(mockMutateAsync).toHaveBeenCalledWith({
        collectionId: 'test-collection-id',
        sourcePath: specialPath,
        config: {
          chunk_size: 1000,
          chunk_overlap: 200,
        },
      });
    });

    it('should handle rapid form submissions', async () => {
      renderComponent();
      
      // Mock a slow mutation to test protection against multiple submissions
      let resolvePromise: () => void;
      const slowPromise = new Promise<{ data: { id: string } }>((resolve) => {
        resolvePromise = () => resolve({ data: { id: 'operation-id' } });
      });
      mockMutateAsync.mockReturnValue(slowPromise);

      const input = screen.getByLabelText('Source Directory Path');
      await userEvent.type(input, '/path');

      const submitButton = screen.getByRole('button', { name: 'Add Data' });
      
      // First click starts the submission
      await userEvent.click(submitButton);
      
      // Button should now be disabled and show loading state
      expect(submitButton).toBeDisabled();
      expect(submitButton).toHaveTextContent('Adding Source...');
      
      // Try clicking multiple times while it's disabled
      await userEvent.click(submitButton);
      await userEvent.click(submitButton);

      // Should only call mutate once due to isSubmitting state
      expect(mockMutateAsync).toHaveBeenCalledTimes(1);
      
      // Resolve the promise to complete the test
      resolvePromise!();
      
      await waitFor(() => {
        expect(mockOnSuccess).toHaveBeenCalled();
      });
    });
  });
});