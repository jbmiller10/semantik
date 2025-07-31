import { render, screen, waitFor } from '@testing-library/react';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import DeleteCollectionModal from '../DeleteCollectionModal';
import { useDeleteCollection } from '../../hooks/useCollections';

// Mock the hooks
vi.mock('../../hooks/useCollections', () => ({
  useDeleteCollection: vi.fn()
}));

const mockUseDeleteCollection = useDeleteCollection as vi.MockedFunction<typeof useDeleteCollection>;

// Test data
const mockStats = {
  total_files: 150,
  total_vectors: 3000,
  total_size: 1048576, // 1 MB
  job_count: 5,
};

const defaultProps = {
  collectionId: 'test-collection-id',
  collectionName: 'Test Collection',
  stats: mockStats,
  onClose: vi.fn(),
  onSuccess: vi.fn(),
};

// Helper function to render component with providers
const renderComponent = (props = {}) => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });

  return render(
    <QueryClientProvider client={queryClient}>
      <DeleteCollectionModal {...defaultProps} {...props} />
    </QueryClientProvider>
  );
};

describe('DeleteCollectionModal', () => {
  let mockMutate: vi.Mock;
  let mockMutation: any;

  beforeEach(() => {
    vi.clearAllMocks();
    
    // Setup default mock mutation
    mockMutate = vi.fn();
    mockMutation = {
      mutate: mockMutate,
      isPending: false,
      isError: false,
      error: null,
      reset: vi.fn(),
    };
    
    mockUseDeleteCollection.mockReturnValue(mockMutation);
  });

  describe('Modal Rendering', () => {
    it('should render the modal with collection details', () => {
      renderComponent();
      
      expect(screen.getByRole('heading', { name: 'Delete Collection' })).toBeInTheDocument();
      expect(screen.getByText(/You are about to permanently delete the collection "Test Collection"/)).toBeInTheDocument();
      expect(screen.getByText('This action cannot be undone')).toBeInTheDocument();
    });

    it('should render the confirmation input field', () => {
      renderComponent();
      
      const input = screen.getByLabelText(/Type DELETE to confirm/);
      expect(input).toBeInTheDocument();
      expect(input).toHaveAttribute('placeholder', 'Type DELETE here');
    });

    it('should render cancel and delete buttons', () => {
      renderComponent();
      
      expect(screen.getByRole('button', { name: 'Cancel' })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Delete Collection' })).toBeInTheDocument();
    });
  });

  describe('Details Toggle', () => {
    it('should show deletion details when toggle is clicked', async () => {
      const user = userEvent.setup();
      renderComponent();
      
      const toggleButton = screen.getByText('What will be deleted?');
      
      // Initially hidden
      expect(screen.queryByText('Jobs:')).not.toBeInTheDocument();
      
      // Click to show
      await user.click(toggleButton);
      
      expect(screen.getByText('Jobs:')).toBeInTheDocument();
      expect(screen.getByText('5')).toBeInTheDocument();
      expect(screen.getByText('Documents:')).toBeInTheDocument();
      expect(screen.getByText('150')).toBeInTheDocument();
      expect(screen.getByText('Vectors:')).toBeInTheDocument();
      expect(screen.getByText('3,000')).toBeInTheDocument();
      expect(screen.getByText('Storage:')).toBeInTheDocument();
      expect(screen.getByText('1 MB')).toBeInTheDocument();
      expect(screen.getByText(/All database records, vector embeddings, and associated files will be permanently removed/)).toBeInTheDocument();
    });

    it('should toggle details visibility', async () => {
      const user = userEvent.setup();
      renderComponent();
      
      const toggleButton = screen.getByText('What will be deleted?');
      
      // Open details
      await user.click(toggleButton);
      expect(screen.getByText('Jobs:')).toBeInTheDocument();
      
      // Close details
      await user.click(toggleButton);
      await waitFor(() => {
        expect(screen.queryByText('Jobs:')).not.toBeInTheDocument();
      });
    });
  });

  describe('Confirmation Flow', () => {
    it('should disable delete button when confirmation text is empty', () => {
      renderComponent();
      
      const deleteButton = screen.getByRole('button', { name: 'Delete Collection' });
      expect(deleteButton).toBeDisabled();
    });

    it('should disable delete button when confirmation text is incorrect', async () => {
      const user = userEvent.setup();
      renderComponent();
      
      const input = screen.getByLabelText(/Type DELETE to confirm/);
      await user.type(input, 'delete'); // lowercase
      
      const deleteButton = screen.getByRole('button', { name: 'Delete Collection' });
      expect(deleteButton).toBeDisabled();
    });

    it('should enable delete button when confirmation text is "DELETE"', async () => {
      const user = userEvent.setup();
      renderComponent();
      
      const input = screen.getByLabelText(/Type DELETE to confirm/);
      await user.type(input, 'DELETE');
      
      const deleteButton = screen.getByRole('button', { name: 'Delete Collection' });
      expect(deleteButton).toBeEnabled();
    });

    it('should handle partial typing of confirmation text', async () => {
      const user = userEvent.setup();
      renderComponent();
      
      const input = screen.getByLabelText(/Type DELETE to confirm/);
      const deleteButton = screen.getByRole('button', { name: 'Delete Collection' });
      
      await user.type(input, 'D');
      expect(deleteButton).toBeDisabled();
      
      await user.type(input, 'E');
      expect(deleteButton).toBeDisabled();
      
      await user.type(input, 'LETE');
      expect(deleteButton).toBeEnabled();
    });
  });

  describe('Deletion Flow', () => {
    it('should call delete mutation when form is submitted with correct confirmation', async () => {
      const user = userEvent.setup();
      renderComponent();
      
      const input = screen.getByLabelText(/Type DELETE to confirm/);
      await user.type(input, 'DELETE');
      
      const deleteButton = screen.getByRole('button', { name: 'Delete Collection' });
      await user.click(deleteButton);
      
      expect(mockMutate).toHaveBeenCalledWith(
        'test-collection-id',
        expect.objectContaining({
          onSuccess: expect.any(Function)
        })
      );
    });

    it('should handle successful deletion', async () => {
      const user = userEvent.setup();
      renderComponent();
      
      const input = screen.getByLabelText(/Type DELETE to confirm/);
      await user.type(input, 'DELETE');
      
      const deleteButton = screen.getByRole('button', { name: 'Delete Collection' });
      await user.click(deleteButton);
      
      // Get the onSuccess callback and call it
      const mutateCall = mockMutate.mock.calls[0];
      const options = mutateCall[1];
      
      // Simulate successful deletion
      options.onSuccess();
      
      expect(defaultProps.onSuccess).toHaveBeenCalled();
      expect(defaultProps.onClose).toHaveBeenCalled();
    });

    it('should show loading state during deletion', async () => {
      const user = userEvent.setup();
      
      // Set mutation to pending state
      mockMutation.isPending = true;
      mockUseDeleteCollection.mockReturnValue(mockMutation);
      
      renderComponent();
      
      const input = screen.getByLabelText(/Type DELETE to confirm/);
      await user.type(input, 'DELETE');
      
      const deleteButton = screen.getByRole('button', { name: 'Deleting...' });
      expect(deleteButton).toBeDisabled();
      
      const cancelButton = screen.getByRole('button', { name: 'Cancel' });
      expect(cancelButton).toBeDisabled();
    });

    it('should not submit when Enter is pressed without correct confirmation', async () => {
      const user = userEvent.setup();
      renderComponent();
      
      const input = screen.getByLabelText(/Type DELETE to confirm/);
      await user.type(input, 'wrong');
      await user.keyboard('{Enter}');
      
      expect(mockMutate).not.toHaveBeenCalled();
    });

    it('should submit when Enter is pressed with correct confirmation', async () => {
      const user = userEvent.setup();
      renderComponent();
      
      const input = screen.getByLabelText(/Type DELETE to confirm/);
      await user.type(input, 'DELETE');
      await user.keyboard('{Enter}');
      
      expect(mockMutate).toHaveBeenCalledWith(
        'test-collection-id',
        expect.objectContaining({
          onSuccess: expect.any(Function)
        })
      );
    });
  });

  describe('User Interactions', () => {
    it('should call onClose when cancel button is clicked', async () => {
      const user = userEvent.setup();
      renderComponent();
      
      const cancelButton = screen.getByRole('button', { name: 'Cancel' });
      await user.click(cancelButton);
      
      expect(defaultProps.onClose).toHaveBeenCalledTimes(1);
    });

    it('should call onClose when backdrop is clicked', async () => {
      const user = userEvent.setup();
      const { container } = renderComponent();
      
      // Find the backdrop (first div with fixed inset-0)
      const backdrop = container.querySelector('.fixed.inset-0.bg-black');
      await user.click(backdrop!);
      
      expect(defaultProps.onClose).toHaveBeenCalledTimes(1);
    });

    it('should focus on confirmation input when modal opens', async () => {
      renderComponent();
      
      const input = screen.getByLabelText(/Type DELETE to confirm/);
      
      // Wait a bit for focus to be set
      await waitFor(() => {
        expect(document.activeElement).toBe(input);
      }, { timeout: 100 });
    });
  });

  describe('Edge Cases', () => {
    it('should handle empty collection name gracefully', () => {
      renderComponent({ collectionName: '' });
      
      expect(screen.getByText(/You are about to permanently delete the collection ""/)).toBeInTheDocument();
    });

    it('should handle zero stats gracefully', async () => {
      const zeroStats = {
        total_files: 0,
        total_vectors: 0,
        total_size: 0,
        job_count: 0,
      };
      
      renderComponent({ stats: zeroStats });
      
      // Open details
      const user = userEvent.setup();
      await user.click(screen.getByText('What will be deleted?'));
      
      expect(screen.getByText('0 Bytes')).toBeInTheDocument();
    });

    it('should format large numbers correctly', async () => {
      const largeStats = {
        total_files: 1234567,
        total_vectors: 9876543,
        total_size: 1073741824, // 1 GB
        job_count: 999,
      };
      
      const user = userEvent.setup();
      renderComponent({ stats: largeStats });
      
      await user.click(screen.getByText('What will be deleted?'));
      
      expect(screen.getByText('1,234,567')).toBeInTheDocument();
      expect(screen.getByText('9,876,543')).toBeInTheDocument();
      expect(screen.getByText('1 GB')).toBeInTheDocument();
    });
  });
});