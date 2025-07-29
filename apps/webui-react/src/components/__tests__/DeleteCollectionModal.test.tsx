import { render, screen, waitFor } from '@testing-library/react';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { AxiosError } from 'axios';
import DeleteCollectionModal from '../DeleteCollectionModal';
import { collectionsV2Api } from '../../services/api/v2/collections';
import { useUIStore } from '../../stores/uiStore';

// Mock dependencies
vi.mock('../../services/api/v2/collections');
vi.mock('../../stores/uiStore');

const mockCollectionsV2Api = collectionsV2Api as vi.Mocked<typeof collectionsV2Api>;
const mockUseUIStore = useUIStore as unknown as vi.Mock;

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
  let mockAddToast: vi.Mock;

  beforeEach(() => {
    vi.clearAllMocks();
    mockAddToast = vi.fn();
    mockUseUIStore.mockReturnValue({ addToast: mockAddToast });
  });

  describe('Modal Rendering', () => {
    it('should render modal with collection name', () => {
      renderComponent();
      
      expect(screen.getByRole('heading', { name: 'Delete Collection' })).toBeInTheDocument();
      expect(screen.getByText(/You are about to permanently delete the collection "Test Collection"/)).toBeInTheDocument();
    });

    it('should render warning message', () => {
      renderComponent();
      
      expect(screen.getByText('This action cannot be undone')).toBeInTheDocument();
    });

    it('should render confirmation input field', () => {
      renderComponent();
      
      const input = screen.getByLabelText(/Type DELETE to confirm/);
      expect(input).toBeInTheDocument();
      expect(input).toHaveAttribute('placeholder', 'Type DELETE here');
      expect(input).toHaveAttribute('autoComplete', 'off');
    });

    it('should render action buttons', () => {
      renderComponent();
      
      expect(screen.getByRole('button', { name: 'Cancel' })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Delete Collection' })).toBeInTheDocument();
    });
  });

  describe('Details Toggle', () => {
    it('should hide details by default', () => {
      renderComponent();
      
      expect(screen.queryByText('Jobs:')).not.toBeInTheDocument();
      expect(screen.queryByText('Documents:')).not.toBeInTheDocument();
    });

    it('should show details when toggle button is clicked', async () => {
      const user = userEvent.setup();
      renderComponent();
      
      const toggleButton = screen.getByText('What will be deleted?');
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
    it('should call delete API when form is submitted with correct confirmation', async () => {
      const user = userEvent.setup();
      mockCollectionsV2Api.delete.mockResolvedValue({ data: {} });
      renderComponent();
      
      const input = screen.getByLabelText(/Type DELETE to confirm/);
      await user.type(input, 'DELETE');
      
      const deleteButton = screen.getByRole('button', { name: 'Delete Collection' });
      await user.click(deleteButton);
      
      expect(mockCollectionsV2Api.delete).toHaveBeenCalledWith('test-collection-id');
    });

    it('should handle successful deletion', async () => {
      const user = userEvent.setup();
      mockCollectionsV2Api.delete.mockResolvedValue({ data: {} });
      renderComponent();
      
      const input = screen.getByLabelText(/Type DELETE to confirm/);
      await user.type(input, 'DELETE');
      
      const deleteButton = screen.getByRole('button', { name: 'Delete Collection' });
      await user.click(deleteButton);
      
      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith({
          type: 'success',
          message: 'Collection "Test Collection" deleted successfully',
        });
        expect(defaultProps.onSuccess).toHaveBeenCalled();
      });
    });

    it('should handle deletion error with specific message', async () => {
      const user = userEvent.setup();
      const error = new AxiosError('Network error');
      error.response = {
        data: { detail: 'Collection is being indexed' },
        status: 409,
        statusText: 'Conflict',
        headers: {},
        config: {} as any,
      };
      mockCollectionsV2Api.delete.mockRejectedValue(error);
      renderComponent();
      
      const input = screen.getByLabelText(/Type DELETE to confirm/);
      await user.type(input, 'DELETE');
      
      const deleteButton = screen.getByRole('button', { name: 'Delete Collection' });
      await user.click(deleteButton);
      
      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith({
          type: 'error',
          message: 'Collection is being indexed',
        });
      });
    });

    it('should handle deletion error with fallback message', async () => {
      const user = userEvent.setup();
      const error = new Error('Network error');
      mockCollectionsV2Api.delete.mockRejectedValue(error);
      renderComponent();
      
      const input = screen.getByLabelText(/Type DELETE to confirm/);
      await user.type(input, 'DELETE');
      
      const deleteButton = screen.getByRole('button', { name: 'Delete Collection' });
      await user.click(deleteButton);
      
      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith({
          type: 'error',
          message: 'Failed to delete collection',
        });
      });
    });
  });

  describe('Loading State', () => {
    it('should show loading state during deletion', async () => {
      const user = userEvent.setup();
      let resolveDelete: () => void;
      const deletePromise = new Promise<void>((resolve) => {
        resolveDelete = resolve;
      });
      mockCollectionsV2Api.delete.mockReturnValue(deletePromise as any);
      renderComponent();
      
      const input = screen.getByLabelText(/Type DELETE to confirm/);
      await user.type(input, 'DELETE');
      
      const deleteButton = screen.getByRole('button', { name: 'Delete Collection' });
      await user.click(deleteButton);
      
      // Check loading state
      expect(screen.getByRole('button', { name: 'Deleting...' })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Deleting...' })).toBeDisabled();
      expect(screen.getByRole('button', { name: 'Cancel' })).toBeDisabled();
      
      // Resolve deletion
      resolveDelete!();
      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalled();
      });
    });
  });

  describe('Modal Close Functionality', () => {
    it('should call onClose when Cancel button is clicked', async () => {
      const user = userEvent.setup();
      renderComponent();
      
      const cancelButton = screen.getByRole('button', { name: 'Cancel' });
      await user.click(cancelButton);
      
      expect(defaultProps.onClose).toHaveBeenCalled();
    });

    it('should call onClose when backdrop is clicked', async () => {
      const user = userEvent.setup();
      renderComponent();
      
      // Find the backdrop (first div with fixed positioning)
      const backdrop = document.querySelector('.fixed.inset-0');
      expect(backdrop).toBeInTheDocument();
      
      await user.click(backdrop!);
      
      expect(defaultProps.onClose).toHaveBeenCalled();
    });

    it('should not close modal when clicking inside modal content', async () => {
      const user = userEvent.setup();
      renderComponent();
      
      const modalContent = screen.getByRole('heading', { name: 'Delete Collection' });
      await user.click(modalContent);
      
      expect(defaultProps.onClose).not.toHaveBeenCalled();
    });
  });

  describe('Form Submission', () => {
    it('should prevent form submission when confirmation text is incorrect', async () => {
      const user = userEvent.setup();
      renderComponent();
      
      const input = screen.getByLabelText(/Type DELETE to confirm/);
      await user.type(input, 'WRONG');
      
      const form = screen.getByRole('button', { name: 'Delete Collection' }).closest('form');
      const submitEvent = new Event('submit', { bubbles: true, cancelable: true });
      form!.dispatchEvent(submitEvent);
      
      expect(mockCollectionsV2Api.delete).not.toHaveBeenCalled();
    });

    it('should handle form submission with Enter key', async () => {
      const user = userEvent.setup();
      mockCollectionsV2Api.delete.mockResolvedValue({ data: {} });
      renderComponent();
      
      const input = screen.getByLabelText(/Type DELETE to confirm/);
      await user.type(input, 'DELETE');
      await user.keyboard('{Enter}');
      
      await waitFor(() => {
        expect(mockCollectionsV2Api.delete).toHaveBeenCalledWith('test-collection-id');
      });
    });
  });

  describe('Accessibility', () => {
    it('should have proper modal structure', () => {
      renderComponent();
      
      // Check for modal overlay and content
      const overlay = document.querySelector('.fixed.inset-0');
      const modalContent = document.querySelector('.fixed.left-1\\/2');
      
      expect(overlay).toBeInTheDocument();
      expect(modalContent).toBeInTheDocument();
    });

    it('should have proper form structure', () => {
      renderComponent();
      
      const form = screen.getByRole('button', { name: 'Delete Collection' }).closest('form');
      expect(form).toBeInTheDocument();
      
      const input = screen.getByLabelText(/Type DELETE to confirm/);
      expect(input).toHaveAttribute('required');
    });

    it('should maintain focus management', async () => {
      const user = userEvent.setup();
      renderComponent();
      
      const input = screen.getByLabelText(/Type DELETE to confirm/);
      await user.click(input);
      
      expect(document.activeElement).toBe(input);
    });
  });

  describe('Data Formatting', () => {
    it('should format bytes correctly', async () => {
      const user = userEvent.setup();
      const customStats = {
        ...mockStats,
        total_size: 0,
      };
      renderComponent({ stats: customStats });
      
      const toggleButton = screen.getByText('What will be deleted?');
      await user.click(toggleButton);
      
      expect(screen.getByText('0 Bytes')).toBeInTheDocument();
    });

    it('should format large byte values correctly', async () => {
      const user = userEvent.setup();
      const customStats = {
        ...mockStats,
        total_size: 1073741824, // 1 GB
      };
      renderComponent({ stats: customStats });
      
      const toggleButton = screen.getByText('What will be deleted?');
      await user.click(toggleButton);
      
      expect(screen.getByText('1 GB')).toBeInTheDocument();
    });

    it('should format numbers with locale separators', async () => {
      const user = userEvent.setup();
      const customStats = {
        ...mockStats,
        total_files: 1500,
        total_vectors: 30000,
      };
      renderComponent({ stats: customStats });
      
      const toggleButton = screen.getByText('What will be deleted?');
      await user.click(toggleButton);
      
      expect(screen.getByText('1,500')).toBeInTheDocument();
      expect(screen.getByText('30,000')).toBeInTheDocument();
    });
  });
});