import { render, screen, waitFor } from '@testing-library/react';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import DeleteConfirmModal from '../DeleteConfirmModal';
import { useDeleteMCPProfile } from '../../../hooks/useMCPProfiles';
import type { MCPProfile } from '../../../types/mcp-profile';

// Mock the hooks
vi.mock('../../../hooks/useMCPProfiles', () => ({
  useDeleteMCPProfile: vi.fn(),
}));

const mockUseDeleteMCPProfile = useDeleteMCPProfile as vi.MockedFunction<typeof useDeleteMCPProfile>;

// Test data
const mockProfile: MCPProfile = {
  id: 'profile-1',
  name: 'test-profile',
  description: 'A test profile',
  enabled: true,
  search_type: 'semantic',
  result_count: 10,
  use_reranker: true,
  score_threshold: null,
  hybrid_alpha: null,
  collections: [],
  created_at: '2025-01-01T00:00:00Z',
  updated_at: '2025-01-01T00:00:00Z',
};

const defaultProps = {
  profile: mockProfile,
  onClose: vi.fn(),
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
      <DeleteConfirmModal {...defaultProps} {...props} />
    </QueryClientProvider>
  );
};

describe('DeleteConfirmModal', () => {
  let mockMutateAsync: vi.Mock;
  let mockMutation: {
    mutateAsync: vi.Mock;
    isPending: boolean;
  };

  beforeEach(() => {
    vi.clearAllMocks();

    mockMutateAsync = vi.fn().mockResolvedValue({});
    mockMutation = {
      mutateAsync: mockMutateAsync,
      isPending: false,
    };

    mockUseDeleteMCPProfile.mockReturnValue(mockMutation as ReturnType<typeof useDeleteMCPProfile>);
  });

  describe('Rendering', () => {
    it('should render the modal with correct title', () => {
      renderComponent();

      expect(screen.getByRole('heading', { name: 'Delete Profile' })).toBeInTheDocument();
    });

    it('should display profile name in confirmation message', () => {
      renderComponent();

      expect(screen.getByText(/Are you sure you want to delete the profile/)).toBeInTheDocument();
      expect(screen.getByText(/"test-profile"/)).toBeInTheDocument();
    });

    it('should display warning about irreversible action', () => {
      renderComponent();

      expect(screen.getByText(/This action cannot be undone/)).toBeInTheDocument();
    });

    it('should display the MCP tool name that will be removed', () => {
      renderComponent();

      expect(screen.getByText('search_test-profile')).toBeInTheDocument();
    });

    it('should render cancel and delete buttons', () => {
      renderComponent();

      expect(screen.getByRole('button', { name: 'Cancel' })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Delete Profile' })).toBeInTheDocument();
    });

    it('should have correct dialog role and aria attributes', () => {
      renderComponent();

      const dialog = screen.getByRole('dialog');
      expect(dialog).toHaveAttribute('aria-modal', 'true');
      expect(dialog).toHaveAttribute('aria-labelledby', 'delete-modal-title');
    });
  });

  describe('Deletion Flow', () => {
    it('should call delete mutation when delete button is clicked', async () => {
      const user = userEvent.setup();
      renderComponent();

      const deleteButton = screen.getByRole('button', { name: 'Delete Profile' });
      await user.click(deleteButton);

      expect(mockMutateAsync).toHaveBeenCalledWith({
        profileId: 'profile-1',
        profileName: 'test-profile',
      });
    });

    it('should call onClose after successful deletion', async () => {
      const user = userEvent.setup();
      renderComponent();

      const deleteButton = screen.getByRole('button', { name: 'Delete Profile' });
      await user.click(deleteButton);

      await waitFor(() => {
        expect(defaultProps.onClose).toHaveBeenCalled();
      });
    });

    it('should show error message on deletion failure', async () => {
      const user = userEvent.setup();
      mockMutateAsync.mockRejectedValue(new Error('Deletion failed'));

      renderComponent();

      const deleteButton = screen.getByRole('button', { name: 'Delete Profile' });
      await user.click(deleteButton);

      await waitFor(() => {
        expect(screen.getByText('Deletion failed')).toBeInTheDocument();
      });
    });

    it('should not call onClose on deletion failure', async () => {
      const user = userEvent.setup();
      mockMutateAsync.mockRejectedValue(new Error('Deletion failed'));

      renderComponent();

      const deleteButton = screen.getByRole('button', { name: 'Delete Profile' });
      await user.click(deleteButton);

      await waitFor(() => {
        expect(screen.getByText('Deletion failed')).toBeInTheDocument();
      });

      expect(defaultProps.onClose).not.toHaveBeenCalled();
    });

    it('should disable buttons during deletion', async () => {
      const user = userEvent.setup();
      // Create a promise that doesn't resolve immediately
      let resolveDelete: () => void;
      mockMutateAsync.mockImplementation(
        () => new Promise((resolve) => { resolveDelete = () => resolve({}); })
      );

      renderComponent();

      const deleteButton = screen.getByRole('button', { name: 'Delete Profile' });
      await user.click(deleteButton);

      // Buttons should be disabled during deletion
      await waitFor(() => {
        expect(screen.getByRole('button', { name: 'Cancel' })).toBeDisabled();
        expect(screen.getByRole('button', { name: 'Delete Profile' })).toBeDisabled();
      });

      // Clean up
      resolveDelete!();
    });

    it('should show loading spinner during deletion', async () => {
      const user = userEvent.setup();
      let resolveDelete: () => void;
      mockMutateAsync.mockImplementation(
        () => new Promise((resolve) => { resolveDelete = () => resolve({}); })
      );

      renderComponent();

      const deleteButton = screen.getByRole('button', { name: 'Delete Profile' });
      await user.click(deleteButton);

      await waitFor(() => {
        // Check for the spinner SVG with animate-spin class
        const spinner = document.querySelector('.animate-spin');
        expect(spinner).toBeInTheDocument();
      });

      // Clean up
      resolveDelete!();
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

      const backdrop = container.querySelector('.fixed.inset-0.bg-black');
      await user.click(backdrop!);

      expect(defaultProps.onClose).toHaveBeenCalledTimes(1);
    });

    it('should not close on backdrop click during deletion', async () => {
      const user = userEvent.setup();
      let resolveDelete: () => void;
      mockMutateAsync.mockImplementation(
        () => new Promise((resolve) => { resolveDelete = () => resolve({}); })
      );

      const { container } = renderComponent();

      const deleteButton = screen.getByRole('button', { name: 'Delete Profile' });
      await user.click(deleteButton);

      const backdrop = container.querySelector('.fixed.inset-0.bg-black');
      await user.click(backdrop!);

      // onClose should not have been called (except possibly from successful deletion)
      expect(defaultProps.onClose).not.toHaveBeenCalled();

      // Clean up
      resolveDelete!();
    });

    it('should close on Escape key press', async () => {
      const user = userEvent.setup();
      renderComponent();

      await user.keyboard('{Escape}');

      expect(defaultProps.onClose).toHaveBeenCalledTimes(1);
    });

    it('should not close on Escape during deletion', async () => {
      const user = userEvent.setup();
      let resolveDelete: () => void;
      mockMutateAsync.mockImplementation(
        () => new Promise((resolve) => { resolveDelete = () => resolve({}); })
      );

      renderComponent();

      const deleteButton = screen.getByRole('button', { name: 'Delete Profile' });
      await user.click(deleteButton);

      await user.keyboard('{Escape}');

      expect(defaultProps.onClose).not.toHaveBeenCalled();

      // Clean up
      resolveDelete!();
    });
  });

  describe('Edge Cases', () => {
    it('should handle profile with special characters in name', () => {
      renderComponent({
        profile: { ...mockProfile, name: 'my-profile_v2.0' },
      });

      expect(screen.getByText(/"my-profile_v2.0"/)).toBeInTheDocument();
      expect(screen.getByText('search_my-profile_v2.0')).toBeInTheDocument();
    });

    it('should handle generic error message', async () => {
      const user = userEvent.setup();
      mockMutateAsync.mockRejectedValue('Unknown error');

      renderComponent();

      const deleteButton = screen.getByRole('button', { name: 'Delete Profile' });
      await user.click(deleteButton);

      await waitFor(() => {
        expect(screen.getByText('Delete failed')).toBeInTheDocument();
      });
    });
  });
});
