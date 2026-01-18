import { render, screen, waitFor } from '@testing-library/react';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import ProfileFormModal from '../ProfileFormModal';
import { useCollections } from '../../../hooks/useCollections';
import { useCreateMCPProfile, useUpdateMCPProfile } from '../../../hooks/useMCPProfiles';
import type { MCPProfile } from '../../../types/mcp-profile';
import type { Collection } from '../../../types/collection';

// Mock the hooks
vi.mock('../../../hooks/useCollections', () => ({
  useCollections: vi.fn(),
}));

vi.mock('../../../hooks/useMCPProfiles', () => ({
  useCreateMCPProfile: vi.fn(),
  useUpdateMCPProfile: vi.fn(),
}));

const mockUseCollections = useCollections as vi.MockedFunction<typeof useCollections>;
const mockUseCreateMCPProfile = useCreateMCPProfile as vi.MockedFunction<typeof useCreateMCPProfile>;
const mockUseUpdateMCPProfile = useUpdateMCPProfile as vi.MockedFunction<typeof useUpdateMCPProfile>;

// Test data
const mockCollections: Collection[] = [
  {
    id: 'col-1',
    name: 'Documents',
    description: 'Document collection',
    document_count: 100,
    vector_count: 500,
    status: 'ready',
    created_at: '2025-01-01T00:00:00Z',
  },
  {
    id: 'col-2',
    name: 'Code',
    description: 'Code snippets',
    document_count: 50,
    vector_count: 200,
    status: 'ready',
    created_at: '2025-01-01T00:00:00Z',
  },
];

const mockProfile: MCPProfile = {
  id: 'profile-1',
  name: 'test-profile',
  description: 'A test profile for searching',
  enabled: true,
  search_type: 'semantic',
  result_count: 10,
  use_reranker: true,
  score_threshold: null,
  hybrid_alpha: null,
  collections: [{ id: 'col-1', name: 'Documents' }],
  created_at: '2025-01-01T00:00:00Z',
  updated_at: '2025-01-01T00:00:00Z',
};

const defaultProps = {
  profile: null,
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
      <ProfileFormModal {...defaultProps} {...props} />
    </QueryClientProvider>
  );
};

describe('ProfileFormModal', () => {
  let mockCreateMutateAsync: vi.Mock;
  let mockUpdateMutateAsync: vi.Mock;

  beforeEach(() => {
    vi.clearAllMocks();

    // Setup collections mock
    mockUseCollections.mockReturnValue({
      data: mockCollections,
      isLoading: false,
      error: null,
    } as ReturnType<typeof useCollections>);

    // Setup create mock
    mockCreateMutateAsync = vi.fn().mockResolvedValue(mockProfile);
    mockUseCreateMCPProfile.mockReturnValue({
      mutateAsync: mockCreateMutateAsync,
    } as unknown as ReturnType<typeof useCreateMCPProfile>);

    // Setup update mock
    mockUpdateMutateAsync = vi.fn().mockResolvedValue(mockProfile);
    mockUseUpdateMCPProfile.mockReturnValue({
      mutateAsync: mockUpdateMutateAsync,
    } as unknown as ReturnType<typeof useUpdateMCPProfile>);
  });

  describe('Create Mode Rendering', () => {
    it('should render modal with "Create MCP Profile" title', () => {
      renderComponent();

      expect(screen.getByRole('heading', { name: 'Create MCP Profile' })).toBeInTheDocument();
    });

    it('should render empty form fields', () => {
      renderComponent();

      expect(screen.getByLabelText(/Profile Name/)).toHaveValue('');
      expect(screen.getByLabelText(/Description/)).toHaveValue('');
    });

    it('should render collections list', () => {
      renderComponent();

      expect(screen.getByText('Documents')).toBeInTheDocument();
      // 'Code' appears in both the collections list and search type dropdown,
      // so check for the collection checkbox label specifically
      expect(screen.getByRole('checkbox', { name: /code/i })).toBeInTheDocument();
    });

    it('should render search type select with default value', () => {
      renderComponent();

      expect(screen.getByLabelText(/Search Type/)).toHaveValue('semantic');
    });

    it('should render result count input with default value', () => {
      renderComponent();

      expect(screen.getByLabelText(/Default Results/)).toHaveValue(10);
    });

    it('should render toggle switches', () => {
      renderComponent();

      const switches = screen.getAllByRole('switch');
      expect(switches).toHaveLength(2);
      expect(switches[0]).toHaveAttribute('aria-label', 'Disable reranker');
      expect(switches[1]).toHaveAttribute('aria-label', 'Disable profile');
    });

    it('should render Create Profile button', () => {
      renderComponent();

      expect(screen.getByRole('button', { name: 'Create Profile' })).toBeInTheDocument();
    });
  });

  describe('Edit Mode Rendering', () => {
    it('should render modal with "Edit MCP Profile" title', () => {
      renderComponent({ profile: mockProfile });

      expect(screen.getByRole('heading', { name: 'Edit MCP Profile' })).toBeInTheDocument();
    });

    it('should populate form with profile data', () => {
      renderComponent({ profile: mockProfile });

      expect(screen.getByLabelText(/Profile Name/)).toHaveValue('test-profile');
      expect(screen.getByLabelText(/Description/)).toHaveValue('A test profile for searching');
    });

    it('should pre-select collections', () => {
      renderComponent({ profile: mockProfile });

      const documentsCheckbox = screen.getByRole('checkbox', { name: /documents/i });
      expect(documentsCheckbox).toBeChecked();
    });

    it('should render Update Profile button', () => {
      renderComponent({ profile: mockProfile });

      expect(screen.getByRole('button', { name: 'Update Profile' })).toBeInTheDocument();
    });
  });

  describe('Collections Loading State', () => {
    it('should show loading indicator when collections are loading', () => {
      mockUseCollections.mockReturnValue({
        data: undefined,
        isLoading: true,
        error: null,
      } as ReturnType<typeof useCollections>);

      renderComponent();

      expect(screen.getByText('Loading collections...')).toBeInTheDocument();
    });

    it('should show error message when collections fail to load', () => {
      mockUseCollections.mockReturnValue({
        data: undefined,
        isLoading: false,
        error: new Error('Failed to load'),
      } as ReturnType<typeof useCollections>);

      renderComponent();

      expect(screen.getByText('Failed to load collections. Please try again.')).toBeInTheDocument();
    });

    it('should show empty state when no collections exist', () => {
      mockUseCollections.mockReturnValue({
        data: [],
        isLoading: false,
        error: null,
      } as ReturnType<typeof useCollections>);

      renderComponent();

      expect(screen.getByText('No collections available. Create a collection first.')).toBeInTheDocument();
    });
  });

  describe('Form Validation', () => {
    it('should show error for empty name', async () => {
      const user = userEvent.setup();
      renderComponent();

      // Fill description and select collection
      await user.type(screen.getByLabelText(/Description/), 'Test description');
      await user.click(screen.getByRole('checkbox', { name: /documents/i }));

      // Submit
      await user.click(screen.getByRole('button', { name: 'Create Profile' }));

      await waitFor(() => {
        expect(screen.getByText('Profile name is required')).toBeInTheDocument();
      });
    });

    it('should show error for invalid name format', async () => {
      const user = userEvent.setup();
      renderComponent();

      await user.type(screen.getByLabelText(/Profile Name/), 'Invalid Name!');
      await user.type(screen.getByLabelText(/Description/), 'Test description');
      await user.click(screen.getByRole('checkbox', { name: /documents/i }));
      await user.click(screen.getByRole('button', { name: 'Create Profile' }));

      await waitFor(() => {
        expect(screen.getByText(/Name must start with a lowercase letter/)).toBeInTheDocument();
      });
    });

    it('should show error for empty description', async () => {
      const user = userEvent.setup();
      renderComponent();

      await user.type(screen.getByLabelText(/Profile Name/), 'testprofile');
      await user.click(screen.getByRole('checkbox', { name: /documents/i }));
      await user.click(screen.getByRole('button', { name: 'Create Profile' }));

      await waitFor(() => {
        expect(screen.getByText('Description is required')).toBeInTheDocument();
      });
    });

    it('should show error when no collections selected', async () => {
      const user = userEvent.setup();
      renderComponent();

      await user.type(screen.getByLabelText(/Profile Name/), 'testprofile');
      await user.type(screen.getByLabelText(/Description/), 'Test description');
      await user.click(screen.getByRole('button', { name: 'Create Profile' }));

      await waitFor(() => {
        expect(screen.getByText('At least one collection is required')).toBeInTheDocument();
      });
    });

    it('should have aria-invalid on inputs with errors', async () => {
      const user = userEvent.setup();
      renderComponent();

      await user.click(screen.getByRole('button', { name: 'Create Profile' }));

      await waitFor(() => {
        expect(screen.getByLabelText(/Profile Name/)).toHaveAttribute('aria-invalid', 'true');
        expect(screen.getByLabelText(/Description/)).toHaveAttribute('aria-invalid', 'true');
      });
    });

    it('should clear error when field is modified', async () => {
      const user = userEvent.setup();
      renderComponent();

      // Trigger validation
      await user.click(screen.getByRole('button', { name: 'Create Profile' }));
      await waitFor(() => {
        expect(screen.getByText('Profile name is required')).toBeInTheDocument();
      });

      // Type in name field
      await user.type(screen.getByLabelText(/Profile Name/), 'test');

      await waitFor(() => {
        expect(screen.queryByText('Profile name is required')).not.toBeInTheDocument();
      });
    });
  });

  describe('Form Submission', () => {
    it('should call create mutation with form data', async () => {
      const user = userEvent.setup();
      renderComponent();

      await user.type(screen.getByLabelText(/Profile Name/), 'newprofile');
      await user.type(screen.getByLabelText(/Description/), 'A new profile');
      await user.click(screen.getByRole('checkbox', { name: /documents/i }));
      await user.click(screen.getByRole('button', { name: 'Create Profile' }));

      await waitFor(() => {
        expect(mockCreateMutateAsync).toHaveBeenCalledWith(
          expect.objectContaining({
            name: 'newprofile',
            description: 'A new profile',
            collection_ids: ['col-1'],
          })
        );
      });
    });

    it('should call update mutation in edit mode', async () => {
      const user = userEvent.setup();
      renderComponent({ profile: mockProfile });

      await user.clear(screen.getByLabelText(/Description/));
      await user.type(screen.getByLabelText(/Description/), 'Updated description');
      await user.click(screen.getByRole('button', { name: 'Update Profile' }));

      await waitFor(() => {
        expect(mockUpdateMutateAsync).toHaveBeenCalledWith({
          profileId: 'profile-1',
          data: expect.objectContaining({
            description: 'Updated description',
          }),
        });
      });
    });

    it('should call onClose after successful creation', async () => {
      const user = userEvent.setup();
      renderComponent();

      await user.type(screen.getByLabelText(/Profile Name/), 'newprofile');
      await user.type(screen.getByLabelText(/Description/), 'A new profile');
      await user.click(screen.getByRole('checkbox', { name: /documents/i }));
      await user.click(screen.getByRole('button', { name: 'Create Profile' }));

      await waitFor(() => {
        expect(defaultProps.onClose).toHaveBeenCalled();
      });
    });

    it('should show error on submission failure', async () => {
      const user = userEvent.setup();
      mockCreateMutateAsync.mockRejectedValue(new Error('Creation failed'));

      renderComponent();

      await user.type(screen.getByLabelText(/Profile Name/), 'newprofile');
      await user.type(screen.getByLabelText(/Description/), 'A new profile');
      await user.click(screen.getByRole('checkbox', { name: /documents/i }));
      await user.click(screen.getByRole('button', { name: 'Create Profile' }));

      await waitFor(() => {
        expect(screen.getByText('Creation failed')).toBeInTheDocument();
      });
    });

    it('should disable form during submission', async () => {
      const user = userEvent.setup();
      let resolveCreate: () => void;
      mockCreateMutateAsync.mockImplementation(
        () => new Promise((resolve) => { resolveCreate = () => resolve(mockProfile); })
      );

      renderComponent();

      await user.type(screen.getByLabelText(/Profile Name/), 'newprofile');
      await user.type(screen.getByLabelText(/Description/), 'A new profile');
      await user.click(screen.getByRole('checkbox', { name: /documents/i }));
      await user.click(screen.getByRole('button', { name: 'Create Profile' }));

      await waitFor(() => {
        expect(screen.getByLabelText(/Profile Name/)).toBeDisabled();
        expect(screen.getByRole('button', { name: 'Cancel' })).toBeDisabled();
      });

      // Clean up
      resolveCreate!();
    });
  });

  describe('Toggle Switches', () => {
    it('should toggle use_reranker', async () => {
      const user = userEvent.setup();
      renderComponent();

      const switches = screen.getAllByRole('switch');
      const rerankerSwitch = switches[0];

      expect(rerankerSwitch).toHaveAttribute('aria-checked', 'true');

      await user.click(rerankerSwitch);

      expect(rerankerSwitch).toHaveAttribute('aria-checked', 'false');
    });

    it('should toggle enabled', async () => {
      const user = userEvent.setup();
      renderComponent();

      const switches = screen.getAllByRole('switch');
      const enabledSwitch = switches[1];

      expect(enabledSwitch).toHaveAttribute('aria-checked', 'true');

      await user.click(enabledSwitch);

      expect(enabledSwitch).toHaveAttribute('aria-checked', 'false');
    });
  });

  describe('Advanced Settings', () => {
    it('should expand advanced settings on click', async () => {
      const user = userEvent.setup();
      renderComponent();

      expect(screen.queryByLabelText(/Score Threshold/)).not.toBeInTheDocument();

      await user.click(screen.getByText('Advanced Settings'));

      expect(screen.getByLabelText(/Score Threshold/)).toBeInTheDocument();
    });

    it('should show hybrid alpha when search type is hybrid', async () => {
      const user = userEvent.setup();
      renderComponent();

      await user.click(screen.getByText('Advanced Settings'));
      expect(screen.queryByLabelText(/Hybrid Alpha/)).not.toBeInTheDocument();

      await user.selectOptions(screen.getByLabelText(/Search Type/), 'hybrid');

      expect(screen.getByLabelText(/Hybrid Alpha/)).toBeInTheDocument();
    });
  });

  describe('User Interactions', () => {
    it('should call onClose when cancel button is clicked', async () => {
      const user = userEvent.setup();
      renderComponent();

      await user.click(screen.getByRole('button', { name: 'Cancel' }));

      expect(defaultProps.onClose).toHaveBeenCalledTimes(1);
    });

    it('should call onClose when backdrop is clicked', async () => {
      const user = userEvent.setup();
      const { container } = renderComponent();

      const backdrop = container.querySelector('.fixed.inset-0.bg-black\\/50');
      await user.click(backdrop!);

      expect(defaultProps.onClose).toHaveBeenCalledTimes(1);
    });

    it('should close on Escape key press', async () => {
      const user = userEvent.setup();
      renderComponent();

      await user.keyboard('{Escape}');

      expect(defaultProps.onClose).toHaveBeenCalledTimes(1);
    });

    it('should convert name to lowercase as user types', async () => {
      const user = userEvent.setup();
      renderComponent();

      const nameInput = screen.getByLabelText(/Profile Name/);
      await user.type(nameInput, 'TestProfile');

      expect(nameInput).toHaveValue('testprofile');
    });
  });

  describe('Edge Cases', () => {
    it('should display MCP tool name preview as user types', async () => {
      const user = userEvent.setup();
      renderComponent();

      expect(screen.getByText(/search_name/)).toBeInTheDocument();

      await user.type(screen.getByLabelText(/Profile Name/), 'myprofile');

      expect(screen.getByText(/search_myprofile/)).toBeInTheDocument();
    });

    it('should show character count for description', async () => {
      const user = userEvent.setup();
      renderComponent();

      // Character count is shown in format "(count/1000)"
      expect(screen.getByText(/0\/1000/)).toBeInTheDocument();

      await user.type(screen.getByLabelText(/Description/), 'Test');

      expect(screen.getByText(/4\/1000/)).toBeInTheDocument();
    });
  });
});
