import { render, screen, waitFor, within } from '@testing-library/react';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import MCPProfilesSettings from '../MCPProfilesSettings';
import { useMCPProfiles } from '../../../hooks/useMCPProfiles';
import type { MCPProfile } from '../../../types/mcp-profile';

// Mock the hooks
vi.mock('../../../hooks/useMCPProfiles', () => ({
  useMCPProfiles: vi.fn(),
  useToggleMCPProfileEnabled: vi.fn(() => ({
    mutateAsync: vi.fn(),
    isPending: false,
  })),
}));

// Mock the child components to simplify testing
vi.mock('../../mcp/ProfileCard', () => ({
  default: ({ profile, onEdit, onDelete, onViewConfig }: {
    profile: MCPProfile;
    onEdit: () => void;
    onDelete: () => void;
    onViewConfig: () => void;
  }) => (
    <div data-testid={`profile-card-${profile.id}`}>
      <span>{profile.name}</span>
      <button onClick={onEdit}>Edit</button>
      <button onClick={onDelete}>Delete</button>
      <button onClick={onViewConfig}>Config</button>
    </div>
  ),
}));

vi.mock('../../mcp/ProfileFormModal', () => ({
  default: ({ profile, onClose }: { profile: MCPProfile | null; onClose: () => void }) => (
    <div data-testid="profile-form-modal">
      <span>{profile ? 'Edit Mode' : 'Create Mode'}</span>
      <button onClick={onClose}>Close Form</button>
    </div>
  ),
}));

vi.mock('../../mcp/DeleteConfirmModal', () => ({
  default: ({ profile, onClose }: { profile: MCPProfile; onClose: () => void }) => (
    <div data-testid="delete-confirm-modal">
      <span>Delete {profile.name}?</span>
      <button onClick={onClose}>Close Delete</button>
    </div>
  ),
}));

vi.mock('../../mcp/ConfigModal', () => ({
  default: ({ profile, onClose }: { profile: MCPProfile; onClose: () => void }) => (
    <div data-testid="config-modal">
      <span>Config for {profile.name}</span>
      <button onClick={onClose}>Close Config</button>
    </div>
  ),
}));

const mockUseMCPProfiles = useMCPProfiles as vi.MockedFunction<typeof useMCPProfiles>;

// Test data
const mockProfiles: MCPProfile[] = [
  {
    id: 'profile-1',
    name: 'docs-profile',
    description: 'Search documentation',
    enabled: true,
    search_type: 'semantic',
    result_count: 10,
    use_reranker: true,
    score_threshold: null,
    hybrid_alpha: null,
    collections: [{ id: 'col-1', name: 'Docs' }],
    created_at: '2025-01-01T00:00:00Z',
    updated_at: '2025-01-01T00:00:00Z',
  },
  {
    id: 'profile-2',
    name: 'code-profile',
    description: 'Search code',
    enabled: false,
    search_type: 'code',
    result_count: 20,
    use_reranker: false,
    score_threshold: null,
    hybrid_alpha: null,
    collections: [{ id: 'col-2', name: 'Code' }],
    created_at: '2025-01-02T00:00:00Z',
    updated_at: '2025-01-02T00:00:00Z',
  },
];

// Helper function to render component with providers
const renderComponent = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });

  return render(
    <QueryClientProvider client={queryClient}>
      <MCPProfilesSettings />
    </QueryClientProvider>
  );
};

describe('MCPProfilesSettings', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Loading State', () => {
    it('should show loading state while fetching profiles', () => {
      mockUseMCPProfiles.mockReturnValue({
        data: undefined,
        isLoading: true,
        error: null,
        refetch: vi.fn(),
      } as unknown as ReturnType<typeof useMCPProfiles>);

      renderComponent();

      expect(screen.getByText('Loading MCP profiles...')).toBeInTheDocument();
    });
  });

  describe('Error State', () => {
    it('should show error message when fetch fails', () => {
      mockUseMCPProfiles.mockReturnValue({
        data: undefined,
        isLoading: false,
        error: new Error('Failed to load profiles'),
        refetch: vi.fn(),
      } as unknown as ReturnType<typeof useMCPProfiles>);

      renderComponent();

      expect(screen.getByText('Error loading MCP profiles')).toBeInTheDocument();
      expect(screen.getByText('Failed to load profiles')).toBeInTheDocument();
    });

    it('should show retry button on error', async () => {
      const mockRefetch = vi.fn();
      mockUseMCPProfiles.mockReturnValue({
        data: undefined,
        isLoading: false,
        error: new Error('Failed to load'),
        refetch: mockRefetch,
      } as unknown as ReturnType<typeof useMCPProfiles>);

      const user = userEvent.setup();
      renderComponent();

      const retryButton = screen.getByRole('button', { name: /try again/i });
      await user.click(retryButton);

      expect(mockRefetch).toHaveBeenCalled();
    });
  });

  describe('Empty State', () => {
    it('should show empty state when no profiles exist', () => {
      mockUseMCPProfiles.mockReturnValue({
        data: [],
        isLoading: false,
        error: null,
        refetch: vi.fn(),
      } as unknown as ReturnType<typeof useMCPProfiles>);

      renderComponent();

      expect(screen.getByText('No MCP profiles')).toBeInTheDocument();
      expect(screen.getByText(/Get started by creating a new profile/)).toBeInTheDocument();
    });

    it('should show create button in empty state', () => {
      mockUseMCPProfiles.mockReturnValue({
        data: [],
        isLoading: false,
        error: null,
        refetch: vi.fn(),
      } as unknown as ReturnType<typeof useMCPProfiles>);

      renderComponent();

      // There are two Create Profile buttons - header and empty state
      const createButtons = screen.getAllByRole('button', { name: /create profile/i });
      expect(createButtons.length).toBeGreaterThanOrEqual(1);
    });
  });

  describe('Profiles List', () => {
    beforeEach(() => {
      mockUseMCPProfiles.mockReturnValue({
        data: mockProfiles,
        isLoading: false,
        error: null,
        refetch: vi.fn(),
      } as unknown as ReturnType<typeof useMCPProfiles>);
    });

    it('should render page title and description', () => {
      renderComponent();

      expect(screen.getByRole('heading', { name: 'MCP Profiles' })).toBeInTheDocument();
      expect(screen.getByText(/Configure search profiles for MCP clients/)).toBeInTheDocument();
    });

    it('should render create profile button in header', () => {
      renderComponent();

      expect(screen.getByRole('button', { name: /create profile/i })).toBeInTheDocument();
    });

    it('should render all profiles', () => {
      renderComponent();

      expect(screen.getByTestId('profile-card-profile-1')).toBeInTheDocument();
      expect(screen.getByTestId('profile-card-profile-2')).toBeInTheDocument();
      expect(screen.getByText('docs-profile')).toBeInTheDocument();
      expect(screen.getByText('code-profile')).toBeInTheDocument();
    });
  });

  describe('Create Profile Modal', () => {
    beforeEach(() => {
      mockUseMCPProfiles.mockReturnValue({
        data: mockProfiles,
        isLoading: false,
        error: null,
        refetch: vi.fn(),
      } as unknown as ReturnType<typeof useMCPProfiles>);
    });

    it('should open create modal when create button is clicked', async () => {
      const user = userEvent.setup();
      renderComponent();

      await user.click(screen.getByRole('button', { name: /create profile/i }));

      expect(screen.getByTestId('profile-form-modal')).toBeInTheDocument();
      expect(screen.getByText('Create Mode')).toBeInTheDocument();
    });

    it('should close create modal when close is triggered', async () => {
      const user = userEvent.setup();
      renderComponent();

      await user.click(screen.getByRole('button', { name: /create profile/i }));
      expect(screen.getByTestId('profile-form-modal')).toBeInTheDocument();

      await user.click(screen.getByRole('button', { name: 'Close Form' }));

      await waitFor(() => {
        expect(screen.queryByTestId('profile-form-modal')).not.toBeInTheDocument();
      });
    });
  });

  describe('Edit Profile Modal', () => {
    beforeEach(() => {
      mockUseMCPProfiles.mockReturnValue({
        data: mockProfiles,
        isLoading: false,
        error: null,
        refetch: vi.fn(),
      } as unknown as ReturnType<typeof useMCPProfiles>);
    });

    it('should open edit modal when edit is clicked on a profile', async () => {
      const user = userEvent.setup();
      renderComponent();

      const profileCard = screen.getByTestId('profile-card-profile-1');
      await user.click(within(profileCard).getByRole('button', { name: 'Edit' }));

      expect(screen.getByTestId('profile-form-modal')).toBeInTheDocument();
      expect(screen.getByText('Edit Mode')).toBeInTheDocument();
    });
  });

  describe('Delete Profile Modal', () => {
    beforeEach(() => {
      mockUseMCPProfiles.mockReturnValue({
        data: mockProfiles,
        isLoading: false,
        error: null,
        refetch: vi.fn(),
      } as unknown as ReturnType<typeof useMCPProfiles>);
    });

    it('should open delete modal when delete is clicked on a profile', async () => {
      const user = userEvent.setup();
      renderComponent();

      const profileCard = screen.getByTestId('profile-card-profile-1');
      await user.click(within(profileCard).getByRole('button', { name: 'Delete' }));

      expect(screen.getByTestId('delete-confirm-modal')).toBeInTheDocument();
      expect(screen.getByText('Delete docs-profile?')).toBeInTheDocument();
    });

    it('should close delete modal when close is triggered', async () => {
      const user = userEvent.setup();
      renderComponent();

      const profileCard = screen.getByTestId('profile-card-profile-1');
      await user.click(within(profileCard).getByRole('button', { name: 'Delete' }));
      expect(screen.getByTestId('delete-confirm-modal')).toBeInTheDocument();

      await user.click(screen.getByRole('button', { name: 'Close Delete' }));

      await waitFor(() => {
        expect(screen.queryByTestId('delete-confirm-modal')).not.toBeInTheDocument();
      });
    });
  });

  describe('Config Modal', () => {
    beforeEach(() => {
      mockUseMCPProfiles.mockReturnValue({
        data: mockProfiles,
        isLoading: false,
        error: null,
        refetch: vi.fn(),
      } as unknown as ReturnType<typeof useMCPProfiles>);
    });

    it('should open config modal when config is clicked on a profile', async () => {
      const user = userEvent.setup();
      renderComponent();

      const profileCard = screen.getByTestId('profile-card-profile-1');
      await user.click(within(profileCard).getByRole('button', { name: 'Config' }));

      expect(screen.getByTestId('config-modal')).toBeInTheDocument();
      expect(screen.getByText('Config for docs-profile')).toBeInTheDocument();
    });

    it('should close config modal when close is triggered', async () => {
      const user = userEvent.setup();
      renderComponent();

      const profileCard = screen.getByTestId('profile-card-profile-1');
      await user.click(within(profileCard).getByRole('button', { name: 'Config' }));
      expect(screen.getByTestId('config-modal')).toBeInTheDocument();

      await user.click(screen.getByRole('button', { name: 'Close Config' }));

      await waitFor(() => {
        expect(screen.queryByTestId('config-modal')).not.toBeInTheDocument();
      });
    });
  });

  describe('Empty State Create Button', () => {
    it('should open create modal from empty state button', async () => {
      mockUseMCPProfiles.mockReturnValue({
        data: [],
        isLoading: false,
        error: null,
        refetch: vi.fn(),
      } as unknown as ReturnType<typeof useMCPProfiles>);

      const user = userEvent.setup();
      renderComponent();

      // There are two Create Profile buttons, get the first one
      const createButtons = screen.getAllByRole('button', { name: /create profile/i });
      await user.click(createButtons[0]);

      expect(screen.getByTestId('profile-form-modal')).toBeInTheDocument();
    });
  });
});
