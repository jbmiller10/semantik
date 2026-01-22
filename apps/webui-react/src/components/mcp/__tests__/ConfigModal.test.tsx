import { render, screen, waitFor } from '@testing-library/react';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import ConfigModal from '../ConfigModal';
import { useMCPProfileConfig } from '../../../hooks/useMCPProfiles';
import type { MCPProfile, MCPClientConfig } from '../../../types/mcp-profile';

// Mock the hooks
vi.mock('../../../hooks/useMCPProfiles', () => ({
  useMCPProfileConfig: vi.fn(),
}));

const mockUseMCPProfileConfig = useMCPProfileConfig as vi.MockedFunction<typeof useMCPProfileConfig>;

// Clipboard mock - will be properly set up in beforeEach
let mockWriteText: ReturnType<typeof vi.fn>;

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

const mockConfig: MCPClientConfig = {
  server_name: 'semantik-test-profile',
  command: 'npx',
  args: ['-y', '@anthropic/mcp-server-semantik', '--profile', 'test-profile'],
  env: {
    SEMANTIK_API_URL: 'http://localhost:8000',
    SEMANTIK_API_KEY: '<your-access-token-or-api-key>',
  },
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
      <ConfigModal {...defaultProps} {...props} />
    </QueryClientProvider>
  );
};

describe('ConfigModal', () => {
  beforeEach(() => {
    vi.clearAllMocks();

    // Set up fresh clipboard mock for each test
    mockWriteText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, 'clipboard', {
      value: { writeText: mockWriteText },
      writable: true,
      configurable: true,
    });

    // Mock execCommand for fallback (it doesn't exist in JSDOM by default)
    // @ts-expect-error - execCommand is deprecated
    document.execCommand = vi.fn().mockReturnValue(true);
  });

  describe('Loading State', () => {
    it('should show loading spinner while fetching config', () => {
      mockUseMCPProfileConfig.mockReturnValue({
        data: undefined,
        isLoading: true,
        error: null,
      } as ReturnType<typeof useMCPProfileConfig>);

      renderComponent();

      expect(screen.getByText('Loading configuration...')).toBeInTheDocument();
    });
  });

  describe('Error State', () => {
    it('should show error message when config fetch fails', () => {
      mockUseMCPProfileConfig.mockReturnValue({
        data: undefined,
        isLoading: false,
        error: new Error('Failed to load config'),
      } as ReturnType<typeof useMCPProfileConfig>);

      renderComponent();

      expect(screen.getByText('Error loading configuration')).toBeInTheDocument();
      expect(screen.getByText('Failed to load config')).toBeInTheDocument();
    });

    it('should handle non-Error objects gracefully', () => {
      mockUseMCPProfileConfig.mockReturnValue({
        data: undefined,
        isLoading: false,
        error: 'Unknown error',
      } as unknown as ReturnType<typeof useMCPProfileConfig>);

      renderComponent();

      expect(screen.getByText('Unknown error')).toBeInTheDocument();
    });
  });

  describe('Config Display', () => {
    beforeEach(() => {
      mockUseMCPProfileConfig.mockReturnValue({
        data: mockConfig,
        isLoading: false,
        error: null,
      } as ReturnType<typeof useMCPProfileConfig>);
    });

    it('should render modal with correct title', () => {
      renderComponent();

      expect(screen.getByRole('heading', { name: 'Connection Info' })).toBeInTheDocument();
      expect(screen.getByText(/Configure MCP client to use "test-profile" profile/)).toBeInTheDocument();
    });

    it('should display the MCP tool name', () => {
      renderComponent();

      expect(screen.getByText('MCP Tool Name')).toBeInTheDocument();
      // Tool name appears multiple times, just check it exists
      expect(screen.getAllByText('search_test-profile').length).toBeGreaterThan(0);
    });

    it('should display config file locations for all platforms', () => {
      renderComponent();

      expect(screen.getByText('macOS:')).toBeInTheDocument();
      // The config path appears multiple times, check it exists at least once
      expect(screen.getAllByText(/claude_desktop_config\.json/).length).toBeGreaterThan(0);
      expect(screen.getByText('Linux:')).toBeInTheDocument();
      expect(screen.getByText('Windows:')).toBeInTheDocument();
    });

    it('should display JSON config', () => {
      renderComponent();

      expect(screen.getByText('Add to mcpServers')).toBeInTheDocument();
      // Check for parts of the JSON config
      const preElement = document.querySelector('pre');
      expect(preElement).toBeInTheDocument();
      expect(preElement?.textContent).toContain('semantik-test-profile');
      expect(preElement?.textContent).toContain('npx');
    });

    it('should display auth token warning', () => {
      renderComponent();

      expect(screen.getByText('Replace the auth token')).toBeInTheDocument();
      // The token placeholder appears in multiple places (JSON and warning), verify at least one exists
      expect(screen.getAllByText(/<your-access-token-or-api-key>/).length).toBeGreaterThan(0);
    });

    it('should display usage note', () => {
      renderComponent();

      expect(screen.getByText('How it works')).toBeInTheDocument();
      // "Restart Claude" appears in multiple places (tool note + usage note), verify at least one exists
      expect(screen.getAllByText(/Restart Claude/).length).toBeGreaterThan(0);
    });

    it('should have correct dialog role and aria attributes', () => {
      renderComponent();

      const dialog = screen.getByRole('dialog');
      expect(dialog).toHaveAttribute('aria-modal', 'true');
      expect(dialog).toHaveAttribute('aria-labelledby', 'config-modal-title');
    });
  });

  describe('Copy Functionality', () => {
    beforeEach(() => {
      mockUseMCPProfileConfig.mockReturnValue({
        data: mockConfig,
        isLoading: false,
        error: null,
      } as ReturnType<typeof useMCPProfileConfig>);
    });

    it('should trigger copy action when Copy tool name button is clicked', async () => {
      const user = userEvent.setup();
      renderComponent();

      const copyButton = screen.getByRole('button', { name: /copy tool name/i });
      await user.click(copyButton);

      // Verify copy happened by checking UI feedback (Copied/Failed state)
      // In JSDOM environment, actual clipboard API behavior varies
      await waitFor(() => {
        // After clicking, the button text should change from "Copy"
        // Either "Copied" (success) or "Failed" (error) - but not "Copy"
        const hasFeedback =
          screen.queryByText('Copied') !== null ||
          screen.queryByText('Failed') !== null;
        expect(hasFeedback).toBe(true);
      });
    });

    it('should show "Copied" feedback after copying tool name', async () => {
      const user = userEvent.setup();
      renderComponent();

      const copyButton = screen.getByRole('button', { name: /copy tool name/i });
      await user.click(copyButton);

      await waitFor(() => {
        expect(screen.getByText('Copied')).toBeInTheDocument();
      });
    });

    it('should trigger copy action when Copy config button is clicked', async () => {
      const user = userEvent.setup();
      renderComponent();

      const copyConfigButton = screen.getByRole('button', { name: /copy config/i });
      await user.click(copyConfigButton);

      // Verify copy happened by checking UI feedback
      await waitFor(() => {
        const hasFeedback =
          screen.queryByText('Copied!') !== null ||
          screen.queryByText('Failed!') !== null;
        expect(hasFeedback).toBe(true);
      });
    });

    it('should show "Copied!" feedback after copying config', async () => {
      const user = userEvent.setup();
      renderComponent();

      const copyConfigButton = screen.getByRole('button', { name: /copy config/i });
      await user.click(copyConfigButton);

      await waitFor(() => {
        expect(screen.getByText('Copied!')).toBeInTheDocument();
      });
    });

    it('should show error state on copy failure', async () => {
      const user = userEvent.setup();

      // Override clipboard mock to reject
      mockWriteText = vi.fn().mockRejectedValue(new Error('Copy failed'));
      Object.defineProperty(navigator, 'clipboard', {
        value: { writeText: mockWriteText },
        writable: true,
        configurable: true,
      });

      // Override execCommand to fail as well (the fallback)
      // @ts-expect-error - execCommand is deprecated but we need to mock it
      document.execCommand = vi.fn().mockReturnValue(false);

      renderComponent();

      const copyButton = screen.getByRole('button', { name: /copy tool name/i });
      await user.click(copyButton);

      await waitFor(() => {
        expect(screen.getByText('Failed')).toBeInTheDocument();
      });
    });
  });

  describe('User Interactions', () => {
    beforeEach(() => {
      mockUseMCPProfileConfig.mockReturnValue({
        data: mockConfig,
        isLoading: false,
        error: null,
      } as ReturnType<typeof useMCPProfileConfig>);
    });

    it('should call onClose when close button in header is clicked', async () => {
      const user = userEvent.setup();
      renderComponent();

      // The X button in the header
      const closeButtons = screen.getAllByRole('button');
      const headerCloseButton = closeButtons.find(btn => btn.querySelector('svg path[d*="M6 18L18 6"]'));
      await user.click(headerCloseButton!);

      expect(defaultProps.onClose).toHaveBeenCalledTimes(1);
    });

    it('should call onClose when Close button in footer is clicked', async () => {
      const user = userEvent.setup();
      renderComponent();

      const closeButton = screen.getByRole('button', { name: 'Close' });
      await user.click(closeButton);

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
  });

  describe('Edge Cases', () => {
    it('should handle profile with special characters in name', () => {
      mockUseMCPProfileConfig.mockReturnValue({
        data: mockConfig,
        isLoading: false,
        error: null,
      } as ReturnType<typeof useMCPProfileConfig>);

      renderComponent({
        profile: { ...mockProfile, name: 'my-profile_v2' },
      });

      // Tool name appears multiple times, check it exists
      expect(screen.getAllByText('search_my-profile_v2').length).toBeGreaterThan(0);
    });
  });
});
