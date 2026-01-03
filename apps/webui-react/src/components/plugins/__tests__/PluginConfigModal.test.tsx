import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@/tests/utils/test-utils';
import userEvent from '@testing-library/user-event';
import PluginConfigModal from '../PluginConfigModal';
import * as usePluginsModule from '@/hooks/usePlugins';
import type { PluginInfo, PluginType, HealthStatus, PluginConfigSchema } from '@/types/plugin';

// Mock the hooks
vi.mock('@/hooks/usePlugins', () => ({
  usePluginConfigSchema: vi.fn(),
  useUpdatePluginConfig: vi.fn(),
}));

// Helper to create mock plugin
const createMockPlugin = (overrides: Partial<PluginInfo> = {}): PluginInfo => ({
  id: 'test-plugin',
  type: 'embedding' as PluginType,
  version: '1.0.0',
  manifest: {
    id: 'test-plugin',
    type: 'embedding' as PluginType,
    version: '1.0.0',
    display_name: 'Test Plugin',
    description: 'A test plugin for testing purposes',
    author: 'Test Author',
    license: 'MIT',
    homepage: 'https://example.com',
    requires: [],
    capabilities: {},
  },
  enabled: true,
  config: { api_key: 'test-key' },
  health_status: 'healthy' as HealthStatus,
  last_health_check: '2025-01-01T00:00:00Z',
  error_message: null,
  requires_restart: false,
  ...overrides,
});

const mockSchema: PluginConfigSchema = {
  type: 'object',
  properties: {
    api_key: {
      type: 'string',
      title: 'API Key',
      description: 'Your API key',
    },
    timeout: {
      type: 'number',
      title: 'Timeout',
      minimum: 0,
      maximum: 60,
    },
  },
  required: ['api_key'],
};

describe('PluginConfigModal', () => {
  const mockOnClose = vi.fn();
  const mockMutateAsync = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();

    vi.mocked(usePluginsModule.usePluginConfigSchema).mockReturnValue({
      data: mockSchema,
      isLoading: false,
      error: null,
      isError: false,
      isSuccess: true,
      isPending: false,
    } as ReturnType<typeof usePluginsModule.usePluginConfigSchema>);

    vi.mocked(usePluginsModule.useUpdatePluginConfig).mockReturnValue({
      mutateAsync: mockMutateAsync,
      isPending: false,
    } as unknown as ReturnType<typeof usePluginsModule.useUpdatePluginConfig>);
  });

  describe('rendering', () => {
    it('renders modal with plugin name and version', () => {
      render(
        <PluginConfigModal plugin={createMockPlugin()} onClose={mockOnClose} />
      );

      expect(screen.getByText('Configure Test Plugin')).toBeInTheDocument();
      expect(screen.getByText('v1.0.0')).toBeInTheDocument();
    });

    it('renders the config form with schema', () => {
      render(
        <PluginConfigModal plugin={createMockPlugin()} onClose={mockOnClose} />
      );

      expect(screen.getByLabelText(/api key/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/timeout/i)).toBeInTheDocument();
    });

    it('pre-fills form with existing config values', () => {
      const plugin = createMockPlugin({
        config: { api_key: 'my-key', timeout: 30 },
      });

      render(<PluginConfigModal plugin={plugin} onClose={mockOnClose} />);

      expect(screen.getByDisplayValue('my-key')).toBeInTheDocument();
      expect(screen.getByDisplayValue('30')).toBeInTheDocument();
    });

    it('shows restart warning message', () => {
      render(
        <PluginConfigModal plugin={createMockPlugin()} onClose={mockOnClose} />
      );

      expect(
        screen.getByText('Changes require service restart')
      ).toBeInTheDocument();
    });
  });

  describe('loading state', () => {
    it('shows loading indicator while fetching schema', () => {
      vi.mocked(usePluginsModule.usePluginConfigSchema).mockReturnValue({
        data: undefined,
        isLoading: true,
        error: null,
        isError: false,
        isSuccess: false,
        isPending: true,
      } as ReturnType<typeof usePluginsModule.usePluginConfigSchema>);

      render(
        <PluginConfigModal plugin={createMockPlugin()} onClose={mockOnClose} />
      );

      expect(screen.getByText('Loading configuration...')).toBeInTheDocument();
    });
  });

  describe('error state', () => {
    it('shows error message when schema fails to load', () => {
      vi.mocked(usePluginsModule.usePluginConfigSchema).mockReturnValue({
        data: undefined,
        isLoading: false,
        error: new Error('Network error'),
        isError: true,
        isSuccess: false,
        isPending: false,
      } as ReturnType<typeof usePluginsModule.usePluginConfigSchema>);

      render(
        <PluginConfigModal plugin={createMockPlugin()} onClose={mockOnClose} />
      );

      expect(
        screen.getByText('Failed to load configuration schema')
      ).toBeInTheDocument();
    });
  });

  describe('no config', () => {
    it('shows message when plugin has no configuration', () => {
      vi.mocked(usePluginsModule.usePluginConfigSchema).mockReturnValue({
        data: null,
        isLoading: false,
        error: null,
        isError: false,
        isSuccess: true,
        isPending: false,
      } as unknown as ReturnType<typeof usePluginsModule.usePluginConfigSchema>);

      render(
        <PluginConfigModal plugin={createMockPlugin()} onClose={mockOnClose} />
      );

      expect(
        screen.getByText('This plugin has no configuration options.')
      ).toBeInTheDocument();
    });
  });

  describe('closing modal', () => {
    it('calls onClose when Cancel button is clicked', async () => {
      const user = userEvent.setup();

      render(
        <PluginConfigModal plugin={createMockPlugin()} onClose={mockOnClose} />
      );

      await user.click(screen.getByRole('button', { name: /cancel/i }));

      expect(mockOnClose).toHaveBeenCalled();
    });

    it('calls onClose when close icon is clicked', async () => {
      const user = userEvent.setup();

      render(
        <PluginConfigModal plugin={createMockPlugin()} onClose={mockOnClose} />
      );

      const closeButton = screen.getByRole('button', { name: /close/i });
      await user.click(closeButton);

      expect(mockOnClose).toHaveBeenCalled();
    });

    it('calls onClose when Cancel button is clicked', async () => {
      const user = userEvent.setup();

      render(
        <PluginConfigModal plugin={createMockPlugin()} onClose={mockOnClose} />
      );

      // Click Cancel button
      const cancelButton = screen.getByRole('button', { name: /cancel/i });
      await user.click(cancelButton);

      expect(mockOnClose).toHaveBeenCalled();
    });
  });

  describe('form submission', () => {
    it('calls mutateAsync with updated config on submit', async () => {
      const user = userEvent.setup();
      mockMutateAsync.mockResolvedValue({});

      render(
        <PluginConfigModal
          plugin={createMockPlugin({ config: { api_key: 'old-key' } })}
          onClose={mockOnClose}
        />
      );

      const input = screen.getByLabelText(/api key/i);
      await user.clear(input);
      await user.type(input, 'new-key');

      await user.click(screen.getByRole('button', { name: /save changes/i }));

      await waitFor(() => {
        expect(mockMutateAsync).toHaveBeenCalledWith({
          pluginId: 'test-plugin',
          config: expect.objectContaining({ api_key: 'new-key' }),
        });
      });
    });

    it('closes modal after successful save', async () => {
      const user = userEvent.setup();
      mockMutateAsync.mockResolvedValue({});

      render(
        <PluginConfigModal plugin={createMockPlugin()} onClose={mockOnClose} />
      );

      await user.click(screen.getByRole('button', { name: /save changes/i }));

      await waitFor(() => {
        expect(mockOnClose).toHaveBeenCalled();
      });
    });

    it('shows error message when save fails', async () => {
      const user = userEvent.setup();
      mockMutateAsync.mockRejectedValue(new Error('Save failed'));

      render(
        <PluginConfigModal plugin={createMockPlugin()} onClose={mockOnClose} />
      );

      await user.click(screen.getByRole('button', { name: /save changes/i }));

      await waitFor(() => {
        expect(screen.getByText('Save failed')).toBeInTheDocument();
      });
    });

    it('does not close modal when save fails', async () => {
      const user = userEvent.setup();
      mockMutateAsync.mockRejectedValue(new Error('Save failed'));

      render(
        <PluginConfigModal plugin={createMockPlugin()} onClose={mockOnClose} />
      );

      await user.click(screen.getByRole('button', { name: /save changes/i }));

      await waitFor(() => {
        expect(screen.getByText('Save failed')).toBeInTheDocument();
      });

      expect(mockOnClose).not.toHaveBeenCalled();
    });
  });

  describe('validation', () => {
    it('shows validation error for required field', async () => {
      const user = userEvent.setup();

      render(
        <PluginConfigModal
          plugin={createMockPlugin({ config: {} })}
          onClose={mockOnClose}
        />
      );

      await user.click(screen.getByRole('button', { name: /save changes/i }));

      await waitFor(() => {
        expect(screen.getByText('This field is required')).toBeInTheDocument();
      });

      expect(mockMutateAsync).not.toHaveBeenCalled();
    });

    it('clears validation errors when values change', async () => {
      const user = userEvent.setup();

      render(
        <PluginConfigModal
          plugin={createMockPlugin({ config: {} })}
          onClose={mockOnClose}
        />
      );

      // Trigger validation error
      await user.click(screen.getByRole('button', { name: /save changes/i }));

      await waitFor(() => {
        expect(screen.getByText('This field is required')).toBeInTheDocument();
      });

      // Type a value - error should clear on next submit
      await user.type(screen.getByLabelText(/api key/i), 'new-value');

      mockMutateAsync.mockResolvedValue({});
      await user.click(screen.getByRole('button', { name: /save changes/i }));

      await waitFor(() => {
        expect(mockMutateAsync).toHaveBeenCalled();
      });
    });
  });

  describe('saving state', () => {
    it('disables Save button and shows spinner while saving', () => {
      vi.mocked(usePluginsModule.useUpdatePluginConfig).mockReturnValue({
        mutateAsync: mockMutateAsync,
        isPending: true,
      } as unknown as ReturnType<typeof usePluginsModule.useUpdatePluginConfig>);

      render(
        <PluginConfigModal plugin={createMockPlugin()} onClose={mockOnClose} />
      );

      const saveButton = screen.getByRole('button', { name: /saving/i });
      expect(saveButton).toBeDisabled();
      expect(screen.getByText('Saving...')).toBeInTheDocument();
    });

    it('disables Save button when schema is loading', () => {
      vi.mocked(usePluginsModule.usePluginConfigSchema).mockReturnValue({
        data: undefined,
        isLoading: true,
        error: null,
        isError: false,
        isSuccess: false,
        isPending: true,
      } as ReturnType<typeof usePluginsModule.usePluginConfigSchema>);

      render(
        <PluginConfigModal plugin={createMockPlugin()} onClose={mockOnClose} />
      );

      expect(
        screen.getByRole('button', { name: /save changes/i })
      ).toBeDisabled();
    });

    it('disables Save button when schema has error', () => {
      vi.mocked(usePluginsModule.usePluginConfigSchema).mockReturnValue({
        data: undefined,
        isLoading: false,
        error: new Error('Error'),
        isError: true,
        isSuccess: false,
        isPending: false,
      } as ReturnType<typeof usePluginsModule.usePluginConfigSchema>);

      render(
        <PluginConfigModal plugin={createMockPlugin()} onClose={mockOnClose} />
      );

      expect(
        screen.getByRole('button', { name: /save changes/i })
      ).toBeDisabled();
    });
  });
});
