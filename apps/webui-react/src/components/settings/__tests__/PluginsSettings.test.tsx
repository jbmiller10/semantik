import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@/tests/utils/test-utils';
import userEvent from '@testing-library/user-event';
import PluginsSettings from '../PluginsSettings';
import * as usePluginsModule from '@/hooks/usePlugins';
import type { PluginInfo, PluginType, HealthStatus } from '@/types/plugin';

// Mock the hooks
vi.mock('@/hooks/usePlugins', () => ({
  usePlugins: vi.fn(),
  useEnablePlugin: vi.fn(),
  useDisablePlugin: vi.fn(),
  useRefreshPluginHealth: vi.fn(),
}));

// Mock PluginConfigModal to simplify testing
vi.mock('../../plugins/PluginConfigModal', () => ({
  default: ({ plugin, onClose }: { plugin: PluginInfo; onClose: () => void }) => (
    <div data-testid="plugin-config-modal">
      <span>Configure {plugin.manifest.display_name}</span>
      <button onClick={onClose}>Close Modal</button>
    </div>
  ),
}));

// Helper to create mock plugins
const createMockPlugin = (
  id: string,
  type: PluginType,
  displayName: string,
  overrides: Partial<PluginInfo> = {}
): PluginInfo => ({
  id,
  type,
  version: '1.0.0',
  manifest: {
    id,
    type,
    version: '1.0.0',
    display_name: displayName,
    description: `A ${type} plugin`,
    author: 'Test Author',
    license: 'MIT',
    homepage: null,
    requires: [],
    capabilities: {},
  },
  enabled: true,
  config: {},
  health_status: 'healthy' as HealthStatus,
  last_health_check: '2025-01-01T00:00:00Z',
  error_message: null,
  requires_restart: false,
  ...overrides,
});

const mockPlugins: PluginInfo[] = [
  createMockPlugin('embed-1', 'embedding', 'OpenAI Embeddings'),
  createMockPlugin('embed-2', 'embedding', 'Local Embeddings'),
  createMockPlugin('chunk-1', 'chunking', 'Semantic Chunker'),
  createMockPlugin('rerank-1', 'reranker', 'Cross Encoder Reranker'),
  createMockPlugin('extract-1', 'extractor', 'Entity Extractor'),
  createMockPlugin('conn-1', 'connector', 'File Connector'),
];

describe('PluginsSettings', () => {
  const mockRefetch = vi.fn();
  const mockEnableMutateAsync = vi.fn();
  const mockDisableMutateAsync = vi.fn();
  const mockRefreshHealthMutateAsync = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();

    vi.mocked(usePluginsModule.usePlugins).mockReturnValue({
      data: mockPlugins,
      isLoading: false,
      error: null,
      refetch: mockRefetch,
      isError: false,
      isSuccess: true,
    } as unknown as ReturnType<typeof usePluginsModule.usePlugins>);

    vi.mocked(usePluginsModule.useEnablePlugin).mockReturnValue({
      mutateAsync: mockEnableMutateAsync,
      isPending: false,
    } as unknown as ReturnType<typeof usePluginsModule.useEnablePlugin>);

    vi.mocked(usePluginsModule.useDisablePlugin).mockReturnValue({
      mutateAsync: mockDisableMutateAsync,
      isPending: false,
    } as unknown as ReturnType<typeof usePluginsModule.useDisablePlugin>);

    vi.mocked(usePluginsModule.useRefreshPluginHealth).mockReturnValue({
      mutateAsync: mockRefreshHealthMutateAsync,
      isPending: false,
    } as unknown as ReturnType<typeof usePluginsModule.useRefreshPluginHealth>);
  });

  describe('loading state', () => {
    it('shows loading spinner when plugins are loading', () => {
      vi.mocked(usePluginsModule.usePlugins).mockReturnValue({
        data: undefined,
        isLoading: true,
        error: null,
        refetch: mockRefetch,
      } as unknown as ReturnType<typeof usePluginsModule.usePlugins>);

      render(<PluginsSettings />);

      expect(screen.getByText('Loading plugins...')).toBeInTheDocument();
    });
  });

  describe('error state', () => {
    it('shows error message when loading fails', () => {
      vi.mocked(usePluginsModule.usePlugins).mockReturnValue({
        data: undefined,
        isLoading: false,
        error: new Error('Network error'),
        refetch: mockRefetch,
      } as unknown as ReturnType<typeof usePluginsModule.usePlugins>);

      render(<PluginsSettings />);

      expect(screen.getByText('Error loading plugins')).toBeInTheDocument();
      expect(screen.getByText('Network error')).toBeInTheDocument();
    });

    it('shows retry button when error occurs', async () => {
      const user = userEvent.setup();
      vi.mocked(usePluginsModule.usePlugins).mockReturnValue({
        data: undefined,
        isLoading: false,
        error: new Error('Network error'),
        refetch: mockRefetch,
      } as unknown as ReturnType<typeof usePluginsModule.usePlugins>);

      render(<PluginsSettings />);

      await user.click(screen.getByText('Try again'));

      expect(mockRefetch).toHaveBeenCalled();
    });
  });

  describe('empty state', () => {
    it('shows message when no plugins are installed', () => {
      vi.mocked(usePluginsModule.usePlugins).mockReturnValue({
        data: [],
        isLoading: false,
        error: null,
        refetch: mockRefetch,
      } as unknown as ReturnType<typeof usePluginsModule.usePlugins>);

      render(<PluginsSettings />);

      expect(screen.getByText('No plugins installed')).toBeInTheDocument();
      expect(
        screen.getByText(
          'Check the Available tab for plugins you can install.'
        )
      ).toBeInTheDocument();
    });
  });

  describe('plugin list', () => {
    it('renders plugins grouped by type', () => {
      render(<PluginsSettings />);

      // Check type headers
      expect(screen.getByText('Embedding Providers')).toBeInTheDocument();
      expect(screen.getByText('Chunking Strategies')).toBeInTheDocument();
      expect(screen.getByText('Connectors')).toBeInTheDocument();
      expect(screen.getByText('Rerankers')).toBeInTheDocument();
      expect(screen.getByText('Extractors')).toBeInTheDocument();
    });

    it('renders all plugin display names', () => {
      render(<PluginsSettings />);

      expect(screen.getByText('OpenAI Embeddings')).toBeInTheDocument();
      expect(screen.getByText('Local Embeddings')).toBeInTheDocument();
      expect(screen.getByText('Semantic Chunker')).toBeInTheDocument();
      expect(screen.getByText('Cross Encoder Reranker')).toBeInTheDocument();
      expect(screen.getByText('Entity Extractor')).toBeInTheDocument();
      expect(screen.getByText('File Connector')).toBeInTheDocument();
    });

    it('shows plugin count per type', () => {
      render(<PluginsSettings />);

      // 2 embedding providers
      expect(screen.getByText('(2)')).toBeInTheDocument();
    });

    it('does not show section for types with no plugins', () => {
      vi.mocked(usePluginsModule.usePlugins).mockReturnValue({
        data: [createMockPlugin('embed-1', 'embedding', 'Test Embed')],
        isLoading: false,
        error: null,
        refetch: mockRefetch,
      } as unknown as ReturnType<typeof usePluginsModule.usePlugins>);

      render(<PluginsSettings />);

      expect(screen.getByText('Embedding Providers')).toBeInTheDocument();
      expect(screen.queryByText('Chunking Strategies')).not.toBeInTheDocument();
      expect(screen.queryByText('Rerankers')).not.toBeInTheDocument();
    });
  });

  describe('header', () => {
    it('shows description text', () => {
      render(<PluginsSettings />);

      expect(
        screen.getByText(
          'Manage installed plugins. Changes require a service restart to take effect.'
        )
      ).toBeInTheDocument();
    });

    it('shows refresh button', () => {
      render(<PluginsSettings />);

      expect(
        screen.getByRole('button', { name: /refresh/i })
      ).toBeInTheDocument();
    });

    it('calls refetch when refresh button is clicked', async () => {
      const user = userEvent.setup();
      render(<PluginsSettings />);

      await user.click(screen.getByRole('button', { name: /refresh/i }));

      expect(mockRefetch).toHaveBeenCalled();
    });
  });

  describe('plugin enable/disable', () => {
    it('calls enable mutation when enabling a plugin', async () => {
      const user = userEvent.setup();
      vi.mocked(usePluginsModule.usePlugins).mockReturnValue({
        data: [createMockPlugin('test-1', 'embedding', 'Test', { enabled: false })],
        isLoading: false,
        error: null,
        refetch: mockRefetch,
      } as unknown as ReturnType<typeof usePluginsModule.usePlugins>);

      mockEnableMutateAsync.mockResolvedValue({});

      render(<PluginsSettings />);

      const toggle = screen.getByRole('switch');
      await user.click(toggle);

      await waitFor(() => {
        expect(mockEnableMutateAsync).toHaveBeenCalledWith('test-1');
      });
    });

    it('calls disable mutation when disabling a plugin', async () => {
      const user = userEvent.setup();
      vi.mocked(usePluginsModule.usePlugins).mockReturnValue({
        data: [createMockPlugin('test-1', 'embedding', 'Test', { enabled: true })],
        isLoading: false,
        error: null,
        refetch: mockRefetch,
      } as unknown as ReturnType<typeof usePluginsModule.usePlugins>);

      mockDisableMutateAsync.mockResolvedValue({});

      render(<PluginsSettings />);

      const toggle = screen.getByRole('switch');
      await user.click(toggle);

      // Confirm disable
      await user.click(screen.getByRole('button', { name: /yes, disable/i }));

      await waitFor(() => {
        expect(mockDisableMutateAsync).toHaveBeenCalledWith('test-1');
      });
    });
  });

  describe('plugin configuration', () => {
    it('opens config modal when configure is clicked', async () => {
      const user = userEvent.setup();
      vi.mocked(usePluginsModule.usePlugins).mockReturnValue({
        data: [createMockPlugin('test-1', 'embedding', 'Test Plugin')],
        isLoading: false,
        error: null,
        refetch: mockRefetch,
      } as unknown as ReturnType<typeof usePluginsModule.usePlugins>);

      render(<PluginsSettings />);

      await user.click(screen.getByRole('button', { name: /configure/i }));

      expect(screen.getByTestId('plugin-config-modal')).toBeInTheDocument();
      expect(screen.getByText('Configure Test Plugin')).toBeInTheDocument();
    });

    it('closes config modal when close is clicked', async () => {
      const user = userEvent.setup();
      vi.mocked(usePluginsModule.usePlugins).mockReturnValue({
        data: [createMockPlugin('test-1', 'embedding', 'Test Plugin')],
        isLoading: false,
        error: null,
        refetch: mockRefetch,
      } as unknown as ReturnType<typeof usePluginsModule.usePlugins>);

      render(<PluginsSettings />);

      await user.click(screen.getByRole('button', { name: /configure/i }));
      await user.click(screen.getByRole('button', { name: /close modal/i }));

      expect(screen.queryByTestId('plugin-config-modal')).not.toBeInTheDocument();
    });
  });

  describe('health check', () => {
    it('calls refresh health mutation when health is clicked', async () => {
      const user = userEvent.setup();
      vi.mocked(usePluginsModule.usePlugins).mockReturnValue({
        data: [createMockPlugin('test-1', 'embedding', 'Test')],
        isLoading: false,
        error: null,
        refetch: mockRefetch,
      } as unknown as ReturnType<typeof usePluginsModule.usePlugins>);

      mockRefreshHealthMutateAsync.mockResolvedValue({});

      render(<PluginsSettings />);

      await user.click(screen.getByText('Healthy'));

      await waitFor(() => {
        expect(mockRefreshHealthMutateAsync).toHaveBeenCalledWith('test-1');
      });
    });
  });
});
