import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@/tests/utils/test-utils';
import userEvent from '@testing-library/user-event';
import PluginCard from '../PluginCard';
import type { PluginInfo, PluginType, HealthStatus } from '@/types/plugin';

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
  config: {},
  health_status: 'healthy' as HealthStatus,
  last_health_check: '2025-01-01T00:00:00Z',
  error_message: null,
  requires_restart: false,
  ...overrides,
});

describe('PluginCard', () => {
  const mockOnEnable = vi.fn();
  const mockOnDisable = vi.fn();
  const mockOnConfigure = vi.fn();
  const mockOnRefreshHealth = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  const renderPluginCard = (plugin: PluginInfo = createMockPlugin()) =>
    render(
      <PluginCard
        plugin={plugin}
        onEnable={mockOnEnable}
        onDisable={mockOnDisable}
        onConfigure={mockOnConfigure}
        onRefreshHealth={mockOnRefreshHealth}
      />
    );

  describe('rendering', () => {
    it('renders plugin display name and version', () => {
      renderPluginCard();

      expect(screen.getByText('Test Plugin')).toBeInTheDocument();
      expect(screen.getByText('v1.0.0')).toBeInTheDocument();
    });

    it('renders plugin description', () => {
      renderPluginCard();

      expect(
        screen.getByText('A test plugin for testing purposes')
      ).toBeInTheDocument();
    });

    it('renders author information when provided', () => {
      renderPluginCard();

      expect(screen.getByText('by Test Author')).toBeInTheDocument();
    });

    it('renders homepage link when provided', () => {
      renderPluginCard();

      const homepageLink = screen.getByRole('link', { name: /homepage/i });
      expect(homepageLink).toHaveAttribute('href', 'https://example.com');
      expect(homepageLink).toHaveAttribute('target', '_blank');
    });

    it('does not render author or homepage when not provided', () => {
      const plugin = createMockPlugin({
        manifest: {
          ...createMockPlugin().manifest,
          author: null,
          homepage: null,
        },
      });
      renderPluginCard(plugin);

      expect(screen.queryByText(/^by /)).not.toBeInTheDocument();
      expect(screen.queryByRole('link', { name: /homepage/i })).not.toBeInTheDocument();
    });

    it('renders restart required badge when applicable', () => {
      const plugin = createMockPlugin({ requires_restart: true });
      renderPluginCard(plugin);

      expect(screen.getByText('Restart required')).toBeInTheDocument();
    });

    it('does not show restart badge when not required', () => {
      const plugin = createMockPlugin({ requires_restart: false });
      renderPluginCard(plugin);

      expect(screen.queryByText('Restart required')).not.toBeInTheDocument();
    });
  });

  describe('health status', () => {
    it('displays healthy status with green indicator', () => {
      const plugin = createMockPlugin({ health_status: 'healthy' });
      renderPluginCard(plugin);

      expect(screen.getByText('Healthy')).toBeInTheDocument();
    });

    it('displays unhealthy status with red indicator', () => {
      const plugin = createMockPlugin({ health_status: 'unhealthy' });
      renderPluginCard(plugin);

      expect(screen.getByText('Unhealthy')).toBeInTheDocument();
    });

    it('displays unknown status when health status is null', () => {
      const plugin = createMockPlugin({ health_status: null });
      renderPluginCard(plugin);

      expect(screen.getByText('Unknown')).toBeInTheDocument();
    });

    it('displays error message when present', () => {
      const plugin = createMockPlugin({
        health_status: 'unhealthy',
        error_message: 'Connection failed',
      });
      renderPluginCard(plugin);

      expect(screen.getByText('Connection failed')).toBeInTheDocument();
    });

    it('calls onRefreshHealth when health button is clicked', async () => {
      const user = userEvent.setup();
      renderPluginCard();

      await user.click(screen.getByText('Healthy'));

      expect(mockOnRefreshHealth).toHaveBeenCalledWith('test-plugin');
    });
  });

  describe('configure button', () => {
    it('renders Configure button', () => {
      renderPluginCard();

      expect(
        screen.getByRole('button', { name: /configure/i })
      ).toBeInTheDocument();
    });

    it('calls onConfigure with plugin when Configure is clicked', async () => {
      const user = userEvent.setup();
      const plugin = createMockPlugin();
      renderPluginCard(plugin);

      await user.click(screen.getByRole('button', { name: /configure/i }));

      expect(mockOnConfigure).toHaveBeenCalledWith(plugin);
    });
  });

  describe('enable/disable toggle', () => {
    it('renders toggle switch in enabled state', () => {
      const plugin = createMockPlugin({ enabled: true });
      renderPluginCard(plugin);

      const toggle = screen.getByRole('switch');
      expect(toggle).toHaveAttribute('aria-checked', 'true');
    });

    it('renders toggle switch in disabled state', () => {
      const plugin = createMockPlugin({ enabled: false });
      renderPluginCard(plugin);

      const toggle = screen.getByRole('switch');
      expect(toggle).toHaveAttribute('aria-checked', 'false');
    });

    it('calls onEnable when enabling a disabled plugin', async () => {
      const user = userEvent.setup();
      const plugin = createMockPlugin({ enabled: false });
      renderPluginCard(plugin);

      await user.click(screen.getByRole('switch'));

      expect(mockOnEnable).toHaveBeenCalledWith('test-plugin');
    });

    it('shows confirmation dialog when disabling an enabled plugin', async () => {
      const user = userEvent.setup();
      const plugin = createMockPlugin({ enabled: true });
      renderPluginCard(plugin);

      await user.click(screen.getByRole('switch'));

      expect(
        screen.getByText(/disabling this plugin requires a service restart/i)
      ).toBeInTheDocument();
      expect(
        screen.getByRole('button', { name: /yes, disable/i })
      ).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /cancel/i })).toBeInTheDocument();
    });

    it('calls onDisable when confirmation is accepted', async () => {
      const user = userEvent.setup();
      const plugin = createMockPlugin({ enabled: true });
      renderPluginCard(plugin);

      await user.click(screen.getByRole('switch'));
      await user.click(screen.getByRole('button', { name: /yes, disable/i }));

      expect(mockOnDisable).toHaveBeenCalledWith('test-plugin');
    });

    it('hides confirmation dialog when Cancel is clicked', async () => {
      const user = userEvent.setup();
      const plugin = createMockPlugin({ enabled: true });
      renderPluginCard(plugin);

      await user.click(screen.getByRole('switch'));
      await user.click(screen.getByRole('button', { name: /cancel/i }));

      expect(
        screen.queryByText(/disabling this plugin requires a service restart/i)
      ).not.toBeInTheDocument();
      expect(mockOnDisable).not.toHaveBeenCalled();
    });
  });

  describe('loading states', () => {
    it('disables toggle when isEnabling is true', () => {
      render(
        <PluginCard
          plugin={createMockPlugin({ enabled: false })}
          onEnable={mockOnEnable}
          onDisable={mockOnDisable}
          onConfigure={mockOnConfigure}
          onRefreshHealth={mockOnRefreshHealth}
          isEnabling={true}
        />
      );

      expect(screen.getByRole('switch')).toBeDisabled();
    });

    it('disables toggle when isDisabling is true', () => {
      render(
        <PluginCard
          plugin={createMockPlugin({ enabled: true })}
          onEnable={mockOnEnable}
          onDisable={mockOnDisable}
          onConfigure={mockOnConfigure}
          onRefreshHealth={mockOnRefreshHealth}
          isDisabling={true}
        />
      );

      expect(screen.getByRole('switch')).toBeDisabled();
    });

    it('shows Disabling... text when isDisabling during confirmation', async () => {
      const user = userEvent.setup();
      const { rerender } = render(
        <PluginCard
          plugin={createMockPlugin({ enabled: true })}
          onEnable={mockOnEnable}
          onDisable={mockOnDisable}
          onConfigure={mockOnConfigure}
          onRefreshHealth={mockOnRefreshHealth}
        />
      );

      await user.click(screen.getByRole('switch'));

      rerender(
        <PluginCard
          plugin={createMockPlugin({ enabled: true })}
          onEnable={mockOnEnable}
          onDisable={mockOnDisable}
          onConfigure={mockOnConfigure}
          onRefreshHealth={mockOnRefreshHealth}
          isDisabling={true}
        />
      );

      expect(screen.getByText('Disabling...')).toBeInTheDocument();
    });
  });
});
