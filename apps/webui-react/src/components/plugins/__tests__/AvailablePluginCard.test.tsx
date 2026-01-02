import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@/tests/utils/test-utils';
import userEvent from '@testing-library/user-event';
import AvailablePluginCard from '../AvailablePluginCard';
import type { AvailablePlugin, PluginType } from '@/types/plugin';

// Helper to create mock available plugin
const createMockAvailablePlugin = (
  overrides: Partial<AvailablePlugin> = {}
): AvailablePlugin => ({
  id: 'test-plugin',
  type: 'embedding' as PluginType,
  name: 'Test Plugin',
  description: 'A test plugin for testing purposes',
  author: 'Test Author',
  repository: 'https://github.com/test/test-plugin',
  pypi: 'semantik-plugin-test',
  verified: true,
  min_semantik_version: '2.0.0',
  tags: ['api', 'cloud'],
  is_compatible: true,
  compatibility_message: null,
  is_installed: false,
  install_command: 'pip install semantik-plugin-test',
  ...overrides,
});

describe('AvailablePluginCard', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('rendering', () => {
    it('renders plugin name and description', () => {
      render(<AvailablePluginCard plugin={createMockAvailablePlugin()} />);

      expect(screen.getByText('Test Plugin')).toBeInTheDocument();
      expect(
        screen.getByText('A test plugin for testing purposes')
      ).toBeInTheDocument();
    });

    it('renders author information', () => {
      render(<AvailablePluginCard plugin={createMockAvailablePlugin()} />);

      expect(screen.getByText('by Test Author')).toBeInTheDocument();
    });

    it('renders GitHub link', () => {
      render(<AvailablePluginCard plugin={createMockAvailablePlugin()} />);

      const githubLink = screen.getByRole('link', { name: /github/i });
      expect(githubLink).toHaveAttribute(
        'href',
        'https://github.com/test/test-plugin'
      );
      expect(githubLink).toHaveAttribute('target', '_blank');
    });

    it('renders tags when present', () => {
      render(<AvailablePluginCard plugin={createMockAvailablePlugin()} />);

      expect(screen.getByText('api')).toBeInTheDocument();
      expect(screen.getByText('cloud')).toBeInTheDocument();
    });

    it('does not render tags section when empty', () => {
      const plugin = createMockAvailablePlugin({ tags: [] });
      render(<AvailablePluginCard plugin={plugin} />);

      expect(screen.queryByText('api')).not.toBeInTheDocument();
    });
  });

  describe('verified badge', () => {
    it('shows Verified badge for verified plugins', () => {
      const plugin = createMockAvailablePlugin({ verified: true });
      render(<AvailablePluginCard plugin={plugin} />);

      expect(screen.getByText('Verified')).toBeInTheDocument();
    });

    it('shows Unverified badge for unverified plugins', () => {
      const plugin = createMockAvailablePlugin({ verified: false });
      render(<AvailablePluginCard plugin={plugin} />);

      expect(screen.getByText('Unverified')).toBeInTheDocument();
    });
  });

  describe('installed status', () => {
    it('shows Installed badge when plugin is installed', () => {
      const plugin = createMockAvailablePlugin({ is_installed: true });
      render(<AvailablePluginCard plugin={plugin} />);

      expect(screen.getByText('Installed')).toBeInTheDocument();
    });

    it('does not show Installed badge when not installed', () => {
      const plugin = createMockAvailablePlugin({ is_installed: false });
      render(<AvailablePluginCard plugin={plugin} />);

      expect(screen.queryByText('Installed')).not.toBeInTheDocument();
    });

    it('hides install command when plugin is installed', () => {
      const plugin = createMockAvailablePlugin({ is_installed: true });
      render(<AvailablePluginCard plugin={plugin} />);

      expect(
        screen.queryByText('pip install semantik-plugin-test')
      ).not.toBeInTheDocument();
    });
  });

  describe('compatibility', () => {
    it('shows Incompatible badge when not compatible', () => {
      const plugin = createMockAvailablePlugin({
        is_compatible: false,
        compatibility_message: 'Requires Semantik >= 3.0.0',
      });
      render(<AvailablePluginCard plugin={plugin} />);

      expect(screen.getByText('Incompatible')).toBeInTheDocument();
    });

    it('shows compatibility message when incompatible', () => {
      const plugin = createMockAvailablePlugin({
        is_compatible: false,
        compatibility_message: 'Requires Semantik >= 3.0.0',
      });
      render(<AvailablePluginCard plugin={plugin} />);

      expect(
        screen.getByText('Requires Semantik >= 3.0.0')
      ).toBeInTheDocument();
    });

    it('hides install command when incompatible', () => {
      const plugin = createMockAvailablePlugin({ is_compatible: false });
      render(<AvailablePluginCard plugin={plugin} />);

      expect(
        screen.queryByText('pip install semantik-plugin-test')
      ).not.toBeInTheDocument();
    });

    it('does not show Incompatible badge when compatible', () => {
      const plugin = createMockAvailablePlugin({ is_compatible: true });
      render(<AvailablePluginCard plugin={plugin} />);

      expect(screen.queryByText('Incompatible')).not.toBeInTheDocument();
    });
  });

  describe('install command', () => {
    it('renders install command when compatible and not installed', () => {
      const plugin = createMockAvailablePlugin({
        is_compatible: true,
        is_installed: false,
      });
      render(<AvailablePluginCard plugin={plugin} />);

      expect(
        screen.getByText('pip install semantik-plugin-test')
      ).toBeInTheDocument();
    });

    it('renders copy button for install command', () => {
      const plugin = createMockAvailablePlugin();
      render(<AvailablePluginCard plugin={plugin} />);

      expect(screen.getByTitle('Copy to clipboard')).toBeInTheDocument();
    });

    it('copy button can be clicked', async () => {
      const user = userEvent.setup();
      const plugin = createMockAvailablePlugin();
      render(<AvailablePluginCard plugin={plugin} />);

      const copyButton = screen.getByTitle('Copy to clipboard');
      // Click should not throw
      await user.click(copyButton);

      // Just verify the button is interactive
      expect(copyButton).toBeInTheDocument();
    });
  });

  describe('styling', () => {
    it('applies installed styling when plugin is installed', () => {
      const plugin = createMockAvailablePlugin({ is_installed: true });
      const { container } = render(<AvailablePluginCard plugin={plugin} />);

      const card = container.firstChild;
      expect(card).toHaveClass('border-green-200');
      expect(card).toHaveClass('bg-green-50');
    });

    it('applies opacity styling when incompatible', () => {
      const plugin = createMockAvailablePlugin({ is_compatible: false });
      const { container } = render(<AvailablePluginCard plugin={plugin} />);

      const card = container.firstChild;
      expect(card).toHaveClass('opacity-60');
    });
  });
});
