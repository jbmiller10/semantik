import { describe, it, expect, beforeEach, vi } from 'vitest';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { render as renderWithProviders } from '../../tests/utils/test-utils';
import SettingsPage from '../SettingsPage';
import { useAuthStore } from '../../stores/authStore';

const mockNavigate = vi.fn();
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom');
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  };
});

// Mock authStore
vi.mock('../../stores/authStore', () => ({
  useAuthStore: vi.fn(),
}));

// Mock alert
global.alert = vi.fn();

// Mock child components to isolate SettingsPage tests
vi.mock('../../components/settings/PreferencesTab', () => ({
  default: () => <div data-testid="preferences-tab">Preferences Content</div>,
}));

vi.mock('../../components/settings/AdminTab', () => ({
  default: () => <div data-testid="admin-tab">Admin Content</div>,
}));

vi.mock('../../components/settings/SystemTab', () => ({
  default: () => <div data-testid="system-tab">System Content</div>,
}));

vi.mock('../../components/settings/PluginsSettings', () => ({
  default: () => <div data-testid="plugins-settings">Plugins Content</div>,
}));

vi.mock('../../components/settings/MCPProfilesSettings', () => ({
  default: () => <div data-testid="mcp-settings">MCP Content</div>,
}));

const mockSuperuser = {
  id: 1,
  username: 'admin',
  email: 'admin@test.com',
  is_active: true,
  is_superuser: true,
  created_at: new Date().toISOString(),
};

const mockRegularUser = {
  id: 2,
  username: 'user',
  email: 'user@test.com',
  is_active: true,
  is_superuser: false,
  created_at: new Date().toISOString(),
};

describe('SettingsPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockNavigate.mockClear();
  });

  describe('header and navigation', () => {
    beforeEach(() => {
      vi.mocked(useAuthStore).mockImplementation((selector) => {
        const state = { user: mockRegularUser };
        return selector(state as ReturnType<typeof useAuthStore.getState>);
      });
    });

    it('renders settings page with header and back button', () => {
      renderWithProviders(<SettingsPage />);

      expect(screen.getByText('Settings')).toBeInTheDocument();
      expect(
        screen.getByText('Manage your preferences, plugins, and system settings')
      ).toBeInTheDocument();
      expect(
        screen.getByRole('button', { name: /back to home/i })
      ).toBeInTheDocument();
    });

    it('navigates back to home when back button is clicked', async () => {
      const user = userEvent.setup();
      renderWithProviders(<SettingsPage />);

      await user.click(screen.getByRole('button', { name: /back to home/i }));

      expect(mockNavigate).toHaveBeenCalledWith('/');
    });
  });

  describe('tab navigation for regular users', () => {
    beforeEach(() => {
      vi.mocked(useAuthStore).mockImplementation((selector) => {
        const state = { user: mockRegularUser };
        return selector(state as ReturnType<typeof useAuthStore.getState>);
      });
    });

    it('shows 4 tabs for regular users (no Admin tab)', () => {
      renderWithProviders(<SettingsPage />);

      expect(screen.getByText('Preferences')).toBeInTheDocument();
      expect(screen.queryByText('Admin')).not.toBeInTheDocument();
      expect(screen.getByText('System')).toBeInTheDocument();
      expect(screen.getByText('Plugins')).toBeInTheDocument();
      expect(screen.getByText('MCP Profiles')).toBeInTheDocument();
    });

    it('defaults to Preferences tab', () => {
      renderWithProviders(<SettingsPage />);

      expect(screen.getByTestId('preferences-tab')).toBeInTheDocument();
    });

    it('can switch between tabs', async () => {
      const user = userEvent.setup();
      renderWithProviders(<SettingsPage />);

      // Switch to System tab
      await user.click(screen.getByText('System'));
      expect(screen.getByTestId('system-tab')).toBeInTheDocument();
      expect(screen.queryByTestId('preferences-tab')).not.toBeInTheDocument();

      // Switch to Plugins tab
      await user.click(screen.getByText('Plugins'));
      expect(screen.getByTestId('plugins-settings')).toBeInTheDocument();

      // Switch to MCP tab
      await user.click(screen.getByText('MCP Profiles'));
      expect(screen.getByTestId('mcp-settings')).toBeInTheDocument();
    });
  });

  describe('tab navigation for superusers', () => {
    beforeEach(() => {
      vi.mocked(useAuthStore).mockImplementation((selector) => {
        const state = { user: mockSuperuser };
        return selector(state as ReturnType<typeof useAuthStore.getState>);
      });
    });

    it('shows 5 tabs for superusers (includes Admin tab)', () => {
      renderWithProviders(<SettingsPage />);

      expect(screen.getByText('Preferences')).toBeInTheDocument();
      expect(screen.getByText('Admin')).toBeInTheDocument();
      expect(screen.getByText('System')).toBeInTheDocument();
      expect(screen.getByText('Plugins')).toBeInTheDocument();
      expect(screen.getByText('MCP Profiles')).toBeInTheDocument();
    });

    it('can access Admin tab', async () => {
      const user = userEvent.setup();
      renderWithProviders(<SettingsPage />);

      await user.click(screen.getByText('Admin'));

      expect(screen.getByTestId('admin-tab')).toBeInTheDocument();
    });
  });

  describe('access control edge cases', () => {
    it('handles null user gracefully', () => {
      vi.mocked(useAuthStore).mockImplementation((selector) => {
        const state = { user: null };
        return selector(state as ReturnType<typeof useAuthStore.getState>);
      });

      renderWithProviders(<SettingsPage />);

      // Should show 4 tabs (no Admin)
      expect(screen.getByText('Preferences')).toBeInTheDocument();
      expect(screen.queryByText('Admin')).not.toBeInTheDocument();
    });

    it('handles undefined user gracefully', () => {
      vi.mocked(useAuthStore).mockImplementation((selector) => {
        const state = { user: undefined };
        return selector(state as ReturnType<typeof useAuthStore.getState>);
      });

      renderWithProviders(<SettingsPage />);

      // Should show 4 tabs (no Admin)
      expect(screen.getByText('Preferences')).toBeInTheDocument();
      expect(screen.queryByText('Admin')).not.toBeInTheDocument();
    });
  });

  describe('tab styling', () => {
    beforeEach(() => {
      vi.mocked(useAuthStore).mockImplementation((selector) => {
        const state = { user: mockRegularUser };
        return selector(state as ReturnType<typeof useAuthStore.getState>);
      });
    });

    it('applies active styling to selected tab', async () => {
      const user = userEvent.setup();
      renderWithProviders(<SettingsPage />);

      // Preferences should be active by default
      const preferencesButton = screen.getByText('Preferences').closest('button');
      expect(preferencesButton).toHaveClass('border-[var(--accent-primary)]');

      // Switch to System
      await user.click(screen.getByText('System'));
      const systemButton = screen.getByText('System').closest('button');
      expect(systemButton).toHaveClass('border-[var(--accent-primary)]');
      expect(preferencesButton).not.toHaveClass('border-[var(--accent-primary)]');
    });
  });
});
