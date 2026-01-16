import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@/tests/utils/test-utils';
import userEvent from '@testing-library/user-event';
import DangerZoneSettings from '../DangerZoneSettings';
import * as reactRouterDom from 'react-router-dom';
import * as reactQuery from '@tanstack/react-query';
import * as settingsApiModule from '@/services/api/v2';
import * as uiStoreModule from '@/stores/uiStore';

// Mock the modules
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom');
  return {
    ...actual,
    useNavigate: vi.fn(),
  };
});

vi.mock('@tanstack/react-query', async () => {
  const actual = await vi.importActual('@tanstack/react-query');
  return {
    ...actual,
    useQueryClient: vi.fn(),
  };
});

vi.mock('@/services/api/v2', () => ({
  settingsApi: {
    resetDatabase: vi.fn(),
  },
}));

vi.mock('@/stores/uiStore', async () => {
  const actual = await vi.importActual('@/stores/uiStore');
  return {
    ...actual,
    useUIStore: vi.fn(),
  };
});

describe('DangerZoneSettings', () => {
  const mockNavigate = vi.fn();
  const mockClear = vi.fn();
  const mockAddToast = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();

    vi.mocked(reactRouterDom.useNavigate).mockReturnValue(mockNavigate);
    vi.mocked(reactQuery.useQueryClient).mockReturnValue({
      clear: mockClear,
    } as unknown as ReturnType<typeof reactQuery.useQueryClient>);
    vi.mocked(uiStoreModule.useUIStore).mockImplementation((selector) => {
      const state = { addToast: mockAddToast };
      return selector ? selector(state as never) : state;
    });
  });

  describe('rendering', () => {
    it('renders the warning box', () => {
      render(<DangerZoneSettings />);

      expect(screen.getByText(/Operations in this section are destructive and cannot be undone/)).toBeInTheDocument();
    });

    it('renders the Reset Database section', () => {
      render(<DangerZoneSettings />);

      expect(screen.getByRole('heading', { name: 'Reset Database' })).toBeInTheDocument();
      expect(screen.getByText(/This will permanently delete all collections/)).toBeInTheDocument();
    });

    it('renders the Reset Database button', () => {
      render(<DangerZoneSettings />);

      expect(screen.getByRole('button', { name: /Reset Database/i })).toBeInTheDocument();
    });
  });

  describe('confirmation dialog', () => {
    it('opens confirmation dialog when reset button is clicked', async () => {
      const user = userEvent.setup();
      render(<DangerZoneSettings />);

      const resetButton = screen.getByRole('button', { name: /Reset Database/i });
      await user.click(resetButton);

      expect(screen.getByText('Confirm Database Reset')).toBeInTheDocument();
      expect(screen.getByText('Type "RESET" to confirm:')).toBeInTheDocument();
    });

    it('shows cancel button in dialog', async () => {
      const user = userEvent.setup();
      render(<DangerZoneSettings />);

      const resetButton = screen.getByRole('button', { name: /Reset Database/i });
      await user.click(resetButton);

      expect(screen.getByRole('button', { name: 'Cancel' })).toBeInTheDocument();
    });

    it('closes dialog when cancel is clicked', async () => {
      const user = userEvent.setup();
      render(<DangerZoneSettings />);

      const resetButton = screen.getByRole('button', { name: /Reset Database/i });
      await user.click(resetButton);

      const cancelButton = screen.getByRole('button', { name: 'Cancel' });
      await user.click(cancelButton);

      expect(screen.queryByText('Confirm Database Reset')).not.toBeInTheDocument();
    });

    it('disables confirm button when text does not match', async () => {
      const user = userEvent.setup();
      render(<DangerZoneSettings />);

      const resetButton = screen.getByRole('button', { name: /Reset Database/i });
      await user.click(resetButton);

      // Find the confirm button inside the dialog
      const dialogButtons = screen.getAllByRole('button', { name: /Reset Database/i });
      const confirmButton = dialogButtons[dialogButtons.length - 1]; // The one in the dialog

      expect(confirmButton).toBeDisabled();
    });

    it('enables confirm button when "RESET" is typed', async () => {
      const user = userEvent.setup();
      render(<DangerZoneSettings />);

      const openDialogButton = screen.getByRole('button', { name: /Reset Database/i });
      await user.click(openDialogButton);

      const input = screen.getByPlaceholderText('Type RESET');
      await user.type(input, 'RESET');

      // Find the confirm button inside the dialog
      const dialogButtons = screen.getAllByRole('button', { name: /Reset Database/i });
      const confirmButton = dialogButtons[dialogButtons.length - 1];

      expect(confirmButton).not.toBeDisabled();
    });
  });

  describe('reset functionality', () => {
    it('calls resetDatabase API when confirmed', async () => {
      const user = userEvent.setup();
      vi.mocked(settingsApiModule.settingsApi.resetDatabase).mockResolvedValueOnce({
        data: { message: 'Database reset successfully' },
        status: 200,
        statusText: 'OK',
        headers: {},
        config: {} as never,
      });

      render(<DangerZoneSettings />);

      const openDialogButton = screen.getByRole('button', { name: /Reset Database/i });
      await user.click(openDialogButton);

      const input = screen.getByPlaceholderText('Type RESET');
      await user.type(input, 'RESET');

      const dialogButtons = screen.getAllByRole('button', { name: /Reset Database/i });
      const confirmButton = dialogButtons[dialogButtons.length - 1];
      await user.click(confirmButton);

      await waitFor(() => {
        expect(settingsApiModule.settingsApi.resetDatabase).toHaveBeenCalled();
      });
    });

    it('clears query client on success', async () => {
      const user = userEvent.setup();
      vi.mocked(settingsApiModule.settingsApi.resetDatabase).mockResolvedValueOnce({
        data: { message: 'Database reset successfully' },
        status: 200,
        statusText: 'OK',
        headers: {},
        config: {} as never,
      });

      render(<DangerZoneSettings />);

      const openDialogButton = screen.getByRole('button', { name: /Reset Database/i });
      await user.click(openDialogButton);

      const input = screen.getByPlaceholderText('Type RESET');
      await user.type(input, 'RESET');

      const dialogButtons = screen.getAllByRole('button', { name: /Reset Database/i });
      const confirmButton = dialogButtons[dialogButtons.length - 1];
      await user.click(confirmButton);

      await waitFor(() => {
        expect(mockClear).toHaveBeenCalled();
      });
    });

    it('shows success toast and navigates to home on success', async () => {
      const user = userEvent.setup();
      vi.mocked(settingsApiModule.settingsApi.resetDatabase).mockResolvedValueOnce({
        data: { message: 'Database reset successfully' },
        status: 200,
        statusText: 'OK',
        headers: {},
        config: {} as never,
      });

      render(<DangerZoneSettings />);

      const openDialogButton = screen.getByRole('button', { name: /Reset Database/i });
      await user.click(openDialogButton);

      const input = screen.getByPlaceholderText('Type RESET');
      await user.type(input, 'RESET');

      const dialogButtons = screen.getAllByRole('button', { name: /Reset Database/i });
      const confirmButton = dialogButtons[dialogButtons.length - 1];
      await user.click(confirmButton);

      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith({ type: 'success', message: 'Database reset successfully' });
        expect(mockNavigate).toHaveBeenCalledWith('/');
      });
    });

    it('shows error toast on failure', async () => {
      const user = userEvent.setup();
      vi.mocked(settingsApiModule.settingsApi.resetDatabase).mockRejectedValueOnce(
        new Error('Reset failed')
      );

      render(<DangerZoneSettings />);

      const openDialogButton = screen.getByRole('button', { name: /Reset Database/i });
      await user.click(openDialogButton);

      const input = screen.getByPlaceholderText('Type RESET');
      await user.type(input, 'RESET');

      const dialogButtons = screen.getAllByRole('button', { name: /Reset Database/i });
      const confirmButton = dialogButtons[dialogButtons.length - 1];
      await user.click(confirmButton);

      await waitFor(() => {
        expect(mockAddToast).toHaveBeenCalledWith({
          type: 'error',
          message: expect.stringContaining('Failed to reset database')
        });
      });
    });
  });
});
