import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import InterfaceSettings from '../InterfaceSettings';
import {
  usePreferences,
  useUpdatePreferences,
  useResetInterfacePreferences,
} from '../../../hooks/usePreferences';

// Mock the hooks
vi.mock('../../../hooks/usePreferences', () => ({
  usePreferences: vi.fn(),
  useUpdatePreferences: vi.fn(),
  useResetInterfacePreferences: vi.fn(),
}));

const mockPreferences = {
  interface: {
    data_refresh_interval_ms: 30000,
    visualization_sample_limit: 200000,
    animation_enabled: true,
  },
};

describe('InterfaceSettings', () => {
  const mockMutateAsync = vi.fn();
  const mockResetMutateAsync = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();

    vi.mocked(usePreferences).mockReturnValue({
      data: mockPreferences,
      isLoading: false,
      error: null,
    } as ReturnType<typeof usePreferences>);

    vi.mocked(useUpdatePreferences).mockReturnValue({
      mutateAsync: mockMutateAsync,
      isPending: false,
    } as unknown as ReturnType<typeof useUpdatePreferences>);

    vi.mocked(useResetInterfacePreferences).mockReturnValue({
      mutateAsync: mockResetMutateAsync,
      isPending: false,
    } as unknown as ReturnType<typeof useResetInterfacePreferences>);
  });

  it('renders loading state', () => {
    vi.mocked(usePreferences).mockReturnValue({
      data: undefined,
      isLoading: true,
      error: null,
    } as ReturnType<typeof usePreferences>);

    render(<InterfaceSettings />);

    expect(screen.getByText('Loading interface preferences...')).toBeInTheDocument();
  });

  it('renders error state', () => {
    vi.mocked(usePreferences).mockReturnValue({
      data: undefined,
      isLoading: false,
      error: new Error('Failed to load'),
    } as ReturnType<typeof usePreferences>);

    render(<InterfaceSettings />);

    expect(screen.getByText('Error loading preferences')).toBeInTheDocument();
    expect(screen.getByText('Failed to load')).toBeInTheDocument();
  });

  it('renders interface preferences form', () => {
    render(<InterfaceSettings />);

    expect(screen.getByText('Interface Preferences')).toBeInTheDocument();
    expect(screen.getByText('Data Refresh Interval')).toBeInTheDocument();
    expect(screen.getByText('Visualization Sample Limit')).toBeInTheDocument();
    expect(screen.getByText('Enable Animations')).toBeInTheDocument();
  });

  it('displays current values from preferences', () => {
    render(<InterfaceSettings />);

    // Check displayed value for refresh interval (30000ms = 30s)
    expect(screen.getByText('30s')).toBeInTheDocument();
    // Check displayed value for sample limit (200000 = 200K)
    expect(screen.getByText('200K')).toBeInTheDocument();
  });

  it('calls save when Save Preferences button is clicked', async () => {
    mockMutateAsync.mockResolvedValue({});
    render(<InterfaceSettings />);

    const saveButton = screen.getByText('Save Preferences');
    fireEvent.click(saveButton);

    await waitFor(() => {
      expect(mockMutateAsync).toHaveBeenCalledWith({
        interface: {
          data_refresh_interval_ms: 30000,
          visualization_sample_limit: 200000,
          animation_enabled: true,
        },
      });
    });
  });

  it('calls reset when Reset to Defaults button is clicked', async () => {
    mockResetMutateAsync.mockResolvedValue({});
    render(<InterfaceSettings />);

    const resetButton = screen.getByText('Reset to Defaults');
    fireEvent.click(resetButton);

    await waitFor(() => {
      expect(mockResetMutateAsync).toHaveBeenCalled();
    });
  });

  it('shows saving state on save button', () => {
    vi.mocked(useUpdatePreferences).mockReturnValue({
      mutateAsync: mockMutateAsync,
      isPending: true,
    } as unknown as ReturnType<typeof useUpdatePreferences>);

    render(<InterfaceSettings />);

    expect(screen.getByText('Saving...')).toBeInTheDocument();
  });

  it('shows resetting state on reset button', () => {
    vi.mocked(useResetInterfacePreferences).mockReturnValue({
      mutateAsync: mockResetMutateAsync,
      isPending: true,
    } as unknown as ReturnType<typeof useResetInterfacePreferences>);

    render(<InterfaceSettings />);

    expect(screen.getByText('Resetting...')).toBeInTheDocument();
  });

  it('updates form state when data refresh interval slider changes', () => {
    render(<InterfaceSettings />);

    const sliders = screen.getAllByRole('slider');
    const refreshIntervalSlider = sliders[0];

    // Change to 45 seconds
    fireEvent.change(refreshIntervalSlider, { target: { value: '45' } });

    expect(screen.getByText('45s')).toBeInTheDocument();
  });

  it('updates form state when visualization sample limit slider changes', () => {
    render(<InterfaceSettings />);

    const sliders = screen.getAllByRole('slider');
    const sampleLimitSlider = sliders[1];

    // Change to 300000
    fireEvent.change(sampleLimitSlider, { target: { value: '300000' } });

    expect(screen.getByText('300K')).toBeInTheDocument();
  });

  it('updates form state when animation checkbox is toggled', async () => {
    mockMutateAsync.mockResolvedValue({});
    render(<InterfaceSettings />);

    const checkbox = screen.getByRole('checkbox');
    expect(checkbox).toBeChecked();

    fireEvent.click(checkbox);

    // Now save and verify the new value is sent
    const saveButton = screen.getByText('Save Preferences');
    fireEvent.click(saveButton);

    await waitFor(() => {
      expect(mockMutateAsync).toHaveBeenCalledWith({
        interface: {
          data_refresh_interval_ms: 30000,
          visualization_sample_limit: 200000,
          animation_enabled: false,
        },
      });
    });
  });

  it('disables save button when mutation is pending', () => {
    vi.mocked(useUpdatePreferences).mockReturnValue({
      mutateAsync: mockMutateAsync,
      isPending: true,
    } as unknown as ReturnType<typeof useUpdatePreferences>);

    render(<InterfaceSettings />);

    const saveButton = screen.getByText('Saving...');
    expect(saveButton).toBeDisabled();
  });

  it('disables reset button when reset mutation is pending', () => {
    vi.mocked(useResetInterfacePreferences).mockReturnValue({
      mutateAsync: mockResetMutateAsync,
      isPending: true,
    } as unknown as ReturnType<typeof useResetInterfacePreferences>);

    render(<InterfaceSettings />);

    const resetButton = screen.getByText('Resetting...');
    expect(resetButton).toBeDisabled();
  });

  it('renders info box with helpful message', () => {
    render(<InterfaceSettings />);

    expect(
      screen.getByText(/These settings control how the UI behaves/)
    ).toBeInTheDocument();
  });
});
