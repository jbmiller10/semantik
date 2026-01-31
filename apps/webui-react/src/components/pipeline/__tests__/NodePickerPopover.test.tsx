import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import '@testing-library/jest-dom';

import { NodePickerPopover } from '../NodePickerPopover';

// Mock the useAvailablePlugins hook
vi.mock('../../../hooks/useAvailablePlugins', () => ({
  useAvailablePlugins: vi.fn(),
}));

import { useAvailablePlugins } from '../../../hooks/useAvailablePlugins';

const mockUseAvailablePlugins = useAvailablePlugins as ReturnType<typeof vi.fn>;

describe('NodePickerPopover', () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });

  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );

  const mockPlugins = [
    { id: 'recursive', name: 'Recursive', description: 'Recursive text splitter' },
    { id: 'semantic', name: 'Semantic', description: 'Semantic chunking' },
    { id: 'fixed', name: 'Fixed Size', description: 'Fixed token count' },
  ];

  beforeEach(() => {
    vi.clearAllMocks();
    mockUseAvailablePlugins.mockReturnValue({
      plugins: mockPlugins,
      isLoading: false,
      error: null,
    });
  });

  it('renders plugin list', () => {
    const onSelect = vi.fn();
    const onCancel = vi.fn();

    render(
      <NodePickerPopover
        tier="chunker"
        position={{ x: 100, y: 200 }}
        onSelect={onSelect}
        onCancel={onCancel}
      />,
      { wrapper }
    );

    expect(screen.getByText('Recursive')).toBeInTheDocument();
    expect(screen.getByText('Semantic')).toBeInTheDocument();
    expect(screen.getByText('Fixed Size')).toBeInTheDocument();
  });

  it('shows tier label in header', () => {
    const onSelect = vi.fn();
    const onCancel = vi.fn();

    render(
      <NodePickerPopover
        tier="chunker"
        position={{ x: 100, y: 200 }}
        onSelect={onSelect}
        onCancel={onCancel}
      />,
      { wrapper }
    );

    expect(screen.getByText(/select chunker/i)).toBeInTheDocument();
  });

  it('shows plugin descriptions', () => {
    const onSelect = vi.fn();
    const onCancel = vi.fn();

    render(
      <NodePickerPopover
        tier="chunker"
        position={{ x: 100, y: 200 }}
        onSelect={onSelect}
        onCancel={onCancel}
      />,
      { wrapper }
    );

    expect(screen.getByText('Recursive text splitter')).toBeInTheDocument();
    expect(screen.getByText('Semantic chunking')).toBeInTheDocument();
  });

  it('calls onSelect with plugin ID when plugin is clicked', async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();
    const onCancel = vi.fn();

    render(
      <NodePickerPopover
        tier="chunker"
        position={{ x: 100, y: 200 }}
        onSelect={onSelect}
        onCancel={onCancel}
      />,
      { wrapper }
    );

    await user.click(screen.getByText('Semantic'));

    expect(onSelect).toHaveBeenCalledWith('semantic');
  });

  it('calls onCancel when close button is clicked', async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();
    const onCancel = vi.fn();

    render(
      <NodePickerPopover
        tier="chunker"
        position={{ x: 100, y: 200 }}
        onSelect={onSelect}
        onCancel={onCancel}
      />,
      { wrapper }
    );

    const closeButton = screen.getByRole('button', { name: /cancel/i });
    await user.click(closeButton);

    expect(onCancel).toHaveBeenCalled();
  });

  it('calls onCancel when Escape key is pressed', async () => {
    const onSelect = vi.fn();
    const onCancel = vi.fn();

    render(
      <NodePickerPopover
        tier="chunker"
        position={{ x: 100, y: 200 }}
        onSelect={onSelect}
        onCancel={onCancel}
      />,
      { wrapper }
    );

    fireEvent.keyDown(document, { key: 'Escape' });

    expect(onCancel).toHaveBeenCalled();
  });

  it('calls onCancel when clicking outside', async () => {
    const onSelect = vi.fn();
    const onCancel = vi.fn();

    render(
      <div>
        <div data-testid="outside">Outside</div>
        <NodePickerPopover
          tier="chunker"
          position={{ x: 100, y: 200 }}
          onSelect={onSelect}
          onCancel={onCancel}
        />
      </div>,
      { wrapper }
    );

    // Wait for the click outside handler to be registered (the component uses setTimeout)
    await new Promise(resolve => setTimeout(resolve, 10));

    // Click outside the popover
    fireEvent.mouseDown(screen.getByTestId('outside'));

    expect(onCancel).toHaveBeenCalled();
  });

  it('shows loading state', () => {
    mockUseAvailablePlugins.mockReturnValue({
      plugins: [],
      isLoading: true,
      error: null,
    });

    const onSelect = vi.fn();
    const onCancel = vi.fn();

    render(
      <NodePickerPopover
        tier="chunker"
        position={{ x: 100, y: 200 }}
        onSelect={onSelect}
        onCancel={onCancel}
      />,
      { wrapper }
    );

    // Should show loading spinner
    expect(document.querySelector('.animate-spin')).toBeInTheDocument();
  });

  it('shows error state', () => {
    mockUseAvailablePlugins.mockReturnValue({
      plugins: [],
      isLoading: false,
      error: new Error('Failed to load'),
    });

    const onSelect = vi.fn();
    const onCancel = vi.fn();

    render(
      <NodePickerPopover
        tier="chunker"
        position={{ x: 100, y: 200 }}
        onSelect={onSelect}
        onCancel={onCancel}
      />,
      { wrapper }
    );

    expect(screen.getByText(/failed to load plugins/i)).toBeInTheDocument();
  });

  it('shows empty state when no plugins available', () => {
    mockUseAvailablePlugins.mockReturnValue({
      plugins: [],
      isLoading: false,
      error: null,
    });

    const onSelect = vi.fn();
    const onCancel = vi.fn();

    render(
      <NodePickerPopover
        tier="chunker"
        position={{ x: 100, y: 200 }}
        onSelect={onSelect}
        onCancel={onCancel}
      />,
      { wrapper }
    );

    expect(screen.getByText(/no plugins available/i)).toBeInTheDocument();
  });

  it('auto-selects when only one plugin is available', async () => {
    mockUseAvailablePlugins.mockReturnValue({
      plugins: [{ id: 'only-one', name: 'Only One', description: 'The only plugin' }],
      isLoading: false,
      error: null,
    });

    const onSelect = vi.fn();
    const onCancel = vi.fn();

    render(
      <NodePickerPopover
        tier="chunker"
        position={{ x: 100, y: 200 }}
        onSelect={onSelect}
        onCancel={onCancel}
      />,
      { wrapper }
    );

    await waitFor(() => {
      expect(onSelect).toHaveBeenCalledWith('only-one');
    });
  });

  it('auto-select fires only once even when effect re-runs due to React Query refetch', async () => {
    const singlePlugin = [{ id: 'only-one', name: 'Only One', description: 'The only plugin' }];

    mockUseAvailablePlugins.mockReturnValue({
      plugins: singlePlugin,
      isLoading: false,
      error: null,
    });

    const onSelect = vi.fn();
    const onCancel = vi.fn();

    const { rerender } = render(
      <NodePickerPopover
        tier="chunker"
        position={{ x: 100, y: 200 }}
        onSelect={onSelect}
        onCancel={onCancel}
      />,
      { wrapper }
    );

    await waitFor(() => {
      expect(onSelect).toHaveBeenCalledWith('only-one');
    });
    expect(onSelect).toHaveBeenCalledTimes(1);

    // Simulate React Query refetch returning a new array reference with same content
    mockUseAvailablePlugins.mockReturnValue({
      plugins: [...singlePlugin], // New array reference
      isLoading: false,
      error: null,
    });

    rerender(
      <NodePickerPopover
        tier="chunker"
        position={{ x: 100, y: 200 }}
        onSelect={onSelect}
        onCancel={onCancel}
      />
    );

    // Give time for any potential extra effect runs
    await new Promise(resolve => setTimeout(resolve, 50));

    // onSelect should still have been called only once
    expect(onSelect).toHaveBeenCalledTimes(1);
  });

  it('has proper accessibility attributes', () => {
    const onSelect = vi.fn();
    const onCancel = vi.fn();

    render(
      <NodePickerPopover
        tier="chunker"
        position={{ x: 100, y: 200 }}
        onSelect={onSelect}
        onCancel={onCancel}
      />,
      { wrapper }
    );

    expect(screen.getByRole('dialog')).toBeInTheDocument();
    expect(screen.getByRole('dialog')).toHaveAttribute('aria-label', 'Select chunker');
  });
});
