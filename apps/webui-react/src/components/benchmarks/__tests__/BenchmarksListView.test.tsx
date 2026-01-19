import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('../../../hooks/useBenchmarks', () => ({
  useBenchmarks: vi.fn(),
  useStartBenchmark: vi.fn(),
  useCancelBenchmark: vi.fn(),
  useDeleteBenchmark: vi.fn(),
}));

vi.mock('../CreateBenchmarkModal', () => ({
  CreateBenchmarkModal: () => <div>create-modal</div>,
}));

import { BenchmarksListView } from '../BenchmarksListView';
import {
  useBenchmarks,
  useStartBenchmark,
  useCancelBenchmark,
  useDeleteBenchmark,
} from '../../../hooks/useBenchmarks';

describe('BenchmarksListView', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.spyOn(window, 'confirm').mockReturnValue(true);

    vi.mocked(useStartBenchmark).mockReturnValue({
      mutate: vi.fn(),
      isPending: false,
      variables: undefined,
    } as ReturnType<typeof useStartBenchmark>);

    vi.mocked(useCancelBenchmark).mockReturnValue({
      mutate: vi.fn(),
      isPending: false,
      variables: undefined,
    } as ReturnType<typeof useCancelBenchmark>);

    vi.mocked(useDeleteBenchmark).mockReturnValue({
      mutate: vi.fn(),
      isPending: false,
      variables: undefined,
    } as ReturnType<typeof useDeleteBenchmark>);
  });

  it('renders loading state', () => {
    vi.mocked(useBenchmarks).mockReturnValue({
      isLoading: true,
      error: null,
      data: undefined,
    } as ReturnType<typeof useBenchmarks>);

    const { container } = render(<BenchmarksListView onViewResults={vi.fn()} />);
    expect(container.querySelector('.animate-spin')).toBeTruthy();
  });

  it('renders error state', () => {
    vi.mocked(useBenchmarks).mockReturnValue({
      isLoading: false,
      error: new Error('boom'),
      data: undefined,
    } as ReturnType<typeof useBenchmarks>);

    render(<BenchmarksListView onViewResults={vi.fn()} />);
    expect(screen.getByText(/Failed to load benchmarks/i)).toBeInTheDocument();
  });

  it('filters by status and triggers mutations', async () => {
    const user = userEvent.setup();

    vi.mocked(useBenchmarks).mockReturnValue({
      isLoading: false,
      error: null,
      data: {
        benchmarks: [
          {
            id: 'b1',
            name: 'Alpha',
            description: null,
            owner_id: 1,
            mapping_id: 1,
            status: 'pending',
            total_runs: 1,
            completed_runs: 0,
            failed_runs: 0,
            created_at: '2025-01-01T00:00:00Z',
            started_at: null,
            completed_at: null,
            operation_uuid: null,
          },
          {
            id: 'b2',
            name: 'Beta',
            description: null,
            owner_id: 1,
            mapping_id: 1,
            status: 'running',
            total_runs: 1,
            completed_runs: 0,
            failed_runs: 0,
            created_at: '2025-01-01T00:00:00Z',
            started_at: null,
            completed_at: null,
            operation_uuid: null,
          },
        ],
      },
    } as ReturnType<typeof useBenchmarks>);

    const start = vi.fn();
    vi.mocked(useStartBenchmark).mockReturnValue({
      mutate: start,
      isPending: false,
      variables: undefined,
    } as ReturnType<typeof useStartBenchmark>);

    const cancel = vi.fn();
    vi.mocked(useCancelBenchmark).mockReturnValue({
      mutate: cancel,
      isPending: false,
      variables: undefined,
    } as ReturnType<typeof useCancelBenchmark>);

    const del = vi.fn();
    vi.mocked(useDeleteBenchmark).mockReturnValue({
      mutate: del,
      isPending: false,
      variables: undefined,
    } as ReturnType<typeof useDeleteBenchmark>);

    render(<BenchmarksListView onViewResults={vi.fn()} />);

    expect(screen.getByText('Alpha')).toBeInTheDocument();
    expect(screen.getByText('Beta')).toBeInTheDocument();

    await user.selectOptions(screen.getByRole('combobox'), 'pending');
    expect(screen.getByText('Alpha')).toBeInTheDocument();
    expect(screen.queryByText('Beta')).not.toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: /Start/i }));
    expect(start).toHaveBeenCalledWith('b1');

    await user.selectOptions(screen.getByRole('combobox'), 'running');
    expect(screen.getByText('Beta')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: /Cancel/i }));
    expect(cancel).toHaveBeenCalledWith('b2');

    await user.click(screen.getByTitle('Cannot delete running benchmark'));
    expect(del).not.toHaveBeenCalled();
  });

  it('prompts and deletes a non-running benchmark', async () => {
    const user = userEvent.setup();
    const del = vi.fn();
    vi.mocked(useDeleteBenchmark).mockReturnValue({
      mutate: del,
      isPending: false,
      variables: undefined,
    } as ReturnType<typeof useDeleteBenchmark>);

    vi.mocked(useBenchmarks).mockReturnValue({
      isLoading: false,
      error: null,
      data: {
        benchmarks: [
          {
            id: 'b1',
            name: 'Alpha',
            description: null,
            owner_id: 1,
            mapping_id: 1,
            status: 'completed',
            total_runs: 1,
            completed_runs: 1,
            failed_runs: 0,
            created_at: '2025-01-01T00:00:00Z',
            started_at: null,
            completed_at: '2025-01-02T00:00:00Z',
            operation_uuid: null,
          },
        ],
      },
    } as ReturnType<typeof useBenchmarks>);

    render(<BenchmarksListView onViewResults={vi.fn()} />);

    await user.click(screen.getByTitle('Delete benchmark'));
    expect(window.confirm).toHaveBeenCalled();
    expect(del).toHaveBeenCalledWith('b1');
  });
});

