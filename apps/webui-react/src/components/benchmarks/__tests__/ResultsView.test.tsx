import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('../../../hooks/useBenchmarks', () => ({
  useBenchmarks: vi.fn(),
  useBenchmarkResults: vi.fn(),
}));

vi.mock('../ResultsComparison', () => ({
  ResultsComparison: ({ benchmarkId }: { benchmarkId: string }) => (
    <div>comparison:{benchmarkId}</div>
  ),
}));

import { ResultsView } from '../ResultsView';
import { useBenchmarks, useBenchmarkResults } from '../../../hooks/useBenchmarks';

describe('ResultsView', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders loading state', () => {
    vi.mocked(useBenchmarks).mockReturnValue({
      isLoading: true,
      error: null,
      data: undefined,
    } as ReturnType<typeof useBenchmarks>);

    const { container } = render(<ResultsView />);
    expect(container.querySelector('.animate-spin')).toBeTruthy();
  });

  it('renders error state', () => {
    vi.mocked(useBenchmarks).mockReturnValue({
      isLoading: false,
      error: new Error('boom'),
      data: undefined,
    } as ReturnType<typeof useBenchmarks>);

    render(<ResultsView />);
    expect(screen.getByText(/Failed to load results/i)).toBeInTheDocument();
    expect(screen.getByText(/boom/i)).toBeInTheDocument();
  });

  it('filters to completed benchmarks and supports search', async () => {
    const user = userEvent.setup();

    vi.mocked(useBenchmarks).mockReturnValue({
      isLoading: false,
      error: null,
      data: {
        benchmarks: [
          {
            id: 'b1',
            name: 'Alpha',
            description: 'first',
            owner_id: 1,
            mapping_id: 1,
            status: 'completed',
            total_runs: 1,
            completed_runs: 1,
            failed_runs: 0,
            created_at: '2025-01-01T00:00:00Z',
            started_at: null,
            completed_at: '2025-01-01T00:00:00Z',
            operation_uuid: null,
          },
          {
            id: 'b2',
            name: 'Beta',
            description: null,
            owner_id: 1,
            mapping_id: 1,
            status: 'running',
            total_runs: 2,
            completed_runs: 0,
            failed_runs: 0,
            created_at: '2025-01-01T00:00:00Z',
            started_at: null,
            completed_at: null,
            operation_uuid: null,
          },
          {
            id: 'b3',
            name: 'Gamma',
            description: 'second',
            owner_id: 1,
            mapping_id: 1,
            status: 'completed',
            total_runs: 2,
            completed_runs: 2,
            failed_runs: 0,
            created_at: '2025-01-01T00:00:00Z',
            started_at: null,
            completed_at: '2025-01-02T00:00:00Z',
            operation_uuid: null,
          },
        ],
      },
    } as ReturnType<typeof useBenchmarks>);

    vi.mocked(useBenchmarkResults).mockImplementation((benchmarkId: string) => {
      if (benchmarkId === 'b1') {
        return {
          data: {
            benchmark_id: 'b1',
            primary_k: 10,
            k_values_for_metrics: [10],
            total_runs: 1,
            summary: {},
            runs: [
              {
                id: 'r1',
                run_order: 0,
                config_hash: 'cfg-1',
                config: { search_mode: 'dense' },
                status: 'completed',
                error_message: null,
                metrics: { mrr: 0.9, precision: { '10': 0.5 }, ndcg: { '10': 0.7 }, recall: { '10': 0.1 } },
                metrics_flat: {},
                timing: { indexing_ms: null, evaluation_ms: null, total_ms: 10 },
              },
            ],
          },
        } as ReturnType<typeof useBenchmarkResults>;
      }

      return {
        data: {
          benchmark_id: benchmarkId,
          primary_k: 10,
          k_values_for_metrics: [10],
          total_runs: 1,
          summary: {},
          runs: [],
        },
      } as ReturnType<typeof useBenchmarkResults>;
    });

    render(<ResultsView />);

    expect(screen.getByText('Alpha')).toBeInTheDocument();
    expect(screen.getByText('Gamma')).toBeInTheDocument();
    expect(screen.queryByText('Beta')).not.toBeInTheDocument();

    await user.type(screen.getByPlaceholderText(/Search completed benchmarks/i), 'alp');
    expect(screen.getByText('Alpha')).toBeInTheDocument();
    expect(screen.queryByText('Gamma')).not.toBeInTheDocument();
  });

  it('navigates to comparison view after selecting a benchmark', async () => {
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
            status: 'completed',
            total_runs: 1,
            completed_runs: 1,
            failed_runs: 0,
            created_at: '2025-01-01T00:00:00Z',
            started_at: null,
            completed_at: '2025-01-01T00:00:00Z',
            operation_uuid: null,
          },
        ],
      },
    } as ReturnType<typeof useBenchmarks>);

    vi.mocked(useBenchmarkResults).mockReturnValue({
      data: { benchmark_id: 'b1', primary_k: 10, k_values_for_metrics: [10], total_runs: 1, summary: {}, runs: [] },
    } as ReturnType<typeof useBenchmarkResults>);

    render(<ResultsView />);

    await user.click(screen.getByRole('button', { name: /Alpha/i }));
    expect(screen.getByText('comparison:b1')).toBeInTheDocument();
  });
});
