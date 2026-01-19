import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

vi.mock('recharts', () => ({
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  BarChart: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Bar: () => null,
  XAxis: () => null,
  YAxis: () => null,
  Tooltip: () => null,
  Legend: () => null,
}));

vi.mock('../../../hooks/useBenchmarks', () => ({
  useBenchmark: vi.fn(),
  useBenchmarkResults: vi.fn(),
}));

vi.mock('../QueryResultsPanel', () => ({
  QueryResultsPanel: ({ runId }: { runId: string }) => <div>Query Panel {runId}</div>,
}));

import { ResultsComparison } from '../ResultsComparison';
import { useBenchmark, useBenchmarkResults } from '../../../hooks/useBenchmarks';

describe('ResultsComparison', () => {
  beforeEach(() => {
    vi.clearAllMocks();

    vi.mocked(useBenchmark).mockReturnValue({
      data: { name: 'Test Benchmark' },
    } as ReturnType<typeof useBenchmark>);

    vi.mocked(useBenchmarkResults).mockReturnValue({
      data: {
        benchmark_id: 'bench-1',
        primary_k: 10,
        k_values_for_metrics: [10],
        total_runs: 3,
        summary: {},
        runs: [
          {
            id: 'run-1',
            run_order: 0,
            config_hash: 'cfg-1',
            config: { search_mode: 'dense', use_reranker: false, top_k: 10 },
            status: 'completed',
            error_message: null,
            metrics: { mrr: 0.2, precision: { '10': 0.5 }, recall: { '10': 0.3 }, ndcg: { '10': 0.1 } },
            metrics_flat: {},
            timing: { indexing_ms: null, evaluation_ms: null, total_ms: 300 },
          },
          {
            id: 'run-2',
            run_order: 1,
            config_hash: 'cfg-2',
            config: { search_mode: 'dense', use_reranker: true, top_k: 10 },
            status: 'completed',
            error_message: null,
            metrics: { mrr: 0.8, precision: { '10': 0.7 }, recall: { '10': 0.6 }, ndcg: { '10': 0.7 } },
            metrics_flat: {},
            timing: { indexing_ms: null, evaluation_ms: null, total_ms: 400 },
          },
          {
            id: 'run-3',
            run_order: 2,
            config_hash: 'cfg-3',
            config: { search_mode: 'dense', use_reranker: false, top_k: 10 },
            status: 'failed',
            error_message: 'boom',
            metrics: { mrr: null, precision: {}, recall: {}, ndcg: {} },
            metrics_flat: {},
            timing: { indexing_ms: null, evaluation_ms: null, total_ms: null },
          },
        ],
      },
      isLoading: false,
      error: null,
    } as ReturnType<typeof useBenchmarkResults>);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('sorts completed runs by MRR descending by default', () => {
    render(<ResultsComparison benchmarkId="bench-1" onBack={vi.fn()} />);

    expect(screen.getByText('Test Benchmark')).toBeInTheDocument();
    expect(screen.getByText('2 configurations evaluated')).toBeInTheDocument();

    const rows = screen.getAllByRole('row');
    // rows[0] is header; rows[1] should be best MRR (run-2)
    expect(rows[1]).toHaveTextContent('dense + rerank + k=10');
    expect(rows[2]).toHaveTextContent('dense + k=10');
  });

  it('exports CSV by creating a blob URL and clicking a download link', async () => {
    const user = userEvent.setup();
    const createObjectURL = vi.fn(() => 'blob:mock-url');
    const revokeObjectURL = vi.fn();
    global.URL.createObjectURL = createObjectURL;
    global.URL.revokeObjectURL = revokeObjectURL;

    const click = vi.fn();
    const originalCreateElement = document.createElement.bind(document);
    vi.spyOn(document, 'createElement').mockImplementation((tagName: string) => {
      if (tagName === 'a') {
        return { click, set href(_v: string) {}, set download(_v: string) {} } as unknown as HTMLAnchorElement;
      }
      return originalCreateElement(tagName);
    });

    render(<ResultsComparison benchmarkId="bench-1" onBack={vi.fn()} />);
    await user.click(screen.getByRole('button', { name: /CSV/i }));

    expect(createObjectURL).toHaveBeenCalled();
    expect(click).toHaveBeenCalled();
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:mock-url');
  });

  it('exports JSON by creating a blob URL and clicking a download link', async () => {
    const user = userEvent.setup();
    const createObjectURL = vi.fn(() => 'blob:json-url');
    const revokeObjectURL = vi.fn();
    global.URL.createObjectURL = createObjectURL;
    global.URL.revokeObjectURL = revokeObjectURL;

    const click = vi.fn();
    const originalCreateElement = document.createElement.bind(document);
    vi.spyOn(document, 'createElement').mockImplementation((tagName: string) => {
      if (tagName === 'a') {
        return { click, set href(_v: string) {}, set download(_v: string) {} } as unknown as HTMLAnchorElement;
      }
      return originalCreateElement(tagName);
    });

    render(<ResultsComparison benchmarkId="bench-1" onBack={vi.fn()} />);
    await user.click(screen.getByRole('button', { name: /JSON/i }));

    expect(createObjectURL).toHaveBeenCalled();
    expect(click).toHaveBeenCalled();
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:json-url');
  });

  it('toggles chart visibility and sort direction', async () => {
    const user = userEvent.setup();
    render(<ResultsComparison benchmarkId="bench-1" onBack={vi.fn()} />);

    expect(screen.getByRole('button', { name: /Hide Chart/i })).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: /Hide Chart/i }));
    expect(screen.getByRole('button', { name: /Show Chart/i })).toBeInTheDocument();

    const configHeader = screen.getByRole('columnheader', { name: /Configuration/i });
    await user.click(configHeader);
    expect(configHeader).toHaveAttribute('aria-sort', 'descending');
    await user.click(configHeader);
    expect(configHeader).toHaveAttribute('aria-sort', 'ascending');
  });

  it('supports selecting a different K value and drilling into a run', async () => {
    vi.mocked(useBenchmarkResults).mockReturnValue({
      data: {
        benchmark_id: 'bench-1',
        primary_k: 5,
        k_values_for_metrics: [5, 10],
        total_runs: 1,
        summary: {},
        runs: [
          {
            id: 'run-2',
            run_order: 1,
            config_hash: 'cfg-2',
            config: { search_mode: 'dense', use_reranker: true, top_k: 10 },
            status: 'completed',
            error_message: null,
            metrics: { mrr: 0.8, precision: { '5': 0.6, '10': 0.7 }, recall: { '10': 0.6 }, ndcg: { '10': 0.7 } },
            metrics_flat: {},
            timing: { indexing_ms: null, evaluation_ms: null, total_ms: 400 },
          },
        ],
      },
      isLoading: false,
      error: null,
    } as ReturnType<typeof useBenchmarkResults>);

    const user = userEvent.setup();
    render(<ResultsComparison benchmarkId="bench-1" onBack={vi.fn()} />);

    const select = screen.getByRole('combobox') as HTMLSelectElement;
    expect(select.value).toBe('5');
    await user.selectOptions(select, '10');

    expect(screen.getByText('P@10')).toBeInTheDocument();

    await user.click(screen.getByText('dense + rerank + k=10'));
    expect(screen.getByText('Query Panel run-2')).toBeInTheDocument();
  });

  it('renders an error message when results fail to load', () => {
    vi.mocked(useBenchmarkResults).mockReturnValue({
      data: undefined,
      isLoading: false,
      error: new Error('nope'),
    } as ReturnType<typeof useBenchmarkResults>);

    render(<ResultsComparison benchmarkId="bench-1" onBack={vi.fn()} />);
    expect(screen.getByText(/Failed to load results/i)).toBeInTheDocument();
  });
});
