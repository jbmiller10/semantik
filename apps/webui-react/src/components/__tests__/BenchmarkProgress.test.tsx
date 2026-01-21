import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@/tests/utils/test-utils'
import userEvent from '@testing-library/user-event'
import { BenchmarkProgress } from '../benchmarks/BenchmarkProgress'
import { useBenchmarkProgress } from '../../hooks/useBenchmarkProgress'
import { useCancelBenchmark } from '../../hooks/useBenchmarks'
import type { Benchmark } from '@/types/benchmark'

vi.mock('../../hooks/useBenchmarkProgress', () => ({
  useBenchmarkProgress: vi.fn(),
}))

vi.mock('../../hooks/useBenchmarks', () => ({
  useCancelBenchmark: vi.fn(),
}))

const benchmark: Benchmark = {
  id: 'bench-1',
  name: 'Benchmark 1',
  description: null,
  owner_id: 1,
  mapping_id: 1,
  status: 'running',
  total_runs: 2,
  completed_runs: 0,
  failed_runs: 0,
  created_at: '2025-01-01T00:00:00Z',
  started_at: '2025-01-01T00:01:00Z',
  completed_at: null,
  operation_uuid: 'op-1',
}

describe('BenchmarkProgress', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders progress and cancels when confirmed', async () => {
    const user = userEvent.setup()

    vi.mocked(useBenchmarkProgress).mockReturnValue({
      progress: {
        totalRuns: 2,
        completedRuns: 1,
        failedRuns: 1,
        primaryK: 10,
        kValuesForMetrics: [10],
        currentRunOrder: 2,
        currentRunConfig: { search_mode: 'dense', use_reranker: true, top_k: 20 },
        status: 'running',
        stage: 'evaluating',
        currentQueries: { total: 10, processed: 5 },
        recentMetrics: [
          {
            runId: 'run-1',
            runOrder: 1,
            config: { search_mode: 'dense', use_reranker: true, top_k: 20 },
            metrics: { mrr: 0.5, precision: { '10': 0.6 }, ndcg: { '10': 0.55 }, recall: {} },
            timing: { search_ms: null, rerank_ms: null, total_ms: 200 },
          },
        ],
      },
      isConnected: false,
    } as never)

    const mutate = vi.fn()
    vi.mocked(useCancelBenchmark).mockReturnValue({ mutate, isPending: false } as never)

    vi.stubGlobal('confirm', vi.fn(() => true))

    render(<BenchmarkProgress benchmark={benchmark} />)

    expect(screen.getByText('Benchmark 1')).toBeInTheDocument()
    expect(screen.getByText(/reconnecting/i)).toBeInTheDocument()
    expect(screen.getByText(/2\s*\/\s*2 runs/i)).toBeInTheDocument()
    expect(screen.getByText(/1 failed/i)).toBeInTheDocument()

    // Recent metrics formatting
    expect(screen.getByText('0.500')).toBeInTheDocument()
    expect(screen.getByText('0.550')).toBeInTheDocument()
    expect(screen.getByText('0.600')).toBeInTheDocument()
    expect(screen.getByText('200ms')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /cancel/i }))
    expect(mutate).toHaveBeenCalledWith('bench-1')
  })

  it('shows completion banner when completed', () => {
    vi.mocked(useBenchmarkProgress).mockReturnValue({
      progress: {
        totalRuns: 2,
        completedRuns: 2,
        failedRuns: 0,
        primaryK: 10,
        kValuesForMetrics: [10],
        currentRunOrder: 0,
        currentRunConfig: null,
        status: 'completed',
        stage: 'completed',
        currentQueries: { total: 0, processed: 0 },
        recentMetrics: [],
      },
      isConnected: true,
    } as never)

    vi.mocked(useCancelBenchmark).mockReturnValue({ mutate: vi.fn(), isPending: false } as never)

    render(<BenchmarkProgress benchmark={benchmark} />)

    expect(screen.getByText(/benchmark complete/i)).toBeInTheDocument()
    expect(screen.getByText(/all 2 runs have finished/i)).toBeInTheDocument()
  })
})
