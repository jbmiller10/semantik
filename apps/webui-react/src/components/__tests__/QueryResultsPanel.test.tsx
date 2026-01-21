import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@/tests/utils/test-utils'
import userEvent from '@testing-library/user-event'
import { QueryResultsPanel } from '../benchmarks/QueryResultsPanel'
import { useBenchmarkQueryResults } from '../../hooks/useBenchmarks'
import type { RunQueryResultsResponse } from '@/types/benchmark'

vi.mock('../../hooks/useBenchmarks', () => ({
  useBenchmarkQueryResults: vi.fn(),
}))

describe('QueryResultsPanel', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders error state', () => {
    vi.mocked(useBenchmarkQueryResults).mockReturnValue({
      data: undefined,
      isLoading: false,
      error: new Error('boom'),
    } as never)

    render(<QueryResultsPanel benchmarkId="b1" runId="r1" onBack={vi.fn()} />)

    expect(screen.getByText(/failed to load query results/i)).toBeInTheDocument()
    expect(screen.getByText(/boom/i)).toBeInTheDocument()
  })

  it('filters results and renders metric formatting', async () => {
    const user = userEvent.setup()

    const data: RunQueryResultsResponse = {
      run_id: 'run-1',
      total: 2,
      page: 1,
      per_page: 20,
      results: [
        {
          query_id: 1,
          query_key: 'q1',
          query_text: 'hello world',
          retrieved_doc_ids: [],
          precision_at_k: 0.85,
          recall_at_k: 0.55,
          reciprocal_rank: 0.2,
          ndcg_at_k: null,
          search_time_ms: 12,
          rerank_time_ms: null,
        },
        {
          query_id: 2,
          query_key: 'q2',
          query_text: 'goodbye',
          retrieved_doc_ids: [],
          precision_at_k: null,
          recall_at_k: null,
          reciprocal_rank: null,
          ndcg_at_k: 0.9,
          search_time_ms: null,
          rerank_time_ms: 3,
        },
      ],
    }

    vi.mocked(useBenchmarkQueryResults).mockReturnValue({
      data,
      isLoading: false,
      error: null,
    } as never)

    render(<QueryResultsPanel benchmarkId="b1" runId="r1" onBack={vi.fn()} />)

    expect(screen.getByText(/2 queries evaluated/i)).toBeInTheDocument()
    expect(screen.getByText('q1')).toBeInTheDocument()
    expect(screen.getByText('q2')).toBeInTheDocument()

    // MetricValue rendering (green/amber/red/default + null)
    expect(screen.getByText('0.850')).toHaveClass('text-green-400')
    expect(screen.getByText('0.550')).toHaveClass('text-amber-400')
    expect(screen.getByText('0.200')).toHaveClass('text-red-400')
    expect(screen.getAllByText('-').length).toBeGreaterThan(0)

    // Filter down to one row
    await user.type(screen.getByPlaceholderText(/filter queries/i), 'good')
    expect(screen.queryByText('q1')).not.toBeInTheDocument()
    expect(screen.getByText('q2')).toBeInTheDocument()

    // No matches
    await user.clear(screen.getByPlaceholderText(/filter queries/i))
    await user.type(screen.getByPlaceholderText(/filter queries/i), 'nomatch')
    expect(screen.getByText(/no matching queries found/i)).toBeInTheDocument()
  })

  it('paginates results', async () => {
    const user = userEvent.setup()

    vi.mocked(useBenchmarkQueryResults).mockImplementation((_benchmarkId, _runId, params) => {
      const page = params?.page ?? 1
      return {
        data: {
          run_id: 'run-1',
          total: 40,
          page,
          per_page: 20,
          results: [
            {
              query_id: page,
              query_key: `q${page}`,
              query_text: `query ${page}`,
              retrieved_doc_ids: [],
              precision_at_k: 0.1,
              recall_at_k: 0.1,
              reciprocal_rank: 0.1,
              ndcg_at_k: 0.1,
              search_time_ms: null,
              rerank_time_ms: null,
            },
          ],
        },
        isLoading: false,
        error: null,
      } as never
    })

    render(<QueryResultsPanel benchmarkId="b1" runId="r1" onBack={vi.fn()} />)

    expect(screen.getByText(/page 1 of 2/i)).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /next page/i }))
    expect(screen.getByText(/page 2 of 2/i)).toBeInTheDocument()

    expect(vi.mocked(useBenchmarkQueryResults)).toHaveBeenLastCalledWith('b1', 'r1', { page: 2, per_page: 20 })
  })
})
