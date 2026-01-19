import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { renderHook, act, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'

import { useAuthStore } from '../../stores/authStore'
import { benchmarkKeys } from '../useBenchmarks'
import { useBenchmarkProgress } from '../useBenchmarkProgress'

let lastOnMessage: ((event: MessageEvent) => void) | undefined

vi.mock('../useWebSocket', () => ({
  useWebSocket: (_url: string | null, options: any) => {
    lastOnMessage = options?.onMessage
    return { readyState: WebSocket.OPEN }
  },
}))

vi.mock('../../services/api/v2/operations', () => ({
  operationsV2Api: {
    getGlobalWebSocketConnectionInfo: vi.fn(() => ({
      url: 'ws://test/ws',
      protocols: ['proto'],
    })),
  },
}))

describe('useBenchmarkProgress', () => {
  it('handles benchmark_progress messages, invalidates caches, and calls callbacks', async () => {
    useAuthStore.setState({ token: 'token', refreshToken: null, user: null })

    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const invalidateSpy = vi.spyOn(queryClient, 'invalidateQueries')
    const onComplete = vi.fn()
    const onError = vi.fn()

    const wrapper = ({ children }: { children: ReactNode }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    )

    const { result } = renderHook(
      () => useBenchmarkProgress('bench-1', 'op-1', { onComplete, onError }),
      { wrapper }
    )

    act(() => {
      lastOnMessage?.(
        new MessageEvent('message', {
          data: JSON.stringify({
            type: 'benchmark_progress',
            data: {
              benchmark_id: 'other',
              operation_id: 'op-1', // backwards-compat should still allow
              total_runs: 10,
              completed_runs: 2,
              failed_runs: 0,
              primary_k: 5,
              k_values_for_metrics: [5, 10],
              status: 'running',
              stage: 'evaluating',
              last_completed_run: {
                run_id: 'run-1',
                run_order: 1,
                config: { alpha: 1 },
                metrics: { mrr: 0.5 },
                timing: { total_ms: 123 },
              },
            },
          }),
        })
      )
    })

    await waitFor(() => {
      expect(result.current.progress.totalRuns).toBe(10)
      expect(result.current.progress.completedRuns).toBe(2)
      expect(result.current.progress.primaryK).toBe(5)
      expect(result.current.progress.recentMetrics).toHaveLength(1)
    })

    // Repeat the same last_completed_run to exercise de-dupe by runId
    act(() => {
      lastOnMessage?.(
        new MessageEvent('message', {
          data: JSON.stringify({
            type: 'benchmark_progress',
            data: {
              benchmark_id: 'bench-1',
              total_runs: 10,
              completed_runs: 3,
              status: 'running',
              last_completed_run: {
                run_id: 'run-1',
                run_order: 1,
                config: { alpha: 1 },
                metrics: { mrr: 0.5 },
                timing: { total_ms: 123 },
              },
            },
          }),
        })
      )
    })

    await waitFor(() => {
      expect(result.current.progress.recentMetrics).toHaveLength(1)
      expect(result.current.progress.completedRuns).toBe(3)
    })

    // Generic progress update can estimate completed runs
    act(() => {
      lastOnMessage?.(
        new MessageEvent('message', {
          data: JSON.stringify({
            type: 'other_progress',
            data: { operation_id: 'op-1', progress_percent: 50 },
          }),
        })
      )
    })

    await waitFor(() => {
      expect(result.current.progress.completedRuns).toBeGreaterThanOrEqual(5)
    })

    act(() => {
      lastOnMessage?.(
        new MessageEvent('message', {
          data: JSON.stringify({
            type: 'benchmark_progress',
            data: {
              benchmark_id: 'bench-1',
              status: 'completed',
              stage: 'completed',
              last_completed_run: { run_id: 'run-2', run_order: 2, config: {}, metrics: { mrr: 0.9 } },
            },
          }),
        })
      )
    })

    await waitFor(() => {
      expect(onComplete).toHaveBeenCalledTimes(1)
    })

    expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: benchmarkKeys.lists() })
    expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: benchmarkKeys.detail('bench-1') })
    expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: benchmarkKeys.results('bench-1') })

    act(() => {
      lastOnMessage?.(
        new MessageEvent('message', {
          data: JSON.stringify({
            type: 'benchmark_progress',
            data: { benchmark_id: 'bench-1', status: 'failed', error_message: 'nope' },
          }),
        })
      )
    })

    await waitFor(() => {
      expect(onError).toHaveBeenCalledWith('nope')
    })
  })
})
