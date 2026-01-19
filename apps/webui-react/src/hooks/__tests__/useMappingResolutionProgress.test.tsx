import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { renderHook, act } from '@testing-library/react'
import type { ReactNode } from 'react'

import { useAuthStore } from '../../stores/authStore'
import { datasetKeys } from '../useBenchmarks'
import { useMappingResolutionProgress } from '../useMappingResolutionProgress'

let lastOnMessage: ((event: MessageEvent) => void) | undefined

vi.mock('../useWebSocket', () => ({
  useWebSocket: (_url: string | null, options: unknown) => {
    const onMessage =
      options && typeof options === 'object' && 'onMessage' in options
        ? (options as { onMessage?: (event: MessageEvent) => void }).onMessage
        : undefined
    lastOnMessage = onMessage
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

describe('useMappingResolutionProgress', () => {
  it('ignores unrelated messages and processes completed/failed updates', () => {
    useAuthStore.setState({ token: 'token', refreshToken: null, user: null })

    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const invalidateSpy = vi.spyOn(queryClient, 'invalidateQueries')
    const onComplete = vi.fn()
    const onError = vi.fn()

    const wrapper = ({ children }: { children: ReactNode }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    )

    const { result } = renderHook(
      () =>
        useMappingResolutionProgress('op-1', {
          datasetId: 'ds-1',
          onComplete,
          onError,
        }),
      { wrapper }
    )

    expect(result.current.isConnected).toBe(true)

    act(() => {
      lastOnMessage?.(
        new MessageEvent('message', {
          data: JSON.stringify({ type: 'other', data: {} }),
        })
      )
    })
    expect(onComplete).not.toHaveBeenCalled()
    expect(onError).not.toHaveBeenCalled()

    act(() => {
      lastOnMessage?.(
        new MessageEvent('message', {
          data: JSON.stringify({
            message: {
              type: 'benchmark_mapping_resolution_progress',
              data: { operation_id: 'op-2', stage: 'completed' },
            },
          }),
        })
      )
    })
    expect(onComplete).not.toHaveBeenCalled()

    act(() => {
      lastOnMessage?.(
        new MessageEvent('message', {
          data: JSON.stringify({
            message: {
              type: 'benchmark_mapping_resolution_progress',
              data: {
                operation_id: 'op-1',
                stage: 'completed',
                total_refs: 10,
                processed_refs: 10,
                resolved_refs: 9,
                ambiguous_refs: 1,
                unresolved_refs: 0,
              },
            },
          }),
        })
      )
    })

    expect(onComplete).toHaveBeenCalledTimes(1)
    expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: datasetKeys.mappings('ds-1') })

    act(() => {
      lastOnMessage?.(
        new MessageEvent('message', {
          data: JSON.stringify({
            message: {
              type: 'benchmark_mapping_resolution_progress',
              data: { operation_id: 'op-1', stage: 'failed' },
            },
          }),
        })
      )
    })

    expect(onError).toHaveBeenCalledWith('Mapping resolution failed')
    expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: datasetKeys.mappings('ds-1') })
  })
})
