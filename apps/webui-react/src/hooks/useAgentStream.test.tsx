import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, waitFor, act } from '../tests/utils/test-utils';
import { useAgentStream } from './useAgentStream';
import { useAuthStore } from '../stores/authStore';

function encodeUtf8(text: string): Uint8Array {
  return new TextEncoder().encode(text);
}

function sse(event: string, data: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

function createStreamingFetch(chunks: Array<string | Uint8Array>) {
  return vi.fn(async () => {
    const reader = {
      idx: 0,
      cancelled: false,
      read: async () => {
        if (reader.cancelled || reader.idx >= chunks.length) {
          return { done: true as const, value: undefined };
        }
        const chunk = chunks[reader.idx++];
        return {
          done: false as const,
          value: typeof chunk === 'string' ? encodeUtf8(chunk) : chunk,
        };
      },
      cancel: async () => {
        reader.cancelled = true;
      },
    };

    return {
      ok: true,
      status: 200,
      json: async () => ({}),
      body: {
        getReader: () => reader,
      },
    } as unknown as Response;
  });
}

describe('useAgentStream', () => {
  const originalFetch = global.fetch;

  beforeEach(() => {
    useAuthStore.setState({ token: 'test-token', refreshToken: null, user: null });
  });

  afterEach(() => {
    global.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it('processes valid SSE events and updates state + callbacks', async () => {
    vi.spyOn(Date, 'now').mockReturnValue(123);

    const callbacks = {
      onContent: vi.fn(),
      onToolCallStart: vi.fn(),
      onToolCallEnd: vi.fn(),
      onSubagentStart: vi.fn(),
      onSubagentEnd: vi.fn(),
      onUncertainty: vi.fn(),
      onPipelineUpdate: vi.fn(),
      onDone: vi.fn(),
      onError: vi.fn(),
      onStatus: vi.fn(),
      onActivity: vi.fn(),
      onQuestion: vi.fn(),
    };

    global.fetch = createStreamingFetch([
      sse('content', { text: 'Hello ' }),
      sse('tool_call_start', { tool: 'search', arguments: { q: 'x' } }),
      sse('tool_call_end', { tool: 'search', success: true, result: { n: 1 } }),
      sse('subagent_start', { name: 'planner', task: 'make plan' }),
      sse('subagent_end', { name: 'planner', success: false, error: 'boom' }),
      sse('uncertainty', { id: 'u1', severity: 'notable', message: 'hmm', context: { a: 1 } }),
      sse('pipeline_update', { pipeline: {} }),
      sse('status', { phase: 'analyzing', message: 'Thinking', progress: { current: 1, total: 2 } }),
      sse('activity', { message: 'did thing', timestamp: '2020-01-01T00:00:00Z' }),
      sse('question', { id: 'q1', message: 'Pick one', options: [{ id: 'o1', label: 'Yes' }], allowCustom: true }),
      sse('done', { pipeline_updated: false, uncertainties_added: [], tool_calls: [], max_turns_reached: false }),
    ]);

    const { result } = renderHook(() => useAgentStream('conv-1', callbacks));

    await act(async () => {
      await result.current.sendMessage('hi');
    });

    expect(result.current.isStreaming).toBe(false);
    expect(result.current.error).toBeNull();
    expect(result.current.currentContent).toBe('Hello ');

    expect(result.current.toolCalls).toHaveLength(1);
    expect(result.current.toolCalls[0]).toMatchObject({
      id: '123-search',
      tool: 'search',
      status: 'success',
      arguments: { q: 'x' },
      result: { n: 1 },
    });

    expect(result.current.subagents).toHaveLength(1);
    expect(result.current.subagents[0]).toMatchObject({
      id: '123-planner',
      name: 'planner',
      task: 'make plan',
      status: 'error',
      error: 'boom',
    });

    expect(result.current.uncertainties).toHaveLength(1);
    expect(result.current.uncertainties[0]).toMatchObject({
      id: 'u1',
      severity: 'notable',
      message: 'hmm',
      resolved: false,
      context: { a: 1 },
    });

    expect(result.current.pipeline).toEqual({});
    expect(result.current.status).toEqual({
      phase: 'analyzing',
      message: 'Thinking',
      progress: { current: 1, total: 2 },
    });
    expect(result.current.activities).toEqual([{ message: 'did thing', timestamp: '2020-01-01T00:00:00Z' }]);
    expect(result.current.pendingQuestions).toHaveLength(1);
    expect(result.current.pendingQuestions[0]).toMatchObject({ id: 'q1', allowCustom: true });

    expect(callbacks.onContent).toHaveBeenCalledWith('Hello ');
    expect(callbacks.onToolCallStart).toHaveBeenCalledOnce();
    expect(callbacks.onToolCallEnd).toHaveBeenCalledOnce();
    expect(callbacks.onSubagentStart).toHaveBeenCalledOnce();
    expect(callbacks.onSubagentEnd).toHaveBeenCalledOnce();
    expect(callbacks.onUncertainty).toHaveBeenCalledOnce();
    expect(callbacks.onPipelineUpdate).toHaveBeenCalledOnce();
    expect(callbacks.onStatus).toHaveBeenCalledOnce();
    expect(callbacks.onActivity).toHaveBeenCalledOnce();
    expect(callbacks.onQuestion).toHaveBeenCalledOnce();
    expect(callbacks.onDone).toHaveBeenCalledOnce();
    expect(callbacks.onError).not.toHaveBeenCalled();
  });

  it('skips invalid known event payloads (schema validation)', async () => {
    const callbacks = { onContent: vi.fn() };

    global.fetch = createStreamingFetch([
      sse('content', { bad: true }), // missing required `text`
      sse('content', { text: 'ok' }),
    ]);

    const { result } = renderHook(() => useAgentStream('conv-1', callbacks));

    await act(async () => {
      await result.current.sendMessage('hi');
    });

    expect(result.current.currentContent).toBe('ok');
    expect(callbacks.onContent).toHaveBeenCalledOnce();
    expect(callbacks.onContent).toHaveBeenCalledWith('ok');
  });

  it('surfaces parse failures and errors after consecutive threshold', async () => {
    const callbacks = { onError: vi.fn() };

    const badJson = 'event: content\ndata: {"text":\n\n';
    global.fetch = createStreamingFetch([badJson, badJson, badJson]);

    const { result } = renderHook(() => useAgentStream('conv-1', callbacks));

    await act(async () => {
      await result.current.sendMessage('hi');
    });

    expect(callbacks.onError).toHaveBeenCalledWith('Some data may be missing due to a communication issue');
    expect(callbacks.onError).toHaveBeenCalledWith(
      'Stream corrupted: 3 parse failures - please refresh and try again'
    );
    expect(result.current.error).toBe('Stream corrupted: 3 parse failures - please refresh and try again');
  });

  it('cancels an in-progress stream and marks running items as cancelled', async () => {
    vi.spyOn(Date, 'now').mockReturnValue(123);

    global.fetch = vi.fn(async (_url: string, init?: RequestInit) => {
      const signal = init?.signal;
      let stage = 0;

      const reader = {
        read: async () => {
          if (stage === 0) {
            stage++;
            return { done: false as const, value: encodeUtf8(sse('tool_call_start', { tool: 'search', arguments: {} })) };
          }

          return await new Promise<{ done: boolean; value?: Uint8Array }>((_resolve, reject) => {
            if (!signal) {
              reject(Object.assign(new Error('Missing abort signal'), { name: 'AbortError' }));
              return;
            }
            signal.addEventListener('abort', () => {
              reject(Object.assign(new Error('Aborted'), { name: 'AbortError' }));
            });
          });
        },
      };

      return {
        ok: true,
        status: 200,
        json: async () => ({}),
        body: { getReader: () => reader },
      } as unknown as Response;
    });

    const { result } = renderHook(() => useAgentStream('conv-1'));

    let sendPromise: Promise<void>;
    act(() => {
      sendPromise = result.current.sendMessage('hi');
    });

    await waitFor(() => expect(result.current.toolCalls).toHaveLength(1));
    expect(result.current.toolCalls[0].status).toBe('running');

    act(() => {
      result.current.cancel();
    });

    await act(async () => {
      await sendPromise;
    });

    expect(result.current.isStreaming).toBe(false);
    expect(result.current.toolCalls[0]).toMatchObject({ tool: 'search', status: 'error', error: 'Cancelled' });
  });

  it('sets error when initial HTTP response is not ok', async () => {
    const callbacks = { onError: vi.fn() };

    global.fetch = vi.fn(async () => {
      return {
        ok: false,
        status: 500,
        json: async () => ({ detail: 'Nope' }),
      } as unknown as Response;
    });

    const { result } = renderHook(() => useAgentStream('conv-1', callbacks));

    await act(async () => {
      await result.current.sendMessage('hi');
    });

    expect(result.current.error).toBe('Nope');
    expect(callbacks.onError).toHaveBeenCalledWith('Nope');
  });
});
