import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act, waitFor } from '@/tests/utils/test-utils';
import { useAuthStore } from '@/stores/authStore';
import { useAgentStream } from '../useAgentStream';

function sse(event: string, data: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

function createStream(chunks: string[], signal?: AbortSignal): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(encoder.encode(chunk));
      }

      if (signal) {
        signal.addEventListener('abort', () => {
          const err = new Error('Aborted');
          (err as Error & { name: string }).name = 'AbortError';
          controller.error(err);
        });
      } else {
        controller.close();
      }
    },
  });
}

describe('useAgentStream', () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    vi.restoreAllMocks();
    useAuthStore.setState({ token: 't', refreshToken: null, user: null });
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it('streams known events and updates state + callbacks', async () => {
    const onContent = vi.fn();
    const onToolCallStart = vi.fn();
    const onToolCallEnd = vi.fn();
    const onStatus = vi.fn();
    const onPipelineUpdate = vi.fn();
    const onQuestion = vi.fn();
    const onDone = vi.fn();

    globalThis.fetch = vi.fn(async () => {
      const stream = createStream([
        sse('status', { phase: 'analyzing', message: 'Working...', progress: { current: 1, total: 3 } }),
        sse('content', { text: 'Hello ' }),
        sse('tool_call_start', { tool: 'build_pipeline', arguments: { a: 1 } }),
        sse('tool_call_end', { tool: 'build_pipeline', success: true, result: { ok: true } }),
        sse('pipeline_update', { pipeline: { chunking_strategy: 'semantic' } }),
        sse('question', { id: 'q1', message: 'Pick one', options: [{ id: 'o1', label: 'Yes' }], allowCustom: false }),
        sse('done', { pipeline_updated: true, uncertainties_added: [], tool_calls: [], max_turns_reached: false }),
      ]);
      return new Response(stream, { status: 200, headers: { 'Content-Type': 'text/event-stream' } });
    }) as unknown as typeof fetch;

    const { result } = renderHook(() =>
      useAgentStream('conv-1', { onContent, onToolCallStart, onToolCallEnd, onStatus, onPipelineUpdate, onQuestion, onDone })
    );

    await act(async () => {
      await result.current.sendMessage('hi');
    });

    expect(result.current.isStreaming).toBe(false);
    expect(result.current.error).toBe(null);
    expect(result.current.currentContent).toBe('Hello ');
    expect(result.current.status).toEqual({ phase: 'analyzing', message: 'Working...', progress: { current: 1, total: 3 } });
    expect(result.current.pipeline).toEqual({ chunking_strategy: 'semantic' });
    expect(result.current.pendingQuestions).toHaveLength(1);
    expect(result.current.toolCalls).toHaveLength(1);
    expect(result.current.toolCalls[0]?.status).toBe('success');

    expect(onContent).toHaveBeenCalledWith('Hello ');
    expect(onToolCallStart).toHaveBeenCalled();
    expect(onToolCallEnd).toHaveBeenCalled();
    expect(onStatus).toHaveBeenCalled();
    expect(onPipelineUpdate).toHaveBeenCalled();
    expect(onQuestion).toHaveBeenCalled();
    expect(onDone).toHaveBeenCalled();
  });

  it('surfaces parse failures and stops after threshold', async () => {
    const onError = vi.fn();

    globalThis.fetch = vi.fn(async () => {
      const bad = 'event: content\ndata: {not-json}\n\n';
      const stream = createStream([bad, bad, bad]);
      return new Response(stream, { status: 200, headers: { 'Content-Type': 'text/event-stream' } });
    }) as unknown as typeof fetch;

    const { result } = renderHook(() => useAgentStream('conv-1', { onError }));

    await act(async () => {
      await result.current.sendMessage('hi');
    });

    expect(onError).toHaveBeenCalled();
    expect(result.current.error).toMatch(/stream corrupted/i);
  });

  it('stops on validation failures for critical events', async () => {
    const onError = vi.fn();

    globalThis.fetch = vi.fn(async () => {
      const stream = createStream([sse('done', { nope: true })]);
      return new Response(stream, { status: 200, headers: { 'Content-Type': 'text/event-stream' } });
    }) as unknown as typeof fetch;

    const { result } = renderHook(() => useAgentStream('conv-1', { onError }));

    await act(async () => {
      await result.current.sendMessage('hi');
    });

    expect(result.current.error).toMatch(/failed to process critical \"done\" event/i);
    expect(onError).toHaveBeenCalled();
  });

  it('marks running tool calls/subagents as cancelled on AbortError', async () => {
    globalThis.fetch = vi.fn(async (_url, init) => {
      const signal = init?.signal as AbortSignal | undefined;
      const stream = createStream(
        [
          sse('tool_call_start', { tool: 'build_pipeline', arguments: {} }),
          sse('subagent_start', { name: 'sub1', task: 't' }),
        ],
        signal
      );
      return new Response(stream, { status: 200, headers: { 'Content-Type': 'text/event-stream' } });
    }) as unknown as typeof fetch;

    const { result } = renderHook(() => useAgentStream('conv-1'));

    let sendPromise: Promise<void> | undefined;
    act(() => {
      sendPromise = result.current.sendMessage('hi');
    });

    await waitFor(() => {
      expect(result.current.toolCalls.some((tc) => tc.status === 'running')).toBe(true);
      expect(result.current.subagents.some((sa) => sa.status === 'running')).toBe(true);
    });

    act(() => {
      result.current.cancel();
    });

    await act(async () => {
      await sendPromise;
    });

    expect(result.current.toolCalls[0]?.status).toBe('error');
    expect(result.current.toolCalls[0]?.error).toBe('Cancelled');
    expect(result.current.subagents[0]?.status).toBe('error');
    expect(result.current.subagents[0]?.error).toBe('Cancelled');
  });

  it('sets error for non-OK responses and includes server detail', async () => {
    const onError = vi.fn();
    globalThis.fetch = vi.fn(async () => {
      return new Response(JSON.stringify({ detail: 'nope' }), { status: 500, headers: { 'Content-Type': 'application/json' } });
    }) as unknown as typeof fetch;

    const { result } = renderHook(() => useAgentStream('conv-1', { onError }));

    await act(async () => {
      await result.current.sendMessage('hi');
    });

    expect(result.current.error).toBe('nope');
    expect(onError).toHaveBeenCalledWith('nope');
  });
});
