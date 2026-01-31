import type { ReactNode } from 'react';
import { describe, it, expect, beforeEach } from 'vitest';
import { renderHook, waitFor, act } from '../tests/utils/test-utils';
import { http, HttpResponse } from 'msw';
import { server } from '../tests/mocks/server';
import { AllTheProviders } from '../tests/utils/providers';
import { createTestQueryClient } from '../tests/utils/queryClient';
import { useUIStore } from '../stores/uiStore';
import { agentKeys } from '../services/api/v2/agent';
import type { ConversationDetail } from '../types/agent';
import {
  useAgentConversation,
  useConversations,
  useCreateConversation,
  useApplyPipeline,
  useAbandonConversation,
  useUpdateConversationCache,
  useConversationDetail,
} from './useAgentConversation';

const createWrapper = () => {
  const queryClient = createTestQueryClient();
  const Wrapper = ({ children }: { children: ReactNode }) => (
    <AllTheProviders queryClient={queryClient}>{children}</AllTheProviders>
  );
  return { wrapper: Wrapper, queryClient };
};

const makeConversationDetail = (overrides: Partial<ConversationDetail> = {}): ConversationDetail => {
  const now = new Date().toISOString();
  return {
    id: 'conv-1',
    status: 'active',
    source_id: 1,
    inline_source_config: null,
    collection_id: null,
    current_pipeline: null,
    source_analysis: null,
    uncertainties: [],
    messages: [],
    summary: null,
    created_at: now,
    updated_at: now,
    ...overrides,
  };
};

describe('useAgentConversation hooks', () => {
  beforeEach(() => {
    useUIStore.setState({ toasts: [] });
  });

  it('combines server messages with optimistic messages and supports streaming updates', async () => {
    const { wrapper } = createWrapper();
    const { result } = renderHook(() => useAgentConversation('conv-test-123'), { wrapper });

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    const baseCount = result.current.messages.length;
    expect(baseCount).toBeGreaterThan(0);

    act(() => {
      result.current.addOptimisticUserMessage('hi');
    });
    expect(result.current.messages).toHaveLength(baseCount + 1);
    expect(result.current.messages.at(-1)).toMatchObject({ role: 'user', content: 'hi' });

    act(() => {
      result.current.updateLastAssistantMessage('partial');
    });
    expect(result.current.messages).toHaveLength(baseCount + 2);
    expect(result.current.messages.at(-1)).toMatchObject({ role: 'assistant', content: 'partial' });

    act(() => {
      result.current.updateLastAssistantMessage('final');
    });
    expect(result.current.messages).toHaveLength(baseCount + 2);
    expect(result.current.messages.at(-1)).toMatchObject({ role: 'assistant', content: 'final' });

    act(() => {
      result.current.clearOptimisticMessages();
    });
    expect(result.current.messages).toHaveLength(baseCount);
  });

  it('syncWithServer clears optimistic messages and refetches conversation', async () => {
    const { wrapper } = createWrapper();
    const { result } = renderHook(() => useAgentConversation('conv-test-123'), { wrapper });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    act(() => {
      result.current.addOptimisticAssistantMessage('temp');
    });
    expect(result.current.messages.at(-1)).toMatchObject({ role: 'assistant', content: 'temp' });

    server.use(
      http.get('*/api/v2/agent/conversations/:id', ({ params }) => {
        const now = new Date().toISOString();
        return HttpResponse.json({
          id: params.id,
          status: 'active',
          source_id: 42,
          inline_source_config: null,
          collection_id: null,
          current_pipeline: null,
          source_analysis: null,
          uncertainties: [],
          messages: [{ role: 'assistant', content: 'server-updated', timestamp: now }],
          summary: null,
          created_at: now,
          updated_at: now,
        });
      })
    );

    await act(async () => {
      await result.current.syncWithServer();
    });

    await waitFor(() => expect(result.current.messages.at(-1)).toMatchObject({ content: 'server-updated' }));
    expect(result.current.messages).toHaveLength(1);
  });

  it('useConversations lists conversations', async () => {
    const { wrapper } = createWrapper();
    const { result } = renderHook(() => useConversations(), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data?.conversations).toHaveLength(2);
  });

  it('useConversationDetail fetches conversation detail', async () => {
    const { wrapper } = createWrapper();
    const { result } = renderHook(() => useConversationDetail('conv-test-123'), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data?.id).toBe('conv-test-123');
  });

  it('useCreateConversation creates and caches detail + invalidates lists', async () => {
    const { wrapper, queryClient } = createWrapper();
    // Seed list query without observers so invalidation doesn't immediately refetch and clear the flag.
    queryClient.setQueryData(agentKeys.list(undefined), { conversations: [], total: 0 });

    const { result } = renderHook(() => useCreateConversation(), { wrapper });

    act(() => {
      result.current.mutate({ source_id: 123 });
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    const created = result.current.data;
    expect(created?.id).toBeTruthy();
    expect(queryClient.getQueryData(agentKeys.detail(created!.id))).toBeTruthy();

    const listState = queryClient.getQueryState(agentKeys.list(undefined));
    expect(listState?.isInvalidated).toBe(true);
  });

  it('useCreateConversation shows an error toast on failure', async () => {
    server.use(
      http.post('*/api/v2/agent/conversations', () => {
        return HttpResponse.json({ detail: 'Boom' }, { status: 500 });
      })
    );

    const { wrapper } = createWrapper();
    const { result } = renderHook(() => useCreateConversation(), { wrapper });

    act(() => {
      result.current.mutate({ source_id: 123 });
    });

    await waitFor(() => expect(result.current.isError).toBe(true));

    const toasts = useUIStore.getState().toasts;
    expect(toasts).toHaveLength(1);
    expect(toasts[0].type).toBe('error');
    expect(toasts[0].message).toContain('Failed to create conversation:');
    expect(toasts[0].message).toContain('Boom');
  });

  it('useApplyPipeline updates cached conversation status + collection and shows success toast', async () => {
    const { wrapper, queryClient } = createWrapper();
    queryClient.setQueryData(agentKeys.detail('conv-1'), makeConversationDetail({ id: 'conv-1' }));
    queryClient.setQueryData(['collections'], []);

    const { result } = renderHook(() => useApplyPipeline(), { wrapper });

    act(() => {
      result.current.mutate({ conversationId: 'conv-1', data: { collection_name: 'My Collection' } });
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    const updated = queryClient.getQueryData(agentKeys.detail('conv-1')) as ConversationDetail;
    expect(updated.status).toBe('applied');
    expect(updated.collection_id).toBe('coll-new-123');

    const toasts = useUIStore.getState().toasts;
    expect(toasts.at(-1)?.type).toBe('success');
    expect(toasts.at(-1)?.message).toBe('Collection "My Collection" created successfully');

    expect(queryClient.getQueryState(['collections'])?.isInvalidated).toBe(true);
  });

  it('useApplyPipeline shows an error toast on failure', async () => {
    server.use(
      http.post('*/api/v2/agent/conversations/:id/apply', () => {
        return HttpResponse.json({ detail: 'Nope' }, { status: 400 });
      })
    );

    const { wrapper } = createWrapper();
    const { result } = renderHook(() => useApplyPipeline(), { wrapper });

    act(() => {
      result.current.mutate({ conversationId: 'conv-1', data: { collection_name: 'My Collection' } });
    });

    await waitFor(() => expect(result.current.isError).toBe(true));

    const toasts = useUIStore.getState().toasts;
    expect(toasts).toHaveLength(1);
    expect(toasts[0].type).toBe('error');
    expect(toasts[0].message).toContain('Failed to apply pipeline:');
    expect(toasts[0].message).toContain('Nope');
  });

  it('useAbandonConversation updates cached status and shows info toast', async () => {
    const { wrapper, queryClient } = createWrapper();
    queryClient.setQueryData(agentKeys.detail('conv-1'), makeConversationDetail({ id: 'conv-1' }));

    const { result } = renderHook(() => useAbandonConversation(), { wrapper });

    act(() => {
      result.current.mutate('conv-1');
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    const updated = queryClient.getQueryData(agentKeys.detail('conv-1')) as ConversationDetail;
    expect(updated.status).toBe('abandoned');

    const toasts = useUIStore.getState().toasts;
    expect(toasts).toHaveLength(1);
    expect(toasts[0].type).toBe('info');
    expect(toasts[0].message).toBe('Conversation abandoned');
  });

  it('useUpdateConversationCache updates pipeline and uncertainties in cache', async () => {
    const { wrapper, queryClient } = createWrapper();
    queryClient.setQueryData(agentKeys.detail('conv-1'), makeConversationDetail({ id: 'conv-1', uncertainties: [] }));

    const { result } = renderHook(() => useUpdateConversationCache(), { wrapper });

    act(() => {
      result.current.updatePipeline('conv-1', { chunking_strategy: 'semantic' });
    });

    const afterPipeline = queryClient.getQueryData(agentKeys.detail('conv-1')) as ConversationDetail;
    expect(afterPipeline.current_pipeline).toEqual({ chunking_strategy: 'semantic' });

    act(() => {
      result.current.addUncertainty('conv-1', { id: 'u1', severity: 'info', message: 'FYI', resolved: false });
    });

    const afterUncertainty = queryClient.getQueryData(agentKeys.detail('conv-1')) as ConversationDetail;
    expect(afterUncertainty.uncertainties).toHaveLength(1);
    expect(afterUncertainty.uncertainties[0]).toMatchObject({ id: 'u1', severity: 'info' });
  });
});
