import { act, renderHook } from '@testing-library/react';
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

import { useOperationsSocket } from '../useOperationsSocket';
import { useAuthStore } from '../../stores/authStore';
import type { useWebSocket as useWebSocketHook } from '../useWebSocket';

// Mock useWebSocket so we can observe URL and reconnect calls
const reconnectMock = vi.fn();
type UseWebSocketOptions = Parameters<typeof useWebSocketHook>[1];
const captured: { url: string | null; opts: UseWebSocketOptions | null } = { url: null, opts: null };

vi.mock('../useWebSocket', () => {
  return {
    useWebSocket: (url: string | null, opts: UseWebSocketOptions) => {
      captured.url = url;
      captured.opts = opts;
      return {
        readyState: 1,
        reconnect: reconnectMock,
        sendMessage: vi.fn(),
        disconnect: vi.fn(),
      };
    },
  };
});

// Provide a minimal WebSocket global so OPEN is defined
global.WebSocket = Object.assign(function () {}, {
  OPEN: 1,
}) as unknown as typeof WebSocket;

describe('useOperationsSocket', () => {
  beforeEach(() => {
    reconnectMock.mockClear();
    captured.url = null;
    captured.opts = null;
    // reset auth store
    useAuthStore.setState({ token: null, refreshToken: null, user: null });

    // Ensure we start from a clean runtime config
    delete (window as typeof window & { __API_BASE_URL__?: string }).__API_BASE_URL__;
  });

  it('builds websocket URL and passes auth token via subprotocol', () => {
    (window as typeof window & { __API_BASE_URL__?: string }).__API_BASE_URL__ = 'https://api.example.com/prefix';

    // seed initial token
    useAuthStore.setState({ token: 'token-1', user: { id: 1, username: 'u', email: 'u@example.com', is_active: true, created_at: 'now' }, refreshToken: null });

    const queryClient = new QueryClient();

    const wrapper = ({ children }: { children: React.ReactNode }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );

    const { rerender } = renderHook(() => useOperationsSocket(), { wrapper });

    // URL should NOT contain token (token is passed via subprotocol for security)
    expect(captured.url).toBe('wss://api.example.com/prefix/ws/operations');
    // Token should be in protocols array
    expect(captured.opts?.protocols).toEqual(['access_token.token-1']);

    // change token
    act(() => {
      useAuthStore.setState({ token: 'token-2' });
    });

    rerender();

    // URL should remain the same, only protocols change
    expect(captured.url).toBe('wss://api.example.com/prefix/ws/operations');
    expect(captured.opts?.protocols).toEqual(['access_token.token-2']);
  });
});
