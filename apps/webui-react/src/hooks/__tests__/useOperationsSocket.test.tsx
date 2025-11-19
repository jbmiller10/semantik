import { act, renderHook } from '@testing-library/react';
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

import { useOperationsSocket } from '../useOperationsSocket';
import { useAuthStore } from '../../stores/authStore';

// Mock useWebSocket so we can observe URL and reconnect calls
const reconnectMock = vi.fn();
const captured: { url: string | null; opts: any } = { url: null, opts: null };

vi.mock('../useWebSocket', () => {
  return {
    useWebSocket: (url: string | null, opts: any) => {
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
  });

  it('builds websocket URL from auth token and reconnects on token change', () => {
    // seed initial token
    useAuthStore.setState({ token: 'token-1', user: { id: 1, username: 'u', email: 'u@example.com', is_active: true, created_at: 'now' }, refreshToken: null });

    const queryClient = new QueryClient();

    const wrapper = ({ children }: { children: React.ReactNode }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );

    const { rerender } = renderHook(() => useOperationsSocket(), { wrapper });

    expect(captured.url).toContain('token-1');
    // Initial effect triggers a reconnect once on mount.
    expect(reconnectMock).toHaveBeenCalledTimes(1);

    // change token
    act(() => {
      useAuthStore.setState({ token: 'token-2' });
    });

    rerender();

    expect(captured.url).toContain('token-2');
    expect(reconnectMock).toHaveBeenCalledTimes(2);
  });
});
