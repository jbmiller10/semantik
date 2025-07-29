import { renderHook, act } from '@testing-library/react';
import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { ReactNode } from 'react';
import { useOperationProgress } from '../useOperationProgress';
import { useWebSocket } from '../useWebSocket';
import { useUpdateOperationInCache } from '../useCollectionOperations';
import { useUIStore } from '../../stores/uiStore';
import { operationsV2Api } from '../../services/api/v2/operations';

// Mock dependencies
vi.mock('../useWebSocket');
vi.mock('../useCollectionOperations');
vi.mock('../../stores/uiStore');
vi.mock('../../services/api/v2/operations');

// Test helpers
const createTestQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });

const createWrapper = (queryClient: QueryClient) => {
  return ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

// Mock WebSocket states
const WS_STATES = {
  CONNECTING: 0,
  OPEN: 1,
  CLOSING: 2,
  CLOSED: 3,
};

describe('useOperationProgress', () => {
  let queryClient: QueryClient;
  let mockAddToast: ReturnType<typeof vi.fn>;
  let mockUpdateOperationInCache: ReturnType<typeof vi.fn>;
  let mockSendMessage: ReturnType<typeof vi.fn>;
  let mockOnMessage: ((event: MessageEvent) => void) | undefined;
  let mockOnError: ((event: Event) => void) | undefined;
  let mockOnClose: (() => void) | undefined;

  beforeEach(() => {
    vi.clearAllMocks();
    queryClient = createTestQueryClient();
    
    // Mock UI store
    mockAddToast = vi.fn();
    vi.mocked(useUIStore).mockReturnValue({ addToast: mockAddToast });

    // Mock update operation cache
    mockUpdateOperationInCache = vi.fn();
    vi.mocked(useUpdateOperationInCache).mockReturnValue(mockUpdateOperationInCache);

    // Mock WebSocket hook
    mockSendMessage = vi.fn();
    vi.mocked(useWebSocket).mockImplementation((url, options) => {
      mockOnMessage = options?.onMessage;
      mockOnError = options?.onError;
      mockOnClose = options?.onClose;
      
      return {
        sendMessage: mockSendMessage,
        readyState: url ? WS_STATES.OPEN : WS_STATES.CLOSED,
        lastMessage: null,
        error: null,
      };
    });

    // Mock operations API
    vi.mocked(operationsV2Api.getWebSocketUrl).mockImplementation(
      (opId) => `ws://localhost:8080/ws/operations/${opId}`
    );
  });

  afterEach(() => {
    mockOnMessage = undefined;
    mockOnError = undefined;
    mockOnClose = undefined;
  });

  describe('WebSocket connection', () => {
    it('should establish WebSocket connection with operation ID', () => {
      const operationId = 'op-123';
      
      renderHook(() => useOperationProgress(operationId), {
        wrapper: createWrapper(queryClient),
      });

      expect(operationsV2Api.getWebSocketUrl).toHaveBeenCalledWith(operationId);
      expect(useWebSocket).toHaveBeenCalledWith(
        `ws://localhost:8080/ws/operations/${operationId}`,
        expect.any(Object)
      );
    });

    it('should not establish connection when operation ID is null', () => {
      renderHook(() => useOperationProgress(null), {
        wrapper: createWrapper(queryClient),
      });

      expect(useWebSocket).toHaveBeenCalledWith(null, expect.any(Object));
    });

    it('should return connection status', () => {
      const { result } = renderHook(() => useOperationProgress('op-123'), {
        wrapper: createWrapper(queryClient),
      });

      expect(result.current.isConnected).toBe(true);
      expect(result.current.readyState).toBe(WS_STATES.OPEN);
    });
  });

  describe('Message handling', () => {
    it('should handle operation started message', () => {
      renderHook(() => useOperationProgress('op-123'), {
        wrapper: createWrapper(queryClient),
      });

      const message = {
        type: 'operation_started',
        data: {
          status: 'processing',
          message: 'Operation started',
        },
      };

      act(() => {
        mockOnMessage?.(new MessageEvent('message', { 
          data: JSON.stringify(message) 
        }));
      });

      expect(mockUpdateOperationInCache).toHaveBeenCalledWith('op-123', {
        status: 'processing',
        progress: undefined,
      });
    });

    it('should handle operation progress updates', () => {
      renderHook(() => useOperationProgress('op-123'), {
        wrapper: createWrapper(queryClient),
      });

      const message = {
        type: 'progress_update',
        data: {
          status: 'processing',
          progress: 50,
          message: '50% complete',
        },
      };

      act(() => {
        mockOnMessage?.(new MessageEvent('message', { 
          data: JSON.stringify(message) 
        }));
      });

      expect(mockUpdateOperationInCache).toHaveBeenCalledWith('op-123', {
        status: 'processing',
        progress: 50,
      });

      // Should show toast for 50% progress
      expect(mockAddToast).toHaveBeenCalledWith({
        type: 'info',
        message: '50% complete',
      });
    });

    it('should only show progress toasts at 25% intervals', () => {
      renderHook(() => useOperationProgress('op-123'), {
        wrapper: createWrapper(queryClient),
      });

      // 25% - should show toast
      act(() => {
        mockOnMessage?.(new MessageEvent('message', { 
          data: JSON.stringify({
            data: { status: 'processing', progress: 25, message: '25% complete' }
          })
        }));
      });
      expect(mockAddToast).toHaveBeenCalledTimes(1);

      // 30% - should not show toast
      mockAddToast.mockClear();
      act(() => {
        mockOnMessage?.(new MessageEvent('message', { 
          data: JSON.stringify({
            data: { status: 'processing', progress: 30, message: '30% complete' }
          })
        }));
      });
      expect(mockAddToast).not.toHaveBeenCalled();

      // 50% - should show toast
      act(() => {
        mockOnMessage?.(new MessageEvent('message', { 
          data: JSON.stringify({
            data: { status: 'processing', progress: 50, message: '50% complete' }
          })
        }));
      });
      expect(mockAddToast).toHaveBeenCalledTimes(1);
    });

    it('should handle operation completed message', () => {
      const onComplete = vi.fn();
      
      renderHook(() => useOperationProgress('op-123', { onComplete }), {
        wrapper: createWrapper(queryClient),
      });

      const message = {
        type: 'operation_completed',
        data: {
          message: 'Operation completed successfully',
        },
      };

      act(() => {
        mockOnMessage?.(new MessageEvent('message', { 
          data: JSON.stringify(message) 
        }));
      });

      expect(mockUpdateOperationInCache).toHaveBeenCalledWith('op-123', {
        status: 'completed',
        progress: undefined,
      });

      expect(mockAddToast).toHaveBeenCalledWith({
        type: 'success',
        message: 'Operation completed successfully',
      });

      expect(onComplete).toHaveBeenCalled();
    });

    it('should prevent duplicate completion messages', () => {
      const onComplete = vi.fn();
      
      renderHook(() => useOperationProgress('op-123', { onComplete }), {
        wrapper: createWrapper(queryClient),
      });

      const message = {
        type: 'operation_completed',
        data: {},
      };

      // Send completion message twice
      act(() => {
        mockOnMessage?.(new MessageEvent('message', { 
          data: JSON.stringify(message) 
        }));
      });

      act(() => {
        mockOnMessage?.(new MessageEvent('message', { 
          data: JSON.stringify(message) 
        }));
      });

      // Should only show one toast and call onComplete once
      expect(mockAddToast).toHaveBeenCalledTimes(1);
      expect(onComplete).toHaveBeenCalledTimes(1);
    });

    it('should handle operation failed message', () => {
      const onError = vi.fn();
      
      renderHook(() => useOperationProgress('op-123', { onError }), {
        wrapper: createWrapper(queryClient),
      });

      const message = {
        type: 'operation_failed',
        data: {
          error_message: 'Out of memory error',
        },
      };

      act(() => {
        mockOnMessage?.(new MessageEvent('message', { 
          data: JSON.stringify(message) 
        }));
      });

      expect(mockUpdateOperationInCache).toHaveBeenCalledWith('op-123', {
        status: 'failed',
        progress: undefined,
      });

      expect(mockAddToast).toHaveBeenCalledWith({
        type: 'error',
        message: 'Out of memory error',
      });

      expect(onError).toHaveBeenCalledWith('Out of memory error');
    });

    it('should handle operation cancelled message', () => {
      renderHook(() => useOperationProgress('op-123'), {
        wrapper: createWrapper(queryClient),
      });

      const message = {
        data: {
          status: 'cancelled',
        },
      };

      act(() => {
        mockOnMessage?.(new MessageEvent('message', { 
          data: JSON.stringify(message) 
        }));
      });

      expect(mockUpdateOperationInCache).toHaveBeenCalledWith('op-123', {
        status: 'cancelled',
        progress: undefined,
      });

      expect(mockAddToast).toHaveBeenCalledWith({
        type: 'warning',
        message: 'Operation was cancelled',
      });
    });

    it('should handle current state message on connection', () => {
      renderHook(() => useOperationProgress('op-123'), {
        wrapper: createWrapper(queryClient),
      });

      const message = {
        type: 'current_state',
        data: {
          status: 'processing',
          progress: 75,
        },
      };

      act(() => {
        mockOnMessage?.(new MessageEvent('message', { 
          data: JSON.stringify(message) 
        }));
      });

      expect(mockUpdateOperationInCache).toHaveBeenCalledWith('op-123', {
        status: 'processing',
        progress: 75,
      });
    });

    it('should handle metadata updates', () => {
      renderHook(() => useOperationProgress('op-123'), {
        wrapper: createWrapper(queryClient),
      });

      const message = {
        data: {
          status: 'processing',
          metadata: {
            eta: '2024-01-01T12:00:00Z',
            current_file: 'document.pdf',
          },
        },
      };

      act(() => {
        mockOnMessage?.(new MessageEvent('message', { 
          data: JSON.stringify(message) 
        }));
      });

      // Should be called twice: once for status update, once for metadata
      expect(mockUpdateOperationInCache).toHaveBeenCalledTimes(2);
      
      // First call for status update
      expect(mockUpdateOperationInCache).toHaveBeenNthCalledWith(1, 'op-123', {
        status: 'processing',
        progress: undefined,
      });
      
      // Second call for metadata update
      expect(mockUpdateOperationInCache).toHaveBeenNthCalledWith(2, 'op-123', {
        eta: '2024-01-01T12:00:00Z',
      });
    });
  });

  describe('Error handling', () => {
    it('should handle WebSocket errors', () => {
      renderHook(() => useOperationProgress('op-123'), {
        wrapper: createWrapper(queryClient),
      });

      const errorEvent = new Event('error');
      
      act(() => {
        mockOnError?.(errorEvent);
      });

      expect(mockAddToast).toHaveBeenCalledWith({
        type: 'error',
        message: 'Lost connection to operation progress updates',
      });
    });

    it('should handle malformed messages', () => {
      const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

      renderHook(() => useOperationProgress('op-123'), {
        wrapper: createWrapper(queryClient),
      });

      // Send invalid JSON
      act(() => {
        mockOnMessage?.(new MessageEvent('message', { 
          data: 'invalid json' 
        }));
      });

      expect(consoleErrorSpy).toHaveBeenCalledWith(
        'Failed to parse operation progress message:',
        expect.any(Error),
        'invalid json'
      );

      // Should not update cache or show toasts
      expect(mockUpdateOperationInCache).not.toHaveBeenCalled();
      expect(mockAddToast).not.toHaveBeenCalled();

      consoleErrorSpy.mockRestore();
    });

    it('should handle WebSocket close event', () => {
      const consoleLogSpy = vi.spyOn(console, 'log').mockImplementation(() => {});

      renderHook(() => useOperationProgress('op-123'), {
        wrapper: createWrapper(queryClient),
      });

      act(() => {
        mockOnClose?.();
      });

      expect(consoleLogSpy).toHaveBeenCalledWith(
        'WebSocket closed for operation',
        'op-123'
      );

      consoleLogSpy.mockRestore();
    });
  });

  describe('Options', () => {
    it('should respect showToasts option', () => {
      renderHook(() => useOperationProgress('op-123', { showToasts: false }), {
        wrapper: createWrapper(queryClient),
      });

      const message = {
        type: 'operation_completed',
        data: {},
      };

      act(() => {
        mockOnMessage?.(new MessageEvent('message', { 
          data: JSON.stringify(message) 
        }));
      });

      // Should update cache but not show toast
      expect(mockUpdateOperationInCache).toHaveBeenCalled();
      expect(mockAddToast).not.toHaveBeenCalled();
    });
  });

  describe('Operation ID changes', () => {
    it('should reset completion tracking when operation changes', async () => {
      const onComplete = vi.fn();
      
      const { rerender } = renderHook(
        ({ operationId }) => useOperationProgress(operationId, { onComplete }),
        {
          wrapper: createWrapper(queryClient),
          initialProps: { operationId: 'op-123' },
        }
      );

      // Complete first operation
      const completionMessage = {
        type: 'operation_completed',
        data: {},
      };

      act(() => {
        mockOnMessage?.(new MessageEvent('message', { 
          data: JSON.stringify(completionMessage) 
        }));
      });

      expect(onComplete).toHaveBeenCalledTimes(1);

      // Change to new operation
      rerender({ operationId: 'op-456' });

      // Reset mock to capture new WebSocket setup
      vi.mocked(useWebSocket).mockClear();
      onComplete.mockClear();

      // Should be able to complete the new operation
      act(() => {
        mockOnMessage?.(new MessageEvent('message', { 
          data: JSON.stringify(completionMessage) 
        }));
      });

      // This would fail if completion tracking wasn't reset
      expect(onComplete).toHaveBeenCalledTimes(1);
    });
  });

  describe('Return values', () => {
    it('should expose sendMessage function', () => {
      const { result } = renderHook(() => useOperationProgress('op-123'), {
        wrapper: createWrapper(queryClient),
      });

      expect(result.current.sendMessage).toBe(mockSendMessage);
      
      // Test that sendMessage can be called
      result.current.sendMessage('test message');
      expect(mockSendMessage).toHaveBeenCalledWith('test message');
    });
  });
});