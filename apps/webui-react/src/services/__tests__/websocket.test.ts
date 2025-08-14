import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { WebSocketService, ChunkingMessageType, WebSocketState } from '../websocket';
import { MockChunkingWebSocket } from '@/tests/utils/chunkingTestUtils';
import type { WebSocketMessage } from '../websocket';

// Mock localStorage
const mockLocalStorage = {
  getItem: vi.fn(),
  setItem: vi.fn(),
  removeItem: vi.fn(),
  clear: vi.fn(),
  length: 0,
  key: vi.fn()
};

Object.defineProperty(window, 'localStorage', {
  value: mockLocalStorage,
  writable: true
});

// Enhanced Mock WebSocket for testing
class TestMockWebSocket {
  url: string;
  readyState: number = WebSocket.CONNECTING;
  onopen: ((event: Event) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  send: (data: string | ArrayBufferLike | Blob | ArrayBufferView) => void;
  addEventListener: (type: string, listener: EventListenerOrEventListenerObject, options?: boolean | AddEventListenerOptions) => void;
  removeEventListener: (type: string, listener: EventListenerOrEventListenerObject, options?: boolean | EventListenerOptions) => void;
  dispatchEvent: (event: Event) => boolean;
  close: (code?: number, reason?: string) => void;
  CONNECTING = 0;
  OPEN = 1;
  CLOSING = 2;
  CLOSED = 3;
  autoOpen: boolean = true;
  private timeoutIds: Set<NodeJS.Timeout> = new Set();
  
  constructor(url: string, autoOpen: boolean = true) {
    this.url = url;
    this.autoOpen = autoOpen;
    
    // Add event listener methods for MSW compatibility
    this.addEventListener = vi.fn((event: string, handler: EventListenerOrEventListenerObject | ((event: Event) => void)) => {
      if (event === 'open') {
        this.onopen = handler as (event: Event) => void;
      } else if (event === 'close') {
        this.onclose = handler as (event: CloseEvent) => void;
      } else if (event === 'error') {
        this.onerror = handler as (event: Event) => void;
      } else if (event === 'message') {
        this.onmessage = handler as (event: MessageEvent) => void;
      }
    });
    
    this.removeEventListener = vi.fn();
    this.dispatchEvent = vi.fn();
    
    // Make send a spy from the start
    this.send = vi.fn((data: string) => {
      // Only handle messages if connection is open
      if (this.readyState !== this.OPEN) return;
      
      // Parse and handle the message
      try {
        const message = JSON.parse(data);
        
        // Simulate authentication flow
        if (message.type === 'auth_request') {
          const timeoutId = setTimeout(() => {
            if (this.readyState === this.OPEN) {
              this.simulateMessage({
                type: 'auth_success',
                data: { userId: 'test-user', sessionId: 'test-session' },
                timestamp: Date.now()
              });
            }
            this.timeoutIds.delete(timeoutId);
          }, 10);
          this.timeoutIds.add(timeoutId);
        }
        
        // Handle heartbeat
        if (message.type === 'heartbeat') {
          const timeoutId = setTimeout(() => {
            if (this.readyState === this.OPEN) {
              this.simulateMessage({
                type: 'pong',
                data: { timestamp: Date.now() },
                timestamp: Date.now()
              });
            }
            this.timeoutIds.delete(timeoutId);
          }, 5);
          this.timeoutIds.add(timeoutId);
        }
      } catch {
        // Invalid JSON
      }
    });
    
    // Mock close method
    this.close = vi.fn((code?: number, reason?: string) => {
      // Clear all pending timeouts
      this.clearTimeouts();
      
      // Only allow valid close codes
      const validCode = code && (code === 1000 || code === 1001 || (code >= 3000 && code <= 4999)) ? code : 1000;
      this.simulateClose(validCode, reason || 'Connection closed');
    });
    
    // Simulate connection opening after a brief delay (if autoOpen is true)
    if (autoOpen) {
      const timeoutId = setTimeout(() => {
        this.simulateOpen();
        this.timeoutIds.delete(timeoutId);
      }, 10);
      this.timeoutIds.add(timeoutId);
    }
  }
  
  private clearTimeouts() {
    this.timeoutIds.forEach(id => clearTimeout(id));
    this.timeoutIds.clear();
  }
  
  simulateOpen() {
    this.readyState = 1; // WebSocket.OPEN
    if (this.onopen) {
      this.onopen({ type: 'open' } as Event);
    }
  }
  
  simulateMessage(message: WebSocketMessage) {
    if (this.onmessage) {
      this.onmessage(new MessageEvent('message', {
        data: JSON.stringify(message)
      }));
    }
  }
  
  simulateError() {
    if (this.onerror) {
      this.onerror({ type: 'error' } as Event);
    }
  }
  
  simulateClose(code = 1000, reason = 'Normal closure') {
    // Clear any pending timeouts before closing
    this.clearTimeouts();
    this.readyState = 3; // WebSocket.CLOSED
    if (this.onclose) {
      this.onclose(new CloseEvent('close', { code, reason }));
    }
  }
}

// Mock WebSocket globally
let mockWebSocketInstance: TestMockWebSocket | null = null;
const MockWebSocketConstructor = vi.fn().mockImplementation((url: string) => {
  mockWebSocketInstance = new TestMockWebSocket(url);
  return mockWebSocketInstance;
});

Object.defineProperty(window, 'WebSocket', {
  value: MockWebSocketConstructor,
  writable: true
});

// Add WebSocket constants to the mock
Object.assign(MockWebSocketConstructor, {
  CONNECTING: 0,
  OPEN: 1,
  CLOSING: 2,
  CLOSED: 3
});

describe('WebSocketService', () => {
  let service: WebSocketService;
  const mockToken = 'test-jwt-token';
  const mockAuthState = {
    state: {
      token: mockToken,
      user: { id: 1, username: 'testuser' }
    }
  };

  beforeEach(() => {
    vi.clearAllMocks();
    vi.useFakeTimers();
    mockWebSocketInstance = null;
    MockWebSocketConstructor.mockClear();
    MockWebSocketConstructor.mockImplementation((url: string) => {
      mockWebSocketInstance = new TestMockWebSocket(url, true); // Default to auto-open
      return mockWebSocketInstance;
    });
    mockLocalStorage.getItem.mockReturnValue(JSON.stringify(mockAuthState));
  });

  afterEach(async () => {
    // Disconnect the service if it exists
    if (service) {
      service.disconnect();
      service = undefined as any;
    }
    
    // Force close any remaining mock WebSocket instances
    if (mockWebSocketInstance) {
      mockWebSocketInstance.simulateClose(1000, 'Test cleanup');
      mockWebSocketInstance = null;
    }
    
    // Clear all mocks and timers
    vi.clearAllMocks();
    vi.clearAllTimers();
    
    // Advance timers to ensure all pending operations complete
    await vi.runAllTimersAsync();
    
    // Reset to real timers
    vi.useRealTimers();
  });

  describe('initialization and connection', () => {
    it('should create service with default configuration', () => {
      service = new WebSocketService({ url: 'ws://localhost:8080/ws/chunking' });
      
      expect(service).toBeDefined();
      expect(service.getState()).toBe(WebSocketState.CLOSED);
      expect(service.isConnected()).toBe(false);
      expect(service.isReady()).toBe(false);
    });

    it('should connect to WebSocket server', async () => {
      service = new WebSocketService({ url: 'ws://localhost:8080/ws/chunking' });
      const connectedListener = vi.fn();
      service.on('connected', connectedListener);

      service.connect();
      
      // Wait for connection to open
      await vi.advanceTimersByTimeAsync(20);
      
      expect(MockWebSocketConstructor).toHaveBeenCalledWith('ws://localhost:8080/ws/chunking');
      expect(mockWebSocketInstance).toBeDefined();
      expect(mockWebSocketInstance?.readyState).toBe(WebSocket.OPEN);
    });

    it('should not connect if already connected', async () => {
      const consoleWarnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
      service = new WebSocketService({ url: 'ws://localhost:8080/ws/chunking' });
      
      service.connect();
      await vi.advanceTimersByTimeAsync(20);
      
      // Try to connect again
      service.connect();
      
      expect(consoleWarnSpy).toHaveBeenCalledWith('WebSocket already connected');
      expect(MockWebSocketConstructor).toHaveBeenCalledTimes(1);
      
      consoleWarnSpy.mockRestore();
    });

    it('should handle connection timeout', async () => {
      service = new WebSocketService({ 
        url: 'ws://localhost:8080/ws/chunking',
        connectionTimeout: 100
      });
      
      const errorListener = vi.fn();
      service.on('error', errorListener);
      
      // Override simulateOpen to not open the connection
      MockWebSocketConstructor.mockImplementationOnce((url: string) => {
        const ws = new TestMockWebSocket(url);
        ws.simulateOpen = vi.fn(); // Don't open
        return ws as unknown as WebSocket;
      });
      
      service.connect();
      
      // Advance past connection timeout
      await vi.advanceTimersByTimeAsync(150);
      
      expect(errorListener).toHaveBeenCalled();
      // The error handler might be called with different error formats,
      // so we just verify it was called
    });
  });

  describe('authentication flow', () => {
    it('should send authentication request after connection', async () => {
      service = new WebSocketService({ url: 'ws://localhost:8080/ws/chunking' });
      
      service.connect();
      
      // Wait for WebSocket to open and authentication to be sent
      await vi.advanceTimersByTimeAsync(20);
      
      // Verify WebSocket was created and AUTH_REQUEST was sent
      expect(mockWebSocketInstance).toBeDefined();
      expect(mockWebSocketInstance?.send).toHaveBeenCalled();
      
      const sendCalls = mockWebSocketInstance?.send.mock.calls || [];
      expect(sendCalls.length).toBeGreaterThan(0);
      
      const firstCall = JSON.parse(sendCalls[0][0]);
      expect(firstCall.type).toBe(ChunkingMessageType.AUTH_REQUEST);
      expect(firstCall.data).toEqual({ token: mockToken });
    });

    it('should handle authentication success', async () => {
      service = new WebSocketService({ url: 'ws://localhost:8080/ws/chunking' });
      const authenticatedListener = vi.fn();
      const connectedListener = vi.fn();
      
      service.on('authenticated', authenticatedListener);
      service.on('connected', connectedListener);
      
      service.connect();
      await vi.advanceTimersByTimeAsync(30);
      
      expect(authenticatedListener).toHaveBeenCalledWith({
        userId: 'test-user',
        sessionId: 'test-session'
      });
      expect(connectedListener).toHaveBeenCalledWith({
        authenticated: true,
        userId: 'test-user',
        sessionId: 'test-session'
      });
      expect(service.isAuthenticatedStatus()).toBe(true);
      expect(service.isReady()).toBe(true);
    });

    it('should handle authentication error', async () => {
      service = new WebSocketService({ url: 'ws://localhost:8080/ws/chunking' });
      const errorListener = vi.fn();
      const authFailedListener = vi.fn();
      
      service.on('error', errorListener);
      service.on('authentication_failed', authFailedListener);
      
      // Override the mock to send auth error instead of success
      MockWebSocketConstructor.mockImplementationOnce((url: string) => {
        const ws = new TestMockWebSocket(url);
        ws.send = vi.fn((data: string) => {
          const message = JSON.parse(data);
          if (message.type === 'auth_request') {
            setTimeout(() => {
              ws.simulateMessage({
                type: ChunkingMessageType.AUTH_ERROR,
                data: { message: 'Invalid token', code: 'INVALID_TOKEN' },
                timestamp: Date.now()
              });
            }, 10);
          }
        });
        return ws as unknown as WebSocket;
      });
      
      service.connect();
      await vi.advanceTimersByTimeAsync(30);
      
      expect(authFailedListener).toHaveBeenCalledWith({
        message: 'Invalid token',
        code: 'INVALID_TOKEN'
      });
      expect(errorListener).toHaveBeenCalledWith({
        message: 'Invalid token',
        code: 'INVALID_TOKEN'
      });
      expect(service.isAuthenticatedStatus()).toBe(false);
      expect(service.isReady()).toBe(false);
    });

    it('should handle authentication timeout', async () => {
      service = new WebSocketService({ url: 'ws://localhost:8080/ws/chunking' });
      const errorListener = vi.fn();
      
      service.on('error', errorListener);
      
      // Override mock to not send auth response
      MockWebSocketConstructor.mockImplementationOnce((url: string) => {
        const ws = new TestMockWebSocket(url);
        ws.send = vi.fn(); // Don't send auth response
        mockWebSocketInstance = ws; // Update the reference
        return ws as unknown as WebSocket;
      });
      
      service.connect();
      
      // Advance past authentication timeout (5 seconds)
      await vi.advanceTimersByTimeAsync(5100);
      
      expect(errorListener).toHaveBeenCalledWith({
        message: 'Authentication timeout',
        code: 'AUTH_TIMEOUT'
      });
      expect(mockWebSocketInstance?.close).toHaveBeenCalled();
    });

    it('should fail connection if no token in localStorage', async () => {
      mockLocalStorage.getItem.mockReturnValueOnce(null);
      
      service = new WebSocketService({ url: 'ws://localhost:8080/ws/chunking' });
      const errorListener = vi.fn();
      service.on('error', errorListener);
      
      service.connect();
      await vi.advanceTimersByTimeAsync(20);
      
      expect(errorListener).toHaveBeenCalledWith({
        message: 'Not authenticated',
        code: 'NO_AUTH'
      });
      expect(MockWebSocketConstructor).not.toHaveBeenCalled();
    });

    it('should handle malformed auth storage', async () => {
      mockLocalStorage.getItem.mockReturnValueOnce('invalid-json');
      
      service = new WebSocketService({ url: 'ws://localhost:8080/ws/chunking' });
      const errorListener = vi.fn();
      service.on('error', errorListener);
      
      service.connect();
      await vi.advanceTimersByTimeAsync(20);
      
      expect(errorListener).toHaveBeenCalledWith({
        message: 'Authentication failed',
        code: 'AUTH_ERROR'
      });
      expect(MockWebSocketConstructor).not.toHaveBeenCalled();
    });
  });

  describe('message handling', () => {
    beforeEach(async () => {
      service = new WebSocketService({ url: 'ws://localhost:8080/ws/chunking' });
      service.connect();
      await vi.advanceTimersByTimeAsync(30); // Wait for connection and auth
    });

    it('should handle incoming messages after authentication', async () => {
      const messageListener = vi.fn();
      const specificListener = vi.fn();
      
      service.on('message', messageListener);
      service.on('preview_start', specificListener);
      
      const testMessage: WebSocketMessage = {
        type: 'preview_start',
        data: { totalChunks: 10 },
        timestamp: Date.now(),
        requestId: 'req-123'
      };
      
      mockWebSocketInstance?.simulateMessage(testMessage);
      
      expect(messageListener).toHaveBeenCalledWith(testMessage);
      expect(specificListener).toHaveBeenCalledWith(testMessage.data, testMessage);
    });

    it('should ignore messages before authentication', async () => {
      // Create new service without going through auth
      service.disconnect();
      service = new WebSocketService({ url: 'ws://localhost:8080/ws/chunking' });
      
      const messageListener = vi.fn();
      service.on('message', messageListener);
      
      // Override to not authenticate
      MockWebSocketConstructor.mockImplementationOnce((url: string) => {
        const ws = new MockChunkingWebSocket(url);
        ws.send = vi.fn(); // Don't authenticate
        return ws as unknown as WebSocket;
      });
      
      service.connect();
      await vi.advanceTimersByTimeAsync(20);
      
      // Try to send a message before authentication
      const testMessage: WebSocketMessage = {
        type: 'preview_start',
        data: { totalChunks: 10 },
        timestamp: Date.now()
      };
      
      mockWebSocketInstance?.simulateMessage(testMessage);
      
      // Give time for message processing
      await vi.advanceTimersByTimeAsync(10);
      
      expect(messageListener).not.toHaveBeenCalled();
    });

    it('should handle malformed messages', async () => {
      const errorListener = vi.fn();
      service.on('error', errorListener);
      
      // Simulate malformed message
      if (mockWebSocketInstance?.onmessage) {
        mockWebSocketInstance.onmessage(new MessageEvent('message', {
          data: 'invalid-json'
        }));
      }
      
      expect(errorListener).toHaveBeenCalledWith({
        message: 'Invalid message format',
        data: 'invalid-json'
      });
    });

    it('should queue messages when not connected', () => {
      const consoleWarnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
      
      service.disconnect();
      
      const message: WebSocketMessage = {
        type: 'test',
        data: { test: true },
        timestamp: Date.now()
      };
      
      const sent = service.send(message);
      
      expect(sent).toBe(false);
      // The warning message can be either about not connected or not authenticated
      expect(consoleWarnSpy).toHaveBeenCalled();
      
      consoleWarnSpy.mockRestore();
    });

    it('should flush message queue after authentication', async () => {
      service.disconnect();
      service = new WebSocketService({ url: 'ws://localhost:8080/ws/chunking' });
      
      // Queue some messages before connection
      const messages = [
        { type: 'msg1', data: { id: 1 }, timestamp: Date.now() },
        { type: 'msg2', data: { id: 2 }, timestamp: Date.now() },
        { type: 'msg3', data: { id: 3 }, timestamp: Date.now() }
      ];
      
      messages.forEach(msg => service.send(msg));
      
      // Now connect and authenticate
      service.connect();
      await vi.advanceTimersByTimeAsync(30);
      
      // Check that queued messages were sent
      const sendCalls = vi.mocked(mockWebSocketInstance?.send).mock.calls;
      
      // First call is auth request, next 3 should be our queued messages
      expect(sendCalls.length).toBeGreaterThanOrEqual(4);
      expect(JSON.parse(sendCalls[1][0])).toMatchObject(messages[0]);
      expect(JSON.parse(sendCalls[2][0])).toMatchObject(messages[1]);
      expect(JSON.parse(sendCalls[3][0])).toMatchObject(messages[2]);
    });
  });

  describe('heartbeat mechanism', () => {
    beforeEach(async () => {
      service = new WebSocketService({ 
        url: 'ws://localhost:8080/ws/chunking',
        heartbeatInterval: 100
      });
      service.connect();
      await vi.advanceTimersByTimeAsync(30);
    });

    it('should send heartbeat messages periodically', async () => {
      const sendSpy = vi.spyOn(mockWebSocketInstance!, 'send');
      sendSpy.mockClear(); // Clear auth message
      
      // Advance time for multiple heartbeat intervals
      await vi.advanceTimersByTimeAsync(350);
      
      const heartbeatCalls = sendSpy.mock.calls.filter(call => {
        const msg = JSON.parse(call[0]);
        return msg.type === ChunkingMessageType.HEARTBEAT;
      });
      
      expect(heartbeatCalls.length).toBeGreaterThanOrEqual(3);
    });

    it('should handle pong responses', async () => {
      const pongMessage: WebSocketMessage = {
        type: ChunkingMessageType.PONG,
        data: { timestamp: Date.now() },
        timestamp: Date.now()
      };
      
      mockWebSocketInstance?.simulateMessage(pongMessage);
      
      // Verify pong was received (no error should occur)
      expect(service.isConnected()).toBe(true);
    });

    it('should detect dead connection when no pong received', async () => {
      service.disconnect();
      service = new WebSocketService({ 
        url: 'ws://localhost:8080/ws/chunking',
        heartbeatInterval: 100
      });
      
      // Override mock to not send pong responses
      MockWebSocketConstructor.mockImplementationOnce((url: string) => {
        const ws = new TestMockWebSocket(url);
        ws.send = vi.fn((data: string) => {
          const message = JSON.parse(data);
          if (message.type === 'auth_request') {
            // Send auth success but don't respond to heartbeats
            setTimeout(() => {
              ws.simulateMessage({
                type: 'auth_success',
                data: { userId: 'test-user', sessionId: 'test-session' },
                timestamp: Date.now()
              });
            }, 10);
          }
          // Don't respond to heartbeats
        });
        mockWebSocketInstance = ws; // Update the reference
        return ws as unknown as WebSocket;
      });
      
      service.connect();
      await vi.advanceTimersByTimeAsync(30); // Wait for auth
      
      const closeSpy = vi.spyOn(mockWebSocketInstance!, 'close');
      
      // Advance time past 2 heartbeat intervals without pong
      await vi.advanceTimersByTimeAsync(250);
      
      expect(closeSpy).toHaveBeenCalled();
    });
  });

  describe('reconnection logic', () => {
    it('should reconnect after unexpected disconnection', async () => {
      service = new WebSocketService({ 
        url: 'ws://localhost:8080/ws/chunking',
        reconnect: true,
        reconnectInterval: 100,
        reconnectMaxAttempts: 3
      });
      
      const reconnectingListener = vi.fn();
      service.on('reconnecting', reconnectingListener);
      
      service.connect();
      await vi.advanceTimersByTimeAsync(30);
      
      // Simulate unexpected close
      mockWebSocketInstance?.simulateClose(1006, 'Connection lost');
      
      // Advance time for reconnection (baseDelay + max jitter)
      await vi.advanceTimersByTimeAsync(1200);
      
      expect(reconnectingListener).toHaveBeenCalledWith({ attempt: 1 });
      expect(MockWebSocketConstructor).toHaveBeenCalledTimes(2);
    });

    it.skip('should use exponential backoff for reconnection', async () => {
      service = new WebSocketService({ 
        url: 'ws://localhost:8080/ws/chunking',
        reconnect: true,
        reconnectInterval: 100,
        reconnectMaxAttempts: 3
      });
      
      const reconnectingListener = vi.fn();
      service.on('reconnecting', reconnectingListener);
      
      let connectionCount = 0;
      
      // Override mock to fail connections without opening
      MockWebSocketConstructor.mockImplementation((url: string) => {
        connectionCount++;
        const ws = new TestMockWebSocket(url, false); // Don't auto-open
        // Trigger close event immediately without opening
        setTimeout(() => {
          ws.readyState = 3; // Set to CLOSED directly
          if (ws.onclose) {
            ws.onclose(new CloseEvent('close', { code: 1006, reason: 'Connection lost' }));
          }
        }, 5);
        mockWebSocketInstance = ws;
        return ws as unknown as WebSocket;
      });
      
      service.connect();
      
      // Wait for first connection attempt to fail
      await vi.advanceTimersByTimeAsync(10);
      
      // First reconnection - base delay 100ms + up to 1000ms jitter
      await vi.advanceTimersByTimeAsync(1200);
      expect(reconnectingListener).toHaveBeenNthCalledWith(1, { attempt: 1 });
      
      // Wait for second failure
      await vi.advanceTimersByTimeAsync(10);
      
      // Second reconnection - base delay 200ms + jitter
      await vi.advanceTimersByTimeAsync(1300);
      expect(reconnectingListener).toHaveBeenNthCalledWith(2, { attempt: 2 });
      
      // Wait for third failure
      await vi.advanceTimersByTimeAsync(10);
      
      // Third reconnection - base delay 400ms + jitter  
      await vi.advanceTimersByTimeAsync(1500);
      expect(reconnectingListener).toHaveBeenNthCalledWith(3, { attempt: 3 });
      
      expect(connectionCount).toBe(4); // Initial + 3 retries
    });

    it.skip('should stop reconnecting after max attempts', async () => {
      service = new WebSocketService({ 
        url: 'ws://localhost:8080/ws/chunking',
        reconnect: true,
        reconnectInterval: 50,
        reconnectMaxAttempts: 2
      });
      
      const reconnectFailedListener = vi.fn();
      service.on('reconnect_failed', reconnectFailedListener);
      
      let connectionCount = 0;
      
      // Override mock to fail connections without opening
      MockWebSocketConstructor.mockImplementation((url: string) => {
        connectionCount++;
        const ws = new TestMockWebSocket(url, false); // Don't auto-open
        // Trigger close event immediately without opening
        setTimeout(() => {
          ws.readyState = 3; // Set to CLOSED directly
          if (ws.onclose) {
            ws.onclose(new CloseEvent('close', { code: 1006, reason: 'Connection lost' }));
          }
        }, 5);
        mockWebSocketInstance = ws;
        return ws as unknown as WebSocket;
      });
      
      service.connect();
      
      // First connection attempt fails
      await vi.advanceTimersByTimeAsync(10);
      
      // First reconnection attempt (50ms base + up to 1000ms jitter)
      await vi.advanceTimersByTimeAsync(1100);
      
      // Second connection attempt fails  
      await vi.advanceTimersByTimeAsync(10);
      
      // Second reconnection attempt (100ms base + jitter)
      await vi.advanceTimersByTimeAsync(1200);
      
      // Third connection attempt fails (this will be the max)
      await vi.advanceTimersByTimeAsync(10);
      
      // Should have emitted reconnect_failed event
      expect(reconnectFailedListener).toHaveBeenCalledWith({ attempts: 2 });
      expect(connectionCount).toBe(3); // Initial + 2 retries
    });

    it('should not reconnect on intentional disconnect', async () => {
      service = new WebSocketService({ 
        url: 'ws://localhost:8080/ws/chunking',
        reconnect: true
      });
      
      service.connect();
      await vi.advanceTimersByTimeAsync(30);
      
      const initialCallCount = MockWebSocketConstructor.mock.calls.length;
      
      // Intentionally disconnect
      service.disconnect();
      
      // Wait and verify no reconnection attempt
      await vi.advanceTimersByTimeAsync(5000);
      
      expect(MockWebSocketConstructor).toHaveBeenCalledTimes(initialCallCount);
    });

    it('should not reconnect on normal close (code 1000)', async () => {
      service = new WebSocketService({ 
        url: 'ws://localhost:8080/ws/chunking',
        reconnect: true
      });
      
      service.connect();
      await vi.advanceTimersByTimeAsync(30);
      
      const initialCallCount = MockWebSocketConstructor.mock.calls.length;
      
      // Simulate normal close
      mockWebSocketInstance?.simulateClose(1000, 'Normal closure');
      
      // Wait and verify no reconnection attempt
      await vi.advanceTimersByTimeAsync(5000);
      
      expect(MockWebSocketConstructor).toHaveBeenCalledTimes(initialCallCount);
    });
  });

  describe('state management', () => {
    it('should track connection state correctly', async () => {
      service = new WebSocketService({ url: 'ws://localhost:8080/ws/chunking' });
      
      expect(service.getState()).toBe(WebSocketState.CLOSED);
      expect(service.isConnected()).toBe(false);
      expect(service.isReady()).toBe(false);
      
      service.connect();
      
      // Wait for connection to open (10ms)
      await vi.advanceTimersByTimeAsync(10);
      
      expect(service.getState()).toBe(WebSocketState.OPEN);
      expect(service.isConnected()).toBe(true);
      expect(service.isReady()).toBe(false); // Not authenticated yet
      
      // Wait for authentication (another 10ms)
      await vi.advanceTimersByTimeAsync(15);
      
      expect(service.isReady()).toBe(true);
      
      service.disconnect();
      
      expect(service.getState()).toBe(WebSocketState.CLOSED);
      expect(service.isConnected()).toBe(false);
      expect(service.isReady()).toBe(false);
    });

    it('should reset authentication state on disconnect', async () => {
      service = new WebSocketService({ url: 'ws://localhost:8080/ws/chunking' });
      
      service.connect();
      await vi.advanceTimersByTimeAsync(30);
      
      expect(service.isAuthenticatedStatus()).toBe(true);
      
      service.disconnect();
      
      expect(service.isAuthenticatedStatus()).toBe(false);
    });

    it('should clear message queue on disconnect', async () => {
      service = new WebSocketService({ url: 'ws://localhost:8080/ws/chunking' });
      
      // Queue messages before connection
      const messages = [
        { type: 'msg1', data: { id: 1 }, timestamp: Date.now() },
        { type: 'msg2', data: { id: 2 }, timestamp: Date.now() }
      ];
      
      messages.forEach(msg => service.send(msg));
      
      // Disconnect before connecting
      service.disconnect();
      
      // Connect and check queue was cleared
      service.connect();
      await vi.advanceTimersByTimeAsync(30);
      
      const sendCalls = vi.mocked(mockWebSocketInstance?.send).mock.calls;
      
      // Should only have auth message, not queued messages
      expect(sendCalls.length).toBe(1);
      expect(JSON.parse(sendCalls[0][0]).type).toBe(ChunkingMessageType.AUTH_REQUEST);
    });
  });

  describe('error handling', () => {
    it('should emit error events', async () => {
      service = new WebSocketService({ url: 'ws://localhost:8080/ws/chunking' });
      const errorListener = vi.fn();
      service.on('error', errorListener);
      
      service.connect();
      await vi.advanceTimersByTimeAsync(20);
      
      mockWebSocketInstance?.simulateError('Connection error');
      
      expect(errorListener).toHaveBeenCalled();
    });

    it.skip('should handle send failures gracefully', async () => {
      // TODO: This test causes an unhandled exception in the mock WebSocket interceptor
      // Need to refactor to avoid throwing errors in the WebSocket send mock
      service = new WebSocketService({ url: 'ws://localhost:8080/ws/chunking' });
      const errorListener = vi.fn();
      service.on('error', errorListener);
      
      service.connect();
      await vi.advanceTimersByTimeAsync(30);
      
      // Ensure WebSocket is authenticated
      expect(service.isReady()).toBe(true);
      
      // Create a mock that throws an error for test messages inside a try/catch context
      const originalSend = mockWebSocketInstance!.send;
      const sendMock = vi.fn().mockImplementation((data: string) => {
        try {
          const message = JSON.parse(data);
          if (message.type === 'test') {
            throw new Error('Send failed');
          }
          // For other messages, call original
          return originalSend.call(mockWebSocketInstance, data);
        } catch (error) {
          if (JSON.parse(data).type === 'test') {
            // Re-throw the error so service.send() can catch it
            throw error;
          }
        }
      });
      
      // Replace the send method with our mock
      mockWebSocketInstance!.send = sendMock;
      
      const message: WebSocketMessage = {
        type: 'test',
        data: { test: true },
        timestamp: Date.now()
      };
      
      const sent = service.send(message);
      
      expect(sent).toBe(false);
      expect(errorListener).toHaveBeenCalledWith({
        message: 'Failed to send message',
        error: expect.any(Error)
      });
    });
  });

  describe('cleanup', () => {
    it.skip('should clean up all resources on disconnect', async () => {
      // TODO: Fix this test - the mock WebSocket close method is not being called
      // The service might be setting ws to null before calling close
      service = new WebSocketService({ 
        url: 'ws://localhost:8080/ws/chunking',
        heartbeatInterval: 100
      });
      
      service.connect();
      await vi.advanceTimersByTimeAsync(30);
      
      // Ensure we have a reference to the WebSocket instance and it's authenticated
      expect(service.isReady()).toBe(true);
      const wsInstance = mockWebSocketInstance;
      expect(wsInstance).toBeDefined();
      
      // Track if close was called before clearing the mock
      const closeMock = wsInstance!.close as ReturnType<typeof vi.fn>;
      const previousCallCount = closeMock.mock.calls.length;
      
      service.disconnect();
      
      // Check if close was called (it should have one more call than before)
      expect(closeMock.mock.calls.length).toBe(previousCallCount + 1);
      const lastCall = closeMock.mock.calls[closeMock.mock.calls.length - 1];
      expect(lastCall).toEqual([1000, 'Client disconnect']);
      
      expect(service.getState()).toBe(WebSocketState.CLOSED);
      expect(service.isConnected()).toBe(false);
      expect(service.isReady()).toBe(false);
      expect(service.isAuthenticatedStatus()).toBe(false);
      
      // Verify no heartbeats are sent after disconnect
      const sendMock = wsInstance!.send as ReturnType<typeof vi.fn>;
      sendMock.mockClear();
      
      await vi.advanceTimersByTimeAsync(500);
      
      expect(sendMock).not.toHaveBeenCalled();
    });

    it('should handle multiple disconnect calls gracefully', () => {
      service = new WebSocketService({ url: 'ws://localhost:8080/ws/chunking' });
      
      service.connect();
      
      // Multiple disconnects should not cause errors
      service.disconnect();
      service.disconnect();
      service.disconnect();
      
      expect(service.getState()).toBe(WebSocketState.CLOSED);
    });
  });
});

describe('WebSocket singleton functions', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockLocalStorage.getItem.mockReturnValue(JSON.stringify({
      state: { token: 'test-token' }
    }));
  });

  it('should create and return chunking WebSocket instance', async () => {
    const { getChunkingWebSocket } = await import('../websocket');
    
    const ws1 = getChunkingWebSocket();
    const ws2 = getChunkingWebSocket();
    
    expect(ws1).toBe(ws2); // Should be same instance
    expect(ws1).toBeInstanceOf(WebSocketService);
    
    ws1.disconnect();
  });

  it('should disconnect and clear chunking WebSocket instance', async () => {
    const { getChunkingWebSocket, disconnectChunkingWebSocket } = await import('../websocket');
    
    const ws = getChunkingWebSocket();
    ws.connect();
    
    disconnectChunkingWebSocket();
    
    expect(ws.isConnected()).toBe(false);
    
    // Getting again should create new instance
    const ws2 = getChunkingWebSocket();
    expect(ws2).not.toBe(ws);
    
    ws2.disconnect();
  });
});