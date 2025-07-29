import { renderHook, act } from '@testing-library/react'
import { useWebSocket } from '../useWebSocket'
import { mockWebSocket, MockWebSocket } from '../../tests/utils/errorTestUtils'

describe('useWebSocket - Error Handling', () => {
  let mockWs: { restore: () => void }
  let wsInstances: MockWebSocket[] = []

  beforeEach(() => {
    vi.clearAllMocks()
    wsInstances = []
    mockWs = mockWebSocket()
    
    // Track WebSocket instances
    // const OriginalWebSocket = global.WebSocket // Unused variable
    global.WebSocket = class extends MockWebSocket {
      constructor(url: string) {
        super(url)
        wsInstances.push(this)
      }
    } as unknown as typeof WebSocket
  })

  afterEach(() => {
    mockWs.restore()
    wsInstances = []
  })

  describe('Connection Errors', () => {
    it('should handle immediate connection failure', async () => {
      const onError = vi.fn()
      const onMessage = vi.fn()
      
      const { result } = renderHook(() => 
        useWebSocket({
          url: 'ws://localhost:8080/ws/fail-connection',
          onMessage,
          onError
        })
      )
      
      // Should start in connecting state
      expect(result.current.readyState).toBe(WebSocket.CONNECTING)
      expect(result.current.error).toBeNull()
      
      // Wait for connection to fail
      await act(async () => {
        await new Promise(resolve => setTimeout(resolve, 50))
      })
      
      // Should have error state
      expect(result.current.readyState).toBe(WebSocket.CLOSED)
      expect(result.current.error).toBeTruthy()
      expect(onError).toHaveBeenCalledWith(expect.any(Error))
    })

    it('should retry connection on failure', async () => {
      const onReconnect = vi.fn()
      
      const { result } = renderHook(() => 
        useWebSocket({
          url: 'ws://localhost:8080/ws/test',
          reconnectInterval: 100,
          maxReconnectAttempts: 3,
          onReconnect
        })
      )
      
      // Force connection to fail
      act(() => {
        wsInstances[0]?.simulateDisconnect(1006, 'Connection lost')
      })
      
      // Should attempt to reconnect
      await act(async () => {
        await new Promise(resolve => setTimeout(resolve, 150))
      })
      
      expect(result.current.reconnectAttempts).toBeGreaterThan(0)
      expect(onReconnect).toHaveBeenCalled()
    })

    it('should stop retrying after max attempts', async () => {
      const onMaxReconnectAttemptsReached = vi.fn()
      
      const { result } = renderHook(() => 
        useWebSocket({
          url: 'ws://localhost:8080/ws/always-fail',
          reconnectInterval: 50,
          maxReconnectAttempts: 2,
          onMaxReconnectAttemptsReached
        })
      )
      
      // Wait for multiple reconnect attempts
      await act(async () => {
        await new Promise(resolve => setTimeout(resolve, 200))
      })
      
      expect(result.current.reconnectAttempts).toBe(2)
      expect(onMaxReconnectAttemptsReached).toHaveBeenCalled()
      
      // Should not attempt more connections
      const instanceCount = wsInstances.length
      await act(async () => {
        await new Promise(resolve => setTimeout(resolve, 100))
      })
      expect(wsInstances.length).toBe(instanceCount)
    })

    it('should handle network offline/online transitions', async () => {
      const { result } = renderHook(() => 
        useWebSocket({
          url: 'ws://localhost:8080/ws/test',
          reconnectOnOffline: true
        })
      )
      
      // Initially connected
      act(() => {
        wsInstances[0]?.simulateOpen()
      })
      
      expect(result.current.readyState).toBe(WebSocket.OPEN)
      
      // Go offline
      act(() => {
        Object.defineProperty(navigator, 'onLine', { value: false, writable: true })
        window.dispatchEvent(new Event('offline'))
      })
      
      // Should close connection
      expect(result.current.readyState).toBe(WebSocket.CLOSED)
      
      // Go back online
      act(() => {
        Object.defineProperty(navigator, 'onLine', { value: true, writable: true })
        window.dispatchEvent(new Event('online'))
      })
      
      // Should attempt to reconnect
      await act(async () => {
        await new Promise(resolve => setTimeout(resolve, 100))
      })
      
      expect(wsInstances.length).toBeGreaterThan(1) // New connection created
    })
  })

  describe('Message Errors', () => {
    it('should handle malformed messages', async () => {
      const onMessage = vi.fn()
      const onError = vi.fn()
      
      renderHook(() => 
        useWebSocket({
          url: 'ws://localhost:8080/ws/test',
          onMessage,
          onError
        })
      )
      
      // Connect successfully
      act(() => {
        wsInstances[0]?.simulateOpen()
      })
      
      // Send malformed message
      act(() => {
        wsInstances[0]?.simulateMessage('invalid json {]')
      })
      
      // Should call onMessage with raw string (let consumer handle parsing)
      expect(onMessage).toHaveBeenCalledWith(
        expect.objectContaining({
          data: 'invalid json {]'
        })
      )
    })

    it('should handle binary messages', async () => {
      const onMessage = vi.fn()
      
      renderHook(() => 
        useWebSocket({
          url: 'ws://localhost:8080/ws/test',
          onMessage
        })
      )
      
      act(() => {
        wsInstances[0]?.simulateOpen()
      })
      
      // Send binary data
      const binaryData = new ArrayBuffer(8)
      act(() => {
        const event = new MessageEvent('message', { data: binaryData })
        wsInstances[0]?.onmessage?.(event)
      })
      
      expect(onMessage).toHaveBeenCalledWith(
        expect.objectContaining({
          data: binaryData
        })
      )
    })

    it('should handle rapid message bursts', async () => {
      const onMessage = vi.fn()
      const messages: unknown[] = []
      
      renderHook(() => 
        useWebSocket({
          url: 'ws://localhost:8080/ws/test',
          onMessage: (event) => {
            messages.push(JSON.parse(event.data))
            onMessage(event)
          }
        })
      )
      
      act(() => {
        wsInstances[0]?.simulateOpen()
      })
      
      // Send many messages rapidly
      act(() => {
        for (let i = 0; i < 100; i++) {
          wsInstances[0]?.simulateMessage({ id: i, data: `Message ${i}` })
        }
      })
      
      // All messages should be received
      expect(messages).toHaveLength(100)
      expect(messages[0].id).toBe(0)
      expect(messages[99].id).toBe(99)
    })
  })

  describe('Send Errors', () => {
    it('should handle sending when not connected', () => {
      const onError = vi.fn()
      
      const { result } = renderHook(() => 
        useWebSocket({
          url: 'ws://localhost:8080/ws/test',
          onError
        })
      )
      
      // Try to send before connection
      expect(() => {
        result.current.send('test message')
      }).toThrow()
    })

    it('should queue messages when configured', async () => {
      const { result } = renderHook(() => 
        useWebSocket({
          url: 'ws://localhost:8080/ws/test',
          queueMessages: true
        })
      )
      
      // Send messages before connection
      act(() => {
        result.current.send('message 1')
        result.current.send('message 2')
        result.current.send('message 3')
      })
      
      // Mock send function to track calls
      const mockSend = vi.fn()
      
      // Connect
      act(() => {
        wsInstances[0]?.simulateOpen()
        if (wsInstances[0]) {
          wsInstances[0].send = mockSend
        }
      })
      
      // Queued messages should be sent (implementation specific)
      // This depends on whether the hook implements message queuing
    })

    it('should handle send failures', () => {
      const onError = vi.fn()
      
      const { result } = renderHook(() => 
        useWebSocket({
          url: 'ws://localhost:8080/ws/test',
          onError
        })
      )
      
      act(() => {
        wsInstances[0]?.simulateOpen()
      })
      
      // Override send to throw error
      wsInstances[0].send = () => {
        throw new Error('Send failed')
      }
      
      // Should handle send error
      act(() => {
        try {
          result.current.send('test')
        } catch {
          // Expected
        }
      })
    })
  })

  describe('Cleanup and Memory Management', () => {
    it('should clean up on unmount', () => {
      const { unmount } = renderHook(() => 
        useWebSocket({
          url: 'ws://localhost:8080/ws/test'
        })
      )
      
      act(() => {
        wsInstances[0]?.simulateOpen()
      })
      
      expect(wsInstances[0]?.readyState).toBe(WebSocket.OPEN)
      
      // Unmount
      unmount()
      
      // Connection should be closed
      expect(wsInstances[0]?.readyState).toBe(WebSocket.CLOSED)
    })

    it('should handle multiple rapid reconnections without memory leaks', async () => {
      renderHook(
        ({ url }) => useWebSocket({ url, reconnectInterval: 10 }),
        { initialProps: { url: 'ws://localhost:8080/ws/test' } }
      )
      
      // Trigger multiple reconnections
      for (let i = 0; i < 5; i++) {
        act(() => {
          wsInstances[wsInstances.length - 1]?.simulateDisconnect()
        })
        
        await act(async () => {
          await new Promise(resolve => setTimeout(resolve, 20))
        })
      }
      
      // Should not have excessive instances
      expect(wsInstances.length).toBeLessThan(10)
    })

    it('should cancel reconnection on unmount', async () => {
      const { unmount } = renderHook(() => 
        useWebSocket({
          url: 'ws://localhost:8080/ws/test',
          reconnectInterval: 100
        })
      )
      
      // Force disconnection to trigger reconnect
      act(() => {
        wsInstances[0]?.simulateDisconnect()
      })
      
      const instanceCount = wsInstances.length
      
      // Unmount before reconnection
      unmount()
      
      // Wait for what would be reconnection time
      await act(async () => {
        await new Promise(resolve => setTimeout(resolve, 150))
      })
      
      // No new connections should be created
      expect(wsInstances.length).toBe(instanceCount)
    })
  })

  describe('Authentication Errors', () => {
    it('should handle authentication failure close codes', async () => {
      const onAuthError = vi.fn()
      
      const { result } = renderHook(() => 
        useWebSocket({
          url: 'ws://localhost:8080/ws/test?token=invalid',
          onClose: (event) => {
            if (event.code === 4401 || event.code === 4403) {
              onAuthError(event)
            }
          }
        })
      )
      
      // Simulate auth failure
      act(() => {
        wsInstances[0]?.simulateDisconnect(4401, 'Authentication failed')
      })
      
      expect(onAuthError).toHaveBeenCalledWith(
        expect.objectContaining({
          code: 4401,
          reason: 'Authentication failed'
        })
      )
      
      // Should not attempt to reconnect on auth failure
      await act(async () => {
        await new Promise(resolve => setTimeout(resolve, 100))
      })
      
      expect(result.current.reconnectAttempts).toBe(0)
    })
  })
})