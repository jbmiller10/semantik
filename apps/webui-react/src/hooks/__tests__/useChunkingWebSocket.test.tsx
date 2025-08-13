import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, act, waitFor } from '@testing-library/react'
import { useChunkingWebSocket } from '../useChunkingWebSocket'
import { 
  MockChunkingWebSocket, 
  simulateChunkingProgress,
  mockChunkingPreviewResponse,
  mockComparisonResults
} from '@/tests/utils/chunkingTestUtils'
import type { 
  WebSocketMessage,
  ChunkingProgressData,
  ChunkingChunkData,
  ChunkingCompleteData,
  ChunkingErrorData
} from '@/services/websocket'

// Mock the WebSocket service module
const mockWebSocketService = {
  connect: vi.fn(),
  disconnect: vi.fn(),
  on: vi.fn(),
  off: vi.fn(),
  send: vi.fn(),
  isConnected: vi.fn(() => false),
  getReconnectAttempts: vi.fn(() => 0),
}

const mockGetChunkingWebSocket = vi.fn(() => mockWebSocketService)
const mockDisconnectChunkingWebSocket = vi.fn()

vi.mock('@/services/websocket', () => ({
  getChunkingWebSocket: () => mockGetChunkingWebSocket(),
  disconnectChunkingWebSocket: () => mockDisconnectChunkingWebSocket(),
  ChunkingMessageType: {
    AUTH_REQUEST: 'auth_request',
    AUTH_SUCCESS: 'auth_success',
    AUTH_ERROR: 'auth_error',
    PREVIEW_START: 'preview_start',
    PREVIEW_PROGRESS: 'preview_progress',
    PREVIEW_CHUNK: 'preview_chunk',
    PREVIEW_COMPLETE: 'preview_complete',
    PREVIEW_ERROR: 'preview_error',
    COMPARISON_START: 'comparison_start',
    COMPARISON_PROGRESS: 'comparison_progress',
    COMPARISON_RESULT: 'comparison_result',
    COMPARISON_COMPLETE: 'comparison_complete',
    COMPARISON_ERROR: 'comparison_error',
    HEARTBEAT: 'heartbeat',
    PONG: 'pong',
  }
}))

describe('useChunkingWebSocket', () => {
  let eventHandlers: Record<string, Function> = {}

  beforeEach(() => {
    vi.clearAllMocks()
    eventHandlers = {}
    
    // Setup mock event handler registration
    mockWebSocketService.on.mockImplementation((event: string, handler: Function) => {
      eventHandlers[event] = handler
      return mockWebSocketService
    })
    
    mockWebSocketService.off.mockImplementation((event: string) => {
      delete eventHandlers[event]
      return mockWebSocketService
    })
    
    mockWebSocketService.isConnected.mockReturnValue(false)
    mockWebSocketService.getReconnectAttempts.mockReturnValue(0)
  })

  afterEach(() => {
    vi.clearAllMocks()
    eventHandlers = {}
  })

  describe('Connection Lifecycle', () => {
    it('should auto-connect when autoConnect is true', () => {
      const { result } = renderHook(() => useChunkingWebSocket({ autoConnect: true }))
      
      expect(mockGetChunkingWebSocket).toHaveBeenCalled()
      expect(mockWebSocketService.connect).toHaveBeenCalled()
      expect(result.current.connectionStatus).toBe('connecting')
    })

    it('should not auto-connect when autoConnect is false', () => {
      renderHook(() => useChunkingWebSocket({ autoConnect: false }))
      
      // WebSocket instance is still created but not connected
      expect(mockWebSocketService.connect).not.toHaveBeenCalled()
    })

    it('should connect manually when connect is called', () => {
      const { result } = renderHook(() => useChunkingWebSocket({ autoConnect: false }))
      
      act(() => {
        result.current.connect()
      })
      
      expect(mockWebSocketService.connect).toHaveBeenCalled()
    })

    it('should disconnect when disconnect is called', () => {
      const { result } = renderHook(() => useChunkingWebSocket())
      
      act(() => {
        result.current.disconnect()
      })
      
      expect(mockDisconnectChunkingWebSocket).toHaveBeenCalled()
      expect(result.current.connectionStatus).toBe('disconnected')
    })

    it('should cleanup on unmount', () => {
      const { unmount } = renderHook(() => useChunkingWebSocket())
      
      unmount()
      
      expect(mockDisconnectChunkingWebSocket).toHaveBeenCalled()
    })

    it('should handle connection state changes', () => {
      const { result } = renderHook(() => useChunkingWebSocket())
      
      // Simulate connected event
      act(() => {
        eventHandlers['connected']()
      })
      
      expect(result.current.connectionStatus).toBe('connected')
      expect(result.current.isConnected).toBe(true)
      expect(result.current.reconnectAttempts).toBe(0)
      
      // Simulate disconnected event
      act(() => {
        eventHandlers['disconnected']({ code: 1000, reason: 'Normal closure' })
      })
      
      expect(result.current.connectionStatus).toBe('disconnected')
      expect(result.current.isConnected).toBe(false)
    })

    it('should handle reconnection attempts', () => {
      const { result } = renderHook(() => useChunkingWebSocket())
      
      // Simulate reconnecting event
      act(() => {
        eventHandlers['reconnecting']({ attempt: 1 })
      })
      
      expect(result.current.connectionStatus).toBe('reconnecting')
      expect(result.current.reconnectAttempts).toBe(1)
      
      // Simulate multiple reconnection attempts
      act(() => {
        eventHandlers['reconnecting']({ attempt: 3 })
      })
      
      expect(result.current.reconnectAttempts).toBe(3)
    })

    it('should handle reconnection failure', () => {
      const { result } = renderHook(() => useChunkingWebSocket())
      
      act(() => {
        eventHandlers['reconnect_failed']({ attempts: 5 })
      })
      
      expect(result.current.connectionStatus).toBe('error')
      expect(result.current.error).toEqual({
        message: 'Failed to reconnect to server',
        code: 'RECONNECT_FAILED',
      })
    })

    it('should not reconnect on clean disconnect', () => {
      const { result } = renderHook(() => useChunkingWebSocket())
      
      // Simulate clean disconnect (code 1000)
      act(() => {
        eventHandlers['disconnected']({ code: 1000, reason: 'Normal closure' })
      })
      
      expect(result.current.progress).toBeNull()
      
      // Simulate unexpected disconnect
      act(() => {
        // Set some progress first
        eventHandlers['preview_progress'](
          { percentage: 50, currentChunk: 5, totalChunks: 10 },
          { type: 'preview_progress', data: {}, timestamp: Date.now() }
        )
      })
      
      expect(result.current.progress).not.toBeNull()
      
      act(() => {
        eventHandlers['disconnected']({ code: 1006, reason: 'Abnormal closure' })
      })
      
      // Progress should be cleared on unexpected disconnect
      expect(result.current.progress).toBeNull()
    })
  })

  describe('Message Handling', () => {
    it('should handle preview start message', () => {
      const { result } = renderHook(() => useChunkingWebSocket())
      
      act(() => {
        eventHandlers['preview_start'](
          { totalChunks: 10 },
          { type: 'preview_start', data: { totalChunks: 10 }, timestamp: Date.now() }
        )
      })
      
      expect(result.current.chunks).toEqual([])
      expect(result.current.progress).toBeNull()
      expect(result.current.statistics).toBeNull()
      expect(result.current.error).toBeNull()
    })

    it('should handle preview progress message', () => {
      const onProgressUpdate = vi.fn()
      const { result } = renderHook(() => 
        useChunkingWebSocket({ onProgressUpdate })
      )
      
      const progressData: ChunkingProgressData = {
        percentage: 50,
        currentChunk: 5,
        totalChunks: 10,
        estimatedTimeRemaining: 30,
      }
      
      act(() => {
        eventHandlers['preview_progress'](
          progressData,
          { type: 'preview_progress', data: progressData, timestamp: Date.now() }
        )
      })
      
      expect(result.current.progress).toEqual(progressData)
      expect(onProgressUpdate).toHaveBeenCalledWith(progressData)
    })

    it('should accumulate chunks as they arrive', () => {
      const onChunkReceived = vi.fn()
      const { result } = renderHook(() => 
        useChunkingWebSocket({ onChunkReceived })
      )
      
      const chunk1 = mockChunkingPreviewResponse.chunks[0]
      const chunk2 = mockChunkingPreviewResponse.chunks[1]
      
      // Receive first chunk
      act(() => {
        eventHandlers['preview_chunk'](
          { chunk: chunk1, index: 0, total: 2 },
          { type: 'preview_chunk', data: { chunk: chunk1, index: 0, total: 2 }, timestamp: Date.now() }
        )
      })
      
      expect(result.current.chunks).toHaveLength(1)
      expect(result.current.chunks[0]).toEqual(chunk1)
      expect(onChunkReceived).toHaveBeenCalledWith(chunk1, 0, 2)
      
      // Receive second chunk
      act(() => {
        eventHandlers['preview_chunk'](
          { chunk: chunk2, index: 1, total: 2 },
          { type: 'preview_chunk', data: { chunk: chunk2, index: 1, total: 2 }, timestamp: Date.now() }
        )
      })
      
      expect(result.current.chunks).toHaveLength(2)
      expect(result.current.chunks[1]).toEqual(chunk2)
      expect(onChunkReceived).toHaveBeenCalledWith(chunk2, 1, 2)
    })

    it('should handle out-of-order chunks correctly', () => {
      const { result } = renderHook(() => useChunkingWebSocket())
      
      const chunk1 = mockChunkingPreviewResponse.chunks[0]
      const chunk2 = mockChunkingPreviewResponse.chunks[1]
      
      // Receive second chunk first
      act(() => {
        eventHandlers['preview_chunk'](
          { chunk: chunk2, index: 1, total: 2 },
          { type: 'preview_chunk', data: { chunk: chunk2, index: 1, total: 2 }, timestamp: Date.now() }
        )
      })
      
      expect(result.current.chunks).toHaveLength(2)
      expect(result.current.chunks[0]).toBeUndefined()
      expect(result.current.chunks[1]).toEqual(chunk2)
      
      // Then receive first chunk
      act(() => {
        eventHandlers['preview_chunk'](
          { chunk: chunk1, index: 0, total: 2 },
          { type: 'preview_chunk', data: { chunk: chunk1, index: 0, total: 2 }, timestamp: Date.now() }
        )
      })
      
      expect(result.current.chunks[0]).toEqual(chunk1)
      expect(result.current.chunks[1]).toEqual(chunk2)
    })

    it('should handle preview complete message', () => {
      const onComplete = vi.fn()
      const { result } = renderHook(() => 
        useChunkingWebSocket({ onComplete })
      )
      
      const completeData: ChunkingCompleteData = {
        statistics: mockChunkingPreviewResponse.statistics,
        performance: mockChunkingPreviewResponse.performance,
      }
      
      act(() => {
        eventHandlers['preview_complete'](
          completeData,
          { type: 'preview_complete', data: completeData, timestamp: Date.now() }
        )
      })
      
      expect(result.current.statistics).toEqual(completeData.statistics)
      expect(result.current.performance).toEqual(completeData.performance)
      expect(result.current.progress).toBeNull()
      expect(onComplete).toHaveBeenCalledWith(completeData.statistics, completeData.performance)
    })

    it('should handle preview error message', () => {
      const onError = vi.fn()
      const { result } = renderHook(() => 
        useChunkingWebSocket({ onError })
      )
      
      const errorData: ChunkingErrorData = {
        message: 'Failed to process document',
        code: 'PROCESSING_ERROR',
        details: { reason: 'Invalid format' },
      }
      
      act(() => {
        eventHandlers['preview_error'](
          errorData,
          { type: 'preview_error', data: errorData, timestamp: Date.now() }
        )
      })
      
      expect(result.current.error).toEqual(errorData)
      expect(result.current.progress).toBeNull()
      expect(onError).toHaveBeenCalledWith(errorData)
    })

    it('should handle comparison messages', () => {
      const { result } = renderHook(() => useChunkingWebSocket())
      
      // Comparison start
      act(() => {
        eventHandlers['comparison_start'](
          { totalStrategies: 3 },
          { type: 'comparison_start', data: { totalStrategies: 3 }, timestamp: Date.now() }
        )
      })
      
      expect(result.current.error).toBeNull()
      
      // Comparison progress
      act(() => {
        eventHandlers['comparison_progress'](
          { 
            percentage: 33, 
            currentStrategy: 1, 
            totalStrategies: 3,
            estimatedTimeRemaining: 60 
          },
          { type: 'comparison_progress', data: {}, timestamp: Date.now() }
        )
      })
      
      expect(result.current.progress).toEqual({
        percentage: 33,
        currentChunk: 1,
        totalChunks: 3,
        estimatedTimeRemaining: 60,
      })
      
      // Comparison complete
      act(() => {
        eventHandlers['comparison_complete'](
          { results: mockComparisonResults },
          { type: 'comparison_complete', data: { results: mockComparisonResults }, timestamp: Date.now() }
        )
      })
      
      expect(result.current.progress).toBeNull()
      
      // Comparison error
      const errorData: ChunkingErrorData = {
        message: 'Comparison failed',
        code: 'COMPARISON_ERROR',
      }
      
      act(() => {
        eventHandlers['comparison_error'](
          errorData,
          { type: 'comparison_error', data: errorData, timestamp: Date.now() }
        )
      })
      
      expect(result.current.error).toEqual(errorData)
      expect(result.current.progress).toBeNull()
    })
  })

  describe('Request ID Tracking', () => {
    it('should filter messages by request ID when provided', () => {
      const requestId = 'test-request-123'
      const { result } = renderHook(() => 
        useChunkingWebSocket({ requestId })
      )
      
      const progressData: ChunkingProgressData = {
        percentage: 50,
        currentChunk: 5,
        totalChunks: 10,
      }
      
      // First, start a preview to set the active request ID
      mockWebSocketService.isConnected.mockReturnValue(true)
      act(() => {
        result.current.startPreview('doc-1', 'recursive', {})
      })
      
      // Message with matching request ID (the one passed in options)
      act(() => {
        eventHandlers['preview_progress'](
          progressData,
          { 
            type: 'preview_progress', 
            data: progressData, 
            timestamp: Date.now(),
            requestId: requestId 
          }
        )
      })
      
      expect(result.current.progress).toEqual(progressData)
      
      // Message with different request ID should be ignored
      act(() => {
        eventHandlers['preview_progress'](
          { ...progressData, percentage: 75 },
          { 
            type: 'preview_progress', 
            data: { ...progressData, percentage: 75 }, 
            timestamp: Date.now(),
            requestId: 'different-request' 
          }
        )
      })
      
      // Progress should not be updated
      expect(result.current.progress?.percentage).toBe(50)
    })

    it('should accept messages without request ID when none specified', () => {
      const { result } = renderHook(() => useChunkingWebSocket())
      
      const progressData: ChunkingProgressData = {
        percentage: 50,
        currentChunk: 5,
        totalChunks: 10,
      }
      
      // Message without request ID
      act(() => {
        eventHandlers['preview_progress'](
          progressData,
          { type: 'preview_progress', data: progressData, timestamp: Date.now() }
        )
      })
      
      expect(result.current.progress).toEqual(progressData)
    })

    it('should update active request ID when starting preview', () => {
      mockWebSocketService.isConnected.mockReturnValue(true)
      const { result } = renderHook(() => useChunkingWebSocket())
      
      act(() => {
        result.current.startPreview('doc-1', 'recursive', { chunk_size: 600 })
      })
      
      expect(mockWebSocketService.send).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'preview_request',
          data: {
            documentId: 'doc-1',
            strategy: 'recursive',
            configuration: { chunk_size: 600 },
          },
          requestId: expect.stringMatching(/^preview-\d+$/),
        })
      )
    })
  })

  describe('Actions', () => {
    beforeEach(() => {
      mockWebSocketService.isConnected.mockReturnValue(true)
    })

    it('should start preview when connected', () => {
      const { result } = renderHook(() => useChunkingWebSocket())
      
      act(() => {
        result.current.startPreview(
          'doc-123',
          'recursive',
          { chunk_size: 600, chunk_overlap: 100 }
        )
      })
      
      expect(mockWebSocketService.send).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'preview_request',
          data: {
            documentId: 'doc-123',
            strategy: 'recursive',
            configuration: { chunk_size: 600, chunk_overlap: 100 },
          },
        })
      )
      
      // Should clear previous data
      expect(result.current.chunks).toEqual([])
      expect(result.current.progress).toBeNull()
      expect(result.current.statistics).toBeNull()
      expect(result.current.error).toBeNull()
    })

    it('should show error when starting preview without connection', () => {
      mockWebSocketService.isConnected.mockReturnValue(false)
      const { result } = renderHook(() => useChunkingWebSocket())
      
      act(() => {
        result.current.startPreview('doc-123', 'recursive', {})
      })
      
      expect(mockWebSocketService.send).not.toHaveBeenCalled()
      expect(result.current.error).toEqual({
        message: 'WebSocket not connected. Please wait for connection.',
        code: 'NOT_CONNECTED',
      })
    })

    it('should start comparison when connected', () => {
      const { result } = renderHook(() => useChunkingWebSocket())
      
      const strategies = [
        { strategy: 'recursive', configuration: { chunk_size: 600 } },
        { strategy: 'semantic', configuration: { max_chunk_size: 1000 } },
      ]
      
      act(() => {
        result.current.startComparison('doc-456', strategies)
      })
      
      expect(mockWebSocketService.send).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'comparison_request',
          data: {
            documentId: 'doc-456',
            strategies,
          },
        })
      )
      
      // Should clear previous data
      expect(result.current.progress).toBeNull()
      expect(result.current.error).toBeNull()
    })

    it('should show error when starting comparison without connection', () => {
      mockWebSocketService.isConnected.mockReturnValue(false)
      const { result } = renderHook(() => useChunkingWebSocket())
      
      act(() => {
        result.current.startComparison('doc-456', [])
      })
      
      expect(mockWebSocketService.send).not.toHaveBeenCalled()
      expect(result.current.error).toEqual({
        message: 'WebSocket not connected. Please wait for connection.',
        code: 'NOT_CONNECTED',
      })
    })

    it('should clear all data when clearData is called', () => {
      const { result } = renderHook(() => useChunkingWebSocket())
      
      // Set some data first
      act(() => {
        eventHandlers['preview_chunk'](
          { chunk: mockChunkingPreviewResponse.chunks[0], index: 0, total: 1 },
          { type: 'preview_chunk', data: {}, timestamp: Date.now() }
        )
        eventHandlers['preview_progress'](
          { percentage: 50, currentChunk: 1, totalChunks: 2 },
          { type: 'preview_progress', data: {}, timestamp: Date.now() }
        )
      })
      
      expect(result.current.chunks).toHaveLength(1)
      expect(result.current.progress).not.toBeNull()
      
      // Clear data
      act(() => {
        result.current.clearData()
      })
      
      expect(result.current.chunks).toEqual([])
      expect(result.current.progress).toBeNull()
      expect(result.current.statistics).toBeNull()
      expect(result.current.performance).toBeNull()
      expect(result.current.error).toBeNull()
    })
  })

  describe('Error Handling', () => {
    it('should handle WebSocket errors', () => {
      const onError = vi.fn()
      const { result } = renderHook(() => 
        useChunkingWebSocket({ onError })
      )
      
      const errorData: ChunkingErrorData = {
        message: 'Connection error',
        code: 'WS_ERROR',
      }
      
      act(() => {
        eventHandlers['error'](errorData)
      })
      
      expect(result.current.connectionStatus).toBe('error')
      expect(result.current.error).toEqual({
        message: 'Connection error',
        code: 'WS_ERROR',
        details: undefined,
      })
      expect(onError).toHaveBeenCalledWith(errorData)
    })

    it('should handle preview errors gracefully', () => {
      const onError = vi.fn()
      const { result } = renderHook(() => 
        useChunkingWebSocket({ onError })
      )
      
      // Start with some progress
      act(() => {
        eventHandlers['preview_progress'](
          { percentage: 50, currentChunk: 5, totalChunks: 10 },
          { type: 'preview_progress', data: {}, timestamp: Date.now() }
        )
      })
      
      expect(result.current.progress).not.toBeNull()
      
      // Then error occurs
      const errorData: ChunkingErrorData = {
        message: 'Processing failed',
        code: 'PROC_ERROR',
      }
      
      act(() => {
        eventHandlers['preview_error'](
          errorData,
          { type: 'preview_error', data: errorData, timestamp: Date.now() }
        )
      })
      
      expect(result.current.error).toEqual(errorData)
      expect(result.current.progress).toBeNull() // Progress cleared on error
      expect(onError).toHaveBeenCalledWith(errorData)
    })
  })

  describe('Progress Updates', () => {
    it('should update progress from chunk messages', () => {
      const { result } = renderHook(() => useChunkingWebSocket())
      
      // Receive chunks and check progress updates
      act(() => {
        eventHandlers['preview_chunk'](
          { chunk: mockChunkingPreviewResponse.chunks[0], index: 0, total: 3 },
          { type: 'preview_chunk', data: {}, timestamp: Date.now() }
        )
      })
      
      expect(result.current.progress).toEqual({
        percentage: 33,
        currentChunk: 1,
        totalChunks: 3,
      })
      
      act(() => {
        eventHandlers['preview_chunk'](
          { chunk: mockChunkingPreviewResponse.chunks[1], index: 1, total: 3 },
          { type: 'preview_chunk', data: {}, timestamp: Date.now() }
        )
      })
      
      expect(result.current.progress).toEqual({
        percentage: 67,
        currentChunk: 2,
        totalChunks: 3,
      })
      
      act(() => {
        eventHandlers['preview_chunk'](
          { chunk: mockChunkingPreviewResponse.chunks[0], index: 2, total: 3 },
          { type: 'preview_chunk', data: {}, timestamp: Date.now() }
        )
      })
      
      expect(result.current.progress).toEqual({
        percentage: 100,
        currentChunk: 3,
        totalChunks: 3,
      })
    })

    it('should handle estimated time in progress', () => {
      const onProgressUpdate = vi.fn()
      const { result } = renderHook(() => 
        useChunkingWebSocket({ onProgressUpdate })
      )
      
      const progressWithTime: ChunkingProgressData = {
        percentage: 40,
        currentChunk: 4,
        totalChunks: 10,
        estimatedTimeRemaining: 120,
      }
      
      act(() => {
        eventHandlers['preview_progress'](
          progressWithTime,
          { type: 'preview_progress', data: progressWithTime, timestamp: Date.now() }
        )
      })
      
      expect(result.current.progress).toEqual(progressWithTime)
      expect(onProgressUpdate).toHaveBeenCalledWith(progressWithTime)
    })
  })

  describe('Edge Cases', () => {
    it('should handle multiple rapid connections', () => {
      const { result } = renderHook(() => useChunkingWebSocket({ autoConnect: false }))
      
      // Multiple connect calls
      act(() => {
        result.current.connect()
        result.current.connect()
        result.current.connect()
      })
      
      // Should only connect once (mock tracks this)
      expect(mockWebSocketService.connect).toHaveBeenCalledTimes(3)
    })

    it('should handle missing statistics sizeDistribution field', () => {
      const { result } = renderHook(() => useChunkingWebSocket())
      
      const incompleteData = {
        statistics: {
          ...mockChunkingPreviewResponse.statistics,
          sizeDistribution: undefined,
        },
        performance: mockChunkingPreviewResponse.performance,
      }
      
      act(() => {
        eventHandlers['preview_complete'](
          incompleteData,
          { type: 'preview_complete', data: incompleteData, timestamp: Date.now() }
        )
      })
      
      // Should add empty array for sizeDistribution
      expect(result.current.statistics?.sizeDistribution).toEqual([])
    })

    it('should handle empty chunks array', () => {
      const onChunkReceived = vi.fn()
      const { result } = renderHook(() => 
        useChunkingWebSocket({ onChunkReceived })
      )
      
      // Complete with no chunks
      act(() => {
        eventHandlers['preview_complete'](
          {
            statistics: { ...mockChunkingPreviewResponse.statistics, totalChunks: 0 },
            performance: mockChunkingPreviewResponse.performance,
          },
          { type: 'preview_complete', data: {}, timestamp: Date.now() }
        )
      })
      
      expect(result.current.chunks).toEqual([])
      expect(result.current.statistics?.totalChunks).toBe(0)
      expect(onChunkReceived).not.toHaveBeenCalled()
    })

    it('should maintain state consistency across reconnections', () => {
      const { result } = renderHook(() => useChunkingWebSocket())
      
      // Set some state
      act(() => {
        eventHandlers['preview_chunk'](
          { chunk: mockChunkingPreviewResponse.chunks[0], index: 0, total: 1 },
          { type: 'preview_chunk', data: {}, timestamp: Date.now() }
        )
      })
      
      expect(result.current.chunks).toHaveLength(1)
      
      // Disconnect
      act(() => {
        result.current.disconnect()
      })
      
      // State should be maintained
      expect(result.current.chunks).toHaveLength(1)
      
      // Reconnect
      act(() => {
        result.current.connect()
      })
      
      // State should still be maintained
      expect(result.current.chunks).toHaveLength(1)
    })
  })
})