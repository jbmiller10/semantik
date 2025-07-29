import React from 'react'
import { screen, waitFor } from '@testing-library/react'
import { OperationProgress } from '../OperationProgress'
import { useOperationProgress } from '../../hooks/useOperationProgress'
import { useCollectionStore } from '../../stores/collectionStore'
import { useUIStore } from '../../stores/uiStore'
import { 
  renderWithErrorHandlers,
  mockWebSocket,
  MockWebSocket
} from '../../tests/utils/errorTestUtils'
import type { MockedFunction, MockOperation } from '@/tests/types/test-types'

// Mock the hooks
vi.mock('../../hooks/useOperationProgress')
vi.mock('../../stores/collectionStore')
vi.mock('../../stores/uiStore')

describe('OperationProgress - WebSocket Error Handling', () => {
  const mockAddToast = vi.fn()
  const mockUpdateOperationProgress = vi.fn()
  
  const mockOperation: MockOperation = {
    id: 'test-op-id',
    collection_id: 'test-coll-id',
    type: 'index' as const,
    status: 'processing' as const,
    progress: 0,
    message: 'Starting...',
    config: { source_path: '/data/documents' },
    created_at: new Date().toISOString()
  }

  let mockWs: { restore: () => void }

  beforeEach(() => {
    vi.clearAllMocks()
    mockWs = mockWebSocket()
    
    vi.mocked(useUIStore).mockReturnValue({
      addToast: mockAddToast
    } as ReturnType<typeof useUIStore>)
    
    vi.mocked(useCollectionStore).mockReturnValue({
      updateOperationProgress: mockUpdateOperationProgress
    } as ReturnType<typeof useCollectionStore>)
  })

  afterEach(() => {
    mockWs.restore()
  })

  describe('Connection Failures', () => {
    it('should handle initial connection failure', async () => {
      vi.mocked(useOperationProgress).mockImplementation(({ onError }) => {
        // Simulate connection failure
        setTimeout(() => {
          onError?.(new Error('Failed to connect to WebSocket'))
        }, 0)
        
        return {
          isConnected: false,
          error: 'Failed to connect to WebSocket',
          retryCount: 1
        }
      })
      
      renderWithErrorHandlers(
        <OperationProgress operation={mockOperation} />,
        []
      )
      
      // Should show operation info even without connection
      expect(screen.getByText('/data/documents')).toBeInTheDocument()
      
      // Should not show "Live" indicator
      expect(screen.queryByText(/live/i)).not.toBeInTheDocument()
      
      // Error might be logged but UI should handle gracefully
      await waitFor(() => {
        expect(screen.getByText('Starting...')).toBeInTheDocument()
      })
    })

    it('should handle connection drop during operation', async () => {
      let onErrorCallback: ((error: Error) => void) | undefined
      let connectionState = { isConnected: true, error: null, retryCount: 0 }
      
      vi.mocked(useOperationProgress).mockImplementation(({ onError }) => {
        onErrorCallback = onError
        return connectionState
      })
      
      const { rerender } = renderWithErrorHandlers(
        <OperationProgress operation={mockOperation} />,
        []
      )
      
      // Initially connected
      expect(screen.getByText(/live/i)).toBeInTheDocument()
      
      // Simulate connection drop
      connectionState = { isConnected: false, error: 'Connection lost', retryCount: 1 }
      onErrorCallback?.(new Error('Connection lost'))
      
      rerender(<OperationProgress operation={mockOperation} />)
      
      // Should no longer show "Live" indicator
      await waitFor(() => {
        expect(screen.queryByText(/live/i)).not.toBeInTheDocument()
      })
      
      // Operation info should still be visible
      expect(screen.getByText('/data/documents')).toBeInTheDocument()
    })

    it('should handle authentication failure', async () => {
      vi.mocked(useOperationProgress).mockImplementation(({ onError }) => {
        setTimeout(() => {
          const error = new Error('WebSocket authentication failed') as Error & { code: number }
          error.code = 4401
          onError?.(error)
        }, 0)
        
        return {
          isConnected: false,
          error: 'Authentication failed',
          retryCount: 0
        }
      })
      
      renderWithErrorHandlers(
        <OperationProgress operation={mockOperation} />,
        []
      )
      
      // Should still show operation status
      expect(screen.getByRole('status')).toBeInTheDocument()
      
      // Should not crash the component
      await waitFor(() => {
        expect(screen.getByText('Starting...')).toBeInTheDocument()
      })
    })

    it('should handle permission denied for operation', async () => {
      vi.mocked(useOperationProgress).mockImplementation(({ onError }) => {
        setTimeout(() => {
          const error = new Error('Permission denied') as Error & { code: number }
          error.code = 4403
          onError?.(error)
        }, 0)
        
        return {
          isConnected: false,
          error: 'Permission denied',
          retryCount: 0
        }
      })
      
      renderWithErrorHandlers(
        <OperationProgress operation={{
          ...mockOperation,
          id: 'other-user-op'
        }} />,
        []
      )
      
      // Component should handle gracefully
      await waitFor(() => {
        expect(screen.getByText('Starting...')).toBeInTheDocument()
      })
    })
  })

  describe('Message Handling Errors', () => {
    it('should handle malformed WebSocket messages', async () => {
      const mockMessages = [
        { type: 'progress', progress: 25, message: 'Processing...' },
        'invalid json string',
        { invalidStructure: true },
        { type: 'progress', progress: 50, message: 'Halfway there' }
      ]
      
      let messageIndex = 0
      
      vi.mocked(useOperationProgress).mockImplementation(({ operationId, onProgress }) => {
        // Send messages sequentially
        const interval = setInterval(() => {
          if (messageIndex < mockMessages.length) {
            try {
              const msg = mockMessages[messageIndex]
              if (typeof msg === 'object' && msg.type === 'progress') {
                onProgress?.(msg.progress, msg.message || '')
              }
            } catch {
              // Malformed message - should be handled gracefully
            }
            messageIndex++
          } else {
            clearInterval(interval)
          }
        }, 50)
        
        return { isConnected: true, error: null, retryCount: 0 }
      })
      
      renderWithErrorHandlers(
        <OperationProgress operation={mockOperation} />,
        []
      )
      
      // Should process valid messages and skip invalid ones
      await waitFor(() => {
        expect(mockUpdateOperationProgress).toHaveBeenCalledWith(
          mockOperation.id,
          50,
          'Halfway there'
        )
      })
      
      // Should have processed 2 valid messages
      expect(mockUpdateOperationProgress).toHaveBeenCalledTimes(2)
    })

    it('should handle missing required fields in messages', async () => {
      vi.mocked(useOperationProgress).mockImplementation(({ operationId, onProgress, onError }) => {
        // Send message with missing progress
        setTimeout(() => {
          try {
            onProgress?.(undefined as unknown as number, 'Missing progress')
          } catch (e) {
            onError?.(e as Error)
          }
        }, 0)
        
        return { isConnected: true, error: null, retryCount: 0 }
      })
      
      renderWithErrorHandlers(
        <OperationProgress operation={mockOperation} />,
        []
      )
      
      // Component should not crash
      await waitFor(() => {
        expect(screen.getByText('Starting...')).toBeInTheDocument()
      })
    })

    it('should handle extremely large messages', async () => {
      const largeMessage = 'A'.repeat(100000) // 100KB message
      
      vi.mocked(useOperationProgress).mockImplementation(({ operationId, onProgress }) => {
        setTimeout(() => {
          onProgress?.(25, largeMessage)
        }, 0)
        
        return { isConnected: true, error: null, retryCount: 0 }
      })
      
      renderWithErrorHandlers(
        <OperationProgress operation={mockOperation} />,
        []
      )
      
      // Should truncate or handle large messages gracefully
      await waitFor(() => {
        expect(mockUpdateOperationProgress).toHaveBeenCalled()
      })
      
      // Check if message was truncated (implementation specific)
      const call = mockUpdateOperationProgress.mock.calls[0]
      expect(call[2].length).toBeLessThanOrEqual(1000) // Reasonable message length
    })
  })

  describe('Reconnection Scenarios', () => {
    it('should show reconnection attempts', async () => {
      let retryCount = 0
      
      vi.mocked(useOperationProgress).mockImplementation(() => {
        return { 
          isConnected: false, 
          error: 'Connection lost', 
          retryCount: retryCount++
        }
      })
      
      const { rerender } = renderWithErrorHandlers(
        <OperationProgress operation={mockOperation} />,
        []
      )
      
      // Trigger rerenders to simulate retry attempts
      for (let i = 0; i < 3; i++) {
        rerender(<OperationProgress operation={mockOperation} />)
        await new Promise(resolve => setTimeout(resolve, 100))
      }
      
      // Component should handle multiple retry attempts
      expect(screen.getByText('Starting...')).toBeInTheDocument()
    })

    it('should restore state after reconnection', async () => {
      let isConnected = false
      let currentProgress = 25
      
      vi.mocked(useOperationProgress).mockImplementation(({ operationId, onProgress }) => {
        if (isConnected) {
          // Simulate receiving current state after reconnection
          setTimeout(() => {
            onProgress?.(currentProgress, 'Resumed after reconnection')
          }, 0)
        }
        
        return { isConnected, error: null, retryCount: isConnected ? 0 : 1 }
      })
      
      const { rerender } = renderWithErrorHandlers(
        <OperationProgress operation={{
          ...mockOperation,
          progress: currentProgress,
          message: 'Processing...'
        }} />,
        []
      )
      
      // Initially disconnected
      expect(screen.queryByText(/live/i)).not.toBeInTheDocument()
      
      // Reconnect
      isConnected = true
      currentProgress = 50
      rerender(<OperationProgress operation={{
        ...mockOperation,
        progress: currentProgress,
        message: 'Processing...'
      }} />)
      
      // Should show live indicator again
      await waitFor(() => {
        expect(screen.getByText(/live/i)).toBeInTheDocument()
      })
      
      // Should receive state update
      await waitFor(() => {
        expect(mockUpdateOperationProgress).toHaveBeenCalledWith(
          mockOperation.id,
          currentProgress,
          'Resumed after reconnection'
        )
      })
    })
  })

  describe('Operation Completion Handling', () => {
    it('should handle completion message during connection issues', async () => {
      let onCompleteCallback: (() => void) | undefined
      
      vi.mocked(useOperationProgress).mockImplementation(({ operationId, onComplete }) => {
        onCompleteCallback = onComplete
        
        // Simulate connection issue
        return { isConnected: false, error: 'Connection lost', retryCount: 1 }
      })
      
      renderWithErrorHandlers(
        <OperationProgress operation={mockOperation} />,
        []
      )
      
      // Operation completes while disconnected
      onCompleteCallback?.()
      
      // Should handle gracefully without errors
      await waitFor(() => {
        expect(screen.getByText('Starting...')).toBeInTheDocument()
      })
    })

    it('should handle error status during operation', async () => {
      vi.mocked(useOperationProgress).mockImplementation(({ onError }) => {
        setTimeout(() => {
          onError?.(new Error('Operation failed: Insufficient disk space'))
        }, 0)
        
        return { isConnected: true, error: null, retryCount: 0 }
      })
      
      renderWithErrorHandlers(
        <OperationProgress operation={{
          ...mockOperation,
          status: 'failed',
          error_message: 'Insufficient disk space'
        }} />,
        []
      )
      
      // Should show error status
      expect(screen.getByText(/failed/i)).toBeInTheDocument()
      expect(screen.getByText(/insufficient disk space/i)).toBeInTheDocument()
    })
  })

  describe('Multiple Operations', () => {
    it('should handle WebSocket errors for multiple operations independently', async () => {
      const operations = [
        { ...mockOperation, id: 'op-1' },
        { ...mockOperation, id: 'op-2' },
        { ...mockOperation, id: 'op-3' }
      ]
      
      vi.mocked(useOperationProgress).mockImplementation(({ operationId }) => {
        // Different connection states for different operations
        if (operationId === 'op-1') {
          return { isConnected: true, error: null, retryCount: 0 }
        } else if (operationId === 'op-2') {
          return { isConnected: false, error: 'Connection failed', retryCount: 2 }
        } else {
          return { isConnected: false, error: 'Authentication failed', retryCount: 0 }
        }
      })
      
      renderWithErrorHandlers(
        <div>
          {operations.map(op => (
            <OperationProgress key={op.id} operation={op} />
          ))}
        </div>,
        []
      )
      
      // Should show different states
      const liveIndicators = screen.queryAllByText(/live/i)
      expect(liveIndicators).toHaveLength(1) // Only op-1 is connected
    })
  })
})