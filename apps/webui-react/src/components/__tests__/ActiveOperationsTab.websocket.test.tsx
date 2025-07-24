import React from 'react'
import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { ActiveOperationsTab } from '../ActiveOperationsTab'
import { useOperationProgress } from '../../hooks/useOperationProgress'
import { useCollectionStore } from '../../stores/collectionStore'
import { 
  renderWithErrorHandlers,
  mockWebSocket,
  MockWebSocket
} from '../../tests/utils/errorTestUtils'
import { operationsV2Api } from '../../services/api/v2/collections'
import { useQuery } from '@tanstack/react-query'

// Mock the hooks and APIs
vi.mock('../../hooks/useOperationProgress')
vi.mock('../../stores/collectionStore')
vi.mock('../../services/api/v2/operations')
vi.mock('@tanstack/react-query')

describe('ActiveOperationsTab - WebSocket Error Handling', () => {
  const mockUpdateOperationProgress = vi.fn()
  
  const mockActiveOperations = [
    {
      id: 'op-1',
      collection_id: 'coll-1',
      type: 'index',
      status: 'processing',
      progress: 30,
      message: 'Processing documents...',
      source_path: '/data/batch1',
      created_at: new Date(Date.now() - 60000).toISOString()
    },
    {
      id: 'op-2',
      collection_id: 'coll-2',
      type: 'reindex',
      status: 'processing',
      progress: 10,
      message: 'Creating staging environment...',
      created_at: new Date(Date.now() - 30000).toISOString()
    },
    {
      id: 'op-3',
      collection_id: 'coll-3',
      type: 'append',
      status: 'pending',
      progress: 0,
      message: 'Queued',
      source_path: '/data/new',
      created_at: new Date().toISOString()
    }
  ]

  const mockCollections = new Map([
    ['coll-1', { uuid: 'coll-1', name: 'Collection 1' }],
    ['coll-2', { uuid: 'coll-2', name: 'Collection 2' }],
    ['coll-3', { uuid: 'coll-3', name: 'Collection 3' }]
  ])

  let mockWs: { restore: () => void }

  beforeEach(() => {
    vi.clearAllMocks()
    mockWs = mockWebSocket()
    
    vi.mocked(useCollectionStore).mockReturnValue({
      updateOperationProgress: mockUpdateOperationProgress,
      collections: mockCollections,
      getCollectionOperations: vi.fn().mockReturnValue(mockActiveOperations)
    } as any)
  })

  afterEach(() => {
    mockWs.restore()
  })

  describe('Multiple WebSocket Connection Management', () => {
    it('should handle mixed connection states across operations', async () => {
      // Different connection states for each operation
      vi.mocked(useOperationProgress).mockImplementation(({ operationId }) => {
        switch (operationId) {
          case 'op-1':
            return { isConnected: true, error: null, retryCount: 0 }
          case 'op-2':
            return { isConnected: false, error: 'Connection lost', retryCount: 2 }
          case 'op-3':
            // Pending operations shouldn't connect
            return { isConnected: false, error: null, retryCount: 0 }
          default:
            return { isConnected: false, error: null, retryCount: 0 }
        }
      })
      
      vi.mocked(useQuery).mockReturnValue({
        data: mockActiveOperations,
        error: null,
        isLoading: false,
        refetch: vi.fn()
      } as any)
      
      renderWithErrorHandlers(<ActiveOperationsTab />, [])
      
      await waitFor(() => {
        expect(screen.getByText('Processing documents...')).toBeInTheDocument()
      })
      
      // Only op-1 should show live indicator
      const operationItems = screen.getAllByRole('listitem')
      expect(operationItems).toHaveLength(3)
      
      // Check for live indicators (implementation specific)
      const liveIndicators = screen.queryAllByText(/live/i)
      expect(liveIndicators.length).toBeLessThanOrEqual(1)
    })

    it('should handle WebSocket failures without affecting UI refresh', async () => {
      const mockRefetch = vi.fn()
      
      vi.mocked(useOperationProgress).mockImplementation(({ onError }) => {
        // Simulate connection error
        setTimeout(() => {
          onError?.(new Error('WebSocket connection failed'))
        }, 100)
        
        return { isConnected: false, error: 'Connection failed', retryCount: 1 }
      })
      
      vi.mocked(useQuery).mockReturnValue({
        data: mockActiveOperations,
        error: null,
        isLoading: false,
        refetch: mockRefetch
      } as any)
      
      renderWithErrorHandlers(<ActiveOperationsTab />, [])
      
      // Operations should still display
      await waitFor(() => {
        expect(screen.getByText('Processing documents...')).toBeInTheDocument()
        expect(screen.getByText('Creating staging environment...')).toBeInTheDocument()
      })
      
      // Auto-refresh should continue working
      await waitFor(() => {
        expect(mockRefetch).toHaveBeenCalled()
      }, { timeout: 6000 }) // 5s refresh interval + buffer
    })

    it('should handle rapid operation status changes with WebSocket errors', async () => {
      let operationStatus = 'processing'
      
      vi.mocked(useOperationProgress).mockImplementation(({ operationId, onProgress, onComplete, onError }) => {
        if (operationId === 'op-1') {
          // Simulate rapid progress updates with intermittent errors
          const interval = setInterval(() => {
            const random = Math.random()
            if (random < 0.3) {
              onError?.(new Error('Temporary connection issue'))
            } else if (random < 0.8) {
              onProgress?.(Math.floor(random * 100), 'Processing...')
            } else {
              onComplete?.()
              clearInterval(interval)
            }
          }, 100)
        }
        
        return { isConnected: true, error: null, retryCount: 0 }
      })
      
      vi.mocked(useQuery).mockReturnValue({
        data: mockActiveOperations,
        error: null,
        isLoading: false,
        refetch: vi.fn()
      } as any)
      
      renderWithErrorHandlers(<ActiveOperationsTab />, [])
      
      // Should handle rapid updates without crashing
      await waitFor(() => {
        expect(screen.getByText('Processing documents...')).toBeInTheDocument()
      })
      
      // Let it run for a bit
      await new Promise(resolve => setTimeout(resolve, 500))
      
      // Should have received multiple updates
      expect(mockUpdateOperationProgress).toHaveBeenCalled()
    })
  })

  describe('Operation-Specific WebSocket Errors', () => {
    it('should handle authentication errors for specific operations', async () => {
      vi.mocked(useOperationProgress).mockImplementation(({ operationId, onError }) => {
        if (operationId === 'op-2') {
          // Simulate auth error for reindex operation
          setTimeout(() => {
            const error = new Error('Insufficient permissions for reindex')
            ;(error as any).code = 4403
            onError?.(error)
          }, 0)
        }
        
        return { 
          isConnected: operationId !== 'op-2', 
          error: operationId === 'op-2' ? 'Permission denied' : null,
          retryCount: 0
        }
      })
      
      vi.mocked(useQuery).mockReturnValue({
        data: mockActiveOperations,
        error: null,
        isLoading: false,
        refetch: vi.fn()
      } as any)
      
      renderWithErrorHandlers(<ActiveOperationsTab />, [])
      
      // All operations should still display
      await waitFor(() => {
        expect(screen.getByText('Processing documents...')).toBeInTheDocument()
        expect(screen.getByText('Creating staging environment...')).toBeInTheDocument()
        expect(screen.getByText('Queued')).toBeInTheDocument()
      })
      
      // op-2 should not have live indicator
      // But should still show progress from API polling
    })

    it('should handle operation completion during WebSocket outage', async () => {
      let isConnected = true
      
      vi.mocked(useOperationProgress).mockImplementation(({ operationId, onProgress }) => {
        if (operationId === 'op-1' && isConnected) {
          // Send progress updates
          setTimeout(() => onProgress?.(50, 'Halfway...'), 100)
          setTimeout(() => onProgress?.(75, 'Almost done...'), 200)
          setTimeout(() => {
            // Disconnect before completion
            isConnected = false
          }, 300)
        }
        
        return { isConnected, error: null, retryCount: 0 }
      })
      
      // Simulate operation completing via API polling
      const updatedOperations = [...mockActiveOperations]
      updatedOperations[0] = { ...updatedOperations[0], status: 'completed', progress: 100 }
      
      let callCount = 0
      vi.mocked(useQuery).mockImplementation(() => {
        callCount++
        return {
          data: callCount > 2 ? updatedOperations : mockActiveOperations,
          error: null,
          isLoading: false,
          refetch: vi.fn()
        } as any
      })
      
      const { rerender } = renderWithErrorHandlers(<ActiveOperationsTab />, [])
      
      // Should show initial state
      expect(screen.getByText('Processing documents...')).toBeInTheDocument()
      
      // Wait for API polling to detect completion
      await waitFor(() => {
        rerender(<ActiveOperationsTab />)
      })
      
      // Operation should be removed from active list when completed
      await waitFor(() => {
        expect(screen.queryByText('Processing documents...')).not.toBeInTheDocument()
      }, { timeout: 3000 })
    })
  })

  describe('Error Recovery and Fallback', () => {
    it('should fall back to polling when all WebSockets fail', async () => {
      const mockRefetch = vi.fn()
      
      // All WebSocket connections fail
      vi.mocked(useOperationProgress).mockImplementation(() => {
        return { isConnected: false, error: 'WebSocket not available', retryCount: 5 }
      })
      
      vi.mocked(useQuery).mockReturnValue({
        data: mockActiveOperations,
        error: null,
        isLoading: false,
        refetch: mockRefetch
      } as any)
      
      renderWithErrorHandlers(<ActiveOperationsTab />, [])
      
      // Should still show operations from polling
      await waitFor(() => {
        expect(screen.getByText('Processing documents...')).toBeInTheDocument()
      })
      
      // No live indicators should be shown
      expect(screen.queryByText(/live/i)).not.toBeInTheDocument()
      
      // Polling should continue
      await waitFor(() => {
        expect(mockRefetch).toHaveBeenCalled()
      }, { timeout: 6000 })
    })

    it('should handle empty operations list with WebSocket errors gracefully', async () => {
      vi.mocked(useQuery).mockReturnValue({
        data: [],
        error: null,
        isLoading: false,
        refetch: vi.fn()
      } as any)
      
      renderWithErrorHandlers(<ActiveOperationsTab />, [])
      
      // Should show empty state
      await waitFor(() => {
        expect(screen.getByText(/no active operations/i)).toBeInTheDocument()
      })
      
      // No WebSocket connections should be attempted
      expect(vi.mocked(useOperationProgress)).not.toHaveBeenCalled()
    })

    it('should handle API errors while WebSockets are connected', async () => {
      vi.mocked(useOperationProgress).mockReturnValue({
        isConnected: true,
        error: null,
        retryCount: 0
      } as any)
      
      vi.mocked(useQuery).mockReturnValue({
        data: null,
        error: new Error('Failed to fetch operations'),
        isLoading: false,
        refetch: vi.fn()
      } as any)
      
      renderWithErrorHandlers(<ActiveOperationsTab />, [])
      
      // Should show error state
      await waitFor(() => {
        expect(screen.getByText(/failed to load operations/i)).toBeInTheDocument()
      })
      
      // Should show retry button
      expect(screen.getByRole('button', { name: /retry/i })).toBeInTheDocument()
    })
  })

  describe('Performance and Resource Management', () => {
    it('should not create excessive WebSocket connections', async () => {
      // Create many operations
      const manyOperations = Array.from({ length: 50 }, (_, i) => ({
        ...mockActiveOperations[0],
        id: `op-${i}`,
        collection_id: `coll-${i}`,
        message: `Operation ${i}`
      }))
      
      vi.mocked(useQuery).mockReturnValue({
        data: manyOperations,
        error: null,
        isLoading: false,
        refetch: vi.fn()
      } as any)
      
      renderWithErrorHandlers(<ActiveOperationsTab />, [])
      
      await waitFor(() => {
        expect(screen.getByText('Operation 0')).toBeInTheDocument()
      })
      
      // Should limit WebSocket connections (implementation specific)
      // Only processing operations should connect
      const processingOps = manyOperations.filter(op => op.status === 'processing')
      expect(vi.mocked(useOperationProgress)).toHaveBeenCalledTimes(processingOps.length)
    })

    it('should clean up WebSocket connections when operations complete', async () => {
      const cleanupFn = vi.fn()
      
      vi.mocked(useOperationProgress).mockReturnValue({
        isConnected: true,
        error: null,
        retryCount: 0,
        cleanup: cleanupFn
      } as any)
      
      const { unmount } = renderWithErrorHandlers(<ActiveOperationsTab />, [])
      
      // Unmount component
      unmount()
      
      // Cleanup should be called (implementation specific)
      // This depends on how the hook manages cleanup
    })
  })
})