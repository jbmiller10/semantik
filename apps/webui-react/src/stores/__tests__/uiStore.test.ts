import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { act, renderHook } from '@testing-library/react'
import { useUIStore } from '../uiStore'

describe('uiStore', () => {
  beforeEach(() => {
    // Clear all timers
    vi.clearAllTimers()
    vi.useFakeTimers()
    
    // Reset the store by getting its initial state
    const store = useUIStore.getState()
    // Clear all toasts properly
    store.toasts.forEach(toast => store.removeToast(toast.id))
    // Reset other state
    store.setActiveTab('create')
    store.setShowJobMetricsModal(null)
    store.setShowDocumentViewer(null)
    store.setShowCollectionDetailsModal(null)
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  describe('toasts', () => {
    it('adds a toast with auto-generated id', () => {
      const { result } = renderHook(() => useUIStore())
      
      act(() => {
        result.current.addToast({
          message: 'Test message',
          type: 'success',
        })
      })

      expect(result.current.toasts).toHaveLength(1)
      expect(result.current.toasts[0]).toMatchObject({
        message: 'Test message',
        type: 'success',
      })
      expect(result.current.toasts[0].id).toBeDefined()
    })

    it('removes a toast by id', () => {
      const { result } = renderHook(() => useUIStore())
      
      let firstToastId: string
      
      // Add two toasts
      act(() => {
        result.current.addToast({
          message: 'Toast 1',
          type: 'info',
        })
        result.current.addToast({
          message: 'Toast 2',
          type: 'error',
        })
      })

      expect(result.current.toasts).toHaveLength(2)
      firstToastId = result.current.toasts[0].id

      // Remove the first toast
      act(() => {
        result.current.removeToast(firstToastId)
      })

      expect(result.current.toasts).toHaveLength(1)
      expect(result.current.toasts[0].message).toBe('Toast 2')
    })

    it('auto-removes toast after default duration', () => {
      const { result } = renderHook(() => useUIStore())
      
      act(() => {
        result.current.addToast({
          message: 'Auto-remove test',
          type: 'info',
        })
      })

      expect(result.current.toasts).toHaveLength(1)

      // Fast-forward time by 5 seconds (default duration)
      act(() => {
        vi.advanceTimersByTime(5000)
      })

      expect(result.current.toasts).toHaveLength(0)
    })

    it('auto-removes toast after custom duration', () => {
      const { result } = renderHook(() => useUIStore())
      
      act(() => {
        result.current.addToast({
          message: 'Custom duration',
          type: 'warning',
          duration: 3000,
        })
      })

      expect(result.current.toasts).toHaveLength(1)

      // Fast-forward time by 2 seconds - toast should still be there
      act(() => {
        vi.advanceTimersByTime(2000)
      })
      expect(result.current.toasts).toHaveLength(1)

      // Fast-forward another 1 second - toast should be removed
      act(() => {
        vi.advanceTimersByTime(1000)
      })
      expect(result.current.toasts).toHaveLength(0)
    })

    it('does not auto-remove toast when duration is 0', () => {
      const { result } = renderHook(() => useUIStore())
      
      act(() => {
        result.current.addToast({
          message: 'Persistent toast',
          type: 'error',
          duration: 0,
        })
      })

      expect(result.current.toasts).toHaveLength(1)

      // Fast-forward time by a long duration
      act(() => {
        vi.advanceTimersByTime(60000) // 60 seconds
      })

      // Toast should still be there
      expect(result.current.toasts).toHaveLength(1)
    })

    it('handles multiple toasts with different durations', () => {
      const { result } = renderHook(() => useUIStore())
      
      act(() => {
        result.current.addToast({
          message: 'Toast 1',
          type: 'info',
          duration: 2000,
        })
        result.current.addToast({
          message: 'Toast 2',
          type: 'success',
          duration: 4000,
        })
        result.current.addToast({
          message: 'Toast 3',
          type: 'error',
          duration: 0, // Persistent
        })
      })

      expect(result.current.toasts).toHaveLength(3)

      // After 2 seconds, first toast should be gone
      act(() => {
        vi.advanceTimersByTime(2000)
      })
      expect(result.current.toasts).toHaveLength(2)
      expect(result.current.toasts.map(t => t.message)).toEqual(['Toast 2', 'Toast 3'])

      // After 2 more seconds, second toast should be gone
      act(() => {
        vi.advanceTimersByTime(2000)
      })
      expect(result.current.toasts).toHaveLength(1)
      expect(result.current.toasts[0].message).toBe('Toast 3')
    })
  })

  describe('activeTab', () => {
    it('has default activeTab as create', () => {
      const { result } = renderHook(() => useUIStore())
      expect(result.current.activeTab).toBe('create')
    })

    it('sets activeTab correctly', () => {
      const { result } = renderHook(() => useUIStore())
      
      act(() => {
        result.current.setActiveTab('jobs')
      })
      expect(result.current.activeTab).toBe('jobs')

      act(() => {
        result.current.setActiveTab('search')
      })
      expect(result.current.activeTab).toBe('search')

      act(() => {
        result.current.setActiveTab('collections')
      })
      expect(result.current.activeTab).toBe('collections')
    })
  })

  describe('modals', () => {
    it('sets and clears showJobMetricsModal', () => {
      const { result } = renderHook(() => useUIStore())
      
      expect(result.current.showJobMetricsModal).toBeNull()

      act(() => {
        result.current.setShowJobMetricsModal('job-123')
      })
      expect(result.current.showJobMetricsModal).toBe('job-123')

      act(() => {
        result.current.setShowJobMetricsModal(null)
      })
      expect(result.current.showJobMetricsModal).toBeNull()
    })

    it('sets and clears showDocumentViewer', () => {
      const { result } = renderHook(() => useUIStore())
      
      expect(result.current.showDocumentViewer).toBeNull()

      const viewerData = {
        jobId: 'job-123',
        docId: 'doc-456',
        chunkId: 'chunk-789',
      }

      act(() => {
        result.current.setShowDocumentViewer(viewerData)
      })
      expect(result.current.showDocumentViewer).toEqual(viewerData)

      act(() => {
        result.current.setShowDocumentViewer(null)
      })
      expect(result.current.showDocumentViewer).toBeNull()
    })

    it('sets showDocumentViewer without chunkId', () => {
      const { result } = renderHook(() => useUIStore())
      
      const viewerData = {
        jobId: 'job-123',
        docId: 'doc-456',
      }

      act(() => {
        result.current.setShowDocumentViewer(viewerData)
      })
      expect(result.current.showDocumentViewer).toEqual(viewerData)
      expect(result.current.showDocumentViewer?.chunkId).toBeUndefined()
    })

    it('sets and clears showCollectionDetailsModal', () => {
      const { result } = renderHook(() => useUIStore())
      
      expect(result.current.showCollectionDetailsModal).toBeNull()

      act(() => {
        result.current.setShowCollectionDetailsModal('my-collection')
      })
      expect(result.current.showCollectionDetailsModal).toBe('my-collection')

      act(() => {
        result.current.setShowCollectionDetailsModal(null)
      })
      expect(result.current.showCollectionDetailsModal).toBeNull()
    })
  })

  describe('state isolation', () => {
    it('maintains separate state for different hooks', () => {
      const { result: hook1 } = renderHook(() => useUIStore())
      const { result: hook2 } = renderHook(() => useUIStore())
      
      // Both hooks should see the same initial state
      expect(hook1.current.activeTab).toBe('create')
      expect(hook2.current.activeTab).toBe('create')
      
      // Change state in hook1
      act(() => {
        hook1.current.setActiveTab('jobs')
      })
      
      // Both hooks should see the updated state (zustand uses global state)
      expect(hook1.current.activeTab).toBe('jobs')
      expect(hook2.current.activeTab).toBe('jobs')
    })
  })
})