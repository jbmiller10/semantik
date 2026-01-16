import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { act, renderHook } from '@testing-library/react'
import { useUIStore } from '../uiStore'

describe('uiStore', () => {
  beforeEach(() => {
    // Reset the store to initial state first
    useUIStore.setState({
      toasts: [],
      activeTab: 'collections',
      showDocumentViewer: null,
      showCollectionDetailsModal: null,
    })
    
    // Then setup fake timers
    vi.clearAllTimers()
    vi.useFakeTimers({ shouldAdvanceTime: true })
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.clearAllMocks()
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
      
      // Add two toasts with no auto-removal
      act(() => {
        // Use a fixed timestamp for consistent IDs
        const now = Date.now()
        vi.setSystemTime(now)
        
        result.current.addToast({
          message: 'Toast 1',
          type: 'info',
          duration: 0,  // No auto-removal
        })
        
        // Advance time slightly for different ID
        vi.setSystemTime(now + 1)
        
        result.current.addToast({
          message: 'Toast 2',
          type: 'error',
          duration: 0,  // No auto-removal
        })
      })

      // Verify toasts were added
      expect(result.current.toasts).toHaveLength(2)
      const firstToastId = result.current.toasts[0].id
      const secondToastId = result.current.toasts[1].id
      
      expect(result.current.toasts[0].message).toBe('Toast 1')
      expect(result.current.toasts[1].message).toBe('Toast 2')

      // Remove the first toast
      act(() => {
        result.current.removeToast(firstToastId)
      })

      // Verify only one toast remains
      expect(result.current.toasts).toHaveLength(1)
      expect(result.current.toasts[0].message).toBe('Toast 2')
      expect(result.current.toasts[0].id).toBe(secondToastId)
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
        // Use fixed timestamps for consistent IDs
        const now = Date.now()
        vi.setSystemTime(now)
        
        result.current.addToast({
          message: 'Toast 1',
          type: 'info',
          duration: 2000,
        })
        
        vi.setSystemTime(now + 1)
        result.current.addToast({
          message: 'Toast 2',
          type: 'success',
          duration: 4000,
        })
        
        vi.setSystemTime(now + 2)
        result.current.addToast({
          message: 'Toast 3',
          type: 'error',
          duration: 0, // Persistent
        })
      })

      expect(result.current.toasts).toHaveLength(3)
      expect(result.current.toasts.map(t => t.message)).toEqual(['Toast 1', 'Toast 2', 'Toast 3'])

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
    it('has default activeTab as collections', () => {
      const { result } = renderHook(() => useUIStore())
      expect(result.current.activeTab).toBe('collections')
    })

    it('sets activeTab correctly', () => {
      const { result } = renderHook(() => useUIStore())
      
      act(() => {
        result.current.setActiveTab('search')
      })
      expect(result.current.activeTab).toBe('search')

      act(() => {
        result.current.setActiveTab('collections')
      })
      expect(result.current.activeTab).toBe('collections')

      act(() => {
        result.current.setActiveTab('operations')
      })
      expect(result.current.activeTab).toBe('operations')
    })
  })

  describe('modals', () => {
    it('sets and clears showDocumentViewer', () => {
      const { result } = renderHook(() => useUIStore())
      
      expect(result.current.showDocumentViewer).toBeNull()

      const viewerData = {
        collectionId: 'collection-123',
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
        collectionId: 'collection-123',
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
      expect(hook1.current.activeTab).toBe('collections')
      expect(hook2.current.activeTab).toBe('collections')

      // Change state in hook1
      act(() => {
        hook1.current.setActiveTab('search')
      })

      // Both hooks should see the updated state (zustand uses global state)
      expect(hook1.current.activeTab).toBe('search')
      expect(hook2.current.activeTab).toBe('search')
    })
  })

  describe('theme', () => {
    let localStorageMock: Record<string, string>
    let classListMock: { add: ReturnType<typeof vi.fn>; remove: ReturnType<typeof vi.fn> }
    let matchMediaMock: ReturnType<typeof vi.fn>

    beforeEach(() => {
      // Setup localStorage mock
      localStorageMock = {}
      vi.spyOn(Storage.prototype, 'getItem').mockImplementation((key) => localStorageMock[key] || null)
      vi.spyOn(Storage.prototype, 'setItem').mockImplementation((key, value) => {
        localStorageMock[key] = value
      })

      // Setup classList mock
      classListMock = {
        add: vi.fn(),
        remove: vi.fn(),
      }
      vi.spyOn(document.documentElement.classList, 'add').mockImplementation(classListMock.add)
      vi.spyOn(document.documentElement.classList, 'remove').mockImplementation(classListMock.remove)

      // Setup matchMedia mock (required by applyTheme function)
      matchMediaMock = vi.fn().mockImplementation((query: string) => ({
        matches: query === '(prefers-color-scheme: dark)' ? false : false,
        media: query,
        onchange: null,
        addListener: vi.fn(),
        removeListener: vi.fn(),
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        dispatchEvent: vi.fn(),
      }))
      vi.spyOn(window, 'matchMedia').mockImplementation(matchMediaMock)
    })

    afterEach(() => {
      vi.restoreAllMocks()
    })

    it('has a theme property in state', () => {
      const { result } = renderHook(() => useUIStore())
      expect(result.current.theme).toBeDefined()
    })

    it('persists theme to localStorage when setTheme is called', () => {
      const { result } = renderHook(() => useUIStore())

      act(() => {
        result.current.setTheme('dark')
      })

      expect(localStorage.setItem).toHaveBeenCalledWith('semantik-theme', 'dark')
      expect(result.current.theme).toBe('dark')
    })

    it('updates store state when setTheme is called with light', () => {
      const { result } = renderHook(() => useUIStore())

      act(() => {
        result.current.setTheme('light')
      })

      expect(localStorage.setItem).toHaveBeenCalledWith('semantik-theme', 'light')
      expect(result.current.theme).toBe('light')
    })

    it('updates store state when setTheme is called with system', () => {
      const { result } = renderHook(() => useUIStore())

      act(() => {
        result.current.setTheme('system')
      })

      expect(localStorage.setItem).toHaveBeenCalledWith('semantik-theme', 'system')
      expect(result.current.theme).toBe('system')
    })

    it('applies dark class to document when theme is dark', () => {
      const { result } = renderHook(() => useUIStore())

      act(() => {
        result.current.setTheme('dark')
      })

      expect(classListMock.add).toHaveBeenCalledWith('dark')
    })

    it('removes dark class from document when theme is light', () => {
      const { result } = renderHook(() => useUIStore())

      act(() => {
        result.current.setTheme('light')
      })

      expect(classListMock.remove).toHaveBeenCalledWith('dark')
    })

    it('cycles through all theme values correctly', () => {
      const { result } = renderHook(() => useUIStore())

      act(() => {
        result.current.setTheme('light')
      })
      expect(result.current.theme).toBe('light')

      act(() => {
        result.current.setTheme('dark')
      })
      expect(result.current.theme).toBe('dark')

      act(() => {
        result.current.setTheme('system')
      })
      expect(result.current.theme).toBe('system')
    })
  })

  describe('timer cleanup on toast removal', () => {
    it('clears timer when toast is manually removed', () => {
      const clearTimeoutSpy = vi.spyOn(globalThis, 'clearTimeout')
      const { result } = renderHook(() => useUIStore())

      // Add a toast with auto-removal (will have a timerId)
      act(() => {
        result.current.addToast({
          message: 'Test',
          type: 'info',
          duration: 5000,
        })
      })

      const toastId = result.current.toasts[0].id
      const timerId = result.current.toasts[0].timerId

      // Verify timerId was set
      expect(timerId).toBeDefined()

      // Remove the toast manually
      act(() => {
        result.current.removeToast(toastId)
      })

      // Verify clearTimeout was called with the timer ID
      expect(clearTimeoutSpy).toHaveBeenCalledWith(timerId)
    })

    it('handles removing non-existent toast gracefully', () => {
      const { result } = renderHook(() => useUIStore())

      // Try to remove a non-existent toast - should not throw
      expect(() => {
        act(() => {
          result.current.removeToast('non-existent-id')
        })
      }).not.toThrow()
    })

    it('does not attempt clearTimeout for toast without timerId', () => {
      const clearTimeoutSpy = vi.spyOn(globalThis, 'clearTimeout')
      const { result } = renderHook(() => useUIStore())

      // Add a toast with duration 0 (no timer)
      act(() => {
        result.current.addToast({
          message: 'Persistent',
          type: 'error',
          duration: 0,
        })
      })

      const toastId = result.current.toasts[0].id

      // Verify timerId is undefined
      expect(result.current.toasts[0].timerId).toBeUndefined()

      // Clear any previous clearTimeout calls
      clearTimeoutSpy.mockClear()

      // Remove the toast
      act(() => {
        result.current.removeToast(toastId)
      })

      // clearTimeout should not be called since there was no timer
      expect(clearTimeoutSpy).not.toHaveBeenCalled()
    })
  })
})