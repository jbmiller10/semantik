import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { useUIStore } from './src/stores/uiStore.ts'

vi.useFakeTimers()

// Add toast
const store = useUIStore.getState()
store.addToast({ message: 'Test', type: 'info' })

console.log('Initial toasts:', useUIStore.getState().toasts.length)

// Clear timers
vi.clearAllTimers()

console.log('After clear timers:', useUIStore.getState().toasts.length)
