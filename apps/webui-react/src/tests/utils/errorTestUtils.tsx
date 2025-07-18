import React, { ReactElement } from 'react'
import { render, screen, waitFor } from '@testing-library/react'
import { RequestHandler } from 'msw'
import { server } from '../mocks/server'
import { BrowserRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: false },
    mutations: { retry: false }
  }
})

const TestWrapper = ({ children }: { children: React.ReactNode }) => (
  <QueryClientProvider client={queryClient}>
    <BrowserRouter>
      {children}
    </BrowserRouter>
  </QueryClientProvider>
)

/**
 * Render a component with specific error handlers
 */
export const renderWithErrorHandlers = (
  ui: ReactElement,
  errorHandlers: RequestHandler[]
) => {
  // Reset handlers and use only error handlers for this test
  server.resetHandlers()
  server.use(...errorHandlers)
  
  return render(ui, { wrapper: TestWrapper })
}

/**
 * Wait for an error message to appear
 */
export const waitForError = async (errorMessage: string | RegExp) => {
  await waitFor(() => {
    const regex = typeof errorMessage === 'string' 
      ? new RegExp(errorMessage, 'i') 
      : errorMessage
    expect(screen.getByText(regex)).toBeInTheDocument()
  })
}

/**
 * Wait for a toast notification
 */
export const waitForToast = async (message: string | RegExp, type?: 'error' | 'success' | 'warning' | 'info') => {
  await waitFor(() => {
    const toast = screen.getByTestId('toast')
    expect(toast).toBeInTheDocument()
    
    const regex = typeof message === 'string' 
      ? new RegExp(message, 'i') 
      : message
    expect(toast).toHaveTextContent(regex)
    
    if (type) {
      expect(toast).toHaveClass(`toast-${type}`)
    }
  })
}

/**
 * Wait for a loading state to appear and disappear
 */
export const waitForLoadingToComplete = async () => {
  // Wait for loading to start
  await waitFor(() => {
    expect(screen.getByText(/loading/i)).toBeInTheDocument()
  })
  
  // Wait for loading to finish
  await waitFor(() => {
    expect(screen.queryByText(/loading/i)).not.toBeInTheDocument()
  })
}

/**
 * Simulate network offline/online
 */
export const simulateOffline = () => {
  Object.defineProperty(navigator, 'onLine', {
    writable: true,
    value: false
  })
  window.dispatchEvent(new Event('offline'))
}

export const simulateOnline = () => {
  Object.defineProperty(navigator, 'onLine', {
    writable: true,
    value: true
  })
  window.dispatchEvent(new Event('online'))
}

/**
 * Assert that a retry button exists and works
 */
export const testRetryFunctionality = async (
  retryButtonText: string | RegExp = /retry/i,
  onRetryHandlers?: RequestHandler[]
) => {
  const retryButton = await screen.findByRole('button', { 
    name: retryButtonText 
  })
  expect(retryButton).toBeInTheDocument()
  
  if (onRetryHandlers) {
    server.use(...onRetryHandlers)
  }
  
  return retryButton
}

/**
 * Test that form data is preserved after error
 */
export const expectFormDataPreserved = (formData: Record<string, string>) => {
  Object.entries(formData).forEach(([name, value]) => {
    const input = screen.getByLabelText(new RegExp(name, 'i')) as HTMLInputElement
    expect(input.value).toBe(value)
  })
}

/**
 * Simulate localStorage auth token removal
 */
export const removeAuthToken = () => {
  localStorage.removeItem('access_token')
  localStorage.removeItem('refresh_token')
  localStorage.removeItem('auth-storage')
}

/**
 * Create a mock console.error that can be asserted
 */
export const mockConsoleError = () => {
  const originalError = console.error
  const mockError = vi.fn()
  console.error = mockError
  
  return {
    mockError,
    restore: () => {
      console.error = originalError
    }
  }
}

/**
 * Test error boundary behavior
 */
export const testErrorBoundary = async (
  triggerError: () => void | Promise<void>,
  expectedErrorMessage?: string | RegExp
) => {
  const { mockError, restore } = mockConsoleError()
  
  try {
    await triggerError()
    
    // Check that error boundary caught the error
    await waitFor(() => {
      expect(screen.getByText(/something went wrong/i)).toBeInTheDocument()
    })
    
    if (expectedErrorMessage) {
      const regex = typeof expectedErrorMessage === 'string' 
        ? new RegExp(expectedErrorMessage, 'i') 
        : expectedErrorMessage
      expect(screen.getByText(regex)).toBeInTheDocument()
    }
    
    // Check that console.error was called
    expect(mockError).toHaveBeenCalled()
    
    // Test reload button if present
    const reloadButton = screen.queryByRole('button', { name: /reload/i })
    if (reloadButton) {
      expect(reloadButton).toBeInTheDocument()
    }
  } finally {
    restore()
  }
}

/**
 * Mock WebSocket for testing
 */
export class MockWebSocket {
  static CONNECTING = 0
  static OPEN = 1
  static CLOSING = 2
  static CLOSED = 3
  
  url: string
  readyState: number = MockWebSocket.CONNECTING
  onopen: ((event: Event) => void) | null = null
  onclose: ((event: CloseEvent) => void) | null = null
  onmessage: ((event: MessageEvent) => void) | null = null
  onerror: ((event: Event) => void) | null = null
  
  private messageQueue: MessageEvent[] = []
  private closeCode?: number
  private closeReason?: string

  constructor(url: string) {
    this.url = url
    
    // Simulate connection based on URL patterns
    setTimeout(() => {
      if (this.shouldFailConnection()) {
        this.simulateConnectionError()
      } else {
        this.simulateOpen()
      }
    }, 0)
  }

  private shouldFailConnection(): boolean {
    // Fail if URL contains "fail" or "error"
    return this.url.includes('fail') || this.url.includes('error')
  }

  private simulateOpen() {
    this.readyState = MockWebSocket.OPEN
    this.onopen?.(new Event('open'))
    
    // Process any queued messages
    this.messageQueue.forEach(msg => this.onmessage?.(msg))
    this.messageQueue = []
  }

  private simulateConnectionError() {
    this.onerror?.(new Event('error'))
    this.close(1006, 'Connection failed')
  }

  send(data: string) {
    if (this.readyState !== MockWebSocket.OPEN) {
      throw new Error('WebSocket is not open')
    }
    
    // Echo back for testing
    const message = new MessageEvent('message', { data })
    this.onmessage?.(message)
  }

  close(code: number = 1000, reason: string = '') {
    this.readyState = MockWebSocket.CLOSING
    this.closeCode = code
    this.closeReason = reason
    
    setTimeout(() => {
      this.readyState = MockWebSocket.CLOSED
      this.onclose?.(new CloseEvent('close', { code, reason }))
    }, 0)
  }

  // Test helper to simulate incoming messages
  simulateMessage(data: any) {
    const message = new MessageEvent('message', { 
      data: typeof data === 'string' ? data : JSON.stringify(data) 
    })
    
    if (this.readyState === MockWebSocket.OPEN) {
      this.onmessage?.(message)
    } else {
      this.messageQueue.push(message)
    }
  }

  // Test helper to simulate errors
  simulateError(error?: Event) {
    this.onerror?.(error || new Event('error'))
  }

  // Test helper to simulate disconnection
  simulateDisconnect(code: number = 1006, reason: string = 'Connection lost') {
    this.close(code, reason)
  }
}

// Replace global WebSocket with mock for tests
export const mockWebSocket = () => {
  const originalWebSocket = global.WebSocket
  global.WebSocket = MockWebSocket as any
  
  return {
    restore: () => {
      global.WebSocket = originalWebSocket
    }
  }
}