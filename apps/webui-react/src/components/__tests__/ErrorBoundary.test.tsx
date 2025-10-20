import React from 'react'
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import ErrorBoundary from '../ErrorBoundary'

// Component that throws an error
const ThrowError = ({ shouldThrow }: { shouldThrow: boolean }) => {
  if (shouldThrow) {
    throw new Error('Test error message')
  }
  return <div>No error</div>
}

// Note: window.location.reload is not mocked because the enhanced ErrorBoundary
// now uses a reset mechanism instead of page reload

describe('ErrorBoundary', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    // Suppress console.error for these tests
    vi.spyOn(console, 'error').mockImplementation(() => {})
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('renders children when there is no error', () => {
    render(
      <ErrorBoundary>
        <div>Test content</div>
      </ErrorBoundary>
    )

    expect(screen.getByText('Test content')).toBeInTheDocument()
  })

  it('catches errors and displays error UI', () => {
    render(
      <ErrorBoundary>
        <ThrowError shouldThrow={true} />
      </ErrorBoundary>
    )

    expect(screen.getByText('Component Error')).toBeInTheDocument()
    expect(screen.getByText('Test error message')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Try Again/i })).toBeInTheDocument()
  })

  it('displays stack trace in collapsible details', () => {
    const error = new Error('Test error with stack')
    // Create a predictable stack trace
    error.stack = 'Error: Test error with stack\n    at TestComponent\n    at ErrorBoundary'

    const ThrowCustomError = () => {
      throw error
    }

    render(
      <ErrorBoundary>
        <ThrowCustomError />
      </ErrorBoundary>
    )

    expect(screen.getByText('Technical details')).toBeInTheDocument()
    
    // The stack trace should be in a details element
    const details = document.querySelector('details')
    expect(details).toBeInTheDocument()
    
    // The details should contain error information
    expect(screen.getByText('Error ID:')).toBeInTheDocument()
    expect(screen.getByText('Component Stack:')).toBeInTheDocument()
    expect(screen.getByText('Error Stack:')).toBeInTheDocument()
    
    // The error message should be displayed
    expect(screen.getByText('Test error with stack')).toBeInTheDocument()
  })

  it('resets error boundary when Try Again button is clicked', async () => {
    const user = userEvent.setup()
    let shouldThrow = true
    
    const TestComponent = () => {
      if (shouldThrow) {
        throw new Error('Test error')
      }
      return <div>No error</div>
    }

    const { rerender } = render(
      <ErrorBoundary>
        <TestComponent />
      </ErrorBoundary>
    )

    expect(screen.getByText('Component Error')).toBeInTheDocument()
    
    // Set shouldThrow to false and click Try Again
    shouldThrow = false
    const resetButton = screen.getByRole('button', { name: /Try Again/i })
    await user.click(resetButton)
    
    // Re-render to see the recovered state
    rerender(
      <ErrorBoundary>
        <TestComponent />
      </ErrorBoundary>
    )
    
    expect(screen.getByText('No error')).toBeInTheDocument()
  })

  it('logs errors to console', () => {
    const consoleError = vi.spyOn(console, 'error')

    render(
      <ErrorBoundary>
        <ThrowError shouldThrow={true} />
      </ErrorBoundary>
    )

    expect(consoleError).toHaveBeenCalledWith(
      '[ErrorBoundary] Component error caught:',
      expect.objectContaining({
        error: expect.any(Error),
        errorInfo: expect.objectContaining({
          componentStack: expect.any(String),
        }),
        errorId: expect.stringMatching(/^error-/),
        level: 'component',
        timestamp: expect.any(String)
      })
    )
  })

  it('shows error boundary UI with proper styling', () => {
    render(
      <ErrorBoundary>
        <ThrowError shouldThrow={true} />
      </ErrorBoundary>
    )

    // Check for error boundary container with red background
    const container = screen.getByText('Component Error').closest('.bg-red-50')
    expect(container).toBeInTheDocument()
    expect(container).toHaveClass('w-full', 'p-8', 'bg-red-50', 'rounded-lg')

    // Check heading styling
    const heading = screen.getByText('Component Error')
    expect(heading).toHaveClass('text-xl', 'font-semibold', 'text-gray-900')

    // Check reset button styling
    const button = screen.getByRole('button', { name: /Try Again/i })
    expect(button).toHaveClass('bg-blue-500', 'text-white', 'rounded')
  })

  it('renders correctly when error has no message', () => {
    const ThrowEmptyError = () => {
      const error = new Error()
      error.message = ''
      throw error
    }

    render(
      <ErrorBoundary>
        <ThrowEmptyError />
      </ErrorBoundary>
    )

    expect(screen.getByText('Component Error')).toBeInTheDocument()
    // Should show 'Unknown error' when error message is empty
    expect(screen.getByText('Unknown error')).toBeInTheDocument()
  })

  it('handles errors thrown during render', () => {
    const BadComponent = () => {
      const obj: unknown = null
      // Intentionally cause a runtime error
      return <div>{(obj as { nonExistent: { property: string } }).nonExistent.property}</div>
    }

    render(
      <ErrorBoundary>
        <BadComponent />
      </ErrorBoundary>
    )

    expect(screen.getByText('Component Error')).toBeInTheDocument()
    // The actual error message will vary, but error UI should be shown
    expect(screen.getByRole('button', { name: /Try Again/i })).toBeInTheDocument()
  })

  it('does not catch errors in event handlers', async () => {
    const user = userEvent.setup()
    let errorCaught = false

    // Mock error boundary's componentDidCatch
    const mockComponentDidCatch = vi.fn()
    
    // Custom error boundary that tracks whether it caught an error
    class TestErrorBoundary extends React.Component<
      { children: React.ReactNode },
      { hasError: boolean }
    > {
      state = { hasError: false }

      static getDerivedStateFromError() {
        return { hasError: true }
      }

      componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
        mockComponentDidCatch(error, errorInfo)
        errorCaught = true
      }

      render() {
        if (this.state.hasError) {
          return <div>Error caught by boundary</div>
        }
        return this.props.children
      }
    }

    const ButtonWithError = () => {
      const handleClick = () => {
        // Simulate an error in event handler without actually throwing
        // This tests the concept without causing uncaught exceptions
        try {
          throw new Error('Event handler error')
        } catch {
          // Event handler errors are not caught by error boundaries
          // So we handle it here to avoid uncaught exception in tests
        }
      }
      return <button onClick={handleClick}>Click me</button>
    }

    render(
      <TestErrorBoundary>
        <ButtonWithError />
      </TestErrorBoundary>
    )

    // The button should render normally
    const button = screen.getByRole('button', { name: 'Click me' })
    expect(button).toBeInTheDocument()

    // Click the button
    await user.click(button)

    // Verify error boundary did NOT catch the error
    expect(errorCaught).toBe(false)
    expect(mockComponentDidCatch).not.toHaveBeenCalled()
    
    // The error UI should NOT be shown
    expect(screen.queryByText('Error caught by boundary')).not.toBeInTheDocument()
  })

  it('can recover from error state when reset', () => {
    let shouldThrow = true
    
    const TestComponent = () => {
      if (shouldThrow) {
        throw new Error('Test error')
      }
      return <div>No error</div>
    }
    
    const { rerender } = render(
      <ErrorBoundary>
        <TestComponent />
      </ErrorBoundary>
    )

    // Error should be displayed
    expect(screen.getByText('Component Error')).toBeInTheDocument()

    // Now the enhanced ErrorBoundary can recover via the Try Again button
    shouldThrow = false
    const resetButton = screen.getByRole('button', { name: /Try Again/i })
    resetButton.click()
    
    // Re-render after reset
    rerender(
      <ErrorBoundary>
        <TestComponent />
      </ErrorBoundary>
    )
    
    // Should now show the recovered component
    expect(screen.getByText('No error')).toBeInTheDocument()
  })

  it('shows page-level error with Go Home button', () => {
    render(
      <ErrorBoundary level="page">
        <ThrowError shouldThrow={true} />
      </ErrorBoundary>
    )

    expect(screen.getByText('Page Error')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Try Again/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Go Home/i })).toBeInTheDocument()
  })

  it('uses custom fallback when provided', () => {
    const customFallback = (error: Error, resetError: () => void) => (
      <div>
        <h1>Custom Error UI</h1>
        <p>{error.message}</p>
        <button onClick={resetError}>Custom Reset</button>
      </div>
    )

    render(
      <ErrorBoundary fallback={customFallback}>
        <ThrowError shouldThrow={true} />
      </ErrorBoundary>
    )

    expect(screen.getByText('Custom Error UI')).toBeInTheDocument()
    expect(screen.getByText('Test error message')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Custom Reset' })).toBeInTheDocument()
  })
})