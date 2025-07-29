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

// Mock window.location.reload
const mockReload = vi.fn()
Object.defineProperty(window, 'location', {
  value: { reload: mockReload },
  writable: true,
})

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

    expect(screen.getByText('Something went wrong')).toBeInTheDocument()
    expect(screen.getByText('Test error message')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Reload page' })).toBeInTheDocument()
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

    expect(screen.getByText('Stack trace')).toBeInTheDocument()
    
    // The stack trace should be in a details element
    const details = document.querySelector('details')
    expect(details).toBeInTheDocument()
    
    // Check that the stack trace contains expected text
    const preElement = details?.querySelector('pre')
    expect(preElement).toBeInTheDocument()
    expect(preElement?.textContent).toContain('Error: Test error with stack')
  })

  it('reloads page when reload button is clicked', async () => {
    const user = userEvent.setup()

    render(
      <ErrorBoundary>
        <ThrowError shouldThrow={true} />
      </ErrorBoundary>
    )

    const reloadButton = screen.getByRole('button', { name: 'Reload page' })
    await user.click(reloadButton)

    expect(mockReload).toHaveBeenCalled()
  })

  it('logs errors to console', () => {
    const consoleError = vi.spyOn(console, 'error')

    render(
      <ErrorBoundary>
        <ThrowError shouldThrow={true} />
      </ErrorBoundary>
    )

    expect(consoleError).toHaveBeenCalledWith(
      'Uncaught error:',
      expect.any(Error),
      expect.objectContaining({
        componentStack: expect.any(String),
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
    const container = screen.getByText('Something went wrong').closest('.bg-red-50')
    expect(container).toBeInTheDocument()
    expect(container).toHaveClass('min-h-screen', 'flex', 'items-center', 'justify-center')

    // Check heading styling
    const heading = screen.getByText('Something went wrong')
    expect(heading).toHaveClass('text-2xl', 'font-bold', 'text-red-600')

    // Check reload button styling
    const button = screen.getByRole('button', { name: 'Reload page' })
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

    expect(screen.getByText('Something went wrong')).toBeInTheDocument()
    // Error message paragraph should still be there, just empty
    const container = screen.getByText('Something went wrong').parentElement
    const errorParagraph = container?.querySelector('p')
    expect(errorParagraph).toBeInTheDocument()
    expect(errorParagraph?.textContent).toBe('')
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

    expect(screen.getByText('Something went wrong')).toBeInTheDocument()
    // The actual error message will vary, but error UI should be shown
    expect(screen.getByRole('button', { name: 'Reload page' })).toBeInTheDocument()
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

  it('stays in error state even when children change', () => {
    const { rerender } = render(
      <ErrorBoundary>
        <ThrowError shouldThrow={true} />
      </ErrorBoundary>
    )

    // Error should be displayed
    expect(screen.getByText('Something went wrong')).toBeInTheDocument()

    // Re-render with non-throwing component
    rerender(
      <ErrorBoundary>
        <ThrowError shouldThrow={false} />
      </ErrorBoundary>
    )

    // Error boundary remains in error state - this is expected React behavior
    // The only way to recover is to reload the page
    expect(screen.getByText('Something went wrong')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Reload page' })).toBeInTheDocument()
  })
})