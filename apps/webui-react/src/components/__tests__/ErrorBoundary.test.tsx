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
    const details = screen.getByRole('group')
    expect(details).toBeInTheDocument()
    
    // Check that the stack trace contains expected text
    const stackTrace = screen.getByText(/Error: Test error with stack/)
    expect(stackTrace).toBeInTheDocument()
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
    expect(button).toHaveClass('bg-blue-500', 'text-white', 'rounded', 'hover:bg-blue-600')
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
    const errorParagraph = screen.getByText('Something went wrong').parentElement?.querySelector('p')
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
    let errorThrown = false

    const ButtonWithError = () => {
      const handleClick = () => {
        errorThrown = true
        throw new Error('Event handler error')
      }
      return <button onClick={handleClick}>Click me</button>
    }

    // Temporarily suppress error logging for this test
    const originalError = console.error
    console.error = vi.fn()

    render(
      <ErrorBoundary>
        <ButtonWithError />
      </ErrorBoundary>
    )

    // The button should render normally
    const button = screen.getByRole('button', { name: 'Click me' })
    expect(button).toBeInTheDocument()

    // Use a try-catch to handle the error from event handler
    try {
      await user.click(button)
    } catch {
      // Expected - event handler errors are not caught by ErrorBoundary
    }

    // Verify the error was thrown
    expect(errorThrown).toBe(true)

    // The error UI should NOT be shown (ErrorBoundary doesn't catch event handler errors)
    expect(screen.queryByText('Something went wrong')).not.toBeInTheDocument()

    // Restore console.error
    console.error = originalError
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