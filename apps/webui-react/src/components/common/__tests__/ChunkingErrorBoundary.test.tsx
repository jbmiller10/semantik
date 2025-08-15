import { render, screen, fireEvent } from '@testing-library/react';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import ChunkingErrorBoundary, { ChunkingErrorBoundaryWrapper } from '../ChunkingErrorBoundary';
import { useChunkingStore } from '../../../stores/chunkingStore';

// Mock the chunking store
vi.mock('../../../stores/chunkingStore', () => ({
  useChunkingStore: vi.fn()
}));

// Component that always throws an error
function ThrowError({ message = 'Test error' }: { message?: string }) {
  throw new Error(message);
}

describe('ChunkingErrorBoundary', () => {
  const mockResetToDefaults = vi.fn();
  
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(useChunkingStore).mockReturnValue({
      resetToDefaults: mockResetToDefaults
    });
    // Suppress console.error during tests
    vi.spyOn(console, 'error').mockImplementation(() => {});
  });

  it('renders children when there is no error', () => {
    render(
      <ChunkingErrorBoundary>
        <div>Test content</div>
      </ChunkingErrorBoundary>
    );
    
    expect(screen.getByText('Test content')).toBeInTheDocument();
  });

  it('catches and displays errors from children', () => {
    render(
      <ChunkingErrorBoundary componentName="TestComponent">
        <ThrowError message="Test chunking error" />
      </ChunkingErrorBoundary>
    );
    
    expect(screen.getByText('Chunking Component Error')).toBeInTheDocument();
    expect(screen.getByText('The TestComponent encountered an error.')).toBeInTheDocument();
    expect(screen.getByText('Test chunking error')).toBeInTheDocument();
  });

  it('shows WebSocket-specific help for WebSocket errors', () => {
    render(
      <ChunkingErrorBoundary>
        <ThrowError message="WebSocket connection failed" />
      </ChunkingErrorBoundary>
    );
    
    expect(screen.getByText('Connection Issue Detected')).toBeInTheDocument();
    expect(screen.getByText(/This might be a temporary network issue/)).toBeInTheDocument();
  });

  it('shows configuration-specific help for configuration errors', () => {
    render(
      <ChunkingErrorBoundary>
        <ThrowError message="Invalid configuration parameter" />
      </ChunkingErrorBoundary>
    );
    
    expect(screen.getByText('Configuration Issue')).toBeInTheDocument();
    expect(screen.getByText(/The current chunking configuration may be invalid/)).toBeInTheDocument();
  });

  it('allows resetting the error boundary', () => {
    let shouldThrow = true;
    
    function TestComponent() {
      if (shouldThrow) {
        throw new Error('Test error');
      }
      return <div>No error</div>;
    }
    
    const { rerender } = render(
      <ChunkingErrorBoundary>
        <TestComponent />
      </ChunkingErrorBoundary>
    );
    
    expect(screen.getByText('Chunking Component Error')).toBeInTheDocument();
    
    // Set shouldThrow to false before clicking reset
    shouldThrow = false;
    
    // Click reset button
    fireEvent.click(screen.getByRole('button', { name: /Try Again/i }));
    
    // Force a re-render
    rerender(
      <ChunkingErrorBoundary>
        <TestComponent />
      </ChunkingErrorBoundary>
    );
    
    expect(screen.getByText('No error')).toBeInTheDocument();
  });

  it('shows reset configuration button for configuration errors', () => {
    render(
      <ChunkingErrorBoundary>
        <ThrowError message="Invalid configuration" />
      </ChunkingErrorBoundary>
    );
    
    expect(screen.getByRole('button', { name: /Reset Settings/i })).toBeInTheDocument();
  });

  it('shows technical details section', () => {
    render(
      <ChunkingErrorBoundary componentName="TestComponent">
        <ThrowError message="Test error" />
      </ChunkingErrorBoundary>
    );
    
    const summaryElement = screen.getByText('Show technical details');
    expect(summaryElement).toBeInTheDocument();
    
    // The details element contains component information
    // Click the summary to ensure it's expanded
    fireEvent.click(summaryElement);
    
    // Now details should be visible
    expect(screen.getByText('Component:')).toBeInTheDocument();
    expect(screen.getByText('TestComponent')).toBeInTheDocument();
  });
});

describe('ChunkingErrorBoundaryWrapper', () => {
  const mockResetToDefaults = vi.fn();
  
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(useChunkingStore).mockImplementation((selector: (state: { resetToDefaults: () => void }) => unknown) => {
      const state = {
        resetToDefaults: mockResetToDefaults
      };
      return selector ? selector(state) : state;
    });
    // Also mock getState for static access
    vi.mocked(useChunkingStore).getState = vi.fn().mockReturnValue({
      resetToDefaults: mockResetToDefaults
    });
    vi.spyOn(console, 'error').mockImplementation(() => {});
    vi.spyOn(console, 'log').mockImplementation(() => {});
  });

  it('renders children normally when no error', () => {
    render(
      <ChunkingErrorBoundaryWrapper componentName="TestWrapper">
        <div>Test content</div>
      </ChunkingErrorBoundaryWrapper>
    );
    
    expect(screen.getByText('Test content')).toBeInTheDocument();
  });

  it('catches errors and displays fallback UI', () => {
    render(
      <ChunkingErrorBoundaryWrapper componentName="TestWrapper">
        <ThrowError message="Wrapper test error" />
      </ChunkingErrorBoundaryWrapper>
    );
    
    expect(screen.getByText('Chunking Component Error')).toBeInTheDocument();
    expect(screen.getByText('The TestWrapper encountered an error.')).toBeInTheDocument();
  });

  it('calls resetToDefaults when resetting without preserveConfiguration', () => {
    render(
      <ChunkingErrorBoundaryWrapper componentName="TestWrapper" preserveConfiguration={false}>
        <ThrowError message="Test error" />
      </ChunkingErrorBoundaryWrapper>
    );
    
    fireEvent.click(screen.getByRole('button', { name: /Try Again/i }));
    
    expect(mockResetToDefaults).toHaveBeenCalled();
  });

  it('preserves configuration when preserveConfiguration is true', () => {
    render(
      <ChunkingErrorBoundaryWrapper componentName="TestWrapper" preserveConfiguration={true}>
        <ThrowError message="Test error" />
      </ChunkingErrorBoundaryWrapper>
    );
    
    fireEvent.click(screen.getByRole('button', { name: /Try Again/i }));
    
    // Should not call resetToDefaults when preserving configuration
    expect(mockResetToDefaults).not.toHaveBeenCalled();
    expect(console.log).toHaveBeenCalledWith('Preserving chunking configuration during reset');
  });
});