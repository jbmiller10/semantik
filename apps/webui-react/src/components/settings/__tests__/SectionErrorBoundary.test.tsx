import { render, screen, fireEvent } from '@testing-library/react';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import SectionErrorBoundary from '../SectionErrorBoundary';

// Component that always throws an error
function ThrowError({ message = 'Test error' }: { message?: string }): never {
  throw new Error(message);
}

describe('SectionErrorBoundary', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Suppress console.error during tests
    vi.spyOn(console, 'error').mockImplementation(() => {});
  });

  describe('normal rendering', () => {
    it('renders children when there is no error', () => {
      render(
        <SectionErrorBoundary sectionName="Test Section">
          <div>Test content</div>
        </SectionErrorBoundary>
      );

      expect(screen.getByText('Test content')).toBeInTheDocument();
    });
  });

  describe('error handling', () => {
    it('catches errors and displays fallback UI', () => {
      render(
        <SectionErrorBoundary sectionName="Test Section">
          <ThrowError message="Test error message" />
        </SectionErrorBoundary>
      );

      expect(screen.getByText('Failed to load Test Section')).toBeInTheDocument();
      expect(screen.getByText('Test error message')).toBeInTheDocument();
    });

    it('shows Try Again button', () => {
      render(
        <SectionErrorBoundary sectionName="Test Section">
          <ThrowError />
        </SectionErrorBoundary>
      );

      expect(
        screen.getByRole('button', { name: /Try Again/i })
      ).toBeInTheDocument();
    });

    it('shows technical details in expandable section', () => {
      render(
        <SectionErrorBoundary sectionName="Test Section">
          <ThrowError />
        </SectionErrorBoundary>
      );

      expect(screen.getByText('Technical details')).toBeInTheDocument();
    });

    it('expands technical details on click', () => {
      render(
        <SectionErrorBoundary sectionName="Test Section">
          <ThrowError message="Test error" />
        </SectionErrorBoundary>
      );

      const summary = screen.getByText('Technical details');
      fireEvent.click(summary);

      // Error stack should be visible in the expanded details (check for pre element with stack)
      const pre = document.querySelector('pre');
      expect(pre).toBeInTheDocument();
      expect(pre?.textContent).toContain('Error: Test error');
    });
  });

  describe('recovery', () => {
    it('resets error state when Try Again is clicked', () => {
      let shouldThrow = true;

      function TestComponent() {
        if (shouldThrow) {
          throw new Error('Test error');
        }
        return <div>No error</div>;
      }

      const { rerender } = render(
        <SectionErrorBoundary sectionName="Test Section">
          <TestComponent />
        </SectionErrorBoundary>
      );

      expect(screen.getByText('Failed to load Test Section')).toBeInTheDocument();

      shouldThrow = false;
      fireEvent.click(screen.getByRole('button', { name: /Try Again/i }));

      rerender(
        <SectionErrorBoundary sectionName="Test Section">
          <TestComponent />
        </SectionErrorBoundary>
      );

      expect(screen.getByText('No error')).toBeInTheDocument();
    });

    it('calls onRetry callback when provided', () => {
      const onRetry = vi.fn();

      render(
        <SectionErrorBoundary sectionName="Test Section" onRetry={onRetry}>
          <ThrowError />
        </SectionErrorBoundary>
      );

      fireEvent.click(screen.getByRole('button', { name: /Try Again/i }));

      expect(onRetry).toHaveBeenCalled();
    });
  });

  describe('error isolation', () => {
    it('does not affect sibling sections', () => {
      render(
        <div>
          <SectionErrorBoundary sectionName="Working Section">
            <div>Working content</div>
          </SectionErrorBoundary>
          <SectionErrorBoundary sectionName="Broken Section">
            <ThrowError />
          </SectionErrorBoundary>
        </div>
      );

      // Working section should still be visible
      expect(screen.getByText('Working content')).toBeInTheDocument();
      // Broken section should show error
      expect(screen.getByText('Failed to load Broken Section')).toBeInTheDocument();
    });
  });

  describe('logging', () => {
    it('logs error to console with section name', () => {
      const consoleError = vi.spyOn(console, 'error');

      render(
        <SectionErrorBoundary sectionName="Test Section">
          <ThrowError message="Logged error" />
        </SectionErrorBoundary>
      );

      expect(consoleError).toHaveBeenCalledWith(
        '[SectionErrorBoundary] Error in Test Section:',
        expect.objectContaining({
          error: expect.any(Error),
          timestamp: expect.any(String),
        })
      );
    });
  });
});
