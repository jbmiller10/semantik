import { render, screen, fireEvent } from '@testing-library/react';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import { ErrorFallback } from '../ErrorFallback';
import { 
  ChunkingErrorFallback,
  PreviewErrorFallback,
  ComparisonErrorFallback,
  AnalyticsErrorFallback,
  ConfigurationErrorFallback
} from '../ChunkingErrorFallback';

describe('ErrorFallback', () => {
  const mockReset = vi.fn();
  const testError = new Error('Test error message');

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders error message and reset button', () => {
    render(
      <ErrorFallback 
        error={testError} 
        resetError={mockReset}
      />
    );
    
    expect(screen.getByText('Component Error')).toBeInTheDocument();
    expect(screen.getByText('Test error message')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Try Again/i })).toBeInTheDocument();
  });

  it('displays custom component name', () => {
    render(
      <ErrorFallback 
        error={testError} 
        resetError={mockReset}
        componentName="Custom Component"
      />
    );
    
    expect(screen.getByText('Custom Component Error')).toBeInTheDocument();
  });

  it('displays custom suggestion', () => {
    render(
      <ErrorFallback 
        error={testError} 
        resetError={mockReset}
        suggestion="Please check your network connection"
      />
    );
    
    expect(screen.getByText('Please check your network connection')).toBeInTheDocument();
  });

  it('toggles error details when clicked', () => {
    render(
      <ErrorFallback 
        error={testError} 
        resetError={mockReset}
        showDetails={true}
      />
    );
    
    // Initially collapsed
    expect(screen.getByText('Show details')).toBeInTheDocument();
    // Check for a pre element with overflow-auto class (the stack trace container)
    expect(screen.queryByText(/Error: Test error message/)).not.toBeInTheDocument();
    
    // Click to expand
    fireEvent.click(screen.getByText('Show details'));
    expect(screen.getByText('Hide details')).toBeInTheDocument();
    // Now the error stack should be visible
    const preElement = screen.getByText((content, element) => {
      return element?.tagName === 'PRE' && content.includes('Error: Test error message');
    });
    expect(preElement).toBeInTheDocument();
    
    // Click to collapse
    fireEvent.click(screen.getByText('Hide details'));
    expect(screen.getByText('Show details')).toBeInTheDocument();
  });

  it('hides details section when showDetails is false', () => {
    render(
      <ErrorFallback 
        error={testError} 
        resetError={mockReset}
        showDetails={false}
      />
    );
    
    expect(screen.queryByText('Show details')).not.toBeInTheDocument();
  });

  it('calls resetError when Try Again is clicked', () => {
    render(
      <ErrorFallback 
        error={testError} 
        resetError={mockReset}
      />
    );
    
    fireEvent.click(screen.getByRole('button', { name: /Try Again/i }));
    expect(mockReset).toHaveBeenCalledTimes(1);
  });
});

describe('ChunkingErrorFallback', () => {
  const mockReset = vi.fn();
  const mockResetConfig = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders preview variant correctly', () => {
    const error = new Error('Preview generation failed');
    render(
      <ChunkingErrorFallback 
        error={error} 
        resetError={mockReset}
        variant="preview"
      />
    );
    
    expect(screen.getByText('Preview Generation Failed')).toBeInTheDocument();
    expect(screen.getByText(/Unable to generate preview/)).toBeInTheDocument();
  });

  it('renders comparison variant correctly', () => {
    const error = new Error('Comparison failed');
    render(
      <ChunkingErrorFallback 
        error={error} 
        resetError={mockReset}
        variant="comparison"
      />
    );
    
    expect(screen.getByText('Strategy Comparison Failed')).toBeInTheDocument();
    expect(screen.getByText(/Could not compare strategies/)).toBeInTheDocument();
  });

  it('renders analytics variant correctly', () => {
    const error = new Error('Analytics failed');
    render(
      <ChunkingErrorFallback 
        error={error} 
        resetError={mockReset}
        variant="analytics"
      />
    );
    
    expect(screen.getByText('Analytics Loading Failed')).toBeInTheDocument();
    expect(screen.getByText(/Failed to load analytics data/)).toBeInTheDocument();
  });

  it('shows network error suggestions for network errors', () => {
    const error = new Error('WebSocket connection failed');
    render(
      <ChunkingErrorFallback 
        error={error} 
        resetError={mockReset}
      />
    );
    
    expect(screen.getByText('Connection Tips:')).toBeInTheDocument();
    expect(screen.getByText(/Check if the backend services are running/)).toBeInTheDocument();
  });

  it('shows configuration error suggestions for config errors', () => {
    const error = new Error('Invalid configuration parameter');
    render(
      <ChunkingErrorFallback 
        error={error} 
        resetError={mockReset}
      />
    );
    
    expect(screen.getByText('Configuration Tips:')).toBeInTheDocument();
    expect(screen.getByText(/Chunk size should be between 100-10000 characters/)).toBeInTheDocument();
  });

  it('shows alternative actions for preview variant', () => {
    const error = new Error('Preview error');
    render(
      <ChunkingErrorFallback 
        error={error} 
        resetError={mockReset}
        variant="preview"
      />
    );
    
    expect(screen.getByText('Alternative Actions:')).toBeInTheDocument();
    expect(screen.getByText(/Try a different chunking strategy/)).toBeInTheDocument();
  });

  it('shows reset configuration button for config errors', () => {
    const error = new Error('Invalid configuration');
    render(
      <ChunkingErrorFallback 
        error={error} 
        resetError={mockReset}
        variant="configuration"
        onResetConfiguration={mockResetConfig}
      />
    );
    
    const resetButton = screen.getByRole('button', { name: /Reset to Defaults/i });
    expect(resetButton).toBeInTheDocument();
    
    fireEvent.click(resetButton);
    expect(mockResetConfig).toHaveBeenCalledTimes(1);
  });

  it('expands technical details', () => {
    const error = new Error('Test error');
    render(
      <ChunkingErrorFallback 
        error={error} 
        resetError={mockReset}
        variant="preview"
      />
    );
    
    const detailsElement = screen.getByText('Technical details for developers');
    fireEvent.click(detailsElement);
    
    // Check that the technical details section is rendered
    expect(screen.getByText('Error Type:')).toBeInTheDocument();
    expect(screen.getByText('Variant:')).toBeInTheDocument();
    expect(screen.getByText('Timestamp:')).toBeInTheDocument();
    expect(screen.getByText('Stack Trace:')).toBeInTheDocument();
    
    // The details div should contain the word preview as the variant value
    const detailsDiv = screen.getByText('Variant:').closest('div');
    expect(detailsDiv).toHaveTextContent('Variant: preview');
  });
});

describe('Specialized Error Fallbacks', () => {
  const mockReset = vi.fn();
  const testError = new Error('Test error');

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('PreviewErrorFallback renders with preview variant', () => {
    render(<PreviewErrorFallback error={testError} resetError={mockReset} />);
    expect(screen.getByText('Preview Generation Failed')).toBeInTheDocument();
  });

  it('ComparisonErrorFallback renders with comparison variant', () => {
    render(<ComparisonErrorFallback error={testError} resetError={mockReset} />);
    expect(screen.getByText('Strategy Comparison Failed')).toBeInTheDocument();
  });

  it('AnalyticsErrorFallback renders with analytics variant', () => {
    render(<AnalyticsErrorFallback error={testError} resetError={mockReset} />);
    expect(screen.getByText('Analytics Loading Failed')).toBeInTheDocument();
  });

  it('ConfigurationErrorFallback renders with configuration variant', () => {
    const mockResetConfig = vi.fn();
    render(
      <ConfigurationErrorFallback 
        error={testError} 
        resetError={mockReset}
        onResetConfiguration={mockResetConfig}
      />
    );
    expect(screen.getByText('Configuration Error')).toBeInTheDocument();
    
    // Should have reset config button
    const resetConfigButton = screen.getByRole('button', { name: /Reset to Defaults/i });
    expect(resetConfigButton).toBeInTheDocument();
    
    fireEvent.click(resetConfigButton);
    expect(mockResetConfig).toHaveBeenCalledTimes(1);
  });
});