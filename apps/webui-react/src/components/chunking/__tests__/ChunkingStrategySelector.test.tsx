import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ChunkingStrategySelector } from '../ChunkingStrategySelector';
import { useChunkingStore } from '../../../stores/chunkingStore';
import type { ChunkingStore } from '../../../stores/chunkingStore';

// Mock the chunking store
vi.mock('../../../stores/chunkingStore', () => ({
  useChunkingStore: vi.fn()
}));

describe('ChunkingStrategySelector', () => {
  const mockSetStrategy = vi.fn();
  const mockGetRecommendedStrategy = vi.fn();

  const defaultStoreState: Partial<ChunkingStore> = {
    selectedStrategy: 'recursive',
    setStrategy: mockSetStrategy,
    getRecommendedStrategy: mockGetRecommendedStrategy,
    strategyConfig: {
      strategy: 'recursive',
      parameters: {
        chunk_size: 600,
        chunk_overlap: 100
      }
    },
    customPresets: [],
    loadPreview: vi.fn(),
    updateConfiguration: vi.fn(),
    applyPreset: vi.fn(),
    saveCustomPreset: vi.fn(),
    selectedPreset: null,
    previewDocument: null,
    previewLoading: false
  };

  beforeEach(() => {
    vi.clearAllMocks();
    (useChunkingStore as unknown as vi.Mock).mockReturnValue(defaultStoreState);
    mockGetRecommendedStrategy.mockReturnValue('recursive');
  });

  it('renders all chunking strategies', () => {
    render(<ChunkingStrategySelector />);
    
    expect(screen.getByText('Character-based')).toBeInTheDocument();
    expect(screen.getByText('Recursive')).toBeInTheDocument();
    expect(screen.getByText('Markdown-aware')).toBeInTheDocument();
    expect(screen.getByText('Semantic')).toBeInTheDocument();
    expect(screen.getByText('Hierarchical')).toBeInTheDocument();
    expect(screen.getByText('Hybrid Auto-Select')).toBeInTheDocument();
  });

  it('highlights the selected strategy', () => {
    render(<ChunkingStrategySelector />);
    
    const recursiveCard = screen.getByRole('button', { name: /Select Recursive strategy/i });
    // Check the parent div for the border classes
    expect(recursiveCard.parentElement).toHaveClass('border-blue-500', 'bg-blue-50');
  });

  it('shows recommended badge for recommended strategies', () => {
    render(<ChunkingStrategySelector />);
    
    // Recursive and Hybrid strategies are marked as recommended
    const recommendedBadges = screen.getAllByText('Recommended');
    expect(recommendedBadges).toHaveLength(2);
  });

  it('displays performance indicators for each strategy', () => {
    render(<ChunkingStrategySelector />);
    
    // Check for performance indicator labels
    expect(screen.getAllByText('Fast')).toBeTruthy();
    expect(screen.getAllByText('Good')).toBeTruthy();
    expect(screen.getAllByText('Low')).toBeTruthy();
  });

  it('calls setStrategy when a strategy is selected', () => {
    const mockOnStrategyChange = vi.fn();
    render(<ChunkingStrategySelector onStrategyChange={mockOnStrategyChange} />);
    
    const semanticCard = screen.getByRole('button', { name: /Select Semantic strategy/i });
    fireEvent.click(semanticCard);
    
    expect(mockSetStrategy).toHaveBeenCalledWith('semantic');
    expect(mockOnStrategyChange).toHaveBeenCalledWith('semantic');
  });

  it('shows file type recommendation when fileType is provided', () => {
    render(<ChunkingStrategySelector fileType="md" />);
    
    expect(screen.getByText('Detected file type:')).toBeInTheDocument();
    expect(screen.getByText('md')).toBeInTheDocument();
  });

  it('expands and collapses strategy details', () => {
    render(<ChunkingStrategySelector />);
    
    const showDetailsButton = screen.getAllByText('Show Details')[0];
    fireEvent.click(showDetailsButton);
    
    // Check if details are shown
    expect(screen.getByText('Parameters')).toBeInTheDocument();
    expect(screen.getByText('Supported Files')).toBeInTheDocument();
    expect(screen.getByText('How it works')).toBeInTheDocument();
    
    // Click to hide details
    const hideDetailsButton = screen.getByText('Hide Details');
    fireEvent.click(hideDetailsButton);
    
    // Details should be hidden
    expect(screen.queryByText('Parameters')).not.toBeInTheDocument();
  });

  it('disables interaction when disabled prop is true', () => {
    render(<ChunkingStrategySelector disabled={true} />);
    
    const strategyCards = screen.getAllByRole('button', { name: /Select .* strategy/i });
    strategyCards.forEach(card => {
      expect(card.parentElement).toHaveClass('opacity-50', 'cursor-not-allowed');
    });
  });

  it('shows and hides parameter tuner', () => {
    render(<ChunkingStrategySelector />);
    
    // Initially parameter tuner should not be visible
    expect(screen.queryByText('Configure Parameters')).toBeInTheDocument();
    
    // Click to show parameter tuner
    const configureButton = screen.getByText('Configure Parameters');
    fireEvent.click(configureButton);
    
    // Parameter tuner should be visible (checking for presence of ChunkingParameterTuner)
    expect(screen.getByText('Preset Configuration')).toBeInTheDocument();
  });

  it('displays strategy-specific recommendations', () => {
    render(<ChunkingStrategySelector />);
    
    // Expand details for Markdown strategy
    const markdownCard = screen.getByRole('button', { name: /Select Markdown-aware strategy/i });
    const showDetailsButton = markdownCard.parentElement?.querySelector('button:last-child');
    
    if (showDetailsButton) {
      fireEvent.click(showDetailsButton);
      
      // Check for Markdown-specific recommendations
      expect(screen.getByText(/Documentation/)).toBeInTheDocument();
    }
  });

  it('handles keyboard navigation', () => {
    render(<ChunkingStrategySelector />);
    
    const characterCard = screen.getByRole('button', { name: /Select Character-based strategy/i });
    
    // Simulate Enter key press
    fireEvent.keyDown(characterCard, { key: 'Enter' });
    expect(mockSetStrategy).toHaveBeenCalledWith('character');
    
    // Simulate Space key press
    vi.clearAllMocks();
    fireEvent.keyDown(characterCard, { key: ' ' });
    expect(mockSetStrategy).toHaveBeenCalledWith('character');
  });

  it('displays correct icons for each strategy', () => {
    render(<ChunkingStrategySelector />);
    
    // The component should render icon containers
    const iconContainers = screen.getAllByRole('button', { name: /Select .* strategy/i })
      .map(button => button.querySelector('svg'));
    
    // Should have 6 icons (one for each strategy)
    expect(iconContainers.filter(Boolean)).toHaveLength(6);
  });

  it('shows technical details when expanded', () => {
    render(<ChunkingStrategySelector />);
    
    // Expand semantic strategy details
    const semanticCard = screen.getByRole('button', { name: /Select Semantic strategy/i });
    const showDetailsButton = semanticCard.parentElement?.querySelector('button:last-child');
    
    if (showDetailsButton) {
      fireEvent.click(showDetailsButton);
      
      // Check for semantic-specific technical details - use more specific text
      expect(screen.getByText(/Uses AI embeddings to identify topic changes/)).toBeInTheDocument();
      expect(screen.getByText(/semantic similarity between sentences/)).toBeInTheDocument();
    }
  });
});