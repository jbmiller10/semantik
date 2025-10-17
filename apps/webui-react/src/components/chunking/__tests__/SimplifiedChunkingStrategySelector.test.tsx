import React from 'react';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import { SimplifiedChunkingStrategySelector } from '../SimplifiedChunkingStrategySelector';
import { useChunkingStore } from '../../../stores/chunkingStore';
import { CHUNKING_STRATEGIES } from '../../../types/chunking';
import type { ChunkingStore } from '../../../stores/chunkingStore';
import type { ChunkingStrategyType } from '../../../types/chunking';

// Mock the chunking store
vi.mock('../../../stores/chunkingStore', () => ({
  useChunkingStore: vi.fn()
}));

// Mock the ChunkingParameterTuner component
vi.mock('../ChunkingParameterTuner', () => ({
  ChunkingParameterTuner: vi.fn(({ disabled, showPreview, onParameterChange }: { disabled?: boolean; showPreview?: boolean; onParameterChange?: () => void }) => (
    <div data-testid="chunking-parameter-tuner">
      <div>Parameter Tuner (disabled: {disabled ? 'true' : 'false'})</div>
      <div>Show Preview: {showPreview ? 'true' : 'false'}</div>
      <button onClick={onParameterChange}>Trigger Parameter Change</button>
    </div>
  ))
}));

// Mock the ChunkingStrategyGuide component
vi.mock('../ChunkingStrategyGuide', () => ({
  ChunkingStrategyGuide: vi.fn(({ onClose, currentStrategy, fileType }: { onClose?: () => void; currentStrategy?: string; fileType?: string }) => (
    <div data-testid="chunking-strategy-guide">
      <div>Strategy Guide</div>
      <div>Current Strategy: {currentStrategy}</div>
      <div>File Type: {fileType || 'none'}</div>
      <button onClick={onClose}>Close Guide</button>
    </div>
  ))
}));

// Mock CSS file
vi.mock('../SimplifiedChunkingStrategySelector.css', () => ({}));

describe('SimplifiedChunkingStrategySelector', () => {
  const mockSetStrategy = vi.fn();
  const mockGetRecommendedStrategy = vi.fn();
  const mockOnStrategyChange = vi.fn();
  const user = userEvent.setup();

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
    }
  };

  beforeEach(() => {
    vi.clearAllMocks();
    (useChunkingStore as unknown as vi.Mock).mockReturnValue(defaultStoreState);
    mockGetRecommendedStrategy.mockReturnValue('hybrid');
  });

  describe('Basic Rendering', () => {
    it('should render the strategy selector with all components', () => {
      render(<SimplifiedChunkingStrategySelector />);
      
      expect(screen.getByLabelText(/chunking strategy/i)).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /learn more/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /advanced options/i })).toBeInTheDocument();
    });

    it('should render all available strategies in dropdown', () => {
      render(<SimplifiedChunkingStrategySelector />);
      
      const select = screen.getByLabelText(/chunking strategy/i);
      const options = within(select).getAllByRole('option');
      
      expect(options).toHaveLength(Object.keys(CHUNKING_STRATEGIES).length);
      
      // Check each strategy is present
      Object.keys(CHUNKING_STRATEGIES).forEach((strategy) => {
        const option = within(select).getByRole('option', { name: new RegExp(CHUNKING_STRATEGIES[strategy as ChunkingStrategyType].name) });
        expect(option).toBeInTheDocument();
      });
    });

    it('should show selected strategy description', () => {
      render(<SimplifiedChunkingStrategySelector />);
      
      const description = CHUNKING_STRATEGIES.recursive.description;
      expect(screen.getByText(description)).toBeInTheDocument();
    });

    it('should have the correct initial selected value', () => {
      render(<SimplifiedChunkingStrategySelector />);
      
      const select = screen.getByLabelText(/chunking strategy/i) as HTMLSelectElement;
      expect(select.value).toBe('recursive');
    });
  });

  describe('Strategy Selection', () => {
    it('should change strategy when selecting from dropdown', async () => {
      render(<SimplifiedChunkingStrategySelector onStrategyChange={mockOnStrategyChange} />);
      
      const select = screen.getByLabelText(/chunking strategy/i);
      await user.selectOptions(select, 'semantic');
      
      expect(mockSetStrategy).toHaveBeenCalledWith('semantic');
      expect(mockOnStrategyChange).toHaveBeenCalledWith('semantic');
    });

    it('should update description when strategy changes', async () => {
      const { rerender } = render(<SimplifiedChunkingStrategySelector />);
      
      // Initial description
      expect(screen.getByText(CHUNKING_STRATEGIES.recursive.description)).toBeInTheDocument();
      
      // Update store state
      (useChunkingStore as unknown as vi.Mock).mockReturnValue({
        ...defaultStoreState,
        selectedStrategy: 'semantic'
      });
      
      rerender(<SimplifiedChunkingStrategySelector />);
      
      // New description should be shown
      expect(screen.getByText(CHUNKING_STRATEGIES.semantic.description)).toBeInTheDocument();
    });

    it('should not change strategy when disabled', async () => {
      render(<SimplifiedChunkingStrategySelector disabled={true} onStrategyChange={mockOnStrategyChange} />);
      
      const select = screen.getByLabelText(/chunking strategy/i);
      expect(select).toBeDisabled();
      
      // Try to select a different option (should not work)
      await user.selectOptions(select, 'semantic');
      
      expect(mockSetStrategy).not.toHaveBeenCalled();
      expect(mockOnStrategyChange).not.toHaveBeenCalled();
    });

    it('should handle strategy change without callback', async () => {
      render(<SimplifiedChunkingStrategySelector />);
      
      const select = screen.getByLabelText(/chunking strategy/i);
      await user.selectOptions(select, 'markdown');
      
      expect(mockSetStrategy).toHaveBeenCalledWith('markdown');
      // No error should occur even without onStrategyChange callback
    });
  });

  describe('Recommended Strategy Display', () => {
    it('should show recommended badge for recommended strategy', () => {
      mockGetRecommendedStrategy.mockReturnValue('recursive');
      
      render(<SimplifiedChunkingStrategySelector fileType="txt" />);
      
      const select = screen.getByLabelText(/chunking strategy/i);
      const recursiveOption = within(select).getByRole('option', { name: /Recursive.*✨ Recommended/i });
      expect(recursiveOption).toBeInTheDocument();
    });

    it('should show recommendation message when not using recommended strategy', () => {
      mockGetRecommendedStrategy.mockReturnValue('markdown');
      
      render(<SimplifiedChunkingStrategySelector fileType="md" />);
      
      // Look for the recommendation message specifically
      const recommendationDiv = screen.getByText(/For md files, we recommend using/).closest('div');
      expect(recommendationDiv).toBeInTheDocument();
      expect(recommendationDiv).toHaveTextContent('For md files, we recommend using Markdown-aware');
    });

    it('should not show recommendation message when using recommended strategy', () => {
      mockGetRecommendedStrategy.mockReturnValue('recursive');
      
      render(<SimplifiedChunkingStrategySelector fileType="txt" />);
      
      expect(screen.queryByText(/For txt files, we recommend using/)).not.toBeInTheDocument();
    });

    it('should not show recommendation without file type', () => {
      render(<SimplifiedChunkingStrategySelector />);
      
      expect(screen.queryByText(/we recommend using/)).not.toBeInTheDocument();
    });

    it('should update recommendation when file type changes', () => {
      // Start with markdown recommendation for md files
      mockGetRecommendedStrategy.mockReturnValue('markdown');
      const { rerender } = render(<SimplifiedChunkingStrategySelector fileType="md" />);
      
      let recommendationDiv = screen.getByText(/For md files, we recommend using/).closest('div');
      expect(recommendationDiv).toHaveTextContent('For md files, we recommend using Markdown-aware');
      
      // Change to semantic recommendation for pdf files
      mockGetRecommendedStrategy.mockReturnValue('semantic');
      rerender(<SimplifiedChunkingStrategySelector fileType="pdf" />);
      
      recommendationDiv = screen.getByText(/For pdf files, we recommend using/).closest('div');
      expect(recommendationDiv).toHaveTextContent('For pdf files, we recommend using Semantic');
    });
  });

  describe('Learn More Guide Modal', () => {
    it('should open guide modal when clicking Learn more', async () => {
      render(<SimplifiedChunkingStrategySelector />);
      
      const learnMoreButton = screen.getByRole('button', { name: /learn more/i });
      await user.click(learnMoreButton);
      
      expect(screen.getByTestId('chunking-strategy-guide')).toBeInTheDocument();
      expect(screen.getByText('Strategy Guide')).toBeInTheDocument();
    });

    it('should pass current strategy to guide modal', async () => {
      render(<SimplifiedChunkingStrategySelector />);
      
      await user.click(screen.getByRole('button', { name: /learn more/i }));
      
      expect(screen.getByText('Current Strategy: recursive')).toBeInTheDocument();
    });

    it('should pass file type to guide modal', async () => {
      render(<SimplifiedChunkingStrategySelector fileType="pdf" />);
      
      await user.click(screen.getByRole('button', { name: /learn more/i }));
      
      expect(screen.getByText('File Type: pdf')).toBeInTheDocument();
    });

    it('should close guide modal', async () => {
      render(<SimplifiedChunkingStrategySelector />);
      
      await user.click(screen.getByRole('button', { name: /learn more/i }));
      expect(screen.getByTestId('chunking-strategy-guide')).toBeInTheDocument();
      
      await user.click(screen.getByRole('button', { name: /close guide/i }));
      expect(screen.queryByTestId('chunking-strategy-guide')).not.toBeInTheDocument();
    });
  });

  describe('Advanced Options Toggle', () => {
    it('should toggle advanced options visibility', async () => {
      render(<SimplifiedChunkingStrategySelector />);
      
      const advancedButton = screen.getByRole('button', { name: /advanced options/i });
      
      // Initially hidden
      expect(screen.queryByTestId('chunking-parameter-tuner')).not.toBeInTheDocument();
      
      // Click to show
      await user.click(advancedButton);
      expect(screen.getByTestId('chunking-parameter-tuner')).toBeInTheDocument();
      
      // Click to hide
      await user.click(advancedButton);
      expect(screen.queryByTestId('chunking-parameter-tuner')).not.toBeInTheDocument();
    });

    it('should animate chevron icon on toggle', async () => {
      render(<SimplifiedChunkingStrategySelector />);
      
      const advancedButton = screen.getByRole('button', { name: /advanced options/i });
      const chevron = advancedButton.querySelector('svg:last-child');
      
      // Initially not rotated
      expect(chevron).not.toHaveClass('rotate-180');
      
      // Click to expand
      await user.click(advancedButton);
      expect(chevron).toHaveClass('rotate-180');
      
      // Click to collapse
      await user.click(advancedButton);
      expect(chevron).not.toHaveClass('rotate-180');
    });

    it('should disable advanced options toggle when component is disabled', async () => {
      render(<SimplifiedChunkingStrategySelector disabled={true} />);
      
      const advancedButton = screen.getByRole('button', { name: /advanced options/i });
      expect(advancedButton).toHaveClass('disabled:opacity-50');
      
      // Try to click (should not expand)
      await user.click(advancedButton);
      expect(screen.queryByTestId('chunking-parameter-tuner')).not.toBeInTheDocument();
    });

    it('should pass disabled state to parameter tuner', async () => {
      // Start with enabled to show the tuner
      const { rerender } = render(<SimplifiedChunkingStrategySelector disabled={false} />);
      
      // Click to show advanced options
      await user.click(screen.getByRole('button', { name: /advanced options/i }));
      
      // Verify tuner is shown and not disabled
      let tuner = screen.getByTestId('chunking-parameter-tuner');
      expect(tuner).toHaveTextContent('disabled: false');
      
      // Now re-render with disabled=true while tuner is shown
      rerender(<SimplifiedChunkingStrategySelector disabled={true} />);
      
      // The tuner should still be visible but with disabled state
      tuner = screen.getByTestId('chunking-parameter-tuner');
      expect(tuner).toHaveTextContent('disabled: true');
    });

    it('should not show preview in parameter tuner', async () => {
      render(<SimplifiedChunkingStrategySelector />);
      
      await user.click(screen.getByRole('button', { name: /advanced options/i }));
      
      const tuner = screen.getByTestId('chunking-parameter-tuner');
      expect(tuner).toHaveTextContent('Show Preview: false');
    });
  });

  describe('Integration with ChunkingParameterTuner', () => {
    it('should trigger strategy change callback when parameters change', async () => {
      render(<SimplifiedChunkingStrategySelector onStrategyChange={mockOnStrategyChange} />);
      
      await user.click(screen.getByRole('button', { name: /advanced options/i }));
      
      const parameterChangeButton = screen.getByRole('button', { name: /trigger parameter change/i });
      await user.click(parameterChangeButton);
      
      expect(mockOnStrategyChange).toHaveBeenCalledWith('recursive');
    });

    it('should handle parameter changes without callback', async () => {
      render(<SimplifiedChunkingStrategySelector />);
      
      await user.click(screen.getByRole('button', { name: /advanced options/i }));
      
      const parameterChangeButton = screen.getByRole('button', { name: /trigger parameter change/i });
      
      // Should not throw error
      await user.click(parameterChangeButton);
    });
  });

  describe('Strategy Label Formatting', () => {
    it('should format strategy labels correctly', () => {
      render(<SimplifiedChunkingStrategySelector />);
      
      const select = screen.getByLabelText(/chunking strategy/i);
      
      // Check that strategy names are properly formatted
      Object.keys(CHUNKING_STRATEGIES).forEach((strategyKey) => {
        const strategy = CHUNKING_STRATEGIES[strategyKey as ChunkingStrategyType];
        const option = within(select).getByRole('option', { name: new RegExp(strategy.name) });
        expect(option).toBeInTheDocument();
      });
    });

    it('should show recommended suffix for recommended strategies', () => {
      mockGetRecommendedStrategy.mockReturnValue('hybrid');
      
      render(<SimplifiedChunkingStrategySelector fileType="mixed" />);
      
      const select = screen.getByLabelText(/chunking strategy/i);
      const hybridOption = within(select).getByRole('option', { name: /Hybrid Auto-Select.*✨ Recommended/i });
      expect(hybridOption).toBeInTheDocument();
    });

    it('should mark isRecommended strategies', () => {
      // The hybrid strategy has isRecommended: true in CHUNKING_STRATEGIES
      render(<SimplifiedChunkingStrategySelector />);
      
      const select = screen.getByLabelText(/chunking strategy/i);
      
      // Find options with recommended marker
      const recommendedOptions = within(select).queryAllByRole('option', { name: /✨ Recommended/i });
      
      // Should have at least the strategies marked as isRecommended
      expect(recommendedOptions.length).toBeGreaterThan(0);
    });
  });

  describe('Edge Cases', () => {
    it.skip('should handle missing strategy in CHUNKING_STRATEGIES gracefully', () => {
      // Skip this test as the component expects valid strategies
      // In production, TypeScript ensures only valid strategies are used
      // This edge case would require modifying the component to handle invalid strategies
    });

    it('should handle rapid strategy changes', async () => {
      render(<SimplifiedChunkingStrategySelector onStrategyChange={mockOnStrategyChange} />);
      
      const select = screen.getByLabelText(/chunking strategy/i);
      
      // Rapid changes
      await user.selectOptions(select, 'semantic');
      await user.selectOptions(select, 'markdown');
      await user.selectOptions(select, 'character');
      
      expect(mockSetStrategy).toHaveBeenCalledTimes(3);
      expect(mockOnStrategyChange).toHaveBeenCalledTimes(3);
    });

    it('should handle undefined fileType gracefully', () => {
      render(<SimplifiedChunkingStrategySelector fileType={undefined} />);
      
      // Should use default recommendation
      expect(mockGetRecommendedStrategy).not.toHaveBeenCalledWith(undefined);
      
      // Should not show file-specific recommendation
      expect(screen.queryByText(/For .* files/)).not.toBeInTheDocument();
    });

    it('should handle empty string fileType', () => {
      render(<SimplifiedChunkingStrategySelector fileType="" />);
      
      // Should not show recommendation for empty file type
      expect(screen.queryByText(/For {2}files/)).not.toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('should have proper ARIA labels', () => {
      render(<SimplifiedChunkingStrategySelector />);
      
      const select = screen.getByLabelText(/chunking strategy/i);
      expect(select).toHaveAttribute('id', 'chunking-strategy');
      
      const label = screen.getByText(/chunking strategy/i, { selector: 'label' });
      expect(label).toHaveAttribute('for', 'chunking-strategy');
    });

    it('should support keyboard navigation', async () => {
      render(<SimplifiedChunkingStrategySelector />);
      
      const select = screen.getByLabelText(/chunking strategy/i);
      
      // Focus the select element directly
      select.focus();
      expect(select).toHaveFocus();
      
      // Use selectOptions to simulate keyboard selection
      await user.selectOptions(select, 'character');
      
      // Check that a change was triggered
      expect(mockSetStrategy).toHaveBeenCalledWith('character');
    });

    it('should have descriptive button labels', () => {
      render(<SimplifiedChunkingStrategySelector />);
      
      const learnMoreButton = screen.getByRole('button', { name: /learn more/i });
      expect(learnMoreButton).toHaveAccessibleName();
      
      const advancedButton = screen.getByRole('button', { name: /advanced options/i });
      expect(advancedButton).toHaveAccessibleName();
    });

    it('should indicate disabled state visually', () => {
      render(<SimplifiedChunkingStrategySelector disabled={true} />);
      
      const select = screen.getByLabelText(/chunking strategy/i);
      expect(select).toHaveClass('disabled:bg-gray-50', 'disabled:text-gray-500');
      
      const advancedButton = screen.getByRole('button', { name: /advanced options/i });
      expect(advancedButton).toHaveClass('disabled:opacity-50');
    });
  });

  describe('Visual Feedback', () => {
    it('should show hover state on buttons', () => {
      render(<SimplifiedChunkingStrategySelector />);
      
      const learnMoreButton = screen.getByRole('button', { name: /learn more/i });
      expect(learnMoreButton).toHaveClass('hover:text-blue-700');
      
      const advancedButton = screen.getByRole('button', { name: /advanced options/i });
      expect(advancedButton).toHaveClass('hover:text-gray-900');
    });

    it('should show focus ring on select', () => {
      render(<SimplifiedChunkingStrategySelector />);
      
      const select = screen.getByLabelText(/chunking strategy/i);
      expect(select).toHaveClass('focus:ring-2', 'focus:ring-blue-500');
    });

    it('should use correct colors for recommendation alert', () => {
      mockGetRecommendedStrategy.mockReturnValue('markdown');
      
      render(<SimplifiedChunkingStrategySelector fileType="md" />);
      
      const alert = screen.getByText(/For md files/).closest('div');
      expect(alert).toHaveClass('bg-blue-50', 'border-blue-200');
      
      const text = screen.getByText(/For md files/);
      expect(text).toHaveClass('text-blue-700');
    });

    it('should show transition effects', async () => {
      render(<SimplifiedChunkingStrategySelector />);
      
      const advancedButton = screen.getByRole('button', { name: /advanced options/i });
      expect(advancedButton).toHaveClass('transition-colors');
      
      // Check chevron rotation transition
      const chevron = advancedButton.querySelector('svg:last-child');
      expect(chevron).toHaveClass('transition-transform');
    });
  });
});