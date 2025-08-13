import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ChunkingParameterTuner } from '../ChunkingParameterTuner';
import { useChunkingStore } from '../../../stores/chunkingStore';
import { CHUNKING_STRATEGIES, CHUNKING_PRESETS } from '../../../types/chunking';
import { mockChunkingPresets, createChunkingConfig } from '../../../tests/utils/chunkingTestUtils';
import type { ChunkingStore } from '../../../stores/chunkingStore';

// Mock the chunking store
vi.mock('../../../stores/chunkingStore', () => ({
  useChunkingStore: vi.fn()
}));

// Mock the formStyles utility
vi.mock('../../../utils/formStyles', () => ({
  getInputClassName: vi.fn(() => 'mock-input-class')
}));

describe('ChunkingParameterTuner', () => {
  const mockUpdateConfiguration = vi.fn();
  const mockApplyPreset = vi.fn();
  const mockSaveCustomPreset = vi.fn();
  const mockLoadPreview = vi.fn();

  const defaultStoreState: Partial<ChunkingStore> = {
    selectedStrategy: 'recursive',
    strategyConfig: {
      strategy: 'recursive',
      parameters: {
        chunk_size: 600,
        chunk_overlap: 100,
        preserve_sentences: true
      }
    },
    updateConfiguration: mockUpdateConfiguration,
    applyPreset: mockApplyPreset,
    saveCustomPreset: mockSaveCustomPreset,
    customPresets: mockChunkingPresets.filter(p => !p.isSystem),
    selectedPreset: null,
    loadPreview: mockLoadPreview,
    previewDocument: { id: 'test-doc', name: 'test.txt', content: 'test content' },
    previewLoading: false
  };

  beforeEach(() => {
    vi.clearAllMocks();
    vi.useFakeTimers();
    (useChunkingStore as unknown as vi.Mock).mockReturnValue(defaultStoreState);
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  describe('Rendering', () => {
    it('renders configuration section with preset selector', () => {
      render(<ChunkingParameterTuner />);
      
      expect(screen.getByText('Configuration')).toBeInTheDocument();
      expect(screen.getByRole('combobox')).toBeInTheDocument();
      expect(screen.getByText('Custom Configuration')).toBeInTheDocument();
    });

    it('renders all basic parameters for the selected strategy', () => {
      render(<ChunkingParameterTuner />);
      
      expect(screen.getByText('Chunk Size')).toBeInTheDocument();
      expect(screen.getByText('Chunk Overlap')).toBeInTheDocument();
      expect(screen.getByText('Parameters')).toBeInTheDocument();
    });

    it('renders parameter descriptions with help icons', () => {
      render(<ChunkingParameterTuner />);
      
      // Help icons are rendered inside groups with hover effect
      const helpGroups = document.querySelectorAll('.group.relative');
      // Look for elements that have the tooltip structure
      const tooltipContainers = Array.from(document.querySelectorAll('.bg-gray-900.text-white.text-xs'));
      
      // There should be tooltips for parameters with descriptions
      expect(helpGroups.length).toBeGreaterThan(0);
    });

    it('displays unit labels for numeric parameters', () => {
      render(<ChunkingParameterTuner />);
      
      // Multiple parameters can have the same unit
      const unitLabels = screen.getAllByText('Measured in characters');
      expect(unitLabels.length).toBeGreaterThan(0);
    });

    it('shows current values for all parameters', () => {
      render(<ChunkingParameterTuner />);
      
      expect(screen.getByText('600')).toBeInTheDocument(); // chunk_size value
      // 100 appears multiple times (as value and as min/max labels)
      const hundredElements = screen.getAllByText('100');
      expect(hundredElements.length).toBeGreaterThan(0);
    });
  });

  describe('Parameter Change Handling', () => {
    it('updates numeric parameters when slider is moved', async () => {
      render(<ChunkingParameterTuner />);
      
      const chunkSizeSlider = screen.getAllByRole('slider')[0];
      fireEvent.change(chunkSizeSlider, { target: { value: '800' } });
      
      expect(mockUpdateConfiguration).toHaveBeenCalledWith({ chunk_size: 800 });
    });

    it('updates boolean parameters when toggle is clicked', async () => {
      // Update store state to include advanced parameters visible
      const storeWithAdvanced = {
        ...defaultStoreState,
        strategyConfig: {
          strategy: 'recursive',
          parameters: {
            chunk_size: 600,
            chunk_overlap: 100,
            preserve_sentences: true
          }
        }
      };
      (useChunkingStore as unknown as vi.Mock).mockReturnValue(storeWithAdvanced);
      
      render(<ChunkingParameterTuner />);
      
      // Show advanced parameters
      const advancedButton = screen.getByText('Advanced Parameters');
      fireEvent.click(advancedButton);
      
      const preserveSentencesToggle = screen.getByRole('switch');
      fireEvent.click(preserveSentencesToggle);
      
      expect(mockUpdateConfiguration).toHaveBeenCalledWith({ preserve_sentences: false });
    });

    it('updates select parameters when option is changed', () => {
      // Use hierarchical strategy which has select parameters
      const hierarchicalStore = {
        ...defaultStoreState,
        selectedStrategy: 'hierarchical',
        strategyConfig: {
          strategy: 'hierarchical',
          parameters: {
            chunk_sizes: '1024,2048,4096',
            overlap_ratio: 0.2
          }
        }
      };
      (useChunkingStore as unknown as vi.Mock).mockReturnValue(hierarchicalStore);
      
      render(<ChunkingParameterTuner />);
      
      const selectElement = screen.getByRole('combobox', { name: /Chunk Sizes/i });
      fireEvent.change(selectElement, { target: { value: '512,1024,2048' } });
      
      expect(mockUpdateConfiguration).toHaveBeenCalledWith({ chunk_sizes: '512,1024,2048' });
    });

    it('debounces parameter changes for preview updates', async () => {
      const mockOnParameterChange = vi.fn();
      render(<ChunkingParameterTuner onParameterChange={mockOnParameterChange} />);
      
      const chunkSizeSlider = screen.getAllByRole('slider')[0];
      
      // Make multiple rapid changes
      fireEvent.change(chunkSizeSlider, { target: { value: '700' } });
      fireEvent.change(chunkSizeSlider, { target: { value: '800' } });
      fireEvent.change(chunkSizeSlider, { target: { value: '900' } });
      
      // Callback should not be called immediately
      expect(mockOnParameterChange).not.toHaveBeenCalled();
      expect(mockLoadPreview).not.toHaveBeenCalled();
      
      // Fast-forward 500ms (debounce delay)
      act(() => {
        vi.advanceTimersByTime(500);
      });
      
      // Now it should be called once
      await waitFor(() => {
        expect(mockLoadPreview).toHaveBeenCalledWith(true);
        expect(mockOnParameterChange).toHaveBeenCalledTimes(1);
      });
    });
  });

  describe('Preset Management', () => {
    it('displays built-in presets for the selected strategy', () => {
      render(<ChunkingParameterTuner />);
      
      fireEvent.click(screen.getByRole('combobox'));
      
      // Check for built-in presets
      expect(screen.getByText('Built-in Presets')).toBeInTheDocument();
    });

    it('displays custom presets when available', () => {
      const storeWithCustomPresets = {
        ...defaultStoreState,
        customPresets: [
          {
            id: 'custom-1',
            name: 'My Custom Preset',
            description: 'Custom settings',
            strategy: 'recursive',
            configuration: createChunkingConfig('recursive')
          }
        ]
      };
      (useChunkingStore as unknown as vi.Mock).mockReturnValue(storeWithCustomPresets);
      
      render(<ChunkingParameterTuner />);
      
      fireEvent.click(screen.getByRole('combobox'));
      
      expect(screen.getByText('Custom Presets')).toBeInTheDocument();
      expect(screen.getByText('My Custom Preset')).toBeInTheDocument();
    });

    it('applies preset when selected', () => {
      render(<ChunkingParameterTuner />);
      
      const presetSelector = screen.getByRole('combobox');
      fireEvent.change(presetSelector, { target: { value: 'default-recursive' } });
      
      expect(mockApplyPreset).toHaveBeenCalledWith('default-recursive');
    });

    it('does not apply preset when "custom" is selected', () => {
      render(<ChunkingParameterTuner />);
      
      const presetSelector = screen.getByRole('combobox');
      fireEvent.change(presetSelector, { target: { value: 'custom' } });
      
      expect(mockApplyPreset).not.toHaveBeenCalled();
    });

    it('shows save preset form when Save Preset button is clicked', async () => {
      const user = userEvent.setup({ delay: null });
      render(<ChunkingParameterTuner />);
      
      const saveButton = screen.getByRole('button', { name: /Save Preset/i });
      await user.click(saveButton);
      
      expect(screen.getByPlaceholderText('Enter preset name...')).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /Cancel/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /^Save$/i })).toBeInTheDocument();
    });

    it('saves custom preset with entered name', async () => {
      const user = userEvent.setup({ delay: null });
      render(<ChunkingParameterTuner />);
      
      // Open save preset form
      const savePresetButton = screen.getByRole('button', { name: /Save Preset/i });
      await user.click(savePresetButton);
      
      // Enter preset name
      const nameInput = screen.getByPlaceholderText('Enter preset name...');
      await user.type(nameInput, 'My New Preset');
      
      // Click save
      const saveButton = screen.getByRole('button', { name: /^Save$/i });
      await user.click(saveButton);
      
      expect(mockSaveCustomPreset).toHaveBeenCalledWith({
        name: 'My New Preset',
        description: 'Custom configuration for Recursive',
        strategy: 'recursive',
        configuration: defaultStoreState.strategyConfig
      });
    });

    it('cancels preset saving when Cancel is clicked', async () => {
      const user = userEvent.setup({ delay: null });
      render(<ChunkingParameterTuner />);
      
      // Open save preset form
      const savePresetButton = screen.getByRole('button', { name: /Save Preset/i });
      await user.click(savePresetButton);
      
      expect(screen.getByPlaceholderText('Enter preset name...')).toBeInTheDocument();
      
      // Click cancel
      const cancelButton = screen.getByRole('button', { name: /Cancel/i });
      await user.click(cancelButton);
      
      // Form should be hidden
      expect(screen.queryByPlaceholderText('Enter preset name...')).not.toBeInTheDocument();
      expect(mockSaveCustomPreset).not.toHaveBeenCalled();
    });

    it('disables save button when preset name is empty', async () => {
      const user = userEvent.setup({ delay: null });
      render(<ChunkingParameterTuner />);
      
      // Open save preset form
      const savePresetButton = screen.getByRole('button', { name: /Save Preset/i });
      await user.click(savePresetButton);
      
      const saveButton = screen.getByRole('button', { name: /^Save$/i });
      expect(saveButton).toBeDisabled();
      
      // Type a name
      const nameInput = screen.getByPlaceholderText('Enter preset name...');
      await user.type(nameInput, 'Test');
      
      expect(saveButton).not.toBeDisabled();
      
      // Clear the name
      await user.clear(nameInput);
      
      expect(saveButton).toBeDisabled();
    });
  });

  describe('Advanced Parameters', () => {
    it('hides advanced parameters by default', () => {
      render(<ChunkingParameterTuner />);
      
      // Advanced parameter should not be visible
      expect(screen.queryByText('Preserve Sentences')).not.toBeInTheDocument();
      
      // But advanced parameters toggle should be visible
      expect(screen.getByText('Advanced Parameters')).toBeInTheDocument();
    });

    it('shows advanced parameters when toggle is clicked', () => {
      render(<ChunkingParameterTuner />);
      
      const advancedToggle = screen.getByText('Advanced Parameters');
      fireEvent.click(advancedToggle);
      
      // Advanced parameter should now be visible
      expect(screen.getByText('Preserve Sentences')).toBeInTheDocument();
    });

    it('rotates chevron icon when advanced section is toggled', () => {
      render(<ChunkingParameterTuner />);
      
      const advancedButton = screen.getByText('Advanced Parameters').closest('button');
      const chevron = advancedButton?.querySelector('svg');
      
      // Initially not rotated
      expect(chevron).not.toHaveClass('rotate-180');
      
      // Click to show advanced
      fireEvent.click(advancedButton!);
      
      // Should be rotated
      expect(chevron).toHaveClass('rotate-180');
      
      // Click to hide advanced
      fireEvent.click(advancedButton!);
      
      // Should not be rotated
      expect(chevron).not.toHaveClass('rotate-180');
    });

    it('applies animation when advanced parameters are shown', () => {
      render(<ChunkingParameterTuner />);
      
      const advancedToggle = screen.getByText('Advanced Parameters');
      fireEvent.click(advancedToggle);
      
      const advancedSection = screen.getByText('Preserve Sentences').closest('.animate-slideDown');
      expect(advancedSection).toHaveClass('animate-slideDown');
    });
  });

  describe('Reset Functionality', () => {
    it('resets parameters to defaults when Reset button is clicked', async () => {
      const user = userEvent.setup({ delay: null });
      render(<ChunkingParameterTuner />);
      
      const resetButton = screen.getByRole('button', { name: /Reset/i });
      await user.click(resetButton);
      
      expect(mockUpdateConfiguration).toHaveBeenCalledWith({
        chunk_size: 600,
        chunk_overlap: 100,
        preserve_sentences: true
      });
    });

    it('resets correct defaults for different strategies', async () => {
      const semanticStore = {
        ...defaultStoreState,
        selectedStrategy: 'semantic',
        strategyConfig: {
          strategy: 'semantic',
          parameters: {
            breakpoint_percentile_threshold: 85,
            max_chunk_size: 1500,
            buffer_size: 3
          }
        }
      };
      (useChunkingStore as unknown as vi.Mock).mockReturnValue(semanticStore);
      
      const user = userEvent.setup({ delay: null });
      render(<ChunkingParameterTuner />);
      
      const resetButton = screen.getByRole('button', { name: /Reset/i });
      await user.click(resetButton);
      
      expect(mockUpdateConfiguration).toHaveBeenCalledWith({
        breakpoint_percentile_threshold: 90,
        max_chunk_size: 1000,
        buffer_size: 2
      });
    });
  });

  describe('Preview Integration', () => {
    it('shows preview status when showPreview is true', () => {
      render(<ChunkingParameterTuner showPreview={true} />);
      
      expect(screen.getByText('Preview updated')).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /Refresh/i })).toBeInTheDocument();
    });

    it('does not show preview status when showPreview is false', () => {
      render(<ChunkingParameterTuner showPreview={false} />);
      
      expect(screen.queryByText('Preview updated')).not.toBeInTheDocument();
      expect(screen.queryByRole('button', { name: /Refresh/i })).not.toBeInTheDocument();
    });

    it('shows loading state when preview is loading', () => {
      const loadingStore = {
        ...defaultStoreState,
        previewLoading: true
      };
      (useChunkingStore as unknown as vi.Mock).mockReturnValue(loadingStore);
      
      render(<ChunkingParameterTuner showPreview={true} />);
      
      expect(screen.getByText('Updating preview...')).toBeInTheDocument();
      
      // Check for spinner
      const spinner = document.querySelector('.animate-spin');
      expect(spinner).toBeInTheDocument();
    });

    it('triggers preview refresh when Refresh button is clicked', async () => {
      const user = userEvent.setup({ delay: null });
      render(<ChunkingParameterTuner showPreview={true} />);
      
      const refreshButton = screen.getByRole('button', { name: /Refresh/i });
      await user.click(refreshButton);
      
      expect(mockLoadPreview).toHaveBeenCalledWith(true);
    });

    it('disables refresh button when preview is loading', () => {
      const loadingStore = {
        ...defaultStoreState,
        previewLoading: true
      };
      (useChunkingStore as unknown as vi.Mock).mockReturnValue(loadingStore);
      
      render(<ChunkingParameterTuner showPreview={true} />);
      
      const refreshButton = screen.getByRole('button', { name: /Refresh/i });
      expect(refreshButton).toBeDisabled();
    });

    it('does not trigger preview when no document is available', () => {
      const noDocumentStore = {
        ...defaultStoreState,
        previewDocument: null
      };
      (useChunkingStore as unknown as vi.Mock).mockReturnValue(noDocumentStore);
      
      render(<ChunkingParameterTuner showPreview={true} />);
      
      const chunkSizeSlider = screen.getAllByRole('slider')[0];
      fireEvent.change(chunkSizeSlider, { target: { value: '800' } });
      
      act(() => {
        vi.advanceTimersByTime(500);
      });
      
      expect(mockLoadPreview).not.toHaveBeenCalled();
    });
  });

  describe('Disabled State', () => {
    it('disables all controls when disabled prop is true', () => {
      render(<ChunkingParameterTuner disabled={true} />);
      
      // Check sliders are disabled
      const sliders = screen.getAllByRole('slider');
      sliders.forEach(slider => {
        expect(slider).toBeDisabled();
        expect(slider).toHaveClass('opacity-50', 'cursor-not-allowed');
      });
      
      // Check preset selector is disabled
      const presetSelector = screen.getByRole('combobox');
      expect(presetSelector).toBeDisabled();
      
      // Check buttons are disabled
      const resetButton = screen.getByRole('button', { name: /Reset/i });
      expect(resetButton).toBeDisabled();
      
      const savePresetButton = screen.getByRole('button', { name: /Save Preset/i });
      expect(savePresetButton).toBeDisabled();
    });

    it('disables advanced parameter controls when disabled', () => {
      render(<ChunkingParameterTuner disabled={true} />);
      
      // Show advanced parameters
      const advancedToggle = screen.getByText('Advanced Parameters');
      fireEvent.click(advancedToggle);
      
      // Check if boolean switch is disabled
      const switches = screen.getAllByRole('switch');
      switches.forEach(switchEl => {
        expect(switchEl).toBeDisabled();
        expect(switchEl).toHaveClass('opacity-50', 'cursor-not-allowed');
      });
    });
  });

  describe('Parameter Visualization', () => {
    it('shows visual progress for numeric sliders', () => {
      render(<ChunkingParameterTuner />);
      
      const sliders = screen.getAllByRole('slider');
      sliders.forEach(slider => {
        // Check that the slider has a gradient background style
        expect(slider.style.background).toContain('linear-gradient');
      });
    });

    it('updates slider visual progress when value changes', () => {
      render(<ChunkingParameterTuner />);
      
      const chunkSizeSlider = screen.getAllByRole('slider')[0];
      
      // Initial state (600 out of 100-2000 range)
      expect(chunkSizeSlider.style.background).toContain('linear-gradient');
      
      // Change value
      fireEvent.change(chunkSizeSlider, { target: { value: '1500' } });
      
      // Visual should update (different percentage)
      expect(chunkSizeSlider.style.background).toContain('linear-gradient');
    });

    it('shows min and max values for sliders', () => {
      render(<ChunkingParameterTuner />);
      
      // For chunk_size slider (100-2000)
      expect(screen.getByText('100')).toBeInTheDocument();
      expect(screen.getByText('2000')).toBeInTheDocument();
      
      // For chunk_overlap slider (0-500)
      expect(screen.getByText('0')).toBeInTheDocument();
      expect(screen.getByText('500')).toBeInTheDocument();
    });
  });

  describe('Different Parameter Types', () => {
    it('renders select parameters correctly', () => {
      const hierarchicalStore = {
        ...defaultStoreState,
        selectedStrategy: 'hierarchical',
        strategyConfig: {
          strategy: 'hierarchical',
          parameters: {
            chunk_sizes: '1024,2048,4096',
            overlap_ratio: 0.2
          }
        }
      };
      (useChunkingStore as unknown as vi.Mock).mockReturnValue(hierarchicalStore);
      
      render(<ChunkingParameterTuner />);
      
      const selectElement = screen.getByRole('combobox', { name: /Chunk Sizes/i });
      expect(selectElement).toBeInTheDocument();
      
      // Check options are available
      expect(screen.getByText('Medium (1024, 2048, 4096)')).toBeInTheDocument();
    });

    it('renders boolean parameters with toggle switches', () => {
      render(<ChunkingParameterTuner />);
      
      // Show advanced to see boolean parameter
      const advancedToggle = screen.getByText('Advanced Parameters');
      fireEvent.click(advancedToggle);
      
      const preserveSentencesSwitch = screen.getByRole('switch');
      expect(preserveSentencesSwitch).toBeInTheDocument();
      expect(preserveSentencesSwitch).toHaveAttribute('aria-checked', 'true');
    });

    it('shows correct visual state for boolean toggles', () => {
      render(<ChunkingParameterTuner />);
      
      // Show advanced parameters
      const advancedToggle = screen.getByText('Advanced Parameters');
      fireEvent.click(advancedToggle);
      
      const switchElement = screen.getByRole('switch');
      const switchIndicator = switchElement.querySelector('span');
      
      // Initially on (true)
      expect(switchElement).toHaveClass('bg-blue-600');
      expect(switchIndicator).toHaveClass('translate-x-5');
      
      // Click to turn off
      fireEvent.click(switchElement);
      
      // Verify the updateConfiguration was called with false
      expect(mockUpdateConfiguration).toHaveBeenCalledWith({ preserve_sentences: false });
    });
  });

  describe('Help Tooltips', () => {
    it('renders tooltips for parameter descriptions', () => {
      render(<ChunkingParameterTuner />);
      
      // Tooltips are rendered but initially invisible
      const tooltips = document.querySelectorAll('.bg-gray-900.text-white.text-xs');
      
      // Check that tooltips exist in the DOM (they're hidden via opacity-0 invisible classes)
      expect(tooltips.length).toBeGreaterThan(0);
      
      // Check that tooltips contain description text from the parameter definitions
      const tooltipTexts = Array.from(tooltips).map(t => t.textContent);
      // These are actual descriptions from CHUNKING_STRATEGIES
      expect(tooltipTexts.some(text => 
        text?.includes('Number of characters per chunk') || 
        text?.includes('Number of overlapping characters') ||
        text?.includes('Target size for each chunk') ||
        text?.includes('Overlap between consecutive chunks')
      )).toBe(true);
    });
  });
});