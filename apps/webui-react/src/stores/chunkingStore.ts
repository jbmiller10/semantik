import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import type { 
  ChunkingStrategyType, 
  ChunkingConfiguration, 
  ChunkingPreset,
  ChunkPreview,
  ChunkingComparisonResult,
  ChunkingAnalytics,
  ChunkingStatistics
} from '../types/chunking';
import { CHUNKING_STRATEGIES, CHUNKING_PRESETS } from '../types/chunking';
import { chunkingApi, handleChunkingError } from '../services/api/v2/chunking';

interface ChunkingStore {
  // Strategy Selection
  selectedStrategy: ChunkingStrategyType;
  strategyConfig: ChunkingConfiguration;
  
  // Preview State
  previewDocument: { id?: string; content?: string; name?: string } | null;
  previewChunks: ChunkPreview[];
  previewStatistics: ChunkingStatistics | null;
  previewLoading: boolean;
  previewError: string | null;
  
  // Comparison State
  comparisonStrategies: ChunkingStrategyType[];
  comparisonResults: Partial<Record<ChunkingStrategyType, ChunkingComparisonResult>>;
  comparisonLoading: boolean;
  comparisonError: string | null;
  
  // Analytics State
  analyticsData: ChunkingAnalytics | null;
  analyticsLoading: boolean;
  
  // Preset Management
  selectedPreset: string | null;
  customPresets: ChunkingPreset[];
  presetsLoading: boolean;
  
  // Actions - Strategy Management
  setStrategy: (strategy: ChunkingStrategyType) => void;
  updateConfiguration: (updates: Partial<ChunkingConfiguration['parameters']>) => void;
  applyPreset: (presetId: string) => void;
  loadPresets: () => Promise<void>;
  saveCustomPreset: (preset: Omit<ChunkingPreset, 'id'>) => Promise<string>;
  deleteCustomPreset: (presetId: string) => Promise<void>;
  
  // Actions - Preview
  setPreviewDocument: (doc: { id?: string; content?: string; name?: string } | null) => void;
  loadPreview: (forceRefresh?: boolean) => Promise<void>;
  clearPreview: () => void;
  
  // Actions - Comparison
  addComparisonStrategy: (strategy: ChunkingStrategyType) => void;
  removeComparisonStrategy: (strategy: ChunkingStrategyType) => void;
  compareStrategies: () => Promise<void>;
  clearComparison: () => void;
  
  // Actions - Analytics
  loadAnalytics: () => Promise<void>;
  
  // Actions - Utility
  reset: () => void;
  resetToDefaults: () => void;
  getRecommendedStrategy: (fileType?: string) => ChunkingStrategyType;
  cancelActiveRequests: () => void;
}

// Default configuration for each strategy
const getDefaultConfiguration = (strategy: ChunkingStrategyType): ChunkingConfiguration => {
  const strategyDef = CHUNKING_STRATEGIES[strategy];
  const parameters: Record<string, number | boolean | string> = {};
  
  strategyDef.parameters.forEach(param => {
    parameters[param.key] = param.defaultValue;
  });
  
  return { strategy, parameters };
};

// Initial state
const initialState = {
  selectedStrategy: 'recursive' as ChunkingStrategyType,
  strategyConfig: getDefaultConfiguration('recursive'),
  previewDocument: null,
  previewChunks: [],
  previewStatistics: null,
  previewLoading: false,
  previewError: null,
  comparisonStrategies: [],
  comparisonResults: {},
  comparisonLoading: false,
  comparisonError: null,
  analyticsData: null,
  analyticsLoading: false,
  selectedPreset: null,
  customPresets: [],
  presetsLoading: false
};

export const useChunkingStore = create<ChunkingStore>()(
  devtools(
    (set, get) => ({
      ...initialState,

      // Strategy Management Actions
      setStrategy: (strategy) => {
        const config = getDefaultConfiguration(strategy);
        set({
          selectedStrategy: strategy,
          strategyConfig: config,
          selectedPreset: null,
          previewError: null
        });
      },

      updateConfiguration: (updates) => {
        const { strategyConfig } = get();
        // Filter out undefined values
        const filteredUpdates = Object.entries(updates).reduce((acc, [key, value]) => {
          if (value !== undefined) {
            acc[key] = value;
          }
          return acc;
        }, {} as Record<string, number | boolean | string>);
        
        set({
          strategyConfig: {
            ...strategyConfig,
            parameters: { ...strategyConfig.parameters, ...filteredUpdates }
          },
          selectedPreset: null // Clear preset when manually updating
        });
      },

      applyPreset: (presetId) => {
        const preset = [...CHUNKING_PRESETS, ...get().customPresets]
          .find(p => p.id === presetId);
        
        if (preset) {
          set({
            selectedStrategy: preset.strategy,
            strategyConfig: preset.configuration,
            selectedPreset: presetId,
            previewError: null
          });
        }
      },

      loadPresets: async () => {
        set({ presetsLoading: true });
        
        try {
          const presets = await chunkingApi.getPresets({
            requestId: `load-presets-${Date.now()}`
          });
          
          // Filter out system presets (they're already in CHUNKING_PRESETS)
          const customPresets = presets.filter(p => p.id.startsWith('custom-'));
          
          set({
            customPresets,
            presetsLoading: false
          });
        } catch (error) {
          console.error('Failed to load presets:', handleChunkingError(error));
          set({ presetsLoading: false });
        }
      },

      saveCustomPreset: async (preset) => {
        try {
          const savedPreset = await chunkingApi.savePreset(preset, {
            requestId: `save-preset-${Date.now()}`
          });
          
          set(state => ({
            customPresets: [...state.customPresets, savedPreset]
          }));
          
          return savedPreset.id;
        } catch (error) {
          console.error('Failed to save preset:', handleChunkingError(error));
          throw error;
        }
      },

      deleteCustomPreset: async (presetId) => {
        try {
          await chunkingApi.deletePreset(presetId, {
            requestId: `delete-preset-${Date.now()}`
          });
          
          set(state => ({
            customPresets: state.customPresets.filter(p => p.id !== presetId),
            selectedPreset: state.selectedPreset === presetId ? null : state.selectedPreset
          }));
        } catch (error) {
          console.error('Failed to delete preset:', handleChunkingError(error));
          throw error;
        }
      },

      // Preview Actions
      setPreviewDocument: (doc) => {
        set({ 
          previewDocument: doc,
          previewChunks: [],
          previewStatistics: null,
          previewError: null
        });
      },

      loadPreview: async (forceRefresh = false) => {
        const { previewDocument, strategyConfig, previewChunks } = get();
        
        if (!previewDocument || (!forceRefresh && previewChunks.length > 0)) {
          return;
        }

        set({ previewLoading: true, previewError: null });

        try {
          const response = await chunkingApi.preview(
            {
              documentId: previewDocument.id,
              content: previewDocument.content,
              strategy: strategyConfig.strategy,
              configuration: strategyConfig.parameters,
            },
            {
              requestId: `preview-${previewDocument.id || 'content'}-${Date.now()}`,
              onProgress: (progress) => {
                // Progress tracking for UI feedback
                console.debug(`Preview progress: ${progress.percentage}%`);
              },
            }
          );

          set({
            previewChunks: response.chunks,
            previewStatistics: response.statistics,
            previewLoading: false
          });
        } catch (error) {
          set({
            previewError: handleChunkingError(error),
            previewLoading: false
          });
        }
      },

      clearPreview: () => {
        set({
          previewDocument: null,
          previewChunks: [],
          previewStatistics: null,
          previewError: null
        });
      },

      // Comparison Actions
      addComparisonStrategy: (strategy) => {
        set(state => ({
          comparisonStrategies: state.comparisonStrategies.includes(strategy)
            ? state.comparisonStrategies
            : [...state.comparisonStrategies, strategy].slice(0, 3) // Max 3 strategies
        }));
      },

      removeComparisonStrategy: (strategy) => {
        set(state => {
          const newResults = { ...state.comparisonResults };
          delete newResults[strategy];
          return {
            comparisonStrategies: state.comparisonStrategies.filter(s => s !== strategy),
            comparisonResults: newResults
          };
        });
      },

      compareStrategies: async () => {
        const { previewDocument, comparisonStrategies } = get();
        
        if (!previewDocument) {
          set({ comparisonError: 'No document selected for comparison' });
          return;
        }
        
        if (comparisonStrategies.length === 0) {
          set({ comparisonError: 'No strategies selected for comparison' });
          return;
        }

        set({ comparisonLoading: true, comparisonError: null });

        try {
          const strategies = comparisonStrategies.map(strategy => ({
            strategy,
            configuration: getDefaultConfiguration(strategy).parameters
          }));

          const results = await chunkingApi.compare(
            {
              documentId: previewDocument.id,
              content: previewDocument.content,
              strategies,
            },
            {
              requestId: `compare-${Date.now()}`,
              onProgress: (progress) => {
                console.debug(`Comparison progress: ${progress.percentage}%`);
              },
            }
          );

          // Convert array to map for store
          const comparisonResults: Partial<Record<ChunkingStrategyType, ChunkingComparisonResult>> = {};
          results.forEach(result => {
            comparisonResults[result.strategy] = result;
          });

          set({
            comparisonResults,
            comparisonLoading: false
          });
        } catch (error) {
          set({
            comparisonError: handleChunkingError(error),
            comparisonLoading: false
          });
        }
      },

      clearComparison: () => {
        set({
          comparisonStrategies: [],
          comparisonResults: {},
          comparisonError: null
        });
      },

      // Analytics Actions
      loadAnalytics: async () => {
        set({ analyticsLoading: true });

        try {
          const analytics = await chunkingApi.getAnalytics(
            undefined,
            {
              requestId: `analytics-${Date.now()}`
            }
          );

          set({
            analyticsData: analytics,
            analyticsLoading: false
          });
        } catch (error) {
          console.error('Failed to load analytics:', handleChunkingError(error));
          set({ analyticsLoading: false });
        }
      },

      // Utility Actions
      reset: () => {
        set(initialState);
      },

      resetToDefaults: () => {
        set(initialState);
      },

      getRecommendedStrategy: (fileType) => {
        if (!fileType) return 'recursive';
        
        // Remove leading dot if present
        const cleanFileType = fileType.startsWith('.') ? fileType.slice(1) : fileType;
        
        // Special cases for markdown
        if (['md', 'mdx', 'markdown'].includes(cleanFileType.toLowerCase())) {
          return 'markdown';
        }
        
        // Check presets for file type match
        const matchingPreset = CHUNKING_PRESETS.find(preset =>
          preset.fileTypes?.includes(cleanFileType.toLowerCase())
        );
        
        if (matchingPreset) {
          return matchingPreset.strategy;
        }
        
        // Default to recursive for most files (including code)
        return 'recursive';
      },

      cancelActiveRequests: () => {
        chunkingApi.cancelAllRequests('User cancelled operation');
      }
    }),
    {
      name: 'chunking-store'
    }
  )
);