import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import type { 
  ChunkingStrategyType, 
  ChunkingConfiguration, 
  ChunkingPreset,
  ChunkPreview,
  ChunkingPreviewResponse,
  ChunkingComparisonResult,
  ChunkingAnalytics,
  ChunkingStatistics
} from '../types/chunking';
import { CHUNKING_STRATEGIES, CHUNKING_PRESETS } from '../types/chunking';

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
  comparisonResults: Map<ChunkingStrategyType, ChunkingComparisonResult>;
  comparisonLoading: boolean;
  comparisonError: string | null;
  
  // Analytics State
  analyticsData: ChunkingAnalytics | null;
  analyticsLoading: boolean;
  
  // Preset Management
  selectedPreset: string | null;
  customPresets: ChunkingPreset[];
  
  // Actions - Strategy Management
  setStrategy: (strategy: ChunkingStrategyType) => void;
  updateConfiguration: (updates: Partial<ChunkingConfiguration['parameters']>) => void;
  applyPreset: (presetId: string) => void;
  saveCustomPreset: (preset: Omit<ChunkingPreset, 'id'>) => void;
  deleteCustomPreset: (presetId: string) => void;
  
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
  getRecommendedStrategy: (fileType?: string) => ChunkingStrategyType;
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
  comparisonResults: new Map(),
  comparisonLoading: false,
  comparisonError: null,
  analyticsData: null,
  analyticsLoading: false,
  selectedPreset: 'default-documents',
  customPresets: []
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

      saveCustomPreset: (preset) => {
        const id = `custom-${Date.now()}`;
        const newPreset = { ...preset, id };
        set(state => ({
          customPresets: [...state.customPresets, newPreset]
        }));
        return id;
      },

      deleteCustomPreset: (presetId) => {
        set(state => ({
          customPresets: state.customPresets.filter(p => p.id !== presetId),
          selectedPreset: state.selectedPreset === presetId ? null : state.selectedPreset
        }));
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
          // TODO: Replace with actual API call
          // const response = await chunkingApi.preview({
          //   documentId: previewDocument.id,
          //   content: previewDocument.content,
          //   strategy: strategyConfig.strategy,
          //   configuration: strategyConfig.parameters
          // });
          
          // Using strategyConfig for future API implementation
          console.debug('Preview requested for strategy:', strategyConfig.strategy);

          // Simulate API response for now
          await new Promise(resolve => setTimeout(resolve, 1000));
          
          const mockResponse: ChunkingPreviewResponse = {
            chunks: [
              {
                id: '1',
                content: 'This is a preview chunk showing how your document will be split...',
                startIndex: 0,
                endIndex: 100,
                tokens: 20
              },
              {
                id: '2',
                content: 'The second chunk continues from here with some overlap...',
                startIndex: 80,
                endIndex: 180,
                tokens: 18,
                overlapWithPrevious: 20
              }
            ],
            statistics: {
              totalChunks: 2,
              avgChunkSize: 90,
              minChunkSize: 80,
              maxChunkSize: 100,
              totalTokens: 38,
              avgTokensPerChunk: 19,
              overlapPercentage: 22,
              sizeDistribution: [
                { range: '0-100', count: 2, percentage: 100 }
              ]
            },
            performance: {
              processingTimeMs: 150,
              estimatedFullProcessingTimeMs: 1500
            }
          };

          set({
            previewChunks: mockResponse.chunks,
            previewStatistics: mockResponse.statistics,
            previewLoading: false
          });
        } catch (error) {
          set({
            previewError: error instanceof Error ? error.message : 'Failed to load preview',
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
        set(state => ({
          comparisonStrategies: state.comparisonStrategies.filter(s => s !== strategy),
          comparisonResults: new Map([...state.comparisonResults].filter(([key]) => key !== strategy))
        }));
      },

      compareStrategies: async () => {
        const { previewDocument, comparisonStrategies } = get();
        
        if (!previewDocument || comparisonStrategies.length === 0) {
          return;
        }

        set({ comparisonLoading: true, comparisonError: null });

        try {
          // TODO: Replace with actual API call
          // const results = await Promise.all(
          //   comparisonStrategies.map(strategy =>
          //     chunkingApi.preview({
          //       documentId: previewDocument.id,
          //       content: previewDocument.content,
          //       strategy,
          //       configuration: getDefaultConfiguration(strategy).parameters
          //     })
          //   )
          // );

          // Simulate API response
          await new Promise(resolve => setTimeout(resolve, 1500));
          
          const mockResults = new Map<ChunkingStrategyType, ChunkingComparisonResult>();
          comparisonStrategies.forEach(strategy => {
            mockResults.set(strategy, {
              strategy,
              configuration: getDefaultConfiguration(strategy),
              preview: {
                chunks: [],
                statistics: {
                  totalChunks: Math.floor(Math.random() * 20) + 5,
                  avgChunkSize: Math.floor(Math.random() * 500) + 300,
                  minChunkSize: 200,
                  maxChunkSize: 800,
                  sizeDistribution: []
                },
                performance: {
                  processingTimeMs: Math.floor(Math.random() * 500) + 100,
                  estimatedFullProcessingTimeMs: Math.floor(Math.random() * 5000) + 1000
                }
              },
              score: {
                quality: Math.random() * 30 + 70,
                performance: Math.random() * 30 + 70,
                overall: Math.random() * 30 + 70
              }
            });
          });

          set({
            comparisonResults: mockResults,
            comparisonLoading: false
          });
        } catch (error) {
          set({
            comparisonError: error instanceof Error ? error.message : 'Failed to compare strategies',
            comparisonLoading: false
          });
        }
      },

      clearComparison: () => {
        set({
          comparisonStrategies: [],
          comparisonResults: new Map(),
          comparisonError: null
        });
      },

      // Analytics Actions
      loadAnalytics: async () => {
        set({ analyticsLoading: true });

        try {
          // TODO: Replace with actual API call
          // const analytics = await chunkingApi.getAnalytics();

          // Simulate API response
          await new Promise(resolve => setTimeout(resolve, 800));
          
          const mockAnalytics: ChunkingAnalytics = {
            strategyUsage: [
              { strategy: 'recursive', count: 145, percentage: 35, trend: 'up' },
              { strategy: 'character', count: 95, percentage: 23, trend: 'stable' },
              { strategy: 'semantic', count: 75, percentage: 18, trend: 'up' },
              { strategy: 'markdown', count: 60, percentage: 14, trend: 'down' },
              { strategy: 'hierarchical', count: 25, percentage: 6, trend: 'stable' },
              { strategy: 'hybrid', count: 15, percentage: 4, trend: 'up' }
            ],
            performanceMetrics: Object.keys(CHUNKING_STRATEGIES).map(strategy => ({
              strategy: strategy as ChunkingStrategyType,
              avgProcessingTimeMs: Math.floor(Math.random() * 2000) + 500,
              avgChunksPerDocument: Math.floor(Math.random() * 50) + 10,
              successRate: Math.random() * 5 + 95
            })),
            fileTypeDistribution: [
              { fileType: 'pdf', count: 120, preferredStrategy: 'recursive' },
              { fileType: 'md', count: 80, preferredStrategy: 'markdown' },
              { fileType: 'txt', count: 60, preferredStrategy: 'character' },
              { fileType: 'py', count: 40, preferredStrategy: 'recursive' }
            ],
            recommendations: [
              {
                id: '1',
                type: 'strategy',
                priority: 'high',
                title: 'Try Semantic Chunking for Better Results',
                description: 'Based on your document types, semantic chunking could improve search quality by 15%',
                action: {
                  label: 'Switch to Semantic',
                  configuration: getDefaultConfiguration('semantic')
                }
              }
            ]
          };

          set({
            analyticsData: mockAnalytics,
            analyticsLoading: false
          });
        } catch {
          set({ analyticsLoading: false });
        }
      },

      // Utility Actions
      reset: () => {
        set(initialState);
      },

      getRecommendedStrategy: (fileType) => {
        if (!fileType) return 'hybrid';
        
        // Check presets for file type match
        const matchingPreset = CHUNKING_PRESETS.find(preset =>
          preset.fileTypes?.includes(fileType.toLowerCase())
        );
        
        if (matchingPreset) {
          return matchingPreset.strategy;
        }
        
        // Special cases
        if (['md', 'mdx', 'markdown'].includes(fileType.toLowerCase())) {
          return 'markdown';
        }
        
        // Default to recursive for most files
        return 'recursive';
      }
    }),
    {
      name: 'chunking-store'
    }
  )
);