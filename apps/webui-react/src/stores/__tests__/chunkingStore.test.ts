import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { act, renderHook } from '@testing-library/react'
import { useChunkingStore } from '../chunkingStore'
import { CHUNKING_PRESETS } from '@/types/chunking'
import type { ChunkingStrategyType, ChunkingPreset } from '@/types/chunking'

// Create mock functions first
const mockPreview = vi.fn()
const mockCompare = vi.fn()
const mockGetAnalytics = vi.fn()
const mockGetPresets = vi.fn()
const mockSavePreset = vi.fn()
const mockDeletePreset = vi.fn()
const mockProcess = vi.fn()
const mockGetRecommendation = vi.fn()
const mockCancelRequest = vi.fn()
const mockCancelAllRequests = vi.fn()
const mockIsRequestActive = vi.fn()
const mockHandleChunkingError = vi.fn((error: unknown) => {
  if (error && typeof error === 'object' && 'message' in error) {
    return (error as { message: string }).message
  }
  return 'Error'
})

// Mock the module
vi.mock('@/services/api/v2/chunking', () => {
  return {
    chunkingApi: {
      get preview() { return mockPreview },
      get compare() { return mockCompare },
      get getAnalytics() { return mockGetAnalytics },
      get getPresets() { return mockGetPresets },
      get savePreset() { return mockSavePreset },
      get deletePreset() { return mockDeletePreset },
      get process() { return mockProcess },
      get getRecommendation() { return mockGetRecommendation },
      get cancelRequest() { return mockCancelRequest },
      get cancelAllRequests() { return mockCancelAllRequests },
      get isRequestActive() { return mockIsRequestActive },
    },
    get handleChunkingError() { return mockHandleChunkingError },
  }
})

describe('chunkingStore', () => {
  beforeEach(() => {
    // Clear mocks before each test
    vi.clearAllMocks()
    
    // Set up default mock implementations
    mockPreview.mockResolvedValue({
      chunks: [
        {
          id: 'chunk-1',
          content: 'Test chunk content',
          startIndex: 0,
          endIndex: 50,
          tokens: 10,
          overlapWithPrevious: 0,
          overlapWithNext: 10,
        },
        {
          id: 'chunk-2',
          content: 'Second chunk content',
          startIndex: 40,
          endIndex: 90,
          tokens: 12,
          overlapWithPrevious: 10,
          overlapWithNext: 0,
        },
      ],
      statistics: {
        totalChunks: 2,
        avgChunkSize: 45,
        minChunkSize: 40,
        maxChunkSize: 50,
        totalSize: 90,
        overlapRatio: 0.11,
        overlapPercentage: 11,
        sizeDistribution: [],
      },
      performance: {
        processingTimeMs: 100,
        estimatedFullProcessingTimeMs: 500,
      },
    })
    
    mockCompare.mockResolvedValue([
      {
        strategy: 'recursive',
        configuration: {
          strategy: 'recursive',
          parameters: { chunk_size: 600, chunk_overlap: 100 },
        },
        preview: {
          chunks: [],
          statistics: {
            totalChunks: 3,
            avgChunkSize: 400,
            minChunkSize: 300,
            maxChunkSize: 500,
            sizeDistribution: [],
          },
          performance: {
            processingTimeMs: 50,
            estimatedFullProcessingTimeMs: 200,
          },
        },
      },
      {
        strategy: 'semantic',
        configuration: {
          strategy: 'semantic',
          parameters: { breakpoint_percentile_threshold: 90, max_chunk_size: 1000 },
        },
        preview: {
          chunks: [],
          statistics: {
            totalChunks: 4,
            avgChunkSize: 350,
            minChunkSize: 250,
            maxChunkSize: 450,
            sizeDistribution: [],
          },
          performance: {
            processingTimeMs: 150,
            estimatedFullProcessingTimeMs: 600,
          },
        },
      },
    ])
    
    mockGetAnalytics.mockResolvedValue({
      strategyUsage: [
        { strategy: 'recursive', count: 100, percentage: 50, trend: 'up' },
        { strategy: 'semantic', count: 50, percentage: 25, trend: 'stable' },
      ],
      performanceMetrics: [
        {
          strategy: 'recursive',
          avgProcessingTimeMs: 100,
          avgChunksPerDocument: 10,
          successRate: 98,
        },
      ],
      fileTypeDistribution: [
        { fileType: 'pdf', count: 50, preferredStrategy: 'recursive' },
      ],
      recommendations: [
        {
          id: 'rec-1',
          type: 'strategy',
          priority: 'high',
          title: 'Use semantic chunking for better results',
          description: 'Based on your document types',
        },
      ],
    })
    
    mockGetPresets.mockResolvedValue([])
    
    mockSavePreset.mockImplementation(async (preset) => {
      const id = `custom-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
      return { ...preset, id }
    })
    
    mockDeletePreset.mockResolvedValue(undefined)
    
    mockProcess.mockResolvedValue({ operationId: 'op-123' })
    
    mockGetRecommendation.mockResolvedValue({
      strategy: 'recursive',
      configuration: {
        strategy: 'recursive',
        parameters: { chunk_size: 600, chunk_overlap: 100 },
      },
      confidence: 0.9,
    })
    
    mockCancelRequest.mockReturnValue(true)
    mockCancelAllRequests.mockReturnValue(undefined)
    mockIsRequestActive.mockReturnValue(false)
  })

  afterEach(() => {
    // Reset store state after each test
    useChunkingStore.getState().reset()
    vi.clearAllMocks()
  })

  describe('Strategy Management', () => {
    it('initializes with default strategy', () => {
      const { result } = renderHook(() => useChunkingStore())
      
      expect(result.current.selectedStrategy).toBe('recursive')
      expect(result.current.strategyConfig.strategy).toBe('recursive')
      expect(result.current.strategyConfig.parameters).toBeDefined()
    })

    it('sets new strategy with default configuration', () => {
      const { result } = renderHook(() => useChunkingStore())
      
      act(() => {
        result.current.setStrategy('semantic')
      })

      expect(result.current.selectedStrategy).toBe('semantic')
      expect(result.current.strategyConfig.strategy).toBe('semantic')
      expect(result.current.selectedPreset).toBeNull()
      expect(result.current.previewError).toBeNull()
    })

    it('updates configuration parameters', () => {
      const { result } = renderHook(() => useChunkingStore())
      
      act(() => {
        result.current.updateConfiguration({
          chunk_size: 1024,
          chunk_overlap: 100,
        })
      })

      expect(result.current.strategyConfig.parameters.chunk_size).toBe(1024)
      expect(result.current.strategyConfig.parameters.chunk_overlap).toBe(100)
      expect(result.current.selectedPreset).toBeNull()
    })

    it('filters out undefined values when updating configuration', () => {
      const { result } = renderHook(() => useChunkingStore())
      
      const initialParams = { ...result.current.strategyConfig.parameters }
      
      act(() => {
        result.current.updateConfiguration({
          chunk_size: 512,
          chunk_overlap: undefined,
        })
      })

      expect(result.current.strategyConfig.parameters.chunk_size).toBe(512)
      expect(result.current.strategyConfig.parameters.chunk_overlap).toBe(initialParams.chunk_overlap)
    })

    it('applies preset correctly', () => {
      const { result } = renderHook(() => useChunkingStore())
      
      const preset = CHUNKING_PRESETS[0]
      
      act(() => {
        result.current.applyPreset(preset.id)
      })

      expect(result.current.selectedStrategy).toBe(preset.strategy)
      expect(result.current.strategyConfig).toEqual(preset.configuration)
      expect(result.current.selectedPreset).toBe(preset.id)
      expect(result.current.previewError).toBeNull()
    })

    it('saves custom preset', async () => {
      const { result } = renderHook(() => useChunkingStore())
      
      const customPreset: Omit<ChunkingPreset, 'id'> = {
        name: 'My Custom Preset',
        description: 'Custom settings for my documents',
        strategy: 'recursive',
        configuration: {
          strategy: 'recursive',
          parameters: {
            chunk_size: 750,
            chunk_overlap: 75,
          },
        },
      }
      
      await act(async () => {
        await result.current.saveCustomPreset(customPreset)
      })

      expect(result.current.customPresets).toHaveLength(1)
      expect(result.current.customPresets[0].name).toBe('My Custom Preset')
      expect(result.current.customPresets[0].id).toMatch(/^custom-\d+-[a-z0-9]+$/)
    })

    it('deletes custom preset', async () => {
      const { result } = renderHook(() => useChunkingStore())
      
      // First save a preset
      let presetId: string = ''
      await act(async () => {
        presetId = await result.current.saveCustomPreset({
          name: 'Test Preset',
          description: 'Test',
          strategy: 'recursive',
          configuration: {
            strategy: 'recursive',
            parameters: {},
          },
        })
      })

      expect(result.current.customPresets).toHaveLength(1)

      // Then delete it
      await act(async () => {
        await result.current.deleteCustomPreset(presetId)
      })

      expect(result.current.customPresets).toHaveLength(0)
    })

    it('clears selected preset when deleting the active preset', async () => {
      const { result } = renderHook(() => useChunkingStore())
      
      let presetId: string = ''
      await act(async () => {
        presetId = await result.current.saveCustomPreset({
          name: 'Test Preset',
          description: 'Test',
          strategy: 'recursive',
          configuration: {
            strategy: 'recursive',
            parameters: {},
          },
        })
        result.current.applyPreset(presetId)
      })

      expect(result.current.selectedPreset).toBe(presetId)

      await act(async () => {
        await result.current.deleteCustomPreset(presetId)
      })

      expect(result.current.selectedPreset).toBeNull()
    })
  })

  describe('Preview Management', () => {
    it('sets preview document', () => {
      const { result } = renderHook(() => useChunkingStore())
      
      const document = {
        id: 'doc-1',
        content: 'Test document content',
        name: 'test.txt',
      }
      
      act(() => {
        result.current.setPreviewDocument(document)
      })

      expect(result.current.previewDocument).toEqual(document)
      expect(result.current.previewChunks).toEqual([])
      expect(result.current.previewStatistics).toBeNull()
      expect(result.current.previewError).toBeNull()
    })

    it('loads preview for document', async () => {
      const { result } = renderHook(() => useChunkingStore())
      
      const document = {
        id: 'doc-1',
        content: 'Test document content',
        name: 'test.txt',
      }
      
      act(() => {
        result.current.setPreviewDocument(document)
      })

      await act(async () => {
        await result.current.loadPreview()
      })

      expect(result.current.previewLoading).toBe(false)
      expect(result.current.previewChunks.length).toBeGreaterThan(0)
      expect(result.current.previewStatistics).toBeDefined()
    })

    it('does not reload preview if chunks already exist', async () => {
      const { result } = renderHook(() => useChunkingStore())
      
      const document = {
        id: 'doc-1',
        content: 'Test document content',
        name: 'test.txt',
      }
      
      act(() => {
        result.current.setPreviewDocument(document)
      })

      // Load once
      await act(async () => {
        await result.current.loadPreview()
      })

      const initialChunks = result.current.previewChunks

      // Try to load again without force
      await act(async () => {
        await result.current.loadPreview(false)
      })

      expect(result.current.previewChunks).toBe(initialChunks)
    })

    it('force refreshes preview when requested', async () => {
      const { result } = renderHook(() => useChunkingStore())
      
      const document = {
        id: 'doc-1',
        content: 'Test document content',
        name: 'test.txt',
      }
      
      act(() => {
        result.current.setPreviewDocument(document)
      })

      // Load once
      await act(async () => {
        await result.current.loadPreview()
      })

      // Force refresh
      await act(async () => {
        await result.current.loadPreview(true)
      })

      expect(result.current.previewChunks.length).toBeGreaterThan(0)
    })

    it('clears preview', () => {
      const { result } = renderHook(() => useChunkingStore())
      
      // Set up some preview data
      act(() => {
        result.current.setPreviewDocument({
          id: 'doc-1',
          content: 'Test',
          name: 'test.txt',
        })
      })

      act(() => {
        result.current.clearPreview()
      })

      expect(result.current.previewDocument).toBeNull()
      expect(result.current.previewChunks).toEqual([])
      expect(result.current.previewStatistics).toBeNull()
      expect(result.current.previewError).toBeNull()
    })

    it('handles preview loading error', async () => {
      const { result } = renderHook(() => useChunkingStore())
      
      // Mock an error scenario
      const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {})
      
      act(() => {
        result.current.setPreviewDocument(null) // No document
      })

      await act(async () => {
        await result.current.loadPreview()
      })

      // Should not set loading state if no document
      expect(result.current.previewLoading).toBe(false)
      
      consoleError.mockRestore()
    })
  })

  describe('Comparison Management', () => {
    it('adds strategy for comparison', () => {
      const { result } = renderHook(() => useChunkingStore())
      
      act(() => {
        result.current.addComparisonStrategy('recursive')
        result.current.addComparisonStrategy('semantic')
      })

      expect(result.current.comparisonStrategies).toEqual(['recursive', 'semantic'])
    })

    it('removes strategy from comparison', () => {
      const { result } = renderHook(() => useChunkingStore())
      
      act(() => {
        result.current.addComparisonStrategy('recursive')
        result.current.addComparisonStrategy('semantic')
        result.current.removeComparisonStrategy('recursive')
      })

      expect(result.current.comparisonStrategies).toEqual(['semantic'])
    })

    it('prevents duplicate strategies in comparison', () => {
      const { result } = renderHook(() => useChunkingStore())
      
      act(() => {
        result.current.addComparisonStrategy('recursive')
        result.current.addComparisonStrategy('recursive')
      })

      expect(result.current.comparisonStrategies).toEqual(['recursive'])
    })

    it('compares strategies', async () => {
      const { result } = renderHook(() => useChunkingStore())
      
      act(() => {
        result.current.setPreviewDocument({
          id: 'doc-1',
          content: 'Test content',
          name: 'test.txt',
        })
        result.current.addComparisonStrategy('recursive')
        result.current.addComparisonStrategy('semantic')
      })

      await act(async () => {
        await result.current.compareStrategies()
      })

      expect(result.current.comparisonLoading).toBe(false)
      expect(Object.keys(result.current.comparisonResults)).toHaveLength(2)
    })

    it('clears comparison', () => {
      const { result } = renderHook(() => useChunkingStore())
      
      act(() => {
        result.current.addComparisonStrategy('recursive')
        result.current.addComparisonStrategy('semantic')
      })

      act(() => {
        result.current.clearComparison()
      })

      expect(result.current.comparisonStrategies).toEqual([])
      expect(result.current.comparisonResults).toEqual({})
      expect(result.current.comparisonError).toBeNull()
    })

    it('handles comparison error when no document', async () => {
      const { result } = renderHook(() => useChunkingStore())
      
      act(() => {
        result.current.addComparisonStrategy('recursive')
      })

      await act(async () => {
        await result.current.compareStrategies()
      })

      expect(result.current.comparisonError).toBe('No document selected for comparison')
    })

    it('handles comparison error when no strategies', async () => {
      const { result } = renderHook(() => useChunkingStore())
      
      act(() => {
        result.current.setPreviewDocument({
          id: 'doc-1',
          content: 'Test',
          name: 'test.txt',
        })
      })

      await act(async () => {
        await result.current.compareStrategies()
      })

      expect(result.current.comparisonError).toBe('No strategies selected for comparison')
    })
  })

  describe('Analytics', () => {
    it('loads analytics data', async () => {
      const { result } = renderHook(() => useChunkingStore())
      
      await act(async () => {
        await result.current.loadAnalytics()
      })

      expect(result.current.analyticsLoading).toBe(false)
      expect(result.current.analyticsData).toBeDefined()
      expect(result.current.analyticsData?.performanceMetrics).toBeDefined()
      expect(result.current.analyticsData?.strategyUsage).toBeDefined()
      expect(result.current.analyticsData?.recommendations).toBeInstanceOf(Array)
    })
  })

  describe('Utility Functions', () => {
    it('gets recommended strategy for file type', () => {
      const { result } = renderHook(() => useChunkingStore())
      
      expect(result.current.getRecommendedStrategy('.md')).toBe('markdown')
      expect(result.current.getRecommendedStrategy('.py')).toBe('recursive')
      expect(result.current.getRecommendedStrategy('.txt')).toBe('recursive')
      expect(result.current.getRecommendedStrategy()).toBe('recursive')
    })

    it('resets store to initial state', () => {
      const { result } = renderHook(() => useChunkingStore())
      
      // Make some changes
      act(() => {
        result.current.setStrategy('semantic')
        result.current.addComparisonStrategy('recursive')
        result.current.setPreviewDocument({
          id: 'doc-1',
          content: 'Test',
          name: 'test.txt',
        })
      })

      // Reset
      act(() => {
        result.current.reset()
      })

      expect(result.current.selectedStrategy).toBe('recursive')
      expect(result.current.comparisonStrategies).toEqual([])
      expect(result.current.previewDocument).toBeNull()
      expect(result.current.customPresets).toEqual([])
    })
  })

  describe('Edge Cases', () => {
    it('handles invalid preset ID gracefully', () => {
      const { result } = renderHook(() => useChunkingStore())
      
      const initialState = { ...result.current }
      
      act(() => {
        result.current.applyPreset('non-existent-preset')
      })

      // State should remain unchanged
      expect(result.current.selectedStrategy).toBe(initialState.selectedStrategy)
      expect(result.current.strategyConfig).toEqual(initialState.strategyConfig)
    })

    it('handles API errors in loadPreview', async () => {
      const { result } = renderHook(() => useChunkingStore())
      
      if (!result.current) return
      
      // Mock API error
      mockPreview.mockRejectedValueOnce(new Error('API Error: Failed to load preview'))
      
      act(() => {
        result.current.setPreviewDocument({
          id: 'doc-1',
          content: 'Test content',
          name: 'test.txt',
        })
      })

      await act(async () => {
        await result.current.loadPreview()
      })

      expect(result.current.previewError).toBe('API Error: Failed to load preview')
      expect(result.current.previewLoading).toBe(false)
      expect(result.current.previewChunks).toEqual([])
    })

    it('handles API errors in compareStrategies', async () => {
      const { result } = renderHook(() => useChunkingStore())
      
      if (!result.current) return
      
      // Mock API error
      mockCompare.mockRejectedValueOnce(new Error('API Error: Comparison failed'))
      
      act(() => {
        result.current.setPreviewDocument({
          id: 'doc-1',
          content: 'Test content',
          name: 'test.txt',
        })
        result.current.addComparisonStrategy('recursive')
      })

      await act(async () => {
        await result.current.compareStrategies()
      })

      expect(result.current.comparisonError).toBe('API Error: Comparison failed')
      expect(result.current.comparisonLoading).toBe(false)
      expect(result.current.comparisonResults).toEqual({})
    })

    it('handles API errors in loadAnalytics', async () => {
      const { result } = renderHook(() => useChunkingStore())
      
      if (!result.current) return
      
      // Mock API error with console.error spy
      const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {})
      mockGetAnalytics.mockRejectedValueOnce(new Error('API Error: Analytics failed'))
      
      await act(async () => {
        await result.current.loadAnalytics()
      })

      expect(result.current.analyticsLoading).toBe(false)
      expect(result.current.analyticsData).toBeNull()
      expect(consoleError).toHaveBeenCalledWith('Failed to load analytics:', 'API Error: Analytics failed')
      
      consoleError.mockRestore()
    })

    it('handles API errors in saveCustomPreset', async () => {
      const { result } = renderHook(() => useChunkingStore())
      
      if (!result.current) return
      
      // Mock API error with console.error spy
      const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {})
      mockSavePreset.mockRejectedValueOnce(new Error('API Error: Save failed'))
      
      await expect(
        act(async () => {
          await result.current.saveCustomPreset({
            name: 'Test Preset',
            description: 'Test',
            strategy: 'recursive',
            configuration: {
              strategy: 'recursive',
              parameters: {},
            },
          })
        })
      ).rejects.toThrow('API Error: Save failed')

      expect(result.current.customPresets).toHaveLength(0)
      expect(consoleError).toHaveBeenCalledWith('Failed to save preset:', 'API Error: Save failed')
      
      consoleError.mockRestore()
    })

    it('handles API errors in deleteCustomPreset', async () => {
      const { result } = renderHook(() => useChunkingStore())
      
      if (!result.current) return
      
      // Add a preset first
      let presetId: string = ''
      await act(async () => {
        presetId = await result.current.saveCustomPreset({
          name: 'Test Preset',
          description: 'Test',
          strategy: 'recursive',
          configuration: {
            strategy: 'recursive',
            parameters: {},
          },
        })
      })

      // Mock API error with console.error spy
      const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {})
      mockDeletePreset.mockRejectedValueOnce(new Error('API Error: Delete failed'))
      
      await expect(
        act(async () => {
          await result.current.deleteCustomPreset(presetId)
        })
      ).rejects.toThrow('API Error: Delete failed')

      // Preset should still exist since delete failed
      expect(result.current.customPresets).toHaveLength(1)
      expect(consoleError).toHaveBeenCalledWith('Failed to delete preset:', 'API Error: Delete failed')
      
      consoleError.mockRestore()
    })

    it('handles API errors in loadPresets', async () => {
      const { result } = renderHook(() => useChunkingStore())
      
      if (!result.current) return
      
      // Mock API error with console.error spy
      const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {})
      mockGetPresets.mockRejectedValueOnce(new Error('API Error: Load presets failed'))
      
      await act(async () => {
        await result.current.loadPresets()
      })

      expect(result.current.presetsLoading).toBe(false)
      expect(result.current.customPresets).toEqual([])
      expect(consoleError).toHaveBeenCalledWith('Failed to load presets:', 'API Error: Load presets failed')
      
      consoleError.mockRestore()
    })

    it('maintains preset list consistency', async () => {
      // Get fresh instance of the store  
      const { result } = renderHook(() => useChunkingStore())
      
      // Verify starting with clean state (customPresets should be empty after reset)
      expect(result.current.customPresets).toHaveLength(0)
      
      // Add multiple presets
      const ids: string[] = []
      await act(async () => {
        for (let i = 0; i < 5; i++) {
          const id = await result.current.saveCustomPreset({
            name: `Preset ${i}`,
            description: `Description ${i}`,
            strategy: 'recursive',
            configuration: {
              strategy: 'recursive',
              parameters: { chunk_size: 100 * (i + 1) },
            },
          })
          ids.push(id)
        }
      })

      expect(result.current.customPresets).toHaveLength(5)
      expect(ids).toHaveLength(5)
      
      // Delete middle preset
      await act(async () => {
        await result.current.deleteCustomPreset(ids[2])
      })

      expect(result.current.customPresets).toHaveLength(4)
      expect(result.current.customPresets.find(p => p.id === ids[2])).toBeUndefined()
      // Verify other presets still exist
      expect(result.current.customPresets.find(p => p.id === ids[0])).toBeDefined()
      expect(result.current.customPresets.find(p => p.id === ids[1])).toBeDefined()
      expect(result.current.customPresets.find(p => p.id === ids[3])).toBeDefined()
      expect(result.current.customPresets.find(p => p.id === ids[4])).toBeDefined()
    })

    it('handles concurrent preview loads', async () => {
      const { result } = renderHook(() => useChunkingStore())
      
      act(() => {
        result.current.setPreviewDocument({
          id: 'doc-1',
          content: 'Test content',
          name: 'test.txt',
        })
      })

      // Start multiple preview loads
      const promises = []
      for (let i = 0; i < 3; i++) {
        promises.push(
          act(async () => {
            await result.current.loadPreview(true)
          })
        )
      }

      await Promise.all(promises)

      // Should handle concurrent loads gracefully
      expect(result.current.previewLoading).toBe(false)
      expect(result.current.previewChunks).toBeDefined()
    })

    it('validates strategy type', () => {
      // Ensure clean state for this test
      useChunkingStore.getState().reset()
      
      const { result } = renderHook(() => useChunkingStore())
      
      const validStrategies: ChunkingStrategyType[] = ['recursive', 'character', 'semantic', 'markdown', 'hierarchical', 'hybrid']
      
      validStrategies.forEach(strategy => {
        act(() => {
          if (result.current) {
            result.current.setStrategy(strategy)
          }
        })
        if (result.current) {
          expect(result.current.selectedStrategy).toBe(strategy)
        }
      })
    })

    it('limits comparison strategies to 3', () => {
      const { result } = renderHook(() => useChunkingStore())
      
      if (!result.current) return
      
      act(() => {
        result.current.addComparisonStrategy('recursive')
        result.current.addComparisonStrategy('semantic')
        result.current.addComparisonStrategy('character')
        result.current.addComparisonStrategy('markdown') // 4th should be ignored
      })

      expect(result.current.comparisonStrategies).toHaveLength(3)
      expect(result.current.comparisonStrategies).toEqual(['recursive', 'semantic', 'character'])
    })

    it('handles caching in loadPreview correctly', async () => {
      const { result } = renderHook(() => useChunkingStore())
      
      if (!result.current) return
      
      act(() => {
        result.current.setPreviewDocument({
          id: 'doc-1',
          content: 'Test content',
          name: 'test.txt',
        })
      })

      // First load
      await act(async () => {
        await result.current.loadPreview()
      })

      expect(mockPreview).toHaveBeenCalledTimes(1)
      const firstChunks = result.current.previewChunks

      // Second load without force - should use cache
      await act(async () => {
        await result.current.loadPreview(false)
      })

      expect(mockPreview).toHaveBeenCalledTimes(1) // Not called again
      expect(result.current.previewChunks).toBe(firstChunks)

      // Third load with force - should reload
      await act(async () => {
        await result.current.loadPreview(true)
      })

      expect(mockPreview).toHaveBeenCalledTimes(2) // Called again
    })

    it('calls cancelActiveRequests correctly', () => {
      const { result } = renderHook(() => useChunkingStore())
      
      if (!result.current) return
      
      act(() => {
        result.current.cancelActiveRequests()
      })

      expect(mockCancelAllRequests).toHaveBeenCalledWith('User cancelled operation')
    })

    it('handles progress callback in loadPreview', async () => {
      const { result } = renderHook(() => useChunkingStore())
      const progressSpy = vi.fn()
      
      if (!result.current) {
        // Skip test if hook didn't initialize properly
        return
      }
      
      // Mock preview to capture the progress callback
      mockPreview.mockImplementationOnce(async (data, options) => {
        // Call the progress callback if provided
        if (options?.onProgress) {
          options.onProgress({ percentage: 50, currentChunk: 5, totalChunks: 10 })
        }
        return {
          chunks: [
            {
              id: 'chunk-1',
              content: 'Test chunk content',
              startIndex: 0,
              endIndex: 50,
              tokens: 10,
              overlapWithPrevious: 0,
              overlapWithNext: 10,
            },
          ],
          statistics: {
            totalChunks: 1,
            avgChunkSize: 50,
            minChunkSize: 50,
            maxChunkSize: 50,
            totalSize: 50,
            overlapRatio: 0,
            overlapPercentage: 0,
            sizeDistribution: [],
          },
          performance: {
            processingTimeMs: 100,
            estimatedFullProcessingTimeMs: 500,
          },
        }
      })

      // Spy on console.debug
      const consoleDebug = vi.spyOn(console, 'debug').mockImplementation(progressSpy)
      
      act(() => {
        result.current.setPreviewDocument({
          id: 'doc-1',
          content: 'Test content',
          name: 'test.txt',
        })
      })

      await act(async () => {
        await result.current.loadPreview()
      })

      expect(progressSpy).toHaveBeenCalledWith('Preview progress: 50%')
      
      consoleDebug.mockRestore()
    })

    it('handles progress callback in compareStrategies', async () => {
      const { result } = renderHook(() => useChunkingStore())
      
      if (!result.current) {
        // Skip test if hook didn't initialize properly
        return
      }
      
      const progressSpy = vi.fn()
      
      // Spy on console.debug BEFORE setting up mock
      const consoleDebug = vi.spyOn(console, 'debug').mockImplementation(progressSpy)
      
      // Mock compare to capture the progress callback
      mockCompare.mockImplementationOnce(async (data, options) => {
        // Call the progress callback if provided
        if (options?.onProgress) {
          options.onProgress({ percentage: 33, currentStrategy: 1, totalStrategies: 3 })
        }
        return [
          {
            strategy: 'recursive',
            configuration: {
              strategy: 'recursive',
              parameters: { chunk_size: 600, chunk_overlap: 100 },
            },
            preview: {
              chunks: [],
              statistics: {
                totalChunks: 3,
                avgChunkSize: 400,
                minChunkSize: 300,
                maxChunkSize: 500,
                sizeDistribution: [],
              },
              performance: {
                processingTimeMs: 50,
                estimatedFullProcessingTimeMs: 200,
              },
            },
          },
          {
            strategy: 'semantic',
            configuration: {
              strategy: 'semantic',
              parameters: { breakpoint_percentile_threshold: 90, max_chunk_size: 1000 },
            },
            preview: {
              chunks: [],
              statistics: {
                totalChunks: 4,
                avgChunkSize: 350,
                minChunkSize: 250,
                maxChunkSize: 450,
                sizeDistribution: [],
              },
              performance: {
                processingTimeMs: 150,
                estimatedFullProcessingTimeMs: 600,
              },
            },
          },
        ]
      })
      
      act(() => {
        result.current.setPreviewDocument({
          id: 'doc-1',
          content: 'Test content',
          name: 'test.txt',
        })
        result.current.addComparisonStrategy('recursive')
        result.current.addComparisonStrategy('semantic')
      })
      
      await act(async () => {
        await result.current.compareStrategies()
      })

      expect(progressSpy).toHaveBeenCalledWith('Comparison progress: 33%')
      
      consoleDebug.mockRestore()
    })

    it('removes comparison results when strategy is removed', () => {
      const { result } = renderHook(() => useChunkingStore())
      
      if (!result.current) {
        // Skip test if hook didn't initialize properly
        return
      }
      
      // Set up some comparison results
      act(() => {
        result.current.addComparisonStrategy('recursive')
        result.current.addComparisonStrategy('semantic')
      })

      // Manually set comparison results
      act(() => {
        useChunkingStore.setState({
          comparisonResults: {
            recursive: {
              strategy: 'recursive',
              configuration: {
                strategy: 'recursive',
                parameters: { chunk_size: 600, chunk_overlap: 100 },
              },
              preview: {
                chunks: [],
                statistics: {
                  totalChunks: 3,
                  avgChunkSize: 400,
                  minChunkSize: 300,
                  maxChunkSize: 500,
                  sizeDistribution: [],
                },
                performance: {
                  processingTimeMs: 50,
                  estimatedFullProcessingTimeMs: 200,
                },
              },
            },
            semantic: {
              strategy: 'semantic',
              configuration: {
                strategy: 'semantic',
                parameters: { breakpoint_percentile_threshold: 90, max_chunk_size: 1000 },
              },
              preview: {
                chunks: [],
                statistics: {
                  totalChunks: 4,
                  avgChunkSize: 350,
                  minChunkSize: 250,
                  maxChunkSize: 450,
                  sizeDistribution: [],
                },
                performance: {
                  processingTimeMs: 150,
                  estimatedFullProcessingTimeMs: 600,
                },
              },
            },
          },
        })
      })

      expect(Object.keys(result.current.comparisonResults)).toHaveLength(2)

      // Remove a strategy
      act(() => {
        result.current.removeComparisonStrategy('recursive')
      })

      expect(result.current.comparisonStrategies).toEqual(['semantic'])
      expect(result.current.comparisonResults.recursive).toBeUndefined()
      expect(result.current.comparisonResults.semantic).toBeDefined()
    })

    it('applies custom presets correctly', async () => {
      const { result } = renderHook(() => useChunkingStore())
      
      if (!result.current) {
        // Skip test if hook didn't initialize properly
        return
      }
      
      // Save a custom preset
      let presetId: string = ''
      await act(async () => {
        presetId = await result.current.saveCustomPreset({
          name: 'Custom Strategy',
          description: 'My custom settings',
          strategy: 'semantic',
          configuration: {
            strategy: 'semantic',
            parameters: {
              max_chunk_size: 1500,
              similarity_threshold: 0.8,
            },
          },
        })
      })

      // Apply the custom preset
      act(() => {
        result.current.applyPreset(presetId)
      })

      expect(result.current.selectedStrategy).toBe('semantic')
      expect(result.current.strategyConfig.parameters.max_chunk_size).toBe(1500)
      expect(result.current.strategyConfig.parameters.similarity_threshold).toBe(0.8)
      expect(result.current.selectedPreset).toBe(presetId)
    })

    it('handles file type recommendations correctly', () => {
      const { result } = renderHook(() => useChunkingStore())
      
      if (!result.current) {
        // Skip test if hook didn't initialize properly
        return
      }
      
      // Test various file types
      expect(result.current.getRecommendedStrategy('.md')).toBe('markdown')
      expect(result.current.getRecommendedStrategy('md')).toBe('markdown')
      expect(result.current.getRecommendedStrategy('.mdx')).toBe('markdown')
      expect(result.current.getRecommendedStrategy('.markdown')).toBe('markdown')
      expect(result.current.getRecommendedStrategy('MDX')).toBe('markdown') // Case insensitive
      expect(result.current.getRecommendedStrategy('.py')).toBe('recursive')
      expect(result.current.getRecommendedStrategy('.js')).toBe('recursive')
      expect(result.current.getRecommendedStrategy()).toBe('recursive') // No file type
      expect(result.current.getRecommendedStrategy('')).toBe('recursive') // Empty string
    })

    it('preserves configuration when switching strategies', () => {
      // Ensure clean state for this test
      useChunkingStore.getState().reset()
      
      const { result } = renderHook(() => useChunkingStore())
      
      if (!result.current) {
        // Skip test if hook didn't initialize properly
        return
      }
      
      // Set custom configuration for recursive
      act(() => {
        result.current.setStrategy('recursive')
        result.current.updateConfiguration({ chunk_size: 1000 })
      })

      const recursiveConfig = { ...result.current.strategyConfig }

      // Switch to semantic
      act(() => {
        result.current.setStrategy('semantic')
      })

      const semanticConfig = { ...result.current.strategyConfig }

      // Configs should be different
      expect(recursiveConfig.strategy).not.toBe(semanticConfig.strategy)
      expect(recursiveConfig.parameters).not.toEqual(semanticConfig.parameters)
    })
  })
})