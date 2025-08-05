import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { act, renderHook } from '@testing-library/react'
import { useChunkingStore } from '../chunkingStore'
import { CHUNKING_STRATEGIES, CHUNKING_PRESETS } from '@/types/chunking'
import type { ChunkingStrategyType, ChunkingPreset } from '@/types/chunking'

// Mock API calls
vi.mock('@/api/chunking', () => ({
  chunkingApi: {
    preview: vi.fn(),
    compare: vi.fn(),
    getAnalytics: vi.fn(),
  },
}))

describe('chunkingStore', () => {
  beforeEach(() => {
    // Reset store state before each test
    const { result } = renderHook(() => useChunkingStore())
    act(() => {
      result.current.reset()
    })
    vi.clearAllMocks()
  })

  afterEach(() => {
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

    it('saves custom preset', () => {
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
      
      let presetId: string
      act(() => {
        presetId = result.current.saveCustomPreset(customPreset)
      })

      expect(result.current.customPresets).toHaveLength(1)
      expect(result.current.customPresets[0].name).toBe('My Custom Preset')
      expect(result.current.customPresets[0].id).toMatch(/^custom-\d+$/)
    })

    it('deletes custom preset', () => {
      const { result } = renderHook(() => useChunkingStore())
      
      // First save a preset
      let presetId: string
      act(() => {
        presetId = result.current.saveCustomPreset({
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
      act(() => {
        result.current.deleteCustomPreset(presetId)
      })

      expect(result.current.customPresets).toHaveLength(0)
    })

    it('clears selected preset when deleting the active preset', () => {
      const { result } = renderHook(() => useChunkingStore())
      
      let presetId: string
      act(() => {
        presetId = result.current.saveCustomPreset({
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

      act(() => {
        result.current.deleteCustomPreset(presetId)
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
      expect(result.current.analyticsData?.performance).toBeDefined()
      expect(result.current.analyticsData?.quality).toBeDefined()
      expect(result.current.analyticsData?.recommendations).toBeInstanceOf(Array)
    })
  })

  describe('Utility Functions', () => {
    it('gets recommended strategy for file type', () => {
      const { result } = renderHook(() => useChunkingStore())
      
      expect(result.current.getRecommendedStrategy('.md')).toBe('markdown')
      expect(result.current.getRecommendedStrategy('.py')).toBe('code')
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

    it('maintains preset list consistency', () => {
      const { result } = renderHook(() => useChunkingStore())
      
      // Add multiple presets
      const ids: string[] = []
      act(() => {
        for (let i = 0; i < 5; i++) {
          ids.push(result.current.saveCustomPreset({
            name: `Preset ${i}`,
            description: `Description ${i}`,
            strategy: 'recursive',
            configuration: {
              strategy: 'recursive',
              parameters: { chunk_size: 100 * i },
            },
          }))
        }
      })

      expect(result.current.customPresets).toHaveLength(5)

      // Delete middle preset
      act(() => {
        result.current.deleteCustomPreset(ids[2])
      })

      expect(result.current.customPresets).toHaveLength(4)
      expect(result.current.customPresets.find(p => p.id === ids[2])).toBeUndefined()
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
      const { result } = renderHook(() => useChunkingStore())
      
      const validStrategies: ChunkingStrategyType[] = ['recursive', 'fixed', 'semantic', 'markdown', 'code']
      
      validStrategies.forEach(strategy => {
        act(() => {
          result.current.setStrategy(strategy)
        })
        expect(result.current.selectedStrategy).toBe(strategy)
      })
    })

    it('preserves configuration when switching strategies', () => {
      const { result } = renderHook(() => useChunkingStore())
      
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