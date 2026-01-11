import { describe, it, expect, beforeEach } from 'vitest'
import { act, renderHook } from '@testing-library/react'
import { useSearchStore, type SearchResult } from '../searchStore'

describe('searchStore', () => {
  const mockSearchResult1: SearchResult = {
    doc_id: 'doc1',
    chunk_id: 'chunk1',
    score: 0.95,
    content: 'This is the first search result',
    file_path: '/path/to/file1.txt',
    file_name: 'file1.txt',
    chunk_index: 0,
    total_chunks: 5,
    collection_id: 'collection1',
    collection_name: 'Test Collection',
  }

  const mockSearchResult2: SearchResult = {
    doc_id: 'doc2',
    chunk_id: 'chunk2',
    score: 0.85,
    content: 'This is the second search result',
    file_path: '/path/to/file2.txt',
    file_name: 'file2.txt',
    chunk_index: 2,
    total_chunks: 3,
  }

  beforeEach(() => {
    // Reset store state before each test
    const { result } = renderHook(() => useSearchStore())
    act(() => {
      result.current.clearResults()
      result.current.setLoading(false)
      result.current.setError(null)
      result.current.updateSearchParams({
        query: '',
        selectedCollections: [],
        topK: 10,
        scoreThreshold: 0.0,
        searchType: 'semantic',
        useReranker: false,
        hybridMode: 'weighted',
        keywordMode: 'any',
        hybridAlpha: 0.7,
        searchMode: 'dense',
        rrfK: 60,
      })
      result.current.setCollections([])
    })
  })

  describe('results management', () => {
    it('starts with empty results', () => {
      const { result } = renderHook(() => useSearchStore())
      expect(result.current.results).toEqual([])
    })

    it('sets results correctly', () => {
      const { result } = renderHook(() => useSearchStore())
      
      act(() => {
        result.current.setResults([mockSearchResult1, mockSearchResult2])
      })

      expect(result.current.results).toHaveLength(2)
      expect(result.current.results[0]).toEqual(mockSearchResult1)
      expect(result.current.results[1]).toEqual(mockSearchResult2)
    })

    it('clears results and resets related state', () => {
      const { result } = renderHook(() => useSearchStore())
      
      act(() => {
        result.current.setResults([mockSearchResult1])
        result.current.setError('Some error')
        result.current.setPartialFailure(true)
        result.current.setFailedCollections([
          { collection_id: '1', collection_name: 'Test', error_message: 'Failed' }
        ])
        result.current.setRerankingMetrics({
          rerankingUsed: true,
          rerankerModel: 'model-123',
          rerankingTimeMs: 150,
        })
      })

      expect(result.current.results).toHaveLength(1)
      expect(result.current.error).toBe('Some error')
      expect(result.current.partialFailure).toBe(true)
      expect(result.current.failedCollections).toHaveLength(1)
      expect(result.current.rerankingMetrics).not.toBeNull()

      act(() => {
        result.current.clearResults()
      })

      expect(result.current.results).toEqual([])
      expect(result.current.error).toBeNull()
      expect(result.current.partialFailure).toBe(false)
      expect(result.current.failedCollections).toEqual([])
      expect(result.current.rerankingMetrics).toBeNull()
    })
  })

  describe('loading and error states', () => {
    it('manages loading state', () => {
      const { result } = renderHook(() => useSearchStore())
      
      expect(result.current.loading).toBe(false)

      act(() => {
        result.current.setLoading(true)
      })
      expect(result.current.loading).toBe(true)

      act(() => {
        result.current.setLoading(false)
      })
      expect(result.current.loading).toBe(false)
    })

    it('manages error state', () => {
      const { result } = renderHook(() => useSearchStore())
      
      expect(result.current.error).toBeNull()

      act(() => {
        result.current.setError('Search failed')
      })
      expect(result.current.error).toBe('Search failed')

      act(() => {
        result.current.setError(null)
      })
      expect(result.current.error).toBeNull()
    })
  })

  describe('search parameters', () => {
    it('has correct default search params', () => {
      const { result } = renderHook(() => useSearchStore())

      expect(result.current.searchParams).toEqual({
        query: '',
        selectedCollections: [],
        topK: 10,
        scoreThreshold: 0.0,
        searchType: 'semantic',
        useReranker: false,
        hybridMode: 'weighted',
        keywordMode: 'any',
        // New sparse/hybrid search params
        searchMode: 'dense',
        rrfK: 60,
        // Legacy params (deprecated)
        hybridAlpha: 0.7,
      })
    })

    it('updates search params partially', () => {
      const { result } = renderHook(() => useSearchStore())
      
      act(() => {
        result.current.updateSearchParams({
          query: 'test query',
          selectedCollections: ['my-collection'],
        })
      })

      expect(result.current.searchParams.query).toBe('test query')
      expect(result.current.searchParams.selectedCollections).toEqual(['my-collection'])
      // Other params should remain unchanged
      expect(result.current.searchParams.topK).toBe(10)
      expect(result.current.searchParams.searchType).toBe('semantic')
    })

    it('updates hybrid search params', () => {
      const { result } = renderHook(() => useSearchStore())
      
      act(() => {
        result.current.updateSearchParams({
          searchType: 'hybrid',
          hybridAlpha: 0.7,
          hybridMode: 'filter',
          keywordMode: 'all',
        })
      })

      expect(result.current.searchParams.searchType).toBe('hybrid')
      expect(result.current.searchParams.hybridAlpha).toBe(0.7)
      expect(result.current.searchParams.hybridMode).toBe('filter')
      expect(result.current.searchParams.keywordMode).toBe('all')
    })

    it('updates reranking params', () => {
      const { result } = renderHook(() => useSearchStore())
      
      act(() => {
        result.current.updateSearchParams({
          useReranker: true,
          rerankModel: 'BAAI/bge-reranker-base',
          rerankQuantization: 'int8',
        })
      })

      expect(result.current.searchParams.useReranker).toBe(true)
      expect(result.current.searchParams.rerankModel).toBe('BAAI/bge-reranker-base')
      expect(result.current.searchParams.rerankQuantization).toBe('int8')
    })

    it('preserves unspecified params when updating', () => {
      const { result } = renderHook(() => useSearchStore())
      
      // Set initial params
      act(() => {
        result.current.updateSearchParams({
          query: 'initial query',
          topK: 20,
          scoreThreshold: 0.5,
        })
      })

      // Update different params
      act(() => {
        result.current.updateSearchParams({
          selectedCollections: ['new-collection'],
          searchType: 'hybrid',
        })
      })

      // Original params should be preserved
      expect(result.current.searchParams.query).toBe('initial query')
      expect(result.current.searchParams.topK).toBe(20)
      expect(result.current.searchParams.scoreThreshold).toBe(0.5)
      // New params should be updated
      expect(result.current.searchParams.selectedCollections).toEqual(['new-collection'])
      expect(result.current.searchParams.searchType).toBe('hybrid')
    })
  })

  describe('collections', () => {
    it('starts with empty collections', () => {
      const { result } = renderHook(() => useSearchStore())
      expect(result.current.collections).toEqual([])
    })

    it('sets collections correctly', () => {
      const { result } = renderHook(() => useSearchStore())
      
      const collections = ['collection1', 'collection2', 'collection3']
      
      act(() => {
        result.current.setCollections(collections)
      })

      expect(result.current.collections).toEqual(collections)
    })
  })

  describe('reranking metrics', () => {
    it('starts with null reranking metrics', () => {
      const { result } = renderHook(() => useSearchStore())
      expect(result.current.rerankingMetrics).toBeNull()
    })

    it('sets reranking metrics correctly', () => {
      const { result } = renderHook(() => useSearchStore())
      
      const metrics = {
        rerankingUsed: true,
        rerankerModel: 'BAAI/bge-reranker-base',
        rerankingTimeMs: 125.5,
      }

      act(() => {
        result.current.setRerankingMetrics(metrics)
      })

      expect(result.current.rerankingMetrics).toEqual(metrics)
    })

    it('sets minimal reranking metrics', () => {
      const { result } = renderHook(() => useSearchStore())
      
      act(() => {
        result.current.setRerankingMetrics({
          rerankingUsed: false,
        })
      })

      expect(result.current.rerankingMetrics).toEqual({
        rerankingUsed: false,
      })
    })

    it('clears reranking metrics', () => {
      const { result } = renderHook(() => useSearchStore())
      
      act(() => {
        result.current.setRerankingMetrics({
          rerankingUsed: true,
          rerankingTimeMs: 100,
        })
      })

      expect(result.current.rerankingMetrics).not.toBeNull()

      act(() => {
        result.current.setRerankingMetrics(null)
      })

      expect(result.current.rerankingMetrics).toBeNull()
    })
  })

  describe('complex scenarios', () => {
    it('handles full search flow', () => {
      const { result } = renderHook(() => useSearchStore())
      
      // Set up search
      act(() => {
        result.current.updateSearchParams({
          query: 'machine learning',
          selectedCollections: ['docs'],
          topK: 5,
          useReranker: true,
          rerankModel: 'model-123',
        })
        result.current.setLoading(true)
      })

      expect(result.current.loading).toBe(true)

      // Simulate search completion
      act(() => {
        result.current.setResults([mockSearchResult1, mockSearchResult2])
        result.current.setRerankingMetrics({
          rerankingUsed: true,
          rerankerModel: 'model-123',
          rerankingTimeMs: 85,
        })
        result.current.setLoading(false)
      })

      expect(result.current.loading).toBe(false)
      expect(result.current.results).toHaveLength(2)
      expect(result.current.rerankingMetrics?.rerankingTimeMs).toBe(85)
    })

    it('handles search error', () => {
      const { result } = renderHook(() => useSearchStore())
      
      // Start search
      act(() => {
        result.current.setLoading(true)
      })

      // Simulate error
      act(() => {
        result.current.setError('Network error occurred')
        result.current.setLoading(false)
      })

      expect(result.current.loading).toBe(false)
      expect(result.current.error).toBe('Network error occurred')
      expect(result.current.results).toEqual([])
    })
  })

  describe('state isolation', () => {
    it('shares state between multiple hooks', () => {
      const { result: hook1 } = renderHook(() => useSearchStore())
      const { result: hook2 } = renderHook(() => useSearchStore())
      
      act(() => {
        hook1.current.updateSearchParams({ query: 'shared query' })
        hook1.current.setResults([mockSearchResult1])
      })
      
      // Both hooks should see the same state
      expect(hook2.current.searchParams.query).toBe('shared query')
      expect(hook2.current.results).toHaveLength(1)
      expect(hook2.current.results[0]).toEqual(mockSearchResult1)
    })
  })
})
