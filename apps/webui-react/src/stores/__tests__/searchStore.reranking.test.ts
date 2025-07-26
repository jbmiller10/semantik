import { describe, it, expect, beforeEach } from 'vitest';
import { useSearchStore } from '../searchStore';

describe('Search Store Reranking Tests', () => {
  beforeEach(() => {
    // Reset store to initial state
    useSearchStore.setState({
      results: [],
      loading: false,
      error: null,
      searchParams: {
        query: '',
        selectedCollections: [],
        topK: 10,
        scoreThreshold: 0.0,
        searchType: 'semantic',
        useReranker: false,
        rerankModel: undefined,
        rerankQuantization: undefined,
      },
      collections: [],
      failedCollections: [],
      partialFailure: false,
      rerankingMetrics: null,
    });
  });

  it('should update reranking parameters correctly', () => {
    const { updateSearchParams, searchParams } = useSearchStore.getState();

    // Enable reranking
    updateSearchParams({ useReranker: true });
    let state = useSearchStore.getState();
    expect(state.searchParams.useReranker).toBe(true);

    // Set reranker model
    updateSearchParams({ rerankModel: 'Qwen/Qwen3-Reranker-0.6B' });
    state = useSearchStore.getState();
    expect(state.searchParams.rerankModel).toBe('Qwen/Qwen3-Reranker-0.6B');

    // Set quantization
    updateSearchParams({ rerankQuantization: 'float16' });
    state = useSearchStore.getState();
    expect(state.searchParams.rerankQuantization).toBe('float16');
  });

  it('should store reranking metrics correctly', () => {
    const { setRerankingMetrics } = useSearchStore.getState();

    const metrics = {
      rerankingUsed: true,
      rerankerModel: 'Qwen/Qwen3-Reranker-0.6B',
      rerankingTimeMs: 75.5,
    };

    setRerankingMetrics(metrics);

    const state = useSearchStore.getState();
    expect(state.rerankingMetrics).toEqual(metrics);
  });

  it('should clear reranking metrics when clearing results', () => {
    const { setRerankingMetrics, clearResults } = useSearchStore.getState();

    // Set some metrics
    setRerankingMetrics({
      rerankingUsed: true,
      rerankerModel: 'Qwen/Qwen3-Reranker-0.6B',
      rerankingTimeMs: 50,
    });

    // Clear results
    clearResults();

    const state = useSearchStore.getState();
    expect(state.rerankingMetrics).toBeNull();
  });

  it('should handle reranking parameters with hybrid search', () => {
    const { updateSearchParams } = useSearchStore.getState();

    // Enable hybrid search with reranking
    updateSearchParams({
      searchType: 'hybrid',
      useReranker: true,
      hybridAlpha: 0.5,
      hybridMode: 'relative_score',
      keywordMode: 'bm25',
      rerankModel: 'Qwen/Qwen3-Reranker-4B',
    });

    const state = useSearchStore.getState();
    expect(state.searchParams.searchType).toBe('hybrid');
    expect(state.searchParams.useReranker).toBe(true);
    expect(state.searchParams.hybridAlpha).toBe(0.5);
    expect(state.searchParams.hybridMode).toBe('relative_score');
    expect(state.searchParams.keywordMode).toBe('bm25');
    expect(state.searchParams.rerankModel).toBe('Qwen/Qwen3-Reranker-4B');
  });

  it('should maintain reranking state when changing other parameters', () => {
    const { updateSearchParams } = useSearchStore.getState();

    // Set reranking parameters
    updateSearchParams({
      useReranker: true,
      rerankModel: 'Qwen/Qwen3-Reranker-0.6B',
    });

    // Change other parameters
    updateSearchParams({
      query: 'new query',
      topK: 20,
      selectedCollections: ['collection-1', 'collection-2'],
    });

    const state = useSearchStore.getState();
    // Reranking parameters should remain unchanged
    expect(state.searchParams.useReranker).toBe(true);
    expect(state.searchParams.rerankModel).toBe('Qwen/Qwen3-Reranker-0.6B');
    // Other parameters should be updated
    expect(state.searchParams.query).toBe('new query');
    expect(state.searchParams.topK).toBe(20);
    expect(state.searchParams.selectedCollections).toEqual(['collection-1', 'collection-2']);
  });

  it('should handle search results with reranked scores', () => {
    const { setResults, setRerankingMetrics } = useSearchStore.getState();

    const resultsWithReranking = [
      {
        doc_id: 'doc_1',
        chunk_id: 'chunk_1',
        score: 0.95, // Reranked score
        content: 'Highly relevant after reranking',
        file_path: '/doc1.txt',
        file_name: 'doc1.txt',
        chunk_index: 0,
        total_chunks: 5,
        collection_id: 'collection-1',
        collection_name: 'Collection 1',
      },
      {
        doc_id: 'doc_2',
        chunk_id: 'chunk_2',
        score: 0.88, // Reranked score
        content: 'Less relevant after reranking',
        file_path: '/doc2.txt',
        file_name: 'doc2.txt',
        chunk_index: 1,
        total_chunks: 3,
        collection_id: 'collection-2',
        collection_name: 'Collection 2',
      },
    ];

    setResults(resultsWithReranking);
    setRerankingMetrics({
      rerankingUsed: true,
      rerankerModel: 'Qwen/Qwen3-Reranker-0.6B',
      rerankingTimeMs: 120,
    });

    const state = useSearchStore.getState();
    expect(state.results).toEqual(resultsWithReranking);
    expect(state.rerankingMetrics?.rerankingUsed).toBe(true);
    expect(state.rerankingMetrics?.rerankingTimeMs).toBe(120);
  });

  it('should reset reranking to default when disabling', () => {
    const { updateSearchParams } = useSearchStore.getState();

    // Enable and configure reranking
    updateSearchParams({
      useReranker: true,
      rerankModel: 'Qwen/Qwen3-Reranker-8B',
      rerankQuantization: 'int8',
    });

    // Disable reranking
    updateSearchParams({ useReranker: false });

    const state = useSearchStore.getState();
    expect(state.searchParams.useReranker).toBe(false);
    // Model and quantization should still be preserved for when user re-enables
    expect(state.searchParams.rerankModel).toBe('Qwen/Qwen3-Reranker-8B');
    expect(state.searchParams.rerankQuantization).toBe('int8');
  });
});