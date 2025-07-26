import { create } from 'zustand';

export interface SearchResult {
  doc_id: string;
  chunk_id: string;
  score: number;
  content: string;
  file_path: string;
  file_name: string;
  chunk_index: number;
  total_chunks: number;
  collection_id?: string;
  collection_name?: string;
}

export interface SearchParams {
  query: string;
  selectedCollections: string[];
  topK: number;
  scoreThreshold: number;
  searchType: 'semantic' | 'question' | 'code' | 'hybrid';
  rerankModel?: string;
  rerankQuantization?: string;
  useReranker: boolean;
  hybridAlpha?: number;
  hybridMode?: 'reciprocal_rank' | 'relative_score';
  keywordMode?: 'bm25';
}

interface FailedCollection {
  collection_id: string;
  collection_name: string;
  error_message: string;
}

interface SearchState {
  results: SearchResult[];
  loading: boolean;
  error: string | null;
  searchParams: SearchParams;
  collections: string[];
  failedCollections: FailedCollection[];
  partialFailure: boolean;
  rerankingMetrics: {
    rerankingUsed: boolean;
    rerankerModel?: string;
    rerankingTimeMs?: number;
  } | null;
  setResults: (results: SearchResult[]) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  updateSearchParams: (params: Partial<SearchParams>) => void;
  setCollections: (collections: string[]) => void;
  setFailedCollections: (failedCollections: FailedCollection[]) => void;
  setPartialFailure: (partialFailure: boolean) => void;
  clearResults: () => void;
  setRerankingMetrics: (metrics: SearchState['rerankingMetrics']) => void;
}

export const useSearchStore = create<SearchState>((set) => ({
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
    hybridMode: 'reciprocal_rank',
    keywordMode: 'bm25',
  },
  collections: [],
  failedCollections: [],
  partialFailure: false,
  rerankingMetrics: null,
  setResults: (results) => set({ results }),
  setLoading: (loading) => set({ loading }),
  setError: (error) => set({ error }),
  updateSearchParams: (params) =>
    set((state) => ({
      searchParams: { ...state.searchParams, ...params },
    })),
  setCollections: (collections) => set({ collections }),
  setFailedCollections: (failedCollections) => set({ failedCollections }),
  setPartialFailure: (partialFailure) => set({ partialFailure }),
  clearResults: () => set({ results: [], error: null, rerankingMetrics: null, failedCollections: [], partialFailure: false }),
  setRerankingMetrics: (metrics) => set({ rerankingMetrics: metrics }),
}));