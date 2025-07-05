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
  job_id?: string;
}

export interface SearchParams {
  query: string;
  collection: string;
  topK: number;
  scoreThreshold: number;
  searchType: 'vector' | 'hybrid';
  rerankModel?: string;
  useReranker: boolean;
  rerankTopK: number;
  hybridAlpha?: number;
  hybridMode?: 'rerank' | 'filter';
  keywordMode?: 'any' | 'all';
}

interface SearchState {
  results: SearchResult[];
  loading: boolean;
  error: string | null;
  searchParams: SearchParams;
  collections: string[];
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
  clearResults: () => void;
  setRerankingMetrics: (metrics: SearchState['rerankingMetrics']) => void;
}

export const useSearchStore = create<SearchState>((set) => ({
  results: [],
  loading: false,
  error: null,
  searchParams: {
    query: '',
    collection: '',
    topK: 10,
    scoreThreshold: 0.0,
    searchType: 'vector',
    useReranker: false,
    rerankTopK: 50,
    hybridMode: 'rerank',
    keywordMode: 'any',
  },
  collections: [],
  rerankingMetrics: null,
  setResults: (results) => set({ results }),
  setLoading: (loading) => set({ loading }),
  setError: (error) => set({ error }),
  updateSearchParams: (params) =>
    set((state) => ({
      searchParams: { ...state.searchParams, ...params },
    })),
  setCollections: (collections) => set({ collections }),
  clearResults: () => set({ results: [], error: null, rerankingMetrics: null }),
  setRerankingMetrics: (metrics) => set({ rerankingMetrics: metrics }),
}));