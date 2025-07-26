import { create } from 'zustand';
import { 
  ValidationError, 
  validateSearchParams, 
  sanitizeQuery,
  clampValue,
  DEFAULT_VALIDATION_RULES 
} from '../utils/searchValidation';

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
  validationErrors: ValidationError[];
  rerankingAvailable: boolean;
  rerankingModelsLoading: boolean;
  setResults: (results: SearchResult[]) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  updateSearchParams: (params: Partial<SearchParams>) => void;
  setCollections: (collections: string[]) => void;
  setFailedCollections: (failedCollections: FailedCollection[]) => void;
  setPartialFailure: (partialFailure: boolean) => void;
  clearResults: () => void;
  setRerankingMetrics: (metrics: SearchState['rerankingMetrics']) => void;
  validateAndUpdateSearchParams: (params: Partial<SearchParams>) => void;
  clearValidationErrors: () => void;
  hasValidationErrors: () => boolean;
  getValidationError: (field: string) => string | undefined;
  setRerankingAvailable: (available: boolean) => void;
  setRerankingModelsLoading: (loading: boolean) => void;
}

export const useSearchStore = create<SearchState>((set, get) => ({
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
    hybridAlpha: 0.7,
    hybridMode: 'reciprocal_rank',
    keywordMode: 'bm25',
  },
  collections: [],
  failedCollections: [],
  partialFailure: false,
  rerankingMetrics: null,
  validationErrors: [],
  rerankingAvailable: true,
  rerankingModelsLoading: false,
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
  
  validateAndUpdateSearchParams: (params) => {
    const currentParams = get().searchParams;
    const updatedParams = { ...currentParams };
    
    // Sanitize and validate query if provided
    if (params.query !== undefined) {
      updatedParams.query = sanitizeQuery(params.query);
    }
    
    // Clamp numeric values to valid ranges
    if (params.topK !== undefined) {
      updatedParams.topK = clampValue(
        params.topK, 
        DEFAULT_VALIDATION_RULES.topK.min, 
        DEFAULT_VALIDATION_RULES.topK.max
      );
    }
    
    if (params.scoreThreshold !== undefined) {
      updatedParams.scoreThreshold = clampValue(
        params.scoreThreshold,
        DEFAULT_VALIDATION_RULES.scoreThreshold.min,
        DEFAULT_VALIDATION_RULES.scoreThreshold.max
      );
    }
    
    if (params.hybridAlpha !== undefined) {
      updatedParams.hybridAlpha = clampValue(
        params.hybridAlpha,
        DEFAULT_VALIDATION_RULES.hybridAlpha.min,
        DEFAULT_VALIDATION_RULES.hybridAlpha.max
      );
    }
    
    // Update other params without validation
    if (params.selectedCollections !== undefined) {
      updatedParams.selectedCollections = params.selectedCollections;
    }
    if (params.searchType !== undefined) {
      updatedParams.searchType = params.searchType;
    }
    if (params.rerankModel !== undefined) {
      updatedParams.rerankModel = params.rerankModel;
    }
    if (params.rerankQuantization !== undefined) {
      updatedParams.rerankQuantization = params.rerankQuantization;
    }
    if (params.useReranker !== undefined) {
      updatedParams.useReranker = params.useReranker;
    }
    if (params.hybridMode !== undefined) {
      updatedParams.hybridMode = params.hybridMode;
    }
    if (params.keywordMode !== undefined) {
      updatedParams.keywordMode = params.keywordMode;
    }
    
    // Validate all params
    const errors = validateSearchParams({
      query: updatedParams.query,
      topK: updatedParams.topK,
      scoreThreshold: updatedParams.scoreThreshold,
      hybridAlpha: updatedParams.hybridAlpha,
      selectedCollections: updatedParams.selectedCollections,
      searchType: updatedParams.searchType,
    });
    
    set({
      searchParams: updatedParams,
      validationErrors: errors,
    });
  },
  
  clearValidationErrors: () => set({ validationErrors: [] }),
  
  hasValidationErrors: () => get().validationErrors.length > 0,
  
  getValidationError: (field) => {
    const errors = get().validationErrors;
    const error = errors.find(e => e.field === field);
    return error?.message;
  },
  
  setRerankingAvailable: (available) => set({ rerankingAvailable: available }),
  
  setRerankingModelsLoading: (loading) => set({ rerankingModelsLoading: loading }),
}));