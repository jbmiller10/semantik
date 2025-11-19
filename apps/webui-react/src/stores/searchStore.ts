import { create } from 'zustand';
import type { ValidationError } from '../utils/searchValidation';
import {
  validateSearchParams,

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
  original_score?: number;
  reranked_score?: number;
  embedding_model?: string;
  metadata?: Record<string, any>;
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
  hybridMode?: 'filter' | 'rerank';
  keywordMode?: 'any' | 'all';
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
    original_count?: number;
    reranked_count?: number;
  } | null;
  gpuMemoryError: {
    message: string;
    suggestion: string;
    currentModel: string;
  } | null;
  validationErrors: ValidationError[];
  touched: Record<string, boolean>;
  abortController: AbortController | null;
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
  setGpuMemoryError: (error: SearchState['gpuMemoryError']) => void;
  validateAndUpdateSearchParams: (params: Partial<SearchParams>) => void;
  setFieldTouched: (field: string, isTouched?: boolean) => void;
  clearValidationErrors: () => void;
  hasValidationErrors: () => boolean;
  getValidationError: (field: string) => string | undefined;
  setRerankingAvailable: (available: boolean) => void;
  setRerankingModelsLoading: (loading: boolean) => void;
  setAbortController: (controller: AbortController | null) => void;
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
    hybridMode: 'rerank',
    keywordMode: 'any',
  },
  collections: [],
  failedCollections: [],
  partialFailure: false,
  rerankingMetrics: null,
  gpuMemoryError: null,
  validationErrors: [],
  touched: {},
  abortController: null,
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
  setGpuMemoryError: (gpuMemoryError) => set({ gpuMemoryError }),

  validateAndUpdateSearchParams: (params) => {
    const currentParams = get().searchParams;
    const updatedParams = { ...currentParams };

    // Sanitize and validate query if provided
    if (params.query !== undefined) {
      updatedParams.query = params.query;
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

    // Validate all params but only return errors for touched fields or critical ones
    const allErrors = validateSearchParams({
      query: updatedParams.query,
      topK: updatedParams.topK,
      scoreThreshold: updatedParams.scoreThreshold,
      hybridAlpha: updatedParams.hybridAlpha,
      selectedCollections: updatedParams.selectedCollections,
      searchType: updatedParams.searchType,
    });

    // Filter errors based on touched state
    // Note: We always show collection errors if they are empty and user tries to search
    const touched = get().touched;
    const visibleErrors = allErrors.filter(error => {
      // Always show collection errors if we have attempted to interact with them
      if (error.field === 'collections' && touched.collections) return true;
      // Show query errors if touched
      if (error.field === 'query' && touched.query) return true;
      // Show other errors if touched
      return touched[error.field];
    });

    set({
      searchParams: updatedParams,
      validationErrors: visibleErrors,
    });
  },

  setFieldTouched: (field, isTouched = true) => {
    set((state) => {
      const newTouched = { ...state.touched, [field]: isTouched };

      // Re-run validation with new touched state
      const currentParams = state.searchParams;
      const allErrors = validateSearchParams({
        query: currentParams.query,
        topK: currentParams.topK,
        scoreThreshold: currentParams.scoreThreshold,
        hybridAlpha: currentParams.hybridAlpha,
        selectedCollections: currentParams.selectedCollections,
        searchType: currentParams.searchType,
      });

      const visibleErrors = allErrors.filter(error => {
        if (error.field === 'collections' && newTouched.collections) return true;
        if (error.field === 'query' && newTouched.query) return true;
        return newTouched[error.field];
      });

      return {
        touched: newTouched,
        validationErrors: visibleErrors
      };
    });
  },

  clearValidationErrors: () => set({ validationErrors: [], touched: {} }),

  hasValidationErrors: () => get().validationErrors.length > 0,

  getValidationError: (field) => {
    const errors = get().validationErrors;
    const error = errors.find(e => e.field === field);
    return error?.message;
  },

  setRerankingAvailable: (available) => set({ rerankingAvailable: available }),

  setRerankingModelsLoading: (loading) => set({ rerankingModelsLoading: loading }),

  setAbortController: (controller) => set({ abortController: controller }),
}));
