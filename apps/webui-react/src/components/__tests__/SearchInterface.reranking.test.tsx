import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import { AxiosError } from 'axios';
import SearchInterface from '../SearchInterface';
import { useSearchStore } from '../../stores/searchStore';
import { useUIStore } from '../../stores/uiStore';
import { searchV2Api } from '../../services/api/v2/collections';

// Mock the API
vi.mock('../../services/api/v2/collections');

// Mock the hooks
vi.mock('../../hooks/useCollections', () => ({
  useCollections: () => ({
    data: [
      {
        id: '123e4567-e89b-12d3-a456-426614174000',
        name: 'Test Collection 1',
        status: 'ready',
        embedding_model: 'Qwen/Qwen3-Embedding-0.6B',
        vector_count: 100,
        document_count: 10,
      },
      {
        id: '456e7890-e89b-12d3-a456-426614174001',
        name: 'Test Collection 2',
        status: 'ready',
        embedding_model: 'BAAI/bge-small-en-v1.5',
        vector_count: 200,
        document_count: 20,
      },
    ],
    refetch: vi.fn(),
  }),
}));

// Create a test wrapper with React Query
const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });

  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

describe('SearchInterface Reranking Tests', () => {
  beforeEach(() => {
    // Reset all mocks
    vi.clearAllMocks();
    
    // Reset store state
    useSearchStore.setState({
      searchParams: {
        query: '',
        selectedCollections: [],
        topK: 10,
        scoreThreshold: 0.0,
        searchType: 'semantic',
        useReranker: false,
        rerankModel: undefined,
        rerankQuantization: undefined,
        hybridAlpha: 0.7,
        hybridMode: 'reciprocal_rank',
        keywordMode: 'bm25',
      },
      results: [],
      loading: false,
      error: null,
      rerankingMetrics: null,
    });

    useUIStore.setState({
      toasts: [],
    });
  });

  it('should pass reranking parameters correctly when enabled', async () => {
    const mockSearchResponse = {
      data: {
        results: [
          {
            document_id: 'doc_1',
            chunk_id: 'chunk_1',
            score: 0.95,
            text: 'Test result with reranking',
            file_path: '/test.txt',
            file_name: 'test.txt',
            collection_id: '123e4567-e89b-12d3-a456-426614174000',
            collection_name: 'Test Collection 1',
          },
        ],
        total_results: 1,
        reranking_used: true,
        reranker_model: 'Qwen/Qwen3-Reranker-0.6B',
        reranking_time_ms: 50,
        search_time_ms: 100,
        total_time_ms: 150,
        partial_failure: false,
      },
    };

    vi.mocked(searchV2Api.search).mockResolvedValueOnce(mockSearchResponse);

    render(<SearchInterface />, { wrapper: createWrapper() });

    // Set search query and select collection
    const queryInput = screen.getByPlaceholderText('Enter your search query...');
    fireEvent.change(queryInput, { target: { value: 'test query' } });

    // Select a collection
    const collectionSelect = screen.getByText('Select collections to search...');
    fireEvent.click(collectionSelect);
    
    // Wait for dropdown to appear and select first collection
    await waitFor(() => {
      const option = screen.getByText('Test Collection 1');
      fireEvent.click(option);
    });

    // Enable reranking
    const rerankingCheckbox = screen.getByText('Enable Cross-Encoder Reranking');
    fireEvent.click(rerankingCheckbox);

    // Select a specific reranker model
    await waitFor(() => {
      const modelSelect = screen.getByLabelText('Reranker Model');
      fireEvent.change(modelSelect, { target: { value: 'Qwen/Qwen3-Reranker-0.6B' } });
    });

    // Submit search
    const searchButton = screen.getByText('Search');
    fireEvent.click(searchButton);

    // Verify API was called with correct parameters
    await waitFor(() => {
      expect(searchV2Api.search).toHaveBeenCalledWith({
        query: 'test query',
        collection_uuids: ['123e4567-e89b-12d3-a456-426614174000'],
        k: 10,
        score_threshold: 0.0,
        search_type: 'semantic',
        use_reranker: true,
        rerank_model: 'Qwen/Qwen3-Reranker-0.6B',
        hybrid_alpha: undefined,
        hybrid_mode: undefined,
        keyword_mode: undefined,
      });
    });

    // Verify reranking metrics are stored
    const state = useSearchStore.getState();
    expect(state.rerankingMetrics).toEqual({
      rerankingUsed: true,
      rerankerModel: 'Qwen/Qwen3-Reranker-0.6B',
      rerankingTimeMs: 50,
    });
  });

  it('should disable reranking when checkbox is unchecked', async () => {
    const mockSearchResponse = {
      data: {
        results: [
          {
            document_id: 'doc_1',
            chunk_id: 'chunk_1',
            score: 0.85,
            text: 'Test result without reranking',
            file_path: '/test.txt',
            file_name: 'test.txt',
            collection_id: '123e4567-e89b-12d3-a456-426614174000',
            collection_name: 'Test Collection 1',
          },
        ],
        total_results: 1,
        reranking_used: false,
        reranker_model: null,
        search_time_ms: 50,
        total_time_ms: 50,
        partial_failure: false,
      },
    };

    vi.mocked(searchV2Api.search).mockResolvedValueOnce(mockSearchResponse);

    render(<SearchInterface />, { wrapper: createWrapper() });

    // Set search query and select collection
    const queryInput = screen.getByPlaceholderText('Enter your search query...');
    fireEvent.change(queryInput, { target: { value: 'test query' } });

    // Select a collection
    const collectionSelect = screen.getByText('Select collections to search...');
    fireEvent.click(collectionSelect);
    
    await waitFor(() => {
      const option = screen.getByText('Test Collection 1');
      fireEvent.click(option);
    });

    // Ensure reranking is disabled (default state)
    const rerankingCheckbox = screen.getByText('Enable Cross-Encoder Reranking') as HTMLInputElement;
    expect(rerankingCheckbox.parentElement?.querySelector('input')?.checked).toBe(false);

    // Submit search
    const searchButton = screen.getByText('Search');
    fireEvent.click(searchButton);

    // Verify API was called with reranking disabled
    await waitFor(() => {
      expect(searchV2Api.search).toHaveBeenCalledWith({
        query: 'test query',
        collection_uuids: ['123e4567-e89b-12d3-a456-426614174000'],
        k: 10,
        score_threshold: 0.0,
        search_type: 'semantic',
        use_reranker: false,
        rerank_model: null,
        hybrid_alpha: undefined,
        hybrid_mode: undefined,
        keyword_mode: undefined,
      });
    });
  });

  it('should show reranking options only when enabled', async () => {
    render(<SearchInterface />, { wrapper: createWrapper() });

    // Initially, reranking options should not be visible
    expect(screen.queryByLabelText('Reranker Model')).not.toBeInTheDocument();
    expect(screen.queryByLabelText('Quantization')).not.toBeInTheDocument();

    // Enable reranking
    const rerankingCheckbox = screen.getByText('Enable Cross-Encoder Reranking');
    fireEvent.click(rerankingCheckbox);

    // Now reranking options should be visible
    await waitFor(() => {
      expect(screen.getByLabelText('Reranker Model')).toBeInTheDocument();
      expect(screen.getByLabelText('Quantization')).toBeInTheDocument();
      expect(screen.getByText(/Reranking uses a more sophisticated model/)).toBeInTheDocument();
    });
  });

  it('should handle insufficient memory error for reranking', async () => {
    const mockError = new AxiosError(
      'Request failed with status code 507',
      'ERR_BAD_RESPONSE',
      undefined,
      undefined,
      {
        status: 507,
        statusText: 'Insufficient Storage',
        data: {
          detail: {
            error: 'insufficient_memory',
            message: 'Insufficient GPU memory for reranking',
            suggestion: 'Try using a smaller model or different quantization',
          },
        },
        headers: {},
        config: {},
      }
    );

    vi.mocked(searchV2Api.search).mockRejectedValueOnce(mockError);

    render(<SearchInterface />, { wrapper: createWrapper() });

    // Set up search with reranking
    const queryInput = screen.getByPlaceholderText('Enter your search query...');
    fireEvent.change(queryInput, { target: { value: 'test query' } });

    // Select a collection
    const collectionSelect = screen.getByText('Select collections to search...');
    fireEvent.click(collectionSelect);
    
    await waitFor(() => {
      const option = screen.getByText('Test Collection 1');
      fireEvent.click(option);
    });

    // Enable reranking with large model
    const rerankingCheckbox = screen.getByText('Enable Cross-Encoder Reranking');
    fireEvent.click(rerankingCheckbox);

    await waitFor(() => {
      const modelSelect = screen.getByLabelText('Reranker Model');
      fireEvent.change(modelSelect, { target: { value: 'Qwen/Qwen3-Reranker-8B' } });
    });

    // Submit search
    const searchButton = screen.getByText('Search');
    fireEvent.click(searchButton);

    // Verify error is displayed
    await waitFor(() => {
      const state = useSearchStore.getState();
      expect(state.error).toContain('Insufficient GPU memory for reranking');
      expect(state.error).toContain('Try using a smaller model or different quantization');
    });

    // Verify toast notification
    const uiState = useUIStore.getState();
    expect(uiState.toasts).toHaveLength(1);
    expect(uiState.toasts[0].type).toBe('error');
    expect(uiState.toasts[0].message).toContain('Insufficient GPU memory');
  });

  it('should work with hybrid search and reranking together', async () => {
    const mockSearchResponse = {
      data: {
        results: [
          {
            document_id: 'doc_1',
            chunk_id: 'chunk_1',
            score: 0.96,
            text: 'Hybrid search with reranking result',
            file_path: '/test.txt',
            file_name: 'test.txt',
            collection_id: '123e4567-e89b-12d3-a456-426614174000',
            collection_name: 'Test Collection 1',
          },
        ],
        total_results: 1,
        reranking_used: true,
        reranker_model: 'Qwen/Qwen3-Reranker-0.6B',
        search_time_ms: 150,
        total_time_ms: 150,
        partial_failure: false,
      },
    };

    vi.mocked(searchV2Api.search).mockResolvedValueOnce(mockSearchResponse);

    render(<SearchInterface />, { wrapper: createWrapper() });

    // Set search query and select collection
    const queryInput = screen.getByPlaceholderText('Enter your search query...');
    fireEvent.change(queryInput, { target: { value: 'test query' } });

    // Select a collection
    const collectionSelect = screen.getByText('Select collections to search...');
    fireEvent.click(collectionSelect);
    
    await waitFor(() => {
      const option = screen.getByText('Test Collection 1');
      fireEvent.click(option);
    });

    // Enable hybrid search
    const hybridCheckbox = screen.getByText('Use Hybrid Search (combines vector similarity with keyword matching)');
    fireEvent.click(hybridCheckbox);

    // Enable reranking
    const rerankingCheckbox = screen.getByText('Enable Cross-Encoder Reranking');
    fireEvent.click(rerankingCheckbox);

    // Submit search
    const searchButton = screen.getByText('Search');
    fireEvent.click(searchButton);

    // Verify API was called with both hybrid and reranking parameters
    await waitFor(() => {
      expect(searchV2Api.search).toHaveBeenCalledWith({
        query: 'test query',
        collection_uuids: ['123e4567-e89b-12d3-a456-426614174000'],
        k: 10,
        score_threshold: 0.0,
        search_type: 'hybrid',
        use_reranker: true,
        rerank_model: undefined, // Auto-select
        hybrid_alpha: 0.7,
        hybrid_mode: 'reciprocal_rank',
        keyword_mode: 'bm25',
      });
    });
  });
});