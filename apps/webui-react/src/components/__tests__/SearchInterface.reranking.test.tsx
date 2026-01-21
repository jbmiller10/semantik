import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import SearchInterface from '../SearchInterface';
import { useSearchStore } from '../../stores/searchStore';
import { useUIStore } from '../../stores/uiStore';
import { mockSearchError, mockSearchSuccess } from '../../tests/mocks/test-utils';

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

// Mock usePreferences to avoid interference with search store state
vi.mock('../../hooks/usePreferences', () => ({
  usePreferences: () => ({
    data: null, // Return null so preferences don't override test state
    isLoading: false,
  }),
  useUpdatePreferences: () => ({
    mutate: vi.fn(),
    isPending: false,
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
        hybridMode: 'weighted',
        keywordMode: 'any',
      },
      results: [],
      loading: false,
      error: null,
      rerankingMetrics: null,
      rerankingAvailable: true,
      rerankingModelsLoading: false,
    });

    useUIStore.setState({
      toasts: [],
    });
  });

  it.skip('should pass reranking parameters correctly when enabled', async () => {
    const mockSearchResponse = {
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
    };

    mockSearchSuccess(mockSearchResponse);

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

    // Wait for reranking options to appear and select a specific reranker model
    await waitFor(() => {
      // Use the actual id of the select element
      const modelSelect = document.getElementById('reranker-model');
      expect(modelSelect).toBeInTheDocument();
    });

    const modelSelect = document.getElementById('reranker-model') as HTMLSelectElement;
    fireEvent.change(modelSelect, { target: { value: 'Qwen/Qwen3-Reranker-0.6B' } });

    // Submit search
    const searchButton = screen.getByText('Search');
    fireEvent.click(searchButton);

    // Wait for search to complete (loading indicator disappears)
    await waitFor(() => {
      expect(screen.queryByText('Searching...')).not.toBeInTheDocument();
    }, { timeout: 5000 });

    // Verify search was performed and results section appears
    await waitFor(() => {
      // The SearchResults component should render something when results are available
      const resultsContainer = screen.getByText('Test result with reranking');
      expect(resultsContainer).toBeInTheDocument();
    });
  });

  it.skip('should disable reranking when checkbox is unchecked', async () => {
    const mockSearchResponse = {
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
    };

    mockSearchSuccess(mockSearchResponse);

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

    // Wait for search to complete
    await waitFor(() => {
      expect(screen.queryByText('Searching...')).not.toBeInTheDocument();
    }, { timeout: 5000 });

    // Verify search results appear
    await waitFor(() => {
      expect(screen.getByText('Test result without reranking')).toBeInTheDocument();
    });
  });

  it('should show reranking options only when enabled', async () => {
    render(<SearchInterface />, { wrapper: createWrapper() });

    // Enable reranking by clicking the checkbox input (now outside Advanced Options)
    const rerankingCheckbox = screen.getByRole('checkbox', { name: /cross-encoder reranking/i });
    fireEvent.click(rerankingCheckbox);

    // Expand Advanced Options to see model/quantization dropdowns
    const advancedButton = screen.getByText('Advanced Options');
    fireEvent.click(advancedButton);

    // Now reranking options should be visible
    await waitFor(() => {
      // Look for the select elements by their labels
      const modelSelect = screen.getByLabelText(/Reranker Model/i);
      const quantizationSelect = screen.getByLabelText(/Quantization/i);

      expect(modelSelect).toBeInTheDocument();
      expect(quantizationSelect).toBeInTheDocument();
      expect(screen.getByText(/Reranking uses a more sophisticated model/)).toBeInTheDocument();
    });
  });

  it('should handle insufficient memory error for reranking', async () => {
    // Mock insufficient memory error
    mockSearchError(507, {
      error: 'insufficient_memory',
      message: 'Insufficient GPU memory for reranking',
      suggestion: 'Try using a smaller model or different quantization',
    });

    render(<SearchInterface />, { wrapper: createWrapper() });

    // Set up search with reranking
    const queryInput = screen.getByPlaceholderText('Enter your search query...');
    fireEvent.change(queryInput, { target: { value: 'test query' } });

    // Select a collection
    const collectionSelect = screen.getByText('Select collections...');
    fireEvent.click(collectionSelect);

    await waitFor(() => {
      const option = screen.getByText('Test Collection 1');
      fireEvent.click(option);
    });

    // Enable reranking with large model
    const rerankingCheckbox = screen.getByRole('checkbox', { name: /cross-encoder reranking/i });
    fireEvent.click(rerankingCheckbox);

    // Expand Advanced Options to access model selection
    const advancedButton = screen.getByText('Advanced Options');
    fireEvent.click(advancedButton);

    await waitFor(() => {
      const modelSelect = screen.getByLabelText(/Reranker Model/i);
      expect(modelSelect).toBeInTheDocument();
      fireEvent.change(modelSelect, { target: { value: 'Qwen/Qwen3-Reranker-8B' } });
    });

    // Submit search
    const searchButton = screen.getByText('Search');
    fireEvent.click(searchButton);

    // Verify error is displayed
    await waitFor(() => {
      const state = useSearchStore.getState();
      // SearchInterface sets error to 'GPU_MEMORY_ERROR' for insufficient memory
      expect(state.error).toBe('GPU_MEMORY_ERROR');

      // Check that the detailed error info was stored in the search store
      expect(state.gpuMemoryError).toBeDefined();
      expect(state.gpuMemoryError?.message).toContain('Insufficient GPU memory for reranking');
      expect(state.gpuMemoryError?.suggestion).toContain('Try using a smaller model or different quantization');
    });

    // Verify toast notification
    const uiState = useUIStore.getState();
    expect(uiState.toasts).toHaveLength(1);
    expect(uiState.toasts[0].type).toBe('error');
    expect(uiState.toasts[0].message).toContain('Insufficient GPU memory');
  });

  it.skip('should work with hybrid search and reranking together', async () => {
    const mockSearchResponse = {
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
    };

    mockSearchSuccess(mockSearchResponse);

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

    // Wait for search to complete
    await waitFor(() => {
      expect(screen.queryByText('Searching...')).not.toBeInTheDocument();
    }, { timeout: 5000 });

    // Verify search results appear
    await waitFor(() => {
      expect(screen.getByText('Hybrid search with reranking result')).toBeInTheDocument();
    });
  });

  it('should update search params when reranking configuration changes', async () => {
    render(<SearchInterface />, { wrapper: createWrapper() });

    // Enable reranking - click the actual checkbox input, not the label text
    const rerankingCheckbox = screen.getByRole('checkbox', { name: /cross-encoder reranking/i });
    fireEvent.click(rerankingCheckbox);

    // Verify search params are updated
    expect(useSearchStore.getState().searchParams.useReranker).toBe(true);

    // Expand Advanced Options to access model/quantization dropdowns
    const advancedButton = screen.getByText('Advanced Options');
    fireEvent.click(advancedButton);

    // Change model
    await waitFor(() => {
      const modelSelect = document.getElementById('reranker-model') as HTMLSelectElement;
      expect(modelSelect).toBeTruthy();
      fireEvent.change(modelSelect, { target: { value: 'Qwen/Qwen3-Reranker-4B' } });
    });

    // Verify model is updated in search params
    expect(useSearchStore.getState().searchParams.rerankModel).toBe('Qwen/Qwen3-Reranker-4B');

    // Change quantization
    const quantizationSelect = document.getElementById('reranker-quantization') as HTMLSelectElement;
    fireEvent.change(quantizationSelect, { target: { value: 'int8' } });

    // Verify quantization is updated
    expect(useSearchStore.getState().searchParams.rerankQuantization).toBe('int8');

    // Disable reranking
    fireEvent.click(rerankingCheckbox);
    expect(useSearchStore.getState().searchParams.useReranker).toBe(false);
  });
});
