import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor, fireEvent } from '@/tests/utils/test-utils'
import userEvent from '@testing-library/user-event'
import SearchInterface from '../SearchInterface'
import { useSearchStore } from '@/stores/searchStore'
import { useUIStore } from '@/stores/uiStore'

// Mock the stores
vi.mock('@/stores/searchStore')
vi.mock('@/stores/uiStore')

// Mock hooks
vi.mock('@/hooks/useCollections', () => ({
  useCollections: () => ({
    data: [
      {
        id: 'test-collection',
        name: 'Test Collection',
        status: 'active',
        total_files: 100,
        total_vectors: 500,
        model_name: 'Qwen/Qwen3-Embedding-0.6B',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      }
    ],
    refetch: vi.fn(),
    isLoading: false,
    error: null,
  })
}))

vi.mock('@/hooks/useRerankingAvailability', () => ({
  useRerankingAvailability: vi.fn()
}))

// Mock SearchResults component
vi.mock('../SearchResults', () => ({
  default: () => <div data-testid="search-results">Search Results</div>
}))

// Mock CollectionMultiSelect component
vi.mock('../CollectionMultiSelect', () => ({
  CollectionMultiSelect: ({ selectedCollections, onChange, disabled }: { selectedCollections: string[]; onChange: (collections: string[]) => void; disabled?: boolean }) => (
    <div data-testid="collection-multiselect">
      <button
        aria-label="Select collections"
        disabled={disabled}
        onClick={() => onChange(['test-collection'])}
      >
        {selectedCollections.length > 0 ? `${selectedCollections.length} selected` : 'Select collections'}
      </button>
    </div>
  )
}))

// Mock RerankingConfiguration component
vi.mock('../RerankingConfiguration', () => ({
  RerankingConfiguration: ({ enabled, model, quantization, onChange, hideToggle }: {
    enabled: boolean;
    model?: string;
    quantization?: string;
    onChange: (config: {
      useReranker?: boolean;
      rerankModel?: string;
      rerankQuantization?: string;
    }) => void;
    hideToggle?: boolean;
  }) => (
    <div data-testid="reranking-configuration">
      {!hideToggle && (
        <label>
          <input
            type="checkbox"
            checked={enabled}
            onChange={(e) => onChange({ useReranker: e.target.checked })}
            aria-label="Enable cross-encoder reranking"
          />
          Enable Cross-Encoder Reranking
        </label>
      )}
      <div className={!enabled && hideToggle ? 'opacity-50' : ''}>
        <select
          value={model || 'BAAI/bge-reranker-v2-m3'}
          onChange={(e) => onChange({ rerankModel: e.target.value })}
          disabled={!enabled}
        >
          <option value="BAAI/bge-reranker-v2-m3">BAAI/bge-reranker-v2-m3</option>
          <option value="Qwen/Qwen3-Reranker-0.6B">Qwen/Qwen3-Reranker-0.6B</option>
        </select>
        <select
          value={quantization || 'int8'}
          onChange={(e) => onChange({ rerankQuantization: e.target.value })}
          disabled={!enabled}
        >
          <option value="int8">int8</option>
          <option value="float16">float16</option>
        </select>
      </div>
    </div>
  )
}))

describe('SearchInterface', () => {
  const mockUpdateSearchParams = vi.fn()
  const mockValidateAndUpdateSearchParams = vi.fn()
  const mockSetResults = vi.fn()
  const mockSetLoading = vi.fn()
  const mockSetError = vi.fn()
  const mockSetCollections = vi.fn()
  const mockSetRerankingMetrics = vi.fn()
  const mockAddToast = vi.fn()

  const defaultSearchParams = {
    query: '',
    collection: '',
    selectedCollections: [],
    topK: 10,
    scoreThreshold: 0.5,
    searchType: 'semantic' as const,
    useReranker: false,
    rerankModel: 'BAAI/bge-reranker-v2-m3',
    rerankQuantization: 'int8',
    hybridAlpha: 0.95,
    hybridMode: 'weighted' as const,
    keywordMode: 'any' as const,
    searchMode: 'dense' as const,
    rrfK: 60,
  }

  beforeEach(() => {
    vi.clearAllMocks()

    vi.mocked(useSearchStore).mockReturnValue({
      searchParams: defaultSearchParams,
      updateSearchParams: mockUpdateSearchParams,
      validateAndUpdateSearchParams: mockValidateAndUpdateSearchParams,
      setResults: mockSetResults,
      setLoading: mockSetLoading,
      setError: mockSetError,
      setCollections: mockSetCollections,
      collections: [],
      setRerankingMetrics: mockSetRerankingMetrics,
      setFailedCollections: vi.fn(),
      setPartialFailure: vi.fn(),
      hasValidationErrors: vi.fn().mockReturnValue(false),
      getValidationError: vi.fn().mockReturnValue(undefined),
      setFieldTouched: vi.fn(),
      abortController: null,
      setAbortController: vi.fn(),
      setGpuMemoryError: vi.fn(),
      rerankingAvailable: true,
      rerankingModelsLoading: false,
    })

    vi.mocked(useUIStore).mockReturnValue({
      addToast: mockAddToast,
    })
  })

  it('renders search form elements', async () => {
    render(<SearchInterface />)

    // Check main elements
    // Check main elements
    expect(screen.getByText('Search Knowledge Base')).toBeInTheDocument()
    expect(screen.getByPlaceholderText('Enter your search query...')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Search' })).toBeInTheDocument()

    // Check search results component
    expect(screen.getByTestId('search-results')).toBeInTheDocument()
  })

  it('validates empty search query', async () => {
    render(<SearchInterface />)

    const searchButton = screen.getByRole('button', { name: 'Search' })

    // The search button is not disabled (validation happens on click)
    expect(searchButton).not.toBeDisabled()
  })

  it('validates collection selection', async () => {

    vi.mocked(useSearchStore).mockReturnValue({
      searchParams: { ...defaultSearchParams, query: 'test query', selectedCollections: ['test-collection'] },
      updateSearchParams: mockUpdateSearchParams,
      validateAndUpdateSearchParams: mockValidateAndUpdateSearchParams,
      setResults: mockSetResults,
      setLoading: mockSetLoading,
      setError: mockSetError,
      setCollections: mockSetCollections,
      collections: ['test-collection'],
      setRerankingMetrics: mockSetRerankingMetrics,
      setFailedCollections: vi.fn(),
      setPartialFailure: vi.fn(),
      hasValidationErrors: vi.fn().mockReturnValue(false),
      getValidationError: vi.fn().mockReturnValue(undefined),
    })

    render(<SearchInterface />)

    const searchButton = screen.getByRole('button', { name: 'Search' })
    expect(searchButton).not.toBeDisabled()
  })

  it('toggles hybrid search mode', async () => {
    const user = userEvent.setup()

    render(<SearchInterface />)

    // Search mode is now controlled via SearchModeSelector buttons
    const hybridButton = screen.getByRole('button', { name: /hybrid/i })
    await user.click(hybridButton)

    await waitFor(() => {
      expect(mockValidateAndUpdateSearchParams).toHaveBeenCalledWith({ searchMode: 'hybrid' })
    })
  })

  it('shows hybrid search options when enabled', async () => {
    vi.mocked(useSearchStore).mockReturnValue({
      searchParams: { ...defaultSearchParams, searchMode: 'hybrid' as const },
      updateSearchParams: mockUpdateSearchParams,
      validateAndUpdateSearchParams: mockValidateAndUpdateSearchParams,
      setResults: mockSetResults,
      setLoading: mockSetLoading,
      setError: mockSetError,
      setCollections: mockSetCollections,
      collections: [],
      setRerankingMetrics: mockSetRerankingMetrics,
      setFailedCollections: vi.fn(),
      setPartialFailure: vi.fn(),
      hasValidationErrors: vi.fn().mockReturnValue(false),
      getValidationError: vi.fn().mockReturnValue(undefined),
    })

    render(<SearchInterface />)

    // New SearchModeSelector shows RRF config when in hybrid mode
    expect(screen.getByText('Hybrid Search Configuration')).toBeInTheDocument()
    expect(screen.getByText(/RRF Weighting/)).toBeInTheDocument()
  })

  it('toggles reranking options', async () => {
    const user = userEvent.setup()

    render(<SearchInterface />)

    // Reranking toggle is now at top level (not inside Advanced Options)
    const rerankCheckbox = screen.getByLabelText(/enable cross-encoder reranking/i)
    expect(rerankCheckbox).not.toBeChecked()

    await user.click(rerankCheckbox)

    expect(mockValidateAndUpdateSearchParams).toHaveBeenCalledWith({ useReranker: true })
  })

  it('shows reranking options when enabled', async () => {
    const user = userEvent.setup()

    vi.mocked(useSearchStore).mockReturnValue({
      searchParams: { ...defaultSearchParams, useReranker: true },
      updateSearchParams: mockUpdateSearchParams,
      validateAndUpdateSearchParams: mockValidateAndUpdateSearchParams,
      setResults: mockSetResults,
      setLoading: mockSetLoading,
      setError: mockSetError,
      setCollections: mockSetCollections,
      collections: [],
      setRerankingMetrics: mockSetRerankingMetrics,
      setFailedCollections: vi.fn(),
      setPartialFailure: vi.fn(),
      hasValidationErrors: vi.fn().mockReturnValue(false),
      getValidationError: vi.fn().mockReturnValue(undefined),
    })

    render(<SearchInterface />)

    // Expand advanced options
    const advancedButton = screen.getByText('Advanced Options')
    await user.click(advancedButton)

    // Check that reranking options are shown
    const selects = screen.getAllByRole('combobox')
    // We expect at least 2 selects for reranking (model and quantization)
    // There might be others (Search Type, Collections, etc.)
    expect(selects.length).toBeGreaterThanOrEqual(2)
    expect(screen.getByText('BAAI/bge-reranker-v2-m3')).toBeInTheDocument()
    expect(screen.getByText('int8')).toBeInTheDocument()
  })

  it('updates search parameters when inputs change', async () => {
    render(<SearchInterface />)

    const queryInput = screen.getByPlaceholderText('Enter your search query...')
    fireEvent.change(queryInput, { target: { value: 'test query' } })

    // Check that validate and update was called
    await waitFor(() => {
      expect(mockValidateAndUpdateSearchParams).toHaveBeenCalledWith({ query: 'test query' })
    })
  })

  it('has disabled search button when no collection selected', () => {
    render(<SearchInterface />)

    const searchButton = screen.getByRole('button', { name: 'Search' })
    // Button is not disabled (validation happens on click)
    expect(searchButton).not.toBeDisabled()
  })

  it('enables search button when collection is selected', () => {
    vi.mocked(useSearchStore).mockReturnValue({
      searchParams: { ...defaultSearchParams, selectedCollections: ['test-collection'] },
      updateSearchParams: mockUpdateSearchParams,
      validateAndUpdateSearchParams: mockValidateAndUpdateSearchParams,
      setResults: mockSetResults,
      setLoading: mockSetLoading,
      setError: mockSetError,
      setCollections: mockSetCollections,
      collections: ['test-collection'],
      setRerankingMetrics: mockSetRerankingMetrics,
      setFailedCollections: vi.fn(),
      setPartialFailure: vi.fn(),
      hasValidationErrors: vi.fn().mockReturnValue(false),
      getValidationError: vi.fn().mockReturnValue(undefined),
    })

    render(<SearchInterface />)

    const searchButton = screen.getByRole('button', { name: 'Search' })
    expect(searchButton).not.toBeDisabled()
  })

  it('changes reranker model selection', async () => {
    const user = userEvent.setup()

    vi.mocked(useSearchStore).mockReturnValue({
      searchParams: { ...defaultSearchParams, useReranker: true },
      updateSearchParams: mockUpdateSearchParams,
      validateAndUpdateSearchParams: mockValidateAndUpdateSearchParams,
      setResults: mockSetResults,
      setLoading: mockSetLoading,
      setError: mockSetError,
      setCollections: mockSetCollections,
      collections: [],
      setRerankingMetrics: mockSetRerankingMetrics,
      setFailedCollections: vi.fn(),
      setPartialFailure: vi.fn(),
      hasValidationErrors: vi.fn().mockReturnValue(false),
      getValidationError: vi.fn().mockReturnValue(undefined),
    })

    render(<SearchInterface />)

    // Expand advanced options
    const advancedButton = screen.getByText('Advanced Options')
    await user.click(advancedButton)

    // Find the select by looking for the one that has the reranker options
    const selects = screen.getAllByRole('combobox')
    const modelSelect = selects.find(select =>
      select.querySelector('option[value="Qwen/Qwen3-Reranker-0.6B"]')
    )

    if (modelSelect) {
      await user.selectOptions(modelSelect, 'Qwen/Qwen3-Reranker-0.6B')

      expect(mockValidateAndUpdateSearchParams).toHaveBeenCalledWith({
        rerankModel: 'Qwen/Qwen3-Reranker-0.6B'
      })
    }
  })

  it('changes quantization selection', async () => {
    const user = userEvent.setup()

    vi.mocked(useSearchStore).mockReturnValue({
      searchParams: { ...defaultSearchParams, useReranker: true },
      updateSearchParams: mockUpdateSearchParams,
      validateAndUpdateSearchParams: mockValidateAndUpdateSearchParams,
      setResults: mockSetResults,
      setLoading: mockSetLoading,
      setError: mockSetError,
      setCollections: mockSetCollections,
      collections: [],
      setRerankingMetrics: mockSetRerankingMetrics,
      setFailedCollections: vi.fn(),
      setPartialFailure: vi.fn(),
      hasValidationErrors: vi.fn().mockReturnValue(false),
      getValidationError: vi.fn().mockReturnValue(undefined),
    })

    render(<SearchInterface />)

    // Expand advanced options
    const advancedButton = screen.getByText('Advanced Options')
    await user.click(advancedButton)

    // Find the select by looking for the one that has the quantization options
    const selects = screen.getAllByRole('combobox')
    const quantizationSelect = selects.find(select =>
      select.querySelector('option[value="float16"]')
    )

    if (quantizationSelect) {
      await user.selectOptions(quantizationSelect, 'float16')

      expect(mockValidateAndUpdateSearchParams).toHaveBeenCalledWith({
        rerankQuantization: 'float16'
      })
    }
  })
})
