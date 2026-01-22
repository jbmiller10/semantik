import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@/tests/utils/test-utils'
import userEvent from '@testing-library/user-event'
import SearchResults from '../SearchResults'
import { useSearchStore } from '@/stores/searchStore'
import { useUIStore } from '@/stores/uiStore'
import type { MockedFunction } from '@/tests/types/test-types'
import type { HyDEInfo } from '@/services/api/v2/types'
import type { SearchResult } from '@/stores/searchStore'

vi.mock('@/stores/searchStore', () => ({
  useSearchStore: vi.fn(),
}))

vi.mock('@/stores/uiStore', () => ({
  useUIStore: vi.fn(),
}))

describe('SearchResults', () => {
  const mockSetShowDocumentViewer = vi.fn()
  const mockOnSelectSmallerModel = vi.fn()
  const mockResults = [
    {
      chunk_id: 'chunk1',
      doc_id: 'doc1',
      collection_id: 'collection1',
      collection_name: 'Test Collection',
      file_path: '/path/to/document1.txt',
      file_name: 'document1.txt',
      content: 'This is the first chunk of document 1',
      chunk_index: 0,
      total_chunks: 2,
      score: 0.95,
    },
    {
      chunk_id: 'chunk2',
      doc_id: 'doc1',
      collection_id: 'collection1',
      collection_name: 'Test Collection',
      file_path: '/path/to/document1.txt',
      file_name: 'document1.txt',
      content: 'This is the second chunk of document 1',
      chunk_index: 1,
      total_chunks: 2,
      score: 0.85,
    },
    {
      chunk_id: 'chunk3',
      doc_id: 'doc2',
      collection_id: 'collection1',
      collection_name: 'Test Collection',
      file_path: '/path/to/document2.txt',
      file_name: 'document2.txt',
      content: 'This is a chunk from document 2',
      chunk_index: 0,
      total_chunks: 1,
      score: 0.75,
    },
  ]

  type RerankingMetrics =
    | {
        rerankingUsed: boolean
        rerankerModel?: string
        rerankingTimeMs?: number
        original_count?: number
        reranked_count?: number
      }
    | null

  type FailedCollection = {
    collection_id: string
    collection_name: string
    error_message?: string
    error?: string
  }

  type GpuMemoryError = {
    message: string
    suggestion: string
    currentModel: string
  }

  type SearchState = {
    results: SearchResult[]
    loading: boolean
    error: string | null
    rerankingMetrics: RerankingMetrics
    failedCollections: FailedCollection[]
    partialFailure: boolean
    hydeUsed: boolean
    hydeInfo: HyDEInfo | null
    gpuMemoryError: GpuMemoryError | null
  }

  let mockSearchState: SearchState

  const setSearchState = (overrides: Partial<SearchState>) => {
    mockSearchState = {
      results: [],
      loading: false,
      error: null,
      rerankingMetrics: null,
      failedCollections: [],
      partialFailure: false,
      hydeUsed: false,
      hydeInfo: null,
      gpuMemoryError: null,
      ...overrides,
    }
  }

  beforeEach(() => {
    vi.clearAllMocks()

    setSearchState({})

    ;(useSearchStore as MockedFunction<typeof useSearchStore>).mockImplementation((selector?: unknown) => {
      if (typeof selector === 'function') {
        return (selector as (state: SearchState) => unknown)(mockSearchState)
      }
      return mockSearchState
    })

    ;(useUIStore as MockedFunction<typeof useUIStore>).mockImplementation((selector?: unknown) => {
      const uiState = { setShowDocumentViewer: mockSetShowDocumentViewer }
      if (typeof selector === 'function') {
        return (selector as (state: typeof uiState) => unknown)(uiState)
      }
      return uiState
    })
  })

  it('renders loading state', () => {
    setSearchState({ loading: true })

    render(<SearchResults />)

    expect(screen.getByText('Searching...')).toBeInTheDocument()
  })

  it('renders GPU memory error UI and forwards model selection', async () => {
    const user = userEvent.setup()
    setSearchState({
      error: 'GPU_MEMORY_ERROR',
      gpuMemoryError: {
        message: 'oom',
        suggestion: 'Try a smaller model',
        currentModel: 'Qwen/Qwen3-Reranker-8B',
      },
    })

    render(<SearchResults onSelectSmallerModel={mockOnSelectSmallerModel} />)

    expect(screen.getByText('Insufficient GPU Memory')).toBeInTheDocument()
    expect(screen.getByText('Try a smaller model')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /Disable reranking/i }))
    expect(mockOnSelectSmallerModel).toHaveBeenCalledWith('disabled')
  })

  it('renders a regular error message', () => {
    setSearchState({ error: 'Something went wrong' })

    render(<SearchResults />)

    expect(screen.getByText('Something went wrong')).toBeInTheDocument()
  })

  it('returns null when there are no results and no failed collections', () => {
    setSearchState({ results: [], failedCollections: [] })

    const { container } = render(<SearchResults />)
    expect(container.firstChild).toBeNull()
  })

  it('renders partial failure warning list', () => {
    setSearchState({
      results: [{ doc_id: 'd', chunk_id: 'c', score: 0.5, content: 'x', file_path: '/p', file_name: 'f', chunk_index: 0, total_chunks: 1 }],
      partialFailure: true,
      failedCollections: [{ collection_id: 'c1', collection_name: 'Collection A', error_message: 'boom' }],
    })

    render(<SearchResults />)

    expect(screen.getByText('Partial Search Failure')).toBeInTheDocument()
    expect(screen.getByText(/Collection A/)).toBeInTheDocument()
    expect(screen.getByText(/boom/)).toBeInTheDocument()
  })

  it('groups results by collection and document', async () => {
    setSearchState({ results: mockResults })

    render(<SearchResults />)

    // Collections auto-expand after initial render.
    await screen.findByText('document1.txt')

    expect(screen.getByText('Search Results')).toBeInTheDocument()
    expect(screen.getByText('Found 3 results across 1 collections')).toBeInTheDocument()
    expect(screen.getByText('Test Collection')).toBeInTheDocument()
    expect(screen.getByText('3 results in 2 documents')).toBeInTheDocument()
    expect(screen.getByText('/path/to/document1.txt')).toBeInTheDocument()
    expect(screen.getByText('/path/to/document2.txt')).toBeInTheDocument()
    expect(screen.getByText('2 chunks')).toBeInTheDocument()
    expect(screen.getByText('1 chunk')).toBeInTheDocument()
    expect(screen.getByText('Score: 0.950')).toBeInTheDocument()
    expect(screen.getByText('Score: 0.750')).toBeInTheDocument()
  })

  it('shows reranking metrics when available', async () => {
    setSearchState({
      results: mockResults,
      rerankingMetrics: {
        rerankingUsed: true,
        rerankingTimeMs: 125.5,
      },
    })

    render(<SearchResults />)

    await screen.findByText('document1.txt')

    expect(screen.getByText('Reranked')).toBeInTheDocument()
    expect(screen.getByText('126ms')).toBeInTheDocument()
  })

  it('renders HyDE expansion section and toggles details', async () => {
    const user = userEvent.setup()
    setSearchState({
      results: [{ doc_id: 'd', chunk_id: 'c', score: 0.5, content: 'x', file_path: '/p', file_name: 'f', chunk_index: 0, total_chunks: 1, collection_id: 'col', collection_name: 'Col' }],
      hydeUsed: true,
      hydeInfo: {
        expanded_query: 'expanded text',
        provider: 'openai',
        model: 'gpt-x',
        tokens_used: 42,
        generation_time_ms: 123.4,
      },
    })

    render(<SearchResults />)

    await user.click(screen.getByRole('button', { name: /HyDE Query Expansion/i }))

    expect(screen.getByText('Generated Hypothetical Document')).toBeInTheDocument()
    expect(screen.getByText('expanded text')).toBeInTheDocument()
    expect(screen.getByText(/openai/)).toBeInTheDocument()
  })

  it('opens document viewer with safe collection id when missing', async () => {
    const user = userEvent.setup()
    setSearchState({
      results: [
        {
          doc_id: 'doc-1',
          chunk_id: 'chunk-1',
          score: 0.81,
          content: 'hello',
          file_path: '/a.txt',
          file_name: 'a.txt',
          chunk_index: 0,
          total_chunks: 1,
          // no collection_id or collection_name => grouped under 'unknown'
          original_score: 0.5,
          reranked_score: 0.8,
          embedding_model: 'model-x',
        },
      ],
    })

    render(<SearchResults />)

    // Collections auto-expand after initial render.
    await screen.findByText('a.txt')

    // Expand the document.
    await user.click(screen.getByText('a.txt'))

    // Click "View Document →" and ensure we default to collectionId="unknown".
    await user.click(screen.getByRole('button', { name: /View Document/i }))

    await waitFor(() => {
      expect(mockSetShowDocumentViewer).toHaveBeenCalledWith({
        collectionId: 'unknown',
        docId: 'doc-1',
        chunkId: 'chunk-1',
      })
    })
  })
})
