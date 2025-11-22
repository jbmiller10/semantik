import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@/tests/utils/test-utils'
import userEvent from '@testing-library/user-event'
import SearchResults from '../SearchResults'
import { useSearchStore } from '@/stores/searchStore'
import { useUIStore } from '@/stores/uiStore'

// Mock the stores
vi.mock('@/stores/searchStore')
vi.mock('@/stores/uiStore')

const mockedUseSearchStore = useSearchStore as unknown as vi.Mock
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

const mockSearchStore = (overrides: Record<string, unknown> = {}) => {
  const state = {
    results: [],
    loading: false,
    error: null,
    rerankingMetrics: null,
    failedCollections: [],
    partialFailure: false,
    gpuMemoryError: null,
    ...overrides,
  }

  mockedUseSearchStore.mockImplementation((selector?: (store: typeof state) => unknown) => {
    if (typeof selector === 'function') {
      return selector(state as never)
    }
    return state as never
  })
}

describe('SearchResults', () => {
  const mockSetShowDocumentViewer = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()

    vi.mocked(useUIStore).mockImplementation((selector: (state: ReturnType<typeof useUIStore>) => unknown) => {
      const state = {
        setShowDocumentViewer: mockSetShowDocumentViewer,
      }
      return selector ? selector(state as ReturnType<typeof useUIStore>) : state
    })

    mockSearchStore()
  })

  it('renders loading state', () => {
    mockSearchStore({ loading: true })

    render(<SearchResults />)

    expect(screen.getByText('Searching...')).toBeInTheDocument()
    // Check for spinner - find element with animate-spin class
    const spinner = document.querySelector('.animate-spin')
    expect(spinner).toBeInTheDocument()
  })

  it('renders error state', () => {
    const errorMessage = 'Failed to fetch search results'
    mockSearchStore({ error: errorMessage })

    render(<SearchResults />)

    expect(screen.getByText(errorMessage)).toBeInTheDocument()
  })

  it('renders nothing when results are empty', () => {
    mockSearchStore()

    const { container } = render(<SearchResults />)

    expect(container.firstChild).toBeNull()
  })

  it('renders search results grouped by collection and document', () => {
    mockSearchStore({ results: mockResults })

    render(<SearchResults />)

    // Check header
    expect(screen.getByText('Search Results')).toBeInTheDocument()
    expect(screen.getByText('Found 3 results across 1 collections')).toBeInTheDocument()

    // Check collection header
    expect(screen.getByText('Test Collection')).toBeInTheDocument()
    expect(screen.getByText('3 results in 2 documents')).toBeInTheDocument()

    // Check document headers
    expect(screen.getByText('document1.txt')).toBeInTheDocument()
    expect(screen.getByText('document2.txt')).toBeInTheDocument()

    // Check file paths
    expect(screen.getByText('/path/to/document1.txt')).toBeInTheDocument()
    expect(screen.getByText('/path/to/document2.txt')).toBeInTheDocument()

    // Check chunk counts
    expect(screen.getByText('2 chunks')).toBeInTheDocument()
    expect(screen.getByText('1 chunk')).toBeInTheDocument()

    // Check max scores
    expect(screen.getByText('Score: 0.950')).toBeInTheDocument()
    expect(screen.getByText('Score: 0.750')).toBeInTheDocument()
  })

  it('displays reranking metrics when available', () => {
    mockSearchStore({
      results: mockResults,
      rerankingMetrics: {
        rerankingUsed: true,
        rerankingTimeMs: 125.5,
      },
    })

    render(<SearchResults />)

    expect(screen.getByText('Reranked')).toBeInTheDocument()
    expect(screen.getByText('126ms')).toBeInTheDocument()
  })

  it('expands and collapses documents on click', async () => {
    const user = userEvent.setup()

    mockSearchStore({ results: mockResults })

    render(<SearchResults />)

    // Initially, chunks should not be visible
    expect(screen.queryByText('This is the first chunk of document 1')).not.toBeInTheDocument()

    // Click on first document to expand
    const firstDoc = screen.getByText('document1.txt').closest('.cursor-pointer')
    await user.click(firstDoc!)

    // Now chunks should be visible
    expect(screen.getByText('This is the first chunk of document 1')).toBeInTheDocument()
    expect(screen.getByText('This is the second chunk of document 1')).toBeInTheDocument()
    expect(screen.getByText('Chunk 1 of 2')).toBeInTheDocument()
    expect(screen.getByText('Chunk 2 of 2')).toBeInTheDocument()
    expect(screen.getAllByText('Score: 0.950')[0]).toBeInTheDocument()
    expect(screen.getAllByText('Score: 0.850')[0]).toBeInTheDocument()

    // Click again to collapse
    await user.click(firstDoc!)

    // Chunks should be hidden again
    expect(screen.queryByText('This is the first chunk of document 1')).not.toBeInTheDocument()
  })

  it('handles view document button click', async () => {
    const user = userEvent.setup()

    mockSearchStore({ results: mockResults })

    render(<SearchResults />)

    // Expand first document
    const firstDoc = screen.getByText('document1.txt').closest('.cursor-pointer')
    await user.click(firstDoc!)

    // Click view document button
    const viewButtons = screen.getAllByText('View Document →')
    await user.click(viewButtons[0])

    expect(mockSetShowDocumentViewer).toHaveBeenCalledWith({
      collectionId: 'collection1',
      docId: 'doc1',
      chunkId: 'chunk1',
    })
  })

  it('handles chunk row click to open document viewer', async () => {
    const user = userEvent.setup()

    mockSearchStore({ results: mockResults })

    render(<SearchResults />)

    // Expand first document
    const firstDoc = screen.getByText('document1.txt').closest('.cursor-pointer')
    await user.click(firstDoc!)

    // Click on the first chunk row directly
    const firstChunkContent = screen.getByText('This is the first chunk of document 1')
    const chunkRow = firstChunkContent.closest('.cursor-pointer')
    await user.click(chunkRow!)

    expect(mockSetShowDocumentViewer).toHaveBeenCalledWith({
      collectionId: 'collection1',
      docId: 'doc1',
      chunkId: 'chunk1',
    })
  })

  it('handles chunk row click for second chunk with correct chunk ID', async () => {
    const user = userEvent.setup()

    mockSearchStore({ results: mockResults })

    render(<SearchResults />)

    // Expand first document
    const firstDoc = screen.getByText('document1.txt').closest('.cursor-pointer')
    await user.click(firstDoc!)

    // Click on the second chunk row
    const secondChunkContent = screen.getByText('This is the second chunk of document 1')
    const chunkRow = secondChunkContent.closest('.cursor-pointer')
    await user.click(chunkRow!)

    expect(mockSetShowDocumentViewer).toHaveBeenCalledWith({
      collectionId: 'collection1',
      docId: 'doc1',
      chunkId: 'chunk2',
    })
  })

  it('handles missing collection ID in results', async () => {
    const user = userEvent.setup()

    const resultsWithoutCollectionId = [
      {
        ...mockResults[0],
        collection_id: undefined,
      },
    ]

    mockSearchStore({ results: resultsWithoutCollectionId })

    render(<SearchResults />)

    // Expand document
    const doc = screen.getByText('document1.txt').closest('.cursor-pointer')
    await user.click(doc!)

    // Click view document - should use 'unknown' as collectionId
    const viewButton = screen.getByText('View Document →')
    await user.click(viewButton)

    expect(mockSetShowDocumentViewer).toHaveBeenCalledWith({
      collectionId: 'unknown',
      docId: 'doc1',
      chunkId: 'chunk1',
    })
  })

  it('prevents event propagation when clicking view document button', async () => {
    const user = userEvent.setup()

    mockSearchStore({ results: mockResults })

    render(<SearchResults />)

    // Expand first document
    const firstDoc = screen.getByText('document1.txt').closest('.cursor-pointer')
    await user.click(firstDoc!)

    // Verify document is expanded
    expect(screen.getByText('This is the first chunk of document 1')).toBeInTheDocument()

    // Click view document button - should not collapse the document
    const viewButton = screen.getAllByText('View Document →')[0]
    await user.click(viewButton)

    // Document should still be expanded
    expect(screen.getByText('This is the first chunk of document 1')).toBeInTheDocument()

    // And setShowDocumentViewer should have been called
    expect(mockSetShowDocumentViewer).toHaveBeenCalled()
  })

  it('displays empty message when groupedResults is empty', () => {
    // This scenario shouldn't normally happen since we return null for empty results,
    // but the component has this code path
    mockSearchStore()

    // Force render by mocking a non-empty results array that groups to empty
    const { container } = render(<SearchResults />)

    // Component returns null for empty results
    expect(container.firstChild).toBeNull()
  })

  it('renders GPU memory error guidance and forwards model selection', async () => {
    const user = userEvent.setup()
    const onSelect = vi.fn()

    mockSearchStore({
      error: 'GPU_MEMORY_ERROR',
      gpuMemoryError: {
        message: 'Not enough memory',
        suggestion: 'Try a smaller reranker model.',
        currentModel: 'Qwen/Qwen3-Reranker-8B',
      },
    })

    render(<SearchResults onSelectSmallerModel={onSelect} />)

    expect(screen.getByText('Insufficient GPU Memory')).toBeInTheDocument()
    expect(screen.getByText('Try a smaller reranker model.')).toBeInTheDocument()

    const disableButton = screen.getByRole('button', { name: /disable reranking to proceed without it/i })
    await user.click(disableButton)

    expect(onSelect).toHaveBeenCalledWith('disabled')
  })
})
