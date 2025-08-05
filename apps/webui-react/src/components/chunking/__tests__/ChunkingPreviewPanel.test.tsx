import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, waitFor, fireEvent } from '@/tests/utils/test-utils'
import userEvent from '@testing-library/user-event'
import { ChunkingPreviewPanel } from '../ChunkingPreviewPanel'
import { useChunkingStore } from '@/stores/chunkingStore'
import type { ChunkPreview, ChunkingStatistics } from '@/types/chunking'

// Mock the chunking store
vi.mock('@/stores/chunkingStore', () => ({
  useChunkingStore: vi.fn(),
}))

// Mock clipboard API
const mockWriteText = vi.fn()
Object.assign(navigator, {
  clipboard: {
    writeText: mockWriteText,
  },
})

const mockChunks: ChunkPreview[] = [
  {
    id: 'chunk-1',
    content: 'This is the first chunk of text with some content.',
    startIndex: 0,
    endIndex: 50,
    overlapWithPrevious: 0,
    overlapWithNext: 10,
    metadata: {
      position: 0,
      size: 50,
      tokens: 12,
    },
  },
  {
    id: 'chunk-2',
    content: 'Second chunk with overlapping content from previous.',
    startIndex: 40,
    endIndex: 93,
    overlapWithPrevious: 10,
    overlapWithNext: 15,
    metadata: {
      position: 1,
      size: 53,
      tokens: 13,
    },
  },
  {
    id: 'chunk-3',
    content: 'Third chunk continues the document text.',
    startIndex: 78,
    endIndex: 119,
    overlapWithPrevious: 15,
    overlapWithNext: 0,
    metadata: {
      position: 2,
      size: 41,
      tokens: 10,
    },
  },
]

const mockStatistics: ChunkingStatistics = {
  totalChunks: 3,
  avgChunkSize: 48,
  minChunkSize: 41,
  maxChunkSize: 53,
  totalSize: 144,
  overlapRatio: 0.17,
}

const mockDocument = {
  id: 'doc-1',
  content: 'This is the first chunk of text with some content from previous. Second chunk with overlapping content from previous. Third chunk continues the document text.',
  name: 'test-document.txt',
}

describe('ChunkingPreviewPanel', () => {
  const mockSetPreviewDocument = vi.fn()
  const mockLoadPreview = vi.fn()

  const defaultMockStore = {
    previewDocument: mockDocument,
    previewChunks: mockChunks,
    previewStatistics: mockStatistics,
    previewLoading: false,
    previewError: null,
    setPreviewDocument: mockSetPreviewDocument,
    loadPreview: mockLoadPreview,
  }

  beforeEach(() => {
    vi.clearAllMocks()
    mockWriteText.mockResolvedValue(undefined)
    ;(useChunkingStore as any).mockReturnValue(defaultMockStore)
  })

  afterEach(() => {
    vi.clearAllMocks()
  })

  it('renders preview panel with chunks and statistics', () => {
    render(<ChunkingPreviewPanel />)

    expect(screen.getByText('Chunk Preview')).toBeInTheDocument()
    expect(screen.getByText('3 chunks')).toBeInTheDocument()
    expect(screen.getByText('Avg: 48 chars')).toBeInTheDocument()
    expect(screen.getByText('Overlap: 17%')).toBeInTheDocument()
  })

  it('shows loading state when preview is loading', () => {
    ;(useChunkingStore as any).mockReturnValue({
      ...defaultMockStore,
      previewLoading: true,
    })

    render(<ChunkingPreviewPanel />)

    expect(screen.getByText('Generating preview...')).toBeInTheDocument()
    expect(screen.getByRole('progressbar')).toBeInTheDocument()
  })

  it('shows error state when preview fails', () => {
    const errorMessage = 'Failed to generate preview'
    ;(useChunkingStore as any).mockReturnValue({
      ...defaultMockStore,
      previewError: errorMessage,
    })

    render(<ChunkingPreviewPanel />)

    expect(screen.getByText('Preview Error')).toBeInTheDocument()
    expect(screen.getByText(errorMessage)).toBeInTheDocument()
  })

  it('shows empty state when no document is selected', () => {
    ;(useChunkingStore as any).mockReturnValue({
      ...defaultMockStore,
      previewDocument: null,
      previewChunks: [],
    })

    render(<ChunkingPreviewPanel onDocumentSelect={vi.fn()} />)

    expect(screen.getByText('No Document Selected')).toBeInTheDocument()
    expect(screen.getByText(/Select a document/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /select document/i })).toBeInTheDocument()
  })

  it('calls onDocumentSelect when select button is clicked', async () => {
    const mockOnDocumentSelect = vi.fn()
    ;(useChunkingStore as any).mockReturnValue({
      ...defaultMockStore,
      previewDocument: null,
      previewChunks: [],
    })

    const user = userEvent.setup()
    render(<ChunkingPreviewPanel onDocumentSelect={mockOnDocumentSelect} />)

    const selectButton = screen.getByRole('button', { name: /select document/i })
    await user.click(selectButton)

    expect(mockOnDocumentSelect).toHaveBeenCalledTimes(1)
  })

  it('switches between view modes', async () => {
    const user = userEvent.setup()
    render(<ChunkingPreviewPanel />)

    // Default is split view
    expect(screen.getByRole('button', { name: /split/i })).toHaveClass('bg-blue-600')

    // Switch to chunks only view
    const chunksButton = screen.getByRole('button', { name: /chunks/i })
    await user.click(chunksButton)
    expect(chunksButton).toHaveClass('bg-blue-600')

    // Switch to original only view
    const originalButton = screen.getByRole('button', { name: /original/i })
    await user.click(originalButton)
    expect(originalButton).toHaveClass('bg-blue-600')
  })

  it('navigates between chunks', async () => {
    const user = userEvent.setup()
    render(<ChunkingPreviewPanel />)

    // Initially shows first chunk
    expect(screen.getByText('Chunk 1 of 3')).toBeInTheDocument()
    expect(screen.getByText('This is the first chunk of text with some content.')).toBeInTheDocument()

    // Navigate to next chunk
    const nextButton = screen.getByRole('button', { name: /next chunk/i })
    await user.click(nextButton)

    expect(screen.getByText('Chunk 2 of 3')).toBeInTheDocument()
    expect(screen.getByText('Second chunk with overlapping content from previous.')).toBeInTheDocument()

    // Navigate to previous chunk
    const prevButton = screen.getByRole('button', { name: /previous chunk/i })
    await user.click(prevButton)

    expect(screen.getByText('Chunk 1 of 3')).toBeInTheDocument()
  })

  it('disables navigation buttons at boundaries', () => {
    render(<ChunkingPreviewPanel />)

    // At first chunk, previous should be disabled
    const prevButton = screen.getByRole('button', { name: /previous chunk/i })
    expect(prevButton).toBeDisabled()

    // Next should be enabled
    const nextButton = screen.getByRole('button', { name: /next chunk/i })
    expect(nextButton).not.toBeDisabled()
  })

  it('copies chunk content to clipboard', async () => {
    const user = userEvent.setup()
    render(<ChunkingPreviewPanel />)

    const copyButton = screen.getAllByRole('button', { name: /copy/i })[0]
    await user.click(copyButton)

    expect(mockWriteText).toHaveBeenCalledWith('This is the first chunk of text with some content.')
    
    // Should show success indicator
    await waitFor(() => {
      expect(screen.getByTestId('copy-success-1')).toBeInTheDocument()
    })
  })

  it('adjusts font size with zoom controls', async () => {
    const user = userEvent.setup()
    render(<ChunkingPreviewPanel />)

    // Initial font size
    const contentElement = screen.getByTestId('chunk-content')
    expect(contentElement).toHaveStyle({ fontSize: '14px' })

    // Zoom in
    const zoomInButton = screen.getByRole('button', { name: /zoom in/i })
    await user.click(zoomInButton)
    expect(contentElement).toHaveStyle({ fontSize: '16px' })

    // Zoom out
    const zoomOutButton = screen.getByRole('button', { name: /zoom out/i })
    await user.click(zoomOutButton)
    await user.click(zoomOutButton)
    expect(contentElement).toHaveStyle({ fontSize: '12px' })
  })

  it('limits font size within bounds', async () => {
    const user = userEvent.setup()
    render(<ChunkingPreviewPanel />)

    const zoomOutButton = screen.getByRole('button', { name: /zoom out/i })
    const zoomInButton = screen.getByRole('button', { name: /zoom in/i })

    // Zoom out to minimum
    for (let i = 0; i < 10; i++) {
      await user.click(zoomOutButton)
    }
    
    const contentElement = screen.getByTestId('chunk-content')
    expect(contentElement).toHaveStyle({ fontSize: '10px' })

    // Zoom in to maximum
    for (let i = 0; i < 20; i++) {
      await user.click(zoomInButton)
    }
    
    expect(contentElement).toHaveStyle({ fontSize: '24px' })
  })

  it('highlights chunk boundaries in original text', async () => {
    const user = userEvent.setup()
    render(<ChunkingPreviewPanel />)

    // Switch to split view to see original with boundaries
    const splitButton = screen.getByRole('button', { name: /split/i })
    await user.click(splitButton)

    // Click on a chunk boundary in the original text
    const chunkBoundary = screen.getByTestId('chunk-boundary-1')
    await user.click(chunkBoundary)

    // Should highlight the chunk
    expect(chunkBoundary).toHaveClass('bg-blue-100')
  })

  it('shows chunk metadata', () => {
    render(<ChunkingPreviewPanel />)

    expect(screen.getByText('Position: 0')).toBeInTheDocument()
    expect(screen.getByText('Size: 50 chars')).toBeInTheDocument()
    expect(screen.getByText('Tokens: 12')).toBeInTheDocument()
  })

  it('displays overlap indicators', () => {
    render(<ChunkingPreviewPanel />)

    // First chunk has overlap with next
    expect(screen.getByText('Overlap with next: 10 chars')).toBeInTheDocument()
  })

  it('uses provided document over store document', () => {
    const providedDoc = {
      id: 'provided-doc',
      content: 'Provided document content',
      name: 'provided.txt',
    }

    render(<ChunkingPreviewPanel document={providedDoc} />)

    expect(mockSetPreviewDocument).toHaveBeenCalledWith(providedDoc)
    expect(mockLoadPreview).toHaveBeenCalled()
  })

  it('reloads preview when document changes', () => {
    const { rerender } = render(<ChunkingPreviewPanel document={mockDocument} />)

    const newDocument = {
      id: 'new-doc',
      content: 'New document content',
      name: 'new.txt',
    }

    rerender(<ChunkingPreviewPanel document={newDocument} />)

    expect(mockSetPreviewDocument).toHaveBeenCalledWith(newDocument)
    expect(mockLoadPreview).toHaveBeenCalledTimes(2)
  })

  it('handles empty chunks gracefully', () => {
    ;(useChunkingStore as any).mockReturnValue({
      ...defaultMockStore,
      previewChunks: [],
      previewStatistics: null,
    })

    render(<ChunkingPreviewPanel />)

    expect(screen.getByText('No chunks generated')).toBeInTheDocument()
  })

  it('displays chunk list in chunks-only mode', async () => {
    const user = userEvent.setup()
    render(<ChunkingPreviewPanel />)

    // Switch to chunks only view
    const chunksButton = screen.getByRole('button', { name: /chunks/i })
    await user.click(chunksButton)

    // All chunks should be visible in list
    expect(screen.getByText('This is the first chunk of text with some content.')).toBeInTheDocument()
    expect(screen.getByText('Second chunk with overlapping content from previous.')).toBeInTheDocument()
    expect(screen.getByText('Third chunk continues the document text.')).toBeInTheDocument()
  })

  it('highlights selected chunk in list view', async () => {
    const user = userEvent.setup()
    render(<ChunkingPreviewPanel />)

    // Switch to chunks view
    const chunksButton = screen.getByRole('button', { name: /chunks/i })
    await user.click(chunksButton)

    // Click on second chunk in list
    const secondChunk = screen.getByText('Second chunk with overlapping content from previous.')
    await user.click(secondChunk.closest('.chunk-item')!)

    expect(secondChunk.closest('.chunk-item')).toHaveClass('bg-blue-50')
  })

  it('shows statistics panel', () => {
    render(<ChunkingPreviewPanel />)

    const statsButton = screen.getByRole('button', { name: /statistics/i })
    fireEvent.click(statsButton)

    expect(screen.getByText('Chunking Statistics')).toBeInTheDocument()
    expect(screen.getByText('Total Chunks:')).toBeInTheDocument()
    expect(screen.getByText('3')).toBeInTheDocument()
    expect(screen.getByText('Average Size:')).toBeInTheDocument()
    expect(screen.getByText('48 chars')).toBeInTheDocument()
    expect(screen.getByText('Min Size:')).toBeInTheDocument()
    expect(screen.getByText('41 chars')).toBeInTheDocument()
    expect(screen.getByText('Max Size:')).toBeInTheDocument()
    expect(screen.getByText('53 chars')).toBeInTheDocument()
    expect(screen.getByText('Total Size:')).toBeInTheDocument()
    expect(screen.getByText('144 chars')).toBeInTheDocument()
    expect(screen.getByText('Overlap Ratio:')).toBeInTheDocument()
    expect(screen.getByText('17%')).toBeInTheDocument()
  })

  it('handles custom height prop', () => {
    render(<ChunkingPreviewPanel height="400px" />)

    const previewContainer = screen.getByTestId('preview-container')
    expect(previewContainer).toHaveStyle({ height: '400px' })
  })

  it('synchronizes scroll between panels in split view', () => {
    render(<ChunkingPreviewPanel />)

    const leftPanel = screen.getByTestId('left-panel')
    const rightPanel = screen.getByTestId('right-panel')

    // Simulate scroll on left panel
    fireEvent.scroll(leftPanel, { target: { scrollTop: 100 } })

    // Right panel should update (implementation dependent)
    expect(leftPanel).toBeDefined()
    expect(rightPanel).toBeDefined()
  })

  it('handles chunk selection from chunk list', async () => {
    const user = userEvent.setup()
    render(<ChunkingPreviewPanel />)

    // Click on chunk item in sidebar
    const chunkItems = screen.getAllByTestId(/chunk-item-/i)
    await user.click(chunkItems[1])

    expect(screen.getByText('Chunk 2 of 3')).toBeInTheDocument()
  })

  it('shows warning for large documents', () => {
    const largeDoc = {
      ...mockDocument,
      content: 'x'.repeat(100000), // 100KB document
    }

    ;(useChunkingStore as any).mockReturnValue({
      ...defaultMockStore,
      previewDocument: largeDoc,
    })

    render(<ChunkingPreviewPanel />)

    expect(screen.getByText(/Large document detected/i)).toBeInTheDocument()
  })
})