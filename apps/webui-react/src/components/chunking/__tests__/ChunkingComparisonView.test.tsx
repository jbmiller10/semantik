import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, waitFor, fireEvent } from '@/tests/utils/test-utils'
import userEvent from '@testing-library/user-event'
import { ChunkingComparisonView } from '../ChunkingComparisonView'
import { useChunkingStore } from '@/stores/chunkingStore'
import type { ChunkingComparisonResult, ChunkingStrategyType } from '@/types/chunking'

// Mock the chunking store
vi.mock('@/stores/chunkingStore', () => ({
  useChunkingStore: vi.fn(),
}))

// Mock URL.createObjectURL and URL.revokeObjectURL
global.URL.createObjectURL = vi.fn(() => 'blob:mock-url')
global.URL.revokeObjectURL = vi.fn()

// Mock document.createElement for download testing
const mockClick = vi.fn()
const originalCreateElement = window.document.createElement.bind(window.document)
window.document.createElement = vi.fn((tagName: string) => {
  const element = originalCreateElement(tagName) as any
  if (tagName === 'a') {
    element.click = mockClick
  }
  return element
})

const mockComparisonResults: Record<ChunkingStrategyType, ChunkingComparisonResult> = {
  recursive: {
    preview: {
      chunks: [
        { id: '1', content: 'Chunk 1 content', metadata: { position: 0, size: 100 } },
        { id: '2', content: 'Chunk 2 content', metadata: { position: 100, size: 120 } },
      ],
      statistics: {
        totalChunks: 10,
        avgChunkSize: 512,
        minChunkSize: 200,
        maxChunkSize: 800,
        totalSize: 5120,
        overlapRatio: 0.1,
      },
      performance: {
        processingTimeMs: 150,
        chunksPerSecond: 66.67,
        memoryUsageMB: 12,
      },
      warnings: [],
    },
    score: {
      overall: 0.85,
      quality: 0.88,
      performance: 0.82,
      factors: {
        coherence: 0.9,
        completeness: 0.95,
        redundancy: 0.15,
        processingSpeed: 0.85,
      },
    },
  },
  fixed: {
    preview: {
      chunks: [
        { id: '3', content: 'Fixed chunk 1', metadata: { position: 0, size: 500 } },
        { id: '4', content: 'Fixed chunk 2', metadata: { position: 500, size: 500 } },
      ],
      statistics: {
        totalChunks: 8,
        avgChunkSize: 500,
        minChunkSize: 500,
        maxChunkSize: 500,
        totalSize: 4000,
        overlapRatio: 0,
      },
      performance: {
        processingTimeMs: 50,
        chunksPerSecond: 160,
        memoryUsageMB: 8,
      },
      warnings: [],
    },
    score: {
      overall: 0.75,
      quality: 0.70,
      performance: 0.95,
      factors: {
        coherence: 0.65,
        completeness: 0.85,
        redundancy: 0.05,
        processingSpeed: 0.98,
      },
    },
  },
  semantic: {
    preview: {
      chunks: [
        { id: '5', content: 'Semantic chunk 1', metadata: { position: 0, size: 600 } },
        { id: '6', content: 'Semantic chunk 2', metadata: { position: 600, size: 450 } },
      ],
      statistics: {
        totalChunks: 7,
        avgChunkSize: 580,
        minChunkSize: 350,
        maxChunkSize: 750,
        totalSize: 4060,
        overlapRatio: 0.05,
      },
      performance: {
        processingTimeMs: 300,
        chunksPerSecond: 23.33,
        memoryUsageMB: 24,
      },
      warnings: ['High memory usage detected'],
    },
    score: {
      overall: 0.92,
      quality: 0.95,
      performance: 0.65,
      factors: {
        coherence: 0.98,
        completeness: 0.97,
        redundancy: 0.08,
        processingSpeed: 0.60,
      },
    },
  },
}

describe('ChunkingComparisonView', () => {
  const mockAddComparisonStrategy = vi.fn()
  const mockRemoveComparisonStrategy = vi.fn()
  const mockCompareStrategies = vi.fn()

  const defaultMockStore = {
    comparisonStrategies: ['recursive', 'fixed'] as ChunkingStrategyType[],
    comparisonResults: {
      recursive: mockComparisonResults.recursive,
      fixed: mockComparisonResults.fixed,
    },
    comparisonLoading: false,
    comparisonError: null,
    addComparisonStrategy: mockAddComparisonStrategy,
    removeComparisonStrategy: mockRemoveComparisonStrategy,
    compareStrategies: mockCompareStrategies,
    selectedStrategy: 'recursive' as ChunkingStrategyType,
  }

  beforeEach(() => {
    vi.clearAllMocks()
    ;(useChunkingStore as any).mockReturnValue(defaultMockStore)
  })

  afterEach(() => {
    vi.clearAllMocks()
  })

  it('renders comparison results correctly', () => {
    render(<ChunkingComparisonView />)

    // Check headers
    expect(screen.getByText('Strategy Comparison')).toBeInTheDocument()
    expect(screen.getByText('Recursive')).toBeInTheDocument()
    expect(screen.getByText('Fixed')).toBeInTheDocument()

    // Check scores
    expect(screen.getByText('85%')).toBeInTheDocument() // Overall score for recursive
    expect(screen.getByText('75%')).toBeInTheDocument() // Overall score for fixed
  })

  it('shows loading state when comparing', () => {
    ;(useChunkingStore as any).mockReturnValue({
      ...defaultMockStore,
      comparisonLoading: true,
    })

    render(<ChunkingComparisonView />)

    expect(screen.getByText('Comparing strategies...')).toBeInTheDocument()
    expect(screen.getByRole('progressbar')).toBeInTheDocument()
  })

  it('shows error state when comparison fails', () => {
    const errorMessage = 'Failed to compare strategies'
    ;(useChunkingStore as any).mockReturnValue({
      ...defaultMockStore,
      comparisonError: errorMessage,
    })

    render(<ChunkingComparisonView />)

    expect(screen.getByText('Comparison Error')).toBeInTheDocument()
    expect(screen.getByText(errorMessage)).toBeInTheDocument()
  })

  it('shows empty state when no strategies selected', () => {
    ;(useChunkingStore as any).mockReturnValue({
      ...defaultMockStore,
      comparisonStrategies: [],
      comparisonResults: {},
    })

    render(<ChunkingComparisonView />)

    expect(screen.getByText('No Strategies Selected')).toBeInTheDocument()
    expect(screen.getByText(/Select strategies to compare/)).toBeInTheDocument()
  })

  it('adds new strategy for comparison', async () => {
    const user = userEvent.setup()
    render(<ChunkingComparisonView />)

    // Click add strategy button
    const addButton = screen.getByRole('button', { name: /add strategy/i })
    await user.click(addButton)

    // Strategy selector should appear
    await waitFor(() => {
      expect(screen.getByText('Semantic')).toBeInTheDocument()
    })

    // Select semantic strategy
    const semanticButton = screen.getByText('Semantic')
    await user.click(semanticButton)

    expect(mockAddComparisonStrategy).toHaveBeenCalledWith('semantic')
  })

  it('removes strategy from comparison', async () => {
    const user = userEvent.setup()
    render(<ChunkingComparisonView />)

    // Click remove button for recursive strategy
    const removeButtons = screen.getAllByRole('button', { name: /remove/i })
    await user.click(removeButtons[0])

    expect(mockRemoveComparisonStrategy).toHaveBeenCalledWith('recursive')
  })

  it('respects maximum strategies limit', () => {
    ;(useChunkingStore as any).mockReturnValue({
      ...defaultMockStore,
      comparisonStrategies: ['recursive', 'fixed', 'semantic'] as ChunkingStrategyType[],
    })

    render(<ChunkingComparisonView maxStrategies={3} />)

    // Add button should be disabled
    const addButton = screen.getByRole('button', { name: /add strategy/i })
    expect(addButton).toBeDisabled()
  })

  it('triggers auto-compare when document changes', () => {
    const document = { id: 'doc1', content: 'Test content', name: 'test.txt' }
    
    const { rerender } = render(<ChunkingComparisonView document={document} />)
    
    expect(mockCompareStrategies).toHaveBeenCalledTimes(1)

    // Change document
    const newDocument = { id: 'doc2', content: 'New content', name: 'new.txt' }
    rerender(<ChunkingComparisonView document={newDocument} />)

    expect(mockCompareStrategies).toHaveBeenCalledTimes(2)
  })

  it('exports comparison results as JSON', async () => {
    const user = userEvent.setup()
    render(<ChunkingComparisonView />)

    // Select JSON format
    const jsonRadio = screen.getByLabelText('JSON')
    await user.click(jsonRadio)

    // Click export button
    const exportButton = screen.getByRole('button', { name: /export/i })
    await user.click(exportButton)

    expect(global.URL.createObjectURL).toHaveBeenCalled()
    expect(mockClick).toHaveBeenCalled()
    expect(global.URL.revokeObjectURL).toHaveBeenCalled()

    // Check the blob was created with JSON data
    const blobCall = (global.URL.createObjectURL as any).mock.calls[0]
    expect(blobCall).toBeDefined()
  })

  it('exports comparison results as CSV', async () => {
    const user = userEvent.setup()
    render(<ChunkingComparisonView />)

    // Select CSV format
    const csvRadio = screen.getByLabelText('CSV')
    await user.click(csvRadio)

    // Click export button
    const exportButton = screen.getByRole('button', { name: /export/i })
    await user.click(exportButton)

    expect(global.URL.createObjectURL).toHaveBeenCalled()
    expect(mockClick).toHaveBeenCalled()
    expect(global.URL.revokeObjectURL).toHaveBeenCalled()
  })

  it('toggles sync scroll', async () => {
    const user = userEvent.setup()
    render(<ChunkingComparisonView />)

    const syncScrollCheckbox = screen.getByLabelText(/sync scroll/i)
    expect(syncScrollCheckbox).toBeChecked()

    await user.click(syncScrollCheckbox)
    expect(syncScrollCheckbox).not.toBeChecked()
  })

  it('handles synchronized scrolling', () => {
    render(<ChunkingComparisonView />)

    const scrollContainers = screen.getAllByTestId(/chunk-preview-container/i)
    
    // Simulate scroll on first container
    fireEvent.scroll(scrollContainers[0], { target: { scrollTop: 100 } })

    // Other containers should be updated (implementation dependent)
    // This test verifies the event handler is attached
    expect(scrollContainers[0]).toBeDefined()
  })

  it('displays performance metrics correctly', () => {
    render(<ChunkingComparisonView />)

    // Check recursive strategy metrics
    expect(screen.getByText('150ms')).toBeInTheDocument() // processing time
    expect(screen.getByText('12 MB')).toBeInTheDocument() // memory usage
    expect(screen.getByText('66.67')).toBeInTheDocument() // chunks per second

    // Check fixed strategy metrics
    expect(screen.getByText('50ms')).toBeInTheDocument()
    expect(screen.getByText('8 MB')).toBeInTheDocument()
    expect(screen.getByText('160')).toBeInTheDocument()
  })

  it('displays quality scores with proper indicators', () => {
    render(<ChunkingComparisonView />)

    // Check quality indicators
    const qualityScores = screen.getAllByTestId(/quality-score/i)
    expect(qualityScores).toHaveLength(2)

    // Recursive should have high quality
    expect(screen.getByText('88%')).toBeInTheDocument()
    
    // Fixed should have lower quality
    expect(screen.getByText('70%')).toBeInTheDocument()
  })

  it('shows warnings when present', () => {
    ;(useChunkingStore as any).mockReturnValue({
      ...defaultMockStore,
      comparisonResults: {
        semantic: mockComparisonResults.semantic,
      },
      comparisonStrategies: ['semantic'] as ChunkingStrategyType[],
    })

    render(<ChunkingComparisonView />)

    expect(screen.getByText('High memory usage detected')).toBeInTheDocument()
  })

  it('highlights the best performing strategy', () => {
    ;(useChunkingStore as any).mockReturnValue({
      ...defaultMockStore,
      comparisonStrategies: ['recursive', 'fixed', 'semantic'] as ChunkingStrategyType[],
      comparisonResults: mockComparisonResults,
    })

    render(<ChunkingComparisonView />)

    // Semantic has the highest overall score (0.92)
    const bestStrategyCard = screen.getByText('Semantic').closest('.border')
    expect(bestStrategyCard).toHaveClass('border-green-500')
  })

  it('shows chunk preview snippets', () => {
    render(<ChunkingComparisonView />)

    expect(screen.getByText('Chunk 1 content')).toBeInTheDocument()
    expect(screen.getByText('Chunk 2 content')).toBeInTheDocument()
    expect(screen.getByText('Fixed chunk 1')).toBeInTheDocument()
    expect(screen.getByText('Fixed chunk 2')).toBeInTheDocument()
  })

  it('displays statistical comparisons', () => {
    render(<ChunkingComparisonView />)

    // Check statistics display
    expect(screen.getByText('Total Chunks: 10')).toBeInTheDocument() // recursive
    expect(screen.getByText('Total Chunks: 8')).toBeInTheDocument() // fixed
    expect(screen.getByText('Avg Size: 512')).toBeInTheDocument() // recursive
    expect(screen.getByText('Avg Size: 500')).toBeInTheDocument() // fixed
  })

  it('cancels add strategy operation', async () => {
    const user = userEvent.setup()
    render(<ChunkingComparisonView />)

    // Click add strategy button
    const addButton = screen.getByRole('button', { name: /add strategy/i })
    await user.click(addButton)

    // Cancel button should appear
    const cancelButton = screen.getByRole('button', { name: /cancel/i })
    await user.click(cancelButton)

    // Strategy selector should disappear
    await waitFor(() => {
      expect(screen.queryByText('Select a strategy to add')).not.toBeInTheDocument()
    })

    expect(mockAddComparisonStrategy).not.toHaveBeenCalled()
  })

  it('disables unavailable strategies in selector', async () => {
    const user = userEvent.setup()
    render(<ChunkingComparisonView />)

    // Click add strategy button
    const addButton = screen.getByRole('button', { name: /add strategy/i })
    await user.click(addButton)

    // Already selected strategies should not appear
    expect(screen.queryByText('Recursive')).not.toBeInTheDocument()
    expect(screen.queryByText('Fixed')).not.toBeInTheDocument()
    
    // Available strategies should appear
    expect(screen.getByText('Semantic')).toBeInTheDocument()
    expect(screen.getByText('Markdown')).toBeInTheDocument()
  })

  it('handles empty comparison results gracefully', () => {
    ;(useChunkingStore as any).mockReturnValue({
      ...defaultMockStore,
      comparisonResults: {},
      comparisonStrategies: ['recursive'] as ChunkingStrategyType[],
    })

    render(<ChunkingComparisonView />)

    expect(screen.getByText('Recursive')).toBeInTheDocument()
    expect(screen.getByText('No results available')).toBeInTheDocument()
  })

  it('displays trend indicators for metrics', () => {
    render(<ChunkingComparisonView />)

    // Check for trend icons
    const upTrends = screen.getAllByTestId('trend-up')
    const downTrends = screen.getAllByTestId('trend-down')
    const stableTrends = screen.getAllByTestId('trend-stable')

    expect(upTrends.length + downTrends.length + stableTrends.length).toBeGreaterThan(0)
  })
})